'Unified client interfaces for interacting with the SO-ARM10x robots using the LeRobot `robots` API, and the Gr00t policy server expected by the Dum-E agent. This module provides:'

import hashlib
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple
from pprint import pformat

import draccus
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import Robot, RobotConfig, make_robot_from_config  # noqa: F401
from lerobot.robots.so_follower import SOFollower
from lerobot.utils.utils import init_logging, log_say

from shared import IPolicyBackend, IRobotController
from policy.gr00t.service import ExternalRobotInferenceClient
from utils import load_config_file

#################################################################################


def _recursive_add_extra_dim(obs: Dict) -> Dict:
    for key, val in obs.items():
        if isinstance(val, np.ndarray):
            obs[key] = val[np.newaxis, ...]
        elif isinstance(val, dict):
            obs[key] = _recursive_add_extra_dim(val)
        else:
            obs[key] = [val]
    return obs


class Gr00tRobotInferenceClient(IPolicyBackend):
    'The ``groot-native`` :class:`~shared.IPolicyBackend`: Isaac-GR00T native ZMQ.'

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        camera_keys: Optional[List[str]] = None,
        robot_state_keys: Optional[List[str]] = None,
        show_images: bool = False,
        language_instruction: Optional[str] = None,
    ) -> None:
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.camera_keys = camera_keys or ["wrist", "front"]
        self.robot_state_keys = robot_state_keys or [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        self.show_images = show_images
        self._language_instruction = language_instruction
        self._closed = False
        assert (
            len(self.robot_state_keys) == 6
        ), f"robot_state_keys should be size 6, but got {len(self.robot_state_keys)}"
        self.modality_keys = ["single_arm", "gripper"]

    @property
    def language_instruction(self) -> Optional[str]:
        """The stored instruction (read-only; set via `set_lang_instruction`)."""
        return self._language_instruction

    def set_lang_instruction(self, lang_instruction: str) -> None:
        self._language_instruction = lang_instruction

    def ping(self) -> bool:
        """Backend reachability. Delegates to the transport; never raises."""
        return self.policy.ping()

    def reset(self) -> None:
        'Recreate the REQ socket so no mid-FSM socket leaks between episodes.'
        self.policy._init_socket()
        self._closed = False

    def close(self) -> None:
        'Release the transport socket + zmq context. Idempotent.'
        if getattr(self, "_closed", False):
            return
        self._closed = True
        policy = getattr(self, "policy", None)
        socket = getattr(policy, "socket", None)
        if socket is not None:
            try:
                socket.close(linger=0)
            except Exception:
                pass
        context = getattr(policy, "context", None)
        if context is not None:
            try:
                context.term()
            except Exception:
                pass

    def get_action(
        self, observation_dict: Dict[str, Any], lang: Optional[str] = None
    ) -> List[Dict[str, float]]:
        # Fail closed on a missing instruction rather than sending a null under
        # the pinned annotation key: the server accepts it and returns motion
        # conditioned on nothing, which reads downstream as checkpoint drift
        # instead of as the configuration error it is.
        instruction = lang or self._language_instruction
        if not instruction:
            raise ValueError(
                "No language instruction: get_action() was called without a `lang` "
                "argument and no instruction is stored on the backend. Pass `lang` "
                "or call set_lang_instruction() first — a null "
                "`annotation.human.task_description` must never reach the policy."
            )

        # Build nested obs dict for new Isaac-GR00T API
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        obs_dict: Dict[str, Any] = {
            "video": {k: observation_dict[k] for k in self.camera_keys},
            "state": {
                "single_arm": state[:5].astype(np.float32),
                "gripper": state[5:6].astype(np.float32),
            },
            "language": {
                # PINNED N1.7 INFERENCE KEY.
                # Confirmed against: checkpoint experiment_cfg/conf.yaml language
                "annotation.human.task_description": instruction
            },
        }

        if self.show_images:
            view_img(obs_dict["video"])

        # Add T=1 dim then B=1 dim
        obs_dict = _recursive_add_extra_dim(obs_dict)
        obs_dict = _recursive_add_extra_dim(obs_dict)

        # Query policy — returns (action_chunk, info)
        action_chunk, _ = self.policy.get_action(obs_dict)

        # Convert to list of dict[str, float]
        # action_chunk keys are "single_arm"/"gripper" with shape (B, T, D)
        lerobot_actions: List[Dict[str, float]] = []
        horizon = action_chunk[self.modality_keys[0]].shape[1]
        for i in range(horizon):
            concat_action = np.concatenate(
                [
                    np.atleast_1d(action_chunk[key][0][i])
                    for key in self.modality_keys
                ],
                axis=0,
            )
            assert len(concat_action) == len(self.robot_state_keys)
            lerobot_actions.append(
                {
                    key: float(concat_action[idx])
                    for idx, key in enumerate(self.robot_state_keys)
                }
            )
        return lerobot_actions


#################################################################################


# Global figure and axis for continuous streaming
_fig = None
_ax = None


def view_img(img, overlay_img=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env.
    Continuously streams images in the same window without creating new ones.
    """
    global _fig, _ax

    if isinstance(img, dict):
        # stack the images horizontally
        img = np.concatenate([img[k] for k in img], axis=1)

    # Calculate new dimensions while maintaining aspect ratio
    h, w = img.shape[:2]
    scale = max(720 / w, 720 / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Create figure and axis only once
    if _fig is None or _ax is None:
        _fig, _ax = plt.subplots(figsize=(new_w / 100, new_h / 100))
        _ax.set_title("Camera View")
        _ax.axis("off")
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)

    # Clear previous image and display new one
    _ax.clear()
    _ax.imshow(img)
    _ax.set_title("Camera View")
    _ax.axis("off")

    # Update the display
    _fig.canvas.draw()
    _fig.canvas.flush_events()
    plt.pause(0.001)  # Small pause to allow GUI to update


# ============================================================================
# Calibration-file resolution and the connect-time assertion
# ============================================================================


# Fixed intermediate path segment upstream inserts between the calibration root
# and the robot class name (`lerobot.utils.constants.ROBOTS`).
_CALIBRATION_ROBOTS_SEGMENT = "robots"

# Upstream's default cache layout when neither HF_LEROBOT_CALIBRATION nor
# HF_LEROBOT_HOME is set: `HF_HOME / "lerobot" / "calibration"`, with HF_HOME
# itself defaulting to the standard huggingface cache directory.
_HF_CACHE_RELATIVE_DEFAULT = (".cache", "huggingface")
_LEROBOT_CACHE_SEGMENT = "lerobot"
_CALIBRATION_SEGMENT = "calibration"


def resolve_lerobot_calibration_root() -> Path:
    "Resolve LeRobot's calibration root from the documented env vars."
    explicit = os.getenv("HF_LEROBOT_CALIBRATION")
    if explicit:
        return Path(explicit).expanduser()

    lerobot_home = os.getenv("HF_LEROBOT_HOME")
    if lerobot_home:
        return Path(lerobot_home).expanduser() / _CALIBRATION_SEGMENT

    hf_home = os.getenv("HF_HOME")
    base = (
        Path(hf_home).expanduser()
        if hf_home
        else Path.home().joinpath(*_HF_CACHE_RELATIVE_DEFAULT)
    )
    return base / _LEROBOT_CACHE_SEGMENT / _CALIBRATION_SEGMENT


def resolve_calibration_file(robot_name: str, robot_id: str) -> Path:
    'Derive the calibration file path exactly as ``Robot.__init__`` does.'
    return (
        resolve_lerobot_calibration_root()
        / _CALIBRATION_ROBOTS_SEGMENT
        / robot_name
        / f"{robot_id}.json"
    )


def assert_calibration_loaded(
    robot_name: Optional[str] = None,
    robot_id: Optional[str] = None,
    expected_path: Optional[Path] = None,
) -> Tuple[Path, str]:
    'Assert the calibration FILE exists; return its path and content checksum.'
    if expected_path is not None:
        path = Path(expected_path)
    else:
        if not robot_name or not robot_id:
            raise ValueError(
                "assert_calibration_loaded needs either an expected_path or both "
                f"robot_name and robot_id (got robot_name={robot_name!r}, "
                f"robot_id={robot_id!r})"
            )
        path = resolve_calibration_file(robot_name, robot_id)

    if not path.is_file():
        raise FileNotFoundError(
            f"LeRobot calibration file not found at {path}. The arm would run "
            f"with an EMPTY in-Python calibration, and because the raw encoder "
            f"tick ranges in that file map physical poses onto the numbers fed to "
            f"the policy, refusing to proceed is the only safe option. Note that "
            f"the parent directory existing is not evidence: lerobot creates it "
            f"unconditionally. Copy the calibration JSON into "
            f"{path.parent} (the pre-0.6.x copy lives in the sibling "
            f"'so101_follower' directory), or run lerobot-calibrate."
        )

    return path, hashlib.sha256(path.read_bytes()).hexdigest()


# ============================================================================
# Safety and stiffness presets
# ============================================================================


# Dum-E's PID preset, keyed by the Feetech register names the motor bus uses.
# These values are *not* new — Dum-E has written 10/0/5 since v1.0 to reduce
DUME_PID: Dict[str, int] = {
    "P_Coefficient": 10,
    "I_Coefficient": 0,
    "D_Coefficient": 5,
}

# Minimum retry count for a coefficient read-back. Upstream's `read` defaults to
# `num_retry=0`, which turns one dropped packet on the Feetech bus into a value
# mismatch — and a mismatch refuses to connect a perfectly healthy arm. The floor is applied over `config.num_read_retries` so a config that
# lowers retries cannot silently disable this one.
_PID_READ_MIN_RETRIES = 2

# Same floor, same reason, for the connect-time `Goal_Position` pre-arm: a
# dropped packet there would read as a failed pre-arm and refuse the connect.
_PREARM_READ_MIN_RETRIES = 2

# Tick slack when confirming the pre-arm took. The Feetech position register can
# report a neighbouring tick between two reads of a stationary joint, so an exact
# equality check would fail on healthy hardware; anything larger than this is a
# write that genuinely did not land.
_PREARM_TICK_TOLERANCE = 1


# The six motor names on an SO-10x follower. `send_action` strips the `.pos`
# suffix before clamping, so a per-motor clamp mapping is keyed on these.
_MOTOR_NAMES: Tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


# The joint-value convention
# `use_degrees` decides whether the motor bus reports and accepts joint values as
DEFAULT_USE_DEGREES: bool = True

# Recognised boolean spellings. The set is closed on purpose: YAML yields the
# bare string "false" for some unquoted shapes, and every non-empty string is
# truthy in Python, so an unrecognised value must RAISE rather than be silently
# read as True — which would silently select the degrees convention.
_TRUE_SPELLINGS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_SPELLINGS = frozenset({"0", "false", "f", "no", "n", "off"})


def coerce_bool(value: Any, name: str) -> bool:
    'Coerce ``value`` to ``bool`` explicitly, raising on anything unrecognised.'
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE_SPELLINGS:
            return True
        if token in _FALSE_SPELLINGS:
            return False
        raise ValueError(
            f"{name} must be one of {sorted(_TRUE_SPELLINGS)} or "
            f"{sorted(_FALSE_SPELLINGS)}; got {value!r}. Refusing to guess: every "
            f"non-empty string is truthy in Python, so guessing would silently "
            f"select the degrees convention."
        )
    raise ValueError(
        f"{name} must be a bool or a recognised boolean string; got {value!r} of "
        f"type {type(value).__name__}"
    )


def resolve_use_degrees(value: "bool | str | None" = None) -> bool:
    'Resolve the joint-value convention: argument, then env, then the default.'
    if value is None:
        raw = os.getenv("DUME_USE_DEGREES")
        if raw is None or not raw.strip():
            return DEFAULT_USE_DEGREES
        return coerce_bool(raw, "DUME_USE_DEGREES")
    return coerce_bool(value, "use_degrees")


# The per-step motion clamp
# PROVENANCE OF THIS VALUE — derived arithmetic on recorded statistics, NOT a
DEFAULT_MAX_RELATIVE_TARGET: float = 160.0

# Divergence threshold for deciding the clamp fired, identical to the
# `abs(safe_goal_pos - goal_pos) > 1e-4` test inside
# `lerobot.robots.utils.ensure_safe_goal_position`. Held equal on purpose: a
# different threshold would let Dum-E and upstream disagree about whether a given
# action was clamped.
CLAMP_DIVERGENCE_THRESHOLD: float = 1e-4

# Upstream's exact clamp wording, re-emitted verbatim inside Dum-E's own warning
# so a single grep finds both signals (research Pitfall 9).
CLAMP_WARNING_TEXT = "Relative goal position magnitude had to be clamped to be safe."

# Environment spellings that explicitly DISABLE the clamp. Disabling has to be
# spelled out, because `None` on the constructor parameter means "resolve from
# the environment then the default" — the clamp is on by default, which is the
# point, given that the production call site passes neither this parameter nor
# the units one.
_CLAMP_DISABLED_SPELLINGS = frozenset({"none", "null", "off", "disabled", "false"})


def resolve_max_relative_target(
    value: "float | Dict[str, float] | None" = None,
) -> "float | Dict[str, float] | None":
    'Resolve and validate the per-step motion clamp before it reaches the config.'
    if value is None:
        raw = os.getenv("DUME_MAX_RELATIVE_TARGET")
        if raw is None or not raw.strip():
            value = DEFAULT_MAX_RELATIVE_TARGET
        elif raw.strip().lower() in _CLAMP_DISABLED_SPELLINGS:
            return None
        else:
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f"DUME_MAX_RELATIVE_TARGET must be a positive float, or one of "
                    f"{sorted(_CLAMP_DISABLED_SPELLINGS)} to disable the clamp; "
                    f"got {raw!r}"
                ) from exc

    if isinstance(value, dict):
        missing = sorted(set(_MOTOR_NAMES) - set(value))
        extra = sorted(set(value) - set(_MOTOR_NAMES))
        if missing or extra:
            raise ValueError(
                "A per-motor max_relative_target mapping must cover exactly the "
                f"motor names {list(_MOTOR_NAMES)} — upstream raises when the key "
                f"sets differ, and only when the clamp actually engages. "
                f"Missing: {missing}. Unexpected: {extra}."
            )
        resolved = {str(motor): float(bound) for motor, bound in value.items()}
        bad = {m: b for m, b in resolved.items() if not (b > 0.0) or not math.isfinite(b)}
        if bad:
            raise ValueError(
                f"Every per-motor max_relative_target bound must be a positive "
                f"finite float; got {bad}"
            )
        return resolved

    # `bool` is a subclass of `int`, and `float(True) == 1.0` would silently
    # become a 1-unit clamp that fires on every action.
    if isinstance(value, bool):
        raise ValueError(
            f"max_relative_target must be a positive float or a per-motor "
            f"mapping, not a bool; got {value!r}"
        )
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"max_relative_target must be a positive float, a per-motor mapping, "
            f"or None; got {value!r} of type {type(value).__name__}"
        )

    resolved_float = float(value)
    if not math.isfinite(resolved_float) or resolved_float <= 0.0:
        raise ValueError(
            f"max_relative_target must be a positive finite float — a value of "
            f"zero or below would clamp every action to the present position, and "
            f"a non-finite one disables the comparison; got {value!r}"
        )
    return resolved_float


def diff_clamped_joints(
    requested: Dict[str, float],
    sent: Dict[str, float],
    threshold: float = CLAMP_DIVERGENCE_THRESHOLD,
) -> Tuple[Tuple[str, float, float], ...]:
    'Return ``(joint, requested, clipped)`` for every joint the clamp moved.'
    clamped: List[Tuple[str, float, float]] = []
    for key, requested_value in requested.items():
        if key not in sent:
            continue
        requested_float = float(requested_value)
        sent_float = float(sent[key])
        if abs(sent_float - requested_float) > threshold:
            clamped.append((key, requested_float, sent_float))
    return tuple(clamped)


# ============================================================================
# Hardware wrapper built on LeRobot API
# ============================================================================


class SO10xArmController(IRobotController):
    """Thin wrapper for SO-100/101 follower arms using LeRobot API.

    Exposes convenience methods for robot agents while delegating to the underlying
    `Robot` implementation.
    """

    def __init__(
        self,
        robot_type: str = "so101_follower",
        robot_port: Optional[str] = None,
        robot_id: str = "my_awesome_follower_arm",
        *,
        wrist_cam_idx: int = 0,
        front_cam_idx: int = 1,
        use_degrees: "bool | str | None" = None,
        max_relative_target: "float | Dict[str, float] | None" = None,
        robot_factory=None,
    ) -> None:
        if robot_port is None:
            robot_port = os.getenv("SO_ARM_PORT")
            if not robot_port:
                raise ValueError(
                    "Robot serial port is required. Set `port` or env `SO_ARM_PORT`."
                )

        # `None` here means "resolve", not "disable" — mirroring the
        # serial-port fallback just above. The clamp is therefore ON by default,
        max_relative_target = resolve_max_relative_target(max_relative_target)

        # Same fallback shape for the units convention. The environment string is coerced
        # explicitly rather than trusted for truthiness.
        use_degrees = resolve_use_degrees(use_degrees)

        # The EFFECTIVE value has to be visible in the log, so the running
        # convention and the active clamp can be read off a session without
        # reconstructing the config precedence chain.
        logger.info(
            "Controller units/safety config: use_degrees={} max_relative_target={} "
            "(recorded units verdict is percent mode — see docs/UNITS-VERDICT.md; "
            "the flip is deferred, so this ships at its current effective value)",
            use_degrees,
            max_relative_target,
        )

        # Store robot_id for the id property
        self._robot_id = robot_id

        # What the connect-time `Goal_Position` pre-arm found and wrote, so a
        # caller can report it instead of re-implementing the pre-arm. `None`
        # until `connect()` runs.
        self.last_prearm_record: Optional[Dict[str, Any]] = None

        cameras = {
            "wrist": OpenCVCameraConfig(
                index_or_path=wrist_cam_idx, fps=30, width=640, height=480
            ),
            "front": OpenCVCameraConfig(
                index_or_path=front_cam_idx, fps=30, width=640, height=480
            ),
        }

        # Build the appropriate config subclass for the chosen robot type.
        # `robot_type` stays "so101_follower"/"so100_follower": lerobot 0.6.1
        if robot_type == "so101_follower":
            from lerobot.robots.so_follower import SO101FollowerConfig

            self.config = SO101FollowerConfig(
                id=robot_id,
                port=robot_port,
                cameras=cameras,
                use_degrees=use_degrees,
                max_relative_target=max_relative_target,
                # Declarative. Upstream `configure()` writes these during
                # `connect()`, so the preset lands on the first pass — no
                # post-connect register overwrite, no second torque-disabled
                # cycle. Verified by `_assert_pid_landed()`.
                position_p_coefficient=DUME_PID["P_Coefficient"],
                position_i_coefficient=DUME_PID["I_Coefficient"],
                position_d_coefficient=DUME_PID["D_Coefficient"],
            )
        elif robot_type == "so100_follower":
            # Fall back to SO-100 if desired
            from lerobot.robots.so_follower import SO100FollowerConfig  # type: ignore

            self.config = SO100FollowerConfig(
                id=robot_id,
                port=robot_port,
                cameras=cameras,
                use_degrees=use_degrees,
                max_relative_target=max_relative_target,
                # Same as the SO-101 branch: an SO-100 arm is not
                # entitled to the library defaults either.
                position_p_coefficient=DUME_PID["P_Coefficient"],
                position_i_coefficient=DUME_PID["I_Coefficient"],
                position_d_coefficient=DUME_PID["D_Coefficient"],
            )
        else:
            raise ValueError(f"Unsupported robot_type: {robot_type}")

        # `SO100Follower is SO101Follower is SOFollower` at 0.6.1, so the single
        # consolidated class is the exact annotation for both branches.
        self.robot: SOFollower = (
            make_robot_from_config if robot_factory is None else robot_factory
        )(self.config)

        # Cache ordering used for vector<->dict conversions
        self._state_keys: List[str] = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

    @property
    def id(self) -> str:
        """Unique identifier for the robot controller."""
        return self._robot_id

    @property
    def camera_keys(self) -> List[str]:
        return list(self.robot.cameras.keys())

    @property
    def robot_state_keys(self) -> List[str]:
        return list(self._state_keys)

    def connect(self, calibrate: bool = True) -> None:
        # Pre-arm Goal_Position FIRST, while torque is still off. `robot.connect()`
        # below re-enables torque, and a stale `Goal_Position` of 0 would become a
        self.last_prearm_record = self._prearm_goal_to_present()
        self.robot.connect(calibrate=calibrate)
        # Assert the calibration FILE loaded before anything reads an observation.
        # `connect()` succeeding is not evidence: the calibrated flag reads the
        # motors, so a firmware-calibrated arm makes connect skip calibration
        # while the in-Python mapping stays empty, and the failure then surfaces
        # at the first bus read instead of here.
        self._assert_calibration_loaded()
        # The Dum-E PID preset is declared on the config and written by upstream
        # `configure()` during the connect above. What remains is proving it
        # landed — and refusing to operate the arm if it did not.
        self._assert_pid_landed()

    def _prearm_goal_to_present(self) -> Dict[str, Any]:
        'Write ``Goal_Position <- Present_Position`` while torque is still OFF.'
        bus = self.robot.bus
        num_retry = max(
            _PREARM_READ_MIN_RETRIES,
            int(getattr(self.config, "num_read_retries", 0) or 0),
        )
        bus.connect()
        try:
            torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=num_retry)
            present = bus.sync_read("Present_Position", normalize=False, num_retry=num_retry)
            goal_before = bus.sync_read("Goal_Position", normalize=False, num_retry=num_retry)
            record: Dict[str, Any] = {
                "torque_enable_before": dict(torque),
                "present_ticks": dict(present),
                "goal_ticks_before": dict(goal_before),
                "worst_pending_jump_ticks": max(
                    (abs(goal_before[m] - present[m]) for m in present), default=0
                ),
            }
            if any(value for value in torque.values()):
                record.update(
                    prearmed=False,
                    reason="torque already enabled — Goal_Position is live",
                )
                logger.info(
                    "Goal_Position pre-arm skipped: torque already enabled, goal is live"
                )
                return record

            for motor, tick in present.items():
                bus.write(
                    "Goal_Position", motor, int(tick), normalize=False, num_retry=num_retry
                )
            goal_after = bus.sync_read("Goal_Position", normalize=False, num_retry=num_retry)
            mismatched = {
                motor: (present[motor], goal_after[motor])
                for motor in present
                if abs(goal_after[motor] - present[motor]) > _PREARM_TICK_TOLERANCE
            }
            if mismatched:
                raise RuntimeError(
                    "Goal_Position pre-arm did not take, so enabling torque would "
                    f"command a jump. Refusing to connect. present vs goal: {mismatched}"
                )
            record.update(
                prearmed=True,
                reason="written and verified",
                goal_ticks_after=dict(goal_after),
            )
            logger.info(
                "Goal_Position pre-armed to present position on {} motors "
                "(neutralised a worst-case {} tick jump at torque-enable)",
                len(present),
                record["worst_pending_jump_ticks"],
            )
            return record
        finally:
            # disable_torque=False: leave the torque state EXACTLY as found. The
            # default (True) would make this guard mutate the state it exists to
            # reason about, and would drop torque under load on a powered arm.
            bus.disconnect(False)

    def _assert_calibration_loaded(self) -> Tuple[Path, str]:
        'Confirm the calibration file loaded, logging its path and checksum.'
        robot = getattr(self, "robot", None)
        path, checksum = assert_calibration_loaded(
            robot_name=getattr(robot, "name", None),
            robot_id=self._robot_id,
            expected_path=getattr(robot, "calibration_fpath", None),
        )
        logger.info(
            "Loaded LeRobot calibration: path={} sha256={}", path, checksum
        )
        return path, checksum

    def disconnect(self) -> None:
        self.robot.disconnect()

    def is_connected(self) -> bool:
        return self.robot.is_connected

    def get_observation(self) -> Dict[str, Any]:
        return self.robot.get_observation()

    def _assert_pid_landed(self) -> Dict[str, Dict[str, int]]:
        'Read the PID preset back from every motor and assert it landed.'
        num_retry = max(
            _PID_READ_MIN_RETRIES,
            int(getattr(self.config, "num_read_retries", 0) or 0),
        )

        observed: Dict[str, Dict[str, int]] = {}
        mismatches: List[str] = []
        for motor in self.robot.bus.motors:
            observed[motor] = {}
            for register, expected in DUME_PID.items():
                try:
                    actual = self.robot.bus.read(
                        register, motor, normalize=False, num_retry=num_retry
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Could not read {register} from motor {motor!r} to verify "
                        f"the Dum-E PID preset {DUME_PID}: the arm will NOT be "
                        f"operated at unknown stiffness. Fix the motor bus rather "
                        f"than softening this check."
                    ) from exc
                observed[motor][register] = actual
                if actual != expected:
                    mismatches.append(
                        f"{motor}.{register}: expected {expected}, observed {actual}"
                    )

        if mismatches:
            # Aggregated on purpose: a read-back that is neither the Dum-E preset
            # nor the library default means the write PARTIALLY landed, which is
            # more dangerous than either endpoint. One message per connect shows
            # which motors took the write and which did not.
            raise ValueError(
                "Dum-E PID preset did not land on the motor bus; refusing to "
                f"operate the arm at unknown stiffness. Expected {DUME_PID}. "
                f"Mismatches: {'; '.join(mismatches)}"
            )

        logger.info("PID read-back verified on every motor: {}", observed)
        return observed

    # ------------------------ Convenience methods ------------------------
    def get_current_state(self) -> np.ndarray:
        obs = self.get_observation()
        return np.array([float(obs[k]) for k in self._state_keys], dtype=np.float64)

    def get_current_images(self) -> Dict[str, np.ndarray]:
        obs = self.get_observation()
        images: Dict[str, np.ndarray] = {}
        for cam in self.camera_keys:
            images[cam] = obs[cam]
        return images

    def set_target_state(self, target_state: Any) -> Dict[str, float]:
        """Accepts a 6-vector (np/torch) or a dict of `*.pos` keys."""
        action_dict: Dict[str, float]
        if isinstance(target_state, dict):
            action_dict = {str(k): float(v) for k, v in target_state.items()}
        else:
            # numpy / torch tensor
            if hasattr(target_state, "detach"):
                target_state = target_state.detach().cpu().numpy()
            target_state = np.asarray(target_state, dtype=np.float64).reshape(-1)
            assert target_state.shape[0] == 6, "Expected 6-dof target state"
            action_dict = {
                k: float(target_state[i]) for i, k in enumerate(self._state_keys)
            }

        sent = self.robot.send_action(action_dict)
        sent_dict = {str(k): float(v) for k, v in sent.items()}
        self._report_clamped_joints(action_dict, sent_dict)
        return sent_dict

    def _report_clamped_joints(
        self, requested: Dict[str, float], sent: Dict[str, float]
    ) -> Tuple[Tuple[str, float, float], ...]:
        "Re-emit any clamp divergence on Dum-E's own loguru stream."
        clamped = diff_clamped_joints(requested, sent)
        if not clamped:
            return ()

        detail = ", ".join(
            f"{joint} requested={requested_value:.4f} clipped={clipped_value:.4f}"
            for joint, requested_value, clipped_value in clamped
        )
        logger.warning(
            "{} max_relative_target={} clamped {} joint(s): {}",
            CLAMP_WARNING_TEXT,
            getattr(self.config, "max_relative_target", None),
            len(clamped),
            detail,
        )
        return clamped

    def move_to_initial_pose(self) -> None:
        'Park at the initial pose, always reached via the retracted ready pose.'
        self.move_to_ready_pose()
        # These target degrees mirror legacy behavior
        self.set_target_state(
            np.array([0.0, -102, 96.0, 76.0, -90.0, 0.0], dtype=np.float64)
        )
        time.sleep(1.0)

    def move_to_ready_pose(self) -> None:
        self.set_target_state(
            np.array([0.0, -90, 75.0, 75.0, -90.0, 0.0], dtype=np.float64)
        )
        time.sleep(1.0)

    def move_to_remote_pose(self) -> None:
        self.set_target_state(
            np.array([0.0, 0.0, 0.0, 50.0, -90.0, 60.0], dtype=np.float64)
        )
        time.sleep(1.0)

    def release_at_remote_pose(self, location: Literal["left", "right"]) -> None:
        """
        This is a pre-defined sequence of poses that the robot will move to release the item relative to the front camera.
        """
        gripper_state = float(self.get_current_state()[-1])
        random_offset = float(np.random.uniform(-5.0, 5.0))
        if location == "left":
            random_offset += 45.0
        else:
            random_offset -= 45.0

        sequence = [
            [random_offset, 0.0, 0.0, 50.0, -90.0, gripper_state],
            [random_offset, 45.0, -45.0, 50.0, -90.0, gripper_state],
            [random_offset, 45.0, -45.0, 50.0, -90.0, min(gripper_state + 10.0, 60.0)],
            [random_offset, 0.0, 0.0, 50.0, -90.0, min(gripper_state + 10.0, 60.0)],
        ]
        for state in sequence:
            self.set_target_state(np.array(state, dtype=np.float64))
            time.sleep(0.5)
        self.move_to_remote_pose()


@dataclass
class EvalConfig:
    # SO-ARM10x robot configuration
    robot_type: str = "so101_follower"
    robot_id: str = "my_awesome_follower_arm"
    robot_port: str = "/dev/tty.usbmodem5A680102371"
    wrist_cam_idx: int = 0
    front_cam_idx: int = 1

    # Policy/eval parameters
    policy_host: str = "localhost"
    policy_port: int = 5555
    # Default action horizon pinned to 16 to match the trained/eval checkpoint
    # (the legacy default of 8 truncated the learned action chunk). The agent
    # path uses PickSkill(action_horizon=16).
    action_horizon: int = 16
    lang_instruction: str = "Grab a banana and put it on the plate"
    play_sounds: bool = False
    timeout: int = 60
    show_images: bool = True


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Step 1: Initialize the robot (wrapper)
    robot = SO10xArmController(
        robot_type=cfg.robot_type,
        robot_port=cfg.robot_port,
        robot_id=cfg.robot_id,
        wrist_cam_idx=cfg.wrist_cam_idx,
        front_cam_idx=cfg.front_cam_idx,
    )
    # Safe connection and initial pose handled inside the eval loop via context manager

    # get camera/state keys
    camera_keys = robot.camera_keys
    print("camera_keys:", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction

    # NOTE: for so100/so101, this should be:
    # ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    robot_state_keys = robot.robot_state_keys
    print("robot_state_keys:", robot.robot_state_keys)

    # Step 2: Initialize the policy
    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
        show_images=cfg.show_images,
    )
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # Step 3: Run the Eval Loop with safe connect/disconnect
    try:
        with robot.activate():
            print("Current robot state:", robot.get_current_state())
            robot.move_to_initial_pose()
            robot.move_to_ready_pose()

            while True:
                observation_dict = robot.get_observation()
                print("observation_dict", observation_dict.keys())
                action_list = policy.get_action(observation_dict, language_instruction)

                horizon = min(cfg.action_horizon, len(action_list))
                for i in range(horizon):
                    action_dict = action_list[i]
                    print("action_dict", action_dict.values())
                    robot.set_target_state(action_dict)
                    time.sleep(0.05)
    except KeyboardInterrupt:
        logging.info(
            "KeyboardInterrupt received. Disconnecting robot and exiting eval."
        )
        # Context manager handles disconnect


if __name__ == "__main__":
    # Support both: 1) YAML/JSON config file and 2) draccus CLI
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    parser.add_argument("--instruction", type=str, help="Robot instruction")
    args, unknown = parser.parse_known_args()

    if args.config:
        cfg_root = load_config_file(args.config)
        ctrl = (
            cfg_root.get("controller", {})
            if isinstance(cfg_root.get("controller"), dict)
            else {}
        )
        cfg = EvalConfig(
            robot_type=ctrl.get("robot_type", "so101_follower"),
            robot_id=ctrl.get("robot_id", "my_awesome_follower_arm"),
            robot_port=ctrl.get("robot_port", "/dev/ttyUSB0"),
            wrist_cam_idx=int(ctrl.get("wrist_cam_idx", 0)),
            front_cam_idx=int(ctrl.get("front_cam_idx", 1)),
            policy_host=ctrl.get("policy_host", "localhost"),
            policy_port=int(ctrl.get("policy_port", 5555)),
            action_horizon=int(ctrl.get("action_horizon", 16)),
            lang_instruction=args.instruction
            or ctrl.get("lang_instruction", "Grab a banana and put it on the plate"),
            play_sounds=bool(ctrl.get("play_sounds", False)),
            timeout=int(ctrl.get("timeout", 60)),
            show_images=bool(ctrl.get("show_images", False)),
        )
        eval(cfg)
    else:
        # Fall back to draccus CLI (supports the dataclass fields directly)
        eval()
