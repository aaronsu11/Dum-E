"The ``lerobot`` :class:`~shared.IPolicyBackend`: LeRobot's async policy server over loopback gRPC."

import os
from typing import Any

from loguru import logger

from policy.lerobot import features
from policy.lerobot.session import LeRobotPolicySession
from shared import IPolicyBackend

#: Env var names for the five D-11 values. Each is forwarded by ``dum_e.py`` from
#: a ``controller.lerobot_*`` config key, and an exported shell value wins.
PORT_ENV_VAR = "DUME_LEROBOT_POLICY_PORT"
POLICY_TYPE_ENV_VAR = "DUME_LEROBOT_POLICY_TYPE"
CHECKPOINT_PATH_ENV_VAR = "DUME_LEROBOT_CHECKPOINT_PATH"
ACTIONS_PER_CHUNK_ENV_VAR = "DUME_LEROBOT_ACTIONS_PER_CHUNK"
DEVICE_ENV_VAR = "DUME_LEROBOT_POLICY_DEVICE"

#: The gRPC port the ``lerobot-policy`` container publishes on loopback.
#: DISTINCT from ``groot-native``'s ZMQ 5555 on purpose — one shared key for both
#: backends is the wrong-port trap this pair exists to avoid.
DEFAULT_PORT = 8080

#: Must be a member of lerobot's own ``SUPPORTED_POLICIES``.
DEFAULT_POLICY_TYPE = "groot"

#: An IN-CONTAINER path: the client only NAMES it, the SERVER resolves it. A path
#: the server cannot resolve falls back to the hub default and would serve BASE
#: weights that look like a working policy — which is what SAFE-01/1 catches.
DEFAULT_CHECKPOINT_PATH = "/checkpoints/model"

#: 16, the checkpoint's real horizon. NOT 40. See the module docstring.
DEFAULT_ACTIONS_PER_CHUNK = 16

#: The device the SERVER puts the policy on.
DEFAULT_DEVICE = "cuda"


class LeRobotPolicyBackend(IPolicyBackend):
    'The ``lerobot`` policy backend over ``LeRobotPolicySession``.'

    def __init__(
        self,
        host: str = "localhost",
        port: int | None = None,
        camera_keys: list[str] | None = None,
        robot_state_keys: list[str] | None = None,
        show_images: bool = False,
        language_instruction: str | None = None,
    ) -> None:
        self._host = host
        # int(...)-coerced: env values are ALWAYS strings, and the stringification
        # trap documented at dum_e.py's env fan-out applies on this reading side.
        self._port = int(port if port is not None else os.getenv(PORT_ENV_VAR, DEFAULT_PORT))
        self._policy_type = os.getenv(POLICY_TYPE_ENV_VAR, DEFAULT_POLICY_TYPE)
        self._checkpoint_path = os.getenv(CHECKPOINT_PATH_ENV_VAR, DEFAULT_CHECKPOINT_PATH)
        self._actions_per_chunk = int(
            os.getenv(ACTIONS_PER_CHUNK_ENV_VAR, DEFAULT_ACTIONS_PER_CHUNK)
        )
        self._device = os.getenv(DEVICE_ENV_VAR, DEFAULT_DEVICE)

        self.camera_keys = list(camera_keys) if camera_keys else list(features.CAMERA_KEYS)
        self.robot_state_keys = (
            list(robot_state_keys) if robot_state_keys else list(features.ROBOT_STATE_KEYS)
        )
        assert (
            len(self.robot_state_keys) == 6
        ), f"robot_state_keys should be size 6, but got {len(self.robot_state_keys)}"

        self.show_images = show_images
        self._language_instruction = language_instruction
        self._closed = False
        self._handshaken = False

        # The features dict is DERIVED from upstream's own hw_to_dataset_features,
        # never hand-written, and its ["observation.state"]["names"] list is the
        # ordering authority for the reindex in get_action.
        self._features = features.build_lerobot_features(
            robot_state_keys=self.robot_state_keys,
            camera_keys=self.camera_keys,
            height=features.FRAME_HEIGHT,
            width=features.FRAME_WIDTH,
        )
        # Fail at CONSTRUCTION, not at the first action, if the derived ordering
        # and the configured keys ever diverge: a permuted state vector produces a
        # plausible-looking pose at the wrong joints, which every shape check
        # passes.
        features.assert_state_ordering(self._features, self.robot_state_keys)
        self._state_names = features.state_names(self._features)

        self._session = LeRobotPolicySession(f"{self._host}:{self._port}")

        if show_images:
            # Do NOT silently discard it: a silently ignored flag is how an
            # operator concludes the preview is BROKEN rather than absent, and then
            # debugs the camera stack instead of reading this line.
            logger.warning(
                "show_images=True is accepted for signature uniformity but is NOT "
                "wired for the lerobot backend — the value is being IGNORED. The "
                "frames go to the policy server over gRPC and there is no local "
                "preview window on this path.",
            )

        logger.info(
            "LeRobot policy backend | host={} port={} policy_type={} "
            "checkpoint_path={} actions_per_chunk={} device={}",
            self._host,
            self._port,
            self._policy_type,
            self._checkpoint_path,
            self._actions_per_chunk,
            self._device,
        )

    @property
    def language_instruction(self) -> str | None:
        """The stored instruction (read-only; set via ``set_lang_instruction``)."""
        return self._language_instruction

    def set_lang_instruction(self, lang_instruction: str) -> None:
        self._language_instruction = lang_instruction

    def ping(self) -> bool:
        """Backend reachability. Delegates to the session; never raises."""
        return self._session.ready()

    def reset(self) -> None:
        "Rebuild the channel and flush the server's per-client observation state."
        self._session.close()
        self._session = LeRobotPolicySession(f"{self._host}:{self._port}")
        self._closed = False
        if self._handshaken:
            # Re-arm the server's per-episode state; keep the loaded weights.
            self._session.probe_ready_or_raise()

    def close(self) -> None:
        'Release the gRPC channel. Idempotent.'
        if getattr(self, "_closed", False):
            return
        self._closed = True
        session = getattr(self, "_session", None)
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001 - best-effort teardown must never raise
                pass

    def get_action(
        self, observation_dict: dict[str, Any], lang: str | None = None
    ) -> list[dict[str, float]]:
        'Run one inference step and return the horizon of named joint targets.'
        # Fail closed on a missing instruction BEFORE any transport send: the
        # server accepts a null task and returns motion conditioned on nothing,
        # which reads downstream as checkpoint drift instead of as the
        # configuration error it is.
        instruction = lang or self._language_instruction
        if not instruction:
            raise ValueError(
                "No language instruction: get_action() was called without a `lang` "
                "argument and no instruction is stored on the backend. Pass `lang` "
                "or call set_lang_instruction() first — a null task description "
                "must never reach the policy."
            )

        if not self._handshaken:
            self._handshake()

        # The FLAT raw observation this wire expects. Deliberately NOT the nested
        # {video, state, language} GR00T shape: that nesting belongs to the ZMQ
        raw_observation: dict[str, Any] = {
            name: float(observation_dict[name]) for name in self._state_names
        }
        for cam in self.camera_keys:
            raw_observation[cam] = observation_dict[cam]
        # "task" is LeRobot's language key (processor_groot.py:1547
        # language_key = "task"). Omitting it is NOT an error but silently
        # substitutes the default prompt "Perform the task."
        # (lerobot/policies/groot/utils.py:239-241) — a silent degradation of
        # policy quality, so it is always supplied.
        raw_observation["task"] = instruction

        timed_actions = self._session.infer(raw_observation)

        if len(timed_actions) != self._actions_per_chunk:
            raise RuntimeError(
                f"LeRobot policy server at {self._session.address} returned "
                f"{len(timed_actions)} actions, expected {self._actions_per_chunk} "
                f"({ACTIONS_PER_CHUNK_ENV_VAR}). Refusing a chunk of the wrong "
                "length: executing a truncated chunk moves the arm partway "
                "through a learned trajectory and looks like policy drift."
            )

        actions: list[dict[str, float]] = []
        for timed_action in timed_actions:
            values = _as_float_list(timed_action.get_action())
            if len(values) != len(self._state_names):
                raise RuntimeError(
                    f"LeRobot policy server at {self._session.address} returned an "
                    f"action of width {len(values)}, expected "
                    f"{len(self._state_names)} for {self._state_names!r}."
                )
            # self._state_names — NOT a hardcoded list — is the ordering
            # authority, because build_dataset_frame builds the state vector by
            actions.append(
                {
                    name: float(value)
                    for name, value in zip(self._state_names, values, strict=True)
                }
            )
        return actions

    def _handshake(self) -> None:
        """Send the ``RemotePolicyConfig`` handshake once per session."""
        # Imported here rather than at module scope only for symmetry with the
        # session's own upstream imports; the module already pays for lerobot.
        from lerobot.async_inference.helpers import RemotePolicyConfig

        self._session.connect(
            RemotePolicyConfig(
                policy_type=self._policy_type,
                pretrained_name_or_path=self._checkpoint_path,
                lerobot_features=self._features,
                actions_per_chunk=self._actions_per_chunk,
                device=self._device,
                # {} on purpose — see the module docstring's rename_map section.
                rename_map={},
            )
        )
        self._handshaken = True


def _as_float_list(action: Any) -> list[float]:
    'Flatten one action into a plain list of floats.'
    if hasattr(action, "detach"):
        action = action.detach().cpu()
    if hasattr(action, "tolist"):
        action = action.tolist()
    return [float(value) for value in action]
