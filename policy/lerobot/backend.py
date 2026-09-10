"""The ``lerobot`` :class:`~shared.IPolicyBackend`: LeRobot's async policy server
over loopback gRPC.

This module is the ADAPTER, not the transport. ``policy/lerobot/session.py`` owns
the channel and the four-method wire; this class owns the things the wire has no
opinion about: the stored language instruction, the handshake payload, the
per-episode reset, and — the load-bearing one — the flat ``(16, 6)``-to-named
action reindex that makes ``get_action`` return the SAME
``list[dict["<joint>.pos", float]]`` contract that ``groot-native`` returns from
its modality dict.

Every member is deliberately SYNCHRONOUS. Concurrency is handled at the ``@tool``
boundary in ``embodiment/so_arm10x/agent.py`` via
``await asyncio.to_thread(sync_method, *args)``; an ``async def`` here would break
that offload pattern and every existing call site, and this module must never
create an event loop of its own.

==================== THE D-11 HANDSHAKE VALUES ====================
Four handshake values plus the port are read from the environment, each with a
documented default, and each is a visible ``controller.lerobot_*`` key in
``config.example.yaml``. ``dum_e.py`` forwards a key ONLY when the config names
it, so every default lives in exactly one place — right here — and cannot drift
between the launcher and the process that owns inference.

``actions_per_chunk`` is the one that matters most: **40 is the well-lit wrong
value**, and it has TWO sources (``GrootConfig``'s own default AND the
checkpoint's ``config.json: action_horizon: 40``), while 16 is correct. The
constructor therefore logs every effective value on ONE line, because D-11's whole
point is that this number must be readable from a log and a config file without
reading code.

==================== WHY rename_map IS {} ====================
``RemotePolicyConfig.rename_map`` is deliberately empty and must not be pressed
into service as the flat-to-named mapper. It feeds
``RenameObservationsProcessorStep`` *inside* the preprocessor
(``policy_server.py:160-163``, ``processor_groot.py:1255``), which runs strictly
AFTER ``raw_observation_to_observation`` has already indexed the observation by
key — so a ``KeyError`` raised in that earlier step can never be repaired by a
rename that runs later. See ``policy/lerobot/features.py`` for the full note.
"""

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
    """The ``lerobot`` policy backend over ``LeRobotPolicySession``.

    Accepts exactly the keyword set ``Gr00tRobotInferenceClient`` accepts, so
    ``policy.factory.make_policy_backend(**kwargs)`` stays uniform across
    backends and ``embodiment/so_arm10x/agent.py``'s
    ``make_policy_backend(host=policy_host)`` call site needs no change. Every
    value that call site does NOT pass is therefore reachable from the
    environment instead.
    """

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
        """Rebuild the channel and flush the server's per-client observation state.

        **Deliberately does NOT clear ``_handshaken``.** Upstream reloads the
        policy INSIDE ``SendPolicyInstructions`` (``policy_server.py:151``), so a
        re-handshake is not a cheap re-ask of a finished question — it is one more
        full multi-GB weight materialization on a GPU that is already holding one
        (this phase measured ``cuda_allocated_MiB=6015`` on a 12288 MiB card, so
        two live copies do not fit). ``scripts/run_pick_baseline.py`` calls this
        between every scored attempt and ``shared``'s ``session()`` calls it on
        entry, so a reload here is a per-episode OOM risk and a multi-minute stall,
        not a one-off cost.

        ``Ready`` alone is what this method actually needs: it calls upstream's
        ``_reset_server()`` (``policy_server.py:107-113``), which clears
        ``observation_queue`` and ``_predicted_timesteps`` and leaves the loaded
        policy in place. It does NOT clear ``last_processed_obs`` — nothing a
        client can send does — which is why every observation carries
        ``must_go=True``; that flag short-circuits the similarity filter
        ``last_processed_obs`` feeds (see ``session.py``'s ``infer``).

        When there IS server-side state to flush — i.e. this backend has already
        handshaken — the ``Ready`` RAISES on an unreachable or refusing server
        rather than returning quietly, unlike :meth:`ping`. A reset that silently
        did nothing is how stale per-client state survives into the next episode
        and then reads as policy drift.

        Before the first handshake the probe is SKIPPED, deliberately: there is no
        per-client state on the server to flush yet, so there is nothing that can
        go stale, and this backend connects lazily — ``IPolicyBackend.session()``
        calls ``reset()`` on entry, and probing there would turn "the server is not
        up yet" into a failure at scope entry instead of at the first
        ``get_action()``, where the handshake's own error message is. The raise
        condition is therefore exactly "we had state to flush and could not".

        Raises:
            RuntimeError: if a handshaken backend's server does not answer ``Ready``.
        """
        self._session.close()
        self._session = LeRobotPolicySession(f"{self._host}:{self._port}")
        self._closed = False
        if self._handshaken:
            # Re-arm the server's per-episode state; keep the loaded weights.
            self._session.probe_ready_or_raise()

    def close(self) -> None:
        """Release the gRPC channel. Idempotent.

        A double close — an explicit ``close()`` inside a ``session()`` body plus
        the ``finally`` — must be a no-op rather than a raise over the original
        error.
        """
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
        """Run one inference step and return the horizon of named joint targets.

        Args:
            observation_dict: The FLAT observation ``IRobotController.get_observation()``
                produces — ``{"<joint>.pos": float}`` for six joints plus
                ``{cam: HxWx3 uint8}`` per camera.
            lang: Instruction to condition on; falls back to the stored one.

        Returns:
            ``list[dict["<joint>.pos", float]]`` of length ``actions_per_chunk``.

        Raises:
            ValueError: when no instruction is available (before any gRPC call).
            RuntimeError: on a transport failure or a malformed chunk.
        """
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
        # wire (embodiment/so_arm10x/controller.py builds it), whereas this wire's
        # SERVER does the packing itself via raw_observation_to_observation() plus
        # the preprocessor. The two backends' observation shapes differ BY DESIGN
        # — do not "unify" them.
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
            # iterating exactly that list
            # (lerobot/utils/feature_utils.py:131-132), so client and server agree
            # by construction rather than by coincidence. strict=True is
            # load-bearing: a silent length mismatch here is a whole-arm failure
            # that every shape check passes.
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
    """Flatten one action into a plain list of floats.

    Duck-typed rather than ``torch``-typed: the real server sends torch tensors
    and the mock sends the same, but this module has no other reason to import
    torch, and a numpy array or a plain list must decode identically.
    """
    if hasattr(action, "detach"):
        action = action.detach().cpu()
    if hasattr(action, "tolist"):
        action = action.tolist()
    return [float(value) for value in action]
