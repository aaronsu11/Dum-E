"""Isaac GR00T client implementing the common policy interface."""
from typing import Any, Dict, List, Optional
import numpy as np
from shared import IPolicyBackend
from .service import ExternalRobotInferenceClient
from shared.visualization import view_img

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
