"""Galaxea adapter for the shared backend contract. No serial or camera ownership."""
import os
import threading
from shared import IPolicyBackend
from .modalities import JOINTS, make_observation, decode_action
from .session import GalaxeaSession


class GalaxeaPolicyBackend(IPolicyBackend):
    def __init__(self, host="127.0.0.1", port=None, camera_keys=None,
                 robot_state_keys=None, show_images=False, language_instruction=None):
        if camera_keys is not None and set(camera_keys) != {"front", "wrist"}:
            raise ValueError("Galaxea requires explicit front and wrist cameras")
        if robot_state_keys is not None and set(robot_state_keys) != set(JOINTS):
            raise ValueError("Galaxea requires the six named SO101 joints")
        self._instruction = language_instruction
        self._lock = threading.RLock()
        if host == "localhost":
            host = "127.0.0.1"
        self._session = GalaxeaSession(
            f"ws://{host}:{port or int(os.getenv('DUME_GALAXEA_POLICY_PORT', '8765'))}")
        self.last_metadata = None

    @property
    def language_instruction(self):
        return self._instruction

    def set_lang_instruction(self, lang_instruction):
        if not isinstance(lang_instruction, str) or not lang_instruction.strip():
            raise ValueError("Nonempty instruction required")
        with self._lock:
            self._instruction = lang_instruction
            self.reset()

    def get_action(self, observation_dict, lang=None, *, seed=None):
        with self._lock:
            self._session.reset()
            raw = make_observation(observation_dict, lang or self._instruction, seed=seed)
            actions = []
            try:
                for index in range(32):
                    reply = self._session.request(raw if index == 0 else {})
                    if reply.get("need_obs") is not (index == 31):
                        raise ValueError("Unexpected chunk-cache boundary")
                    actions.append(decode_action(reply["action"]))
                    if index == 0:
                        self.last_metadata = {k: v for k, v in reply.items()
                                              if k in ("timings", "health", "cot_text",
                                                       "generated_tokens", "seed")}
                return actions
            except Exception:
                self._session.close()
                raise

    def ping(self):
        with self._lock:
            try:
                return self._session.request({"__health__": True}).get("status") == "ready"
            except Exception:
                return False

    def reset(self):
        with self._lock:
            if self._session.socket is not None:
                self._session.reset()

    def close(self):
        with self._lock:
            self._session.close()
