"""Experimental pinned SO101 Pi0.5 bridge; saved server processors own scaling."""
import os
import time
import numpy as np

from policy.molmo_backend import MolmoPolicyBackend
from policy.galaxea.modalities import JOINTS
from policy_lab.profiles import get_profile
from policy_lab.protocol import encode_image, prefix_digest


class Pi05SO101PolicyBackend(MolmoPolicyBackend):
    def __init__(self, host="127.0.0.1", port=None, camera_keys=None,
                 robot_state_keys=None, show_images=False, language_instruction=None):
        if os.getenv("DUME_ASYNC_INFERENCE", "0") == "1":
            raise ValueError("Pi0.5 SO101 bridge currently supports synchronous chunks only")
        if host not in ("127.0.0.1", "localhost"):
            raise ValueError("Pi0.5 SO101 requires a loopback server or tunnel")
        if camera_keys is not None and set(camera_keys) != {"front", "wrist"}:
            raise ValueError("Pi0.5 SO101 requires front and wrist cameras")
        if robot_state_keys is not None and set(robot_state_keys) != set(JOINTS):
            raise ValueError("Pi0.5 SO101 requires the six named joints")
        if show_images:
            raise ValueError("Image display is not implemented for the Pi0.5 bridge")
        super().__init__(port=port or int(os.getenv("DUME_PI05_POLICY_PORT", "18081")),
                         language_instruction=language_instruction)
        self.profile = get_profile("pi05-so101")

    def rtc_ping(self):
        """Independent health connection: never wait for the inference lock."""
        try:
            health = self._request("/health", timeout_s=0.1)
            self._check_health(health)
            return health.get("rtc_contract") == "pi05-bounded-prefix-v1"
        except Exception:
            return False

    def get_rtc_action(self, observation, prefix, *, epoch, request_id, delay_steps):
        started = time.monotonic()
        with self._lock:
            instruction = self._instruction
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("RTC requires an instruction")
        state = np.asarray([observation[k] for k in JOINTS], dtype=float)
        if not np.isfinite(state).all():
            raise ValueError("Invalid RTC state")
        images = {}
        for role in ("front", "wrist"):
            frame = np.asarray(observation[role])
            if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                raise ValueError(f"Invalid {role} image")
            images[role] = encode_image(frame)
        if prefix is not None:
            prefix = np.asarray(prefix, dtype=float)
            if prefix.shape != (25, 6) or not np.isfinite(prefix).all():
                raise ValueError("RTC requires 25 bounded queued targets")
        rtc = {"epoch": epoch, "request_id": request_id,
               "delay_steps": delay_steps,
               "prefix_arm": None if prefix is None else prefix.tolist()}
        reply = self._request("/infer/rtc", {
            "state": state.tolist(), **images, "task": instruction,
            "seed": 20265907, "rtc": rtc})
        self._check_health(reply["health"])
        expected = {"epoch": epoch, "request_id": request_id,
                    "delay_steps": delay_steps, "prefix_sha256": prefix_digest(prefix)}
        if (reply.get("rtc") != expected or reply.get("seed") != 20265907
                or reply["health"].get("rtc_contract") != "pi05-bounded-prefix-v1"):
            raise ValueError("Stale or mismatched RTC reply")
        values = np.asarray(reply["actions"])
        if (values.shape != (50, 6) or values.dtype.kind not in "fiu"
                or not np.isfinite(values).all()):
            raise ValueError("Invalid RTC action chunk")
        metadata = {key: reply[key] for key in ("health", "timings", "seed", "rtc")}
        metadata["client_rpc_ms"] = 1000 * (time.monotonic() - started)
        return values.astype(float), metadata
