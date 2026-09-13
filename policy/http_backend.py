"""Shared loopback HTTP exchange for explicitly mapped SO101 model adapters."""
import json
import os
import threading
import urllib.request

import numpy as np

from policy.galaxea.modalities import JOINTS
from policy.so101_contract import to_model_frame, to_arm_frame
from policy_lab.profiles import get_profile
from policy_lab.protocol import encode_image
from shared import IPolicyBackend


class SO101HTTPPolicyBackend(IPolicyBackend):
    def __init__(self, port=18081, language_instruction=None, *, profile="molmoact2-so101"):
        if os.getenv("DUME_ASYNC_INFERENCE", "0") == "1":
            raise ValueError("HTTP controller bridge supports synchronous chunks only")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Invalid loopback port")
        self.endpoint = f"http://127.0.0.1:{port}"
        self.timeout_s = 10.
        self._instruction = language_instruction
        self._lock = threading.RLock()
        self.profile = get_profile(profile)
        self.last_metadata = None

    @property
    def language_instruction(self):
        return self._instruction

    def set_lang_instruction(self, lang_instruction):
        if not isinstance(lang_instruction, str) or not lang_instruction.strip():
            raise ValueError("Nonempty instruction required")
        with self._lock:
            self._instruction = lang_instruction

    def _request(self, path, data=None, *, timeout_s=None):
        raw = None if data is None else json.dumps(data, allow_nan=False).encode()
        request = urllib.request.Request(self.endpoint + path, data=raw,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(
                request, timeout=self.timeout_s if timeout_s is None else timeout_s) as response:
            if response.url != self.endpoint + path:
                raise ValueError("Inference endpoint redirected")
            payload = response.read(1024 * 1024 + 1)
            if len(payload) > 1024 * 1024:
                raise ValueError("Excessive inference response")
            return json.loads(payload)

    def _check_health(self, health):
        if (health.get("status") != "ready" or health.get("fault")
                or health.get("profile") != self.profile.to_dict()):
            raise ValueError("Policy server not ready or pinned checkpoint/profile mismatch")

    def _to_model(self, values):
        return to_model_frame(values, self.profile.name)

    def _to_arm(self, values):
        return to_arm_frame(values, self.profile.name)

    def get_action(self, observation_dict, lang=None, *, seed=20265907):
        instruction = lang or self._instruction
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1024:
            raise ValueError("Nonempty instruction of at most 1024 characters required")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Invalid seed")
        state = self._to_model([observation_dict[k] for k in JOINTS])
        images = {}
        for role in ("front", "wrist"):
            frame = np.asarray(observation_dict[role])
            if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
                raise ValueError(f"{role}: expected RGB uint8 480x640")
            images[role] = encode_image(frame)
        with self._lock:
            self._check_health(self._request("/health"))
            reply = self._request("/infer", {
                "state": state.tolist(), **images, "task": instruction, "seed": seed})
            self._check_health(reply["health"])
            if reply.get("seed") != seed:
                raise ValueError("Inference seed mismatch")
            values = np.asarray(reply["actions"])
            if (values.shape != (self.profile.horizon, 6) or values.dtype.kind not in "fiu"
                    or not np.isfinite(values).all()):
                raise ValueError(f"Expected {self.profile.horizon} finite six-joint absolute targets")
            self.last_metadata = {k: reply[k] for k in ("health", "timings", "seed")}
            self.last_metadata["server_physical_ready"] = reply.get("physical_ready")
            return [dict(zip(JOINTS, map(float, self._to_arm(v))))
                    for v in values]

    def ping(self):
        with self._lock:
            try:
                self._check_health(self._request("/health"))
                return True
            except Exception:
                return False

    def reset(self):
        # The server resets the policy before every inference request.
        pass

    def close(self):
        # Each HTTP request owns and closes its connection.
        pass
