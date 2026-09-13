"""Calibrated GR00T HTTP bridge for bounded integration trials.

The production native/LeRobot backends are unchanged. This explicit bridge maps
centered controller degrees to the checkpoint's RANGE_M100_100 convention.
"""
import hashlib
import os
from pathlib import Path

from policy.galaxea.modalities import vector
from policy.http_backend import SO101HTTPPolicyBackend
from policy.so101_contract import calibration_scale


class GrootTrialBackend(SO101HTTPPolicyBackend):
    def __init__(self, *, calibration_path, port=8081, language_instruction=None):
        if os.getenv("DUME_ASYNC_INFERENCE", "0") == "1":
            raise ValueError("GR00T HTTP trial bridge is synchronous; use LeRobot for async")
        super().__init__(port=port, language_instruction=language_instruction, profile="groot-so101")
        self.calibration_path = Path(calibration_path).resolve(strict=True)
        self.scales, self.calibration_sha256 = calibration_scale(self.calibration_path)

    def _check_calibration(self):
        if hashlib.sha256(self.calibration_path.read_bytes()).hexdigest() != self.calibration_sha256:
            raise ValueError("GR00T trial calibration changed")

    def _to_model(self, values):
        return vector(values) / self.scales

    def _to_arm(self, values):
        return vector(values) * self.scales

    def get_action(self, observation_dict, lang=None, *, seed=20265907):
        self._check_calibration()
        actions = super().get_action(observation_dict, lang, seed=seed)
        self._check_calibration()
        self.last_metadata["calibration_sha256"] = self.calibration_sha256
        self.last_metadata["units"] = "controller degrees <-> checkpoint RANGE_M100_100; gripper 0-100"
        return actions
