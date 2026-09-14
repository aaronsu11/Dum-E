"""Explicit software frame conversions; these do not certify physical calibration.

The current controller uses centered degrees on five joints and 0–100 on the
gripper. Never infer a frame from numeric ranges or silently change the controller.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from embodiment.so_arm10x.mappings.galaxea import JOINTS, arm_to_model, model_to_arm, vector


def calibration_scale(path):
    """Centered LeRobot degrees per RANGE_M100_100 unit, gripper unchanged."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    scales = []
    for key in JOINTS[:-1]:
        motor = data[key.removesuffix(".pos")]
        lo, hi = motor["range_min"], motor["range_max"]
        if not 0 <= lo < hi <= 4095:
            raise ValueError(f"Invalid calibration range for {key}")
        scales.append((hi - lo) * 360 / (4095 * 200))
    return np.array([*scales, 1.], dtype=np.float32), hashlib.sha256(raw).hexdigest()


def to_model_frame(values, profile, *, calibration_path=None):
    values = vector(values)
    if profile == "pi05-so101":
        # Project-IRA records with LeRobot 0.5.1's use_degrees=True default.
        # Mean/std normalization belongs to the saved checkpoint processor.
        return values.copy()
    if profile in ("g05-so101", "molmoact2-so101"):
        return arm_to_model(values)
    if profile == "groot-so101":
        if calibration_path is None:
            raise ValueError("Explicit calibration required for normalized GR00T frame")
        return values / calibration_scale(calibration_path)[0]
    if profile == "pi05-base":
        raise ValueError("Pi0.5 base has no verified SO101 action/state mapping")
    raise ValueError(f"Unknown SO101 profile: {profile}")


def to_arm_frame(values, profile, *, calibration_path=None):
    values = vector(values)
    if profile == "pi05-so101":
        return values.copy()
    if profile in ("g05-so101", "molmoact2-so101"):
        return model_to_arm(values)
    if profile == "groot-so101":
        if calibration_path is None:
            raise ValueError("Explicit calibration required for normalized GR00T frame")
        return values * calibration_scale(calibration_path)[0]
    if profile == "pi05-base":
        raise ValueError("Pi0.5 base has no verified SO101 action/state mapping")
    raise ValueError(f"Unknown SO101 profile: {profile}")
