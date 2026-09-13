"""Small-motion plumbing checks, explicitly distinct from policy task evaluation."""
import numpy as np
from policy.galaxea.modalities import JOINTS, vector

PROTOCOL = {
    "profile": "g05-so101", "scheduler": "sync", "chunks": 1,
    "actions": 32, "period_s": 0.05, "max_step": 0.25,
    "max_excursion": 5.0, "reset_pose": None,
    "instruction": "Grab a banana and put it on the plate",
    "accuracy_scored": False,
}


def check_camera(frame, name):
    frame = np.asarray(frame)
    if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise ValueError(f"{name}: expected RGB uint8 480x640")
    lo, hi = np.percentile(frame, [1, 99])
    if hi - lo < 5:
        raise ValueError(f"{name}: nearly uniform/black image; check lens and exposure")
    return {"shape": list(frame.shape), "p01": float(lo), "p99": float(hi)}


def joint_limits(calibration):
    spans = []
    for key in JOINTS[:-1]:
        row = calibration[key.removesuffix(".pos")]
        low, high = row["range_min"], row["range_max"]
        if not 0 <= low < high <= 4095:
            raise ValueError(f"Invalid calibration span: {key}")
        spans.append((high - low) * 360 / (2 * 4095))
    return np.array([-v for v in spans] + [0.]), np.array(spans + [100.])


def bounded_command(raw, observed, previous, origin, limits):
    """Project deliberately; preserve raw predictions separately in evidence.

    All arm limits are degrees; gripper limits are normalized 0–100 points.
    Intersect limits around both the measured pose and the previous command,
    so lag cannot accumulate into a later jump.
    """
    raw, observed, previous, origin = map(vector, (raw, observed, previous, origin))
    lower, upper = limits
    if np.any(observed < lower) or np.any(observed > upper):
        raise ValueError("Observed pose outside calibrated limits")
    if np.any(abs(observed - origin) > PROTOCOL["max_excursion"] + 0.5):
        raise ValueError("Observed excursion exceeded trial envelope")
    step, span = PROTOCOL["max_step"], PROTOCOL["max_excursion"]
    low = np.maximum.reduce([lower, origin - span, observed - step, previous - step])
    high = np.minimum.reduce([upper, origin + span, observed + step, previous + step])
    if np.any(low > high):
        raise ValueError("Measured and commanded pose diverged; stop trial")
    return np.clip(raw, low, high)
