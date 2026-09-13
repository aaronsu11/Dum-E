"""Portable numeric observation fixture; never imports or connects hardware."""
from pathlib import Path
import numpy as np
from policy_guard.contracts import JOINT_ORDER, load_numeric


def load_observation(path):
    """Read centered SO101 degrees, gripper 0–100 and RGB uint8 camera frames."""
    arrays = load_numeric(Path(path), expected_keys={"state", "front", "wrist"})
    state = arrays["state"]
    if state.shape != (6,) or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
        raise ValueError("Observation requires six finite state values")
    for role in ("front", "wrist"):
        frame = arrays[role]
        if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
            raise ValueError(f"{role}: expected RGB uint8 480x640")
    return {**dict(zip(JOINT_ORDER, map(float, state))),
            "front": arrays["front"], "wrist": arrays["wrist"]}
