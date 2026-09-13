"""Named SO101 frame conversion from the pinned Galaxea reference client.

These are reference conventions, not empirical approval of a robot calibration.
The five joints are degrees; the gripper retains LeRobot's 0–100 normalization.
"""
import numpy as np
CHECKPOINT_REVISION = "e312be81e90c56a55bcb26b57429bd39a335b449"

JOINTS = ("shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos")
FRAME = {
    "shoulder_pan.pos": (1., 0.),
    "shoulder_lift.pos": (-1., 90.),
    "elbow_flex.pos": (1., 90.),
    "wrist_flex.pos": (1., 0.),
    "wrist_roll.pos": (1., 0.),
    "gripper.pos": (1., 0.),
}
CAMERAS = {"front": "exterior", "wrist": "wrist_right"}
EXPECTED_ACTION_GROUPS = frozenset({"right_arm"})
CODEC_PARTS = {"left_control": 9, "left_gripper": 1,
               "right_control": 9, "right_gripper": 1}
UNUSED_CODEC_GROUPS = frozenset({"left_control", "left_gripper", "right_gripper"})


def validate_codec_presence(absent):
    """SO101 packs all six values in right_control; the other slots are virtual."""
    if set(absent) != UNUSED_CODEC_GROUPS:
        raise ValueError(f"Unexpected SO101 codec presence: absent={absent}")


def vector(values):
    result = np.asarray(values)
    if result.dtype.kind not in "iuf" or result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError("Expected six finite numeric SO101 values")
    return result.astype(np.float32)


def arm_to_model(values):
    values = vector(values)
    return np.array([values[i] * FRAME[key][0] + FRAME[key][1]
                     for i, key in enumerate(JOINTS)], dtype=np.float32)


def model_to_arm(values):
    values = vector(values)
    return np.array([(values[i] - FRAME[key][1]) / FRAME[key][0]
                     for i, key in enumerate(JOINTS)], dtype=np.float32)


def make_observation(observation, instruction, *, seed=None):
    if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 1024:
        raise ValueError("Nonempty instruction of at most1024 characters required")
    state = arm_to_model([observation[key] for key in JOINTS])
    images = {}
    for source, destination in CAMERAS.items():
        frame = np.asarray(observation[source])
        if frame.dtype != np.uint8 or frame.shape != (480, 640, 3):
            raise ValueError(f"{source} must be RGB uint8 480x640; missing cameras are not padded")
        images[destination] = np.ascontiguousarray(frame.transpose(2, 0, 1))
    images["wrist_left"] = np.zeros_like(images["wrist_right"])
    result = {"images": images, "state": {"right_arm": state}, "task": instruction,
              "embodiment_type": "so100", "frequency": 20.}
    if seed is not None:
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Expected uint32 seed")
        result["_dume_seed"] = seed
    return result


def decode_action(action):
    if not isinstance(action, dict) or set(action) != EXPECTED_ACTION_GROUPS:
        raise ValueError("Expected exactly the right_arm action group; refusing absent/extra groups")
    return dict(zip(JOINTS, map(float, model_to_arm(action["right_arm"]))))
