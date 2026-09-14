'LeRobot handshake features — and the BACK-05 joint-ordering authority.'

from lerobot.utils.constants import OBS_STATE
from lerobot.utils.feature_utils import hw_to_dataset_features

#: The six SO-ARM10x follower joints, in the ONLY order that is authoritative:
#: ``embodiment/so_arm10x/controller.py``'s ``robot_state_keys``. Dims 0:5 are
#: the ``single_arm`` group and dim 5 is ``gripper`` — the split the checkpoint's
#: ``action_configs`` encodes as RELATIVE (arm) + ABSOLUTE (gripper).
ROBOT_STATE_KEYS: tuple[str, ...] = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)

#: The two cameras, matching ``controller.py``'s ``camera_keys`` default.
#: NOTE this tuple's ORDER is not the order the policy consumes the cameras in.
CAMERA_KEYS: tuple[str, ...] = ("wrist", "front")

#: Camera frame geometry. 480x640 is the size ``scripts/test_live_policy_server.py``
#: already uses and the size PAR-05's verdicts are measured at — do not
FRAME_HEIGHT: int = 480
FRAME_WIDTH: int = 640


def build_lerobot_features(
    robot_state_keys: tuple[str, ...] | list[str] = ROBOT_STATE_KEYS,
    camera_keys: tuple[str, ...] | list[str] = CAMERA_KEYS,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
) -> dict[str, dict]:
    "Build the ``lerobot_features`` handshake dict via upstream's own builder."
    hw: dict[str, type | tuple] = {joint: float for joint in robot_state_keys}
    for cam in camera_keys:
        hw[cam] = (height, width, 3)

    # use_video=False: these are live camera frames, not decoded video files, so
    # the features must be dtype "image". "video" would make build_dataset_frame
    # treat them as dataset video references.
    return hw_to_dataset_features(hw, "observation", use_video=False)


def state_names(features: dict[str, dict]) -> list[str]:
    """Return the authoritative joint-name ordering from a features dict."""
    return list(features[OBS_STATE]["names"])


def assert_state_ordering(
    features: dict[str, dict],
    robot_state_keys: tuple[str, ...] | list[str] = ROBOT_STATE_KEYS,
) -> None:
    'Raise if the built features disagree with the expected joint ordering.'
    observed = state_names(features)
    expected = list(robot_state_keys)
    if observed != expected:
        raise ValueError(
            f"lerobot_features['{OBS_STATE}']['names'] is {observed!r}, expected "
            f"{expected!r}. These MUST match: build_dataset_frame iterates the "
            "'names' list to build the state vector, so a disagreement silently "
            "permutes the joints on the wire. Refusing to hand a permuted state "
            "vector to the policy — a plausible-looking pose at the wrong joints "
            "is worse than a crash."
        )
