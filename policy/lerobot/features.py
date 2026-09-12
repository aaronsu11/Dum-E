"""LeRobot handshake features — and the BACK-05 joint-ordering authority.

The ``lerobot_features`` dict this module builds is one half of the
``RemotePolicyConfig`` handshake payload (see ``policy/lerobot/session.py``). It
is deliberately NOT hand-written: it is produced by upstream's own
``lerobot.utils.feature_utils.hw_to_dataset_features``, because the
``["observation.state"]["names"]`` list it returns IS the joint-ordering
authority for the whole flat<->named mapping. ``build_dataset_frame`` builds the
state vector by iterating exactly that list::

    elif ft["dtype"] == "float32" and len(ft["shape"]) == 1:
        frame[key] = np.array([values[name] for name in ft["names"]], dtype=np.float32)
    # lerobot/utils/feature_utils.py:131-132

So a hand-written dict that happened to disagree with ``ROBOT_STATE_KEYS`` would
silently permute the joints on the wire — the arm would move, plausibly, to the
wrong pose. Call upstream's builder and ASSERT its output instead
(``assert_state_ordering``).

==================== HANDSHAKE PAYLOAD TRAP ====================
``RemotePolicyConfig.lerobot_features`` is ANNOTATED ``dict[str, PolicyFeature]``
(``lerobot/async_inference/helpers.py:266-273``) but every consumer treats it as
the dataset-features dict of PLAIN DICTS —
``raw_observation_to_observation(..., lerobot_features: dict[str, dict])``
(``helpers.py:88-92``) and ``build_dataset_frame(ds_features: dict[str, dict], ...)``
(``feature_utils.py:110-112``). Pass the ``hw_to_dataset_features`` output, NOT
``PolicyFeature`` objects. The annotation is wrong; the usage is authoritative.

==================== WHY rename_map IS DELIBERATELY EMPTY ====================
``RemotePolicyConfig.rename_map`` is ``{}`` on purpose, and it must not be
pressed into service as the flat->named mapper. It feeds
``RenameObservationsProcessorStep`` *inside* the preprocessor
(``policy_server.py:160-163``, ``processor_groot.py:1255``), which runs strictly
AFTER ``raw_observation_to_observation`` has already indexed the observation by
key. A ``KeyError`` raised in that earlier step can therefore never be repaired
by a rename that runs later. ``rename_map`` is the wrong tool for this job; the
ordering authority above is the right one.
"""

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
#:
#: NOTE this tuple's ORDER is not the order the policy consumes the cameras in.
#: The GR00T pack step orders images by the CHECKPOINT's own
#: ``modality_configs["new_embodiment"]["video"]["modality_keys"]``, which is
#: ``["front", "wrist"]`` (``processor_groot.py:_ordered_image_keys``). This
#: tuple only decides which feature keys exist in the handshake, so it matches
#: the incumbent controller default rather than inventing a third ordering.
CAMERA_KEYS: tuple[str, ...] = ("wrist", "front")

#: Camera frame geometry. 480x640 is the size ``scripts/test_live_policy_server.py``
#: already uses and the size PAR-05's verdicts are measured at — do not
#: "simplify" it to a square. The served pipeline now forces the letterbox pad
#: so this frame preprocesses to ``(256, 256, 3)``, matching the checkpoint's
#: training-time geometry; unpatched upstream would yield ``(256, 340, 3)``.
#: See ``docs/LEROBOT-SERVING-VERDICTS.md``.
FRAME_HEIGHT: int = 480
FRAME_WIDTH: int = 640


def build_lerobot_features(
    robot_state_keys: tuple[str, ...] | list[str] = ROBOT_STATE_KEYS,
    camera_keys: tuple[str, ...] | list[str] = CAMERA_KEYS,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
) -> dict[str, dict]:
    """Build the ``lerobot_features`` handshake dict via upstream's own builder.

    Args:
        robot_state_keys: The joint keys, in the order they must appear in
            ``observation.state``. Insertion order is preserved by
            ``hw_to_dataset_features`` (``names = list(joint_fts)``), so this
            argument's order IS the resulting authority.
        camera_keys: Camera names. Each becomes ``observation.images.<name>``.
        height: Camera frame height in pixels.
        width: Camera frame width in pixels.

    Returns:
        The LeRobot dataset-features dict of plain dicts, ready to be placed on
        ``RemotePolicyConfig.lerobot_features``.
    """
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
    """Raise if the built features disagree with the expected joint ordering.

    Raises:
        ValueError: naming BOTH lists when they differ, so the operator can see
            the permutation rather than infer it.
    """
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
