"""Pinned G05 codec layout; no robot IO or coordinate conversion."""
CODEC_PARTS = {"left_control": 9, "left_gripper": 1,
               "right_control": 9, "right_gripper": 1}
UNUSED_CODEC_GROUPS = frozenset({"left_control", "left_gripper", "right_gripper"})


def validate_codec_presence(absent):
    """SO101 packs all six values in right_control; the other slots are virtual."""
    if set(absent) != UNUSED_CODEC_GROUPS:
        raise ValueError(f"Unexpected SO101 codec presence: absent={absent}")



CHECKPOINT_REVISION = "e312be81e90c56a55bcb26b57429bd39a335b449"
