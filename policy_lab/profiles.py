"""Pinned candidates and explicit limits of their action-space compatibility."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Profile:
    name: str
    repo: str
    revision: str
    policy_type: str
    horizon: int
    action_dim: int
    scope: str
    physical_ready: bool = False

    def to_dict(self):
        return asdict(self)


PROFILES = {
    "groot-so101": Profile(
        "groot-so101", "Dum-E/GR00T-N1.7-3B-SO101",
        "9bd09a2a40c04637b5d6e010790ac2e3b6cbd75f81c1d59443856e4e37f27c09",
        "groot", 16, 6,
        "Validated Dum-E checkpoint and BF16 LeRobot loader with serving guard; "
        "evaluation HTTP transport, observer off, no physical execution.",
    ),
    "pi05-base": Profile(
        "pi05-base", "lerobot/pi05_base",
        "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba",
        "pi05", 50, 32,
        "Base-model smoke only: padded raw SO101 state, two mapped images and "
        "one missing-camera placeholder; no SO101 normalization or physical action mapping.",
    ),
    "pi05-so101": Profile(
        "pi05-so101", "Project-IRA/TPSoSe2026_Pi05_LeRobot_SO101_Finetuning_V7_Full_V2",
        "4b48932cc74a61f685841a4fff467ef31caa9ce1", "pi05", 50, 6,
        "Author-recommended checkpoint 008000; six named absolute SO101 joints in "
        "degrees with 0-100 gripper; saved MEAN_STD processors; wrist_left/desk_view "
        "live cameras. Dum-E calibration and task behavior require bounded physical validation.",
    ),
    "molmoact2-so101": Profile(
        "molmoact2-so101", "allenai/MolmoAct2-SO100_101",
        "152569fe57914d97be91055800035f54e250d009",
        "molmoact2", 30, 6,
        "Official SO100/101 mixture, absolute joint-pose output. Checkpoint normalization "
        "is used, but compatibility with Dum-E calibration and camera placement is unverified.",
    ),
}


def get_profile(name):
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown model profile: {name}") from None
