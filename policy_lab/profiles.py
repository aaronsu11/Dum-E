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
    "pi05-base": Profile(
        "pi05-base", "lerobot/pi05_base",
        "b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba",
        "pi05", 50, 32,
        "Base-model smoke only: padded raw SO101 state, two mapped images and "
        "one missing-camera placeholder; no SO101 normalization or physical action mapping.",
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
