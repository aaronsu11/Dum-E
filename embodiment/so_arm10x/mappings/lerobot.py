"""SO101 binding for the LeRobot HTTP model-coordinate protocol."""
import hashlib
from pathlib import Path
from shared import IPolicyMapping
from .galaxea import JOINTS, vector
from .frames import to_model_frame, to_arm_frame, calibration_scale


class SO101Mapping(IPolicyMapping):
    def __init__(self, profile, *, calibration_path=None):
        if profile not in {"groot-so101", "pi05-so101", "molmoact2-so101"}:
            raise ValueError(f"No qualified SO101 mapping for {profile}")
        self.profile = profile
        self.calibration_path = None
        if profile == "groot-so101":
            if calibration_path is None:
                raise ValueError("GR00T requires explicit calibration")
            self.calibration_path = Path(calibration_path).resolve(strict=True)
            self.scales, self.calibration_sha256 = calibration_scale(self.calibration_path)

    @property
    def joint_names(self):
        return JOINTS

    @property
    def camera_names(self):
        return ("front", "wrist")

    def validate(self):
        if self.calibration_path is not None:
            if hashlib.sha256(self.calibration_path.read_bytes()).hexdigest() != self.calibration_sha256:
                raise ValueError("GR00T calibration changed")

    def to_model(self, values):
        if self.profile == "groot-so101":
            return vector(values) / self.scales
        return to_model_frame(values, self.profile)

    def to_arm(self, values):
        if self.profile == "groot-so101":
            return vector(values) * self.scales
        return to_arm_frame(values, self.profile)

    @property
    def metadata(self):
        if self.calibration_path is None:
            return {}
        return {"calibration_sha256": self.calibration_sha256,
                "units": "controller degrees <-> checkpoint RANGE_M100_100; gripper 0-100"}
