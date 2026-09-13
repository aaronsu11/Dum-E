import json
import numpy as np
import pytest
from policy.so101_contract import calibration_scale, to_model_frame, to_arm_frame
from policy.galaxea.modalities import JOINTS


def test_calibration_endpoints_and_gripper_are_distinct(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({key.removesuffix(".pos"): {"range_min": 1000, "range_max": 3000}
                                for key in JOINTS}))
    degrees_at_upper_limit = 1000 * 360 / 4095
    values = [degrees_at_upper_limit] * 5 + [37.]
    model = to_model_frame(values, "groot-so101", calibration_path=path)
    np.testing.assert_allclose(model, [100] * 5 + [37], rtol=1e-6)
    np.testing.assert_allclose(to_arm_frame(model, "groot-so101", calibration_path=path),
                               values, rtol=1e-6)
    assert len(calibration_scale(path)[1]) == 64


def test_reference_frames_and_unmapped_base():
    for profile in ("g05-so101", "molmoact2-so101"):
        np.testing.assert_array_equal(to_model_frame([0, -30, 25, 40, 0, 20], profile),
                                      [0, 120, 115, 40, 0, 20])
        np.testing.assert_array_equal(to_arm_frame([0, 120, 115, 40, 0, 20], profile),
                                      [0, -30, 25, 40, 0, 20])
    for transform in (to_model_frame, to_arm_frame):
        with pytest.raises(ValueError, match="no verified"):
            transform(np.zeros(6), "pi05-base")
        with pytest.raises(ValueError, match="Explicit calibration"):
            transform(np.zeros(6), "groot-so101")
