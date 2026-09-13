import numpy as np
import pytest
from policy_guard.integration_trial import bounded_command, check_camera, joint_limits
from policy.galaxea.modalities import JOINTS


def test_projection_limits_speed_total_excursion_and_gripper():
    origin = np.array([0.] * 5 + [50.])
    previous = origin.copy()
    limits = (np.array([-180.] * 5 + [0.]), np.array([180.] * 5 + [100.]))
    for _ in range(32):
        target = bounded_command([1000.] * 6, previous, previous, origin, limits)
        assert np.max(abs(target - previous)) <= 0.25
        assert np.max(abs(target - origin)) <= 5
        previous = target
    np.testing.assert_array_equal(previous, origin + 5)
    with pytest.raises(ValueError, match="diverged"):
        bounded_command(origin, origin + 2, origin, origin, limits)


def test_missing_camera_content_and_invalid_actions_are_refused():
    with pytest.raises(ValueError, match="uniform"):
        check_camera(np.zeros((480, 640, 3), np.uint8), "wrist")
    with pytest.raises(ValueError, match="RGB"):
        check_camera(np.zeros((224, 224, 3), np.uint8), "front")
    limits = (np.full(6, -100.), np.full(6, 100.))
    with pytest.raises(ValueError):
        bounded_command([float("nan")] * 6, np.zeros(6), np.zeros(6), np.zeros(6), limits)


def test_gripper_limits_are_not_degrees():
    calibration = {k.removesuffix(".pos"): {"range_min": 1000, "range_max": 3000}
                   for k in JOINTS}
    lower, upper = joint_limits(calibration)
    assert lower[-1] == 0 and upper[-1] == 100
    assert upper[0] == pytest.approx(1000 * 360 / 4095)
