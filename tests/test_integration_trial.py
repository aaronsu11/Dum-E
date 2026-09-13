import numpy as np
import pytest
from policy_guard.integration_trial import bounded_command, check_camera, joint_limits, probe_target
from policy.galaxea.modalities import JOINTS


def test_projection_limits_speed_total_excursion_and_gripper():
    origin = np.array([0.] * 5 + [50.])
    previous = origin.copy()
    limits = (np.array([-180.] * 5 + [0.]), np.array([180.] * 5 + [100.]))
    for _ in range(96):
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


def test_stationary_feedback_does_not_accumulate_targets_across_chunks():
    origin = np.array([0.] * 5 + [50.])
    previous = origin.copy()
    limits = (np.array([-180.] * 5 + [0.]), np.array([180.] * 5 + [100.]))
    for _ in range(96):
        command = bounded_command([1000.] * 6, origin, previous, origin, limits)
        assert np.max(abs(command - origin)) <= 0.25
        previous = command


def test_probe_moves_only_pan_with_bounded_slew_and_tracking():
    origin = np.array([0.] * 5 + [50.])
    limits = (np.array([-180.] * 5 + [0.]), np.array([180.] * 5 + [100.]))
    target = probe_target(origin)
    np.testing.assert_array_equal(target - origin, [3, 0, 0, 0, 0, 0])
    for follows in (False, True):
        previous = origin.copy()
        for _ in range(40):
            observed = previous if follows else origin
            command = bounded_command(target, observed, previous, origin, limits,
                                      max_tracking_error=0.75)
            assert np.max(abs(command - previous)) <= 0.25
            assert np.max(abs(command - observed)) <= 0.75
            assert np.max(abs(command - origin)) <= 3
            np.testing.assert_array_equal(command[1:], origin[1:])
            previous = command
        assert previous[0] == (3 if follows else 0.75)
    with pytest.raises(ValueError, match="diverged"):
        bounded_command(target, origin + 2, origin, origin, limits,
                        max_tracking_error=0.75)


def test_relaxed_probe_reaches_target_without_feedback_but_keeps_slew_and_total_cap():
    origin = np.array([0.] * 5 + [50.])
    limits = (np.array([-180.] * 5 + [0.]), np.array([180.] * 5 + [100.]))
    for target, expected in ((probe_target(origin), 3.), (np.full(6, 1000.), 3.75)):
        previous = origin.copy()
        for _ in range(40):
            command = bounded_command(target, origin, previous, origin, limits,
                                      max_tracking_error=3.75)
            assert np.max(abs(command - previous)) <= 0.25
            assert np.max(abs(command - origin)) <= 3.75
            previous = command
        assert previous[0] == expected
    previous = origin.copy()
    for _ in range(96):
        command = bounded_command(np.full(6, 1000.), previous, previous, origin, limits,
                                  max_tracking_error=3.75)
        assert np.max(abs(command - previous)) <= 0.25
        assert np.max(abs(command - origin)) <= 5.
        previous = command
    with pytest.raises(ValueError, match="Invalid tracking"):
        bounded_command(origin, origin, origin, origin, limits, max_tracking_error=4.)


def test_six_degree_probe_keeps_other_joints_and_tracking_limit(monkeypatch):
    from policy_guard.integration_trial import PROTOCOL
    monkeypatch.setitem(PROTOCOL, "max_excursion", 6.)
    origin = np.array([0.] * 5 + [50.])
    target = probe_target(origin, 6.)
    limits = (np.array([-180.] * 5 + [0.]), np.array([180.] * 5 + [100.]))
    for follows in (False, True):
        previous = origin.copy()
        for _ in range(40):
            observed = previous if follows else origin
            command = bounded_command(target, observed, previous, origin, limits,
                                      max_tracking_error=3.75)
            assert np.max(abs(command - previous)) <= 0.25
            assert np.max(abs(command - observed)) <= 3.75
            assert np.max(abs(command - origin)) <= 6.
            np.testing.assert_array_equal(command[1:], origin[1:])
            previous = command
        assert previous[0] == (6. if follows else 3.75)
    with pytest.raises(ValueError, match="offset"):
        probe_target(origin, 6.1)
