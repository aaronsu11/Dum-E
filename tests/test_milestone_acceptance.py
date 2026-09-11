"""Physical-unit milestone limits are acceptance criteria, not motion limits."""
import numpy as np
import pytest
from policy_guard.milestone_acceptance import physical_metrics


def test_degree_conversion_accepts_observed_scale_but_rejects_larger_error():
    delta = np.zeros((12,16,6))
    delta[0,0,1] = .9857101440429688
    scales = [1.139340659,1.032087912,.9657142857,1.00395604,1.8,1]
    stats, failures = physical_metrics(delta, scales)
    assert stats['max_abs'][1] == pytest.approx(1.0173395244)
    assert failures == []
    delta[0,0,1] = 2
    assert 'max_abs' in physical_metrics(delta, scales)[1]


def test_bias_and_mean_remain_gated_even_below_peak_limit():
    delta = np.zeros((12,16,6)); delta[:,:,0] = .6
    _, failures = physical_metrics(delta, [1]*6)
    assert set(failures) == {'mean_abs','bias'}


def test_gripper_uses_normalized_points_not_degrees():
    delta = np.zeros((12,16,6)); delta[0,0,5] = 1.01
    assert 'max_abs' in physical_metrics(delta, [1]*6)[1]


def test_missing_cases_nan_and_invalid_scale_fail_closed():
    with pytest.raises(ValueError): physical_metrics(np.zeros((11,16,6)), [1]*6)
    bad = np.zeros((12,16,6)); bad[0,0,0] = np.nan
    with pytest.raises(ValueError): physical_metrics(bad, [1]*6)
    with pytest.raises(ValueError): physical_metrics(np.zeros((12,16,6)), [0]*6)
