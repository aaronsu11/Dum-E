"""Small reducer tests; fabricated arrays, no inference."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("milestone", Path(__file__).resolve().parents[1] / "scripts/check_milestone_parity.py")
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def captures():
    return [{"raw": np.zeros((1, 40, 132), np.float32),
             "noise": np.zeros((1, 40, 132), np.float32),
             "decoded": np.zeros((16, 6), np.float32),
             "preprocessing.tokens": np.ones((1, 4), np.int64),
             "preprocessing.state": np.zeros((1, 6), np.float32),
             "collated.state": np.zeros((1, 6), np.float32)} for _ in range(12)]


def rule():
    return {"noise_policy": "independent", "thresholds": {
        "preprocessing": {"atol": 1e-6, "rtol": 1e-6},
        "raw": {"atol": 1e-3, "rtol": 1e-3},
        "decoded": {"max_abs": [.1]*6, "mean_abs": [.05]*6, "bias": [.02]*6, "slope": [.002]*6}}}


def test_selection_is_midpoint_first_seed_for_every_episode():
    lock = {"records": [{"episode_index": e, "frame_index": f, "file": f"r{e}-{f}", "seeds": [e*100+f, -1]}
                        for e in range(12) for f in range(10)]}
    assert api.selected_cases(lock) == [{"record": f"r{e}-5", "seed": e*100+5} for e in range(12)]
    lock["records"] = lock["records"][:-10]
    with pytest.raises(ValueError):
        api.selected_cases(lock)


def test_identical_captures_pass_and_partial_coverage_refused():
    a = captures()
    result, matrix = api.compare_arrays(a, copy.deepcopy(a), rule())
    assert result["passed"] and matrix["delta"].shape == (12, 16, 6)
    with pytest.raises(ValueError):
        api.compare_arrays(a[:-1], a[:-1], rule())


def test_local_bias_cannot_cancel_across_cases():
    a, b = captures(), captures()
    b[0]["decoded"][:, 0] = .03
    b[1]["decoded"][:, 0] = -.03
    result, _ = api.compare_arrays(a, b, rule())
    assert result["metrics"]["bias"][0] == 0
    assert "decoded.trace_bias_max_abs" in result["failures"]


def test_slope_tokens_and_raw_are_independently_checked():
    a, b = captures(), captures()
    b[0]["decoded"][:, 1] = (np.arange(16)-7.5)*.003
    b[0]["preprocessing.tokens"][0, 0] += 1
    b[0]["raw"][0, 39, 131] = .1
    result, _ = api.compare_arrays(a, b, rule())
    assert {"decoded.trace_slope_max_abs", "inputs[0]", "raw"} <= set(result["failures"])


def test_noise_difference_reported_without_false_matched_seed_claim():
    a, b = captures(), captures()
    b[0]["noise"][0, 0, 0] = 1
    result, _ = api.compare_arrays(a, b, rule())
    assert result["passed"] and result["noise"] == {"policy": "independent", "equal": False, "matched_seed_proves_equal_noise": False}
    exact = rule(); exact["noise_policy"] = "exact"
    assert "noise" in api.compare_arrays(a, b, exact)[0]["failures"]


def test_nonfinite_and_cropped_raw_are_rejected():
    a, b = captures(), captures()
    b[0]["raw"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        api.compare_arrays(a, b, rule())
    a, b = captures(), captures()
    for row in a+b:
        row["raw"] = row["raw"][:, :16]
    with pytest.raises(ValueError):
        api.compare_arrays(a, b, rule())


def test_input_schema_mismatch_reports_metrics_instead_of_aborting():
    a, b = captures(), captures()
    for row in b:
        row["collated.extra_mask"] = np.ones(1, np.int64)
        row["collated.state"] = row["collated.state"].astype(np.float64)
        row["dtype.extra_mask"] = np.array(1, np.int64)
    result, _ = api.compare_arrays(a, b, rule())
    assert not result["passed"]
    assert result["metrics"]["max_abs"] == [0]*6
    assert len(result["inputs"][0]["schema_differences"]) == 2
    assert result["inputs"][0]["captured_dtype_codes"]["lerobot"] == {"dtype.extra_mask": 1}
