"""Stock subprocess fixtures are fabricated and never run the actual consumer."""

import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from policy_guard.replay_contract import write_tensors

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def api():
    assert importlib.util.find_spec("replay_upstream_parity"), (
        "Stock wrapper must reject skipped coverage and cropped-only parity"
    )
    return importlib.import_module("replay_upstream_parity")


def test_stock_junit_requires_exact_positive_case_identity():
    xml = b'<testsuites><testsuite tests="1"><testcase name="test_groot_get_action_parity[new_embodiment]" classname="test_groot_vs_original"/></testsuite></testsuites>'
    assert api().parse_junit(xml) == [{"name": "new_embodiment", "outcome": "passed"}]


@pytest.mark.parametrize("fault", ["empty", "skip", "xfail", "xpass", "failure", "error", "wrong", "duplicate"])
def test_stock_false_positive_coverage_refused(fault):
    node = '<testcase name="test_groot_get_action_parity[new_embodiment]">{}</testcase>'
    child = {
        "skip": '<skipped/>', "xfail": '<skipped type="pytest.xfail"/>',
        "xpass": '<properties><property name="outcome" value="xpassed"/></properties>',
        "failure": '<failure/>', "error": '<error/>',
    }.get(fault, "")
    body = node.format(child)
    if fault == "empty":
        body = ""
    elif fault == "wrong":
        body = body.replace("new_embodiment", "libero_sim")
    elif fault == "duplicate":
        body *= 2
    with pytest.raises(ValueError):
        api().parse_junit(f"<testsuites><testsuite>{body}</testsuite></testsuites>".encode())


@pytest.mark.parametrize("output", [
    "", "[fail] new_embodiment: MemoryError\nDumped 0 tags: []",
    "[skip] new_embodiment\nDumped 0 tags: []",
    "Dumped 1 tags: ['libero_sim']", "Dumped 1 tags: ['new_embodiment']\nSkipped/failed 1 tags: ['other']",
])
def test_producer_exit_zero_is_not_positive_coverage(output):
    with pytest.raises(ValueError):
        api().validate_producer_output(output, 0)


def test_producer_summary_requires_expected_tag():
    api().validate_producer_output("Dumped 1 tags: ['new_embodiment']", 0)
    with pytest.raises(ValueError):
        api().validate_producer_output("Dumped 1 tags: ['new_embodiment']", 1)


def test_stock_pre_crop_shape_mismatch_cannot_pass(tmp_path):
    left = write_tensors(tmp_path, {"raw": np.zeros((2, 40, 132), np.float32)})
    right = write_tensors(tmp_path, {"raw": np.zeros((2, 16, 132), np.float32)})
    with pytest.raises(ValueError, match="shape"):
        api().compare_raw(tmp_path, left, right, {"atol": 1e-3, "rtol": 1e-3})


def test_source_pin_checked_before_execution(tmp_path):
    with pytest.raises((ValueError, FileNotFoundError)):
        api().validate_harness(tmp_path, {"commit": "main"})


def test_stock_run_requires_agreement_before_subprocess(tmp_path):
    calls = []
    with pytest.raises(FileNotFoundError):
        api().run(tmp_path, executor=lambda *a, **kw: calls.append(a))
    assert calls == []


def test_pickle_artifacts_cannot_be_supplied_from_an_existing_directory(tmp_path):
    directory = tmp_path / "trusted"
    directory.mkdir()
    (directory / "original_n1_7_new_embodiment.npz").write_bytes(b"untrusted")
    with pytest.raises(FileExistsError):
        api().create_producer_directory(directory)
