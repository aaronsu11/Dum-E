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


def test_stock_observer_is_inert_and_retains_full_noise_and_raw(tmp_path):
    import torch
    import torch.nn.functional as functional
    from types import SimpleNamespace
    from policy_guard.replay_contract import read_tensors

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(config=SimpleNamespace(
                _attn_implementation="sdpa",
                text_config=SimpleNamespace(_attn_implementation="sdpa"),
                vision_config=SimpleNamespace(_attn_implementation="sdpa")))

        def forward(self, inputs):
            q = inputs["state"].reshape(2, 1, 1, 6)
            return functional.scaled_dot_product_attention(q, q, q)

    class Gr00tN1d7(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.backbone = Backbone()
            self.action_head = torch.nn.Module()
            self.action_head.action_encoder = torch.nn.Identity()

        def get_action(self, inputs, options=None):
            hidden = self.backbone(inputs)
            result = torch.randn(2, 40, 132) + hidden.sum() * self.weight
            for _ in range(4):
                result = self.action_head.action_encoder(result)
            return {"action_pred": result}

    model = Gr00tN1d7().eval()
    inputs = {"state": torch.zeros(2, 6)}
    torch.manual_seed(42)
    expected = model.get_action(inputs)["action_pred"]
    expected_next = torch.randn(3)
    torch.manual_seed(42)
    with api().observe_stock(tmp_path, "native", {"evidence_kind": "test_only"}) as observations:
        actual = model.get_action(inputs)["action_pred"]
    assert torch.equal(actual, expected)
    assert torch.equal(torch.randn(3), expected_next)
    assert len(observations) == 1
    value = observations[0]
    assert value["observed"]["observer_inert"]
    assert value["observed"]["flow_steps"] == 4
    assert value["observed"]["noise_shape"] == [2, 40, 132]
    assert value["observed"]["compute_dtypes"] == ["torch.float32"]
    assert value["observed"]["sdpa_calls"] == 1
    np.testing.assert_array_equal(read_tensors(tmp_path, value["raw"])["raw"], actual.detach().numpy())
    assert sys.getprofile() is None


def test_stock_subprocess_fixtures_complete_and_recheck_strict_archive(tmp_path):
    from types import SimpleNamespace
    from policy_guard.parity_gate import Evidence
    from policy_guard.replay_contract import write_evidence
    from tests.test_checkpoint_parity import fixture

    ev, _, _ = fixture(tmp_path)
    proposal = ev.json("tolerance-proposal.json")
    write_evidence(tmp_path, "harness-source.json", {"source": str(tmp_path / "external"),
                                                  "harness": proposal["harness"]})
    calls = []
    raw = write_tensors(tmp_path, {"raw": np.zeros((2, 40, 132), np.float32)})
    noise = write_tensors(tmp_path, {"noise": np.ones((2, 40, 132), np.float32)})
    inputs = write_tensors(tmp_path, {"state": np.ones((2, 6), np.float32)})

    def execute(argv, stdout, **kwargs):
        stage = argv[argv.index("--stage") + 1]
        calls.append((stage, argv))
        backend = "native" if stage == "producer" else "lerobot"
        profile = next(p["observed"] for p in ev.json("profiles.json")["profiles"]
                       if p["backend"] == backend and p["purpose"] == "diagnostic")
        observed = {**profile, "raw_shape": [2, 40, 132], "noise_shape": [2, 40, 132]}
        write_evidence(tmp_path, f"stock/{stage}-observation.json", {
            **profile, "status": "complete", "evidence_kind": "test_only", "observed": observed,
            "inputs": inputs, "raw": raw, "noise": noise,
        })
        if stage == "producer":
            stdout.write("Dumped 1 tags: ['new_embodiment']\n")
            (tmp_path / "stock/producer" / api().ARTIFACT).write_bytes(b"fresh trusted producer fixture")
        else:
            nodeid = api().CONSUMER + "::" + api().CASE
            write_evidence(tmp_path, "stock/coverage.json", {
                "collected": [nodeid],
                "reports": [{"nodeid": nodeid, "when": phase, "outcome": "passed", "wasxfail": None}
                            for phase in ("setup", "call", "teardown")],
            })
            (tmp_path / "stock/junit.xml").write_text(
                f'<testsuites><testsuite tests="1"><testcase name="{api().CASE}"/></testsuite></testsuites>')
        return SimpleNamespace(returncode=0)

    args = SimpleNamespace(workspace=tmp_path, corpus=tmp_path / "corpus", checkpoint=tmp_path / "checkpoint",
                           native_cache=tmp_path / "cache", worker_timeout=10)
    result = api().run(ev, executor=execute, args=args, source_validator=lambda source, _: Path(source),
                       input_validator=lambda *_: ev.json("input-lock.json"),
                       resources=lambda: {"compute_processes": ""})
    assert result["status"] == "complete", result["prerequisite_errors"]
    assert [stage for stage, _ in calls] == ["producer", "consumer"]
    assert api().check(Evidence(tmp_path, test_only=True))["status"] == "complete"
    for _, argv in calls:
        assert "--network" in argv and "none" in argv
        assert "--device" in argv and "GROOT_PARITY_ATOL=0.001" in argv
        assert not any("--privileged" in value or "/dev/tty" in value for value in argv)
    # Keep the same apparently passing report, but contradict actual coverage.
    from tests.test_parity_gate import rewrite
    rewrite(tmp_path, "stock/coverage.json", lambda c: c["reports"][1].update(wasxfail="unexpected pass"))
    with pytest.raises(ValueError):
        api().check(Evidence(tmp_path, test_only=True))
