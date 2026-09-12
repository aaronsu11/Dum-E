"""Fabricated numerical experiments; no model, server, or hardware evidence."""

import copy
import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from policy_guard.parity_gate import Evidence
from policy_guard.replay_contract import CAMERA_ORDER, JOINT_ORDER, read_json, write_tensors
from tests.test_parity_gate import fabricated_repeats, fabricated_worker, fixture_workspace, review, rewrite, ts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def api():
    assert importlib.util.find_spec("policy_guard.parity_report"), (
        "The numerical instrument must produce trace-level bias and slope verdicts"
    )
    return importlib.import_module("policy_guard.parity_report")


def fixture(tmp_path, delta=None, approve=True):
    _, lock, pairs = fixture_workspace(tmp_path)
    if approve:
        review(tmp_path, "tolerances", 7)
    ev = Evidence(tmp_path, test_only=True)
    arrays = {
        "raw": np.zeros((600, 40, 132), np.float32),
        "noise": np.zeros((600, 40, 132), np.float32),
        "decoded": np.zeros((600, 16, 6), np.float32),
    }
    processed = write_tensors(tmp_path, {
        "image_front": np.zeros((1, 3, 2, 2), np.float32),
        "image_wrist": np.ones((1, 3, 2, 2), np.float32),
        "state": np.zeros((1, 6), np.float32),
        "tokens": np.ones((1, 4), np.int64), "mask": np.ones((1, 4), np.int64),
    })
    left = {
        "cases": lock["schedule"], "joint_order": list(JOINT_ORDER),
        "camera_order": list(CAMERA_ORDER), "input_fingerprint": lock["fingerprint"],
        "profile": pairs["diagnostic"][0], "tensors": write_tensors(tmp_path, arrays),
        "preprocessing": [processed] * 600, "common_inputs": [processed] * 600,
        "common_collated": [processed] * 600, "independent_collated": [processed] * 600,
    }
    right = copy.deepcopy(left)
    right["profile"] = pairs["diagnostic"][1]
    if delta is not None:
        arrays["decoded"] += delta
    right["tensors"] = write_tensors(tmp_path, arrays)
    return ev, left, right


def compare(ev, left, right, **kwargs):
    return api().compare_tiers(
        ev, "diagnostic", left, right, started_at=kwargs.pop("started_at", ts(8)),
        ended_at=ts(9), **kwargs,
    )


def test_equal_fabricated_arrays_produce_complete_matrix(tmp_path):
    ev, left, right = fixture(tmp_path)
    result = compare(ev, left, right)
    assert result["passed"] is True
    assert result["cases"] == left["cases"] and len(result["cases"]) == 600
    assert result["units"] == ["percent"] * 5 + ["gripper_percent"]
    matrix = ev.tensors(result["matrix"])
    assert matrix["delta"].shape == (600, 16, 6)
    assert matrix["delta"].dtype == np.float64
    assert matrix["trace_bias"].shape == (600, 6)
    assert result["delta_definition"] == "right-minus-left"
    assert result["metrics"]["bias"] == [0.0] * 6


def test_opposite_offsets_cannot_cancel_trace_bias(tmp_path):
    delta = np.zeros((600, 16, 6), np.float32)
    delta[0, :, 2], delta[1, :, 2] = 0.125, -0.125
    ev, left, right = fixture(tmp_path, delta)
    result = compare(ev, left, right)
    assert result["passed"] is False
    assert result["metrics"]["bias"][2] == 0
    assert result["metrics"]["trace_bias_max_abs"][2] == 0.125
    assert "decoded.trace_bias" in result["failures"]


def test_opposite_time_ramps_cannot_cancel_trace_slopes(tmp_path):
    delta = np.zeros((600, 16, 6), np.float32)
    ramp = (np.arange(16) - 7.5) * 0.125
    delta[0, :, 4], delta[1, :, 4] = ramp, -ramp
    ev, left, right = fixture(tmp_path, delta)
    result = compare(ev, left, right)
    assert result["passed"] is False
    assert result["metrics"]["slope"][4] == 0
    assert result["metrics"]["trace_slope_max_abs"][4] == 0.125
    assert "decoded.trace_slope" in result["failures"]
    matrix = ev.tensors(result["matrix"])
    np.testing.assert_array_equal(matrix["delta"][0, :, 4], ramp)


@pytest.mark.parametrize("fault", ["missing", "permuted", "joint", "camera", "identity", "raw", "decoded"])
def test_no_partial_or_reordered_or_cropped_comparison(tmp_path, fault):
    ev, left, right = fixture(tmp_path)
    if fault == "missing":
        right["cases"] = right["cases"][:-1]
    elif fault == "permuted":
        right["cases"][0], right["cases"][1] = right["cases"][1], right["cases"][0]
    elif fault in ("joint", "camera"):
        right[f"{fault}_order"].reverse()
    elif fault == "identity":
        right["input_fingerprint"] = "0" * 64
    else:
        values = ev.tensors(right["tensors"]).copy()
        values[fault] = values[fault][:, :-1]
        right["tensors"] = write_tensors(tmp_path, values)
    with pytest.raises(ValueError):
        compare(ev, left, right)


@pytest.mark.parametrize("fault", ["tokens", "mask", "common_inputs", "common_collated"])
def test_exact_metadata_and_common_model_inputs(tmp_path, fault):
    ev, left, right = fixture(tmp_path)
    field = fault if fault.startswith("common") else "preprocessing"
    values = {key: value.copy() for key, value in ev.tensors(right[field][0]).items()}
    values["state" if fault.startswith("common") else fault].flat[0] += 1
    right[field][0] = write_tensors(tmp_path, values)
    result = compare(ev, left, right)
    assert not result["passed"]
    assert any(field in failure for failure in result["failures"])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_data_refused_before_reduction(value):
    data = np.zeros((2, 16, 6), np.float32)
    data[0, 0, 0] = value
    with pytest.raises(ValueError, match="finite"):
        api().joint_deviations(np.zeros_like(data), data)


@pytest.mark.parametrize("fault", ["missing", "late", "stale"])
def test_approval_required_before_arithmetic(tmp_path, fault):
    ev, left, right = fixture(tmp_path, approve=fault != "missing")
    if fault == "stale":
        rewrite(tmp_path, "tolerance-proposal.json", lambda p: p["caveats"].append("changed"))
    with pytest.raises((ValueError, FileNotFoundError)):
        compare(ev, left, right, started_at=ts(7) if fault == "late" else ts(8))
    assert not list((tmp_path / "comparisons").glob("*"))


def test_three_operational_bridges_do_not_claim_seed_implies_noise_equality(tmp_path):
    ev, left, right = fixture(tmp_path)
    proposal = ev.json("tolerance-proposal.json")
    values = ev.tensors(right["tensors"]).copy()
    values["noise"] = np.ones((600, 40, 132), np.float32)
    right["tensors"] = write_tensors(tmp_path, values)
    for name in ("native_bridge", "lerobot_bridge", "operational"):
        left["profile"], right["profile"] = proposal["comparisons"][name]["profiles"]
        result = api().compare_tiers(ev, name, left, right, started_at=ts(8), ended_at=ts(9))
        assert result["passed"]
        assert result["noise"]["policy"] == "independent"
        assert result["noise"]["equal"] is False
        assert result["noise"]["matched_seed_proves_equal_noise"] is False


def test_full_cli_refuses_absent_agreement_before_worker_launch(tmp_path):
    fixture_workspace(tmp_path)
    before = {p: p.read_bytes() for p in (tmp_path / "workers").rglob("*") if p.is_file()}
    result = subprocess.run([
        sys.executable, str(ROOT / "scripts/replay_checkpoint_parity.py"),
        "full", "--workspace", str(tmp_path),
    ], capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert '"status": "not_run"' in result.stdout
    assert "Starting" not in result.stdout
    assert {p: p.read_bytes() for p in (tmp_path / "workers").rglob("*") if p.is_file()} == before


def test_full_fake_workers_require_every_profile_and_common_input_pass(tmp_path):
    ev, left, right = fixture(tmp_path)
    module = importlib.import_module("replay_checkpoint_parity")
    assert hasattr(module, "full"), "Full orchestration must exist before agreement"
    calls = []

    class Workers:
        def collect(self, name, schedule, *, common_from=None):
            calls.append((name, len(schedule), common_from))
            return fabricated_worker(ev, name, schedule, f"full-{len(calls)}",
                                     second=8.5 if common_from is None else 9,
                                     common_from=common_from)[0]

    # Stock validity is tested independently. Explicit fixture collaborator only.
    result = module.full(ev, workers=Workers(), stock_check=lambda _: {"status": "complete"},
                         started_at=ts(8), clock=lambda: ts(9))
    assert result["status"] == "complete"
    assert len(result["comparisons"]) == 4
    assert len(calls) == 12 and all(count == 600 for _, count, _ in calls)
    assert sum(common is not None for _, _, common in calls) == 8
    assert read_json(tmp_path / "offline-report.json")["evidence_kind"] == "test_only"
    assert module.check_offline(Evidence(tmp_path, test_only=True),
                                stock_check=lambda _: {"status": "complete"})["status"] == "complete"


def test_captured_common_input_is_real_forwarded_data_without_mutating_original():
    import torch
    module = importlib.import_module("replay_checkpoint_parity")

    class Raw(torch.nn.Module):
        config = {}

        def get_action(self, inputs, options=None):
            return {"action_pred": inputs["state"] + 2}

    capture = module.capture_model(Raw())
    original = {"state": torch.zeros(1, 3)}
    common = {"state": torch.ones(1, 3)}
    capture.common = common
    output = capture.get_action(original)
    assert torch.equal(output["action_pred"], torch.full((1, 3), 3.0))
    assert torch.equal(original["state"], torch.zeros(1, 3))
    assert np.array_equal(capture.original_inputs["state"], np.zeros((1, 3)))
    assert torch.equal(capture.last_inputs["state"], common["state"])


def test_projection_retains_two_actual_camera_patch_ranges():
    import torch
    module = importlib.import_module("replay_checkpoint_parity")
    inputs = {
        "pixel_values": torch.arange(36, dtype=torch.float32).reshape(6, 6),
        "image_grid_thw": torch.tensor([[1, 1, 2], [1, 2, 2]]),
        "state": torch.ones(1, 132), "input_ids": torch.ones(1, 4, dtype=torch.int64),
        "attention_mask": torch.ones(1, 4, dtype=torch.int64),
    }
    processed, full = module.project_preprocessing(inputs)
    np.testing.assert_array_equal(processed["image_front"], full["pixel_values"][:2])
    np.testing.assert_array_equal(processed["image_wrist"], full["pixel_values"][2:])
    assert full["image_grid_thw"].shape == (2, 3)
    inputs["image_grid_thw"][0, 2] = 3
    with pytest.raises(ValueError):
        module.project_preprocessing(inputs)


def test_partial_worker_failure_is_preserved_and_never_passes(tmp_path):
    ev, _, _ = fixture(tmp_path)
    module = importlib.import_module("replay_checkpoint_parity")

    class Unavailable:
        def collect(self, *args, **kwargs):
            raise FileNotFoundError("injected missing model environment")

    result = module.full(ev, workers=Unavailable(), stock_check=lambda _: {},
                         started_at=ts(8), clock=lambda: ts(9))
    assert result["status"] == "not_run" and result["parity_passed"] is False
    assert result["comparisons"] == []
    assert read_json(tmp_path / "offline-report.json")["prerequisite_errors"]
    with pytest.raises(Exception):
        module.check_offline(Evidence(tmp_path, test_only=True), stock_check=lambda _: {})


def test_reduce_repeatability_never_pairs_different_backends(tmp_path):
    from policy_guard.replay_contract import repeatability_schedule, write_evidence
    ev, _, _ = fixture(tmp_path)
    module = importlib.import_module("replay_checkpoint_parity")
    identity = {**ev.identity(), "schema_version": 1, "evidence_kind": "test_only",
                "status": "complete", "started_at": ts(3), "ended_at": ts(4)}
    _, collection = fabricated_repeats(ev, identity, prefix="offset", offsets=True)
    result = module.reduce_repeatability(ev, collection)
    assert len(result["measurements"]) == 4
    for measurement in result["measurements"]:
        assert measurement["statistics"]["max_abs"] == [0.0] * 6
        assert measurement["raw_max_abs"] == 0
        assert len(measurement["cases"]) == 48 and len(measurement["groups"]) == 13


def test_common_collaborator_keeps_model_in_eval_and_restores_bf16_exactly():
    import torch
    module = importlib.import_module("replay_checkpoint_parity")
    raw = torch.nn.Linear(3, 3).eval()
    capture = module.capture_model(raw)
    assert all(not m.training for m in capture.modules())
    tensors = {"state": np.array([[1.125, -2.5, 0.0]], np.float32)}
    result = module.restored_inputs(tensors, "cpu", {"state": "torch.bfloat16"})
    assert result["state"].dtype == torch.bfloat16
    np.testing.assert_array_equal(result["state"].float().numpy(), tensors["state"])


def test_archived_matrix_cannot_be_regenerated_by_check(tmp_path):
    ev, left, right = fixture(tmp_path)
    result = compare(ev, left, right)
    path = tmp_path / result["matrix"]["path"]
    path.unlink()
    fresh = Evidence(tmp_path, test_only=True)
    with pytest.raises(FileNotFoundError):
        api().compare_tiers(fresh, "diagnostic", left, right, started_at=ts(8), ended_at=ts(9),
                           matrix_reference=result["matrix"])
    assert not path.exists()


def test_diagnostic_requires_actual_noise_equality(tmp_path):
    ev, left, right = fixture(tmp_path)
    values = ev.tensors(right["tensors"]).copy()
    values["noise"] = values["noise"] + np.float32(1)
    right["tensors"] = write_tensors(tmp_path, values)
    result = compare(ev, left, right)
    assert result["passed"] is False and "noise" in result["failures"]


def test_repeatability_diagnostic_and_operational_attention_are_independent(tmp_path):
    from policy_guard.replay_contract import validate_profile
    ev, _, _ = fixture(tmp_path)
    native = next(p["observed"] for p in ev.json("profiles.json")["profiles"]
                  if p["backend"] == "native" and p["purpose"] == "operational")
    native["attention"] = ["flash_attention_2"]
    native["parameter_dtypes"] = ["torch.bfloat16"]
    native["compute_dtypes"] = ["torch.float32", "torch.bfloat16"]
    native["sdpa_calls"] = 0
    validate_profile(native)
    native["purpose"] = "diagnostic"
    with pytest.raises(ValueError):
        validate_profile(native)


def test_full_archive_agreement_reference_and_parity_flag_are_required(tmp_path):
    ev, left, right = fixture(tmp_path)
    module = importlib.import_module("replay_checkpoint_parity")

    class Workers:
        counter = 0

        def collect(self, name, schedule, *, common_from=None):
            self.counter += 1
            return fabricated_worker(ev, name, schedule, f"archive-{self.counter}",
                                     second=8.5 if common_from is None else 9,
                                     common_from=common_from)[0]

    result = module.full(ev, workers=Workers(), stock_check=lambda _: {}, started_at=ts(8), clock=lambda: ts(9))
    assert result["status"] == "complete"
    rewrite(tmp_path, "offline-report.json", lambda r: r.update(parity_passed=False))
    with pytest.raises(ValueError, match="passing archive"):
        module.check_offline(Evidence(tmp_path, test_only=True), stock_check=lambda _: {})


def test_actual_empty_raw_or_nan_cannot_be_tolerated():
    with pytest.raises(ValueError):
        api().array_comparison(np.empty((0,)), np.empty((0,)), {"atol": 1, "rtol": 1})
    with pytest.raises(ValueError):
        api().array_comparison(np.ones(1), np.array([np.nan]), {"atol": 1, "rtol": 1})


def test_independent_collated_categorical_metadata_cannot_hide_behind_matching_actions(tmp_path):
    ev, left, right = fixture(tmp_path)
    base = ev.tensors(left["independent_collated"][0])
    left_ref = write_tensors(tmp_path, {**base, "embodiment_id": np.array([24], np.int32)})
    right_ref = write_tensors(tmp_path, {**base, "embodiment_id": np.array([25], np.int32)})
    left["independent_collated"] = [left_ref] * 600
    right["independent_collated"] = [right_ref] * 600
    result = compare(ev, left, right)
    assert result["passed"] is False
    assert result["metrics"]["max_abs"] == [0.0] * 6
    assert "independent_collated[0]" in result["failures"]


def test_review_offline_requires_archived_worker_witnesses(tmp_path):
    ev, left, right = fixture(tmp_path)
    module = importlib.import_module("replay_checkpoint_parity")
    class Workers:
        def collect(self, name, schedule, *, common_from=None):
            bundle = copy.deepcopy(left if name.startswith("native") else right)
            bundle["profile"] = name
            # Aggregate-only fixture intentionally has no captured worker evidence.
            return bundle
    result = module.full(ev, workers=Workers(), stock_check=lambda _: {"status": "complete"},
                         started_at=ts(8), clock=lambda: ts(9))
    assert result["status"] != "complete"


@pytest.mark.parametrize("witness", ["worker", "launch", "case"])
def test_review_offline_and_release_reject_deleted_process_witness(tmp_path, witness):
    from tests.test_parity_gate import api as gate, complete_offline
    identity, lock, pairs = fixture_workspace(tmp_path)
    complete_offline(tmp_path, identity, lock, pairs)
    ev = Evidence(tmp_path, test_only=True)
    module = importlib.import_module("replay_checkpoint_parity")
    assert module.check_offline(ev)["status"] == "complete"
    report = ev.json("offline-report.json")
    bundle = ev.json(report["comparisons"][0]["provenance"]["left"]["independent"])
    reference = (ev.json(bundle["worker"])["cases"][0]["evidence"]
                 if witness == "case" else bundle[witness])
    (tmp_path / reference["path"]).unlink()
    for validator in (module.check_offline, gate().validate_release_evidence):
        with pytest.raises(FileNotFoundError):
            validator(Evidence(tmp_path, test_only=True))


def test_review_valid_arithmetic_cannot_replace_actual_case_aggregate(tmp_path):
    from tests.test_parity_gate import complete_offline
    identity, lock, pairs = fixture_workspace(tmp_path)
    complete_offline(tmp_path, identity, lock, pairs)
    ev = Evidence(tmp_path, test_only=True)
    report = ev.json("offline-report.json")
    original = report["comparisons"][0]
    bundles = []
    for side, profile in zip(("left", "right"), original["profiles"]):
        bundle = {**ev.json(original["provenance"][side]["independent"]), "tensors": original[side]}
        if side == "right":
            arrays = {k: v.copy() for k, v in ev.tensors(bundle["tensors"]).items()}
            arrays["decoded"] += np.float32(.001)  # Still within the approved synthetic bound.
            bundle["tensors"] = write_tensors(tmp_path, arrays)
        bundles.append(bundle)
    forged = api().compare_tiers(ev, "diagnostic", *bundles,
                                started_at=original["started_at"], ended_at=original["ended_at"])
    assert forged["passed"]
    forged["provenance"] = original["provenance"]
    rewrite(tmp_path, "offline-report.json", lambda r: r["comparisons"].__setitem__(0, forged))
    module = importlib.import_module("replay_checkpoint_parity")
    with pytest.raises(ValueError, match="comparison aggregate"):
        module.check_offline(Evidence(tmp_path, test_only=True))


@pytest.mark.parametrize("purpose, expected", [("diagnostic", False), ("operational", True)])
def test_numerical_worker_preserves_operational_tf32_before_model_load(tmp_path, monkeypatch, purpose, expected):
    """Only diagnostics may override deployed controls; no model is constructed."""
    import torch
    from types import SimpleNamespace
    from policy_guard.replay_contract import PrerequisiteError, write_evidence
    module = importlib.import_module("replay_checkpoint_parity")
    native = importlib.import_module("replay_groot_native")
    records = [{"file": f"record_{i:04d}.npz", "seeds": list(range(5))} for i in range(120)]
    lock = {
        "fingerprint": "a" * 64, "joint_order": list(JOINT_ORDER),
        "camera_order": list(CAMERA_ORDER), "records": records,
        "schedule": [{"record": r["file"], "seed": s} for r in records for s in r["seeds"]],
        "checkpoint_fp32_tensor_bytes": 1,
    }
    output = "workers/test-control.json"
    write_evidence(tmp_path, "session.json", {"schema_version": 1, "session_id": "test-controls"})
    write_evidence(tmp_path, "input-lock.json", lock)
    write_evidence(tmp_path, "schedule.json", {
        "kind": "tracer", "session": "test-controls", "input_fingerprint": lock["fingerprint"],
        "cases": lock["schedule"][:1], "execution": {
            "id": "b" * 32, "collection": "numerical-tracer",
            "started_at": ts(1), "worker_manifest": output,
        },
    })
    seen = []

    def observe_before_load(*args):
        seen.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        raise PrerequisiteError("intentional test stop before any model load")

    monkeypatch.setattr(module, "load_input_lock", lambda *args: lock)
    monkeypatch.setattr(native, "NativeReplay", observe_before_load)
    prior = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.get_num_threads())
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        result = module.numerical_worker(SimpleNamespace(
            workspace=tmp_path, output_manifest=output, schedule=tmp_path / "schedule.json",
            corpus=tmp_path, checkpoint=tmp_path, backend="native", profile=purpose,
            device="cpu", image_digest="sha256:" + "d" * 64,
        ))
        assert result == 2
        assert seen == [(expected, expected)], "Operational replay must retain the deployed TF32 flags"
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prior[:2]
        torch.set_num_threads(prior[2])
