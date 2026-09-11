"""Hermetic replay contract tests; these never certify real-model feasibility."""

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def contract():
    assert (ROOT / "policy_guard/replay_contract.py").is_file(), (
        "The replay evidence contract must exist before accepting worker evidence"
    )
    return importlib.import_module("policy_guard.replay_contract")


def test_missing_prerequisites_report_not_run_without_model_import(tmp_path):
    result = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/replay_checkpoint_parity.py"),
            "feasibility", "--corpus", str(tmp_path / "absent-corpus"),
            "--checkpoint", str(tmp_path / "absent-checkpoint"),
            "--workspace", str(tmp_path / "evidence"), "--record", "record_0000.npz",
        ],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 2
    assert '"status": "not_run"' in result.stdout, (
        "Missing inference prerequisites must yield machine-readable not_run evidence"
    )
    assert "not run" in result.stdout


def profile():
    return {
        "backend": "native", "purpose": "diagnostic",
        "parameter_dtypes": ["torch.float32"], "buffer_dtypes": ["torch.float32"],
        "input_dtypes": ["torch.float32"], "backbone_dtypes": ["torch.float32"],
        "compute_dtypes": ["torch.float32"], "noise_dtype": "torch.float32",
        "raw_dtype": "torch.float32", "attention": ["sdpa"],
        "sdpa_calls": 4, "eval": True, "autocast": False, "tf32": False,
        "flow_steps": 4, "raw_shape": [1, 40, 132], "noise_shape": [1, 40, 132],
        "decoded_shape": [16, 6], "noise_draws": 1,
        "seed_at_sampling_boundary": True, "observer_inert": True,
        "device": "cpu", "rng_algorithm": "torch.default_generator.cpu",
        "source": {"sha256": "1" * 64}, "packages": {"torch": "test-fixture"},
        "image_digest": "sha256:" + "2" * 64,
        "checkpoint_fingerprint": "3" * 64, "backbone_fingerprint": "4" * 64,
        "instrumentation_fingerprint": "5" * 64,
        "joint_order": list((
            "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
            "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
        )),
        "camera_order": ["front", "wrist"],
    }


@pytest.mark.parametrize(
    ("key", "bad"),
    [
        ("parameter_dtypes", ["torch.bfloat16"]),
        ("buffer_dtypes", ["torch.bfloat16"]),
        ("input_dtypes", ["torch.bfloat16"]),
        ("backbone_dtypes", ["torch.bfloat16"]),
        ("compute_dtypes", ["torch.bfloat16"]),
        ("attention", ["flash_attention_2"]), ("sdpa_calls", 0),
        ("autocast", True), ("tf32", True), ("eval", False),
        ("flow_steps", 3), ("flow_steps", 5),
        ("raw_shape", [1, 16, 132]), ("noise_shape", [1, 16, 132]),
        ("decoded_shape", [40, 6]), ("noise_draws", 0),
        ("seed_at_sampling_boundary", False), ("observer_inert", False),
        ("image_digest", ""), ("backbone_fingerprint", ""),
    ],
)
def test_diagnostic_rejects_false_effective_controls(key, bad):
    observed = profile()
    contract().validate_profile(observed)
    observed[key] = bad
    with pytest.raises(ValueError):
        contract().validate_profile(observed)


def test_output_cast_is_not_fp32_evidence():
    observed = profile()
    observed["raw_dtype"] = "torch.float32"
    observed["compute_dtypes"] = ["torch.bfloat16"]
    with pytest.raises(ValueError, match="compute"):
        contract().validate_profile(observed)


def test_evidence_is_atomic_immutable_and_json_finite(tmp_path):
    api = contract()
    api.write_evidence(tmp_path, "feasibility.json", {"status": "not_run"})
    original = (tmp_path / "feasibility.json").read_bytes()
    with pytest.raises(FileExistsError):
        api.write_evidence(tmp_path, "feasibility.json", {"status": "complete"})
    assert (tmp_path / "feasibility.json").read_bytes() == original
    with pytest.raises(ValueError):
        api.write_evidence(tmp_path, "bad.json", {"value": float("nan")})
    assert not (tmp_path / "bad.json").exists()
    with pytest.raises(ValueError):
        api.write_evidence(tmp_path, "../escape.json", {})


def test_numeric_evidence_rejects_objects_and_tampering(tmp_path):
    api = contract()
    with pytest.raises(ValueError):
        api.write_tensors(tmp_path, {"raw": np.array([object()], dtype=object)})
    with pytest.raises(ValueError):
        api.write_tensors(tmp_path, {"raw": np.array([np.inf])})
    ref = api.write_tensors(tmp_path, {"raw": np.zeros((1, 40, 132), np.float32)})
    assert api.read_tensors(tmp_path, ref)["raw"].shape == (1, 40, 132)
    path = tmp_path / ref["path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="digest"):
        api.read_tensors(tmp_path, ref)


def test_real_manifest_cannot_be_passed_by_missing_or_fabricated_worker(tmp_path):
    api = contract()
    expected = [{"record": "record_0000.npz", "seed": 42}]
    report = {
        "schema_version": 1, "session": "test", "stage": "tracer",
        "status": "complete", "input_fingerprint": "1" * 64,
        "profile_fingerprint": api.fingerprint_configuration(profile()),
        "started_at": "2026-09-11T00:00:00+00:00",
        "ended_at": "2026-09-11T00:00:01+00:00",
        "expected_cases": expected, "executed_cases": [],
        "cases": [], "prerequisite_errors": [], "profile": profile(),
        "evidence_kind": "real_model",
    }
    with pytest.raises(ValueError, match="coverage"):
        api.validate_replay_manifest(report, tmp_path, expected)
    report["executed_cases"] = copy.deepcopy(expected)
    report["evidence_kind"] = "synthetic"
    with pytest.raises(ValueError, match="real_model"):
        api.validate_replay_manifest(report, tmp_path, expected)


def test_observer_forwards_real_noise_and_does_not_consume_extra_rng():
    import torch

    api = contract()
    torch.manual_seed(42)
    expected = torch.randn((1, 40, 132))
    next_expected = torch.randn(3)
    with api.sampling_observer(seed=42) as observer:
        actual = torch.randn((1, 40, 132))
    assert torch.equal(actual, expected)
    assert torch.equal(torch.randn(3), next_expected)
    assert torch.equal(observer.noise, actual)
    assert observer.noise_draws == 1


def test_observer_records_actual_sdpa_compute():
    import torch
    import torch.nn.functional as functional

    api = contract()
    q = torch.ones(1, 1, 3, 8)
    expected = functional.scaled_dot_product_attention(q, q, q)
    with api.sampling_observer(seed=42) as observer:
        actual = functional.scaled_dot_product_attention(q, q, q)
    assert torch.equal(actual, expected)
    assert observer.sdpa_calls == 1
    assert observer.compute_dtypes == {"torch.float32"}


def test_input_digest_change_is_rejected_before_inference(tmp_path):
    api = contract()
    target = tmp_path / "record.npz"
    np.savez(target, state=np.zeros(6, np.float32))
    locked = {"path": "record.npz", "sha256": api.sha256_file(target)}
    np.savez(target, state=np.ones(6, np.float32))
    with pytest.raises(ValueError, match="digest"):
        api.verify_file(tmp_path, locked)


def test_feasibility_has_no_approval_or_comparison_subcommand(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/replay_checkpoint_parity.py"), "--help"],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0
    assert "feasibility" in result.stdout
    rejected = subprocess.run(
        [sys.executable, str(ROOT / "scripts/replay_checkpoint_parity.py"), "approve"],
        capture_output=True, text=True, timeout=20,
    )
    assert rejected.returncode == 2
    assert "invalid choice" in rejected.stderr


def test_offline_metadata_and_oom_are_prerequisites_not_success():
    api = contract()
    for name in ("OfflineModeIsEnabled", "OutOfMemoryError", "LocalEntryNotFoundError"):
        error = type(name, (RuntimeError,), {})("measured failure")
        assert api.prerequisite_exception(error)
    assert not api.prerequisite_exception(ValueError("wrong manifest"))


def test_log_reference_supports_relative_workspace(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    orchestrator = importlib.import_module("replay_checkpoint_parity")
    import os

    log = tmp_path / "worker.log"
    log.write_text("actual worker failure")
    relative_workspace = Path(os.path.relpath(tmp_path))
    ref = orchestrator.evidence_reference(relative_workspace, log.resolve())
    assert ref["path"] == "worker.log"
    assert ref["sha256"] == contract().sha256_file(log)
