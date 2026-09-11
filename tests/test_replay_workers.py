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
    report, expected, identity = complete_manifest_fixture(tmp_path)
    report["executed_cases"] = []
    report["cases"] = []
    with pytest.raises(ValueError, match="coverage"):
        api.validate_replay_manifest(report, tmp_path, expected, **identity)
    report["executed_cases"] = copy.deepcopy(expected)
    report["evidence_kind"] = "synthetic"
    with pytest.raises(ValueError, match="real_model"):
        api.validate_replay_manifest(report, tmp_path, expected, **identity)


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


def native_worker():
    sys.path.insert(0, str(ROOT / "scripts"))
    module = importlib.import_module("replay_groot_native")
    assert hasattr(module, "pinned_native_cache"), "Native loader must provide a pinned local-cache context"
    return module


def cache_fixture(tmp_path):
    model = tmp_path / "hub/models--nvidia--Cosmos-Reason2-2B"
    snapshot = model / "snapshots" / contract().BACKBONE_REVISION
    snapshot.mkdir(parents=True)
    (model / "refs").mkdir()
    (model / "refs/main").write_text(contract().BACKBONE_REVISION)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (snapshot / name).write_text("{}")
    return tmp_path / "hub", snapshot


def test_pinned_native_cache_preserves_literal_name_and_restores_cwd(tmp_path):
    module = native_worker()
    hub, snapshot = cache_fixture(tmp_path)
    before = Path.cwd()
    with module.pinned_native_cache(hub):
        temporary = Path.cwd()
        local = Path("nvidia/Cosmos-Reason2-2B")
        assert local.is_dir()
        assert local.resolve() == snapshot.resolve()
        assert local.is_symlink()
    assert Path.cwd() == before
    assert not temporary.exists()
    assert snapshot.is_dir()


def test_pinned_native_cache_restores_cwd_after_loader_failure(tmp_path):
    module = native_worker()
    hub, snapshot = cache_fixture(tmp_path)
    before = Path.cwd()
    with pytest.raises(RuntimeError, match="loader failure"):
        with module.pinned_native_cache(hub):
            temporary = Path.cwd()
            raise RuntimeError("loader failure")
    assert Path.cwd() == before
    assert not temporary.exists()


@pytest.mark.parametrize("corruption", ["missing", "wrong_ref", "escape"])
def test_pinned_native_cache_rejects_unpinned_or_escaping_snapshot(tmp_path, corruption):
    module = native_worker()
    hub, snapshot = cache_fixture(tmp_path)
    if corruption == "missing":
        (snapshot / "tokenizer.json").unlink()
    elif corruption == "wrong_ref":
        (hub / "models--nvidia--Cosmos-Reason2-2B/refs/main").write_text("0" * 40)
    else:
        (snapshot / "config.json").unlink()
        outside = tmp_path / "outside.json"
        outside.write_text("{}")
        (snapshot / "config.json").symlink_to(outside)
    before = Path.cwd()
    with pytest.raises((ValueError, contract().PrerequisiteError)):
        with module.pinned_native_cache(hub):
            pytest.fail("invalid snapshot admitted")
    assert Path.cwd() == before


@pytest.mark.parametrize("reduction", ["manual", "autocast", "aten"])
def test_hidden_reduced_precision_compute_cannot_pass_fp32_profile(reduction):
    import torch
    import torch.nn.functional as functional

    api = contract()
    x = torch.ones(1, 8)
    weight = torch.ones(8, 8)
    q = torch.ones(1, 1, 3, 8)
    with api.sampling_observer(42) as observer:
        functional.scaled_dot_product_attention(q, q, q)
        if reduction == "autocast":
            with torch.autocast("cpu", dtype=torch.bfloat16):
                hidden = functional.linear(x, weight)
        elif reduction == "aten":
            hidden = torch.ops.aten.mm.default(x.bfloat16(), weight.bfloat16())
        else:
            hidden = functional.linear(x.bfloat16(), weight.bfloat16())
        returned = hidden.float()
    assert returned.dtype == torch.float32
    observed = profile()
    observed.update(compute_dtypes=sorted(observer.compute_dtypes), autocast=observer.autocast)
    with pytest.raises(ValueError, match="compute|autocast"):
        api.validate_profile(observed)


def test_verified_tensor_bytes_survive_path_mutation(tmp_path):
    api = contract()
    target = tmp_path / "record.npz"
    original = np.zeros((1, 40, 132), np.float32)
    np.savez(target, raw=original)
    ref = {"path": target.name, "sha256": api.sha256_file(target)}
    captured = api.verify_file(tmp_path, ref)
    np.savez(target, raw=np.ones_like(original))
    decoded = api.load_numeric(captured, expected_keys={"raw"})
    assert np.array_equal(decoded["raw"], original), "Must parse the exact bytes whose digest passed"


def complete_manifest_fixture(tmp_path):
    api = contract()
    expected = [{"record": "record_0000.npz", "seed": 42}]
    locked = {
        "fingerprint": "1" * 64, "checkpoint_fingerprint": "3" * 64,
        "schedule": expected,
        "records": [{"file": "record_0000.npz", "sha256": "6" * 64, "instruction": "real locked task", "seeds": [42]}],
        "joint_order": list(api.JOINT_ORDER), "camera_order": list(api.CAMERA_ORDER),
    }
    current = profile()
    current["owned_source_files"] = {"worker.py": "7" * 64}
    current["owned_source_fingerprint"] = api.fingerprint_configuration(current["owned_source_files"])
    reference = api.write_tensors(tmp_path, {
        "raw": np.zeros((1, 40, 132), np.float32), "noise": np.ones((1, 40, 132), np.float32),
        "decoded": np.zeros((16, 6), np.float32),
    })
    report = {
        "schema_version": 1, "session": "expected-session", "stage": "diagnostic",
        "status": "complete", "input_fingerprint": locked["fingerprint"],
        "profile_fingerprint": api.fingerprint_configuration(current), "profile": current,
        "started_at": "2026-09-11T00:00:00+00:00", "ended_at": "2026-09-11T00:00:01+00:00",
        "expected_cases": expected, "executed_cases": expected, "prerequisite_errors": [],
        "evidence_kind": "real_model", "cases": [{"key": expected[0], "record_sha256": "6" * 64,
        "instruction": "real locked task", "observer_inert": True, "tensors": reference}],
    }
    identity = {
        "expected_session": "expected-session", "input_lock": locked,
        "backend": "native", "purpose": "diagnostic", "checkpoint_fingerprint": "3" * 64,
        "image_digest": current["image_digest"], "source_files": current["owned_source_files"], "device": "cpu",
    }
    return report, expected, identity


@pytest.mark.parametrize("corruption", ["session", "input", "record", "instruction", "backend", "purpose", "checkpoint", "image", "source", "device"])
def test_manifest_binds_exact_launcher_identity(tmp_path, corruption):
    api = contract()
    report, expected, identity = complete_manifest_fixture(tmp_path)
    import inspect

    assert "expected_session" in inspect.signature(api.validate_replay_manifest).parameters, "Manifest validator must require the launcher identity"
    api.validate_replay_manifest(report, tmp_path, expected, **identity)
    if corruption == "session":
        report["session"] = "another-session"
    elif corruption == "input":
        report["input_fingerprint"] = "0" * 64
    elif corruption == "record":
        report["cases"][0]["record_sha256"] = "0" * 64
    elif corruption == "instruction":
        report["cases"][0]["instruction"] = "different task"
    else:
        key = {"checkpoint": "checkpoint_fingerprint", "image": "image_digest", "source": "owned_source_files"}.get(corruption, corruption)
        report["profile"][key] = {"worker.py": "0" * 64} if corruption == "source" else "different"
        report["profile_fingerprint"] = api.fingerprint_configuration(report["profile"])
    with pytest.raises(ValueError):
        api.validate_replay_manifest(report, tmp_path, expected, **identity)


def test_explicit_cpu_diagnostic_keeps_operational_cuda_and_arguments(tmp_path):
    from types import SimpleNamespace
    module = importlib.import_module("scripts.replay_checkpoint_parity")
    args = SimpleNamespace(
        device="cuda:0", diagnostic_device="cpu", stock_device="cpu",
        corpus=tmp_path, checkpoint=tmp_path, workspace=tmp_path, native_cache=tmp_path,
    )
    diagnostic = module.worker_argv(args, "native", "diagnostic", "image", "worker.json", "test")
    operational = module.worker_argv(args, "native", "operational", "image", "worker.json", "test")
    assert diagnostic[diagnostic.index("--device") + 1] == "cpu", "CPU diagnostic selection must reach the actual worker"
    assert "--gpus" not in diagnostic
    assert operational[operational.index("--device") + 1] == "cuda:0"
    assert operational[operational.index("--gpus") + 1] == "all"
    stock = module.worker_argv(args, "native", "stock-capacity", "image", "worker.json", "test")
    assert stock[stock.index("--device") + 1] == "cpu"
    assert stock[stock.index("--gpus") + 1] == "all"


def schedule_lock():
    api = contract()
    records = [{"file": f"record_{i:04d}.npz", "seeds": [42, 43, 44, 45, 46],
                "instruction": ("banana", "apple", "orange")[i % 3], "sha256": "6" * 64}
               for i in range(120)]
    return {"records": records, "schedule": [{"record": r["file"], "seed": seed} for r in records for seed in r["seeds"]],
            "joint_order": list(api.JOINT_ORDER), "camera_order": list(api.CAMERA_ORDER), "fingerprint": "1" * 64}


@pytest.mark.parametrize("corruption", ["missing_record", "duplicate_record", "missing_seed", "duplicate_seed", "permuted_camera", "permuted_joint", "permuted_schedule"])
def test_complete_schedule_rejects_missing_duplicate_or_permuted_membership(corruption):
    api = contract()
    assert callable(getattr(api, "validate_schedule", None)), "Complete schedule validation must exist"
    lock = schedule_lock()
    api.validate_schedule(lock, lock["schedule"], "replay")
    if corruption == "missing_record":
        lock["records"].pop()
    elif corruption == "duplicate_record":
        lock["records"][-1] = lock["records"][0]
    elif corruption == "missing_seed":
        lock["records"][0]["seeds"].pop()
    elif corruption == "duplicate_seed":
        lock["records"][0]["seeds"][-1] = 42
    elif corruption == "permuted_camera":
        lock["camera_order"].reverse()
    elif corruption == "permuted_joint":
        lock["joint_order"].reverse()
    else:
        lock["schedule"].reverse()
    with pytest.raises(ValueError):
        api.validate_schedule(lock, lock["schedule"], "replay")


def test_repeatability_schedule_has_fixed_warm_changed_and_cold_processes():
    api = contract()
    assert callable(getattr(api, "repeatability_schedule", None)), "Repeatability scheduling must be executable"
    lock = schedule_lock()
    schedule = api.repeatability_schedule(lock)
    assert len(schedule["groups"]) == 13
    warm = schedule["groups"][0]
    assert warm["id"] == "warm" and len(warm["cases"]) == 36
    for record in (0, 60, 80, 90, 100, 119):
        name = f"record_{record:04d}.npz"
        cases = [c for c in warm["cases"] if c["record"] == name]
        assert [c["seed"] for c in cases] == [42] * 5 + [43]
        assert [c["mode"] for c in cases] == ["warm"] * 5 + ["changed"]
        cold = [g for g in schedule["groups"][1:] if g["cases"][0]["record"] == name]
        assert len(cold) == 2 and all(len(g["cases"]) == 1 for g in cold)
        assert len({g["id"] for g in cold}) == 2
        assert all(g["cases"][0]["seed"] == 42 for g in cold)


def fixture_trace(key, lock):
    arrays = {"raw": np.zeros((1, 40, 132), np.float32), "noise": np.ones((1, 40, 132), np.float32),
              "decoded": np.zeros((16, 6), np.float32)}
    entry = next(r for r in lock["records"] if r["file"] == key["record"])
    return profile(), arrays, entry


def test_streaming_retains_each_success_and_failure_without_completing_partial_result(tmp_path):
    api = contract()
    assert callable(getattr(api, "execute_cases", None)), "Case streaming must publish durable per-case evidence"
    lock = schedule_lock()
    cases = lock["schedule"][:3]
    report = api.ReplayManifest("fixture", "diagnostic", lock["fingerprint"], cases, evidence_kind="hermetic_fixture")
    seen = []
    def trace(key):
        seen.append(key)
        if len(seen) == 2:
            raise api.PrerequisiteError("measured fixture failure")
        return fixture_trace(key, lock)
    with pytest.raises(api.PrerequisiteError):
        api.execute_cases(report, tmp_path, lock, cases, trace, "fixture")
    assert seen == cases[:2]
    assert report.status != "complete" and report.executed_cases == cases[:1]
    saved = api.read_json(tmp_path / report.cases[0]["evidence"]["path"])
    assert saved["key"] == cases[0] and saved["evidence_kind"] == "hermetic_fixture"
    failures = list((tmp_path / "workers/failures").glob("*.json"))
    assert len(failures) == 1 and api.read_json(failures[0])["key"] == cases[1]


def test_streaming_reuses_order_and_rejects_mixed_effective_configuration(tmp_path):
    api = contract()
    assert callable(getattr(api, "execute_cases", None)), "Case streaming must bind effective configuration"
    lock = schedule_lock()
    cases = lock["schedule"][:2]
    report = api.ReplayManifest("fixture", "diagnostic", lock["fingerprint"], cases, evidence_kind="hermetic_fixture")
    seen = []
    def trace(key):
        seen.append(key)
        observed, tensors, entry = fixture_trace(key, lock)
        if len(seen) == 2:
            observed["checkpoint_fingerprint"] = "0" * 64
        return observed, tensors, entry
    with pytest.raises(ValueError, match="configuration"):
        api.execute_cases(report, tmp_path, lock, cases, trace, "fixture")
    assert seen == cases and report.status != "complete"


def test_native_operational_mapping_observes_real_policy_collaborator_changes():
    module = native_worker()
    adapter = module.NativeReplay.__new__(module.NativeReplay)
    adapter.purpose = "operational"
    class ServingPolicy:
        offset = 0
        def get_action(self, observation):
            assert observation["state"]["single_arm"].shape == (1, 1, 5)
            assert observation["language"]["annotation.human.task_description"] == [["banana"]]
            value = observation["state"]["single_arm"][0, 0, 0] + self.offset
            return {"single_arm": np.full((1, 16, 5), value, np.float32), "gripper": np.full((1, 16, 1), 7, np.float32)}, {}
    adapter.policy = ServingPolicy()
    arrays = {"state": np.arange(6, dtype=np.float32), "video_front": np.zeros((480, 640, 3), np.uint8), "video_wrist": np.zeros((480, 640, 3), np.uint8)}
    try:
        original = adapter.predict(arrays, {"instruction": "banana"})
        adapter.policy.offset = 3
        changed = adapter.predict(arrays, {"instruction": "banana"})
    except ImportError as exc:
        pytest.fail(f"Operational mapping must be testable through the serving collaborator without diagnostic imports: {exc}")
    assert np.all(changed[:, :5] - original[:, :5] == 3)
    assert np.all(changed[:, 5] == original[:, 5])


def test_full_600_case_stream_preserves_order_instructions_and_distinct_seeds(tmp_path):
    api = contract()
    lock = schedule_lock()
    cases = lock["schedule"]
    api.validate_schedule(lock, cases, "replay")
    report = api.ReplayManifest("fixture", "diagnostic", lock["fingerprint"], cases, evidence_kind="hermetic_fixture")
    seen = []
    def trace(key):
        seen.append(key)
        observed, tensors, entry = fixture_trace(key, lock)
        observed["floating_operation_count"] = 100 + len(seen)
        return observed, tensors, entry
    api.execute_cases(report, tmp_path, lock, cases, trace, "fixture")
    assert seen == cases == report.executed_cases
    assert len(report.cases) == 600 and not report.failure_ledger
    assert {case["instruction"] for case in report.cases} == {"banana", "apple", "orange"}
    assert {case["key"]["seed"] for case in report.cases[:5]} == {42, 43, 44, 45, 46}
    assert len(list((tmp_path / "workers/cases").glob("*.json"))) == 600
    assert report.evidence_kind == "hermetic_fixture"



def golden_execution_fixture(workspace, *, collection="pre-live", identity="a" * 32):
    api = contract()
    lock = schedule_lock()
    schedule = {"schema_version": 1, "session": "TEST_ONLY-execution",
                "input_fingerprint": lock["fingerprint"], "kind": "replay", "cases": lock["schedule"],
                "evidence_kind": "test_only", "execution": {
                    "id": identity, "collection": collection,
                    "started_at": "2026-09-11T00:00:00+00:00",
                    "worker_manifest": f"workers/native-operational-golden-{collection}.json"}}
    reference = api.write_evidence(workspace, f"golden-{collection}-schedule.json", schedule)
    report = api.ReplayManifest("TEST_ONLY-execution", "operational", lock["fingerprint"],
                                lock["schedule"], evidence_kind="test_only",
                                started_at="2026-09-11T00:00:01+00:00")
    binding = {**schedule["execution"], "schedule": reference}
    return api, lock, report, schedule, binding


def test_golden_case_producer_preserves_execution_in_durable_bytes(tmp_path):
    api, lock, report, schedule, binding = golden_execution_fixture(tmp_path)
    # Assigning the expected extension works on the old dataclass too: RED must
    # demonstrate that the actual writer drops it, not fail at construction.
    report.execution = binding
    def trace(key):
        observed, arrays, entry = fixture_trace(key, lock)
        observed["purpose"] = "operational"
        return observed, arrays, entry
    api.execute_cases(report, tmp_path, lock, lock["schedule"][:2], trace,
                      "native-operational-golden-pre-live")
    for row in report.cases:
        saved = api.read_json(tmp_path / row["evidence"]["path"])
        assert saved.get("execution") == binding, "Case writer must retain actual execution and schedule identity"
        assert row["execution"] == saved["execution"]
        assert report.started_at <= saved["started_at"] <= saved["ended_at"]


@pytest.mark.parametrize("problem", ["wrong-worker", "wrong-schedule", "no-execution", "bad-id", "future-start"])
def test_golden_worker_binds_schedule_before_any_trace(tmp_path, problem):
    api, lock, report, schedule, binding = golden_execution_fixture(tmp_path)
    assert callable(getattr(api, "bind_worker_execution", None)), "Worker entrypoint needs schedule binding"
    output = binding["worker_manifest"]
    path = tmp_path / binding["schedule"]["path"]
    if problem == "wrong-worker":
        output = "workers/native-operational-golden-candidate.json"
    elif problem == "wrong-schedule":
        path = tmp_path / "TEST_ONLY-wrong-schedule.json"
        api.write_evidence(tmp_path, path.name, schedule)
    elif problem == "no-execution":
        schedule.pop("execution")
    elif problem == "bad-id":
        schedule["execution"]["id"] = "old-case"
    elif problem == "future-start":
        schedule["execution"]["started_at"] = "2026-09-11T01:00:00+00:00"
    with pytest.raises(ValueError):
        api.bind_worker_execution(report, tmp_path, path, schedule, output)
    assert not report.cases


def test_golden_worker_schedule_binding_reaches_case_producer(tmp_path):
    api, lock, report, schedule, binding = golden_execution_fixture(tmp_path)
    assert callable(getattr(api, "bind_worker_execution", None)), "Worker entrypoint needs schedule binding"
    api.bind_worker_execution(report, tmp_path, tmp_path / binding["schedule"]["path"],
                              schedule, binding["worker_manifest"])
    assert report.execution == binding
    times = iter(["2026-09-11T00:00:02+00:00", "2026-09-11T00:00:03+00:00"])
    def trace(key):
        observed, arrays, entry = fixture_trace(key, lock)
        observed["purpose"] = "operational"
        return observed, arrays, entry
    api.execute_cases(report, tmp_path, lock, lock["schedule"][:1], trace,
                      "native-operational-golden-pre-live", clock=lambda: next(times))
    row = report.cases[0]
    assert row["execution"] == binding
    assert row["started_at"] == "2026-09-11T00:00:02+00:00"
    assert row["ended_at"] == "2026-09-11T00:00:03+00:00"
    assert api.read_json(tmp_path / row["evidence"]["path"])["execution"] == binding
