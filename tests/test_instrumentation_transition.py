"""Synthetic transition witnesses only; no model, transport, or device calls."""
import ast
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_parity_gate as fixtures
from policy_guard import instrumentation_transition as api
from policy_guard import parity_gate as gate
from policy_guard.replay_contract import write_evidence, write_tensors


@pytest.fixture
def ast_pair(monkeypatch):
    before = b"""class ServingAttestor:
    def __init__(self): self.identity = 1
    def complete(self): return self.identity
class DumEGrootPolicyServer:
    def _predict_action_chunk(self): return self._predict_action_chunk_impl()
    def _predict_action_chunk_impl(self): return 16
"""
    after = before.replace(b"self.identity = 1", b"self.identity = 1; self.integrity = 2")
    after = after.replace(b"return self.identity", b"assert self.integrity == 2; return self.identity")
    after = after.replace(b"return self._predict_action_chunk_impl()", b"result = self._predict_action_chunk_impl(); return result")
    after += b"def file_metadata(paths): return len(paths)\ndef capture_serving_metadata(paths): return file_metadata(paths)\n"
    approved = {}
    for node in ast.parse(after).body:
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                key = node.name + "." + method.name
                if key in api.AST_CHANGES:
                    approved[key] = api.sha(ast.dump(method, include_attributes=False).encode())
        elif node.name in api.AST_CHANGES:
            approved[node.name] = api.sha(ast.dump(node, include_attributes=False).encode())
    monkeypatch.setattr(api, "AST_CHANGES", approved)
    monkeypatch.setattr(api, "BEFORE_SERVER", api.sha(before))
    return before, after


def test_exact_instrumentation_ast_and_formatting_only(ast_pair):
    before, after = ast_pair
    api.validate_server_change(before, after)
    api.validate_server_change(before, b"# formatting is not inference\n" + after)


@pytest.mark.parametrize("old,new", [
    (b"return 16", b"return 40"),
    (b"self.identity = 1", b"self.identity = 0"),
    (b"return len(paths)", b"return 0"),
    (b"assert self.integrity == 2", b"assert True"),
    (b"return result", b"return []"),
])
def test_inference_changes_and_unreviewed_instrumentation_are_refused(ast_pair, old, new):
    before, after = ast_pair
    with pytest.raises(ValueError):
        api.validate_server_change(before, after.replace(old, new))


def test_wrong_before_bytes_and_added_module_execution_refused(ast_pair):
    before, after = ast_pair
    with pytest.raises(ValueError, match="baseline"):
        api.validate_server_change(before + b"\n", after)
    with pytest.raises(ValueError, match="outside"):
        api.validate_server_change(before, after + b"raise RuntimeError('new behavior')\n")


def test_projected_profiles_only_change_serving_source(monkeypatch):
    profile = fixtures.observed("lerobot", "operational")
    monkeypatch.setattr(api, "BEFORE_SERVER", profile["owned_source_files"][api.SERVER])
    profiles = {"profiles": [{"backend": "lerobot", "purpose": "operational", "observed": profile}],
                "serving_configuration": gate.operational_semantics(profile)}
    original = copy.deepcopy(profiles)
    updated = api.projected_profiles(profiles, "f" * 64)
    assert profiles == original and updated["profiles"] == profiles["profiles"]
    before, after = profiles["serving_configuration"], updated["serving_configuration"]
    assert {key for key in before if before[key] != after[key]} == {"source_fingerprint"}


@pytest.fixture
def smoke_case(tmp_path, monkeypatch):
    target, previous = tmp_path / "new", tmp_path / "old"
    target.mkdir(); previous.mkdir()
    case = {"record": "record_0005.npz", "seed": 20265907}
    old_tensor = write_tensors(target, {"decoded": np.zeros((16, 6), np.float32)})
    row = {"key": case, "status": "complete", "tensors": old_tensor}
    row["evidence"] = write_evidence(target, "original-case.json", row)
    worker = write_evidence(target, "original-worker.json", {"cases": [row], "executed_cases": [case]})
    write_evidence(target, "milestone-report.json", {
        "cases": [case], "proofs": {"lerobot-operational": {"worker": worker}},
    })
    write_evidence(target, "input-lock.json", {})
    write_evidence(target, "profiles.json", {"serving_configuration": fixtures.semantics()})
    write_evidence(previous, "live-run.json", {"ended_at": fixtures.ts(5)})
    inputs = {"state": np.zeros(6), "video_front": np.zeros((480, 640, 3), np.uint8),
              "video_wrist": np.zeros((480, 640, 3), np.uint8)}
    monkeypatch.setattr(api, "load_case", lambda *_: (inputs, {"instruction": "banana"}))
    observation = dict.fromkeys(gate.JOINT_ORDER, 0.)
    observation.update(front=inputs["video_front"], wrist=inputs["video_wrist"], task="banana")
    runtime = fixtures.runtime(30)
    runtime["attestation"]["loaded_at"] = fixtures.ts(20)
    runtime["attestation"]["evidence_kind"] = "real_model"
    runtime["attestation"]["semantic_configuration"]["seed_policy"] = {"mode": "fixed", "seed": case["seed"]}
    runtime["attestation"]["configuration_fingerprint"] = fixtures.digest(runtime["attestation"]["semantic_configuration"])
    runtime["request"].update(observation_sha256=gate.observation_fingerprint(observation),
                              output_sha256=gate.array_fingerprint(np.zeros((16, 6), np.float32)))
    smoke = {"schema_version": 1, "status": "complete", "evidence_kind": "real_model",
             "case": case, "server_sha256": "f" * 64, "baseline_case": row["evidence"],
             "decoded": old_tensor, "corpus": str(tmp_path / "never-read"), "runtime": runtime,
             "started_at": fixtures.ts(29), "ended_at": fixtures.ts(33)}
    return target, previous, smoke


def check_smoke(setup):
    target, previous, smoke = setup
    ref = write_evidence(target, "smoke.json", smoke)
    return api.validate_smoke(gate.Evidence(target), ref, gate.Evidence(previous), "f" * 64)


def test_full_fixed_seed_observation_output_witness(smoke_case):
    assert check_smoke(smoke_case)["status"] == "complete"


@pytest.mark.parametrize("fault", ["seed", "observation", "server", "fixture", "stale-load", "partial", "changed-output", "nan"])
def test_smoke_refuses_non_equivalent_or_unbound_witness(smoke_case, fault):
    target, _, smoke = smoke_case
    if fault == "seed":
        smoke["runtime"]["attestation"]["semantic_configuration"]["seed_policy"] = {"mode": "ambient", "seed": None}
    elif fault == "observation":
        smoke["runtime"]["request"]["observation_sha256"] = "0" * 64
    elif fault == "server":
        smoke["server_sha256"] = "0" * 64
    elif fault == "fixture":
        smoke["evidence_kind"] = "test_only"
    elif fault == "stale-load":
        smoke["runtime"]["attestation"]["loaded_at"] = fixtures.ts(2)
    else:
        array = np.zeros((15 if fault == "partial" else 16, 6), np.float32)
        if fault != "partial":
            array[0, 0] = np.nan if fault == "nan" else 1e-7
        if fault == "nan":
            # Numeric artifact writer itself rejects invalid evidence.
            with pytest.raises(ValueError):
                write_tensors(target, {"decoded": array})
            return
        smoke["decoded"] = write_tensors(target, {"decoded": array})
    with pytest.raises(ValueError):
        check_smoke(smoke_case)


def test_existing_runtime_preserved_and_repeated_setup_refused(tmp_path, monkeypatch):
    baseline, target = tmp_path / "old", tmp_path / "new"
    baseline.mkdir()
    write_evidence(baseline, "profiles.json", {"baseline": True})
    write_evidence(baseline, "accepted.json", {"passed": True})
    write_evidence(baseline, "live-run.json", {"status": "failed", "stop_reason": "operator signal 2"})
    write_evidence(baseline, "live-approval.json", {"decision": "approved"})
    (baseline / "runtime-live").mkdir()
    (baseline / "runtime-live/old").write_text("old")
    (target / "runtime-live").mkdir(parents=True)
    (target / "runtime-live/current").write_text("active parent server")
    (target / "runtime-smoke").mkdir()
    (target / "runtime-smoke/current").write_text("fixed seed server")
    for name in ("serving-loaded-01.json", "latency-smoke.json", "latency-server-timings.json"):
        (target / name).write_text("parent-owned diagnostics")
        (baseline / name).write_text("old diagnostic")
    before = {p.name: p.read_bytes() for p in baseline.iterdir() if p.is_file()}
    monkeypatch.setattr(api, "BASELINE", {name: api.sha(data) for name, data in before.items()})
    monkeypatch.setattr(api, "validate_server_change", lambda *_: None)
    monkeypatch.setattr(api, "projected_profiles", lambda *_: {"new serving source": True})
    args = api.prepare_restart_view(target, baseline_workspace=baseline, before_server=b"archived source")
    assert args["baseline_workspace"] == str(baseline)
    assert (target / "runtime-live/current").read_text() == "active parent server"
    assert not (target / "runtime-live/old").exists()
    assert (target / "runtime-smoke/current").read_text() == "fixed seed server"
    assert (target / "latency-smoke.json").read_text() == "parent-owned diagnostics"
    assert not (target / "live-run.json").exists() and not (target / "live-approval.json").exists()
    assert (target / "accepted.json").read_bytes() == before["accepted.json"]
    assert {p.name: p.read_bytes() for p in baseline.iterdir() if p.is_file()} == before
    with pytest.raises(ValueError, match="already contains evidence"):
        api.prepare_restart_view(target, baseline_workspace=baseline, before_server=b"archived source")


def test_broken_transition_is_not_legacy_fallback(tmp_path):
    (tmp_path / api.RECORD).symlink_to("missing.json")
    assert api.present(tmp_path)
    with pytest.raises(FileNotFoundError):
        api.require_source(gate.Evidence(tmp_path), api.SERVER, api.BEFORE_SERVER)
