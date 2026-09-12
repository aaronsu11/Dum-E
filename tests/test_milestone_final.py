"""Final-regression fixtures only: no model, Docker, serial, or camera calls."""
import ast
import copy
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import numpy as np
import pytest

from policy_guard import milestone_golden as golden
from policy_guard import parity_gate as gate
from policy_guard.replay_contract import (
    CAMERA_ORDER, JOINT_ORDER, ReplayManifest, fingerprint_configuration as digest,
    profile_configuration, write_evidence, write_tensors,
)
from scripts import replay_milestone_final as api
from tests.test_parity_gate import observed


def ts(second):
    return (datetime(2026, 9, 12, tzinfo=timezone.utc) + timedelta(seconds=second)).isoformat()


def physical_files(root):
    rows = []
    for index in range(1, 4):
        workspace = root / f"physical-{index}"
        run = {"status": "partial", "started_at": ts(index*10),
               "ended_at": ts(index*10+5)}
        ref = write_evidence(workspace, "live-run.json", run)
        rows.append({"trial": index, "workspace": str(workspace), "live_run": ref,
                     "recorded_end": run["ended_at"]})
    write_evidence(root, api.SUMMARY, {
        "kind": "physical_trials_summary", "status": "complete", "trial_count": 3,
        "trial_records": rows, "recorded_at": ts(40),
    })
    return rows


def fixture(root, fault=None):
    records = [{"episode_index": e, "frame_index": f, "file": f"r{e}-{f}.npz",
                "seeds": [e*1000+f*10+s for s in range(5)], "sha256": digest([e, f]),
                "instruction": "banana"}
               for e in range(12) for f in range(10)]
    lock = {"records": records, "schedule": [{"record": r["file"], "seed": seed}
                                           for r in records for seed in r["seeds"]],
            "fingerprint": digest("lock"), "checkpoint_fingerprint": digest("checkpoint"),
            "joint_order": list(JOINT_ORDER), "camera_order": list(CAMERA_ORDER)}
    scope = {"selected": [{"record": records[e*10+5]["file"], "seed": records[e*10+5]["seeds"][0]}
                          for e in range(12)]}
    write_evidence(root, "input-lock.json", lock)
    scope_ref = write_evidence(root, "milestone-scope.json", scope)
    profile = observed("native", "operational")
    profile.update(device="cuda:0", rng_algorithm="torch.default_generator.cuda",
                   parameter_dtypes=["torch.bfloat16"], backbone_dtypes=["torch.bfloat16"])
    tensors = {k: np.zeros(shape, np.float32) for k, shape in golden.FIELDS.items()}
    candidate = {"cases": scope["selected"], "scope": scope_ref, "profile": profile,
                 "ended_at": ts(1), "tensors": write_tensors(root, tensors)}
    candidate_ref = write_evidence(root, golden.CANDIDATE, candidate)
    physical_files(root)
    ev = gate.Evidence(root, test_only=True)
    ev.identity = lambda: {"session": "test"}
    physical = api.physical_basis(ev)
    plan = {"schema_version": 1, "kind": api.KIND, "id": uuid4().hex,
            "started_at": ts(50), "worker_manifest": api.WORKER, "candidate": candidate_ref,
            "scope": scope_ref, "cases": candidate["cases"], "source_files": api.source_identity(),
            "expected_profile": profile, "physical": physical, "bounds": golden.BOUNDS}
    schedule_ref = write_evidence(root, api.SCHEDULE, plan)
    execution = api.execution_binding(ev, plan)
    report = ReplayManifest(
        session="test", stage="operational", input_fingerprint=lock["fingerprint"],
        expected_cases=candidate["cases"], started_at=ts(52), ended_at=ts(77),
        status="complete", profile=profile, profile_fingerprint=digest(profile),
        configuration_fingerprint=digest(profile_configuration(profile)), execution=execution,
        resources={"source_files": api.source_identity(), "prediction_calls": 24,
                   "observer_control_calls": 12, "process_identity": {
                       "pid": 123, "process_start_ticks": 456, "process_started_at": ts(51),
                       "boot_id": "fixture-boot"}},
    )
    values = {"raw": np.zeros((1, 40, 132), np.float32),
              "noise": np.zeros((1, 40, 132), np.float32),
              "decoded": np.zeros((16, 6), np.float32)}
    if fault == "cropped":
        values["raw"] = values["raw"][:, :16]
    if fault == "numeric":
        values["decoded"][0, 0] = 2e-5
    for index, key in enumerate(candidate["cases"]):
        record = next(r for r in records if r["file"] == key["record"])
        row = {"key": key, "record_sha256": record["sha256"], "instruction": record["instruction"],
               "tensors": write_tensors(root, values), "profile_fingerprint": digest(profile),
               "observer_inert": fault != "control", "execution": execution,
               "started_at": ts(53+index*2), "ended_at": ts(54+index*2)}
        path = f"workers/cases/{Path(api.WORKER).stem}-{index:04d}.json"
        row["evidence"] = write_evidence(root, path, {
            **row, "session": "test", "stage": "operational", "status": "complete",
            "input_fingerprint": lock["fingerprint"], "evidence_kind": "real_model", "profile": profile,
        })
        report.cases.append(row)
        report.executed_cases.append(key)
    if fault == "partial":
        report.executed_cases = report.executed_cases[:-1]
    if fault == "count":
        report.resources["prediction_calls"] = 12
    if fault == "old-process":
        report.resources["process_identity"]["process_started_at"] = ts(1)
    worker_ref = write_evidence(root, api.WORKER, asdict(report))
    (root / api.LOG).write_text("fixture only; no real model execution\n")
    inputs = {k: str(root / k) for k in ("corpus", "checkpoint", "native_cache")}
    inputs["input_fingerprint"] = lock["fingerprint"]
    container = "fixture-final"
    launch = {"schedule": schedule_ref, "manifest": worker_ref, "status": "complete", "exit_code": 0,
              "started_at": ts(51), "ended_at": ts(78), "log": ev.reference(api.LOG),
              "inputs": inputs, "container": container, "source_files": api.source_identity(),
              "argv": api.worker_command(root, inputs, container, profile["image_digest"])}
    return ev, candidate, profile, physical, plan, launch


def test_final_worker_accepts_twelve_fresh_bound_cases(tmp_path):
    ev, candidate, profile, physical, _, launch = fixture(tmp_path)
    report, arrays = api.validate_worker(ev, candidate, profile, physical, launch)
    assert len(report["cases"]) == 12 and arrays["decoded"].shape == (12, 16, 6)
    assert report["resources"]["prediction_calls"] == 24


@pytest.mark.parametrize("fault", ["partial", "cropped", "control", "count", "old-process"])
def test_worker_refuses_incomplete_or_uncontrolled_evidence(tmp_path, fault):
    ev, candidate, profile, physical, _, launch = fixture(tmp_path, fault)
    with pytest.raises(ValueError):
        api.validate_worker(ev, candidate, profile, physical, launch)


@pytest.mark.parametrize("fault", ["exit", "argv", "sources", "old-golden"])
def test_worker_refuses_failed_or_relabelled_launch(tmp_path, fault):
    ev, candidate, profile, physical, _, launch = fixture(tmp_path)
    if fault == "exit":
        launch["exit_code"] = 1
    elif fault == "argv":
        launch["argv"][launch["argv"].index("--device")+1] = "cpu"
    elif fault == "sources":
        launch["source_files"] = {k: v for k, v in launch["source_files"].items() if k != api.SCRIPT}
    else:
        launch["manifest"] = {"path": golden.WORKER, "sha256": "a"*64}
    with pytest.raises(ValueError):
        api.validate_worker(ev, candidate, profile, physical, launch)


def test_final_schedule_cannot_precede_last_physical_run_or_change_seed(tmp_path):
    ev, candidate, profile, physical, plan, _ = fixture(tmp_path)
    plan = copy.deepcopy(plan)
    plan["started_at"] = ts(34)
    with pytest.raises(ValueError, match="all physical trials"):
        api.validate_plan(ev, plan, candidate, profile, physical)
    plan["started_at"] = ts(50)
    plan["cases"][0]["seed"] += 1
    with pytest.raises(ValueError, match="cases"):
        api.validate_plan(ev, plan, candidate, profile, physical)


def test_physical_end_is_bound_to_hashed_live_run_not_summary_claim(tmp_path):
    rows = physical_files(tmp_path)
    ev = gate.Evidence(tmp_path, test_only=True)
    assert api.physical_basis(ev)["last_physical_ended_at"] == ts(35)
    # A changed physical record cannot be hidden by an unchanged summary reference.
    (Path(rows[-1]["workspace"]) / "live-run.json").write_text("{}")
    with pytest.raises(ValueError, match="digest changed"):
        api.physical_basis(gate.Evidence(tmp_path, test_only=True))


@pytest.mark.parametrize("fault", [None, "numeric"])
def test_final_closeout_interface_recomputes_comparison(tmp_path, monkeypatch, fault):
    ev, candidate, profile, physical, plan, launch = fixture(tmp_path, fault)
    launch_ref = write_evidence(tmp_path, api.LAUNCH, launch)
    _, arrays = api.validate_worker(ev, candidate, profile, physical, launch)
    comparison = golden.compare_arrays(ev.tensors(candidate["tensors"]), arrays)
    record = {
        "schema_version": 1, "kind": api.KIND, "status": "complete", "evidence_kind": "real_model",
        "candidate": ev.reference(golden.CANDIDATE), "scope": candidate["scope"],
        "cases": candidate["cases"], "case_count": 12, "bounds": golden.BOUNDS,
        "physical": physical, "source_files": api.source_identity(), "prediction_calls": 24,
        "observer_control_calls": 12, "launch": launch_ref, "started_at": plan["started_at"],
        "ended_at": ts(79), "tensors": write_tensors(tmp_path, arrays), "comparison": comparison,
    }
    write_evidence(tmp_path, api.OUTPUT, record)
    monkeypatch.setattr(api, "context", lambda _: (ev, candidate, profile, physical))
    from scripts import replay_milestone_golden
    monkeypatch.setattr(replay_milestone_golden, "current_inputs", lambda *_: launch["inputs"])
    if fault:
        with pytest.raises(ValueError, match="regression failed"):
            api.validate(ev)
    else:
        result = api.validate(ev)
        assert result["status"] == "complete"
        assert result["final_regression"] == ev.reference(api.OUTPUT)
        assert result["physical_summary"] == ev.reference(api.SUMMARY)
        assert result["last_physical_ended_at"] == ts(35)


def test_worker_argv_is_gpu_only_and_final_names_are_disjoint(tmp_path):
    inputs = {k: str(tmp_path / k) for k in ("corpus", "checkpoint", "native_cache")}
    cmd = api.worker_command(tmp_path, inputs, "fixture-final", "sha256:"+"a"*64)
    assert cmd[cmd.index("--network")+1] == "none"
    assert cmd[cmd.index("--device")+1] == "cuda:0"
    assert "--gpus" in cmd and "--read-only" in cmd
    assert cmd[cmd.index("--output-manifest")+1] == api.WORKER != golden.WORKER
    assert cmd[cmd.index("--schedule")+1] == "/evidence/" + api.SCHEDULE
    assert "--privileged" not in cmd
    assert not any("/dev/tty" in arg or "/dev/video" in arg for arg in cmd)


def test_python310_syntax_and_lazy_worker_import_without_models():
    ast.parse((api.ROOT / api.SCRIPT).read_text(), feature_version=(3, 10))
    code = (
        "import importlib.abc, sys\n"
        "class Guard(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        assert fullname not in ('policy_guard.golden', 'torch', 'gr00t'), fullname\n"
        "sys.meta_path.insert(0, Guard())\n"
        "from scripts import replay_milestone_final\n"
        "replay_milestone_final.main(['_worker', '--help'])\n"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=api.ROOT,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_fresh_replay_persists_schedule_before_launch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import replay_checkpoint_parity, replay_milestone_golden
    ev, candidate, profile, physical, _, launch = fixture(tmp_path)
    for path in (api.SCHEDULE, api.WORKER, api.LOG):
        (tmp_path / path).unlink()
    monkeypatch.setattr(api, 'context', lambda _: (ev, candidate, profile, physical))
    monkeypatch.setattr(replay_milestone_golden, 'current_inputs', lambda *args: launch['inputs'])
    monkeypatch.setattr(replay_checkpoint_parity, 'command_output', lambda _: profile['image_digest'])
    monkeypatch.setattr(replay_checkpoint_parity, 'device_snapshot', lambda: {'compute_processes': []})
    launched = []
    def unavailable(*args, **kwargs):
        assert (tmp_path / api.SCHEDULE).is_file()
        launched.append(True)
        raise OSError('fixture launch unavailable')
    monkeypatch.setattr(api.subprocess, 'Popen', unavailable)
    result = api.replay(SimpleNamespace(workspace=tmp_path, worker_timeout=1))
    assert launched == [True]
    assert result['status'] == 'failed'
    assert result['errors'] == [{'type': 'OSError', 'message': 'fixture launch unavailable'}]
    assert (tmp_path / api.OUTPUT).is_file()
