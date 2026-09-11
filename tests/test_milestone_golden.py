"""Focused scoped-golden tests: numeric fixtures only, no Docker/model calls."""
import copy
from uuid import uuid4

import numpy as np
import pytest

from policy_guard import milestone_golden as api
from policy_guard import parity_gate as gate
from policy_guard.replay_contract import write_evidence, write_tensors
from scripts import replay_milestone_golden as cli


def arrays():
    return {key: np.zeros(shape, np.float32) for key, shape in api.FIELDS.items()}


def test_full_native_raw_and_decoded_bounds_and_exact_noise():
    a, b = arrays(), arrays()
    assert api.compare_arrays(a, b)["passed"]
    b["raw"][11, 39, 131] = 2e-5
    assert not api.compare_arrays(a, b)["tensors"]["raw"]["passed"]
    a, b = arrays(), arrays()
    b["decoded"][11, 15, 5] = 2e-5
    assert not api.compare_arrays(a, b)["tensors"]["decoded"]["passed"]
    a, b = arrays(), arrays()
    b["noise"][0, 0, 0] = 1e-9
    assert not api.compare_arrays(a, b)["passed"]


@pytest.mark.parametrize("mutation", ["partial", "crop", "dtype", "nan"])
def test_partial_crop_dtype_and_nonfinite_refused(mutation):
    a, b = arrays(), arrays()
    if mutation == "partial":
        b["decoded"] = b["decoded"][:-1]
    elif mutation == "crop":
        b["raw"] = b["raw"][:, :16]
    elif mutation == "dtype":
        b["decoded"] = b["decoded"].astype(np.float64)
    else:
        b["decoded"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        api.compare_arrays(a, b)


def test_zero_rtol_even_for_large_values():
    a, b = arrays(), arrays()
    a["decoded"].fill(100)
    b["decoded"].fill(100.001)
    assert not api.compare_arrays(a, b)["passed"]


def selection():
    # Deliberately nonchronological frame numbers: stored order is authoritative.
    lock = {"records": [{"episode_index": e, "file": f"r{e}-{f}.npz",
                         "frame_index": 9-f, "seeds": [e*100+f, e*100+f+1]}
                        for e in range(12) for f in range(10)]}
    scope = {"selected": [{"record": f"r{e}-5.npz", "seed": e*100+5} for e in range(12)]}
    return lock, scope


def test_frozen_stored_middle_and_first_seed_no_reselection():
    lock, scope = selection()
    assert api.scope_cases(lock, scope) == scope["selected"]
    changed = copy.deepcopy(scope)
    changed["selected"][0]["seed"] += 1
    with pytest.raises(ValueError, match="selection changed"):
        api.scope_cases(lock, changed)
    lock["records"].pop()
    with pytest.raises(ValueError, match="ten-frame"):
        api.scope_cases(lock, scope)


def test_worker_command_has_only_scoped_gpu_inference_and_readonly_inputs(tmp_path):
    inputs = {key: str(tmp_path / key) for key in ("corpus", "checkpoint", "native_cache")}
    cmd = cli.worker_command(tmp_path, inputs, "test-milestone", "sha256:"+"a"*64)
    assert cmd[:2] == ["docker", "run"]
    assert cmd[cmd.index("--network")+1] == "none"
    assert "--read-only" in cmd and "--device" in cmd
    assert cmd[cmd.index("--device")+1] == "cuda:0"
    assert "--privileged" not in cmd
    assert "/replay/scripts/replay_milestone_golden.py" in cmd
    assert "_worker" in cmd and "/replay/scripts/replay_groot_native.py" not in cmd
    mounts = [cmd[i+1] for i, token in enumerate(cmd) if token == "--mount"]
    assert all(m.endswith(",readonly") for m in mounts if "dst=/evidence" not in m)
    assert not any("/dev/tty" in m or "/dev/video" in m for m in mounts)


def test_execution_binding_cannot_relabel_an_old_full_worker(tmp_path):
    ev = gate.Evidence(tmp_path, test_only=True)
    plan = {"kind": "milestone_native_golden_replay", "worker_manifest": api.WORKER,
            "id": uuid4().hex, "started_at": "2026-09-11T20:00:00+00:00"}
    ref = write_evidence(tmp_path, api.SCHEDULE, plan)
    assert api.execution_binding(ev, plan)["schedule"] == ref
    plan["worker_manifest"] = "workers/native-operational-full-03.json"
    with pytest.raises(ValueError, match="wrong scoped"):
        api.execution_binding(ev, plan)


def test_candidate_import_is_exact_immutable_and_never_approved(tmp_path, monkeypatch):
    ev = gate.Evidence(tmp_path, test_only=True)
    _, scope = selection()
    cases = scope["selected"]
    for name in ("milestone-scope.json", "tolerance-agreement.json"):
        write_evidence(tmp_path, name, {})
    accepted = {"report": {"path": "accepted.json", "sha256": "a"*64}, "calibration_sha256": "b"*64}
    source = {"worker": {"path": "old-worker.json", "sha256": "c"*64}}
    profile = {"backend": "native"}
    rows = [{"evidence": {"path": f"case-{i}.json", "sha256": "d"*64}} for i in range(12)]
    launch = {"started_at": "2020-01-01T00:00:00+00:00", "ended_at": "2020-01-01T00:01:00+00:00"}
    monkeypatch.setattr(api, "context", lambda _: (ev, accepted, scope, cases))
    monkeypatch.setattr(api, "captured_native", lambda *_: (source, {"profile": profile}, launch, rows, arrays()))
    ref = api.create_candidate(ev)
    record = api.validate_candidate(ev)
    assert ref == ev.reference(api.CANDIDATE)
    assert record["inference_runs_added"] == 0
    assert record["approved"] is False and record["promoted"] is False
    with pytest.raises(ValueError, match="already exists"):
        api.create_candidate(ev)
    # A within-bound substitution still is not an exact import.
    bad = arrays()
    bad["decoded"][0, 0, 0] = 1e-6
    record["tensors"] = write_tensors(tmp_path, bad)
    with pytest.raises(ValueError, match="not captured"):
        api.validate_candidate(ev)


def test_launch_validator_refuses_nonzero_exit_before_consuming_worker(tmp_path):
    ev = gate.Evidence(tmp_path, test_only=True)
    lock, scope = selection()
    write_evidence(tmp_path, "input-lock.json", lock)
    scope_ref = write_evidence(tmp_path, "milestone-scope.json", scope)
    candidate_ref = write_evidence(tmp_path, api.CANDIDATE, {})
    schedule = {"cases": scope["selected"], "candidate": candidate_ref,
                "source_files": api.source_identity(), "scope": scope_ref}
    schedule_ref = write_evidence(tmp_path, api.SCHEDULE, schedule)
    with pytest.raises(ValueError, match="worker launch failed"):
        api.validate_worker(ev, {"scope": scope_ref},
                            {"schedule": schedule_ref, "status": "complete", "exit_code": 1})


def test_cli_has_no_approval_or_promotion_command():
    for name in ("approve", "review", "promote"):
        with pytest.raises(SystemExit) as exc:
            cli.main([name, "--workspace", "/unused"])
        assert exc.value.code == 2


def test_worker_entrypoint_does_not_import_host_golden_or_model_modules():
    import subprocess
    import sys
    code = (
        "import importlib.abc, sys\n"
        "class Guard(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        assert fullname not in (\"policy_guard.golden\", \"torch\", \"gr00t\"), fullname\n"
        "sys.meta_path.insert(0, Guard())\n"
        "from scripts import replay_milestone_golden\n"
        "replay_milestone_golden.main([\"--help\"])\n"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code],
                            capture_output=True, text=True, timeout=20, cwd=api.ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
