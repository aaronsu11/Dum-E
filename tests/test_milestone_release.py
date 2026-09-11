"""Scoped release integration only: synthetic metadata, no models or devices."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_parity_gate as fixtures
from policy_guard import milestone_acceptance, milestone_golden
from policy_guard import parity_gate as gate
from policy_guard.replay_contract import CAMERA_ORDER, JOINT_ORDER


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Replace only the separately tested numerical validators, not release gates."""
    fixtures.save(tmp_path, "session.json", {"schema_version": 1, "session_id": "test-session"})
    lock = {
        "schema_version": 1, "records": [
            {"file": f"record_{i:04d}.npz", "seeds": [1, 2, 3, 4, 5],
             "sha256": fixtures.digest(i), "instruction": "banana"} for i in range(120)
        ], "joint_order": list(JOINT_ORDER), "camera_order": list(CAMERA_ORDER),
        "checkpoint_fingerprint": fixtures.digest("checkpoint"),
    }
    lock["schedule"] = [{"record": r["file"], "seed": s} for r in lock["records"] for s in r["seeds"]]
    lock["fingerprint"] = fixtures.digest(lock)
    fixtures.save(tmp_path, "input-lock.json", lock)
    fixtures.save(tmp_path, "profiles.json", {
        "schema_version": 1, "session": "test-session",
        "profiles": [{"backend": b, "purpose": p, "observed": fixtures.observed(b, p)}
                     for b in ("native", "lerobot") for p in ("diagnostic", "operational")],
        "serving_configuration": fixtures.semantics(),
    })
    fixtures.save(tmp_path, "calibration.json", {
        "session": "test-session", "status": "complete", "evidence_kind": "test_only",
        "calibration_sha256": fixtures.digest("calibration"), "drift_errors": [],
    })
    fixtures.save(tmp_path, "milestone-acceptance.json", {
        "status": "complete", "ended_at": fixtures.ts(10),
        "limits": {"max_abs": [2, 2, 2, 2, 2, 1]},
        "units": ["degrees"] * 5 + ["gripper_normalized_points"],
        "metrics": {"max_abs": [0.1] * 6}, "caveats": ["Only 12 selected observations"],
    })
    for name in ("milestone-criteria-authorization.json", "milestone-scope.json"):
        fixtures.save(tmp_path, name, {"fixture": name})
    fixtures.save(tmp_path, "milestone-report.json", {"status": "failed"})
    fixtures.save(tmp_path, "milestone-golden-candidate.json", {"approved": False, "promoted": False})
    fixtures.save(tmp_path, "milestone-golden-replay.json", {
        "status": "complete", "ended_at": fixtures.ts(12), "approved": False, "promoted": False,
    })

    def acceptance(ev):
        record = ev.json("milestone-acceptance.json")
        gate.require(record["status"] == "complete", "fixture acceptance failed")
        calibration, ref = gate._calibration(ev)
        return {**ev.identity(), "status": "complete",
                "report": ev.reference("milestone-acceptance.json"),
                "configuration_fingerprint": fixtures.digest(fixtures.semantics()),
                "calibration_sha256": calibration, "calibration": ref,
                "ended_at": record["ended_at"]}

    def golden(ev):
        record = ev.json("milestone-golden-replay.json")
        gate.require(record["status"] == "complete", "fixture golden replay failed")
        return {"candidate": ev.reference("milestone-golden-candidate.json"),
                "replay": ev.reference("milestone-golden-replay.json"),
                "ended_at": record["ended_at"], "status": "complete"}

    monkeypatch.setattr(milestone_acceptance, "validate_milestone_acceptance", acceptance)
    monkeypatch.setattr(milestone_golden, "validate_scoped_golden", golden)
    return tmp_path


def ev(workspace):
    return gate.Evidence(workspace, test_only=True)


def prepare(workspace):
    ref, _ = fixtures.preflight(workspace, "review", 20)
    fixtures.cli().prepare_live(ev(workspace), ref["path"], clock=lambda: fixtures.ts(24))
    return ref


def approve(workspace):
    ref = prepare(workspace)
    fixtures.review(workspace, "live", 25)
    return fixtures.preflight(workspace, "live", 30, previous=ref, reason="Explicit approved live freshness")


def test_review_readiness_is_not_approval_and_cli_binds_combined_scope(workspace, capsys):
    prepare(workspace)
    assert gate.validate_release_review(ev(workspace))["release_evidence"]["acceptance_mode"] == "milestone_12"
    with pytest.raises(FileNotFoundError):
        gate.validate_live_approval(ev(workspace))
    prompts = []
    answers = iter(["Operator Fixture", "Accept scoped reference and physical trial", "approve"])

    def answer(prompt):
        prompts.append(prompt)
        return next(answers)

    fixtures.cli().record_decision(workspace, "live", prompt=answer,
                                   clock=lambda: fixtures.ts(25), test_only=True)
    record = gate.validate_live_approval(ev(workspace))
    assert record["approval_scope"] == ["scoped-native-reference", "three-trial-physical-test"]
    refs = {r["path"]: r["sha256"] for r in record["evidence_reviewed"]}
    for name in ("milestone-acceptance", "milestone-criteria-authorization", "milestone-scope",
                 "milestone-report", "milestone-golden-candidate", "milestone-golden-replay"):
        assert refs[name + ".json"] == ev(workspace).reference(name + ".json")["sha256"]
    output = capsys.readouterr().out
    assert "12 observations / one seed" in output and "600 cases" not in output
    assert "preserved strict result: failed" in output and "degrees" in output
    assert refs["milestone-golden-candidate.json"] in output
    assert refs["milestone-golden-replay.json"] in output
    assert "reference AND three-trial physical test" in prompts[-1]
    assert not (workspace / "golden-approval.json").exists()
    assert ev(workspace).json("milestone-golden-candidate.json")["approved"] is False
    assert ev(workspace).json("milestone-report.json")["status"] == "failed"


@pytest.mark.parametrize("name", [
    "milestone-acceptance", "milestone-criteria-authorization", "milestone-scope",
    "milestone-report", "milestone-golden-candidate", "milestone-golden-replay",
])
def test_changed_bound_evidence_refuses_even_with_cached_caller(workspace, name):
    ref, runtime = approve(workspace)
    snapshot = ev(workspace)
    gate.validate_live_approval(snapshot)
    fixtures.rewrite(workspace, name + ".json", lambda r: r.update(tampered=True))
    with pytest.raises(ValueError, match="identity changed"):
        gate.assert_live_release(snapshot, ref, expected_stage="live", runtime=runtime,
                                 current_calibration_sha256=fixtures.digest("calibration"),
                                 now=fixtures.ts(33))


@pytest.mark.parametrize("scope", [None, ["three-trial-physical-test"], ["scoped-native-reference"]])
def test_generic_or_partial_approval_cannot_authorize_combined_release(workspace, scope):
    approve(workspace)
    decision = copy.deepcopy(ev(workspace).json("live-approval.json"))
    decision["approval_scope"] = scope
    with pytest.raises(ValueError, match="explicit combined"):
        gate.validate_live_approval(ev(workspace), decision=decision)


@pytest.mark.parametrize("artifact,end", [
    ("milestone-acceptance.json", 21), ("milestone-golden-replay.json", 20),
])
def test_review_must_follow_completed_acceptance_and_golden(workspace, artifact, end):
    fixtures.rewrite(workspace, artifact, lambda r: r.update(ended_at=fixtures.ts(end)))
    with pytest.raises(ValueError, match="review precedes milestone"):
        prepare(workspace)
    assert not (workspace / "release-review.json").exists()


@pytest.mark.parametrize("fault", ["missing-golden", "failed-golden", "failed-acceptance", "broken-acceptance"])
def test_declared_milestone_never_falls_back_to_legacy(workspace, monkeypatch, fault):
    monkeypatch.setattr(gate, "validate_offline_evidence",
                        lambda _: pytest.fail("must not fall back to exhaustive release"))
    if fault == "missing-golden":
        (workspace / "milestone-golden-replay.json").unlink()
    elif fault == "broken-acceptance":
        path = workspace / "milestone-acceptance.json"
        path.unlink()
        path.symlink_to("missing.json")
    else:
        name = "milestone-golden-replay.json" if fault == "failed-golden" else "milestone-acceptance.json"
        fixtures.rewrite(workspace, name, lambda r: r.update(status="failed"))
    with pytest.raises((ValueError, FileNotFoundError)):
        gate.validate_release_evidence(ev(workspace))


@pytest.mark.parametrize("fault", ["calibration", "stale", "runtime", "review-stage"])
def test_scoped_release_preserves_existing_live_safety_checks(workspace, fault):
    ref, runtime = approve(workspace)
    options = dict(expected_stage="live", runtime=runtime,
                   current_calibration_sha256=fixtures.digest("calibration"), now=fixtures.ts(33))
    gate.assert_live_release(ev(workspace), ref, **options)
    if fault == "calibration":
        options["current_calibration_sha256"] = fixtures.digest("changed")
    elif fault == "stale":
        options["now"] = fixtures.ts(100)
    elif fault == "runtime":
        options["runtime"] = fixtures.runtime(30, "restarted")
    else:
        options["expected_stage"] = "review"
    with pytest.raises(ValueError):
        gate.assert_live_release(ev(workspace), ref, **options)


def test_rejection_and_interrupted_review_do_not_approve(workspace):
    prepare(workspace)

    def interrupted(_):
        raise EOFError

    with pytest.raises(gate.PrerequisiteError):
        fixtures.cli().record_decision(workspace, "live", prompt=interrupted, test_only=True)
    assert not (workspace / "live-approval.json").exists()
    fixtures.review(workspace, "live", 25, ["Operator Fixture", "Reject this reference", "reject"])
    assert ev(workspace).json("live-approval.json")["decision"] == "rejected"
    with pytest.raises(ValueError, match="affirmative"):
        gate.validate_live_approval(ev(workspace))


def test_absent_milestone_retains_legacy_release_review_and_decision_scope(workspace, monkeypatch):
    (workspace / "milestone-acceptance.json").unlink()
    monkeypatch.setattr(milestone_acceptance, "validate_milestone_acceptance",
                        lambda _: pytest.fail("absent milestone must use legacy release"))
    calls = []
    monkeypatch.setattr(gate, "validate_offline_evidence", lambda _: calls.append("offline"))
    monkeypatch.setattr(gate, "_golden_replay", lambda _, name: calls.append(name))
    fixtures.save(workspace, "offline-report.json", {"archived_at": fixtures.ts(10)})
    fixtures.save(workspace, "golden-replay.json", {"ended_at": fixtures.ts(12)})
    for name in ("tolerance-agreement.json", "golden-candidate.json", "golden-approval.json"):
        fixtures.save(workspace, name, {"fixture": name})
    prepare(workspace)
    release = gate.validate_release_evidence(ev(workspace))
    assert release["report"]["path"] == "offline-report.json" and "acceptance_mode" not in release
    assert "offline" in calls and "golden-replay.json" in calls
    assert {ref["path"] for ref in gate.decision_evidence(ev(workspace), "live")} == {
        "session.json", "input-lock.json", "profiles.json", "release-review.json",
        "offline-report.json", "tolerance-agreement.json", "golden-candidate.json",
        "golden-approval.json", "golden-replay.json", "preflights/review-0001.json",
    }


@pytest.mark.parametrize("changed_source", [
    "policy_guard/milestone_acceptance.py", "policy_guard/milestone_golden.py",
    "scripts/replay_milestone_golden.py",
])
def test_runner_binds_scoped_sources_and_refuses_drift_before_construction(workspace, monkeypatch, changed_source):
    import test_checkpoint_sanity as safety

    runner = safety.runner()
    lock = ev(workspace).json("input-lock.json")
    monkeypatch.setattr(runner, "load_input_lock", lambda *_: lock)
    monkeypatch.setattr(runner, "controller_inputs", lambda _: {"controller": {"robot_port": "FAKE"}})
    monkeypatch.setenv("DUME_POLICY_BACKEND", "lerobot")
    actual = runner.RuntimeSource(SimpleNamespace(
        workspace=workspace, attestation=None, corpus=workspace, checkpoint=workspace,
        endpoint="127.0.0.1:8080", checkpoint_mount="/checkpoint",
    ))
    snapshot = actual.inputs()  # Local reads only; no attach/collect/connect.
    assert snapshot["source_files"][changed_source] == runner.sha256_file(runner.ROOT / changed_source)
    clock = safety.Clock(19)
    source = safety.Runtime(clock, fixtures.runtime(15))
    source.instance = source.value["attestation"]["instance"]
    source.inputs = actual.inputs
    review = runner.preflight(ev(workspace), stage="review", attempt=1, runtime_source=source, clock=clock)
    fixtures.cli().prepare_live(ev(workspace), review["path"], clock=clock)
    answers = iter(["Operator Fixture", "Approve scoped reference and trials", "approve"])
    fixtures.cli().record_decision(workspace, "live", prompt=lambda _: next(answers), clock=clock, test_only=True)
    live = runner.preflight(ev(workspace), stage="live", attempt=1, runtime_source=source, clock=clock)
    original_hash = runner.sha256_file
    monkeypatch.setattr(runner, "sha256_file", lambda path: (
        fixtures.digest("changed source") if path == runner.ROOT / changed_source else original_hash(path)
    ))
    constructed = []
    with pytest.raises(ValueError, match="controller/safety inputs changed"):
        runner.run(ev(workspace), preflight=live["path"], preflight_attempt=1,
                   runtime_source=source, clock=clock,
                   controller_factory=lambda **_: constructed.append(True))
    assert not constructed and not (workspace / "live-run.json").exists()


def test_trial1_only_approval_cannot_release_later_trials(workspace):
    prepare(workspace)
    answers = iter(["Session user", "Let's rerun trial 1", "approve"])
    fixtures.cli().record_decision(workspace, "live", prompt=lambda _: next(answers),
                                   clock=lambda: fixtures.ts(25), test_only=True, trial1_only=True)
    record = gate.validate_live_approval(ev(workspace))
    assert record["approval_scope"] == list(gate.MILESTONE_TRIAL1_SCOPE)
    for stage in ("trial-02", "trial-03"):
        with pytest.raises(ValueError, match="trial-1-only approval"):
            gate.assert_live_release(ev(workspace), {}, expected_stage=stage, runtime={},
                                     current_calibration_sha256="unused")
