"""Synthetic evidence only; never an approval or a robot/model invocation."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from policy_guard.replay_contract import (
    CAMERA_ORDER, JOINT_ORDER, canonical, fingerprint_configuration as digest,
    read_json, repeatability_schedule, write_evidence, write_tensors,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def api():
    assert importlib.util.find_spec("policy_guard.parity_gate"), (
        "Evidence review must have an executable release validator before connect"
    )
    return importlib.import_module("policy_guard.parity_gate")


def cli():
    assert importlib.util.find_spec("approve_parity_evidence"), (
        "An explicit decision writer must drive the same release validator"
    )
    return importlib.import_module("approve_parity_evidence")


def ts(second):
    return (datetime(2026, 9, 11, tzinfo=timezone.utc) + timedelta(seconds=second)).isoformat()


def save(workspace, name, value):
    return write_evidence(workspace, name, value)


def rewrite(workspace, name, edit):
    """Deliberate corruption of test-only files, never a production writer."""
    path = workspace / name
    value = read_json(path)
    edit(value)
    path.write_bytes(canonical(value) + b"\n")


def observed(backend, purpose):
    return {
        "backend": backend, "purpose": purpose, "source": {"sha256": digest(backend)},
        "packages": {"torch": "test"}, "image_digest": "sha256:" + digest(backend),
        "checkpoint_fingerprint": digest("checkpoint"), "backbone_fingerprint": digest("backbone"),
        "instrumentation_fingerprint": digest("instrument"), "device": "cpu",
        "rng_algorithm": "torch.default_generator.cpu", "raw_shape": [1, 40, 132],
        "noise_shape": [1, 40, 132], "decoded_shape": [16, 6], "flow_steps": 4,
        "noise_draws": 1, "eval": True, "seed_at_sampling_boundary": True,
        "observer_inert": True, "joint_order": list(JOINT_ORDER), "camera_order": list(CAMERA_ORDER),
        "parameter_dtypes": ["torch.float32"], "buffer_dtypes": [], "input_dtypes": ["torch.float32"],
        "backbone_dtypes": ["torch.float32"], "compute_dtypes": ["torch.float32"],
        "noise_dtype": "torch.float32", "raw_dtype": "torch.float32",
        "attention": ["sdpa"], "autocast": False, "tf32": False, "sdpa_calls": 1,
    }


def semantics():
    return {
        "backend": "lerobot", "purpose": "operational",
        "checkpoint_fingerprint": digest("checkpoint"), "backbone_fingerprint": digest("backbone"),
        "source_fingerprint": digest("source"), "packages_fingerprint": digest("packages"),
        "image_digest": "sha256:" + digest("lerobot"),
        "effective_configuration": {"letter_box_transform": True, "actions_per_chunk": 16},
        "parameter_dtypes": ["torch.bfloat16"], "buffer_dtypes": [],
        "compute_dtypes": ["torch.bfloat16", "torch.float32"],
        "attention": ["sdpa"], "flow_steps": 4, "eval": True,
        "autocast": False, "tf32": False, "device": "cuda:0",
        "seed_policy": {"mode": "ambient", "seed": None},
        "joint_order": list(JOINT_ORDER), "camera_order": list(CAMERA_ORDER),
        "raw_shape": [1, 40, 132], "decoded_shape": [16, 6],
    }


def runtime(second, instance="first"):
    sem = semantics()
    process = {
        "container_id": digest(instance), "container_started_at": ts(0),
        "pid": 17, "process_started_at": ts(1), "process_start_ticks": 123,
        "boot_id": "test-boot", "load_id": digest(instance + "-load"),
    }
    request = {
        "observation_sha256": digest(f"observation-{second}"),
        "timestamp": float(second), "timestep": second,
        "started_at": ts(second), "completed_at": ts(second + 1),
        "output_sha256": digest(f"output-{second}"), "decoded_shape": [16, 6],
    }
    attestation = {
        "schema_version": 1, "evidence_kind": "test_only", "status": "complete",
        "semantic_configuration": sem, "configuration_fingerprint": digest(sem),
        "instance": process, "loaded_at": ts(2), "request": request,
        "endpoint": {"host": "0.0.0.0", "port": 8080},
    }
    host = {
        **process, "image_digest": sem["image_digest"], "running": True,
        "endpoint": "127.0.0.1:8080", "container_port": 8080,
        "checked_at": ts(second + 2),
    }
    return {"attestation": attestation, "host": host, "request": request}


def fixture_workspace(workspace):
    """Actual schema shape and full membership, all data explicitly test_only."""
    save(workspace, "session.json", {"schema_version": 1, "session_id": "test-session"})
    lock = {
        "schema_version": 1, "records": [
            {"file": f"record_{i:04d}.npz", "seeds": [1, 2, 3, 4, 5],
             "sha256": digest(i), "instruction": "banana"} for i in range(120)
        ], "joint_order": list(JOINT_ORDER), "camera_order": list(CAMERA_ORDER),
        "checkpoint_fingerprint": digest("checkpoint"),
    }
    lock["schedule"] = [{"record": r["file"], "seed": seed} for r in lock["records"] for seed in r["seeds"]]
    lock["fingerprint"] = digest(lock)
    save(workspace, "input-lock.json", lock)
    profiles = {
        "schema_version": 1, "session": "test-session",
        "profiles": [{"backend": b, "purpose": p, "observed": observed(b, p)}
                     for b, p in (("native", "diagnostic"), ("lerobot", "diagnostic"),
                                  ("native", "operational"), ("lerobot", "operational"))],
        "serving_configuration": semantics(),
    }
    save(workspace, "profiles.json", profiles)
    identity = {"schema_version": 1, "session": "test-session", "evidence_kind": "test_only",
                "input_fingerprint": lock["fingerprint"], "profiles_fingerprint": digest(profiles),
                "status": "complete", "started_at": ts(3), "ended_at": ts(4)}
    save(workspace, "calibration.json", {
        **identity, "calibration_sha256": digest("calibration"),
        "configuration_fingerprint": digest("robot-config"), "drift_errors": [],
    })
    schedule = repeatability_schedule(lock)
    keys = [key for group in schedule["groups"] for key in group["cases"]]
    values = np.zeros((len(keys), 16, 6), np.float32)
    noise = np.zeros((len(keys), 40, 132), np.float32)
    for i, key in enumerate(keys):
        if key["mode"] == "changed":
            values[i] = 1
            noise[i] = 1
    samples = write_tensors(workspace, {"decoded": values, "noise": noise})
    repeats = []
    for item in profiles["profiles"]:
        repeats.append({
            "profile": item["backend"] + "-" + item["purpose"], "cases": keys,
            "groups": [{"id": group["id"], "process_id": digest(item["backend"] + item["purpose"] + group["id"]),
                        "cases": group["cases"]} for group in schedule["groups"]],
            "tensors": samples, "started_at": ts(3), "ended_at": ts(4),
        })
    repeat_ref = save(workspace, "repeatability.json", {**identity, "schedule": schedule, "measurements": repeats})
    thresholds = {
        "preprocessing": {"atol": 0.0001, "rtol": 0.0001},
        "raw": {"atol": 0.001, "rtol": 0.001},
        "decoded": {name: [0.01] * 6 for name in ("max_abs", "mean_abs", "bias", "slope")},
    }
    pairs = {
        "diagnostic": ["native-diagnostic", "lerobot-diagnostic"],
        "native_bridge": ["native-diagnostic", "native-operational"],
        "lerobot_bridge": ["lerobot-diagnostic", "lerobot-operational"],
        "operational": ["native-operational", "lerobot-operational"],
    }
    proposal = {
        **identity, "started_at": ts(5), "ended_at": ts(6), "repeatability": repeat_ref,
        "harness": {"commit": "a" * 40, "producer_sha256": digest("producer"), "consumer_sha256": digest("consumer")},
        "comparisons": {name: {"profiles": pair, "thresholds": thresholds,
                                "noise_policy": "exact" if name == "diagnostic" else "independent",
                                "rationale": "Test-only measured basis"} for name, pair in pairs.items()},
        "golden": {"atol": 0.001, "rtol": 0.001, "rationale": "Test-only native repeatability"},
        "units": ["percent"] * 5 + ["gripper_percent"],
        "aggregation": "trace-and-aggregate-ols-0..15",
        "caveats": ["Test-only; not model evidence", "Training geometry remains an assumption"],
    }
    save(workspace, "tolerance-proposal.json", proposal)
    return identity, lock, pairs


def review(workspace, kind, second, answers=None, **kwargs):
    answers = iter(answers or ["Operator Fixture", "Intentional test-only review", "approve"])
    return cli().record_decision(
        workspace, kind, prompt=lambda _: next(answers), clock=lambda: ts(second),
        test_only=True, **kwargs,
    )


def complete_offline(workspace, identity, lock, pairs):
    """No real backend comparisons; all arrays and subjects originate here."""
    review(workspace, "tolerances", 10)
    zeros = write_tensors(workspace, {
        "decoded": np.zeros((600, 16, 6), np.float32),
        "raw": np.zeros((600, 40, 132), np.float32),
        "noise": np.zeros((600, 40, 132), np.float32),
    })
    pre = write_tensors(workspace, {
        "image_front": np.zeros((1, 3, 256, 256), np.float32),
        "image_wrist": np.zeros((1, 3, 256, 256), np.float32),
        "state": np.zeros((1, 6), np.float32),
        "tokens": np.ones((1, 3), np.int64), "mask": np.ones((1, 3), np.int64),
    })
    ev = api().Evidence(workspace, test_only=True)
    agreement = ev.reference("tolerance-agreement.json")
    comparisons = []
    for name, pair in pairs.items():
        comparisons.append({
            "name": name, "profiles": pair, "started_at": ts(11), "ended_at": ts(12),
            "cases": lock["schedule"], "joint_order": list(JOINT_ORDER),
            "left": zeros, "right": zeros,
            "preprocessing": [{"key": key, "left": pre, "right": pre} for key in lock["schedule"]],
            "common_inputs": [{"key": key, "left": pre, "right": pre} for key in lock["schedule"]],
            "metrics": {
                "max_abs": [0.] * 6, "mean_abs": [0.] * 6, "bias": [0.] * 6,
                "slope": [0.] * 6, "per_index_bias": [[0.] * 6] * 16,
                "trace_bias_max_abs": [0.] * 6, "trace_slope_max_abs": [0.] * 6,
            }, "passed": True,
        })
    upstream = {
        **identity, "started_at": ts(11), "ended_at": ts(12), "agreement": agreement,
        "harness": read_json(workspace / "tolerance-proposal.json")["harness"],
        "seed": 42, "tag": "new_embodiment", "producer_exit": 0, "consumer_exit": 0,
        "tests": [{"name": "new_embodiment", "outcome": "passed"}],
        "left": write_tensors(workspace, {"raw": np.zeros((2, 40, 132), np.float32)}),
        "right": write_tensors(workspace, {"raw": np.zeros((2, 40, 132), np.float32)}),
        "checkpoint_fingerprint": lock["checkpoint_fingerprint"],
    }
    up_ref = save(workspace, "upstream-result.json", upstream)
    save(workspace, "offline-report.json", {
        **identity, "started_at": ts(11), "ended_at": ts(13), "archived_at": ts(14),
        "agreement": agreement, "upstream": up_ref, "comparisons": comparisons,
        "caveats": ["Test-only arrays", "Training geometry caveat"], "parity_passed": True,
    })
    save(workspace, "golden-candidate.json", {
        **identity, "started_at": ts(11), "ended_at": ts(13),
        "profile": "native-operational", "cases": lock["schedule"], "tensors": zeros,
        "agreement": agreement, "previous": None, "reason": "Initial fixture candidate",
    })
    review(workspace, "golden", 15)
    ev = api().Evidence(workspace, test_only=True)
    save(workspace, "golden-replay.json", {
        **identity, "started_at": ts(16), "ended_at": ts(17), "profile": "native-operational",
        "cases": lock["schedule"], "tensors": zeros,
        "candidate": ev.reference("golden-candidate.json"), "approval": ev.reference("golden-approval.json"),
    })


def preflight(workspace, stage, second, *, attempt=1, previous=None, instance="first", reason=""):
    g = api()
    ev = g.Evidence(workspace, test_only=True)
    live = stage != "review"
    rv = runtime(second, instance)
    record = g.PreflightRecord(
        session="test-session", stage=stage, attempt=attempt, started_at=ts(second),
        ended_at=ts(second + 2), status="complete",
        configuration_fingerprint=digest(semantics()), calibration_sha256=digest("calibration"),
        attestation=rv["attestation"], attestation_sha256=digest(rv["attestation"]),
        host=rv["host"], request=rv["request"], previous=previous, reason=reason,
        approval=ev.reference("live-approval.json") if live else None,
        review_preflight=ev.json("release-review.json")["review_preflight"] if live else None,
        evidence_kind="test_only",
    )
    return g.write_preflight_record(ev, record), rv


def approved(workspace):
    identity, lock, pairs = fixture_workspace(workspace)
    complete_offline(workspace, identity, lock, pairs)
    review_ref, rv = preflight(workspace, "review", 20)
    cli().prepare_live(api().Evidence(workspace, test_only=True), review_ref["path"], clock=lambda: ts(24))
    review(workspace, "live", 25)
    live_ref, rv = preflight(workspace, "live", 30, previous=review_ref, reason="Approved live freshness")
    return live_ref, rv


def release(workspace, reference, rv, events):
    api().assert_live_release(
        api().Evidence(workspace, test_only=True), reference, expected_stage="live",
        runtime=rv, current_calibration_sha256=digest("calibration"), now=ts(33),
    )
    events.append("construct")
    events.append("connect")
    events.append("reset")


def test_review_approval_release_restart_lifecycle(tmp_path):
    live_ref, rv = approved(tmp_path)
    saved = {p: p.read_bytes() for p in (tmp_path / "preflights").iterdir()}
    events = []
    release(tmp_path, live_ref, rv, events)
    assert events == ["construct", "connect", "reset"]
    renewed, new_runtime = preflight(tmp_path, "live", 40, attempt=2, previous=live_ref,
                                     instance="second", reason="Same-profile server restart")
    api().assert_live_release(
        api().Evidence(tmp_path, test_only=True), renewed, expected_stage="live",
        runtime=new_runtime, current_calibration_sha256=digest("calibration"), now=ts(43),
    )
    with pytest.raises(ValueError):
        api().assert_live_release(api().Evidence(tmp_path, test_only=True), live_ref,
                                 expected_stage="live", runtime=new_runtime,
                                 current_calibration_sha256=digest("calibration"), now=ts(43))
    assert all(p.read_bytes() == data for p, data in saved.items())


@pytest.mark.parametrize("corruption", [
    "missing_approval", "rejected", "early", "changed_report", "changed_profile",
    "changed_input", "partial", "not_run", "missing_golden", "changed_tolerance",
    "wrong_subject", "late_tolerance", "changed_calibration",
])
def test_refusal_precedes_fake_construction_connect_and_reset(tmp_path, corruption):
    live_ref, rv = approved(tmp_path)
    mutations = {
        "rejected": ("live-approval.json", lambda d: d.update(decision="rejected")),
        "early": ("live-approval.json", lambda d: d.update(decided_at=ts(12))),
        "changed_report": ("offline-report.json", lambda d: d["caveats"].append("new caveat")),
        "changed_profile": ("profiles.json", lambda d: d["serving_configuration"].update(flow_steps=5)),
        "changed_input": ("input-lock.json", lambda d: d["records"][0].update(sha256=digest("other"))),
        "partial": ("offline-report.json", lambda d: d["comparisons"][0]["cases"].pop()),
        "not_run": ("offline-report.json", lambda d: d.update(status="not_run")),
        "changed_tolerance": ("tolerance-proposal.json", lambda d: d["golden"].update(atol=100)),
        "wrong_subject": ("live-approval.json", lambda d: d.update(subject_digest=digest("other"))),
        "late_tolerance": ("tolerance-agreement.json", lambda d: d.update(decided_at=ts(12))),
        "changed_calibration": ("calibration.json", lambda d: d.update(calibration_sha256=digest("new"))),
    }
    if corruption in ("missing_approval", "missing_golden"):
        (tmp_path / ("live-approval.json" if corruption == "missing_approval" else "golden-approval.json")).unlink()
    else:
        name, edit = mutations[corruption]
        rewrite(tmp_path, name, edit)
    events = []
    with pytest.raises((ValueError, FileNotFoundError, api().PrerequisiteError)):
        release(tmp_path, live_ref, rv, events)
    assert events == []


def test_prepare_live_is_review_readiness_and_never_motion_permission(tmp_path):
    identity, lock, pairs = fixture_workspace(tmp_path)
    complete_offline(tmp_path, identity, lock, pairs)
    ref, rv = preflight(tmp_path, "review", 20)
    bundle = cli().prepare_live(api().Evidence(tmp_path, test_only=True), ref["path"], clock=lambda: ts(24))
    assert read_json(tmp_path / bundle["path"])["review_preflight"] == ref
    assert not (tmp_path / "live-approval.json").exists()
    with pytest.raises((ValueError, FileNotFoundError, api().PrerequisiteError)):
        release(tmp_path, ref, rv, [])


@pytest.mark.parametrize("error", ["overwrite", "predecessor_hash", "predecessor_stage", "missing_reason", "review_stage"])
def test_preflight_paths_and_predecessors_are_exact(tmp_path, error):
    live_ref, rv = approved(tmp_path)
    with pytest.raises((ValueError, FileExistsError)):
        if error == "overwrite":
            preflight(tmp_path, "live", 40, previous=live_ref, reason="new")
        elif error == "review_stage":
            release(tmp_path, api().Evidence(tmp_path, test_only=True).reference("preflights/review-0001.json"), rv, [])
        else:
            previous = dict(live_ref)
            if error == "predecessor_hash":
                previous["sha256"] = digest("wrong")
            if error == "predecessor_stage":
                previous = api().Evidence(tmp_path, test_only=True).reference("preflights/review-0001.json")
            preflight(tmp_path, "live", 40, attempt=2, previous=previous,
                      reason="" if error == "missing_reason" else "restart")


def test_blank_input_reprompts_and_rejected_decision_is_preserved(tmp_path):
    fixture_workspace(tmp_path)
    ref = review(tmp_path, "tolerances", 10, ["", "Operator Fixture", "", "Reason", "", "reject"])
    decision = read_json(tmp_path / ref["path"])
    assert decision["decision"] == "rejected"
    assert (tmp_path / f"decisions/{digest(decision)}.json").read_bytes() == (tmp_path / ref["path"]).read_bytes()
    with pytest.raises(ValueError, match="affirmative"):
        api().validate_tolerance_agreement(api().Evidence(tmp_path, test_only=True))


def test_noninteractive_cli_never_writes_approval(tmp_path, monkeypatch):
    fixture_workspace(tmp_path)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert cli().main(["tolerances", "--workspace", str(tmp_path)]) == 2
    assert not (tmp_path / "tolerance-agreement.json").exists()


def test_live_review_displays_all_joints_trends_limits_and_caveats(tmp_path, capsys):
    approved(tmp_path)
    output = capsys.readouterr().out
    for text in (*JOINT_ORDER, "bias", "slope", "per_index_bias", "thresholds", "600", "caveat"):
        assert text in output


def test_fixture_evidence_cannot_be_used_by_production_validator(tmp_path):
    approved(tmp_path)
    with pytest.raises(ValueError, match="test_only|real_model"):
        api().validate_release_evidence(tmp_path)


def test_same_bytes_are_hashed_and_consumed(tmp_path, monkeypatch):
    approved(tmp_path)
    g = api()
    original = g.capture_bytes
    def replacing_read(path, *args, **kwargs):
        data = original(path, *args, **kwargs)
        if Path(path).name == "tolerance-proposal.json":
            rewrite(tmp_path, "tolerance-proposal.json", lambda d: d["golden"].update(atol=100))
        return data
    monkeypatch.setattr(g, "capture_bytes", replacing_read)
    ev = g.Evidence(tmp_path, test_only=True)
    agreement = g.validate_tolerance_agreement(ev)
    assert agreement["decision"] == "approved"
    assert ev.json("tolerance-proposal.json")["golden"]["atol"] == 0.001
    with pytest.raises(ValueError):
        g.validate_tolerance_agreement(g.Evidence(tmp_path, test_only=True))


def test_complete_status_does_not_hide_seed_ignoring_repeatability(tmp_path):
    fixture_workspace(tmp_path)
    zero = write_tensors(tmp_path, {"decoded": np.zeros((48, 16, 6), np.float32),
                                   "noise": np.zeros((48, 40, 132), np.float32)})
    rewrite(tmp_path, "repeatability.json", lambda d: d["measurements"][0].update(tensors=zero))
    # Rebind the proposal deliberately: even internally consistent digests cannot
    # turn a seed-ignoring measurement into a completed repeatability basis.
    ev = api().Evidence(tmp_path, test_only=True)
    rewrite(tmp_path, "tolerance-proposal.json", lambda d: d.update(repeatability=ev.reference("repeatability.json")))
    with pytest.raises(ValueError, match="seed"):
        review(tmp_path, "tolerances", 10)


def test_comparison_start_must_follow_agreement(tmp_path):
    identity, lock, pairs = fixture_workspace(tmp_path)
    complete_offline(tmp_path, identity, lock, pairs)
    rewrite(tmp_path, "offline-report.json", lambda d: d["comparisons"][0].update(started_at=ts(9)))
    with pytest.raises(ValueError, match="agreement|chronology"):
        api().validate_release_evidence(api().Evidence(tmp_path, test_only=True))


def test_true_pass_flag_cannot_hide_numerical_failure(tmp_path):
    identity, lock, pairs = fixture_workspace(tmp_path)
    complete_offline(tmp_path, identity, lock, pairs)
    bad = write_tensors(tmp_path, {"decoded": np.ones((600, 16, 6), np.float32),
                                  "raw": np.zeros((600, 40, 132), np.float32),
                                  "noise": np.zeros((600, 40, 132), np.float32)})
    rewrite(tmp_path, "offline-report.json", lambda d: d["comparisons"][0].update(right=bad))
    with pytest.raises(ValueError, match="numerical|metric|threshold"):
        api().validate_release_evidence(api().Evidence(tmp_path, test_only=True))


def closeout_fixture(workspace):
    live_ref, rv = approved(workspace)
    run_ref, _ = preflight(workspace, "run", 40, previous=live_ref, reason="Construction")
    previous = run_ref
    trials = []
    for i in range(1, 4):
        ref, _ = preflight(workspace, f"trial-{i:02d}", 40 + 10 * i, previous=previous, reason="Trial reset")
        trials.append({
            "index": i, "preflight": ref, "started_at": ts(43 + 10 * i), "ended_at": ts(48 + 10 * i),
            "iterations": 20, "actions_per_chunk": 16, "action_delay": 0.05,
            "coherent": i != 3, "wrong_target": False, "erratic": i == 3, "grasp": False,
            "clamp_warnings": 0, "safety_stop": False, "operator": "Fixture Operator",
        })
        previous = ref
    ev = api().Evidence(workspace, test_only=True)
    identity = {k: ev.json("offline-report.json")[k] for k in (
        "schema_version", "session", "evidence_kind", "input_fingerprint", "profiles_fingerprint", "status",
    )}
    save(workspace, "live-run.json", {
        **identity, "started_at": ts(43), "ended_at": ts(79), "constructed_at": ts(43),
        "live_preflight": live_ref, "run_preflight": run_ref, "trials": trials,
        "approval": ev.reference("live-approval.json"), "clamp_warnings": 0, "safety_stop": False,
        "instruction": "Grab a banana and put it on the plate",
    })
    final = copy.deepcopy(ev.json("golden-replay.json"))
    final.update(started_at=ts(80), ended_at=ts(90), live_run=api().Evidence(workspace, test_only=True).reference("live-run.json"))
    save(workspace, "final-regression.json", final)


@pytest.mark.parametrize("error", [None, "missing_final", "partial_final", "safety", "two_trials", "wrong_preflight", "wrong_hash", "early_trial"])
def test_closeout_requires_final_replay_and_exact_three_trial_links(tmp_path, error):
    closeout_fixture(tmp_path)
    if error == "missing_final":
        (tmp_path / "final-regression.json").unlink()
    elif error == "partial_final":
        rewrite(tmp_path, "final-regression.json", lambda d: d["cases"].pop())
    elif error:
        def edit(d):
            if error == "safety":
                d["safety_stop"] = True
            elif error == "two_trials":
                d["trials"].pop()
            elif error == "wrong_preflight":
                d["trials"][0]["preflight"] = d["run_preflight"]
            elif error == "wrong_hash":
                d["trials"][0]["preflight"]["sha256"] = digest("wrong")
            elif error == "early_trial":
                d["trials"][0]["started_at"] = ts(49)
        rewrite(tmp_path, "live-run.json", edit)
    ev = api().Evidence(tmp_path, test_only=True)
    if error:
        with pytest.raises((ValueError, FileNotFoundError, api().PrerequisiteError)):
            api().validate_closeout(ev)
        assert not (tmp_path / "closeout.json").exists()
    else:
        assert api().validate_closeout(ev)["status"] == "complete"
