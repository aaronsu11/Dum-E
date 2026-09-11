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
    result = {
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
    if backend == "lerobot" and purpose == "operational":
        sem = semantics()
        result.update({key: value for key, value in sem.items() if key not in
                       ("source_fingerprint", "packages_fingerprint", "seed_policy")})
        result["serving_seed_policy"] = sem["seed_policy"]
        result["owned_source_files"] = {name: digest(name) for name in (
            "policy_guard/groot_guard.py", "policy_guard/replay_contract.py", "docker/lerobot-policy/server.py",
        )}
    result["owned_source_files"] = fixture_sources()
    result["owned_source_fingerprint"] = digest(result["owned_source_files"])
    return result


def semantics():
    return {
        "backend": "lerobot", "purpose": "operational",
        "checkpoint_fingerprint": digest("checkpoint"), "backbone_fingerprint": digest("backbone"),
        "source_fingerprint": digest({"source": {"sha256": digest("lerobot")}, "owned": {
            name: fixture_sources()[name] for name in (
                "policy_guard/groot_guard.py", "policy_guard/replay_contract.py", "docker/lerobot-policy/server.py",
            )}}), "packages_fingerprint": digest({"torch": "test"}),
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
        "observations": {
            "noise_shape": [1, 40, 132], "noise_dtype": "torch.bfloat16", "noise_device": "cuda:0",
            "noise_draws": 1, "sdpa_calls": 1, "kernels": ["aten._scaled_dot_product_attention"],
            "flow_steps": 4, "floating_operation_count": 1, "raw_shape": [1, 40, 132],
            "raw_dtype": "torch.float32", "input_dtypes": ["torch.float32"], "backbone_dtypes": ["torch.bfloat16"],
        },
    }
    host = {
        **process, "image_digest": sem["image_digest"], "running": True,
        "endpoint": "127.0.0.1:8080", "container_port": 8080,
        "checked_at": ts(second + 2),
    }
    return {"attestation": attestation, "host": host, "request": request}


def fixture_workspace(workspace, operational_edit=None):
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
    if operational_edit is not None:
        operational_edit(profiles["profiles"][3]["observed"])
    save(workspace, "profiles.json", profiles)
    identity = {"schema_version": 1, "session": "test-session", "evidence_kind": "test_only",
                "input_fingerprint": lock["fingerprint"], "profiles_fingerprint": digest(profiles),
                "status": "complete", "started_at": ts(3), "ended_at": ts(4)}
    save(workspace, "calibration.json", {
        **identity, "calibration_sha256": digest("calibration"),
        "configuration_fingerprint": digest("robot-config"), "drift_errors": [],
    })
    repeat, _ = fabricated_repeats(api().Evidence(workspace, test_only=True), identity)
    repeat_ref = save(workspace, "repeatability.json", repeat)
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
        "instrument_files": api().instrument_identity(),
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
    independent = {name: fabricated_worker(ev, name, lock["schedule"], "full-independent-" + name,
                                           second=11.1)[0] for name in api().PROFILE_NAMES}
    comparisons = []
    for name, pair in pairs.items():
        source = independent[pair[0]]["reference"]
        bundles, proofs = [], {}
        for side, profile in zip(("left", "right"), pair):
            common = fabricated_worker(ev, profile, lock["schedule"], name + "-" + side,
                                       second=12.1, common_from=source)[0]
            bundles.append(common)
            proofs[side] = {"independent": independent[profile]["reference"], "common": common["reference"]}
        record = api().compare_tiers(ev, name, *bundles, started_at=ts(12), ended_at=ts(12.5))
        record["provenance"] = proofs
        comparisons.append(record)
    up_ref = fabricated_stock(ev, identity)
    save(workspace, "offline-report.json", {
        **identity, "started_at": ts(11), "ended_at": ts(13), "archived_at": ts(14),
        "agreement": agreement, "upstream": up_ref, "comparisons": comparisons,
        "instrument_files": api().instrument_identity(),
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
    with pytest.raises(ValueError, match="seed|aggregate"):
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
    with pytest.raises(ValueError, match="numerical|metric|threshold|aggregate"):
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
    events, previous_event = [], None
    for index in range(3):
        event = {"index": index, "previous": previous_event, "kind": "returned",
                 "operation": "fabricated trial", "trial": index + 1}
        previous_event = digest(event)
        events.append({**event, "sha256": previous_event})
    save(workspace, "live-run.json", {
        **identity, "started_at": ts(43), "ended_at": ts(79), "constructed_at": ts(43),
        "live_preflight": live_ref, "run_preflight": run_ref, "trials": trials,
        "approval": ev.reference("live-approval.json"), "clamp_warnings": 0, "safety_stop": False,
        "events": events, "stop_events": [], "stop_reason": "",
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


def test_actual_plan02_calibration_schema_consumes_validated_bytes(tmp_path):
    """Exercise the production schema using local pinned inputs, without hardware."""
    import scripts.pose_sweep_units_probe as probe
    calibration_path = tmp_path / "calibration-input.json"
    statistics_path = tmp_path / "statistics.json"
    calibration_path.write_bytes(canonical(probe.CALIBRATION_TICK_RANGES))
    statistics_path.write_bytes(canonical({"new_embodiment": {
        "state": probe.CHECKPOINT_STATE_STATS, "action": probe.CHECKPOINT_ACTION_STATS,
    }}))
    record = probe.derive_current_calibration(calibration_path, statistics_path)
    record.update(session_id="test-session", started_at=ts(1), ended_at=ts(2))
    save(tmp_path, "session.json", {"schema_version": 1, "session_id": "test-session"})
    save(tmp_path, "calibration.json", record)
    assert api()._calibration(api().Evidence(tmp_path))[0] == record["calibration"]["sha256"]
    rewrite(tmp_path, "calibration.json", lambda d: d["scale_deg_per_pct"].update(wrist_roll=99))
    with pytest.raises(ValueError, match="scale"):
        api()._calibration(api().Evidence(tmp_path))


def serving_module():
    name = "parity_observation_test_server"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "docker/lerobot-policy/server.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def test_serving_observation_is_inert_and_measures_actual_compute():
    import torch
    from types import SimpleNamespace
    module = serving_module()
    assert hasattr(module, "ServingObservation"), "Serving attestation must observe real inference without reseeding it"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Identity()
            self.action_head = torch.nn.Module()
            self.action_head.action_encoder = torch.nn.Identity()

        def get_action(self):
            inputs = self.backbone(torch.ones((1, 1, 4, 4)))
            attention = torch.nn.functional.scaled_dot_product_attention(inputs, inputs, inputs)
            raw = torch.randn((1, 40, 132))
            for _ in range(4):
                raw = self.action_head.action_encoder(raw) + attention.mean()
            return raw

    model = Model().eval()
    torch.manual_seed(124)
    reference = model.get_action()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(124)
    with module.ServingObservation(model) as observed:
        result = model.get_action()
    assert torch.equal(reference, result)
    assert torch.equal(rng, torch.get_rng_state())
    assert observed.flow_steps == 4
    assert observed.noise_shape == [1, 40, 132]
    assert observed.noise_draws == 1
    assert observed.compute_dtypes == {"torch.float32"}
    assert observed.sdpa_calls == 1
    assert not model.backbone._forward_hooks
    assert not model.action_head.action_encoder._forward_pre_hooks


def test_serving_disabled_attestation_preserves_result_and_skips_fact_collection(monkeypatch):
    module = serving_module()
    assert hasattr(module.DumEGrootPolicyServer, "_predict_action_chunk_impl"), "Optional attestation must wrap the unchanged serving path"
    result = object()
    calls = []

    class Server(module.DumEGrootPolicyServer):
        def _predict_action_chunk_impl(self, observation):
            calls.append(observation)
            return result

    monkeypatch.delenv("DUME_PARITY_ATTESTATION_PATH", raising=False)
    server = Server.__new__(Server)
    assert server._predict_action_chunk("observation") is result
    assert calls == ["observation"]
    assert not hasattr(server, "_parity_attestor")


@pytest.mark.parametrize("mutation", [
    "pid", "container", "image", "checkpoint", "preprocessor", "seed", "incomplete",
    "endpoint", "request", "stale_request", "secret", "compute", "steps",
])
def test_runtime_attestation_rejects_stale_changed_or_incomplete_facts(mutation):
    g = api()
    rv = runtime(30)
    att = rv["attestation"]
    if mutation == "pid":
        rv["host"]["pid"] += 1
    elif mutation == "container":
        rv["host"]["container_id"] = digest("different")
    elif mutation == "image":
        rv["host"]["image_digest"] = "sha256:" + digest("different")
    elif mutation == "checkpoint":
        att["semantic_configuration"]["checkpoint_fingerprint"] = digest("different")
    elif mutation == "preprocessor":
        att["semantic_configuration"]["effective_configuration"]["letter_box_transform"] = False
    elif mutation == "seed":
        att["semantic_configuration"]["seed_policy"] = {"mode": "fixed", "seed": 1}
    elif mutation == "incomplete":
        att["status"] = "loaded"
    elif mutation == "endpoint":
        att["endpoint"]["port"] = 8081
    elif mutation == "request":
        rv["request"] = {**rv["request"], "observation_sha256": digest("different")}
    elif mutation == "stale_request":
        rv["host"]["checked_at"] = ts(200)
    elif mutation == "secret":
        att["environment"] = {"TOKEN": "must not serialize"}
    elif mutation == "compute":
        att["semantic_configuration"]["compute_dtypes"] = []
    elif mutation == "steps":
        att["semantic_configuration"]["flow_steps"] = 3
    with pytest.raises(ValueError):
        g.validate_runtime_attestation(att, host=rv["host"], request=rv["request"],
                                       expected_configuration=semantics(), now=ts(201 if mutation == "stale_request" else 33),
                                       test_only=True)


def test_host_container_identity_checks_actual_port_mapping_and_readonly_mount():
    g = api()
    assert hasattr(g, "container_binding"), "A tag or requested endpoint cannot stand in for inspected container facts"
    inspected = {
        "Id": digest("container"), "Image": "sha256:" + digest("image"),
        "State": {"Running": True, "StartedAt": ts(0)},
        "HostConfig": {"NetworkMode": "bridge"},
        "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8080"}]}},
        "Mounts": [{"Source": "/fixture/checkpoint", "Destination": "/checkpoints/model", "RW": False}],
    }
    binding = g.container_binding(inspected, "127.0.0.1:8080", "/checkpoints/model")
    assert binding["image_digest"] == inspected["Image"]
    for field in ("port", "mount", "network", "running"):
        changed = copy.deepcopy(inspected)
        if field == "port":
            changed["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostIp"] = "0.0.0.0"
        elif field == "mount":
            changed["Mounts"][0]["RW"] = True
        elif field == "network":
            changed["HostConfig"]["NetworkMode"] = "host"
        else:
            changed["State"]["Running"] = False
        with pytest.raises(ValueError):
            g.container_binding(changed, "127.0.0.1:8080", "/checkpoints/model")


@pytest.mark.parametrize("field,value", [
    ("device", "cpu"), ("parameter_dtypes", ["torch.float32"]),
    ("compute_dtypes", ["torch.float64"]), ("attention", ["flash_attention_2"]),
    ("source", {"sha256": "a" * 64}), ("packages", {"torch": "changed"}),
    ("effective_configuration", {"letter_box_transform": False}),
    ("serving_seed_policy", {"mode": "fixed", "seed": 999}),
])
def test_measured_operational_profile_mismatch_cannot_release(tmp_path, field, value):
    identity, lock, pairs = fixture_workspace(tmp_path, lambda p: p.update({field: value}))
    complete_offline(tmp_path, identity, lock, pairs)
    with pytest.raises(ValueError, match="operational|serving|measured"):
        api().validate_release_evidence(api().Evidence(tmp_path, test_only=True))


def test_old_preflight_cannot_be_revived_by_fresh_host_timestamp(tmp_path):
    live_ref, rv = approved(tmp_path)
    rv["host"]["checked_at"] = ts(200)
    with pytest.raises(ValueError):
        api().assert_live_release(api().Evidence(tmp_path, test_only=True), live_ref,
                                 expected_stage="live", runtime=rv,
                                 current_calibration_sha256=digest("calibration"), now=ts(201))


def test_complete_predecessor_requires_historical_semantic_validation(tmp_path):
    live_ref, rv = approved(tmp_path)
    rewrite(tmp_path, live_ref["path"], lambda d: d.update(
        host={}, request={}, approval=None, attestation={}, attestation_sha256=digest({}),
    ))
    malformed = api().Evidence(tmp_path, test_only=True).reference(live_ref["path"])
    events = []
    with pytest.raises((ValueError, KeyError)):
        run_ref, current = preflight(tmp_path, "run", 40, previous=malformed, reason="Construction")
        api().assert_live_release(api().Evidence(tmp_path, test_only=True), run_ref,
                                 expected_stage="run", runtime=current,
                                 current_calibration_sha256=digest("calibration"), now=ts(43))
        events.append("construct")
    assert events == []


def test_serving_attestor_publishes_only_completed_observed_request(tmp_path):
    import torch
    from types import SimpleNamespace
    from lerobot.async_inference.helpers import TimedAction, TimedObservation
    module = serving_module()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.backbone = torch.nn.Identity()
            self.action_head = torch.nn.Module()
            self.action_head.action_encoder = torch.nn.Identity()

        def get_action(self):
            inputs = self.backbone(torch.ones((1, 1, 4, 4)))
            attention = torch.nn.functional.scaled_dot_product_attention(inputs, inputs, inputs)
            raw = torch.randn((1, 40, 132))
            for _ in range(4):
                raw = self.action_head.action_encoder(raw) + attention.mean() + self.weight
            return {"action_pred": raw}

    class Server(module.DumEGrootPolicyServer):
        def _predict_action_chunk_impl(self, observation):
            raw = self.policy._groot_model.get_action()["action_pred"]
            return [TimedAction(timestamp=observation.get_timestamp(), timestep=i, action=row)
                    for i, row in enumerate(raw[0, :16, :6])]

    server = Server.__new__(Server)
    server.config = SimpleNamespace(host="0.0.0.0", port=8080)
    server.policy = torch.nn.Module()
    server.policy.config = SimpleNamespace(base_model_path="/fixture/checkpoint")
    server.policy._groot_model = Model().eval()
    identity = {"container": {"container_id": digest("first"), "container_started_at": ts(0),
                               "image_digest": "sha256:" + digest("lerobot")}}
    current_identity = copy.deepcopy(identity)
    def profile_reader(server, model, identity, measured, raw):
        profile = observed("lerobot", "operational")
        profile.update(parameter_dtypes=["torch.float32"], compute_dtypes=sorted(measured.compute_dtypes),
                       device=measured.noise_device, flow_steps=measured.flow_steps)
        return profile
    moments = iter([ts(2), ts(30), ts(31), ts(40)])
    attestor = module.ServingAttestor(
        server, tmp_path / "lerobot.json", identity,
        identity_reader=lambda *args: current_identity, profile_reader=profile_reader,
        clock=lambda: next(moments), process_reader=lambda: {
            k: v for k, v in runtime(30)["host"].items()
            if k in ("pid", "process_start_ticks", "process_started_at", "boot_id")
        }, test_only=True,
    )
    server._parity_attestor = attestor
    assert read_json(tmp_path / "lerobot.json")["status"] == "loaded"
    raw_observation = {key: 0.0 for key in JOINT_ORDER}
    raw_observation.update({key: np.zeros((480, 640, 3), np.uint8) for key in CAMERA_ORDER})
    raw_observation["task"] = "fixture banana"
    obs = TimedObservation(timestamp=30.0, timestep=0, observation=raw_observation, must_go=True)
    result = server._predict_action_chunk(obs)
    complete = read_json(tmp_path / "lerobot.json")
    assert complete["status"] == "complete"
    assert complete["request"]["observation_sha256"] == api().observation_fingerprint(raw_observation)
    assert complete["request"]["output_sha256"] == api().array_fingerprint(
        torch.stack([item.get_action().detach() for item in result]).numpy())
    assert complete["observations"]["flow_steps"] == 4
    assert complete["semantic_configuration"]["device"] == "cpu"
    assert (tmp_path / "lerobot.json").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "lerobot.json").stat().st_uid == tmp_path.stat().st_uid
    assert b"environment" not in (tmp_path / "lerobot.json").read_bytes()
    # No stale completion survives an identity change or failed inference.
    current_identity["container"]["image_digest"] = "sha256:" + digest("changed")
    with pytest.raises(ValueError, match="changed"):
        server._predict_action_chunk(obs)
    assert read_json(tmp_path / "lerobot.json")["status"] == "failed"
    assert "request" not in read_json(tmp_path / "lerobot.json")


def test_arbitrary_model_config_values_are_hashed_not_disclosed():
    profile = observed("lerobot", "operational")
    profile["effective_configuration"] = {
        "model": {"token": "fixture-private-value"},
        "policy": {"base_model_path": "/fixture/checkpoint"},
        "serving": {"base_model_path": "/fixture/checkpoint", "served_letter_box_transform": True},
    }
    sem = api().operational_semantics(profile)
    assert "fixture-private-value" not in canonical(sem).decode()
    assert sem["effective_configuration"]["serving"]["base_model_path"] == "checkpoint-sha256:" + digest("checkpoint")


def test_host_snapshot_change_requires_new_immutable_preflight_even_while_recent(tmp_path):
    live_ref, rv = approved(tmp_path)
    rv["host"]["checked_at"] = ts(33)
    with pytest.raises(ValueError, match="persisted preflight"):
        api().assert_live_release(api().Evidence(tmp_path, test_only=True), live_ref,
                                 expected_stage="live", runtime=rv,
                                 current_calibration_sha256=digest("calibration"), now=ts(34))


def test_release_recaptures_evidence_even_if_caller_reuses_a_snapshot(tmp_path):
    live_ref, rv = approved(tmp_path)
    cached = api().Evidence(tmp_path, test_only=True)
    api().validate_live_approval(cached)
    rewrite(tmp_path, "offline-report.json", lambda d: d["caveats"].append("changed after earlier validation"))
    with pytest.raises(ValueError):
        api().assert_live_release(cached, live_ref, expected_stage="live", runtime=rv,
                                 current_calibration_sha256=digest("calibration"), now=ts(33))


def test_failed_attempt_can_be_preserved_for_explicit_same_stage_renewal(tmp_path):
    live_ref, rv = approved(tmp_path)
    ev = api().Evidence(tmp_path, test_only=True)
    failed = copy.deepcopy(ev.json(live_ref))
    failed.update(attempt=2, started_at=ts(40), ended_at=ts(42), status="not_run",
                  previous=live_ref, reason="Fresh request unavailable", attestation={},
                  attestation_sha256=digest({}), host={}, request={})
    ref = api().write_preflight_record(ev, failed)
    saved = (tmp_path / ref["path"]).read_bytes()
    renewed, rv = preflight(tmp_path, "live", 50, attempt=3, previous=ref, reason="Explicit retry after unavailable request")
    api().assert_live_release(api().Evidence(tmp_path, test_only=True), renewed,
                             expected_stage="live", runtime=rv,
                             current_calibration_sha256=digest("calibration"), now=ts(53))
    assert (tmp_path / ref["path"]).read_bytes() == saved
    with pytest.raises(ValueError):
        preflight(tmp_path, "run", 60, previous=ref, reason="Cannot construct after failure")


def test_opt_in_identity_failure_drops_previous_policy_before_refusal(tmp_path, monkeypatch):
    module = serving_module()
    server = module.DumEGrootPolicyServer.__new__(module.DumEGrootPolicyServer)
    server.policy = server.preprocessor = server.postprocessor = object()
    monkeypatch.setenv("DUME_PARITY_ATTESTATION_PATH", str(tmp_path / "lerobot.json"))
    def refused(*args):
        raise ValueError("fixture content identity changed")
    monkeypatch.setattr(module, "capture_serving_identity", refused)
    with pytest.raises(ValueError, match="changed"):
        server._prepare_parity_attestation("/fixture/checkpoint")
    assert server.policy is server.preprocessor is server.postprocessor is None
    assert read_json(tmp_path / "lerobot.json")["status"] == "loading"


def configured_capture_server(checkpoint):
    """Real pinned configs/processors with a weight-free model collaborator."""
    import torch
    from types import SimpleNamespace
    module = serving_module()
    config = module.GrootConfig(base_model_path=str(checkpoint), embodiment_tag="new_embodiment", model_params_fp32=False)
    assert config.num_inference_timesteps is None
    module.fixup_policy_features(config, camera_keys=("wrist", "front"), height=480, width=640, state_dim=6, action_dim=6)
    pre, post = module.make_pre_post_processors(
        config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": "cpu"}, **module.serving_preprocessor_overrides()},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(1))
    attention = SimpleNamespace(_attn_implementation="sdpa")
    backbone_config = SimpleNamespace(_attn_implementation="sdpa", text_config=attention, vision_config=attention)
    model.backbone = SimpleNamespace(model=SimpleNamespace(config=backbone_config))
    model.action_head = SimpleNamespace(num_inference_timesteps=4)
    model.config = SimpleNamespace(to_dict=lambda: {"num_inference_timesteps": 4, "_name_or_path": str(checkpoint)})
    model.eval()
    server = SimpleNamespace(policy=SimpleNamespace(config=config), preprocessor=pre, postprocessor=post, actions_per_chunk=16)
    measured = SimpleNamespace(flow_steps=4, noise_draws=1, noise_shape=[1, 40, 132], sdpa_calls=1,
                               floating_operation_count=1, compute_dtypes={"torch.float32"},
                               autocast=False, tf32=False, noise_device="cpu")
    return server, model, measured


def test_serving_profile_reads_effective_steps_when_request_uses_checkpoint_default():
    import torch
    module = serving_module()
    server, model, measured = configured_capture_server(ROOT / "checkpoints/GR00T-N1.7-3B-SO101")
    identity = observed("lerobot", "operational")
    identity["container"] = {"image_digest": identity["image_digest"]}
    try:
        profile = module.serving_profile(server, model, identity, measured, torch.zeros(1, 40, 132))
    except ValueError as exc:
        pytest.fail(f"Observed four-step inference must accept the pinned None request default: {exc}")
    assert profile["flow_steps"] == 4
    assert profile["parameter_dtypes"] == ["torch.float32"]
    assert profile["effective_configuration"]["processors"]["pre"]
    assert api().operational_semantics(profile)["effective_configuration"]["processors_sha256"]


@pytest.mark.parametrize("mutation", ["preprocessor_config", "postprocessor_order"])
def test_independent_replay_and_serving_capture_match_and_detect_processor_changes(tmp_path, monkeypatch, mutation):
    import torch
    from policy_guard.replay_contract import configuration_value

    monkeypatch.delenv("DUME_POLICY_SEED", raising=False)
    spec = importlib.util.spec_from_file_location("parity_capture_test_replay", ROOT / "docker/lerobot-policy/replay_checkpoint.py")
    replay_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay_module)
    mounts = [tmp_path / name for name in ("replay-checkpoint", "serving-checkpoint")]
    for mount in mounts:
        mount.symlink_to(ROOT / "checkpoints/GR00T-N1.7-3B-SO101", target_is_directory=True)
    replay_server, replay_model, _ = configured_capture_server(mounts[0])
    serving_server, serving_model, measured = configured_capture_server(mounts[1])
    assert replay_server.preprocessor is not serving_server.preprocessor
    assert replay_server.postprocessor is not serving_server.postprocessor
    assert replay_server.policy.config.base_model_path != serving_server.policy.config.base_model_path

    # Invoke the actual adapter capture without constructing/loading a real policy.
    adapter = replay_module.LeRobotReplay.__new__(replay_module.LeRobotReplay)
    adapter.server, adapter.raw_model = replay_server, replay_model

    def replay_semantics():
        profile = observed("lerobot", "operational")
        profile.update(
            effective_configuration=configuration_value(adapter.effective_configuration()),
            parameter_dtypes=sorted({str(p.dtype) for p in replay_model.parameters()}),
            buffer_dtypes=sorted({str(b.dtype) for b in replay_model.buffers()}),
            compute_dtypes=["torch.float32"], attention=sorted(adapter.attention_implementations()),
            device="cpu", serving_seed_policy={"mode": "ambient", "seed": None},
        )
        return api().operational_semantics(profile)

    identity = observed("lerobot", "operational")
    identity["container"] = {"image_digest": identity["image_digest"]}
    served = serving_module().serving_profile(
        serving_server, serving_model, identity, measured, torch.zeros(1, 40, 132))
    served_semantics = api().operational_semantics(served)
    assert replay_semantics() == served_semantics, (
        "Independent replay and serving capture must include the same effective processor configuration"
    )
    assert served_semantics["effective_configuration"]["processors_sha256"]

    if mutation == "preprocessor_config":
        pack = next(step for step in replay_server.preprocessor.steps if hasattr(step, "state_dropout_prob"))
        pack.state_dropout_prob = 0.125
    else:
        assert len(replay_server.postprocessor.steps) > 1
        replay_server.postprocessor.steps = list(reversed(replay_server.postprocessor.steps))
    assert replay_semantics() != served_semantics
    unchanged = serving_module().serving_profile(
        serving_server, serving_model, identity, measured, torch.zeros(1, 40, 132))
    assert api().operational_semantics(unchanged) == served_semantics


@pytest.mark.parametrize("field", ["workers", "statistics", "raw_max_abs", "preprocessing_max_abs", "instrument_files"])
def test_review_repeatability_rejects_unverified_measurement_basis(tmp_path, field):
    fixture_workspace(tmp_path)
    def corrupt(record):
        if field == "instrument_files":
            record[field] = {"policy_guard/parity_gate.py": "0" * 64}
        elif field == "workers":
            record["measurements"][0][field] = []
        elif field == "statistics":
            record["measurements"][0][field]["max_abs"] = [1000.] * 6
        else:
            record["measurements"][0][field] = 1000.
    rewrite(tmp_path, "repeatability.json", corrupt)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        api().validate_repeatability(api().Evidence(tmp_path, test_only=True))


@pytest.mark.parametrize("field", ["matrix", "independent_collated", "common_collated", "provenance",
                                  "junit", "coverage", "producer_observation", "launches"])
def test_review_release_requires_complete_numerical_and_stock_witnesses(tmp_path, field):
    identity, lock, pairs = fixture_workspace(tmp_path)
    complete_offline(tmp_path, identity, lock, pairs)
    api().validate_release_evidence(api().Evidence(tmp_path, test_only=True))
    if field in ("matrix", "independent_collated", "common_collated", "provenance"):
        rewrite(tmp_path, "offline-report.json", lambda r: r["comparisons"][0].pop(field, None))
    else:
        rewrite(tmp_path, "upstream-result.json", lambda r: r.pop(field, None))
        ref = api().Evidence(tmp_path, test_only=True).reference("upstream-result.json")
        rewrite(tmp_path, "offline-report.json", lambda r: r.update(upstream=ref))
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        api().validate_release_evidence(api().Evidence(tmp_path, test_only=True))


def fabricated_save(workspace, name, value):
    """Write fabricated temporary files without exercising publication fsync."""
    import hashlib
    path = workspace / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}


def fixture_sources():
    current = api().instrument_identity()
    return {name: current[name] for name in api().WORKER_SOURCE_FILES}


def fabricated_worker(ev, name, cases, label, *, second, common_from=None, offset=0):
    """Full immutable schema, deliberately test_only; never invokes a process."""
    from policy_guard.replay_contract import profile_configuration
    g = api()
    workspace = ev.workspace
    profile = copy.deepcopy(next(p["observed"] for p in ev.json("profiles.json")["profiles"]
                                 if p["backend"] + "-" + p["purpose"] == name))
    backend, purpose = name.split("-")
    kind = "repeatability" if "mode" in cases[0] else "replay"
    clock = ts(second)
    plan = fabricated_save(workspace, f"schedules/{label}.json", {
        "schema_version": 1, **ev.identity(), "kind": kind, "cases": cases, "common_from": common_from,
        "execution": {"id": digest(label)[:32], "collection": "numerical-" + kind,
                      "started_at": clock, "worker_manifest": f"workers/{label}.json"},
    })
    execution = {**ev.json(plan)["execution"], "schedule": plan}
    pre = {
        "image_front": np.zeros((1, 3, 2, 2), np.float32),
        "image_wrist": np.ones((1, 3, 2, 2), np.float32),
        "state": np.zeros((1, 6), np.float32),
        "tokens": np.ones((1, 4), np.int64), "mask": np.ones((1, 4), np.int64),
    }
    pre_ref = write_tensors(workspace, pre)
    collated = [{ "tensors": pre_ref, "dtypes": {k: "torch." + str(v.dtype) for k, v in pre.items()}}] * len(cases)
    if common_from is not None:
        collated = ev.json(common_from)["collated"]
    rows, samples, sample_refs = [], [], {}
    profile_hash = digest(profile)
    lock = ev.json("input-lock.json")
    records = {r["file"]: r for r in lock["records"]}
    for index, key in enumerate(cases):
        changed = key.get("mode") == "changed"
        code = (changed, key["seed"] if kind == "repeatability" else 0)
        if code not in sample_refs:
            arrays = {
                "raw": np.full((1, 40, 132), offset + int(changed), np.float32),
                "noise": np.full((1, 40, 132), code[1], np.float32),
                "decoded": np.full((16, 6), offset + int(changed), np.float32),
                **{"preprocessing." + k: v for k, v in pre.items()},
                **{"collated." + k: v for k, v in pre.items()},
                **{"dtype." + k: np.array(g.DTYPE_NAMES.index("torch." + str(v.dtype)), np.int8)
                   for k, v in pre.items()},
            }
            sample_refs[code] = (write_tensors(workspace, arrays), arrays)
        ref, arrays = sample_refs[code]
        samples.append(arrays)
        record = records[key["record"]]
        row = {"key": key, "record_sha256": record["sha256"], "instruction": record["instruction"],
               "tensors": ref, "observer_inert": True, "profile_fingerprint": profile_hash,
               "execution": execution, "started_at": clock, "ended_at": clock}
        durable = fabricated_save(workspace, f"workers/cases/{label}-{index:04d}.json", {
            **row, "schema_version": 1, "session": ev.identity()["session"], "stage": purpose,
            "input_fingerprint": lock["fingerprint"], "evidence_kind": "test_only",
            "profile": profile, "status": "complete", "error": None,
        })
        rows.append({**row, "evidence": durable})
    pid = int(digest(label)[:12], 16) + 1
    worker = {
        "schema_version": 1, "session": ev.identity()["session"], "stage": purpose,
        "input_fingerprint": lock["fingerprint"], "evidence_kind": "test_only", "status": "complete",
        "started_at": clock, "ended_at": clock, "expected_cases": cases, "executed_cases": cases,
        "cases": rows, "profile": profile, "profile_fingerprint": profile_hash, "execution": execution,
        "configuration_fingerprint": digest(profile_configuration(profile)),
        "resources": {"process_identity": {"pid": pid, "process_start_ticks": pid,
                       "process_started_at": clock, "boot_id": "fabricated-boot"},
                      "instrument_files": g.instrument_identity()},
        "prerequisite_errors": [],
    }
    worker_ref = fabricated_save(workspace, f"workers/{label}.json", worker)
    log_path = workspace / f"workers/{label}.log"
    log_path.write_text("fabricated fixture; no process ran\n")
    launch = {
        "backend": backend, "purpose": purpose, "manifest": worker_ref, "status": "complete", "exit_code": 0,
        "started_at": clock, "ended_at": clock, "schedule": plan,
        "instrument_files": g.instrument_identity(), "log": ev.reference(str(log_path.relative_to(workspace))),
        "argv": ["docker", "run", "--entrypoint", "python", profile["image_digest"],
                 "/replay/scripts/replay_checkpoint_parity.py", "_worker", "--backend", backend,
                 "--profile", purpose, "--device", profile["device"], "--image-digest", profile["image_digest"],
                 "--checkpoint", "/inputs/checkpoint", "--schedule", "/evidence/" + plan["path"],
                 "--output-manifest", worker_ref["path"]],
    }
    launch_ref = fabricated_save(workspace, f"launches/{label}.json", launch)
    bundle = {
        "cases": cases, "profile": name, "input_fingerprint": lock["fingerprint"],
        "joint_order": list(JOINT_ORDER), "camera_order": list(CAMERA_ORDER),
        "tensors": write_tensors(workspace, {k: np.stack([r[k][0] if k != "decoded" else r[k] for r in samples])
                                            for k in ("raw", "noise", "decoded")}),
        "preprocessing": [pre_ref] * len(cases), "common_inputs": [pre_ref] * len(cases),
        "independent_collated": [item["tensors"] for item in collated],
        "common_collated": [item["tensors"] for item in collated],
        "collated": collated, "worker": worker_ref, "launch": launch_ref,
    }
    bundle["reference"] = fabricated_save(workspace, f"bundles/{label}.json", bundle)
    return bundle, {**launch, "reference": launch_ref}


def fabricated_repeats(ev, identity, *, prefix="repeat", offsets=False):
    """The between-profile offset proves reductions never pair backends."""
    g = api()
    schedule = repeatability_schedule(ev.json("input-lock.json"))
    measured, launches = [], []
    for index, name in enumerate(g.PROFILE_NAMES):
        workers, refs, groups, samples = [], [], [], []
        for count, group in enumerate(schedule["groups"]):
            bundle, launch = fabricated_worker(ev, name, group["cases"], f"{prefix}-{name}-{count}",
                                               second=3.5, offset=index * 1000 if offsets else 0)
            launches.append(launch)
            workers.append(bundle["worker"])
            refs.append(bundle["launch"])
            worker = ev.json(bundle["worker"])
            groups.append({"id": group["id"], "cases": group["cases"],
                           "process_id": digest(worker["resources"]["process_identity"])})
            samples.append(ev.tensors(bundle["tensors"]))
        measured.append({
            "profile": name, "cases": [k for group in schedule["groups"] for k in group["cases"]],
            "groups": groups, "workers": workers, "launches": refs,
            "started_at": ts(3.5), "ended_at": ts(3.5),
            "tensors": write_tensors(ev.workspace, {k: np.concatenate([s[k] for s in samples])
                                                   for k in ("decoded", "noise")}),
            "statistics": {"max_abs": [0.] * 6, "mean_abs": [0.] * 6, "bias": [0.] * 6,
                           "slope": [0.] * 6, "per_index_bias": [[0.] * 6] * 16,
                           "trace_bias_max_abs": [0.] * 6, "trace_slope_max_abs": [0.] * 6},
            "raw_max_abs": 0., "preprocessing_max_abs": 0.,
        })
    record = {**identity, "schedule": schedule, "measurements": measured, "instrument_files": g.instrument_identity()}
    return record, {"status": "complete", "started_at": ts(3), "ended_at": ts(4), "workers": launches}


def fabricated_stock(ev, identity):
    g = api()
    workspace = ev.workspace
    result = {
        **identity, "started_at": ts(10.1), "ended_at": ts(10.9),
        "agreement": ev.reference("tolerance-agreement.json"),
        "harness": ev.json("tolerance-proposal.json")["harness"], "seed": 42, "tag": "new_embodiment",
        "checkpoint_fingerprint": ev.json("input-lock.json")["checkpoint_fingerprint"],
        "producer_exit": 0, "consumer_exit": 0, "tests": [{"name": "new_embodiment", "outcome": "passed"}],
        "launches": [],
    }
    raw = write_tensors(workspace, {"raw": np.zeros((2, 40, 132), np.float32)})
    noise = write_tensors(workspace, {"noise": np.zeros((2, 40, 132), np.float32)})
    inputs = write_tensors(workspace, {"state": np.zeros((2, 6), np.float32)})
    bounds = ev.json("tolerance-proposal.json")["comparisons"]["diagnostic"]["thresholds"]["raw"]
    for stage, backend, second in (("producer", "native", 10.2), ("consumer", "lerobot", 10.6)):
        profile = next(p["observed"] for p in ev.json("profiles.json")["profiles"]
                       if p["backend"] == backend and p["purpose"] == "diagnostic")
        observed = {**profile, "raw_shape": [2, 40, 132], "noise_shape": [2, 40, 132]}
        result[stage + "_observation"] = fabricated_save(workspace, f"stock/{stage}-observation.json", {
            **profile, "status": "complete", "evidence_kind": "test_only", "observed": observed,
            "raw": raw, "noise": noise, "inputs": inputs,
        })
        path = workspace / f"stock/{stage}.log"
        path.write_text("Dumped 1 tags: ['new_embodiment']\n" if stage == "producer" else "1 passed\n")
        result[stage + "_log"] = ev.reference(str(path.relative_to(workspace)))
        env = {
            "GROOT_N1_7_PARITY_DIR": "/evidence/stock/producer", "GROOT_N1_7_LIBERO_CKPT": "/inputs/checkpoint",
            "GROOT_PARITY_DEVICE": profile["device"], "GROOT_PARITY_ATOL": str(bounds["atol"]),
            "GROOT_PARITY_RTOL": str(bounds["rtol"]), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        }
        argv = ["docker", "run", *[v for k, value in env.items() for v in ("--env", k + "=" + value)],
                "--entrypoint", "python", profile["image_digest"], "--stage", stage,
                "--checkpoint", "/inputs/checkpoint"]
        result["launches"].append({"stage": stage, "argv": argv, "exit_code": 0,
                                   "started_at": ts(second), "ended_at": ts(second + .1)})
    node = g.CONSUMER + "::" + g.CASE
    result["coverage"] = fabricated_save(workspace, "stock/coverage.json", {
        "collected": [node], "reports": [{"nodeid": node, "when": phase, "outcome": "passed", "wasxfail": None}
                                        for phase in ("setup", "call", "teardown")],
    })
    (workspace / "stock/junit.xml").write_text(
        f'<testsuites><testsuite tests="1"><testcase name="{g.CASE}"/></testsuite></testsuites>')
    result["junit"] = ev.reference("stock/junit.xml")
    (workspace / "stock/artifact.npz").write_bytes(b"explicit fabricated producer output")
    result["artifact"] = ev.reference("stock/artifact.npz")
    result["left"] = result["right"] = raw
    return fabricated_save(workspace, "upstream-result.json", result)


@pytest.mark.parametrize("fault", ["execution", "case_time"])
def test_review_numerical_worker_execution_cannot_be_relabelled(tmp_path, fault):
    fixture_workspace(tmp_path)
    g = api()
    ev = g.Evidence(tmp_path, test_only=True)
    measured = ev.json("repeatability.json")["measurements"][0]
    worker_ref, launch_ref = measured["workers"][0], measured["launches"][0]
    worker = ev.json(worker_ref)
    if fault == "execution":
        worker["execution"] = {"id": "wrong execution"}
    else:
        case = worker["cases"][0]
        case["started_at"] = ts(99)
        rewrite(tmp_path, case["evidence"]["path"], lambda row: row.update(started_at=ts(99)))
        case["evidence"] = g.Evidence(tmp_path, test_only=True).reference(case["evidence"]["path"])
    rewrite(tmp_path, worker_ref["path"], lambda row: (row.clear(), row.update(worker)))
    worker_ref = g.Evidence(tmp_path, test_only=True).reference(worker_ref["path"])
    rewrite(tmp_path, launch_ref["path"], lambda launch: launch.update(manifest=worker_ref))
    launch_ref = g.Evidence(tmp_path, test_only=True).reference(launch_ref["path"])
    with pytest.raises(ValueError, match="execution|chronology"):
        g.validate_numerical_worker(
            g.Evidence(tmp_path, test_only=True), {"worker": worker_ref, "launch": launch_ref},
            measured["profile"], worker["expected_cases"], kind="repeatability", start=ts(3), end=ts(4),
        )
