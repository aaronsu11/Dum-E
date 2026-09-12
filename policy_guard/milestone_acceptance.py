"""Explicit physical-unit acceptance of the operator's small GPU milestone.

The strict numerical report stays failed and immutable. This policy records the
operator-authorized retrospective scope/criteria change; it is not a claim of
bitwise parity, a safety envelope, or permission to construct a robot.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from policy_guard import parity_gate as gate
from policy_guard import instrumentation_transition as transition
from policy_guard.replay_contract import now, write_evidence

ACCEPTANCE = "milestone-acceptance.json"
AUTHORIZATION = "milestone-criteria-authorization.json"
USER_INSTRUCTION = "This doesn't sound too bad given the different inference configuration. Let's relax our criteria to tolerate that and move on to physical test."
LIMITS = {
    "max_abs": [2., 2., 2., 2., 2., 1.],
    "mean_abs": [.5, .5, .5, .5, .5, .25],
    "bias": [.5, .5, .5, .5, .5, .25],
}
UNITS = ["degrees"] * 5 + ["gripper_normalized_points"]
ROOT = Path(__file__).resolve().parents[1]


def physical_metrics(delta, scales):
    delta = np.asarray(delta, dtype=np.float64)
    gate.require(delta.shape == (12, 16, 6) and np.isfinite(delta).all(), "finite twelve-case full action matrix required")
    scales = np.asarray(scales, dtype=np.float64)
    gate.require(scales.shape == (6,) and np.isfinite(scales).all() and (scales > 0).all(), "valid calibrated scales required")
    stats = gate.metrics(delta * scales)
    failures = [name for name, bounds in LIMITS.items() if np.any(np.abs(stats[name]) > bounds)]
    return {name: values.tolist() for name, values in stats.items()}, failures


def source_data(ev):
    report = ev.json("milestone-report.json")
    scope = ev.json(report["scope"])
    gate.require(report["kind"] == "milestone_gpu_12" and report["case_count"] == 12, "wrong milestone sample")
    gate.require(report["evidence_kind"] == "real_model_reduction", "real captured evidence required")
    gate.require(report["scope"] == ev.reference("milestone-scope.json"), "scope changed")
    gate.require(scope["input_lock"] == ev.reference("input-lock.json"), "source input lock changed")
    gate.require(report["agreement"] == ev.reference("tolerance-agreement.json"), "original agreement changed")
    gate.require(report["proofs"] == scope["sources"], "capture sources changed")
    gate.require(scope["authorization"]["selected_user_text"] == "12 observations, one seed", "small scope not recorded")
    lock = ev.json("input-lock.json")
    groups = {}
    for row in lock["records"]:
        groups.setdefault(row["episode_index"], []).append(row)
    expected = []
    for _, records in sorted(groups.items()):
        row = sorted(records, key=lambda item: (item["frame_index"], item["file"]))[len(records)//2]
        expected.append({"record": row["file"], "seed": row["seeds"][0]})
    gate.require(len(expected) == 12 and report["cases"] == expected, "selected sample changed")
    gate.require([{k: row[k] for k in ("record", "seed")} for row in scope["selected"]] == expected, "scope sample changed")
    gate.require(report["reducer_sha256"] == hashlib.sha256((ROOT / "scripts/check_milestone_parity.py").read_bytes()).hexdigest(), "reviewed reducer changed")
    # Only the release-validator integration may differ from the old instrument.
    # All actual inference/capture source remains identical to the original run.
    for name, digest in report["source_instrument"].items():
        if name != "policy_guard/parity_gate.py":
            transition.require_source(ev, name, digest)
    decoded, noise, profiles = [], [], []
    for name in ("native-operational", "lerobot-operational"):
        proof = report["proofs"][name]
        worker, launch = ev.json(proof["worker"]), ev.json(proof["launch"])
        gate.require(worker["status"] == launch["status"] == "complete" and launch["exit_code"] == 0, "source worker did not complete")
        gate.require(launch["manifest"] == proof["worker"], "launch/worker link changed")
        gate.require(worker["resources"]["instrument_files"] == launch["instrument_files"] == report["source_instrument"], "captured instrument changed")
        ev.bytes(launch["log"]["path"], launch["log"]["sha256"])
        profile = worker["profile"]
        gate.validate_profile(profile)
        gate.require(profile["purpose"] == "operational" and profile["device"].startswith("cuda") and profile["parameter_dtypes"] == ["torch.bfloat16"], "operational GPU BF16 required")
        profiles.append(profile)
        rows, noises = [], []
        for key in expected:
            case = worker["cases"][worker["executed_cases"].index(key)]
            saved = ev.json(case["evidence"])
            gate.require(all(saved.get(k) == v for k, v in case.items() if k != "evidence"), "captured case changed")
            gate.require(saved["status"] == "complete" and saved["key"] == key and saved["observer_inert"] is True, "invalid captured case")
            arrays = ev.tensors(case["tensors"])
            for field, shape in (("raw", (1,40,132)), ("noise", (1,40,132)), ("decoded", (16,6))):
                gate.require(arrays[field].shape == shape and np.isfinite(arrays[field]).all(), "invalid full " + field)
            rows.append(arrays["decoded"])
            noises.append(arrays["noise"])
        decoded.append(np.stack(rows)); noise.append(np.stack(noises))
    delta = gate.joint_deviations(*decoded)
    matrix = ev.tensors(report["matrix"])
    gate.require(np.array_equal(matrix["delta"], delta), "reported matrix differs from captured outputs")
    gate.require(report["metrics"] == {k: v.tolist() for k, v in gate.metrics(delta).items()}, "reported metrics changed")
    gate.require(report["noise"]["equal"] == bool(np.array_equal(*noise)), "noise report changed")
    calibration_sha, calibration_ref = gate._calibration(ev)
    calibration = ev.json("calibration.json")
    scales = [calibration["scale_deg_per_pct"][joint.removesuffix(".pos")] for joint in lock["joint_order"][:5]] + [1.]
    return report, delta, profiles, scales, calibration_sha, calibration_ref


def authorize(workspace, *, operator, user_instruction):
    ev = gate.evidence(workspace)
    gate.require(user_instruction == USER_INSTRUCTION and bool(operator.strip()), "actual operator instruction required")
    record = {"schema_version": 1, **ev.identity(), "operator": operator,
              "recorded_at": now(), "user_instruction": user_instruction,
              "basis": "Explicit retrospective acceptance change after reviewing measured differences; not a preregistered experiment",
              "strict_report": ev.reference("milestone-report.json"), "scope": ev.reference("milestone-scope.json"),
              "limits": LIMITS, "units": UNITS,
              "raw_and_preprocessing": "diagnostic only; preserve finite/full-shape validation and actual captured inputs",
              "safety_limits": False, "hardware_presence_confirmed": False,
              "golden_approval_granted": False}
    return write_evidence(ev.workspace, AUTHORIZATION, record)


def assess(workspace):
    ev = gate.evidence(workspace)
    authorization = ev.json(AUTHORIZATION)
    gate.require(authorization["limits"] == LIMITS and authorization["units"] == UNITS, "authorized limits changed")
    gate.require(authorization["user_instruction"] == USER_INSTRUCTION, "operator criteria instruction missing")
    gate.require(authorization["strict_report"] == ev.reference("milestone-report.json"), "authorized report changed")
    started = now()
    report, delta, profiles, scales, calibration, calibration_ref = source_data(ev)
    stats, failures = physical_metrics(delta, scales)
    result = {"schema_version": 1, **ev.identity(), "kind": "milestone_physical_units",
              "status": "complete" if not failures else "failed", "passed": not failures,
              "started_at": started, "ended_at": now(), "authorization": ev.reference(AUTHORIZATION),
              "strict_report": ev.reference("milestone-report.json"), "scope": ev.reference("milestone-scope.json"),
              "limits": LIMITS, "units": UNITS, "scale_per_normalized_point": scales,
              "metrics": stats, "failures": failures, "calibration_sha256": calibration,
              "calibration": calibration_ref, "case_count": 12,
              "policy_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "old_strict_result": report["status"], "release_ready": False,
              "caveats": ["Retrospective operator relaxation, not new evidence of strict parity",
                          "Different sampling precision, attention and preprocessing remain",
                          "Cartesian error and physical task success were not measured",
                          "This record does not confirm operator presence or authorize robot construction"]}
    return write_evidence(ev.workspace, ACCEPTANCE, result)


def validate_milestone_acceptance(workspace):
    ev = gate.evidence(workspace)
    accepted = ev.json(ACCEPTANCE)
    authorization = ev.json(accepted["authorization"])
    gate.require(accepted["authorization"] == ev.reference(AUTHORIZATION), "criteria authorization changed")
    gate.require(authorization["user_instruction"] == USER_INSTRUCTION and authorization["limits"] == LIMITS, "criteria authorization invalid")
    transition.require_source(ev, "policy_guard/milestone_acceptance.py", accepted["policy_sha256"])
    gate.require(accepted["strict_report"] == authorization["strict_report"] == ev.reference("milestone-report.json"), "accepted report changed")
    gate.require(accepted["scope"] == authorization["scope"] == ev.reference("milestone-scope.json"), "accepted sample changed")
    report, delta, profiles, scales, calibration, calibration_ref = source_data(ev)
    stats, failures = physical_metrics(delta, scales)
    gate.require(not failures and accepted["status"] == "complete" and accepted["passed"] is True, "milestone acceptance failed")
    gate.require(accepted["metrics"] == stats and accepted["limits"] == LIMITS and accepted["units"] == UNITS, "acceptance metrics/limits changed")
    gate.require(accepted["scale_per_normalized_point"] == scales and accepted["calibration_sha256"] == calibration, "accepted calibration changed")
    gate.require(gate.timestamp(report["ended_at"]) <= gate.timestamp(authorization["recorded_at"]) <= gate.timestamp(accepted["started_at"]) <= gate.timestamp(accepted["ended_at"]), "acceptance chronology invalid")
    sem = ev.json("profiles.json")["serving_configuration"]
    gate.validate_semantic_configuration(sem)
    gate.require(sem == gate.operational_semantics(transition.serving_profile(ev, profiles[1])),
                 "serving semantics differ from captured operational profile")
    return {**ev.identity(), "status": "complete", "report": ev.reference(ACCEPTANCE),
            "configuration_fingerprint": gate.fingerprint_configuration(sem),
            "calibration_sha256": calibration, "calibration": calibration_ref,
            "ended_at": accepted["ended_at"], **transition.release_fields(ev)}
