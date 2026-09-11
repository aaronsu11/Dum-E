"""Operational native golden integrity and comparison; never creates decisions.

The canonical Plan03 decision schema remains authoritative. Fixture injection is
explicit and cannot be enabled from the command line. All arrays are read through
the bounded numeric-only, captured-byte evidence reader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from policy_guard.parity_gate import (
    Evidence, evidence, require, timestamp, validate_golden_approval,
    validate_golden_candidate, validate_tolerance_agreement,
)
from policy_guard.replay_contract import (
    fingerprint_configuration as digest, now, profile_configuration,
    validate_profile, validate_schedule, write_evidence,
)

NATIVE_PATH = "Gr00tPolicy.get_action -> upstream _get_action -> client joint mapping"
STAGE_FILES = {"pre-live": "golden-replay.json", "final": "final-regression.json"}


@dataclass(frozen=True)
class GoldenManifest:
    session: str
    input_fingerprint: str
    profiles_fingerprint: str
    evidence_kind: str
    started_at: str
    ended_at: str
    candidate: dict
    approval: dict
    previous: dict | None
    schema_version: int = 1
    status: str = "complete"
    profile: str = "native-operational"


def native_profile(ev):
    ev.identity()
    return next(row["observed"] for row in ev.json("profiles.json")["profiles"]
                if row["backend"] == "native" and row["purpose"] == "operational")


def validate_native_profile(profile):
    validate_profile(profile)
    require(profile["backend"] == "native" and profile["purpose"] == "operational",
            "golden requires native-operational, never diagnostic")
    require(profile.get("path") == NATIVE_PATH, "actual Gr00tPolicy.get_action path required")
    require(profile.get("parameter_dtypes") == ["torch.bfloat16"],
            "deployed native bf16 parameter profile required")
    require(profile.get("attention") == ["sdpa"] and profile.get("sdpa_calls", 0) > 0,
            "observed native SDPA required")
    require(bool(profile.get("effective_configuration")), "actual processor/model configuration required")


def consume_worker(ev, launch, expected_cases):
    """Validate durable operational case evidence and retain actual RNG provenance."""
    require(launch["status"] == "complete" and launch["exit_code"] == 0,
            "native worker failed or incomplete")
    report = ev.json(launch["manifest"])
    require(report["evidence_kind"] == ("test_only" if ev.test_only else "real_model"),
            "test_only worker cannot certify real_model replay")
    require(report["status"] == "complete" and not report["prerequisite_errors"] and
            not report.get("failure_ledger"), "native worker incomplete")
    identity = ev.identity()
    require(report["session"] == identity["session"] and report["stage"] == "operational" and
            report["input_fingerprint"] == identity["input_fingerprint"], "worker identity mismatch")
    require(report["expected_cases"] == expected_cases and report["executed_cases"] == expected_cases and
            len(report["cases"]) == len(expected_cases), "complete ordered worker coverage required")
    validate_schedule(ev.json("input-lock.json"), expected_cases, "replay")
    profile = report["profile"]
    validate_native_profile(profile)
    require(report["profile_fingerprint"] == digest(profile), "worker profile digest mismatch")
    require(report["configuration_fingerprint"] == digest(profile_configuration(profile)),
            "worker configuration digest mismatch")
    require(profile_configuration(profile) == profile_configuration(native_profile(ev)),
            "measured native profile changed; new repeatability/agreement required")
    require(timestamp(report["started_at"]) <= timestamp(report["ended_at"]), "worker chronology reversed")
    records = {row["file"]: row for row in ev.json("input-lock.json")["records"]}
    collected = {name: [] for name in ("raw", "noise", "decoded")}
    for key, row in zip(expected_cases, report["cases"], strict=True):
        require(row["key"] == key, "worker case order changed")
        record = records[key["record"]]
        require(row["record_sha256"] == record["sha256"] and row["instruction"] == record["instruction"],
                "record bytes or language identity changed")
        saved = ev.json(row["evidence"])
        require(all(saved.get(k) == v for k, v in row.items() if k != "evidence"),
                "durable case differs from worker manifest")
        for name, value in (("session", identity["session"]), ("stage", "operational"),
                            ("input_fingerprint", identity["input_fingerprint"]),
                            ("evidence_kind", report["evidence_kind"]), ("status", "complete")):
            require(saved.get(name) == value, f"durable case {name} mismatch")
        validate_native_profile(saved["profile"])
        require(row["profile_fingerprint"] == digest(saved["profile"]) and
                profile_configuration(saved["profile"]) == profile_configuration(profile) and
                row["observer_inert"] is True, "mixed or non-inert native observation")
        arrays = ev.tensors(row["tensors"])
        require(set(arrays) == {"raw", "noise", "decoded", "rng_before", "rng_after"},
                "actual raw/noise/decoded and RNG observation required")
        for name, shape in (("raw", (1, 40, 132)), ("noise", (1, 40, 132)), ("decoded", (16, 6))):
            require(arrays[name].shape == shape and arrays[name].dtype == np.float32,
                    f"full native {name} dimensions/dtype required")
            collected[name].append(arrays[name][0] if name != "decoded" else arrays[name])
        for name in ("rng_before", "rng_after"):
            require(arrays[name].ndim == 1 and arrays[name].size > 0 and arrays[name].dtype == np.uint8,
                    "observed RNG state bytes required")
        require(not np.array_equal(arrays["rng_before"], arrays["rng_after"]),
                "sampler did not advance observed RNG")
    return {name: np.stack(values) for name, values in collected.items()}


def validate_candidate(workspace):
    ev = evidence(workspace)
    if "native_candidate" in ev._validated:
        return ev._validated["native_candidate"]
    candidate = validate_golden_candidate(ev)
    arrays = consume_worker(ev, candidate["worker"], candidate["cases"])
    stored = ev.tensors(candidate["tensors"])
    require(all(np.array_equal(arrays[k], stored[k]) for k in arrays),
            "candidate tensors differ from actual worker")
    worker = ev.json(candidate["worker"]["manifest"])
    require(timestamp(candidate["started_at"]) <= timestamp(worker["started_at"]) <=
            timestamp(worker["ended_at"]) <= timestamp(candidate["ended_at"]), "candidate worker chronology")
    ev._validated["native_candidate"] = candidate
    return candidate


def approved_decision(ev):
    # A present decision with missing archive is tampered/incomplete, not absent approval.
    ev.json("golden-approval.json")
    try:
        return validate_golden_approval(ev)
    except FileNotFoundError as exc:
        raise ValueError("canonical approval archive missing or changed") from exc


def validate_golden(workspace, reference="golden-manifest.json"):
    ev = evidence(workspace)
    manifest = ev.record(reference)
    candidate = validate_candidate(ev)
    approval = approved_decision(ev)
    require(manifest["profile"] == "native-operational", "operational golden required")
    require(manifest["candidate"] == ev.reference("golden-candidate.json") and
            manifest["approval"] == ev.reference("golden-approval.json"), "golden manifest identity changed")
    require(manifest["previous"] == candidate.get("previous"), "golden predecessor changed")
    require(timestamp(approval["decided_at"]) <= timestamp(manifest["started_at"]),
            "promotion precedes approval")
    archive = f"goldens/{digest(manifest)}.json"
    require(ev.bytes(archive) == ev.bytes(reference), "immutable golden archive missing or changed")
    return manifest


def promote_reviewed_candidate(workspace, *, candidate_sha256, clock=now):
    ev = evidence(workspace)
    require(not (ev.workspace / "golden-manifest.json").exists(), "immutable approved manifest already exists")
    candidate = validate_candidate(ev)
    require(ev.reference("golden-candidate.json")["sha256"] == candidate_sha256,
            "exact candidate digest required")
    approval = approved_decision(ev)
    when = clock()
    require(timestamp(approval["decided_at"]) <= timestamp(when), "promotion precedes explicit decision")
    manifest = asdict(GoldenManifest(
        **ev.identity(), evidence_kind=candidate["evidence_kind"], started_at=when, ended_at=when,
        candidate=ev.reference("golden-candidate.json"), approval=ev.reference("golden-approval.json"),
        previous=candidate.get("previous"),
    ))
    write_evidence(ev.workspace, f"goldens/{digest(manifest)}.json", manifest)
    return write_evidence(ev.workspace, "golden-manifest.json", manifest)


def compare_native_replay(workspace, replay):
    """Narrow native agreement reducer; never imports the cross-backend reducer."""
    ev = evidence(workspace)
    validate_golden(ev)
    candidate = ev.json("golden-candidate.json")
    approval = approved_decision(ev)
    require(replay["profile"] == "native-operational", "operational replay required")
    require(replay["cases"] == candidate["cases"], "full ordered 600-case replay required")
    require(replay["candidate"] == ev.reference("golden-candidate.json") and
            replay["approval"] == ev.reference("golden-approval.json"), "replay reference changed")
    require(timestamp(approval["decided_at"]) < timestamp(replay["started_at"]),
            "fresh replay must follow approval")
    validate_tolerance_agreement(ev, comparison_started_at=replay["started_at"])
    left, right = ev.tensors(candidate["tensors"]), ev.tensors(replay["tensors"])
    bounds = ev.json("tolerance-proposal.json")["golden"]
    result = {}
    for name, shape in (("raw", (600, 40, 132)), ("noise", (600, 40, 132)), ("decoded", (600, 16, 6))):
        require(left[name].shape == right[name].shape == shape and
                left[name].dtype == right[name].dtype == np.float32, f"full {name} shape/dtype required")
        delta = right[name].astype(np.float64) - left[name].astype(np.float64)
        passed = np.array_equal(left[name], right[name]) if name == "noise" else np.all(
            np.abs(delta) <= bounds["atol"] + bounds["rtol"] * np.abs(left[name].astype(np.float64)))
        result[name] = {"passed": bool(passed), "max_abs": float(np.abs(delta).max())}
    return {"passed": all(row["passed"] for row in result.values()), "bounds": bounds, "tensors": result}
