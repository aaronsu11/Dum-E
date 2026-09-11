"""Operational native golden integrity and comparison; never creates decisions.

The canonical Plan03 decision schema remains authoritative. Fixture injection is
explicit and cannot be enabled from the command line. All arrays are read through
the bounded numeric-only, captured-byte evidence reader.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from policy_guard.parity_gate import (
    Evidence, _calibration, evidence, require, timestamp, validate_golden_approval,
    validate_golden_candidate, validate_tolerance_agreement,
)
from policy_guard.replay_contract import (
    _publish, contained, fingerprint_configuration as digest, now, profile_configuration,
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
    # Operational native is measured FlashAttention2 with SDPA in other layers.
    # Diagnostic SDPA is a separate contract; compare the full observed profile below.
    require(profile.get("attention") in (["sdpa"], ["flash_attention_2"]),
            "observed native attention implementation required")
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


def validate_candidate(workspace, *, _depth=0):
    ev = evidence(workspace)
    if "native_candidate" in ev._validated:
        return ev._validated["native_candidate"]
    candidate = validate_golden_candidate(ev)
    require(candidate["current_fingerprint"] == digest(candidate["current_inputs"]),
            "candidate current identity digest changed")
    if candidate.get("previous") is not None:
        validate_previous(ev, candidate["previous"], _depth=_depth)
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


def validate_golden(workspace, reference="golden-manifest.json", *, _depth=0):
    ev = evidence(workspace)
    manifest = ev.record(reference)
    candidate = validate_candidate(ev, _depth=_depth)
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
    context = evidence(workspace)
    # Promotion is a mutation boundary: never trust an earlier cached snapshot.
    ev = Evidence(context.workspace, test_only=context.test_only)
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
    observed = consume_worker(ev, replay["worker"], replay["cases"])
    right = ev.tensors(replay["tensors"])
    require(set(right) == set(observed) and all(np.array_equal(observed[k], right[k]) for k in observed),
            "replay tensors differ from actual worker output")
    worker = ev.json(replay["worker"]["manifest"])
    require(timestamp(replay["started_at"]) <= timestamp(worker["started_at"]) <=
            timestamp(worker["ended_at"]) <= timestamp(replay["ended_at"]), "replay worker chronology")
    require(replay["manifest"] == ev.reference("golden-manifest.json"), "replay promotion reference changed")
    for key, value in ev.identity().items():
        require(replay[key] == value, "replay identity mismatch")
    require(replay["evidence_kind"] == candidate["evidence_kind"], "replay evidence kind mismatch")
    require(replay["agreement"] == ev.reference("tolerance-agreement.json"), "replay agreement changed")
    require(replay["current_fingerprint"] == digest(replay["current_inputs"]), "replay fingerprint changed")
    result = compare_arrays(ev.tensors(candidate["tensors"]), right, ev.json("tolerance-proposal.json")["golden"])
    result["numerically_passed"] = result["passed"]
    result["identity_changed"] = replay["current_fingerprint"] != candidate["current_fingerprint"]
    result["passed"] = result["passed"] and not result["identity_changed"]
    return result


def compare_arrays(left, right, bounds):
    # The canonical schema currently specifies the same native atol/rtol for raw
    # and decoded output, measured from repeatability, with exact controlled noise.
    # Refuse additional rules rather than silently ignoring an agreed bound.
    require(set(bounds) <= {"atol", "rtol", "rationale", "noise_policy"}, "unsupported native agreement rule")
    require(bounds.get("noise_policy", "exact") == "exact", "native replay requires exact controlled noise")
    require(set(left) == set(right) == {"raw", "noise", "decoded"}, "complete native tensor archive required")
    result = {}
    for name, shape in (("raw", (600, 40, 132)), ("noise", (600, 40, 132)), ("decoded", (600, 16, 6))):
        require(left[name].shape == right[name].shape == shape and
                left[name].dtype == right[name].dtype == np.float32, f"full {name} shape/dtype required")
        require(np.isfinite(left[name]).all() and np.isfinite(right[name]).all(), "nonfinite native tensor")
        delta = right[name].astype(np.float64) - left[name].astype(np.float64)
        passed = np.array_equal(left[name], right[name]) if name == "noise" else np.all(
            np.abs(delta) <= bounds["atol"] + bounds["rtol"] * np.abs(left[name].astype(np.float64)))
        result[name] = {"passed": bool(passed), "max_abs": float(np.abs(delta).max())}
    return {"passed": all(row["passed"] for row in result.values()), "bounds": bounds, "tensors": result}


def calibration_identity(workspace):
    # Plan03 owns current byte/hash, session reference and offline arithmetic checks.
    digest_value, reference = _calibration(evidence(workspace))
    return {"sha256": digest_value, "reference": reference}


def validate_previous(ev, reference, *, _depth=0):
    require(_depth < 16, "excessive golden replacement history")
    receipt = ev.json(reference)
    manifest_ref = receipt["manifest"]
    prefix = f"history/{manifest_ref["sha256"]}/"
    require(manifest_ref["path"] == prefix + "golden-manifest.json", "exact retained predecessor required")
    ev.json(manifest_ref)
    previous = Evidence(contained(ev.workspace, prefix.rstrip("/")), test_only=ev.test_only)
    manifest = validate_golden(previous, _depth=_depth + 1)
    require(previous.reference("golden-manifest.json")["sha256"] == manifest_ref["sha256"], "old manifest changed")
    for name in ("candidate", "approval"):
        expected = {**manifest[name], "path": prefix + manifest[name]["path"]}
        require(receipt[name] == expected, "predecessor receipt changed")
        ev.json(receipt[name])
    # Retain the recursively captured bytes when importing a history chain.
    ev._bytes.update({prefix + path: data for path, data in previous._bytes.items()})
    return previous


def import_previous(workspace, path, expected_sha256):
    ev = evidence(workspace)
    path = Path(path)
    require(path.name == "golden-manifest.json", "explicit predecessor golden-manifest.json required")
    old = Evidence(path.parent, test_only=ev.test_only)
    reference = old.reference(path.name)
    require(reference["sha256"] == expected_sha256, "exact previous manifest digest required")
    manifest = validate_golden(old)
    require(old.workspace != ev.workspace and manifest["session"] != ev.identity()["session"],
            "replacement requires an explicit distinct successor session")
    prefix = f"history/{expected_sha256}/"
    # Copy only validated captured bytes, never glob, hardlink mutable inputs or rewrite references.
    for name, data in old._bytes.items():
        target = contained(ev.workspace, prefix + name)
        try:
            _publish(target, data)
        except FileExistsError:
            require(target.read_bytes() == data, "immutable predecessor snapshot conflict")
    receipt = {
        "manifest": {"path": prefix + "golden-manifest.json", "sha256": expected_sha256},
        "candidate": {**manifest["candidate"], "path": prefix + manifest["candidate"]["path"]},
        "approval": {**manifest["approval"], "path": prefix + manifest["approval"]["path"]},
        "source_workspace": str(old.workspace), "evidence_kind": manifest["evidence_kind"],
    }
    return write_evidence(ev.workspace, f"predecessors/{expected_sha256}.json", receipt)


def previous_digest(ev, candidate):
    if candidate.get("previous") is None:
        return None
    return ev.json(candidate["previous"])["manifest"]["sha256"]
