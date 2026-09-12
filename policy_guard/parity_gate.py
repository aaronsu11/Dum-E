"""Local evidence gates. No robot imports, model loading, or operator decisions.

Public consumers accept either a workspace Path or an Evidence snapshot. A fresh
snapshot is required for each physical release. ``test_only=True`` is an explicit
injection seam for hermetic consumers, never a CLI option.

Stage schema (v1), shared with the future numerical/golden/runner producers:
* All new measured stage records bind session, input_fingerprint and
  profiles_fingerprint (canonical digest of profiles.json), with UTC start/end.
* profiles.json retains Plan 01's list of observed profiles; Plan 04 adds the
  measured ``serving_configuration`` from the arm-free attestation.
* repeatability.json adds the exact repeatability_schedule and four measurements.
  Each binds immutable worker/launch references, ordered durable case captures,
  actual process identities, current instrument hashes and decoded/noise tensors.
  All statistics, raw maxima and preprocessing maxima are independently recomputed.
* tolerance-proposal.json defines COMPARISON_PROFILES, thresholds, noise policies,
  six units, trace/aggregate OLS definitions, harness pins and measured basis.
* offline-report.json holds four comparisons. Each has ordered 600 cases, full
  raw/noise/decoded tensor references, independent preprocessing and common-input
  references per case, actual metrics, start/end and the agreement reference.
  Independent/common bundle references retain worker/launch/schedule provenance;
  aggregates and the signed matrix are reconstructed from case captures.
  Tensor axes are (case, 40, 132) and (case, 16, 6); no shared-prefix slicing.
* validate_offline_evidence and validate_upstream_evidence are the only numerical
  acceptance paths, shared by CLI checks and validate_release_evidence. Stock
  acceptance includes positive producer output, JUnit, collection/runtime coverage,
  observed controls, actual inputs/noise, launches and the full raw boundary.
* golden candidate/replays bind native-operational, the exact 600 cases and
  raw/noise/decoded tensors. Replays link the exact candidate AND decision.
* decisions are DecisionRecord extensions, archived by canonical content digest
  and published once at the fixed stage name. Hashes provide local integrity,
  not authentication against the workstation owner.
* PreflightRecord is the runner contract. Historical links retain their original
  instances. Only assert_live_release checks freshness at the point of use.

Unknown/missing evidence fails closed. A positive status is necessary but never
sufficient: coverage, numerical witnesses, identities and chronology are checked.
"""

from __future__ import annotations

import ast
import hashlib
from xml.etree import ElementTree
import math
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from policy_guard.replay_contract import (
    CAMERA_ORDER, JOINT_ORDER, MAX_JSON, DecisionRecord, PrerequisiteError,
    canonical, capture_bytes, contained, fingerprint_configuration, load_numeric,
    now as utc_now, read_json, repeatability_schedule, validate_profile,
    validate_schedule, write_evidence, write_tensors, profile_configuration, validate_tf32_controls,
)

COMPARISON_PROFILES = {
    "diagnostic": ["native-diagnostic", "lerobot-diagnostic"],
    "native_bridge": ["native-diagnostic", "native-operational"],
    "lerobot_bridge": ["lerobot-diagnostic", "lerobot-operational"],
    "operational": ["native-operational", "lerobot-operational"],
}
PROFILE_NAMES = ("native-diagnostic", "lerobot-diagnostic", "native-operational", "lerobot-operational")
SERVING_SOURCE_FILES = (
    "policy_guard/groot_guard.py", "policy_guard/replay_contract.py",
    "docker/lerobot-policy/server.py",
)
DECISIONS = {
    "tolerances": ("tolerance-proposal.json", "tolerance-agreement.json"),
    "golden": ("golden-candidate.json", "golden-approval.json"),
    "live": ("release-review.json", "live-approval.json"),
}
STAGES = ("review", "live", "run", "trial-01", "trial-02", "trial-03")
MILESTONE_MODE = "milestone_12"
MILESTONE_LIVE_SCOPE = ("scoped-native-reference", "three-trial-physical-test")
MILESTONE_TRIAL1_SCOPE = ("scoped-native-reference", "physical-trial-1-only")


def single_trial_scope(index):
    require(type(index) is int and index in (1, 2, 3), "single trial must be 1, 2, or 3")
    return ["scoped-native-reference", f"physical-trial-{index}-only"]


def approved_trial_indices(record):
    scope = record.get("approval_scope")
    for index in (1, 2, 3):
        if scope == single_trial_scope(index):
            return (index,)
    require(scope in (None, list(MILESTONE_LIVE_SCOPE)), "invalid physical-trial scope")
    return (1, 2, 3)

PREPROCESSING_KEYS = {"image_front", "image_wrist", "state", "tokens", "mask"}
SEMANTIC_KEYS = {
    "backend", "purpose", "checkpoint_fingerprint", "backbone_fingerprint",
    "source_fingerprint", "packages_fingerprint", "image_digest",
    "effective_configuration", "parameter_dtypes", "buffer_dtypes", "compute_dtypes",
    "attention", "flow_steps", "eval", "autocast", "tf32", "tf32_matmul", "tf32_cudnn", "device", "seed_policy",
    "joint_order", "camera_order", "raw_shape", "decoded_shape",
}
INSTANCE_KEYS = {
    "container_id", "container_started_at", "pid", "process_started_at",
    "process_start_ticks", "boot_id", "load_id",
}
REQUEST_KEYS = {
    "observation_sha256", "timestamp", "timestep", "started_at", "completed_at",
    "output_sha256", "decoded_shape",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def timestamp(value):
    require(isinstance(value, str), "UTC timestamp required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0, "timestamp must be UTC")
    return result


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "SHA-256 required")
    return value


def text(value, name):
    require(isinstance(value, str) and bool(value.strip()), f"missing {name}")
    return value


class Evidence:
    """Read, hash and parse the same captured bytes once per validation operation."""

    def __init__(self, workspace, *, test_only=False):
        self.workspace = Path(workspace).resolve()
        self.test_only = test_only
        self._bytes = {}
        self._json = {}
        self._arrays = {}
        self._validated = {}

    def bytes(self, path, expected=None):
        target = contained(self.workspace, path)
        if path not in self._bytes:
            self._bytes[path] = capture_bytes(target)
        data = self._bytes[path]
        if expected is not None:
            require(hashlib.sha256(data).hexdigest() == sha(expected), f"evidence digest changed: {path}")
        return data

    def reference(self, path):
        return {"path": path, "sha256": hashlib.sha256(self.bytes(path)).hexdigest()}

    def json(self, reference):
        if isinstance(reference, str):
            reference = self.reference(reference)
        require(set(reference) == {"path", "sha256"}, "exact path/hash reference required")
        data = self.bytes(reference["path"], reference["sha256"])
        key = reference["path"]
        if key not in self._json:
            self._json[key] = read_json(data)
        return self._json[key]

    def tensors(self, reference):
        require(set(reference) == {"path", "sha256", "arrays"}, "complete tensor reference required")
        data = self.bytes(reference["path"], reference["sha256"])
        key = (reference["path"], reference["sha256"])
        if key not in self._arrays:
            self._arrays[key] = load_numeric(data, expected_keys=set(reference["arrays"]))
        arrays = self._arrays[key]
        require(set(arrays) == set(reference["arrays"]), "tensor keys changed")
        for name, array in arrays.items():
            require(reference["arrays"][name] == {"shape": list(array.shape), "dtype": str(array.dtype)},
                    f"tensor shape/dtype changed: {name}")
        return arrays

    def identity(self):
        if "identity" in self._validated:
            return self._validated["identity"]
        session = self.json("session.json")
        require(session.get("schema_version") == 1, "session schema mismatch")
        text(session.get("session_id"), "session")
        if session.get("supersedes") is not None:
            text(session.get("reason"), "successor session reason")
            previous = session["supersedes"]
            sha(previous["sha256"])
            text(previous["path"], "explicit predecessor")
        lock = self.json("input-lock.json")
        require(lock.get("fingerprint") == fingerprint_configuration({k: v for k, v in lock.items() if k != "fingerprint"}),
                "input lock fingerprint changed")
        validate_schedule(lock, lock["schedule"], "replay")
        profiles = self.json("profiles.json")
        require(profiles.get("schema_version") == 1 and profiles.get("session") == session["session_id"],
                "profile session mismatch")
        observed = {}
        for entry in profiles["profiles"]:
            if entry["purpose"] == "stock-capacity":
                continue
            name = entry["backend"] + "-" + entry["purpose"]
            require(name in PROFILE_NAMES and name not in observed, "duplicate/unknown measured profile")
            validate_profile(entry["observed"])
            require(entry["observed"]["backend"] == entry["backend"] and
                    entry["observed"]["purpose"] == entry["purpose"], "profile declaration mismatch")
            require(entry["observed"]["checkpoint_fingerprint"] == lock["checkpoint_fingerprint"],
                    "profile checkpoint mismatch")
            observed[name] = entry["observed"]
        require(set(observed) == set(PROFILE_NAMES), "all four measured profiles required")
        result = {"session": session["session_id"], "input_fingerprint": lock["fingerprint"],
                  "profiles_fingerprint": fingerprint_configuration(profiles)}
        self._validated["identity"] = result
        return result

    def record(self, reference):
        record = self.json(reference)
        require(record.get("schema_version") == 1, "stage schema mismatch")
        for key, value in self.identity().items():
            require(record.get(key) == value, f"stage {key} mismatch")
        require(record.get("evidence_kind") == ("test_only" if self.test_only else "real_model"),
                "only real_model evidence accepted; test_only requires explicit fixture injection")
        if record.get("status") == "not_run":
            raise PrerequisiteError("not run: required evidence did not run")
        require(record.get("status") == "complete" and not record.get("prerequisite_errors"),
                "required evidence is failed or incomplete")
        require(timestamp(record["started_at"]) <= timestamp(record["ended_at"]), "stage chronology reversed")
        return record


def evidence(workspace):
    return workspace if isinstance(workspace, Evidence) else Evidence(workspace)


def _number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "finite nonnegative threshold required")


def _bounds(value):
    require(set(value) >= {"atol", "rtol"}, "atol/rtol required")
    _number(value["atol"])
    _number(value["rtol"])


def _tensor_contract(ev, ref, count):
    arrays = ev.tensors(ref)
    require(set(arrays) == {"raw", "noise", "decoded"}, "full raw/noise/decoded archive required")
    for name, shape in (("raw", (count, 40, 132)), ("noise", (count, 40, 132)), ("decoded", (count, 16, 6))):
        require(arrays[name].shape == shape and arrays[name].dtype == np.float32, f"full {name} shape/dtype required")
    return arrays


def validate_repeatability(workspace):
    ev = evidence(workspace)
    if "repeatability" in ev._validated:
        return ev._validated["repeatability"]
    record = ev.record("repeatability.json")
    require(record.get("instrument_files") == instrument_identity(), "repeatability instrument became stale")
    schedule = repeatability_schedule(ev.json("input-lock.json"))
    require(record.get("schedule") == schedule, "repeatability schedule was not preregistered exactly")
    keys = [key for group in schedule["groups"] for key in group["cases"]]
    measurements = record["measurements"]
    require([m["profile"] for m in measurements] == list(PROFILE_NAMES), "repeatability requires all four profiles in order")
    for measured in measurements:
        require(measured["cases"] == keys, "partial repeatability coverage")
        groups = measured["groups"]
        require(len(groups) == len(schedule["groups"]), "cold processes missing")
        process_ids = []
        for group, expected in zip(groups, schedule["groups"], strict=True):
            require(group["id"] == expected["id"] and group["cases"] == expected["cases"], "repeatability group mismatch")
            process_ids.append(text(group["process_id"], "measured process identity"))
        require(len(set(process_ids)) == len(process_ids), "cold repeat must run in a new process")
        require(timestamp(record["started_at"]) <= timestamp(measured["started_at"]) <=
                timestamp(measured["ended_at"]) <= timestamp(record["ended_at"]), "repeatability chronology mismatch")
        basis = repeatability_basis(ev, measured, record)
        require(all(measured.get(key) == value for key, value in basis.items()),
                "reported repeatability statistics differ from validated captures")
        arrays = ev.tensors(measured["tensors"])
        require(set(arrays) == {"decoded", "noise"}, "repeatability must contain actual decoded actions and noise")
        for name, shape in (("decoded", (len(keys), 16, 6)), ("noise", (len(keys), 40, 132))):
            require(arrays[name].shape == shape and arrays[name].dtype == np.float32, "repeatability tensor shape/dtype mismatch")
            for record_name in dict.fromkeys(key["record"] for key in keys):
                same = [i for i, key in enumerate(keys) if key["record"] == record_name and key["mode"] != "changed"]
                changed = next(i for i, key in enumerate(keys) if key["record"] == record_name and key["mode"] == "changed")
                require(any(not np.array_equal(arrays[name][i], arrays[name][changed]) for i in same),
                        f"seed-ignoring {name} control")
                # Noise must be exactly replayed; output variability is measured,
                # then reviewed as tolerance basis rather than assumed to be zero.
                if name == "noise":
                    require(all(np.array_equal(arrays[name][same[0]], arrays[name][i]) for i in same),
                            "same seed did not reproduce actual noise")
    process_ids = [g["process_id"] for m in measurements for g in m["groups"]]
    require(len(set(process_ids)) == len(process_ids), "repeatability reused a process across profiles")
    launches = [ev.json(ref) for m in measurements for ref in m["launches"]]
    for before, after in zip(launches, launches[1:]):
        require(timestamp(before["ended_at"]) <= timestamp(after["started_at"]),
                "repeatability processes overlap or changed preregistered order")
    ev._validated["repeatability"] = record
    return record


def validate_tolerance_proposal(workspace):
    ev = evidence(workspace)
    proposal = ev.record("tolerance-proposal.json")
    repeat = validate_repeatability(ev)
    assert_instrument(ev)
    require(proposal["repeatability"] == ev.reference("repeatability.json"), "repeatability subject changed")
    require(timestamp(repeat["ended_at"]) <= timestamp(proposal["started_at"]), "proposal precedes repeatability")
    harness = proposal["harness"]
    require(re.fullmatch(r"[a-f0-9]{40}", harness["commit"]), "immutable harness commit required")
    sha(harness["producer_sha256"])
    sha(harness["consumer_sha256"])
    require(set(proposal["comparisons"]) == set(COMPARISON_PROFILES), "diagnostic and three operational bridges required")
    for name, pair in COMPARISON_PROFILES.items():
        comparison = proposal["comparisons"][name]
        require(comparison["profiles"] == pair, "comparison profile identity changed")
        bounds = comparison["thresholds"]
        require(set(bounds) == {"preprocessing", "raw", "decoded"}, "distinct tier thresholds required")
        _bounds(bounds["preprocessing"])
        _bounds(bounds["raw"])
        require(set(bounds["decoded"]) == {"max_abs", "mean_abs", "bias", "slope"}, "all decoded bounds required")
        for values in bounds["decoded"].values():
            require(isinstance(values, list) and len(values) == 6, "six joint thresholds required")
            for value in values:
                _number(value)
        require(comparison["noise_policy"] == ("exact" if name == "diagnostic" else "independent"),
                "diagnostic noise must match; cross-dtype RNG must be labelled independent")
        text(comparison["rationale"], "independent threshold rationale")
    _bounds(proposal["golden"])
    text(proposal["golden"]["rationale"], "golden threshold rationale")
    require(proposal["units"] == ["percent"] * 5 + ["gripper_percent"], "explicit checkpoint/gripper units required")
    require(proposal["aggregation"] == "trace-and-aggregate-ols-0..15", "trace and aggregate OLS contract required")
    require(isinstance(proposal["caveats"], list) and all(isinstance(v, str) for v in proposal["caveats"]),
            "caveats must be explicit")
    return proposal


def _decision(ev, kind, *, subject=None):
    subject_name, decision_name = DECISIONS[kind]
    record = ev.json(decision_name) if subject is None else subject
    require(record.get("schema_version") == 1 and record.get("decision_type") == kind, "decision schema/type mismatch")
    require(record.get("decision") == "approved", "explicit affirmative approval required")
    for key, value in ev.identity().items():
        require(record.get(key) == value, f"decision {key} mismatch")
    require(record.get("evidence_kind") == ("test_only" if ev.test_only else "real_model"),
            "test_only decision cannot authorize production")
    text(record.get("operator"), "operator identity")
    text(record.get("rationale"), "intentional decision rationale")
    timestamp(record["decided_at"])
    require(record["subject_digest"] == ev.reference(subject_name)["sha256"], "decision subject digest changed")
    expected_refs = decision_evidence(ev, kind)
    require(record.get("evidence_reviewed") == expected_refs, "reviewed evidence identity changed")
    for ref in record["evidence_reviewed"]:
        ev.json(ref)
    if subject is None:
        archive = f"decisions/{fingerprint_configuration(record)}.json"
        require(ev.bytes(archive) == ev.bytes(decision_name), "content-addressed decision archive missing or changed")
    return record


def decision_evidence(workspace, kind):
    ev = evidence(workspace)
    names = ["session.json", "input-lock.json", "profiles.json", DECISIONS[kind][0]]
    if kind == "live" and is_milestone_release(ev):
        names += ["milestone-acceptance.json", "milestone-criteria-authorization.json",
                  "milestone-scope.json", "milestone-report.json",
                  "milestone-golden-candidate.json", "milestone-golden-replay.json"]
    else:
        names += {
            "tolerances": ["repeatability.json"],
            "golden": ["tolerance-agreement.json"],
            "live": ["offline-report.json", "tolerance-agreement.json", "golden-candidate.json",
                     "golden-approval.json", "golden-replay.json"],
        }[kind]
    if kind == "live" and (ev.workspace / "trial-continuation.json").exists():
        names.append("trial-continuation.json")
    refs = [ev.reference(name) for name in names]
    if kind == "live":
        refs.append(ev.json("release-review.json")["review_preflight"])
    if kind == "golden" and ev.json("golden-candidate.json").get("previous") is not None:
        refs.append(ev.json("golden-candidate.json")["previous"])
    return refs


def validate_tolerance_agreement(workspace, *, comparison_started_at=None, decision=None):
    ev = evidence(workspace)
    proposal = validate_tolerance_proposal(ev)
    record = _decision(ev, "tolerances", subject=decision)
    require(timestamp(proposal["ended_at"]) <= timestamp(record["decided_at"]), "agreement precedes measured proposal")
    if comparison_started_at is not None:
        require(timestamp(record["decided_at"]) < timestamp(comparison_started_at), "comparison start must follow agreement")
    # Exact named records only, no newest-file search. Even a prospective check
    # must refuse an agreement recorded after an already-started comparison.
    for name in ("offline-report.json", "upstream-result.json"):
        if contained(ev.workspace, name).exists():
            comparison = ev.json(name)
            starts = [comparison["started_at"], *[c["started_at"] for c in comparison.get("comparisons", [])]]
            require(all(timestamp(record["decided_at"]) < timestamp(start) for start in starts),
                    "agreement chronology: comparison already started")
    return record


def _coverage(ev, record):
    require(record.get("cases") == ev.json("input-lock.json")["schedule"], "full ordered 120-record/600-case coverage required")


def validate_golden_candidate(workspace):
    ev = evidence(workspace)
    record = ev.record("golden-candidate.json")
    require(record.get("profile") == "native-operational", "golden requires actual native-operational path")
    _coverage(ev, record)
    _tensor_contract(ev, record["tensors"], 600)
    validate_tolerance_agreement(ev, comparison_started_at=record["started_at"])
    require(record["agreement"] == ev.reference("tolerance-agreement.json"), "golden tolerance identity mismatch")
    text(record["reason"], "intentional candidate reason")
    if record.get("previous") is not None:
        previous = ev.json(record["previous"])
        require(previous.get("candidate") and previous.get("approval"), "replacement requires approved predecessor manifest")
        prior_candidate = ev.json(previous["candidate"])
        prior_approval = ev.json(previous["approval"])
        require(prior_approval.get("decision") == "approved" and
                prior_approval.get("decision_type") == "golden" and
                prior_approval.get("subject_digest") == previous["candidate"]["sha256"],
                "replacement predecessor lacks subject-bound approval")
        require(timestamp(prior_candidate["ended_at"]) <= timestamp(prior_approval["decided_at"]) <
                timestamp(record["started_at"]), "replacement predecessor chronology")
    return record


def validate_golden_approval(workspace, *, decision=None):
    ev = evidence(workspace)
    candidate = validate_golden_candidate(ev)
    record = _decision(ev, "golden", subject=decision)
    require(timestamp(candidate["ended_at"]) <= timestamp(record["decided_at"]), "golden approved before candidate completed")
    return record


def _allclose(left, right, bounds, label):
    require(left.shape == right.shape, f"{label}: unequal full shapes")
    require(left.dtype == right.dtype, f"{label}: unequal dtypes")
    if left.dtype.kind in "biu":
        require(np.array_equal(left, right), f"{label}: categorical mismatch")
    else:
        require(np.all(np.abs(right.astype(np.float64) - left.astype(np.float64)) <=
                       bounds["atol"] + bounds["rtol"] * np.abs(left.astype(np.float64))),
                f"{label}: numerical threshold failed")


def _preprocessing(ev, rows, schedule, bounds, *, exact=False):
    require(len(rows) == len(schedule), "partial preprocessing/common-input coverage")
    checked = set()
    for row, key in zip(rows, schedule, strict=True):
        require(row["key"] == key, "preprocessing/common-input case identity mismatch")
        cache_key = fingerprint_configuration([row["left"], row["right"], bounds, exact])
        if cache_key in checked:
            continue
        left, right = ev.tensors(row["left"]), ev.tensors(row["right"])
        require(set(left) == set(right) == PREPROCESSING_KEYS, "complete independently captured preprocessing required")
        for name in PREPROCESSING_KEYS:
            require(left[name].size > 0, "empty preprocessing witness")
            if name in ("tokens", "mask"):
                require(left[name].dtype.kind in "biu", "tokens/masks must be exact integer metadata")
            _allclose(left[name], right[name], {"atol": 0, "rtol": 0} if exact else bounds, name)
        checked.add(cache_key)


def _metrics(delta):
    delta = delta.astype(np.float64)
    trace_bias = delta.mean(axis=1)
    indices = np.arange(16, dtype=np.float64) - 7.5
    trace_slope = np.einsum("ctj,t->cj", delta, indices) / np.dot(indices, indices)
    return {
        "max_abs": np.abs(delta).max(axis=(0, 1)), "mean_abs": np.abs(delta).mean(axis=(0, 1)),
        "bias": trace_bias.mean(axis=0), "slope": trace_slope.mean(axis=0),
        "per_index_bias": delta.mean(axis=0), "trace_bias_max_abs": np.abs(trace_bias).max(axis=0),
        "trace_slope_max_abs": np.abs(trace_slope).max(axis=0),
    }


def _golden_replay(ev, name):
    record = ev.record(name)
    approval = validate_golden_approval(ev)
    candidate = ev.json("golden-candidate.json")
    require(record["profile"] == "native-operational", "full operational native replay required")
    _coverage(ev, record)
    require(record["candidate"] == ev.reference("golden-candidate.json") and
            record["approval"] == ev.reference("golden-approval.json"), "golden replay subject/approval mismatch")
    require(timestamp(approval["decided_at"]) < timestamp(record["started_at"]), "native replay predates approval")
    left = _tensor_contract(ev, candidate["tensors"], 600)
    right = _tensor_contract(ev, record["tensors"], 600)
    for key in ("raw", "decoded"):
        _allclose(left[key], right[key], ev.json("tolerance-proposal.json")["golden"], f"native golden {key}")
    require(np.array_equal(left["noise"], right["noise"]), "native regression sampling changed")
    return record


def _calibration(ev):
    """Accept Plan 02's exact persisted schema, including explicit predecessor session."""
    if "calibration" in ev._validated:
        return ev._validated["calibration"]
    session = ev.json("session.json")
    reference = session.get("calibration_reference")
    if reference is not None:
        # This explicitly selected input is allowed to be outside the workspace.
        # It is never discovered by glob, and the very same bytes are parsed.
        data = capture_bytes(Path(reference["path"]), MAX_JSON)
        require(hashlib.sha256(data).hexdigest() == sha(reference["sha256"]), "calibration reference changed")
        record = read_json(data)
    else:
        reference = ev.reference("calibration.json")
        record = ev.json(reference)
    if ev.test_only:
        require(record["session"] == ev.identity()["session"] and record["status"] == "complete" and
                record["evidence_kind"] == "test_only" and record["drift_errors"] == [], "invalid fixture calibration")
        result = (sha(record["calibration_sha256"]), reference)
    else:
        require(record.get("kind") == "offline_arithmetic" and record.get("status") == "passed",
                "completed calibration derivation required")
        require(record.get("schema_version") == 1 and record.get("session_id"), "calibration session missing")
        if session.get("calibration_reference") is None:
            require(record["session_id"] == session["session_id"], "calibration session mismatch")
        require(timestamp(record["started_at"]) <= timestamp(record["ended_at"]), "calibration chronology")
        require(record["checks"]["pinned_snapshot"] == record["checks"]["reset_reachability"] == "passed",
                "calibration drift guard failed")
        # Read only: reuse the offline snapshot validator; no controller construction.
        from scripts.pose_sweep_units_probe import (
            ARM_JOINTS, DUME_POSES, MAX_RES, assert_pose_reachable,
            calibration_bounds, validate_pinned_snapshot,
        )
        calibration_bytes = capture_bytes(Path(record["calibration"]["path"]), MAX_JSON)
        statistics_bytes = capture_bytes(Path(record["statistics"]["path"]), MAX_JSON)
        require(hashlib.sha256(calibration_bytes).hexdigest() == record["calibration"]["sha256"] and
                hashlib.sha256(statistics_bytes).hexdigest() == record["statistics"]["sha256"], "current calibration/statistics changed")
        calibration = read_json(calibration_bytes)
        read_json(statistics_bytes)  # Reject duplicate keys/nonfinite JSON too.
        validate_pinned_snapshot(statistics_bytes, calibration_bytes)
        require(calibration == record["calibration_snapshot"], "calibration snapshot changed")
        require(record["joint_order"] == [joint.removesuffix(".pos") for joint in JOINT_ORDER],
                "calibration joint order changed")
        for joint in ARM_JOINTS:
            lo, hi, drive = calibration_bounds(calibration, joint)
            require(drive == 0 and record["scale_deg_per_pct"][joint] == (hi - lo) * 360 / (MAX_RES * 200),
                    "calibration derived scale changed")
        for name, pose in DUME_POSES.items():
            assert_pose_reachable(pose, calibration)
            require(record["reset_targets"][name]["vector"] == list(pose) and
                    record["reset_targets"][name]["reachable"] is True, "calibration reset derivation changed")
        result = (sha(record["calibration"]["sha256"]), reference)
    ev._validated["calibration"] = result
    return result


def is_milestone_release(workspace):
    # A broken declared artifact must fail validation, never fall back to legacy.
    return os.path.lexists(evidence(workspace).workspace / "milestone-acceptance.json")


def _milestone_release_evidence(ev):
    if "release:milestone" in ev._validated:
        return ev._validated["release:milestone"]
    from policy_guard.milestone_acceptance import validate_milestone_acceptance
    try:
        from policy_guard.milestone_golden import validate_scoped_golden
    except ModuleNotFoundError as exc:
        if exc.name != "policy_guard.milestone_golden":
            raise
        raise PrerequisiteError("not run: scoped native golden validator unavailable") from exc
    accepted = validate_milestone_acceptance(ev)
    golden = validate_scoped_golden(ev)
    require(accepted["status"] == golden["status"] == "complete", "milestone evidence incomplete")
    require(accepted["report"] == ev.reference("milestone-acceptance.json"), "milestone acceptance link changed")
    require(golden["candidate"] == ev.reference("milestone-golden-candidate.json") and
            golden["replay"] == ev.reference("milestone-golden-replay.json"), "scoped golden links changed")
    require(golden["ended_at"] == ev.json(golden["replay"])["ended_at"], "scoped golden completion changed")
    result = {**accepted, "acceptance_mode": MILESTONE_MODE, "scoped_golden": golden,
              "golden_approval_policy": "included-in-explicit-live-decision"}
    ev._validated["release:milestone"] = result
    return result


def validate_release_evidence(workspace):
    ev = evidence(workspace)
    if is_milestone_release(ev):
        return _milestone_release_evidence(ev)
    if "release" in ev._validated:
        return ev._validated["release"]
    report = validate_offline_evidence(ev)
    _golden_replay(ev, "golden-replay.json")
    calibration, calibration_ref = _calibration(ev)
    sem = ev.json("profiles.json").get("serving_configuration")
    if sem is None:
        raise PrerequisiteError("not run: measured operational serving attestation is missing")
    validate_semantic_configuration(sem)
    require(sem["checkpoint_fingerprint"] == ev.json("input-lock.json")["checkpoint_fingerprint"], "serving checkpoint changed")
    operational = next(item["observed"] for item in ev.json("profiles.json")["profiles"]
                       if item["backend"] == "lerobot" and item["purpose"] == "operational")
    require(sem == operational_semantics(operational),
            "serving semantics differ from the measured LeRobot operational profile")
    result = {
        **ev.identity(), "status": "complete", "report": ev.reference("offline-report.json"),
        "configuration_fingerprint": fingerprint_configuration(sem),
        "calibration_sha256": calibration, "calibration": calibration_ref,
    }
    ev._validated["release"] = result
    return result


def operational_semantics(profile):
    """Project measured replay facts onto deployed semantics, without replay RNG.

    Plan 04 must capture serving_seed_policy explicitly from the real server;
    seed_at_sampling_boundary describes an exogenous replay intervention and is
    never interpreted as a deployed fixed seed. Common serving sources are the
    same files in replay and the image; extra replay launcher sources stay in the
    locked profile identity. Only the content-addressed checkpoint path spelling
    is normalized between /inputs/checkpoint and /checkpoints/model.
    """
    require(profile["backend"] == "lerobot" and profile["purpose"] == "operational",
            "measured LeRobot operational profile required")
    require("serving_seed_policy" in profile and "effective_configuration" in profile and
            "owned_source_files" in profile, "measured operational serving/config/source facts missing")
    effective = profile["effective_configuration"]
    checkpoint_path = effective.get("policy", {}).get("base_model_path")

    def normalized(value, key=None):
        if isinstance(value, dict):
            return {k: normalized(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [normalized(v) for v in value]
        if checkpoint_path and key in ("base_model_path", "_name_or_path") and value == checkpoint_path:
            return "checkpoint-sha256:" + profile["checkpoint_fingerprint"]
        return value

    special = {"source_fingerprint", "packages_fingerprint", "effective_configuration", "seed_policy"}
    result = {key: profile[key] for key in SEMANTIC_KEYS - special}
    effective_value = normalized(effective)
    if set(effective_value) >= {"model", "policy", "serving"}:
        # Full configuration still participates in identity; publish only hashes
        # plus the explicitly allowlisted SAFE-01 facts, never arbitrary kwargs.
        effective_value = {
            "model_sha256": fingerprint_configuration(effective_value["model"]),
            "policy_sha256": fingerprint_configuration(effective_value["policy"]),
            "serving": effective_value["serving"],
            **({"processors_sha256": fingerprint_configuration(effective_value["processors"])}
               if "processors" in effective_value else {}),
        }
    result.update(
        source_fingerprint=fingerprint_configuration({
            "source": profile["source"],
            "owned": {name: profile["owned_source_files"][name] for name in SERVING_SOURCE_FILES},
        }),
        packages_fingerprint=fingerprint_configuration(profile["packages"]),
        effective_configuration=effective_value,
        seed_policy=profile["serving_seed_policy"],
    )
    validate_semantic_configuration(result)
    return result


def validate_semantic_configuration(sem):
    require(isinstance(sem, dict) and set(sem) == SEMANTIC_KEYS, "allowlisted measured semantic configuration required")
    for key in ("checkpoint_fingerprint", "backbone_fingerprint", "source_fingerprint", "packages_fingerprint"):
        sha(sem[key])
    require(re.fullmatch(r"sha256:[a-f0-9]{64}", sem["image_digest"]), "immutable image digest required")
    for key, expected in (("backend", "lerobot"), ("purpose", "operational"), ("flow_steps", 4),
                          ("eval", True), ("raw_shape", [1, 40, 132]), ("decoded_shape", [16, 6]),
                          ("joint_order", list(JOINT_ORDER)), ("camera_order", list(CAMERA_ORDER))):
        require(sem[key] == expected, f"observed serving {key} mismatch")
    for key in ("parameter_dtypes", "compute_dtypes", "attention"):
        require(isinstance(sem[key], list) and sem[key] and len(set(sem[key])) == len(sem[key]), f"missing actual {key}")
    require(isinstance(sem["buffer_dtypes"], list), "observed buffer dtypes required")
    require(type(sem["autocast"]) is bool, "actual autocast control required")
    validate_tf32_controls(sem)
    require(isinstance(sem["effective_configuration"], dict) and sem["effective_configuration"], "loaded configuration required")
    text(sem["device"], "observed device")
    seed = sem["seed_policy"]
    require(set(seed) == {"mode", "seed"} and (
        (seed["mode"] == "ambient" and seed["seed"] is None) or
        (seed["mode"] == "fixed" and type(seed["seed"]) is int)
    ), "deployed ambient/fixed seed policy required; replay seed is not a serving policy")


def validate_runtime_attestation(attestation, *, host, request, expected_configuration, now=None, test_only=False):
    require(set(attestation) == {"schema_version", "evidence_kind", "status", "semantic_configuration",
                                "configuration_fingerprint", "instance", "loaded_at", "request", "endpoint", "observations"},
            "attestation contains missing or non-allowlisted fields")
    require(attestation["schema_version"] == 1 and attestation["status"] == "complete", "completed real inference required")
    require(attestation["evidence_kind"] == ("test_only" if test_only else "real_model"), "test_only runtime is not real_model evidence")
    sem = attestation["semantic_configuration"]
    validate_semantic_configuration(sem)
    require(sem == expected_configuration and attestation["configuration_fingerprint"] == fingerprint_configuration(sem),
            "actual loaded configuration changed")
    measured = attestation["observations"]
    require(set(measured) == {"noise_shape", "noise_dtype", "noise_device", "noise_draws", "sdpa_calls",
                              "kernels", "flow_steps", "floating_operation_count", "raw_shape", "raw_dtype",
                              "input_dtypes", "backbone_dtypes"}, "allowlisted actual inference observations required")
    for key, value in (("noise_shape", [1, 40, 132]), ("noise_draws", 1), ("raw_shape", sem["raw_shape"]),
                       ("noise_device", sem["device"]), ("flow_steps", sem["flow_steps"])):
        require(measured[key] == value, f"actual inference {key} mismatch")
    require(type(measured["floating_operation_count"]) is int and measured["floating_operation_count"] > 0,
            "no actual floating compute observed")
    if "sdpa" in sem["attention"]:
        require(type(measured["sdpa_calls"]) is int and measured["sdpa_calls"] > 0, "no actual SDPA observed")
    for key in ("noise_dtype", "raw_dtype"):
        require(measured[key] in sem["compute_dtypes"], "actual tensor dtype is outside measured compute profile")
    require(isinstance(measured["kernels"], list) and measured["input_dtypes"] and measured["backbone_dtypes"],
            "actual kernel/input/backbone observations required")
    instance = attestation["instance"]
    require(set(instance) == INSTANCE_KEYS and set(host) == INSTANCE_KEYS | {
        "image_digest", "running", "endpoint", "container_port", "checked_at",
    }, "allowlisted host/process identity required")
    require(all(instance[key] == host[key] for key in INSTANCE_KEYS), "stale process/container/load instance")
    require(host["running"] is True and host["image_digest"] == sem["image_digest"], "running immutable container identity mismatch")
    sha(instance["container_id"])
    sha(instance["load_id"])
    require(type(instance["pid"]) is int and instance["pid"] > 0 and
            type(instance["process_start_ticks"]) is int and instance["process_start_ticks"] > 0, "actual process identity required")
    text(instance["boot_id"], "boot identity")
    require(re.fullmatch(r"127\.0\.0\.1:[0-9]{1,5}", host["endpoint"]), "release endpoint must be loopback")
    require(attestation["endpoint"] == {"host": "0.0.0.0", "port": host["container_port"]} or
            attestation["endpoint"] == {"host": "127.0.0.1", "port": host["container_port"]},
            "selected endpoint/container port mismatch")
    require(set(request) == REQUEST_KEYS and attestation["request"] == request, "fresh completed request identity mismatch")
    for key in ("observation_sha256", "output_sha256"):
        sha(request[key])
    require(type(request["timestamp"]) in (int, float) and math.isfinite(request["timestamp"]) and
            type(request["timestep"]) is int and request["timestep"] >= 0, "request timestamp/timestep invalid")
    require(request["decoded_shape"] == [16, 6], "completed inference must decode a full chunk")
    # procfs boot time is integral seconds and start ticks are quantized. Docker
    # records StartedAt after process creation. Bind both independent facts while
    # allowing their documented measurement granularity, never a stale process.
    require(abs((timestamp(instance["process_started_at"]) - timestamp(instance["container_started_at"])).total_seconds()) <= 2,
            "container/process start identity differs")
    order = [instance["process_started_at"], attestation["loaded_at"],
             request["started_at"], request["completed_at"], host["checked_at"]]
    require(all(timestamp(a) <= timestamp(b) for a, b in zip(order, order[1:])), "runtime/request chronology mismatch")
    if now is not None:
        for instant in (host["checked_at"], request["completed_at"]):
            age = (timestamp(now) - timestamp(instant)).total_seconds()
            require(0 <= age <= 60, "stale runtime attestation/request")
    return attestation


def write_runtime_json(path, payload):
    """Atomically replace ONLY transient runtime data, with private file mode."""
    path = Path(path)
    require(path.name in ("lerobot.json", "container.json"), "runtime filename must be lerobot.json or container.json")
    require(not path.is_symlink(), "runtime output cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".attestation-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            if os.geteuid() == 0:
                owner = path.parent.stat()
                os.fchown(stream.fileno(), owner.st_uid, owner.st_gid)
            stream.write(canonical(payload) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def array_fingerprint(array):
    array = np.asarray(array)
    require(array.dtype.kind in "biuf" and np.isfinite(array).all(), "finite numeric array required")
    return fingerprint_configuration({"shape": list(array.shape), "dtype": str(array.dtype),
                                      "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()})


def observation_fingerprint(observation):
    require(set(observation) == set(JOINT_ORDER) | set(CAMERA_ORDER) | {"task"}, "exact observation fields required")
    state = np.asarray([observation[joint] for joint in JOINT_ORDER], dtype=np.float64)
    require(state.shape == (6,) and np.isfinite(state).all(), "finite six-joint observation required")
    text(observation["task"], "instruction")
    frames = {}
    for camera in CAMERA_ORDER:
        frame = np.asarray(observation[camera])
        require(frame.shape == (480, 640, 3) and frame.dtype == np.uint8, "original camera geometry/dtype required")
        frames[camera] = array_fingerprint(frame)
    return fingerprint_configuration({"state": state.tolist(), "frames": frames, "task": observation["task"]})


def process_identity(pid=None):
    """Read actual Linux process identity; no environment or command-line dump."""
    pid = os.getpid() if pid is None else pid
    require(type(pid) is int and pid > 0, "positive process PID required")
    stat = Path(f"/proc/{pid}/stat").read_text()
    ticks = int(stat[stat.rfind(")") + 2:].split()[19])
    boot = int(next(line.split()[1] for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime ")))
    return {"pid": pid, "process_start_ticks": ticks,
            "process_started_at": datetime.fromtimestamp(boot + ticks / os.sysconf("SC_CLK_TCK"), timezone.utc).isoformat(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def container_binding(inspected, endpoint, checkpoint_path):
    """Reduce actual Docker inspect output to a nonsecret, read-only binding.

    Plan 04/06 writes this to runtime/container.json before the guarded handshake;
    the opt-in server reads that same mounted directory. Reinspect after the fresh
    request through collect_runtime_host; image tags and supplied config are not
    accepted as substitutes for these engine facts.
    """
    require(re.fullmatch(r"127\.0\.0\.1:[0-9]{1,5}", endpoint), "loopback endpoint required")
    host_port = endpoint.rsplit(":", 1)[1]
    require(1 <= int(host_port) <= 65535, "invalid host port")
    sha(inspected["Id"])
    require(re.fullmatch(r"sha256:[a-f0-9]{64}", inspected["Image"]), "actual image ID required")
    require(inspected["State"]["Running"] is True and inspected["HostConfig"]["NetworkMode"] != "host",
            "running isolated container required")
    timestamp(inspected["State"]["StartedAt"])
    matching = []
    for port, mappings in inspected["NetworkSettings"]["Ports"].items():
        if mappings == [{"HostIp": "127.0.0.1", "HostPort": host_port}] and port.endswith("/tcp"):
            matching.append(int(port.split("/")[0]))
    require(len(matching) == 1, "selected endpoint must map uniquely and only to loopback")
    mounts = [m for m in inspected["Mounts"] if m["Destination"] == checkpoint_path]
    require(len(mounts) == 1 and mounts[0]["RW"] is False, "exact read-only checkpoint mount required")
    return {"schema_version": 1, "container_id": inspected["Id"], "image_digest": inspected["Image"],
            "container_started_at": inspected["State"]["StartedAt"], "endpoint": endpoint,
            "container_port": matching[0], "checkpoint_path": checkpoint_path,
            "checkpoint_source": mounts[0]["Source"]}


def inspect_container_binding(container, endpoint, checkpoint_path):
    require(re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", container), "explicit container name or ID required")
    result = subprocess.run(["docker", "inspect", "--type", "container", "--format", "{{json .}}", container],
                            capture_output=True, timeout=20, check=True)
    return container_binding(read_json(result.stdout), endpoint, checkpoint_path)


def collect_runtime_host(attestation, container, endpoint, checkpoint_path):
    """Read-only host observation after a fresh arm-free request; no model load."""
    binding = inspect_container_binding(container, endpoint, checkpoint_path)
    instance = attestation["instance"]
    pid = instance["pid"]
    require(type(pid) is int and pid > 0, "actual server PID required")
    code = "import json,sys; from policy_guard.parity_gate import process_identity; print(json.dumps(process_identity(int(sys.argv[1]))))"
    result = subprocess.run(["docker", "exec", binding["container_id"], "python3", "-c", code, str(pid)],
                            capture_output=True, timeout=20, check=True)
    process = read_json(result.stdout)
    # load_id belongs to the completed request; PID/ticks/boot/start are checked
    # independently through procfs, and the request/output identities bind load_id.
    return {**process, "load_id": instance["load_id"], "container_id": binding["container_id"],
            "container_started_at": binding["container_started_at"], "image_digest": binding["image_digest"],
            "running": True, "endpoint": binding["endpoint"], "container_port": binding["container_port"],
            "checked_at": utc_now()}


@dataclass(frozen=True)
class PreflightRecord:
    session: str
    stage: str
    attempt: int
    started_at: str
    ended_at: str
    status: str
    configuration_fingerprint: str
    calibration_sha256: str
    attestation: dict
    attestation_sha256: str
    host: dict
    request: dict
    previous: dict | None = None
    reason: str = ""
    approval: dict | None = None
    review_preflight: dict | None = None
    evidence_kind: str = "real_model"
    schema_version: int = 1


def _preflight_path(record):
    require(record["stage"] in STAGES and type(record["attempt"]) is int and 1 <= record["attempt"] <= 9999,
            "explicit valid preflight stage and positive four-digit attempt required")
    return f"preflights/{record['stage']}-{record['attempt']:04d}.json"


def _preflight_record(ev, record, *, path, success=True, seen=None):
    require(path == _preflight_path(record), "preflight path/stage/attempt mismatch")
    require(set(record) == set(PreflightRecord.__dataclass_fields__), "preflight schema mismatch")
    require(record["schema_version"] == 1 and record["session"] == ev.identity()["session"], "preflight session mismatch")
    require(record["evidence_kind"] == ("test_only" if ev.test_only else "real_model"), "test_only preflight not permitted")
    require(record["status"] in ("complete", "failed", "not_run"), "unknown preflight status")
    require(timestamp(record["started_at"]) <= timestamp(record["ended_at"]), "preflight chronology")
    require(record["attestation_sha256"] == fingerprint_configuration(record["attestation"]), "embedded attestation changed")
    seen = set() if seen is None else seen
    require(path not in seen and len(seen) < 100, "cyclic/excessive preflight history")
    seen.add(path)
    cache_key = "preflight:" + path
    if cache_key in ev._validated:
        return record
    previous = record["previous"]
    stage, attempt = record["stage"], record["attempt"]
    if stage == "review" and attempt == 1:
        require(previous is None, "first review has no predecessor")
    else:
        require(previous is not None, "explicit preflight predecessor required")
        text(record["reason"], "preflight renewal/transition reason")
        prior = ev.json(previous)
        _preflight_record(ev, prior, path=previous["path"], success=prior["status"] == "complete", seen=seen)
        if attempt > 1:
            require(prior["stage"] == stage and prior["attempt"] == attempt - 1, "renewal must link previous attempt of same stage")
        else:
            expected_prior = STAGES[STAGES.index(stage) - 1]
            if stage.startswith("trial-") and prior["stage"] == "run":
                indices = approved_trial_indices(validate_live_approval(ev))
                if len(indices) == 1 and stage == f"trial-{indices[0]:02d}":
                    expected_prior = "run"
            require(prior["stage"] == expected_prior and prior["status"] == "complete",
                    "initial stage must follow successful preceding stage")
        require(timestamp(prior["ended_at"]) < timestamp(record["started_at"]), "predecessor chronology")
    if not success and record["status"] != "complete":
        return record
    require(record["status"] == "complete", "failed/not_run preflight cannot release")
    release = validate_release_evidence(ev)
    for key in ("configuration_fingerprint", "calibration_sha256"):
        require(record[key] == release[key], f"preflight {key} changed")
    validate_runtime_attestation(
        record["attestation"], host=record["host"], request=record["request"],
        expected_configuration=ev.json("profiles.json")["serving_configuration"],
        now=record["ended_at"], test_only=ev.test_only,
    )
    require(timestamp(record["started_at"]) <= timestamp(record["request"]["started_at"]) <=
            timestamp(record["host"]["checked_at"]) <= timestamp(record["ended_at"]), "request outside preflight interval")
    if stage == "review":
        require(record["approval"] is None and record["review_preflight"] is None, "review readiness cannot claim approval")
        if release.get("acceptance_mode") == MILESTONE_MODE:
            require(timestamp(release["scoped_golden"]["ended_at"]) < timestamp(record["started_at"]) and
                    timestamp(release["ended_at"]) <= timestamp(record["started_at"]),
                    "review precedes milestone acceptance or scoped golden replay")
        else:
            require(timestamp(ev.json("golden-replay.json")["ended_at"]) < timestamp(record["started_at"]) and
                    timestamp(ev.json("offline-report.json")["archived_at"]) <= timestamp(record["started_at"]), "review precedes archived evidence")
    else:
        approval = validate_live_approval(ev)
        require(record["approval"] == ev.reference("live-approval.json") and
                record["review_preflight"] == ev.json("release-review.json")["review_preflight"], "preflight approval/review link mismatch")
        require(timestamp(approval["decided_at"]) < timestamp(record["started_at"]), "live preflight predates approval")
    ev._validated[cache_key] = record
    return record


def write_preflight_record(workspace, record):
    ev = evidence(workspace)
    value = asdict(record) if isinstance(record, PreflightRecord) else record
    path = _preflight_path(value)
    require(not contained(ev.workspace, path).exists(), "immutable preflight path already exists")
    _preflight_record(ev, value, path=path, success=value["status"] == "complete")
    return write_evidence(ev.workspace, path, value)


def validate_preflight_record(workspace, reference, *, expected_stage=None):
    ev = evidence(workspace)
    if isinstance(reference, str):
        reference = ev.reference(reference)
    record = ev.json(reference)
    if expected_stage is not None:
        require(record["stage"] == expected_stage, "wrong preflight stage")
    return _preflight_record(ev, record, path=reference["path"])


def validate_release_review(workspace):
    ev = evidence(workspace)
    release = validate_release_evidence(ev)
    review = ev.record("release-review.json")
    require(review["release_evidence"] == release, "release review identity changed")
    preflight = validate_preflight_record(ev, review["review_preflight"], expected_stage="review")
    require(timestamp(preflight["ended_at"]) <= timestamp(review["started_at"]), "release review predates exact review preflight")
    return review


def validate_live_approval(workspace, *, decision=None):
    ev = evidence(workspace)
    review = validate_release_review(ev)
    record = _decision(ev, "live", subject=decision)
    if review["release_evidence"].get("acceptance_mode") == MILESTONE_MODE:
        require(record.get("approval_scope") in (list(MILESTONE_LIVE_SCOPE), *(single_trial_scope(i) for i in (1, 2, 3))),
                "explicit combined scoped-reference and physical-test approval required")
    require(timestamp(review["ended_at"]) < timestamp(record["decided_at"]), "live approval must follow archive/review")
    return record


def assert_live_release(workspace, preflight, *, expected_stage, runtime,
                        current_calibration_sha256, now=None):
    supplied = evidence(workspace)
    # A caller may have cached an earlier review/check. Physical release always
    # captures current bytes again, while sharing each capture within this call.
    ev = Evidence(supplied.workspace, test_only=supplied.test_only)
    require(expected_stage in STAGES[1:], "review-stage readiness cannot release hardware")
    approval = validate_live_approval(ev)
    indices = approved_trial_indices(approval)
    if len(indices) == 1:
        require(expected_stage in ("live", "run", f"trial-{indices[0]:02d}"),
                f"trial-{indices[0]}-only approval cannot release another trial")
    record = validate_preflight_record(ev, preflight, expected_stage=expected_stage)
    require(current_calibration_sha256 == record["calibration_sha256"], "current calibration changed")
    validate_runtime_attestation(
        runtime["attestation"], host=runtime["host"], request=runtime["request"],
        expected_configuration=ev.json("profiles.json")["serving_configuration"],
        now=now or utc_now(), test_only=ev.test_only,
    )
    require(record["attestation"] == runtime["attestation"] and record["request"] == runtime["request"] and
            record["host"] == runtime["host"],
            "persisted preflight is for a stale runtime instance/request")
    age = (timestamp(now or utc_now()) - timestamp(record["ended_at"])).total_seconds()
    require(0 < age <= 60, "release requires a fresh persisted preflight; create an explicit new attempt")
    return record


def validate_run_safety_journal(run):
    events = run.get("events")
    require(isinstance(events, list) and bool(events), "operation journal missing")
    previous, stops = None, []
    for index, entry in enumerate(events):
        require(isinstance(entry, dict), "invalid operation journal entry")
        event = dict(entry)
        checksum = event.pop("sha256", None)
        require(type(event.get("index")) is int and event["index"] == index and
                event.get("previous") == previous and fingerprint_configuration(event) == checksum,
                "operation journal changed")
        require(event.get("kind") in ("dispatch", "returned", "stop"), "unknown operation journal event")
        if event["kind"] == "stop":
            require(type(event.get("clamp")) is bool, "invalid journal clamp flag")
            text(event.get("reason"), "journal stop reason")
            stops.append(entry)
        previous = checksum
    require(run.get("stop_events") == stops, "journal stop_events disagree")
    clamps = sum(event["clamp"] for event in stops)
    require(type(run.get("safety_stop")) is bool and run["safety_stop"] == bool(stops) and
            type(run.get("clamp_warnings")) is int and run["clamp_warnings"] == clamps and
            run.get("stop_reason") == (stops[0]["reason"] if stops else ""),
            "journal safety summary disagrees")
    require(not stops and clamps == 0, "journal safety event overrides score")



def validate_closeout(workspace):
    ev = evidence(workspace)
    validate_live_approval(ev)
    run = ev.record("live-run.json")
    validate_run_safety_journal(run)
    require(run["approval"] == ev.reference("live-approval.json"), "run approval mismatch")
    require(run["safety_stop"] is False and type(run["clamp_warnings"]) is int and run["clamp_warnings"] == 0,
            "safety event overrides score")
    require(run["instruction"] == "Grab a banana and put it on the plate", "changed directional instruction")
    live = validate_preflight_record(ev, run["live_preflight"], expected_stage="live")
    construction = validate_preflight_record(ev, run["run_preflight"], expected_stage="run")
    require(construction["previous"] == run["live_preflight"], "run must link the selected live preflight")
    require(timestamp(live["ended_at"]) < timestamp(construction["started_at"]) and
            timestamp(construction["ended_at"]) < timestamp(run["constructed_at"]) == timestamp(run["started_at"]),
            "controller construction preceded current preflight")
    require(len(run["trials"]) == 3, "exactly three completed trials required")
    previous, last_end, successes = run["run_preflight"], run["constructed_at"], 0
    for index, trial in enumerate(run["trials"], 1):
        require(trial["index"] == index, "trial order/denominator changed")
        record = validate_preflight_record(ev, trial["preflight"], expected_stage=f"trial-{index:02d}")
        require(record["previous"] == previous, "trial preflight predecessor mismatch")
        require(timestamp(last_end) < timestamp(record["started_at"]) <= timestamp(record["ended_at"]) <
                timestamp(trial["started_at"]) <= timestamp(trial["ended_at"]) <= timestamp(run["ended_at"]),
                "trial/reset must follow its current preflight")
        require(trial["iterations"] == 20 and trial["actions_per_chunk"] == 16 and trial["action_delay"] == 0.05,
                "fixed trial movement budget changed")
        require(trial["safety_stop"] is False and type(trial["clamp_warnings"]) is int and trial["clamp_warnings"] == 0,
                "trial safety event overrides score")
        for flag in ("coherent", "wrong_target", "erratic", "grasp"):
            require(type(trial[flag]) is bool, "named directional judgments required")
        text(trial["operator"], "trial operator identity")
        successes += int(trial["coherent"] and not trial["wrong_target"] and not trial["erratic"])
        previous, last_end = trial["preflight"], trial["ended_at"]
    require(successes >= 2, "fewer than two coherent directional trials")
    final = _golden_replay(ev, "final-regression.json")
    require(final["live_run"] == ev.reference("live-run.json") and
            timestamp(run["ended_at"]) < timestamp(final["started_at"]), "final full native regression must follow this run")
    return {**ev.identity(), "schema_version": 1, "status": "complete",
            "live_run": ev.reference("live-run.json"), "final_regression": ev.reference("final-regression.json")}


HARNESS_COMMIT = "7e241bd630a3719a56157a497ce5d08f244784f1"
PRODUCER = "tests/policies/groot/utils/dump_original_n1_7.py"
CONSUMER = "tests/policies/groot/test_groot_vs_original.py"
CASE = "test_groot_get_action_parity[new_embodiment]"
ARTIFACT = "original_n1_7_new_embodiment.npz"


DTYPE_NAMES = ("torch.bfloat16", "torch.float32", "torch.float64", "torch.float16",
               "torch.int64", "torch.int32", "torch.int16", "torch.int8", "torch.uint8", "torch.bool")

def validate_coverage(cases, expected):
    require(bool(expected) and cases == expected, "complete ordered case coverage required")
    keys = [canonical(key) for key in cases]
    require(len(set(keys)) == len(keys), "duplicate case identity")


def _finite(values):
    values = np.asarray(values)
    require(values.dtype.kind in "biuf" and values.size > 0 and np.isfinite(values).all(),
            "nonempty finite numeric arrays required")
    return values


def joint_deviations(left, right):
    left, right = _finite(left), _finite(right)
    require(left.shape == right.shape and left.ndim == 3 and left.shape[1:] == (16, 6),
            "full decoded trace shape must be (case,16,6)")
    return right.astype(np.float64) - left.astype(np.float64)


def signed_bias(delta):
    delta = _finite(delta)
    require(delta.ndim == 3 and delta.shape[1:] == (16, 6), "full decoded delta required")
    return delta.astype(np.float64).mean(axis=1)


def chunk_index_slopes(delta):
    delta = _finite(delta)
    require(delta.ndim == 3 and delta.shape[1:] == (16, 6), "full decoded delta required")
    indices = np.arange(16, dtype=np.float64) - 7.5
    return np.einsum("ctj,t->cj", delta.astype(np.float64), indices) / np.dot(indices, indices)


def metrics(delta):
    """Same float64 definitions consumed independently by the release validator."""
    delta = _finite(delta).astype(np.float64)
    bias, slopes = signed_bias(delta), chunk_index_slopes(delta)
    return {
        "max_abs": np.abs(delta).max(axis=(0, 1)),
        "mean_abs": np.abs(delta).mean(axis=(0, 1)),
        "bias": bias.mean(axis=0), "slope": slopes.mean(axis=0),
        "per_index_bias": delta.mean(axis=0),
        "trace_bias_max_abs": np.abs(bias).max(axis=0),
        "trace_slope_max_abs": np.abs(slopes).max(axis=0),
    }


def array_comparison(left, right, bounds, *, exact=False):
    left, right = _finite(left), _finite(right)
    require(left.shape == right.shape, "unequal full tensor shape")
    require(left.dtype == right.dtype, "unequal serialized tensor dtype")
    if exact or left.dtype.kind in "biu":
        return bool(np.array_equal(left, right))
    delta = np.abs(right.astype(np.float64) - left.astype(np.float64))
    return bool(np.all(delta <= bounds["atol"] + bounds["rtol"] * np.abs(left.astype(np.float64))))


def _rows(ev, left, right, keys, field, bounds):
    require(len(left[field]) == len(right[field]) == len(keys), f"partial {field} coverage")
    rows, failures, checked = [], [], {}
    for index, (key, lref, rref) in enumerate(zip(keys, left[field], right[field], strict=True)):
        row = {"key": key, "left": lref, "right": rref}
        rows.append(row)
        cache = fingerprint_configuration([lref, rref])
        if cache not in checked:
            lvalues, rvalues = ev.tensors(lref), ev.tensors(rref)
            require(set(lvalues) == set(rvalues), f"{field}: different input keys")
            if field in ("preprocessing", "common_inputs"):
                require(set(lvalues) == PREPROCESSING_KEYS, f"{field}: incomplete preprocessing witness")
            else:
                require(bool(lvalues), "empty full collated input witness")
            passed = True
            for name in lvalues:
                if name in ("tokens", "mask"):
                    require(lvalues[name].dtype.kind in "biu", "tokens/masks must be exact integers")
                passed &= array_comparison(lvalues[name], rvalues[name], bounds,
                                           exact=field in ("common_inputs", "common_collated"))
            checked[cache] = passed
        if not checked[cache]:
            failures.append(f"{field}[{index}]")
    return rows, failures


def _bundle(ev, bundle, profile, schedule):
    validate_coverage(bundle["cases"], schedule)
    require(bundle["profile"] == profile, "comparison profile mismatch")
    require(bundle["input_fingerprint"] == ev.identity()["input_fingerprint"], "input identity mismatch")
    require(bundle["joint_order"] == list(JOINT_ORDER), "joint ordering mismatch")
    require(bundle["camera_order"] == list(CAMERA_ORDER), "camera ordering mismatch")
    arrays = ev.tensors(bundle["tensors"])
    require(set(arrays) == {"raw", "noise", "decoded"}, "raw/noise/decoded witnesses required")
    for name, shape in (("raw", (600, 40, 132)), ("noise", (600, 40, 132)), ("decoded", (600, 16, 6))):
        require(arrays[name].shape == shape and arrays[name].dtype == np.float32,
                f"full {name} shape/dtype required")
        _finite(arrays[name])
    return arrays


def compare_tiers(workspace, name, left, right, *, started_at, ended_at, matrix_reference=None):
    """Emit discriminating metrics even on numerical failure; structural faults raise."""
    ev = evidence(workspace)
    agreement = validate_tolerance_agreement(ev, comparison_started_at=started_at)
    proposal = validate_tolerance_proposal(ev)
    require(timestamp(started_at) <= timestamp(ended_at), "comparison chronology reversed")
    require(name in COMPARISON_PROFILES, "unknown comparison")
    rule = proposal["comparisons"][name]
    profiles = {p["backend"] + "-" + p["purpose"]: p["observed"] for p in ev.json("profiles.json")["profiles"] if p["purpose"] != "stock-capacity"}
    if name == "diagnostic":
        for key in ("device", "rng_algorithm", "noise_dtype", "noise_shape"):
            require(profiles[rule["profiles"][0]][key] == profiles[rule["profiles"][1]][key], "diagnostic actual sampler controls differ")
    schedule = ev.json("input-lock.json")["schedule"]
    validate_schedule(ev.json("input-lock.json"), schedule, "replay")
    la = _bundle(ev, left, rule["profiles"][0], schedule)
    ra = _bundle(ev, right, rule["profiles"][1], schedule)
    rows, failures = {}, []
    for field in ("preprocessing", "independent_collated", "common_inputs", "common_collated"):
        rows[field], issues = _rows(ev, left, right, schedule, field, rule["thresholds"]["preprocessing"])
        failures.extend(issues)
    if not array_comparison(la["raw"], ra["raw"], rule["thresholds"]["raw"]):
        failures.append("raw")
    noise_equal = bool(np.array_equal(la["noise"], ra["noise"]))
    if rule["noise_policy"] == "exact" and not noise_equal:
        failures.append("noise")
    delta = joint_deviations(la["decoded"], ra["decoded"])
    actual = metrics(delta)
    for metric in ("max_abs", "mean_abs", "bias", "slope"):
        if np.any(np.abs(actual[metric]) > rule["thresholds"]["decoded"][metric]):
            failures.append("decoded." + metric)
    for metric, bound in (("trace_bias_max_abs", "bias"), ("trace_slope_max_abs", "slope")):
        if np.any(actual[metric] > rule["thresholds"]["decoded"][bound]):
            failures.append("decoded.trace_" + bound)
    matrix_values = {
        "delta": delta, "trace_bias": signed_bias(delta), "trace_slope": chunk_index_slopes(delta),
        "trace_max_abs": np.abs(delta).max(axis=1), "trace_mean_abs": np.abs(delta).mean(axis=1),
    }
    if matrix_reference is None:
        matrix = write_tensors(ev.workspace, matrix_values)
    else:
        saved = ev.tensors(matrix_reference)
        require(set(saved) == set(matrix_values) and all(np.array_equal(saved[k], v) for k, v in matrix_values.items()), "archived signed matrix differs")
        matrix = matrix_reference
    return {
        "name": name, "profiles": rule["profiles"], "cases": schedule,
        "started_at": started_at, "ended_at": ended_at,
        "agreement": ev.reference("tolerance-agreement.json"),
        "agreement_decided_at": agreement["decided_at"],
        "joint_order": list(JOINT_ORDER), "camera_order": list(CAMERA_ORDER),
        "left": left["tensors"], "right": right["tensors"], **rows,
        "metrics": {key: value.tolist() for key, value in actual.items()},
        "matrix": matrix, "units": proposal["units"],
        "slope_units": [unit + "/action_index" for unit in proposal["units"]],
        "axes": ["record_seed_repeat", "chunk_index_0..15", "joint"],
        "delta_definition": "right-minus-left", "aggregation": proposal["aggregation"],
        "thresholds": rule["thresholds"], "passed": not failures, "failures": failures,
        "noise": {"policy": rule["noise_policy"], "equal": noise_equal,
                  "matched_seed_proves_equal_noise": False,
                  "profiles": [{key: profiles[p][key] for key in ("device", "rng_algorithm", "noise_dtype", "noise_shape")} for p in rule["profiles"]],
                  "treatment": rule["rationale"]},
        "caveats": proposal["caveats"],
    }


# The dependency direction is producers -> gate -> replay_contract. No validator
# imports an executable producer, a controller, torch, or a model adapter.
INSTRUMENT_FILES = (
    "policy_guard/replay_contract.py", "policy_guard/parity_gate.py", "policy_guard/parity_report.py",
    "policy_guard/groot_guard.py", "scripts/replay_checkpoint_parity.py",
    "scripts/replay_upstream_parity.py", "scripts/replay_groot_native.py",
    "docker/lerobot-policy/replay_checkpoint.py", "docker/lerobot-policy/server.py",
    "policy/lerobot/features.py",
)
WORKER_SOURCE_FILES = tuple(name for name in INSTRUMENT_FILES if name not in (
    "policy_guard/parity_gate.py", "policy_guard/parity_report.py", "scripts/replay_upstream_parity.py",
))


def instrument_identity():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256(capture_bytes(root / name)).hexdigest() for name in INSTRUMENT_FILES}


def assert_instrument(workspace):
    ev = evidence(workspace)
    require(ev.json("tolerance-proposal.json").get("instrument_files") == instrument_identity(),
            "instrument changed: fresh repeatability and explicit agreement required")


def _within(start, end, outer_start, outer_end, label):
    require(timestamp(outer_start) <= timestamp(start) <= timestamp(end) <= timestamp(outer_end),
            label + " chronology mismatch")


def _profile(ev, name):
    return next(p["observed"] for p in ev.json("profiles.json")["profiles"]
                if p["backend"] + "-" + p["purpose"] == name)


def _same_arrays(left, right, label):
    require(set(left) == set(right) and bool(left), label + " keys differ")
    for key in left:
        require(array_comparison(left[key], right[key], {}, exact=True), label + " bytes differ: " + key)


def numerical_execution_binding(plan, schedule_reference, worker_manifest):
    """Bind numerical collection metadata to the shared execute_cases schema."""
    value = plan.get("execution")
    require(isinstance(value, dict) and set(value) == {"id", "collection", "started_at", "worker_manifest"},
            "complete numerical execution binding required")
    require(isinstance(value["id"], str) and re.fullmatch(r"[0-9a-f]{32}", value["id"]),
            "invalid numerical execution identity")
    require(plan["kind"] in ("tracer", "repeatability", "replay") and
            value["collection"] == "numerical-" + plan["kind"] and value["worker_manifest"] == worker_manifest,
            "numerical execution collection/destination mismatch")
    require(set(schedule_reference) == {"path", "sha256"}, "execution schedule reference missing")
    sha(schedule_reference["sha256"])
    timestamp(value["started_at"])
    return {**value, "schedule": schedule_reference}


def validate_numerical_worker(workspace, proof, name, cases, *, kind, start, end, common_from=None):
    """Validate launch -> immutable worker -> immutable cases before reducing.

    ``proof`` has exactly worker and launch file/hash references. Launches also
    bind the schedule, current instrument and log. Test fixtures use this same
    schema and validation; only the required evidence_kind differs.
    """
    ev = evidence(workspace)
    require(set(proof) == {"worker", "launch"}, "worker and launch provenance required")
    cache_key = ("numerical-worker", fingerprint_configuration([proof, name, cases, kind, common_from]))
    if cache_key in ev._validated:
        result = ev._validated[cache_key]
        _within(result["launch"]["started_at"], result["launch"]["ended_at"], start, end, "worker launch")
        return result
    worker, launch = ev.json(proof["worker"]), ev.json(proof["launch"])
    profile = _profile(ev, name)
    backend, purpose = name.split("-")
    identity = ev.identity()
    instrument = instrument_identity()
    require(launch["manifest"] == proof["worker"] and launch["status"] == "complete" and
            type(launch["exit_code"]) is int and launch["exit_code"] == 0, "worker launch failed or changed")
    require(launch["backend"] == backend and launch["purpose"] == purpose, "worker launch profile mismatch")
    require(launch["instrument_files"] == instrument and
            worker["resources"]["instrument_files"] == instrument, "worker instrument became stale")
    ev.bytes(launch["log"]["path"], launch["log"]["sha256"])
    plan = ev.json(launch["schedule"])
    require(plan["schema_version"] == 1 and plan["session"] == identity["session"] and
            plan["input_fingerprint"] == identity["input_fingerprint"] and plan["kind"] == kind and
            plan["cases"] == cases and plan.get("common_from") == common_from, "worker schedule provenance mismatch")
    execution = numerical_execution_binding(plan, launch["schedule"], proof["worker"]["path"])
    require(worker.get("execution") == execution, "worker execution differs from captured schedule")
    require(timestamp(execution["started_at"]) <= timestamp(launch["started_at"]),
            "execution collection postdates worker launch")
    argv = launch["argv"]
    require(argv[:2] == ["docker", "run"], "actual worker invocation required")
    for flag, expected in (
        ("--schedule", "/evidence/" + launch["schedule"]["path"]),
        ("--backend", backend), ("--profile", purpose), ("--device", profile["device"]),
        ("--image-digest", profile["image_digest"]), ("--checkpoint", "/inputs/checkpoint"),
        ("--output-manifest", proof["worker"]["path"]),
    ):
        require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == expected, "worker argv differs: " + flag)
    require(argv[argv.index("--entrypoint") + 2] == profile["image_digest"] and
            "/replay/scripts/replay_checkpoint_parity.py" in argv and "_worker" in argv,
            "wrong worker image or instrument entrypoint")
    require(worker["schema_version"] == 1 and worker["status"] == "complete" and
            not worker.get("prerequisite_errors") and worker["stage"] == purpose and
            worker["session"] == identity["session"] and worker["input_fingerprint"] == identity["input_fingerprint"] and
            worker["evidence_kind"] == ("test_only" if ev.test_only else "real_model"),
            "incomplete or foreign numerical worker")
    require(worker["expected_cases"] == worker["executed_cases"] == cases and
            [row["key"] for row in worker["cases"]] == cases, "worker case coverage differs")
    validate_schedule(ev.json("input-lock.json"), cases, kind)
    validate_profile(worker["profile"])
    bound = fingerprint_configuration(profile_configuration(profile))
    require(fingerprint_configuration(profile_configuration(worker["profile"])) == bound and
            worker["configuration_fingerprint"] == bound and
            fingerprint_configuration(worker["profile"]) == worker["profile_fingerprint"],
            "worker actual profile differs from independently measured profile")
    sources = {key: instrument[key] for key in WORKER_SOURCE_FILES}
    require(worker["profile"]["owned_source_files"] == sources and
            worker["profile"]["owned_source_fingerprint"] == fingerprint_configuration(sources),
            "worker runtime source profile is stale")
    _within(launch["started_at"], launch["ended_at"], start, end, "worker launch")
    _within(worker["started_at"], worker["ended_at"], launch["started_at"], launch["ended_at"], "worker")
    process = worker["resources"]["process_identity"]
    require(set(process) == {"pid", "process_start_ticks", "process_started_at", "boot_id"},
            "complete process identity required")
    for key in ("pid", "process_start_ticks"):
        require(type(process[key]) is int and process[key] > 0, "invalid process identity")
    text(process["boot_id"], "boot identity")
    # Linux btime is rounded to seconds; permit only that measurement resolution.
    age = (timestamp(process["process_started_at"]) - timestamp(launch["started_at"])).total_seconds()
    require(age >= -1 and timestamp(process["process_started_at"]) <= timestamp(worker["started_at"]),
            "process identity predates launch or postdates worker")
    locked = {r["file"]: r for r in ev.json("input-lock.json")["records"]}
    captures = []
    previous_case_end = worker["started_at"]
    for index, case in enumerate(worker["cases"]):
        require(case["evidence"]["path"] == f"workers/cases/{Path(proof['worker']['path']).stem}-{index:04d}.json",
                "durable case belongs to a different worker stream")
        saved = ev.json(case["evidence"])
        require(case.get("execution") == execution, "durable case execution mismatch")
        _within(case["started_at"], case["ended_at"], previous_case_end, worker["ended_at"], "case")
        previous_case_end = case["ended_at"]
        require(all(saved.get(k) == v for k, v in case.items() if k != "evidence"),
                "durable case differs from manifest")
        for key, expected in (
            ("schema_version", 1), ("session", identity["session"]), ("stage", purpose),
            ("input_fingerprint", identity["input_fingerprint"]), ("status", "complete"),
            ("evidence_kind", worker["evidence_kind"]), ("error", None),
        ):
            require(saved.get(key) == expected, "durable case identity/status mismatch: " + key)
        entry = locked[case["key"]["record"]]
        require(case["record_sha256"] == entry["sha256"] and case["instruction"] == entry["instruction"] and
                case["observer_inert"] is True, "case locked record/instruction/observer mismatch")
        validate_profile(saved["profile"])
        require(fingerprint_configuration(saved["profile"]) == case["profile_fingerprint"] and
                fingerprint_configuration(profile_configuration(saved["profile"])) == bound,
                "case actual profile differs")
        arrays = ev.tensors(case["tensors"])
        for key, shape in (("raw", (1, 40, 132)), ("noise", (1, 40, 132)), ("decoded", (16, 6))):
            require(arrays[key].shape == shape and arrays[key].dtype == np.float32, "full captured " + key + " required")
        pre = {k.removeprefix("preprocessing."): v for k, v in arrays.items() if k.startswith("preprocessing.")}
        collated = {k.removeprefix("collated."): v for k, v in arrays.items() if k.startswith("collated.")}
        require(set(pre) == PREPROCESSING_KEYS and bool(collated), "complete captured model inputs required")
        dtype_names = {k.removeprefix("dtype.") for k in arrays if k.startswith("dtype.")}
        require(dtype_names == set(collated), "captured original input dtypes required")
        for key in collated:
            value = arrays["dtype." + key]
            require(value.shape == () and value.dtype.kind in "iu" and 0 <= int(value) < len(DTYPE_NAMES),
                    "invalid captured original dtype")
        require(pre["tokens"].dtype.kind in "biu" and pre["mask"].dtype.kind in "biu", "categorical input dtype")
        captures.append(arrays)
    result = {"worker": worker, "launch": launch, "captures": captures, "process_id": fingerprint_configuration(process)}
    ev._validated[cache_key] = result
    return result


def repeatability_basis(workspace, measured, record):
    """Derive ALL proposal statistics and aggregates from validated case captures."""
    ev = evidence(workspace)
    groups = record["schedule"]["groups"]
    require(len(measured["workers"]) == len(measured["launches"]) == len(groups), "repeatability workers missing")
    captures, actual_groups, processes, starts, ends = [], [], [], [], []
    previous_end = record["started_at"]
    for group, worker_ref, launch_ref in zip(groups, measured["workers"], measured["launches"], strict=True):
        result = validate_numerical_worker(ev, {"worker": worker_ref, "launch": launch_ref}, measured["profile"],
                                          group["cases"], kind="repeatability",
                                          start=previous_end, end=record["ended_at"])
        worker, launch = result["worker"], result["launch"]
        previous_end = launch["ended_at"]
        starts.append(worker["started_at"])
        ends.append(worker["ended_at"])
        processes.append(result["process_id"])
        actual_groups.append({"id": group["id"], "cases": group["cases"], "process_id": result["process_id"]})
        captures.extend(result["captures"])
    require(len(set(processes)) == len(processes), "cold repeat reused a process identity")
    require(measured["groups"] == actual_groups and measured["started_at"] == min(starts) and
            measured["ended_at"] == max(ends), "repeatability process/chronology claims differ")
    keys = [key for group in groups for key in group["cases"]]
    stacked = {"decoded": np.stack([row["decoded"] for row in captures]),
               "noise": np.concatenate([row["noise"] for row in captures])}
    _same_arrays(ev.tensors(measured["tensors"]), stacked, "repeatability aggregate")
    deltas, raw_max, pre_max = [], 0., 0.
    for record_name in dict.fromkeys(key["record"] for key in keys):
        same = [i for i, key in enumerate(keys) if key["record"] == record_name and key["mode"] != "changed"]
        first = captures[same[0]]
        for i in same[1:]:
            current = captures[i]
            deltas.append(current["decoded"].astype(np.float64) - first["decoded"].astype(np.float64))
            raw_max = max(raw_max, float(np.abs(current["raw"].astype(np.float64) - first["raw"]).max()))
            for key in PREPROCESSING_KEYS:
                a, b = first["preprocessing." + key], current["preprocessing." + key]
                require(a.shape == b.shape and a.dtype == b.dtype, "repeatability preprocessing shape/dtype differs")
                if a.dtype.kind in "biu":
                    require(np.array_equal(a, b), "same-seed categorical preprocessing differs")
                else:
                    pre_max = max(pre_max, float(np.abs(b.astype(np.float64) - a).max()))
    return {"statistics": {k: v.tolist() for k, v in _metrics(np.stack(deltas)).items()},
            "raw_max_abs": raw_max, "preprocessing_max_abs": pre_max}


def validate_bundle(workspace, reference, name, *, start, end, common_from=None):
    ev = evidence(workspace)
    cache_key = ("numerical-bundle", fingerprint_configuration([reference, name, common_from]))
    if cache_key in ev._validated:
        bundle, result = ev._validated[cache_key]
        _within(result["launch"]["started_at"], result["launch"]["ended_at"], start, end, "worker launch")
        return bundle, result
    bundle = ev.json(reference)
    schedule = ev.json("input-lock.json")["schedule"]
    aggregate = _bundle(ev, bundle, name, schedule)
    proof = {"worker": bundle["worker"], "launch": bundle["launch"]}
    result = validate_numerical_worker(ev, proof, name, schedule, kind="replay", start=start, end=end,
                                      common_from=common_from)
    captures = result["captures"]
    for key in ("raw", "noise", "decoded"):
        values = np.stack([row[key][0] if key != "decoded" else row[key] for row in captures])
        require(array_comparison(aggregate[key], values, {}, exact=True), "bundle aggregate differs from cases: " + key)
    require(len(bundle["collated"]) == len(schedule), "bundle collated coverage missing")
    for field, prefix in (("preprocessing", "preprocessing."), ("common_inputs", "preprocessing."),
                          ("independent_collated", "collated."), ("common_collated", "collated.")):
        require(len(bundle[field]) == len(schedule), "bundle input coverage differs")
        checked = set()
        for ref, arrays, case in zip(bundle[field], captures, result["worker"]["cases"], strict=True):
            cache = (ref["sha256"], case["tensors"]["sha256"])
            if cache in checked:
                continue
            checked.add(cache)
            projected = {k.removeprefix(prefix): v for k, v in arrays.items() if k.startswith(prefix)}
            _same_arrays(ev.tensors(ref), projected, "bundle " + field)
    for item, ref, arrays in zip(bundle["collated"], bundle["independent_collated"], captures, strict=True):
        require(item["tensors"] == ref and item["dtypes"] ==
                {k.removeprefix("collated."): DTYPE_NAMES[int(arrays["dtype." + k.removeprefix("collated.")])]
                 for k in arrays if k.startswith("collated.")}, "forwarded common input dtype/reference differs")
    ev._validated[cache_key] = (bundle, result)
    return bundle, result


def validate_comparison(workspace, stored, *, report_start, report_end):
    ev = evidence(workspace)
    _within(stored["started_at"], stored["ended_at"], report_start, report_end, "comparison")
    require(set(stored["provenance"]) == {"left", "right"}, "independent/common worker provenance missing")
    bundles = []
    proofs = stored["provenance"]
    common_source = proofs["left"]["independent"]
    workers = []
    for side, profile in zip(("left", "right"), COMPARISON_PROFILES[stored["name"]], strict=True):
        require(set(proofs[side]) == {"independent", "common"}, "both numerical passes required")
        independent, first = validate_bundle(ev, proofs[side]["independent"], profile,
                                            start=report_start, end=stored["started_at"])
        common, second = validate_bundle(ev, proofs[side]["common"], profile, start=stored["started_at"],
                                         end=stored["ended_at"], common_from=common_source)
        workers.extend([first, second])
        values = ev.tensors(stored[side])
        expected = dict(ev.tensors(common["tensors"]))
        expected["decoded"] = ev.tensors(independent["tensors"])["decoded"]
        _same_arrays(values, expected, "comparison aggregate")
        bundles.append({**common, "tensors": stored[side],
                        "preprocessing": independent["preprocessing"],
                        "independent_collated": independent["independent_collated"]})
    require(len({item["process_id"] for item in workers}) == 4, "numerical passes reused process identity")
    source = ev.json(common_source)
    for bundle in bundles:
        require(bundle["collated"] == source["collated"], "common pass did not forward exact captured model inputs")
    actual = compare_tiers(ev, stored["name"], *bundles, started_at=stored["started_at"],
                           ended_at=stored["ended_at"], matrix_reference=stored["matrix"])
    actual["provenance"] = proofs
    require(actual == stored and actual["passed"], "archived numerical verdict/metrics differ")
    return workers


def validate_offline_evidence(workspace, *, stock_check=None, report=None):
    """Canonical numerical acceptance, used by CLI and physical release.

    A fixture stock collaborator is allowed only on Evidence(test_only=True).
    Physical release always calls this without a collaborator.
    """
    ev = evidence(workspace)
    require(ev.test_only or stock_check is None, "fixture stock check forbidden")
    if report is None:
        report = ev.record("offline-report.json")
    else:
        require(report.get("schema_version") == 1 and report.get("status") == "complete" and
                not report.get("prerequisite_errors") and report.get("evidence_kind") ==
                ("test_only" if ev.test_only else "real_model"), "invalid prospective offline report")
        require(all(report.get(k) == v for k, v in ev.identity().items()), "offline identity mismatch")
        require(timestamp(report["started_at"]) <= timestamp(report["ended_at"]), "offline chronology reversed")
    require(report.get("parity_passed") is True and
            report["agreement"] == ev.reference("tolerance-agreement.json") and
            isinstance(report.get("caveats"), list), "passing archive and exact agreement required")
    require(report["instrument_files"] == instrument_identity(), "offline instrument became stale")
    validate_tolerance_agreement(ev, comparison_started_at=report["started_at"])
    require([c["name"] for c in report["comparisons"]] == list(COMPARISON_PROFILES),
            "all four complete comparisons required")
    if stock_check is None:
        require(report["upstream"] == ev.reference("upstream-result.json"), "exact upstream archive required")
        upstream = validate_upstream_evidence(ev)
        require(timestamp(upstream["ended_at"]) <= timestamp(report["started_at"]), "stock must finish before full replay")
    else:
        stock_check(ev)
    all_workers = {}
    for stored in report["comparisons"]:
        for result in validate_comparison(ev, stored, report_start=report["started_at"], report_end=report["ended_at"]):
            ref = fingerprint_configuration(result["launch"]["manifest"])
            all_workers[ref] = result
    require(len(all_workers) == 12 and len({v["process_id"] for v in all_workers.values()}) == 12,
            "four independent plus eight common-input worker runs required")
    ordered = sorted(all_workers.values(), key=lambda v: timestamp(v["launch"]["started_at"]))
    for before, after in zip(ordered, ordered[1:]):
        require(timestamp(before["launch"]["ended_at"]) <= timestamp(after["launch"]["started_at"]),
                "numerical worker processes overlap")
    require(timestamp(report["ended_at"]) <= timestamp(report["archived_at"]), "archive chronology mismatch")
    return report


def validate_producer_output(output, exit_code):
    require(type(exit_code) is int and exit_code == 0, "producer process failed")
    require(not re.search(r"\[(?:fail|skip)\]|Skipped/failed", output), "producer per-tag failure")
    summaries = re.findall(r"^Dumped (\d+) tags: (.+)$", output, re.MULTILINE)
    require(len(summaries) == 1 and summaries[0][0] == "1" and
            ast.literal_eval(summaries[0][1]) == ["new_embodiment"],
            "producer must actually dump exactly new_embodiment")


def parse_junit(data):
    require(len(data) <= 4 * 1024 * 1024 and b"<!DOCTYPE" not in data and b"<!ENTITY" not in data,
            "invalid/bounded JUnit required")
    root = ElementTree.fromstring(data)
    cases = root.findall(".//testcase")
    require(len(cases) == 1 and cases[0].get("name") == CASE, "missing/extra/uncollected stock case")
    case = cases[0]
    require(not any(case.find(name) is not None for name in ("skipped", "failure", "error")),
            "stock skip/xfail/failure/error is not passing coverage")
    for prop in case.findall(".//property"):
        require(prop.get("value") not in ("xpassed", "xfailed", "skipped"),
                "unexpected stock pytest outcome")
    for suite in root.iter("testsuite"):
        for count in ("failures", "errors", "skipped"):
            require(int(suite.get(count, 0)) == 0, "unsuccessful stock suite")
    return [{"name": "new_embodiment", "outcome": "passed"}]


def compare_raw(workspace, left, right, bounds):
    ev = evidence(workspace)
    la, ra = ev.tensors(left), ev.tensors(right)
    require(set(la) == set(ra) == {"raw"}, "stock raw witness required")
    require(la["raw"].shape == ra["raw"].shape == (2, 40, 132), "stock full pre-crop shape mismatch")
    require(la["raw"].dtype == ra["raw"].dtype == np.float32, "stock raw storage must be fp32")
    require(array_comparison(la["raw"], ra["raw"], bounds), "stock raw threshold failed")


def validate_observation(ev, reference, profile):
    value = ev.json(reference)
    require(value["status"] == "complete" and value["evidence_kind"] ==
            ("test_only" if ev.test_only else "real_model"), "stock observation incomplete")
    require(value["backend"] == profile["backend"] and value["image_digest"] == profile["image_digest"],
            "stock observation environment mismatch")
    require(value["checkpoint_fingerprint"] == ev.json("input-lock.json")["checkpoint_fingerprint"],
            "stock observation checkpoint mismatch")
    for key in ("source", "packages", "backbone_fingerprint"):
        require(value[key] == profile[key], "stock source/package/backbone identity mismatch")
    measured = value["observed"]
    validate_tf32_controls(measured)
    for key in ("parameter_dtypes", "input_dtypes", "backbone_dtypes", "compute_dtypes"):
        require(measured[key] == ["torch.float32"], f"stock {key} is not actual fp32")
    require(all(v == "torch.float32" for v in measured["buffer_dtypes"]), "stock buffer precision")
    for key, expected in (
        ("flow_steps", 4), ("raw_shape", [2, 40, 132]), ("noise_shape", [2, 40, 132]),
        ("noise_dtype", "torch.float32"), ("raw_dtype", "torch.float32"),
        ("noise_draws", 1), ("autocast", False), ("tf32", False),
        ("tf32_matmul", False), ("tf32_cudnn", False),
        ("eval", True), ("observer_inert", True), ("attention", ["sdpa"]),
    ):
        require(measured[key] == expected, f"stock effective {key} mismatch")
    require(measured["sdpa_calls"] > 0, "stock SDPA was not observed")
    require(measured["device"] == profile["device"], "stock device differs from approved diagnostic")
    return value


def validate_launches(result, profiles, bounds):
    require([item["stage"] for item in result["launches"]] == ["producer", "consumer"],
            "both stock processes must actually execute")
    for launch, backend in zip(result["launches"], ("native", "lerobot"), strict=True):
        argv = launch["argv"]
        require(argv[:2] == ["docker", "run"] and launch["exit_code"] == 0, "stock process invocation mismatch")
        env_rows = [argv[i + 1] for i, arg in enumerate(argv) if arg == "--env"]
        env = dict(item.split("=", 1) for item in env_rows)
        require(len(env) == len(env_rows), "duplicate stock environment override")
        for key, value in {
            "GROOT_N1_7_PARITY_DIR": "/evidence/stock/producer",
            "GROOT_N1_7_LIBERO_CKPT": "/inputs/checkpoint",
            "GROOT_PARITY_DEVICE": profiles[backend]["device"],
            "GROOT_PARITY_ATOL": str(bounds["atol"]), "GROOT_PARITY_RTOL": str(bounds["rtol"]),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        }.items():
            require(env.get(key) == value, f"stock unapproved environment: {key}")
        require(argv[argv.index("--entrypoint") + 2] == profiles[backend]["image_digest"],
                "stock image differs from measured profile")
        require(argv[argv.index("--stage") + 1] == launch["stage"], "stock stage invocation mismatch")
        require(argv[argv.index("--checkpoint") + 1] == "/inputs/checkpoint", "stock checkpoint override")
        require(timestamp(result["started_at"]) <= timestamp(launch["started_at"]) <=
                timestamp(launch["ended_at"]) <= timestamp(result["ended_at"]), "stock launch chronology")
    require(timestamp(result["launches"][0]["ended_at"]) <= timestamp(result["launches"][1]["started_at"]),
            "stock processes were not serial")


def validate_upstream_evidence(workspace, *, result=None):
    ev = evidence(workspace)
    assert_instrument(ev)
    if result is None:
        result = ev.record("upstream-result.json")
    else:
        require(result.get("status") == "complete" and not result.get("prerequisite_errors") and
                result.get("schema_version") == 1 and result.get("evidence_kind") ==
                ("test_only" if ev.test_only else "real_model"), "invalid prospective stock result")
        require(all(result.get(k) == v for k, v in ev.identity().items()), "stock result identity changed")
        require(timestamp(result["started_at"]) <= timestamp(result["ended_at"]), "stock chronology reversed")
    validate_tolerance_agreement(ev, comparison_started_at=result["started_at"])
    proposal = validate_tolerance_proposal(ev)
    require(result["harness"] == proposal["harness"], "stock source pins differ from agreement")
    require(result["agreement"] == ev.reference("tolerance-agreement.json"), "stock agreement changed")
    require(result["seed"] == 42 and result["tag"] == "new_embodiment", "stock scope changed")
    require(result["checkpoint_fingerprint"] == ev.json("input-lock.json")["checkpoint_fingerprint"],
            "stock checkpoint mismatch")
    validate_producer_output(ev.bytes(result["producer_log"]["path"], result["producer_log"]["sha256"]).decode(),
                             result["producer_exit"])
    require(type(result["consumer_exit"]) is int and result["consumer_exit"] == 0, "stock consumer failed")
    ev.bytes(result["consumer_log"]["path"], result["consumer_log"]["sha256"])
    require(parse_junit(ev.bytes(result["junit"]["path"], result["junit"]["sha256"])) == result["tests"],
            "stock coverage differs from JUnit")
    coverage = ev.json(result["coverage"])
    expected_node = CONSUMER + "::" + CASE
    require(coverage["collected"] == [expected_node] and coverage["reports"] == [
        {"nodeid": expected_node, "when": phase, "outcome": "passed", "wasxfail": None}
        for phase in ("setup", "call", "teardown")
    ], "stock collection or runtime coverage incomplete/xfail/xpass")
    profiles = {p["backend"]: p["observed"] for p in ev.json("profiles.json")["profiles"]
                if p["purpose"] == "diagnostic"}
    validate_launches(result, profiles, proposal["comparisons"]["diagnostic"]["thresholds"]["raw"])
    left = validate_observation(ev, result["producer_observation"], profiles["native"])
    right = validate_observation(ev, result["consumer_observation"], profiles["lerobot"])
    require(left["observed"]["rng_algorithm"] == right["observed"]["rng_algorithm"],
            "stock RNG algorithms differ")
    for field in ("inputs", "noise"):
        la, ra = ev.tensors(left[field]), ev.tensors(right[field])
        require(set(la) == set(ra) and bool(la), f"stock {field} keys differ")
        for key in la:
            require(array_comparison(la[key], ra[key], {}, exact=True), f"stock actual {field} differ")
    require(result["left"] == left["raw"] and result["right"] == right["raw"], "stock boundary references changed")
    compare_raw(ev, result["left"], result["right"],
                proposal["comparisons"]["diagnostic"]["thresholds"]["raw"])
    ev.bytes(result["artifact"]["path"], result["artifact"]["sha256"])
    return result
