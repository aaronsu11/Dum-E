"""Local evidence gates. No robot imports, model loading, or operator decisions.

Public consumers accept either a workspace Path or an Evidence snapshot. A fresh
snapshot is required for each physical release. ``test_only=True`` is an explicit
injection seam for hermetic consumers, never a CLI option.

Stage schema (v1), shared with the future numerical/golden/runner producers:
* All new measured stage records bind session, input_fingerprint and
  profiles_fingerprint (canonical digest of profiles.json), with UTC start/end.
* profiles.json retains Plan 01's list of observed profiles; Plan 04 adds the
  measured ``serving_configuration`` from the arm-free attestation.
* repeatability.json adds the exact repeatability_schedule, and four measurements
  with ordered cases, explicit process groups, decoded/noise numeric tensors.
* tolerance-proposal.json defines COMPARISON_PROFILES, thresholds, noise policies,
  six units, trace/aggregate OLS definitions, harness pins and measured basis.
* offline-report.json holds four comparisons. Each has ordered 600 cases, full
  raw/noise/decoded tensor references, independent preprocessing and common-input
  references per case, actual metrics, start/end and the agreement reference.
  Tensor axes are (case, 40, 132) and (case, 16, 6); no shared-prefix slicing.
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

import hashlib
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
    validate_schedule, write_evidence,
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
PREPROCESSING_KEYS = {"image_front", "image_wrist", "state", "tokens", "mask"}
SEMANTIC_KEYS = {
    "backend", "purpose", "checkpoint_fingerprint", "backbone_fingerprint",
    "source_fingerprint", "packages_fingerprint", "image_digest",
    "effective_configuration", "parameter_dtypes", "buffer_dtypes", "compute_dtypes",
    "attention", "flow_steps", "eval", "autocast", "tf32", "device", "seed_policy",
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
    ev._validated["repeatability"] = record
    return record


def validate_tolerance_proposal(workspace):
    ev = evidence(workspace)
    proposal = ev.record("tolerance-proposal.json")
    repeat = validate_repeatability(ev)
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
    names += {
        "tolerances": ["repeatability.json"],
        "golden": ["tolerance-agreement.json"],
        "live": ["offline-report.json", "tolerance-agreement.json", "golden-candidate.json",
                 "golden-approval.json", "golden-replay.json"],
    }[kind]
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


def _comparison(ev, record, proposal, agreement):
    name = record["name"]
    rule = proposal["comparisons"][name]
    require(record["profiles"] == rule["profiles"], "report comparison profiles mismatch")
    require(timestamp(agreement["decided_at"]) < timestamp(record["started_at"]) <= timestamp(record["ended_at"]),
            "comparison chronology must start after agreement")
    _coverage(ev, record)
    require(record["joint_order"] == list(JOINT_ORDER), "report joint ordering mismatch")
    left, right = _tensor_contract(ev, record["left"], 600), _tensor_contract(ev, record["right"], 600)
    _preprocessing(ev, record["preprocessing"], record["cases"], rule["thresholds"]["preprocessing"])
    _preprocessing(ev, record["common_inputs"], record["cases"], rule["thresholds"]["preprocessing"], exact=True)
    _allclose(left["raw"], right["raw"], rule["thresholds"]["raw"], "raw")
    if rule["noise_policy"] == "exact":
        require(np.array_equal(left["noise"], right["noise"]), "actual fp32 sampler noise differs")
    actual = _metrics(right["decoded"].astype(np.float64) - left["decoded"].astype(np.float64))
    require(set(record["metrics"]) == set(actual), "complete signed bias and trend metrics required")
    for key, values in actual.items():
        recorded = np.asarray(record["metrics"][key], dtype=np.float64)
        require(recorded.shape == values.shape and np.array_equal(recorded, values), f"reported metric differs: {key}")
    for key in ("max_abs", "mean_abs", "bias", "slope"):
        require(np.all(np.abs(actual[key]) <= np.asarray(rule["thresholds"]["decoded"][key])), f"decoded {key} threshold failed")
    for key, bound in (("trace_bias_max_abs", "bias"), ("trace_slope_max_abs", "slope")):
        require(np.all(actual[key] <= np.asarray(rule["thresholds"]["decoded"][bound])), f"per-trace {bound} threshold failed")
    require(record.get("passed") is True, "comparison verdict is not passing")


def _upstream(ev, reference, proposal, agreement):
    result = ev.record(reference)
    require(result["agreement"] == ev.reference("tolerance-agreement.json"), "upstream agreement mismatch")
    require(result["harness"] == proposal["harness"], "upstream source pins changed")
    require(timestamp(agreement["decided_at"]) < timestamp(result["started_at"]), "upstream started before agreement")
    require(result["seed"] == 42 and result["tag"] == "new_embodiment", "wrong stock seed/embodiment")
    require(result["checkpoint_fingerprint"] == ev.json("input-lock.json")["checkpoint_fingerprint"], "wrong stock checkpoint")
    require(type(result["producer_exit"]) is int and result["producer_exit"] == 0 and
            type(result["consumer_exit"]) is int and result["consumer_exit"] == 0, "stock harness did not execute successfully")
    require(result["tests"] == [{"name": "new_embodiment", "outcome": "passed"}],
            "stock consumer requires exactly one expected passed case; no skips/xfails")
    left, right = ev.tensors(result["left"]), ev.tensors(result["right"])
    require(set(left) == set(right) == {"raw"} and left["raw"].shape == right["raw"].shape == (2, 40, 132),
            "stock full pre-crop raw boundary required")
    require(left["raw"].dtype == right["raw"].dtype == np.float32, "stock raw witness must remain fp32")
    _allclose(left["raw"], right["raw"], proposal["comparisons"]["diagnostic"]["thresholds"]["raw"], "stock raw")
    return result


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


def validate_release_evidence(workspace):
    ev = evidence(workspace)
    if "release" in ev._validated:
        return ev._validated["release"]
    report = ev.record("offline-report.json")
    agreement = validate_tolerance_agreement(ev, comparison_started_at=report["started_at"])
    proposal = ev.json("tolerance-proposal.json")
    require(report["agreement"] == ev.reference("tolerance-agreement.json"), "report agreement identity mismatch")
    require([c["name"] for c in report["comparisons"]] == list(COMPARISON_PROFILES), "complete ordered tier/bridge comparisons required")
    for comparison in report["comparisons"]:
        _comparison(ev, comparison, proposal, agreement)
        require(timestamp(report["started_at"]) <= timestamp(comparison["started_at"]) <=
                timestamp(comparison["ended_at"]) <= timestamp(report["ended_at"]), "comparison/report chronology mismatch")
    upstream = _upstream(ev, report["upstream"], proposal, agreement)
    require(report["upstream"] == ev.reference("upstream-result.json"), "exact upstream archive required")
    require(timestamp(upstream["ended_at"]) <= timestamp(report["ended_at"]) <= timestamp(report["archived_at"]),
            "report archive chronology mismatch")
    require(report.get("parity_passed") is True and isinstance(report.get("caveats"), list), "passing archived report and caveats required")
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
    require(type(sem["autocast"]) is bool and type(sem["tf32"]) is bool, "actual compute controls required")
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
            require(prior["stage"] == STAGES[STAGES.index(stage) - 1] and prior["status"] == "complete",
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
    require(timestamp(review["ended_at"]) < timestamp(record["decided_at"]), "live approval must follow archive/review")
    return record


def assert_live_release(workspace, preflight, *, expected_stage, runtime,
                        current_calibration_sha256, now=None):
    supplied = evidence(workspace)
    # A caller may have cached an earlier review/check. Physical release always
    # captures current bytes again, while sharing each capture within this call.
    ev = Evidence(supplied.workspace, test_only=supplied.test_only)
    require(expected_stage in STAGES[1:], "review-stage readiness cannot release hardware")
    validate_live_approval(ev)
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


def validate_closeout(workspace):
    ev = evidence(workspace)
    validate_live_approval(ev)
    run = ev.record("live-run.json")
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
