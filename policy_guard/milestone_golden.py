"""Validate the twelve-case native reference without approving or promoting it."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from policy_guard import parity_gate as gate
from policy_guard import instrumentation_transition as transition
from policy_guard.milestone_acceptance import validate_milestone_acceptance
from policy_guard.replay_contract import (
    fingerprint_configuration as digest, now, profile_configuration,
    sha256_file, validate_replay_manifest, write_evidence, write_tensors,
)

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "milestone-golden-candidate.json"
REPLAY = "milestone-golden-replay.json"
SCHEDULE = "milestone-golden/schedule.json"
WORKER = "workers/native-operational-milestone-golden.json"
LAUNCH = "milestone-golden/launch.json"
BOUNDS = {"atol": 1e-5, "rtol": 0}
FIELDS = {"raw": (12, 40, 132), "noise": (12, 40, 132), "decoded": (12, 16, 6)}


def source_identity():
    # The parent separately binds release-guard integration in the live review.
    # Producer/capture sources, comparison helpers and this new implementation
    # remain byte-bound here; changing a threshold is never an integration edit.
    names = set(gate.INSTRUMENT_FILES) - {"policy_guard/parity_gate.py"}
    names.update(("policy_guard/milestone_golden.py", "scripts/replay_milestone_golden.py",
                  "policy_guard/golden.py", "scripts/replay_native_golden.py",
                  "policy_guard/milestone_acceptance.py"))
    return {name: sha256_file(ROOT / name) for name in sorted(names)}


def scope_cases(lock, scope):
    groups = {}
    for row in lock["records"]:
        groups.setdefault(row["episode_index"], []).append(row)
    gate.require(len(groups) == 12 and all(len(rows) == 10 for rows in groups.values()),
                 "twelve ten-frame episodes required")
    expected = [{"record": rows[5]["file"], "seed": rows[5]["seeds"][0]}
                for rows in groups.values()]
    actual = [{k: row[k] for k in ("record", "seed")} for row in scope["selected"]]
    gate.require(actual == expected and len({digest(k) for k in actual}) == 12,
                 "frozen middle-frame/first-seed selection changed")
    return actual


def context(workspace):
    ev = gate.evidence(workspace)
    accepted = validate_milestone_acceptance(ev)
    scope = ev.json("milestone-scope.json")
    gate.require(scope["input_lock"] == ev.reference("input-lock.json"), "input lock changed")
    bounds = ev.json("tolerance-proposal.json")["golden"]
    gate.require(bounds["atol"] == BOUNDS["atol"] and bounds["rtol"] == 0
                 and bounds.get("noise_policy", "exact") == "exact", "native bounds changed")
    cases = scope_cases(ev.json("input-lock.json"), scope)
    return ev, accepted, scope, cases


def tensors_from_cases(ev, cases):
    captures = [ev.tensors(row["tensors"]) for row in cases]
    return {name: (np.stack([a[name] for a in captures]) if name == "decoded"
                   else np.concatenate([a[name] for a in captures])) for name in FIELDS}


def compare_arrays(left, right):
    gate.require(set(left) == set(right) == set(FIELDS), "complete native archive required")
    results = {}
    for name, shape in FIELDS.items():
        a, b = left[name], right[name]
        gate.require(a.shape == b.shape == shape and a.dtype == b.dtype == np.float32,
                     "full twelve-case float32 " + name + " required")
        passed = gate.array_comparison(a, b, BOUNDS, exact=name == "noise")
        results[name] = {"passed": passed,
                         "max_abs": float(np.abs(b.astype(np.float64) - a.astype(np.float64)).max())}
    return {"passed": all(r["passed"] for r in results.values()),
            "bounds": BOUNDS, "noise_policy": "exact", "tensors": results}


def captured_native(ev, scope, cases):
    # Host-only: the legacy golden module contains syntax unavailable in the
    # pinned native Python 3.10 image. Fresh workers never import it.
    from policy_guard.golden import validate_native_profile

    # Acceptance already revalidates the immutable original GPU captures and
    # source identities. Keep the selected durable case references in the import.
    proof = scope["sources"]["native-operational"]
    worker, launch = ev.json(proof["worker"]), ev.json(proof["launch"])
    gate.require(worker["expected_cases"] == worker["executed_cases"] ==
                 ev.json("input-lock.json")["schedule"], "original worker was incomplete")
    validate_native_profile(worker["profile"])
    rows = [worker["cases"][worker["executed_cases"].index(key)] for key in cases]
    for key, row in zip(cases, rows, strict=True):
        saved = ev.json(row["evidence"])
        gate.require(row["key"] == key and saved["key"] == key
                     and saved["status"] == "complete", "source case differs")
    arrays = tensors_from_cases(ev, rows)
    compare_arrays(arrays, arrays)
    return proof, worker, launch, rows, arrays


def create_candidate(workspace):
    ev, accepted, scope, cases = context(workspace)
    gate.require(not (ev.workspace / CANDIDATE).exists(), "immutable candidate already exists")
    started = now()
    proof, worker, launch, rows, arrays = captured_native(ev, scope, cases)
    record = {
        "schema_version": 1, "kind": "milestone_native_golden_candidate",
        "status": "complete", "evidence_kind": "real_model_import",
        "started_at": started, "ended_at": now(), "scope": ev.reference("milestone-scope.json"),
        "acceptance": accepted["report"], "agreement": ev.reference("tolerance-agreement.json"),
        "cases": cases, "case_count": 12, "bounds": BOUNDS,
        "source": proof, "source_cases": [r["evidence"] for r in rows],
        "captured_at": {"started_at": launch["started_at"], "ended_at": launch["ended_at"]},
        "profile": worker["profile"], "tensors": write_tensors(ev.workspace, arrays),
        "source_files": source_identity(), "calibration_sha256": accepted["calibration_sha256"],
        "inference_runs_added": 0, "approved": False, "promoted": False,
        "review_required": "Final explicit live review must bind this native reference and the physical test",
    }
    return write_evidence(ev.workspace, CANDIDATE, record)


def validate_candidate(workspace):
    ev, accepted, scope, cases = context(workspace)
    record = ev.json(CANDIDATE)
    gate.require(record["kind"] == "milestone_native_golden_candidate" and
                 record["status"] == "complete" and record["evidence_kind"] == "real_model_import",
                 "actual imported native candidate required")
    for key, value in (("scope", ev.reference("milestone-scope.json")),
                       ("acceptance", accepted["report"]),
                       ("agreement", ev.reference("tolerance-agreement.json")),
                       ("cases", cases), ("case_count", 12), ("bounds", BOUNDS),
                       ("source_files", transition.historical_sources(ev, source_identity())),
                       ("calibration_sha256", accepted["calibration_sha256"]),
                       ("inference_runs_added", 0), ("approved", False), ("promoted", False)):
        gate.require(record[key] == value, "candidate changed: " + key)
    proof, worker, launch, rows, arrays = captured_native(ev, scope, cases)
    gate.require(record["source"] == proof and record["profile"] == worker["profile"] and
                 record["source_cases"] == [r["evidence"] for r in rows] and
                 record["captured_at"] == {k: launch[k] for k in ("started_at", "ended_at")},
                 "candidate import provenance differs")
    archived = ev.tensors(record["tensors"])
    compare_arrays(archived, archived)
    gate.require(all(np.array_equal(archived[k], arrays[k]) for k in FIELDS), "candidate is not captured native output")
    gate.require(gate.timestamp(launch["ended_at"]) < gate.timestamp(record["started_at"]) <=
                 gate.timestamp(record["ended_at"]), "candidate import chronology invalid")
    return record


def execution_binding(ev, schedule):
    gate.require(schedule["kind"] == "milestone_native_golden_replay"
                 and schedule["worker_manifest"] == WORKER, "wrong scoped worker schedule")
    from uuid import UUID
    gate.require(UUID(schedule["id"]).hex == schedule["id"], "invalid execution id")
    return {"id": schedule["id"], "collection": schedule["kind"],
            "started_at": schedule["started_at"], "worker_manifest": WORKER,
            "schedule": ev.reference(SCHEDULE)}


def validate_worker(ev, candidate, launch):
    schedule = ev.json(launch["schedule"])
    cases = scope_cases(ev.json("input-lock.json"), ev.json(candidate["scope"]))
    gate.require(schedule["cases"] == cases and schedule["candidate"] == ev.reference(CANDIDATE)
                 and schedule["source_files"] == transition.historical_sources(ev, source_identity())
                 and schedule["scope"] == candidate["scope"], "scheduled candidate/scope/source changed")
    gate.require(launch["status"] == "complete" and type(launch["exit_code"]) is int
                 and launch["exit_code"] == 0 and launch["manifest"] == ev.reference(WORKER)
                 and launch["schedule"] == ev.reference(SCHEDULE), "worker launch failed or changed")
    ev.bytes(launch["log"]["path"], launch["log"]["sha256"])
    gate.require(launch["source_files"] == transition.historical_sources(ev, source_identity()), "launch sources changed")
    # Reconstruct the only permitted argv from bound host paths and pinned image.
    from scripts.replay_milestone_golden import worker_command
    gate.require(launch["argv"] == worker_command(transition.historical_workspace(ev), launch["inputs"], launch["container"],
                                                  candidate["profile"]["image_digest"]),
                 "worker invocation differs")
    worker = ev.json(launch["manifest"])
    expected = candidate["profile"]
    validate_replay_manifest(
        worker, ev.workspace, cases, expected_session=ev.identity()["session"],
        input_lock=ev.json("input-lock.json"), backend="native", purpose="operational",
        checkpoint_fingerprint=expected["checkpoint_fingerprint"],
        image_digest=expected["image_digest"], source_files=expected["owned_source_files"], device="cuda:0",
    )
    gate.require(profile_configuration(worker["profile"]) == profile_configuration(expected),
                 "fresh native operational profile changed")
    binding = execution_binding(ev, schedule)
    gate.require(worker["execution"] == binding and worker["resources"]["source_files"] == transition.historical_sources(ev, source_identity()),
                 "worker execution/source binding differs")
    process = worker["resources"]["process_identity"]
    original = ev.json(candidate["source"]["worker"])["resources"]["process_identity"]
    gate.require(set(process) == set(original) and process != original and
                 type(process["pid"]) is int and process["pid"] > 0 and
                 type(process["process_start_ticks"]) is int and process["process_start_ticks"] > 0
                 and isinstance(process["boot_id"], str) and bool(process["boot_id"]),
                 "fresh worker process identity required")
    gate.require(-1 <= (gate.timestamp(process["process_started_at"]) -
                       gate.timestamp(launch["started_at"])).total_seconds()
                 and gate.timestamp(process["process_started_at"]) <= gate.timestamp(worker["started_at"]),
                 "worker process predates launch or postdates worker")
    gate.require(gate.timestamp(candidate["ended_at"]) <= gate.timestamp(schedule["started_at"]) <=
                 gate.timestamp(launch["started_at"]) <= gate.timestamp(worker["started_at"]) <=
                 gate.timestamp(worker["ended_at"]) <= gate.timestamp(launch["ended_at"]),
                 "fresh replay chronology invalid")
    previous = worker["started_at"]
    for index, row in enumerate(worker["cases"]):
        gate.require(row["evidence"]["path"] ==
                     f"workers/cases/{Path(WORKER).stem}-{index:04d}.json"
                     and row["execution"] == binding, "wrong scoped case provenance")
        gate.require(gate.timestamp(previous) <= gate.timestamp(row["started_at"]) <=
                     gate.timestamp(row["ended_at"]) <= gate.timestamp(worker["ended_at"]),
                     "case chronology invalid")
        previous = row["ended_at"]
    return worker, tensors_from_cases(ev, worker["cases"])


def validate_scoped_golden(workspace):
    """Verify evidence only. The caller must separately obtain explicit review."""
    ev = gate.evidence(workspace)
    candidate = validate_candidate(ev)
    record = ev.json(REPLAY)
    gate.require(record["kind"] == "milestone_native_golden_replay" and
                 record["evidence_kind"] == "real_model" and record["status"] == "complete"
                 and record["approved"] is False and record["promoted"] is False,
                 "completed unpromoted native replay required")
    for key, value in (("candidate", ev.reference(CANDIDATE)), ("scope", candidate["scope"]),
                       ("source_files", transition.historical_sources(ev, source_identity())), ("case_count", 12),
                       ("cases", candidate["cases"]), ("bounds", BOUNDS)):
        gate.require(record[key] == value, "replay changed: " + key)
    launch = ev.json(record["launch"])
    gate.require(record["launch"] == ev.reference(LAUNCH), "wrong native launch reference")
    worker, arrays = validate_worker(ev, candidate, launch)
    from types import SimpleNamespace
    from scripts.replay_milestone_golden import current_inputs
    current = current_inputs(SimpleNamespace(
        workspace=ev.workspace, **{k: Path(launch["inputs"][k])
                                  for k in ("corpus", "checkpoint", "native_cache")}),
        candidate["profile"])
    gate.require(current == launch["inputs"], "current native inputs/cache changed")
    archived = ev.tensors(record["tensors"])
    compare_arrays(archived, archived)
    gate.require(all(np.array_equal(archived[k], arrays[k]) for k in FIELDS),
                 "replay archive differs from actual worker")
    comparison = compare_arrays(ev.tensors(candidate["tensors"]), arrays)
    gate.require(comparison["passed"] and record["comparison"] == comparison,
                 "native golden verification failed")
    gate.require(gate.timestamp(record["started_at"]) <= gate.timestamp(launch["started_at"]) <=
                 gate.timestamp(launch["ended_at"]) <= gate.timestamp(record["ended_at"]),
                 "replay report chronology invalid")
    gate.require(record["prediction_calls"] == 24 and record["observer_control_calls"] == 12,
                 "trace/control workload count changed")
    return {"candidate": ev.reference(CANDIDATE), "replay": ev.reference(REPLAY),
            "ended_at": record["ended_at"], "status": "complete"}
