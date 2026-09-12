"""Bounded native regression after the physical trials; no hardware or approvals."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from uuid import UUID, uuid4

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard import instrumentation_transition as transition
from policy_guard import milestone_golden as golden
from policy_guard import parity_gate as gate
from policy_guard.replay_contract import (
    ReplayManifest, configuration_value, execute_cases,
    fingerprint_configuration as digest, load_case, load_input_lock, now,
    profile_configuration, runtime_identity, sha256_file, trace_prediction,
    validate_replay_manifest, write_evidence, write_tensors,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "scripts/replay_milestone_final.py"
OUTPUT = "final-regression.json"
SUMMARY = "physical-trials.json"
SCHEDULE = "milestone-final/schedule.json"
LAUNCH = "milestone-final/launch.json"
LOG = "milestone-final/worker.log"
WORKER = "workers/native-operational-milestone-final.json"
KIND = "milestone_native_final_regression"


def source_identity():
    return {**golden.source_identity(), SCRIPT: sha256_file(ROOT / SCRIPT)}


def physical_basis(ev):
    """Bind chronology to actual run bytes; the parent validates trial outcomes."""
    summary = ev.json(SUMMARY)
    gate.require(summary["kind"] == "physical_trials_summary" and summary["status"] == "complete"
                 and summary["trial_count"] == 3, "completed three-trial summary required")
    rows = summary["trial_records"]
    gate.require([r["trial"] for r in rows] == [1, 2, 3], "three ordered physical trials required")
    previous = None
    runs = []
    for row in rows:
        old = gate.Evidence(Path(row["workspace"]))
        gate.require(row["live_run"]["path"] == "live-run.json", "exact physical run reference required")
        run = old.json(row["live_run"])
        gate.require(run["ended_at"] == row["recorded_end"] and
                     gate.timestamp(run["started_at"]) <= gate.timestamp(run["ended_at"]),
                     "physical run chronology differs from summary")
        if previous is not None:
            gate.require(gate.timestamp(previous) < gate.timestamp(run["started_at"]),
                         "physical trials overlap or are reordered")
        previous = run["ended_at"]
        runs.append({"workspace": str(old.workspace), "live_run": row["live_run"],
                     "ended_at": previous})
    gate.require(gate.timestamp(previous) <= gate.timestamp(summary["recorded_at"]),
                 "physical summary predates last trial")
    return {"summary": ev.reference(SUMMARY), "live_runs": runs, "last_physical_ended_at": previous}


def context(workspace):
    ev = gate.evidence(workspace)
    candidate = golden.validate_candidate(ev)
    expected = transition.serving_profile(ev, candidate["profile"])
    gate.validate_profile(expected)
    gate.require(expected["backend"] == "native" and expected["purpose"] == "operational"
                 and expected["device"] == "cuda:0"
                 and expected["parameter_dtypes"] == expected["backbone_dtypes"] == ["torch.bfloat16"],
                 "native operational CUDA BF16 profile required")
    return ev, candidate, expected, physical_basis(ev)


def worker_command(workspace, inputs, container, image):
    from scripts.replay_checkpoint_parity import worker_argv
    args = SimpleNamespace(workspace=Path(workspace), corpus=Path(inputs["corpus"]),
                           checkpoint=Path(inputs["checkpoint"]), native_cache=Path(inputs["native_cache"]),
                           device="cuda:0", numerical=False)
    argv = worker_argv(args, "native", "operational", image, WORKER, container, SCHEDULE)
    index = argv.index("/replay/scripts/replay_groot_native.py")
    argv[index:index + 1] = ["/replay/" + SCRIPT, "_worker"]
    return argv


def execution_binding(ev, plan):
    gate.require(plan["schema_version"] == 1 and plan["kind"] == KIND
                 and plan["worker_manifest"] == WORKER and UUID(plan["id"]).hex == plan["id"],
                 "invalid final execution namespace")
    return {"id": plan["id"], "collection": KIND, "started_at": plan["started_at"],
            "worker_manifest": WORKER, "schedule": ev.reference(SCHEDULE)}


def validate_plan(ev, plan, candidate, expected, physical):
    cases = golden.scope_cases(ev.json("input-lock.json"), ev.json("milestone-scope.json"))
    for key, value in (
        ("candidate", ev.reference(golden.CANDIDATE)), ("scope", candidate["scope"]),
        ("cases", cases), ("source_files", source_identity()), ("expected_profile", expected),
        ("physical", physical), ("bounds", golden.BOUNDS),
    ):
        gate.require(plan[key] == value, "final schedule changed: " + key)
    gate.require(gate.timestamp(candidate["ended_at"]) < gate.timestamp(plan["started_at"]) and
                 gate.timestamp(physical["last_physical_ended_at"]) < gate.timestamp(plan["started_at"]),
                 "final replay must follow candidate and all physical trials")
    return execution_binding(ev, plan)


def validate_worker(ev, candidate, expected, physical, launch):
    gate.require(launch["schedule"] == ev.reference(SCHEDULE), "wrong final schedule reference")
    plan = ev.json(launch["schedule"])
    binding = validate_plan(ev, plan, candidate, expected, physical)
    gate.require(launch["status"] == "complete" and type(launch["exit_code"]) is int
                 and launch["exit_code"] == 0 and launch["manifest"] == ev.reference(WORKER),
                 "final worker launch failed or changed")
    gate.require(launch["source_files"] == source_identity()
                 and launch["argv"] == worker_command(ev.workspace, launch["inputs"],
                                                      launch["container"], expected["image_digest"]),
                 "final launch source or actual argv differs")
    gate.require(launch["log"]["path"] == LOG, "wrong final log path")
    ev.bytes(launch["log"]["path"], launch["log"]["sha256"])
    report = ev.json(launch["manifest"])
    validate_replay_manifest(
        report, ev.workspace, plan["cases"], expected_session=ev.identity()["session"],
        input_lock=ev.json("input-lock.json"), backend="native", purpose="operational",
        checkpoint_fingerprint=expected["checkpoint_fingerprint"],
        image_digest=expected["image_digest"], source_files=expected["owned_source_files"], device="cuda:0",
    )
    gate.require(profile_configuration(report["profile"]) == profile_configuration(expected),
                 "actual final native profile differs")
    gate.require(report["execution"] == binding and
                 report["resources"]["source_files"] == source_identity(),
                 "final worker source/execution binding differs")
    process = report["resources"]["process_identity"]
    gate.require(set(process) == {"pid", "process_start_ticks", "process_started_at", "boot_id"}
                 and type(process["pid"]) is int and process["pid"] > 0
                 and type(process["process_start_ticks"]) is int and process["process_start_ticks"] > 0
                 and isinstance(process["boot_id"], str) and bool(process["boot_id"]),
                 "complete final process identity required")
    gate.require(-1 <= (gate.timestamp(process["process_started_at"]) -
                       gate.timestamp(launch["started_at"])).total_seconds()
                 and gate.timestamp(process["process_started_at"]) <= gate.timestamp(report["started_at"]),
                 "process identity predates final launch or postdates worker")
    gate.require(gate.timestamp(plan["started_at"]) <= gate.timestamp(launch["started_at"]) <=
                 gate.timestamp(report["started_at"]) <= gate.timestamp(report["ended_at"]) <=
                 gate.timestamp(launch["ended_at"]), "final launch chronology invalid")
    previous = report["started_at"]
    for index, row in enumerate(report["cases"]):
        gate.require(row["evidence"]["path"] ==
                     f"workers/cases/{Path(WORKER).stem}-{index:04d}.json"
                     and row["execution"] == binding and row["observer_inert"] is True,
                     "wrong final case provenance or failed observer control")
        gate.require(gate.timestamp(previous) <= gate.timestamp(row["started_at"]) <=
                     gate.timestamp(row["ended_at"]) <= gate.timestamp(report["ended_at"]),
                     "final case chronology invalid")
        previous = row["ended_at"]
    gate.require(report["resources"]["prediction_calls"] == 24 and
                 report["resources"]["observer_control_calls"] == 12,
                 "twelve unchanged trace/control pairs required")
    arrays = golden.tensors_from_cases(ev, report["cases"])
    golden.compare_arrays(arrays, arrays)
    return report, arrays


def validate(workspace):
    """Final regression evidence only; does not aggregate or approve physical trials."""
    ev, candidate, expected, physical = context(workspace)
    record = ev.json(OUTPUT)
    for key, value in (
        ("schema_version", 1), ("kind", KIND), ("status", "complete"), ("evidence_kind", "real_model"),
        ("candidate", ev.reference(golden.CANDIDATE)), ("scope", candidate["scope"]),
        ("cases", candidate["cases"]), ("case_count", 12), ("bounds", golden.BOUNDS),
        ("physical", physical), ("source_files", source_identity()), ("prediction_calls", 24),
        ("observer_control_calls", 12), ("launch", ev.reference(LAUNCH)),
    ):
        gate.require(record[key] == value, "final regression changed or failed: " + key)
    launch = ev.json(record["launch"])
    _, arrays = validate_worker(ev, candidate, expected, physical, launch)
    from scripts.replay_milestone_golden import current_inputs
    current = current_inputs(SimpleNamespace(
        workspace=ev.workspace, **{k: Path(launch["inputs"][k])
                                  for k in ("corpus", "checkpoint", "native_cache")}), expected)
    gate.require(current == launch["inputs"], "current final inputs/cache changed")
    saved = ev.tensors(record["tensors"])
    golden.compare_arrays(saved, saved)
    gate.require(all(np.array_equal(saved[k], arrays[k]) for k in golden.FIELDS),
                 "final tensor archive differs from fresh worker")
    comparison = golden.compare_arrays(ev.tensors(candidate["tensors"]), arrays)
    gate.require(comparison["passed"] and record["comparison"] == comparison,
                 "final native regression failed")
    gate.require(record["started_at"] == ev.json(SCHEDULE)["started_at"] and
                 gate.timestamp(launch["ended_at"]) <= gate.timestamp(record["ended_at"]),
                 "final report chronology invalid")
    return {"status": "complete", "final_regression": ev.reference(OUTPUT),
            "candidate": ev.reference(golden.CANDIDATE), "physical_summary": physical["summary"],
            "last_physical_ended_at": physical["last_physical_ended_at"], "ended_at": record["ended_at"]}


def worker(args):
    """One GPU model load, exactly twelve unmodified trace_prediction calls."""
    ev = gate.Evidence(args.workspace)
    gate.require(not (ev.workspace / WORKER).exists(), "immutable final worker exists")
    lock, plan = ev.json("input-lock.json"), ev.json(SCHEDULE)
    cases = golden.scope_cases(lock, ev.json("milestone-scope.json"))
    candidate = ev.json(plan["candidate"])
    expected = plan["expected_profile"]
    gate.require(args.input_lock.resolve() == (ev.workspace / "input-lock.json").resolve()
                 and args.schedule.resolve() == (ev.workspace / SCHEDULE).resolve()
                 and args.output_manifest == WORKER and args.profile == "operational"
                 and args.device == "cuda:0", "wrong final worker invocation")
    gate.require(plan["cases"] == cases and plan["candidate"] == ev.reference(golden.CANDIDATE)
                 and plan["scope"] == candidate["scope"] and plan["source_files"] == source_identity()
                 and args.image_digest == expected["image_digest"] == candidate["profile"]["image_digest"],
                 "final worker schedule/candidate/source mismatch")
    gate.require(plan["physical"]["summary"] == ev.reference(SUMMARY) and
                 gate.timestamp(plan["physical"]["last_physical_ended_at"]) < gate.timestamp(plan["started_at"]),
                 "final worker must follow physical trials")
    report = ReplayManifest(session=ev.identity()["session"], stage="operational",
                            input_fingerprint=lock["fingerprint"], expected_cases=cases)
    report.execution = execution_binding(ev, plan)
    report.resources.update(source_files=source_identity(), process_identity=gate.process_identity(),
                            prediction_calls=None, observer_control_calls=None)
    try:
        gate.require(load_input_lock(args.corpus, args.checkpoint) == lock, "current worker inputs changed")
        import torch
        from scripts.replay_groot_native import NativeReplay
        gate.require(torch.cuda.is_available(), "CUDA unavailable")
        torch.set_num_threads(4)
        adapter = NativeReplay(args.checkpoint, "operational", "cuda:0")
        identity = runtime_identity("native", args.image_digest, lock)
        effective = configuration_value(adapter.effective_configuration())
        identity.update(effective_configuration=effective, effective_configuration_fingerprint=digest(effective))

        def trace(key):
            arrays, entry = load_case(args.corpus, lock, key)
            profile, tensors = trace_prediction(adapter, arrays, entry, key["seed"], identity)
            if profile_configuration(profile) != profile_configuration(expected):
                write_evidence(ev.workspace, "milestone-final/rejected-profile.json", profile)
                raise ValueError("native final operational profile changed")
            return profile, tensors, entry

        execute_cases(report, ev.workspace, lock, cases, trace, Path(WORKER).stem)
        gate.require(source_identity() == plan["source_files"], "source changed during final replay")
        report.resources.update(prediction_calls=24, observer_control_calls=12)
        report.status = "complete"
    except Exception as exc:
        report.status = "failed"
        report.prerequisite_errors.append({"type": type(exc).__name__, "message": str(exc)})
    report.resources["completed_trace_pairs"] = len(report.executed_cases)
    report.ended_at = now()
    write_evidence(ev.workspace, WORKER, asdict(report))
    return 0 if report.status == "complete" else 1


def replay(args):
    from scripts.replay_checkpoint_parity import command_output, device_snapshot
    from scripts.replay_milestone_golden import current_inputs
    ev, candidate, expected, physical = context(args.workspace)
    gate.require(not any((ev.workspace / path).exists() for path in (OUTPUT, SCHEDULE, LAUNCH, WORKER, LOG)),
                 "immutable final attempt exists; preserve it")
    inputs = current_inputs(args, expected)
    image = expected["image_digest"]
    gate.require(command_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"]) == image,
                 "pinned native image missing")
    gate.require(not device_snapshot()["compute_processes"], "GPU already owned")
    started, sources = now(), source_identity()
    plan = {"schema_version": 1, "kind": KIND, "id": uuid4().hex, "started_at": started,
            "worker_manifest": WORKER, "candidate": ev.reference(golden.CANDIDATE),
            "scope": candidate["scope"], "cases": candidate["cases"], "source_files": sources,
            "expected_profile": expected, "physical": physical, "bounds": golden.BOUNDS}
    schedule = write_evidence(ev.workspace, SCHEDULE, plan)
    validate_plan(ev, plan, candidate, expected, physical)
    container = "dume-milestone-native-final-" + str(os.getpid())
    argv = worker_command(ev.workspace, inputs, container, image)
    launch = {"started_at": now(), "container": container, "argv": argv, "inputs": inputs,
              "source_files": sources, "schedule": schedule, "status": "failed"}
    record = {"schema_version": 1, "kind": KIND, "status": "failed", "evidence_kind": "real_model",
              "started_at": started, "candidate": ev.reference(golden.CANDIDATE), "scope": candidate["scope"],
              "cases": candidate["cases"], "case_count": 12, "physical": physical, "source_files": sources,
              "bounds": golden.BOUNDS, "expected_prediction_calls": 24, "prediction_calls": None,
              "observer_control_calls": None, "errors": []}
    process = None
    old_term = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise KeyboardInterrupt("final replay coordinator received SIGTERM")

    signal.signal(signal.SIGTERM, terminate)
    try:
        with (ev.workspace / LOG).open("x") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            try:
                launch["exit_code"] = process.wait(timeout=args.worker_timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                subprocess.run(["docker", "stop", "--time", "5", container], capture_output=True, timeout=20)
                process.wait(timeout=20)
                raise
        launch["ended_at"] = now()
        launch["log"] = ev.reference(LOG)
        if (ev.workspace / WORKER).exists():
            launch["manifest"] = ev.reference(WORKER)
            launch["status"] = "complete" if launch["exit_code"] == 0 and ev.json(WORKER)["status"] == "complete" else "failed"
        record["launch"] = write_evidence(ev.workspace, LAUNCH, launch)
        gate.require(source_identity() == sources and current_inputs(args, expected) == inputs
                     and physical_basis(gate.Evidence(ev.workspace)) == physical,
                     "sources, inputs or physical evidence changed during final replay")
        _, arrays = validate_worker(ev, candidate, expected, physical, launch)
        record.update(tensors=write_tensors(ev.workspace, arrays), prediction_calls=24, observer_control_calls=12)
        record["comparison"] = golden.compare_arrays(ev.tensors(candidate["tensors"]), arrays)
        record["status"] = "complete" if record["comparison"]["passed"] else "failed"
    except (Exception, KeyboardInterrupt) as exc:
        record["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        if not (ev.workspace / LAUNCH).exists():
            launch.update(ended_at=now(), exit_code=process.returncode if process else None)
            if (ev.workspace / LOG).exists():
                launch["log"] = ev.reference(LOG)
            record["launch"] = write_evidence(ev.workspace, LAUNCH, launch)
    finally:
        signal.signal(signal.SIGTERM, old_term)
    record["ended_at"] = now()
    write_evidence(ev.workspace, OUTPUT, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("replay", "check", "_worker"):
        sub = commands.add_parser(name)
        sub.add_argument("--workspace", type=Path, required=True)
        if name in ("replay", "_worker"):
            sub.add_argument("--corpus", type=Path, default=ROOT / "corpus/frozen_v1_0")
            sub.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/GR00T-N1.7-3B-SO101")
        if name == "replay":
            sub.add_argument("--native-cache", type=Path, default=Path.home() / ".cache/huggingface")
            sub.add_argument("--worker-timeout", type=int, default=1800)
        if name == "_worker":
            sub.add_argument("--input-lock", type=Path, required=True)
            sub.add_argument("--schedule", type=Path, required=True)
            sub.add_argument("--output-manifest", required=True)
            sub.add_argument("--image-digest", required=True)
            sub.add_argument("--profile", required=True)
            sub.add_argument("--device", required=True)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        return worker(args)
    result = replay(args) if args.command == "replay" else validate(args.workspace)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
