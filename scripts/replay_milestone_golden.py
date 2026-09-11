"""Import, replay and check the scoped native golden; never approve or promote."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard import milestone_golden as golden
from policy_guard import parity_gate as gate
from policy_guard.replay_contract import (
    ReplayManifest, configuration_value, execute_cases, fingerprint_configuration as digest,
    load_case, load_input_lock, now, runtime_identity, trace_prediction,
    write_evidence, write_tensors,
)

ROOT = Path(__file__).resolve().parents[1]


def worker_command(workspace, inputs, container, image):
    from scripts.replay_checkpoint_parity import worker_argv
    args = SimpleNamespace(workspace=Path(workspace), corpus=Path(inputs["corpus"]),
                           checkpoint=Path(inputs["checkpoint"]), native_cache=Path(inputs["native_cache"]),
                           device="cuda:0", numerical=False)
    argv = worker_argv(args, "native", "operational", image, golden.WORKER, container, golden.SCHEDULE)
    position = argv.index("/replay/scripts/replay_groot_native.py")
    argv[position:position + 1] = ["/replay/scripts/replay_milestone_golden.py", "_worker"]
    return argv


def current_inputs(args, expected):
    from scripts.replay_native_golden import native_cache_identity
    from policy_guard.golden import validate_backbone_identity
    lock = load_input_lock(args.corpus, args.checkpoint)
    gate.require(lock == gate.Evidence(args.workspace).json("input-lock.json"), "current inputs changed")
    validate_backbone_identity(native_cache_identity(args.native_cache), expected)
    return {"corpus": str(args.corpus.resolve()), "checkpoint": str(args.checkpoint.resolve()),
            "native_cache": str(args.native_cache.resolve()), "input_fingerprint": lock["fingerprint"]}


def worker(args):
    """One loaded NativeReplay instance; twelve unchanged trace/control pairs."""
    import torch
    from scripts.replay_groot_native import NativeReplay
    ev = gate.Evidence(args.workspace)
    gate.require(not (ev.workspace / golden.WORKER).exists(), "immutable worker exists")
    plan = ev.json(golden.SCHEDULE)
    lock = ev.json("input-lock.json")
    cases = golden.scope_cases(lock, ev.json("milestone-scope.json"))
    gate.require(args.schedule.resolve() == (ev.workspace / golden.SCHEDULE).resolve()
                 and args.input_lock.resolve() == (ev.workspace / "input-lock.json").resolve()
                 and args.output_manifest == golden.WORKER and args.profile == "operational"
                 and args.device == "cuda:0", "wrong native worker invocation")
    gate.require(plan["cases"] == cases and plan["source_files"] == golden.source_identity(),
                 "worker scoped schedule/source changed")
    candidate = ev.json(plan["candidate"])
    gate.require(plan["candidate"] == ev.reference(golden.CANDIDATE) and
                 plan["scope"] == candidate["scope"] and
                 args.image_digest == candidate["profile"]["image_digest"], "worker candidate changed")
    gate.require(load_input_lock(args.corpus, args.checkpoint) == lock, "current worker inputs changed")
    report = ReplayManifest(session=ev.identity()["session"], stage="operational",
                            input_fingerprint=lock["fingerprint"], expected_cases=cases)
    report.execution = golden.execution_binding(ev, plan)
    report.resources.update(source_files=golden.source_identity(), process_identity=gate.process_identity())
    try:
        gate.require(torch.cuda.is_available(), "CUDA unavailable")
        torch.set_num_threads(4)
        adapter = NativeReplay(args.checkpoint, "operational", "cuda:0")
        identity = runtime_identity("native", args.image_digest, lock)
        effective = configuration_value(adapter.effective_configuration())
        identity.update(effective_configuration=effective, effective_configuration_fingerprint=digest(effective))

        def trace(key):
            arrays, entry = load_case(args.corpus, lock, key)
            profile, tensors = trace_prediction(adapter, arrays, entry, key["seed"], identity)
            gate.require(golden.profile_configuration(profile) ==
                         golden.profile_configuration(candidate["profile"]), "native operational profile changed")
            return profile, tensors, entry

        execute_cases(report, ev.workspace, lock, cases, trace, Path(golden.WORKER).stem)
        gate.require(golden.source_identity() == plan["source_files"], "worker source changed during replay")
        report.status = "complete"
    except Exception as exc:
        report.status = "failed"
        report.prerequisite_errors.append({"type": type(exc).__name__, "message": str(exc)})
    report.ended_at = now()
    write_evidence(ev.workspace, golden.WORKER, asdict(report))
    return 0 if report.status == "complete" else 1


def replay(args):
    from scripts.replay_checkpoint_parity import command_output, device_snapshot
    ev = gate.Evidence(args.workspace)
    gate.require(not any((ev.workspace / p).exists() for p in
                        (golden.REPLAY, golden.SCHEDULE, golden.WORKER, golden.LAUNCH)),
                 "immutable replay attempt exists; preserve it")
    candidate = golden.validate_candidate(ev)
    image = candidate["profile"]["image_digest"]
    inputs = current_inputs(args, candidate["profile"])
    gate.require(command_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"]) == image,
                 "pinned native image missing")
    gate.require(not device_snapshot()["compute_processes"], "GPU already owned")
    start = now()
    source_files = golden.source_identity()
    plan = {"schema_version": 1, "kind": "milestone_native_golden_replay", "id": uuid4().hex,
            "started_at": start, "worker_manifest": golden.WORKER, "cases": candidate["cases"],
            "candidate": ev.reference(golden.CANDIDATE), "scope": candidate["scope"],
            "source_files": source_files}
    schedule = write_evidence(ev.workspace, golden.SCHEDULE, plan)
    name = "dume-milestone-native-golden-" + str(os.getpid())
    argv = worker_command(ev.workspace, inputs, name, image)
    log_path = "milestone-golden/worker.log"
    launch = {"started_at": now(), "argv": argv, "container": name, "inputs": inputs,
              "schedule": schedule, "source_files": source_files, "status": "failed"}
    record = {"schema_version": 1, "kind": "milestone_native_golden_replay",
              "evidence_kind": "real_model", "started_at": start, "candidate": ev.reference(golden.CANDIDATE),
              "scope": candidate["scope"], "source_files": source_files, "cases": candidate["cases"],
              "case_count": 12, "bounds": golden.BOUNDS, "status": "failed", "errors": [],
              "expected_prediction_calls": 24, "prediction_calls": None, "observer_control_calls": None,
              "approved": False, "promoted": False}
    process = None
    try:
        with (ev.workspace / log_path).open("x") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            try:
                launch["exit_code"] = process.wait(timeout=args.worker_timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                subprocess.run(["docker", "stop", "--time", "5", name], capture_output=True, timeout=20)
                process.wait(timeout=20)
                raise
        launch["ended_at"] = now()
        launch["log"] = ev.reference(log_path)
        if (ev.workspace / golden.WORKER).exists():
            launch["manifest"] = ev.reference(golden.WORKER)
            launch["status"] = "complete" if launch["exit_code"] == 0 and ev.json(golden.WORKER)["status"] == "complete" else "failed"
        record["launch"] = write_evidence(ev.workspace, golden.LAUNCH, launch)
        gate.require(current_inputs(args, candidate["profile"]) == inputs and
                     golden.source_identity() == source_files, "current inputs/source changed during replay")
        _, arrays = golden.validate_worker(ev, candidate, launch)
        record.update(prediction_calls=24, observer_control_calls=12)
        record["tensors"] = write_tensors(ev.workspace, arrays)
        record["comparison"] = golden.compare_arrays(ev.tensors(candidate["tensors"]), arrays)
        record["status"] = "complete" if record["comparison"]["passed"] else "failed"
    except (Exception, KeyboardInterrupt) as exc:
        record["errors"].append({"type": type(exc).__name__, "message": str(exc)})
        if not (ev.workspace / golden.LAUNCH).exists():
            launch.update(ended_at=now(), exit_code=process.returncode if process else None)
            if (ev.workspace / log_path).exists():
                launch["log"] = ev.reference(log_path)
            record["launch"] = write_evidence(ev.workspace, golden.LAUNCH, launch)
    record["ended_at"] = now()
    write_evidence(ev.workspace, golden.REPLAY, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("candidate", "replay", "check", "_worker"):
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
            sub.add_argument("--profile", required=True)
            sub.add_argument("--output-manifest", required=True)
            sub.add_argument("--image-digest", required=True)
            sub.add_argument("--device", required=True)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        return worker(args)
    if args.command == "candidate":
        result = golden.create_candidate(args.workspace)
    elif args.command == "check":
        result = golden.validate_scoped_golden(args.workspace)
    else:
        result = replay(args)
    print(__import__("json").dumps(result, sort_keys=True))
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
