"""Generate, explicitly review, promote and verify immutable operational native goldens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard.golden import (  # noqa: E402
    Evidence, STAGE_FILES, compare_native_replay, consume_worker, native_profile,
    promote_reviewed_candidate, require, timestamp, validate_candidate, validate_golden,
    validate_tolerance_agreement,
)
from policy_guard.replay_contract import (  # noqa: E402
    PrerequisiteError, canonical, fingerprint_configuration as digest,
    load_input_lock, now, sha256_file, write_evidence, write_tensors,
)

ROOT = Path(__file__).resolve().parents[1]


def current_inputs(args):
    """Host-only byte identities. No model, transport or robot import."""
    current = load_input_lock(args.corpus, args.checkpoint)
    sources = {name: sha256_file(ROOT / name) for name in (
        "policy_guard/golden.py", "scripts/replay_native_golden.py",
        "scripts/replay_groot_native.py", "scripts/replay_checkpoint_parity.py",
        "policy_guard/replay_contract.py", "policy_guard/groot_guard.py",
        "policy/factory.py", "policy/gr00t/service.py", "embodiment/so_arm10x/controller.py",
        "pyproject.toml", "uv.lock",
    )}
    ev = Evidence(args.workspace)
    from replay_checkpoint_parity import command_output
    image = command_output(["docker", "image", "inspect", args.native_image, "--format", "{{.Id}}"])
    calibration = ev.json("calibration.json")
    return {"input_fingerprint": current["fingerprint"], "source_files": sources,
            "calibration_sha256": calibration["calibration_sha256"], "image_digest": image}


def operational_worker(args, cases, schedule_file, suffix):
    from replay_checkpoint_parity import device_snapshot, run_worker
    if device_snapshot()["compute_processes"]:
        raise PrerequisiteError("GPU already owned; native replay requires exclusive ownership")
    image = native_profile(Evidence(args.workspace))["image_digest"]
    return run_worker(args, "native", "operational", image, cases, schedule_file, suffix)


def collect(args, ev, worker, probe, clock, *, stage):
    destination = "golden-candidate.json" if stage == "candidate" else STAGE_FILES[stage]
    require(not (args.workspace / destination).exists(), "immutable stage exists; select an explicit successor session")
    start = clock()
    identity = ev.identity()
    validate_tolerance_agreement(ev, comparison_started_at=start)
    if stage != "candidate":
        validate_golden(ev)
    current = probe(args)
    require(current["input_fingerprint"] == identity["input_fingerprint"], "current input lock changed")
    require(current["image_digest"] == native_profile(ev)["image_digest"], "current native image changed")
    schedule = ev.json("input-lock.json")["schedule"]
    schedule_file = f"golden-{stage}-schedule.json"
    write_evidence(args.workspace, schedule_file, {
        "schema_version": 1, **identity, "kind": "replay", "cases": schedule,
        "evidence_kind": "test_only" if ev.test_only else "real_model",
    })
    record = {
        "schema_version": 1, **identity, "status": "not_run", "started_at": start,
        "evidence_kind": "test_only" if ev.test_only else "real_model", "stage": stage,
        "profile": "native-operational", "cases": schedule,
        "agreement": ev.reference("tolerance-agreement.json"), "current_inputs": current,
        "current_fingerprint": digest(current), "prerequisite_errors": [],
    }
    try:
        launch, _ = worker(args, schedule, schedule_file, "-golden-" + stage)
        record["worker"] = launch
        if launch["status"] == "not_run":
            raise PrerequisiteError("native operational worker did not run")
        arrays = consume_worker(ev, launch, schedule)
        record["tensors"] = write_tensors(args.workspace, arrays)
        report = ev.json(launch["manifest"])
        record["ended_at"] = max(clock(), report["ended_at"], key=timestamp)
        if stage == "candidate":
            record.update(reason=args.reason, previous=None, status="complete")
        else:
            record.update(candidate=ev.reference("golden-candidate.json"),
                          approval=ev.reference("golden-approval.json"),
                          manifest=ev.reference("golden-manifest.json"))
            record["comparison"] = compare_native_replay(ev, record)
            record["status"] = "complete" if record["comparison"]["passed"] else "failed"
    except (PrerequisiteError, FileNotFoundError) as exc:
        record["status"] = "not_run"
        record["prerequisite_errors"].append(str(exc))
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        record["status"] = "failed"
        record["prerequisite_errors"].append(str(exc))
    record.setdefault("ended_at", clock())
    write_evidence(args.workspace, destination, record)
    return record


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("generate", "review", "promote", "verify", "check"):
        command = commands.add_parser(name)
        command.add_argument("--workspace", type=Path, required=True)
        command.add_argument("--corpus", type=Path, default=ROOT / "corpus/frozen_v1_0")
        command.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/GR00T-N1.7-3B-SO101")
        command.add_argument("--native-image", default="gr00t:latest")
        command.add_argument("--native-cache", type=Path, default=Path.home() / ".cache/huggingface")
        command.add_argument("--device", choices=("cuda:0",), default="cuda:0")
        command.add_argument("--worker-timeout", type=int, default=14400)
        if name == "generate":
            command.add_argument("--reason", required=True)
        if name in ("review", "promote"):
            command.add_argument("--candidate-sha256", required=True)
        if name in ("verify", "check"):
            command.add_argument("--stage", required=True, choices=tuple(STAGE_FILES))
    return result


def main(argv=None, *, worker=None, probe=None, prompt=None, clock=now, test_only=False):
    args = parser().parse_args(argv)
    args.workspace = args.workspace.resolve()
    try:
        require(test_only or (worker is None and probe is None and prompt is None),
                "injected workers/prompts require test_only; never production evidence")
        ev = Evidence(args.workspace, test_only=test_only)
        if args.command in ("generate", "verify"):
            if args.command == "generate":
                require(bool(args.reason.strip()), "intentional candidate reason required")
            record = collect(args, ev, worker or operational_worker, probe or current_inputs, clock,
                             stage="candidate" if args.command == "generate" else args.stage)
            print(canonical({"status": record["status"], "message": "not run" if record["status"] == "not_run"
                             else record["status"], "errors": record["prerequisite_errors"]}).decode())
            return {"complete": 0, "failed": 1, "not_run": 2}[record["status"]]
        if args.command in ("review", "promote"):
            validate_candidate(ev)
            require(ev.reference("golden-candidate.json")["sha256"] == args.candidate_sha256,
                    "exact candidate digest required")
            if args.command == "review":
                if prompt is None and not sys.stdin.isatty():
                    raise PrerequisiteError("interactive named operator review required")
                from approve_parity_evidence import record_decision
                ref = record_decision(args.workspace, "golden", prompt=prompt or input,
                                      clock=clock, test_only=test_only)
                record = Evidence(args.workspace, test_only=test_only).json(ref)
                print(canonical({"status": record["decision"], "decision": ref}).decode())
                return 0 if record["decision"] == "approved" else 1
            promote_reviewed_candidate(ev, candidate_sha256=args.candidate_sha256, clock=clock)
        else:
            record = ev.record(STAGE_FILES[args.stage])
            require(compare_native_replay(ev, record)["passed"], "native golden drift")
        print('{"status":"complete"}')
        return 0
    except (PrerequisiteError, FileNotFoundError) as exc:
        print(canonical({"status": "not_run", "message": f"not run: {exc}"}).decode())
        return 2
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
        print(canonical({"status": "failed", "message": str(exc)}).decode())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
