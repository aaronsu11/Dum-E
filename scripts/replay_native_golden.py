"""Generate, explicitly review, promote and verify immutable operational native goldens."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard.golden import (  # noqa: E402
    Evidence, STAGE_FILES, calibration_identity, compare_arrays, compare_native_replay,
    consume_worker, import_previous, native_profile, previous_digest, validate_previous,
    validate_backbone_identity,
    promote_reviewed_candidate, require, timestamp, validate_candidate, validate_golden,
    validate_tolerance_agreement,
)
from policy_guard.replay_contract import (  # noqa: E402
    BACKBONE_REVISION, PrerequisiteError, canonical, fingerprint_configuration as digest,
    load_input_lock, now, sha256_file, write_evidence, write_tensors,
)

ROOT = Path(__file__).resolve().parents[1]


def native_cache_identity(cache):
    # Match runtime_identity inventory names/hashes and the exact mounted cache.
    root = Path(cache).resolve()
    hub = root / "hub"
    model = hub / "models--nvidia--Cosmos-Reason2-2B"
    snapshot = model / "snapshots" / BACKBONE_REVISION
    revision = model / "refs/main"
    require(hub.resolve().is_relative_to(root) and model.resolve().is_relative_to(hub.resolve()),
            "native cache escapes mounted root")
    require(snapshot.resolve().is_relative_to(model.resolve()) and
            revision.resolve().is_relative_to(model.resolve()), "native snapshot escapes model cache")
    if not snapshot.is_dir() or not revision.is_file():
        raise PrerequisiteError("pinned native backbone snapshot is absent")
    if revision.read_text().strip() != BACKBONE_REVISION:
        raise PrerequisiteError("native backbone default revision differs from pin")
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        if not (snapshot / name).is_file():
            raise PrerequisiteError(f"pinned snapshot missing {name}")
    files = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_symlink():
            require(path.exists() and path.resolve().is_relative_to(model.resolve()),
                    "native snapshot blob missing or escapes model cache")
            require(not path.is_dir(), "native snapshot directory links cannot be inventoried")
        if path.is_file():
            files.append({"path": str(path.relative_to(snapshot)), "sha256": sha256_file(path)})
    return {"backbone_revision": BACKBONE_REVISION, "backbone_files": files,
            "backbone_fingerprint": digest(files)}


def current_inputs(args):
    """Host-only byte identities. No model, transport or robot import."""
    current = load_input_lock(args.corpus, args.checkpoint)
    sources = {name: sha256_file(ROOT / name) for name in (
        "policy_guard/golden.py", "scripts/replay_native_golden.py",
        "scripts/replay_groot_native.py", "scripts/replay_checkpoint_parity.py",
        "policy_guard/replay_contract.py", "policy_guard/groot_guard.py",
        "policy/factory.py", "policy/gr00t/service.py", "embodiment/so_arm10x/controller.py",
        "policy_guard/parity_gate.py", "scripts/approve_parity_evidence.py",
        "scripts/pose_sweep_units_probe.py", "config.example.yaml", "pyproject.toml", "uv.lock",
    )}
    ev = Evidence(args.workspace)
    from replay_checkpoint_parity import command_output
    image = command_output(["docker", "image", "inspect", args.native_image, "--format", "{{.Id}}"])
    calibration = calibration_identity(ev)
    backbone = native_cache_identity(args.native_cache)
    return {"input_fingerprint": current["fingerprint"], "source_files": sources,
            "calibration_sha256": calibration["sha256"], "image_digest": image, **backbone}


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
    live_ref = None
    if stage == "final":
        live = ev.record("live-run.json")
        require(timestamp(live["ended_at"]) < timestamp(start), "final replay must follow this live run")
        live_ref = ev.reference("live-run.json")
    previous = None
    if stage == "candidate":
        require(bool(args.previous_manifest) == bool(args.previous_sha256),
                "replacement requires explicit previous path and digest")
        if args.previous_manifest:
            previous = import_previous(ev, args.previous_manifest, args.previous_sha256)
            old = validate_previous(ev, previous)
            require(timestamp(old.json("golden-manifest.json")["ended_at"]) < timestamp(start),
                    "replacement must follow old promotion")
    current = probe(args)
    validate_backbone_identity(current, native_profile(ev))
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
    if stage == "candidate":
        record.update(reason=args.reason, previous=previous)
    if live_ref is not None:
        record["live_run"] = live_ref
    try:
        launch, _ = worker(args, schedule, schedule_file, "-golden-" + stage)
        record["worker"] = launch
        if launch["status"] == "not_run":
            raise PrerequisiteError("native operational worker did not run")
        arrays = consume_worker(ev, launch, schedule)
        record["tensors"] = write_tensors(args.workspace, arrays)
        after = probe(args)
        require(after == current, "relevant source/configuration changed during replay")
        require(Evidence(args.workspace, test_only=ev.test_only).identity() == identity,
                "locked evidence changed during replay")
        report = ev.json(launch["manifest"])
        require(timestamp(start) <= timestamp(report["started_at"]), "worker started before this collection")
        record["ended_at"] = max(clock(), report["ended_at"], key=timestamp)
        if stage == "candidate":
            record["status"] = "complete"
            if previous is not None:
                old = validate_previous(ev, previous)
                record["previous_comparison"] = compare_arrays(
                    old.tensors(old.json("golden-candidate.json")["tensors"]), arrays,
                    old.json("tolerance-proposal.json")["golden"])
        else:
            record.update(candidate=ev.reference("golden-candidate.json"),
                          approval=ev.reference("golden-approval.json"),
                          manifest=ev.reference("golden-manifest.json"))
            record["comparison"] = compare_native_replay(ev, record)
            record["identity_changed"] = record["comparison"]["identity_changed"]
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
            command.add_argument("--reason")
            command.add_argument("--previous-manifest", type=Path)
            command.add_argument("--previous-sha256")
        if name in ("review", "promote"):
            command.add_argument("--candidate-sha256", required=True)
            command.add_argument("--previous-sha256")
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
                if args.reason is None and not args.previous_manifest:
                    args.reason = "Initial operational native candidate; explicit review pending"
                require(isinstance(args.reason, str) and bool(args.reason.strip()),
                        "intentional candidate replacement reason required")
            record = collect(args, ev, worker or operational_worker, probe or current_inputs, clock,
                             stage="candidate" if args.command == "generate" else args.stage)
            print(canonical({"status": record["status"], "message": "not run" if record["status"] == "not_run"
                             else record["status"], "errors": record["prerequisite_errors"]}).decode())
            return {"complete": 0, "failed": 1, "not_run": 2}[record["status"]]
        if args.command in ("review", "promote"):
            candidate = validate_candidate(ev)
            require(previous_digest(ev, candidate) == args.previous_sha256, "exact previous manifest digest required")
            current = (probe or current_inputs)(args)
            validate_backbone_identity(current, native_profile(ev))
            require(digest(current) == candidate["current_fingerprint"],
                    "current source/configuration changed; complete replay and new review required")
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
            require(record["stage"] == args.stage, "native replay stage mismatch")
            current = (probe or current_inputs)(args)
            validate_backbone_identity(current, native_profile(ev))
            require(digest(current) == record["current_fingerprint"],
                    "relevant source/configuration changed since replay")
            require(timestamp(record["ended_at"]) <= timestamp(clock()), "replay has not completed yet")
            if args.stage == "final":
                live = ev.record("live-run.json")
                require(record["live_run"] == ev.reference("live-run.json") and
                        timestamp(live["ended_at"]) < timestamp(record["started_at"]),
                        "final replay must follow exact live evidence")
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
