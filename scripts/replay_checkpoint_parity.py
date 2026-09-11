"""Isolated real-worker feasibility. This command cannot compare or approve."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard.replay_contract import (  # noqa: E402
    PrerequisiteError, contained, fingerprint_configuration, load_input_lock,
    now, read_json, sha256_file, validate_replay_manifest, write_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCHEDULE = (
    ("native", "diagnostic"), ("lerobot", "diagnostic"),
    ("native", "operational"), ("lerobot", "operational"),
    ("native", "stock-capacity"),
)


def command_output(argv, timeout=30):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise PrerequisiteError(f"{argv[0]} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def evidence_reference(workspace: Path, path: Path) -> dict:
    return {"path": str(path.resolve().relative_to(workspace.resolve())), "sha256": sha256_file(path)}


def device_snapshot():
    gpu = command_output([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv",
    ])
    processes = command_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader",
    ])
    return {"gpu": gpu, "compute_processes": processes, "meminfo": Path("/proc/meminfo").read_text()}


def profile_device(args, purpose):
    if purpose == "diagnostic":
        return args.diagnostic_device or args.device
    if purpose == "stock-capacity":
        return args.stock_device or args.device
    return args.device


def worker_argv(args, backend, purpose, image, output, container_name):
    device = profile_device(args, purpose)
    script = "/replay/scripts/replay_groot_native.py" if backend == "native" else "/replay/docker/lerobot-policy/replay_checkpoint.py"
    argv = [
        "docker", "run", "--rm", "--name", container_name, "--network", "none",
        "--read-only", "--memory", "24g", "--memory-swap", "24g",
        "--tmpfs", "/tmp:rw,size=2g",
    ]
    # Stock policy keeps its configured FlashAttention loader, which requires
    # CUDA visibility even when measuring both resident models on the CPU.
    if device.startswith("cuda") or purpose == "stock-capacity":
        argv += ["--gpus", "all"]
    # The only writable persistent mount is this explicit evidence workspace.
    for source, destination, mode in (
        (ROOT / "scripts", "/replay/scripts", "ro"),
        (ROOT / "policy_guard", "/replay/policy_guard", "ro"),
        (ROOT / "policy", "/replay/policy", "ro"),
        (ROOT / "docker/lerobot-policy", "/replay/docker/lerobot-policy", "ro"),
        (args.corpus.resolve(), "/inputs/corpus", "ro"),
        (args.checkpoint.resolve(), "/inputs/checkpoint", "ro"),
        (args.workspace.resolve(), "/evidence", "rw"),
    ):
        argv += ["--mount", f"type=bind,src={source},dst={destination},readonly" if mode == "ro" else f"type=bind,src={source},dst={destination}"]
    if backend == "native":
        argv += ["--mount", f"type=bind,src={args.native_cache.resolve()},dst=/root/.cache/huggingface,readonly"]
    # Existing baked cache is root-only. Container root has no host devices
    # except CUDA; source/input mounts and the root filesystem remain read-only.
    for key, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_CACHE": "/root/.cache/huggingface/hub", "HF_HOME": "/root/.cache/huggingface",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": "/replay", "UV_NO_SYNC": "1", "UV_PYTHON_DOWNLOADS": "never",
        "NO_ALBUMENTATIONS_UPDATE": "1", "TOKENIZERS_PARALLELISM": "false",
        "DUME_REPLAY_UID": str(os.getuid()), "DUME_REPLAY_GID": str(os.getgid()),
    }.items():
        argv += ["--env", f"{key}={value}"]
    argv += [
        "--entrypoint", "python" if backend == "native" else "python3", image, script,
        "--input-lock", "/evidence/input-lock.json", "--schedule", "/evidence/tracer-schedule.json",
        "--corpus", "/inputs/corpus", "--checkpoint", "/inputs/checkpoint",
        "--workspace", "/evidence", "--profile", purpose, "--output-manifest", output,
        "--image-digest", image, "--device", device,
    ]
    return argv


def run_worker(args, backend, purpose, image, cases):
    label = f"{backend}-{purpose}"
    output = f"workers/{label}.json"
    destination = contained(args.workspace, output)
    if destination.exists():
        raise FileExistsError(f"immutable worker evidence exists: {destination}")
    container_name = f"dume-replay-{label}-{os.getpid()}"
    argv = worker_argv(args, backend, purpose, image, output, container_name)
    expected_session = read_json(args.workspace / "session.json")["session_id"]
    input_lock = read_json(args.workspace / "input-lock.json")
    source_files = {name: sha256_file(ROOT / name) for name in (
        "policy_guard/replay_contract.py", "policy_guard/groot_guard.py",
        "scripts/replay_groot_native.py", "scripts/replay_checkpoint_parity.py",
        "docker/lerobot-policy/replay_checkpoint.py", "docker/lerobot-policy/server.py",
        "policy/lerobot/features.py",
    )}
    start = now()
    device = profile_device(args, purpose)
    print(f"Starting {label} on {device}", flush=True)
    # Logs persist incrementally, including Docker/OOM failures before Python can write.
    log_path = contained(args.workspace, f"workers/{label}.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as log:
        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, text=True)
        try:
            exit_code = process.wait(timeout=args.worker_timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "stop", "--time", "5", container_name], capture_output=True, timeout=20)
            process.wait(timeout=20)
            exit_code = 124
    launch = {
        "backend": backend, "purpose": purpose, "argv": argv,
        "started_at": start, "ended_at": now(), "exit_code": exit_code,
        "log": evidence_reference(args.workspace, log_path),
    }
    if not destination.exists():
        # This is launcher failure evidence, explicitly not an invented worker trace.
        write_evidence(args.workspace, output, {
            "schema_version": 1, "session": read_json(args.workspace / "session.json")["session_id"],
            "status": "not_run", "stage": purpose, "evidence_kind": "launcher_failure",
            "prerequisite_errors": [{"message": f"worker exited {exit_code} without a result; inspect retained log"}],
            "started_at": start, "ended_at": now(), "expected_cases": cases,
            "executed_cases": [], "cases": [], "launch": launch,
        })
    report = read_json(destination)
    launch["status"] = report["status"]
    if exit_code != 0 and report["status"] == "complete":
        launch["status"] = "failed"
        launch["error"] = "nonzero worker exit cannot complete"
    if report["status"] == "complete" and purpose != "stock-capacity":
        validate_replay_manifest(
            report, args.workspace, cases, expected_session=expected_session,
            input_lock=input_lock, backend=backend, purpose=purpose,
            checkpoint_fingerprint=input_lock["checkpoint_fingerprint"],
            image_digest=image, source_files=source_files, device=device,
        )
    launch["manifest"] = {"path": output, "sha256": sha256_file(destination)}
    print(json.dumps({"worker": label, "status": launch["status"], "exit_code": exit_code}), flush=True)
    return launch, report


def feasibility(args):
    started = now()
    report = {"schema_version": 1, "stage": "feasibility", "status": "not_run", "started_at": started, "workers": [], "prerequisite_errors": []}
    workspace_ready = False
    try:
        # Absent inputs report not_run even before a session/workspace exists.
        for path in (args.corpus / "manifest.json", args.checkpoint / "config.json"):
            if not path.is_file():
                raise PrerequisiteError(f"missing input: {path}")
        session_path = args.workspace / "session.json"
        session = read_json(session_path)
        if session.get("schema_version") != 1 or not session.get("session_id"):
            raise ValueError("invalid existing session identity")
        for name in ("input-lock.json", "profiles.json", "feasibility.json", "tracer-schedule.json"):
            if (args.workspace / name).exists():
                raise FileExistsError(f"immutable stage conflict: {args.workspace / name}; select an explicit successor session")
        workspace_ready = True
        for name in ("workers", "tensors"):
            (args.workspace / name).mkdir(exist_ok=True)
        report["session"] = session["session_id"]
        report["session_reference"] = {"path": "session.json", "sha256": sha256_file(session_path)}
        lock = load_input_lock(args.corpus, args.checkpoint)
        entry = next((item for item in lock["records"] if item["file"] == args.record), None)
        if entry is None:
            raise ValueError("record not in locked corpus")
        cases = [{"record": args.record, "seed": entry["seeds"][0]}]
        write_evidence(args.workspace, "input-lock.json", lock)
        write_evidence(args.workspace, "tracer-schedule.json", {
            "schema_version": 1, "session": session["session_id"], "cases": cases,
            "profiles": PROFILE_SCHEDULE,
            "devices": {purpose: profile_device(args, purpose) for _, purpose in PROFILE_SCHEDULE},
            "observer_control": "same seed, same backend, capture disabled",
        })
        report["input_fingerprint"] = lock["fingerprint"]
        sources = {}
        for path in (
            "policy_guard/replay_contract.py", "scripts/replay_checkpoint_parity.py",
            "scripts/replay_groot_native.py", "docker/lerobot-policy/replay_checkpoint.py",
            "docker/lerobot-policy/server.py", "policy_guard/groot_guard.py",
            "policy/lerobot/features.py", "embodiment/so_arm10x/controller.py",
        ):
            sources[path] = sha256_file(ROOT / path)
        report["source_files"] = sources
        report["instrument_fingerprint"] = fingerprint_configuration(sources)
        images = {
            backend: command_output(["docker", "image", "inspect", ref, "--format", "{{.Id}}"])
            for backend, ref in (("native", args.native_image), ("lerobot", args.lerobot_image))
        }
        report["resources_before"] = device_snapshot()
        if report["resources_before"]["compute_processes"]:
            raise PrerequisiteError("GPU already owned by a compute process; release it before an explicit new session")
        profiles = []
        for backend, purpose in PROFILE_SCHEDULE:
            if device_snapshot()["compute_processes"]:
                raise PrerequisiteError("previous GPU owner still alive; sequential worker contract refused")
            launch, worker_report = run_worker(args, backend, purpose, images[backend], cases)
            report["workers"].append(launch)
            profiles.append({"backend": backend, "purpose": purpose, "status": launch["status"], "observed": worker_report.get("profile"), "manifest": launch["manifest"]})
        write_evidence(args.workspace, "profiles.json", {"schema_version": 1, "session": session["session_id"], "profiles": profiles})
        statuses = [worker["status"] for worker in report["workers"]]
        report["status"] = "complete" if statuses and all(s == "complete" for s in statuses) else ("failed" if "failed" in statuses else "not_run")
        report["resources_after"] = device_snapshot()
    except FileExistsError as exc:
        # Conflicts never append to or replace the completed session stage.
        print(json.dumps({"status": "failed", "message": str(exc)}))
        return 1
    except Exception as exc:
        report["status"] = "not_run" if isinstance(exc, (PrerequisiteError, FileNotFoundError)) else "failed"
        report["prerequisite_errors"].append({"type": type(exc).__name__, "message": str(exc)})
    report["ended_at"] = now()
    if workspace_ready:
        write_evidence(args.workspace, "feasibility.json", report)
    print(json.dumps({"status": report["status"], "message": "not run" if report["status"] == "not_run" else report["status"], "errors": report["prerequisite_errors"]}))
    return {"complete": 0, "failed": 1, "not_run": 2}[report["status"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("feasibility", "repeatability", "replay"):
        stage = stages.add_parser(name)
        stage.add_argument("--corpus", type=Path, required=True)
        stage.add_argument("--checkpoint", type=Path, required=True)
        stage.add_argument("--workspace", type=Path, required=True)
        stage.add_argument("--record", default="record_0000.npz")
        stage.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"))
        stage.add_argument("--diagnostic-device", choices=("cuda:0", "cpu"), help="Explicit diagnostic override; operational device is unchanged")
        stage.add_argument("--stock-device", choices=("cuda:0", "cpu"), help="Explicit device for the stock two-model capacity measurement")
        stage.add_argument("--native-image", default="gr00t:latest")
        stage.add_argument("--lerobot-image", default="lerobot-policy:latest")
        stage.add_argument("--native-cache", type=Path, default=Path.home() / ".cache/huggingface")
        stage.add_argument("--worker-timeout", type=int, default=900)
    args = parser.parse_args()
    if args.stage != "feasibility":
        print(json.dumps({"status": "not_run", "message": "not run: Task 1 tracer gate must pass before schedule expansion"}))
        return 2
    return feasibility(args)


if __name__ == "__main__":
    raise SystemExit(main())
