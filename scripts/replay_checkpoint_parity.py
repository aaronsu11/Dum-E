"""Arm-free replay with prospective agreement gates; never operator approval."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard.replay_contract import (  # noqa: E402
    PrerequisiteError, contained, fingerprint_configuration, load_input_lock,
    now, read_json, repeatability_schedule, sha256_file, validate_replay_manifest,
    validate_schedule, write_evidence,
)

from policy_guard.parity_gate import COMPARISON_PROFILES, Evidence, evidence, require, timestamp, validate_repeatability, validate_tolerance_agreement, validate_tolerance_proposal
from policy_guard.parity_report import compare_tiers

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


def worker_argv(args, backend, purpose, image, output, container_name, schedule_file="tracer-schedule.json"):
    device = profile_device(args, purpose)
    script = "/replay/scripts/replay_groot_native.py" if backend == "native" else "/replay/docker/lerobot-policy/replay_checkpoint.py"
    argv = [
        "docker", "run", "--rm", "--name", container_name, "--network", "none",
        "--read-only", "--memory", "24g", "--memory-swap", "24g",
        "--tmpfs", "/tmp:rw,size=2g",
    ]
    if backend == "lerobot":
        # The installed PolicyServer logger creates logs/ during import.
        # Keep that runtime output on tmpfs, with all sources/inputs read-only.
        argv += ["--workdir", "/tmp"]
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
        "--input-lock", "/evidence/input-lock.json", "--schedule", f"/evidence/{schedule_file}",
        "--corpus", "/inputs/corpus", "--checkpoint", "/inputs/checkpoint",
        "--workspace", "/evidence", "--profile", purpose, "--output-manifest", output,
        "--image-digest", image, "--device", device,
    ]
    if getattr(args, "numerical", False) and purpose != "stock-capacity":
        index = argv.index(script)
        argv[index:index + 1] = ["/replay/scripts/replay_checkpoint_parity.py", "_worker", "--backend", backend]
    return argv


def run_worker(args, backend, purpose, image, cases, schedule_file="tracer-schedule.json", suffix=""):
    label = f"{backend}-{purpose}{suffix}"
    output = f"workers/{label}.json"
    destination = contained(args.workspace, output)
    if destination.exists():
        raise FileExistsError(f"immutable worker evidence exists: {destination}")
    container_name = f"dume-replay-{label}-{os.getpid()}"
    if getattr(args, "numerical", False) and purpose != "stock-capacity":
        plan = read_json(contained(args.workspace, schedule_file))
        plan["execution"] = {"id": uuid.uuid4().hex, "collection": "numerical-" + plan["kind"],
                             "started_at": now(), "worker_manifest": output}
        schedule_file = f"schedules/{label}-execution.json"
        write_evidence(args.workspace, schedule_file, plan)
    argv = worker_argv(args, backend, purpose, image, output, container_name, schedule_file)
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
        try:
            validate_replay_manifest(
                report, args.workspace, cases, expected_session=expected_session,
                input_lock=input_lock, backend=backend, purpose=purpose,
                checkpoint_fingerprint=input_lock["checkpoint_fingerprint"],
                image_digest=image, source_files=source_files, device=device,
            )
        except ValueError as exc:
            launch["status"] = "failed"
            launch["validation_error"] = str(exc)
    launch["manifest"] = {"path": output, "sha256": sha256_file(destination)}
    launch["instrument_files"] = instrument_identity()
    launch["schedule"] = evidence_reference(args.workspace, contained(args.workspace, schedule_file))
    launch_ref = write_evidence(args.workspace, f"launches/{label}.json", launch)
    launch["reference"] = launch_ref
    print(json.dumps({"worker": label, "status": launch["status"], "exit_code": exit_code}), flush=True)
    return launch, report


def feasibility(args):
    started = now()
    stage = args.stage
    schedule_name = "tracer-schedule.json" if stage == "feasibility" else f"{stage}-schedule.json"
    profiles_name = "profiles.json" if stage == "feasibility" else f"{stage}-profiles.json"
    report = {"schema_version": 1, "stage": stage, "status": "not_run", "started_at": started, "workers": [], "prerequisite_errors": [], "purpose": "trace collection only"}
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
        for name in (profiles_name, f"{stage}.json", schedule_name):
            if (args.workspace / name).exists():
                raise FileExistsError(f"immutable stage conflict: {args.workspace / name}; select an explicit successor session")
        workspace_ready = True
        for name in ("workers", "tensors"):
            (args.workspace / name).mkdir(exist_ok=True)
        report["session"] = session["session_id"]
        report["session_reference"] = {"path": "session.json", "sha256": sha256_file(session_path)}
        lock = load_input_lock(args.corpus, args.checkpoint)
        if (args.workspace / "input-lock.json").exists():
            if read_json(args.workspace / "input-lock.json") != lock:
                raise ValueError("existing input lock differs from current immutable inputs")
        else:
            write_evidence(args.workspace, "input-lock.json", lock)
        if stage == "feasibility":
            entry = next((item for item in lock["records"] if item["file"] == args.record), None)
            if entry is None:
                raise ValueError("record not in locked corpus")
            groups = [{"id": "tracer", "cases": [{"record": args.record, "seed": entry["seeds"][0]}]}]
            kind = "tracer"
        elif stage in ("repeatability", "repeatability-collection"):
            groups = repeatability_schedule(lock)["groups"]
            kind = "repeatability"
        else:
            groups = [{"id": "full", "cases": lock["schedule"]}]
            kind = "replay"
        schedule = {
            "schema_version": 1, "session": session["session_id"], "kind": kind,
            "input_fingerprint": lock["fingerprint"], "groups": groups,
            "profiles": [(backend, purpose) for backend, purpose in PROFILE_SCHEDULE
                         if stage == "feasibility" or purpose != "stock-capacity"],
            "devices": {purpose: profile_device(args, purpose) for _, purpose in PROFILE_SCHEDULE},
            "observer_control": "same seed, same backend, capture disabled",
        }
        if len(groups) == 1:
            schedule["cases"] = groups[0]["cases"]
        write_evidence(args.workspace, schedule_name, schedule)
        group_files = []
        for group in groups:
            validate_schedule(lock, group["cases"], kind)
            if len(groups) == 1:
                group_files.append(schedule_name)
            else:
                group_id = group["id"]
                relative = f"schedules/{stage}-{group_id}.json"
                write_evidence(args.workspace, relative, {**schedule, "cases": group["cases"], "group": group["id"]})
                group_files.append(relative)
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
        for backend, purpose in schedule["profiles"]:
            for group, schedule_file in zip(groups, group_files, strict=True):
                if device_snapshot()["compute_processes"]:
                    raise PrerequisiteError("previous GPU owner still alive; sequential worker contract refused")
                suffix = "" if stage == "feasibility" else "-" + stage + "-" + group["id"]
                launch, worker_report = run_worker(args, backend, purpose, images[backend], group["cases"], schedule_file, suffix)
                report["workers"].append(launch)
                profiles.append({"backend": backend, "purpose": purpose, "group": group["id"], "status": launch["status"], "observed": worker_report.get("profile"), "manifest": launch["manifest"]})
        profile_record = {"schema_version": 1, "session": session["session_id"], "profiles": profiles}
        if stage == "feasibility" and getattr(args, "numerical", False):
            from policy_guard.parity_gate import operational_semantics
            measured = next((p["observed"] for p in profiles if p["backend"] == "lerobot" and p["purpose"] == "operational" and p["status"] == "complete"), None)
            if measured is not None:
                profile_record["serving_configuration"] = operational_semantics(measured)
        write_evidence(args.workspace, profiles_name, profile_record)
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
        write_evidence(args.workspace, f"{stage}.json", report)
    print(json.dumps({"status": report["status"], "message": "not run" if report["status"] == "not_run" else report["status"], "errors": report["prerequisite_errors"]}))
    return {"complete": 0, "failed": 1, "not_run": 2}[report["status"]]


from policy_guard.parity_gate import (
    INSTRUMENT_FILES, assert_instrument, instrument_identity, numerical_execution_binding, repeatability_basis,
    validate_bundle, validate_comparison, validate_numerical_worker, validate_offline_evidence,
)


def project_preprocessing(inputs):
    """Project actual ordered camera patches, retaining full collated inputs."""
    import numpy as np
    from replay_upstream_parity import flatten_tensors
    flat = flatten_tensors(inputs)
    require({"pixel_values", "image_grid_thw", "input_ids", "attention_mask", "state"} <= set(flat),
            "unsupported pinned collated preprocessing surface")
    grid, pixels = flat["image_grid_thw"], flat["pixel_values"]
    require(grid.shape == (2, 3) and grid.dtype.kind in "iu" and (grid > 0).all(),
            "two ordered actual camera grids required")
    sizes = np.prod(grid, axis=1, dtype=np.int64)
    require(pixels.shape[0] == sum(sizes), "full camera patch count differs from grid")
    return {
        "image_front": pixels[:sizes[0]], "image_wrist": pixels[sizes[0]:],
        "state": flat["state"], "tokens": flat["input_ids"], "mask": flat["attention_mask"],
    }, flat


def restored_inputs(flat, device, dtypes):
    import torch
    result = {}
    for dotted, array in flat.items():
        path = dotted.split(".")
        target = result
        for name in path[:-1]:
            target = target.setdefault(name, {})
        dtype = getattr(torch, dtypes[dotted].removeprefix("torch."))
        target[path[-1]] = torch.from_numpy(array.copy()).to(device=device, dtype=dtype)
    return result


def capture_model(inner):
    """Owned collaborator recording original inputs and explicit common runs."""
    import torch
    from replay_upstream_parity import flatten_tensors

    class Capture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = inner
            self.last_raw = self.last_inputs = self.original_inputs = None
            self.input_dtypes = self.common = None

        @property
        def config(self):
            return self.inner.config

        def get_action(self, *args, **kwargs):
            inputs = args[0] if args else kwargs["inputs"]
            self.original_inputs = flatten_tensors(inputs)
            if self.common is not None:
                inputs = self.common
            self.last_inputs = dict(inputs)
            self.input_dtypes = model_input_dtypes(inputs)
            options = kwargs.get("options", args[1] if len(args) > 1 else None)
            result = self.inner.get_action(inputs, options=options)
            self.last_raw = result["action_pred"].detach().clone()
            return result

    return Capture().eval()


def numerical_worker(args):
    """Extend the existing adapters through owned composition, without forks."""
    import importlib.util
    import resource
    import torch
    from dataclasses import asdict
    from policy_guard.replay_contract import (
        ReplayManifest, configuration_value, execute_cases, load_case, read_tensors,
        runtime_identity, trace_prediction,
    )
    workspace = args.workspace
    if contained(workspace, args.output_manifest).exists():
        raise FileExistsError("immutable numerical worker exists")
    ev = Evidence(workspace)
    schedule = read_json(args.schedule)
    if schedule.get("kind") == "replay":
        validate_tolerance_agreement(ev, comparison_started_at=now())
        assert_instrument(ev)
    else:
        require(schedule.get("common_from") is None, "common inputs require prior agreement")
    lock = ev.json("input-lock.json")
    require(load_input_lock(args.corpus, args.checkpoint) == lock, "current inputs changed")
    validate_schedule(lock, schedule["cases"], schedule["kind"])
    session_id = ev.json("session.json")["session_id"]
    require(schedule["session"] == session_id and schedule["input_fingerprint"] == lock["fingerprint"],
            "numerical schedule subject changed")
    report = ReplayManifest(session=session_id, stage=args.profile,
                            input_fingerprint=lock["fingerprint"], expected_cases=schedule["cases"])
    relative = str(args.schedule.resolve().relative_to(workspace.resolve()))
    report.execution = numerical_execution_binding(schedule, ev.reference(relative), args.output_manifest)
    require(timestamp(report.execution["started_at"]) <= timestamp(report.started_at), "worker precedes collection")
    start = time.perf_counter()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(4)
        report.resources.update(requested_device=args.device, checkpoint_fp32_tensor_bytes=lock["checkpoint_fp32_tensor_bytes"], host_meminfo_before=Path("/proc/meminfo").read_text())
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise PrerequisiteError("CUDA unavailable in numerical worker")
            free, total = torch.cuda.mem_get_info()
            report.resources.update(free_before_bytes=free, total_bytes=total, device_name=torch.cuda.get_device_name())
            torch.cuda.reset_peak_memory_stats()
        if args.backend == "native":
            from replay_groot_native import NativeReplay
            adapter = NativeReplay(args.checkpoint, args.profile, args.device)
        else:
            sys.path.insert(0, str(ROOT / "docker/lerobot-policy"))
            spec = importlib.util.spec_from_file_location("parity_lerobot_replay", ROOT / "docker/lerobot-policy/replay_checkpoint.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            adapter = module.LeRobotReplay(args.checkpoint, args.profile, args.device)
        from policy_guard.parity_gate import process_identity
        report.resources["process_identity"] = process_identity()
        report.resources["instrument_files"] = instrument_identity()
        identity = runtime_identity(args.backend, args.image_digest, lock)
        effective = configuration_value(adapter.effective_configuration())
        identity.update(effective_configuration=effective,
                        effective_configuration_fingerprint=fingerprint_configuration(effective))
        if args.backend == "lerobot" and args.profile == "operational":
            seed = os.environ.get("DUME_POLICY_SEED")
            identity["serving_seed_policy"] = {"mode": "fixed" if seed is not None else "ambient",
                                                "seed": int(seed) if seed is not None else None}
        capture = capture_model(adapter.raw_model)
        adapter.observed = capture
        if args.backend == "lerobot":
            adapter.server.policy._groot_model = capture
        elif args.profile == "operational":
            adapter.policy.model = capture
        common = ev.json(schedule["common_from"]) if schedule.get("common_from") else None
        if common is not None:
            require(common["cases"] == schedule["cases"], "common source schedule mismatch")
        index = 0

        def trace_case(key):
            nonlocal index
            arrays, entry = load_case(args.corpus, lock, key)
            if common:
                item = common["collated"][index]
                capture.common = restored_inputs(read_tensors(workspace, item["tensors"]), args.device, item["dtypes"])
            profile, tensors = trace_prediction(adapter, arrays, entry, key["seed"], identity)
            processed, flat = project_preprocessing(capture.last_inputs)
            tensors.update({"preprocessing." + k: v for k, v in processed.items()})
            tensors.update({"collated." + k: v for k, v in flat.items()})
            import numpy as np
            tensors.update({"dtype." + k: np.array(DTYPE_NAMES.index(v), dtype=np.int8) for k, v in capture.input_dtypes.items()})
            index += 1
            return profile, tensors, entry

        execute_cases(report, workspace, lock, schedule["cases"], trace_case, Path(args.output_manifest).stem)
        report.status = "complete"
    except Exception as exc:
        report.status = "not_run" if isinstance(exc, (PrerequisiteError, FileNotFoundError)) else "failed"
        report.prerequisite_errors.append({"type": type(exc).__name__, "message": str(exc)})
    report.ended_at = now()
    if torch.cuda.is_initialized():
        report.resources.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    report.resources.update(elapsed_seconds=time.perf_counter() - start,
                            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_evidence(workspace, args.output_manifest, asdict(report))
    return {"complete": 0, "failed": 1, "not_run": 2}[report.status]


class FullWorkers:
    def __init__(self, args, ev):
        self.args, self.ev, self.counter = args, ev, 0

    def collect(self, name, schedule, *, common_from=None):
        import numpy as np
        from policy_guard.replay_contract import (
            CAMERA_ORDER, JOINT_ORDER, profile_configuration, read_tensors, write_tensors,
        )
        assert_instrument(self.ev)
        require(not device_snapshot()["compute_processes"], "GPU already owned")
        backend, purpose = name.split("-")
        self.counter += 1
        label = f"full-{self.counter:02d}"
        plan = {
            "schema_version": 1, **self.ev.identity(), "kind": "replay", "cases": schedule,
            "common_from": common_from,
        }
        relative = f"schedules/{label}.json"
        write_evidence(self.ev.workspace, relative, plan)
        profile = next(p["observed"] for p in self.ev.json("profiles.json")["profiles"]
                       if p["backend"] == backend and p["purpose"] == purpose)
        self.args.numerical = True
        self.args.device = profile["device"]
        self.args.diagnostic_device = profile["device"]
        launch, report = run_worker(self.args, backend, purpose, profile["image_digest"],
                                    schedule, relative, "-" + label)
        if launch["status"] == "not_run":
            raise PrerequisiteError(f"full worker not run: {launch}")
        require(launch["status"] == "complete", f"full worker failed: {launch}")
        require(profile_configuration(report["profile"]) == profile_configuration(profile),
                "full worker effective profile differs from agreed repeatability")
        tensors, preprocessing, collated = [], [], []
        for case in report["cases"]:
            arrays = read_tensors(self.ev.workspace, case["tensors"])
            tensors.append({k: arrays[k].squeeze(0) if k != "decoded" else arrays[k]
                            for k in ("raw", "noise", "decoded")})
            pre = {k.removeprefix("preprocessing."): v for k, v in arrays.items() if k.startswith("preprocessing.")}
            inputs = {k.removeprefix("collated."): v for k, v in arrays.items() if k.startswith("collated.")}
            require(pre and inputs, "full worker missing actual preprocessing/common input witnesses")
            preprocessing.append(write_tensors(self.ev.workspace, pre))
            collated.append({"tensors": write_tensors(self.ev.workspace, inputs),
                             "dtypes": {k: DTYPE_NAMES[int(arrays["dtype." + k])] for k in inputs}})
        bundle = {
            "cases": schedule, "profile": name, "input_fingerprint": self.ev.identity()["input_fingerprint"],
            "joint_order": list(JOINT_ORDER), "camera_order": list(CAMERA_ORDER),
            "tensors": write_tensors(self.ev.workspace, {k: np.stack([row[k] for row in tensors])
                                                        for k in ("raw", "noise", "decoded")}),
            "preprocessing": preprocessing, "common_inputs": preprocessing,
            "common_collated": [item["tensors"] for item in collated],
            "independent_collated": [item["tensors"] for item in collated],
            "collated": collated, "worker": launch["manifest"], "launch": launch["reference"],
        }
        bundle["reference"] = write_evidence(self.ev.workspace, f"bundles/{label}.json", bundle)
        return bundle


def _combined(ev, independent, common):
    from policy_guard.replay_contract import write_tensors
    arrays = dict(ev.tensors(common["tensors"]))
    arrays["decoded"] = ev.tensors(independent["tensors"])["decoded"]
    require(independent.get("reference") and common.get("reference"), "archived independent/common workers required")
    return {**common, "provenance": {"independent": independent["reference"], "common": common["reference"]},
            "preprocessing": independent["preprocessing"],
            "independent_collated": independent["independent_collated"],
            "tensors": write_tensors(ev.workspace, arrays)}


def full(workspace, *, workers=None, stock_check=None, started_at=None, clock=now):
    ev = evidence(workspace)
    start = started_at or clock()
    if not (ev.workspace / "tolerance-agreement.json").is_file():
        raise FileNotFoundError("tolerance agreement required before comparison")
    validate_tolerance_agreement(ev, comparison_started_at=start)
    assert_instrument(ev)
    require(workers is not None, "full worker runner required")
    if not ev.test_only:
        require(isinstance(workers, FullWorkers) and stock_check is None, "fixture runner forbidden for real evidence")
    from replay_upstream_parity import check as stock
    (stock_check or stock)(ev)
    if (ev.workspace / "offline-report.json").exists():
        raise FileExistsError("immutable offline report exists")
    report = {
        "schema_version": 1, **ev.identity(), "evidence_kind": "test_only" if ev.test_only else "real_model",
        "stage": "offline", "started_at": start, "status": "not_run", "comparisons": [],
        "agreement": ev.reference("tolerance-agreement.json"),
        "instrument_files": instrument_identity(), "prerequisite_errors": [],
    }
    if not ev.test_only:
        report["upstream"] = ev.reference("upstream-result.json")
        report["historical_context"] = historical_context(workers.args, ev)
    schedule = ev.json("input-lock.json")["schedule"]
    validate_schedule(ev.json("input-lock.json"), schedule, "replay")
    try:
        profiles = dict.fromkeys(profile for pair in COMPARISON_PROFILES.values() for profile in pair)
        independent = {name: workers.collect(name, schedule) for name in profiles}
        for name, pair in COMPARISON_PROFILES.items():
            comparison_start = clock()
            source = independent[pair[0]].get("reference", pair[0])
            lcommon = workers.collect(pair[0], schedule, common_from=source)
            rcommon = workers.collect(pair[1], schedule, common_from=source)
            left = _combined(ev, independent[pair[0]], lcommon)
            right = _combined(ev, independent[pair[1]], rcommon)
            comparison = compare_tiers(ev, name, left, right, started_at=comparison_start, ended_at=clock())
            comparison["provenance"] = {"left": left["provenance"], "right": right["provenance"]}
            validate_comparison(ev, comparison, report_start=start, report_end=comparison["ended_at"])
            report["comparisons"].append(comparison)
        report["status"] = "complete" if all(c["passed"] for c in report["comparisons"]) else "failed"
    except Exception as exc:
        report["status"] = "not_run" if isinstance(exc, (FileNotFoundError, PrerequisiteError)) else "failed"
        report["prerequisite_errors"].append({"type": type(exc).__name__, "message": str(exc)})
    report["ended_at"] = clock()
    report["archived_at"] = clock()
    report["parity_passed"] = report["status"] == "complete"
    report["caveats"] = ev.json("tolerance-proposal.json")["caveats"]
    if report["status"] == "complete":
        try:
            validate_offline_evidence(ev, stock_check=stock_check, report=report)
        except Exception as exc:
            report["status"] = "not_run" if isinstance(exc, (FileNotFoundError, PrerequisiteError)) else "failed"
            report["parity_passed"] = False
            report["prerequisite_errors"].append({"type": type(exc).__name__, "message": str(exc)})
    write_evidence(ev.workspace, "offline-report.json", report)
    return report


def check_offline(workspace, *, stock_check=None):
    return validate_offline_evidence(workspace, stock_check=stock_check)


def repeatability(args):
    """Run only the fixed same-backend schedule, then reduce within each profile."""
    args.stage = "repeatability-collection"
    args.numerical = True
    if (args.workspace / "repeatability.json").exists():
        raise FileExistsError("immutable repeatability exists")
    code = feasibility(args)
    if code:
        return code
    ev = Evidence(args.workspace)
    collection = ev.json("repeatability-collection.json")
    report = reduce_repeatability(ev, collection)
    write_evidence(ev.workspace, "repeatability.json", report)
    validate_repeatability(Evidence(args.workspace))
    print(json.dumps({"status": report["status"], "message": "same-backend measurements only"}))
    return 0


def reduce_repeatability(workspace, collection):
    import numpy as np
    from policy_guard.replay_contract import profile_configuration, write_tensors
    from policy_guard.parity_report import joint_deviations, metrics
    ev = evidence(workspace)
    require(collection["status"] == "complete", "repeatability collection incomplete")
    schedule = repeatability_schedule(ev.json("input-lock.json"))
    expected = [key for group in schedule["groups"] for key in group["cases"]]
    profiles = ev.json("profiles.json")["profiles"]
    report = {
        "schema_version": 1, **ev.identity(), "evidence_kind": "test_only" if ev.test_only else "real_model",
        "status": "complete", "stage": "repeatability", "schedule": schedule,
        "started_at": collection["started_at"], "ended_at": collection["ended_at"],
        "measurements": [], "instrument_files": instrument_identity(),
        "scope": "same backend and same profile only; no cross-backend residuals",
    }
    for backend, purpose in PROFILE_SCHEDULE[:4]:
        name = backend + "-" + purpose
        launches = [item for item in collection["workers"]
                    if item["backend"] == backend and item["purpose"] == purpose]
        require(len(launches) == len(schedule["groups"]), "repeatability process groups incomplete")
        bound = next(p["observed"] for p in profiles if p["backend"] == backend and p["purpose"] == purpose)
        values, groups, worker_refs, launch_refs = [], [], [], []
        raw_max, preprocessing_max = 0.0, 0.0
        all_arrays = []
        starts, ends = [], []
        for launch, group in zip(launches, schedule["groups"], strict=True):
            require(launch["exit_code"] == 0 and launch["status"] == "complete", "repeatability worker failed")
            validated = validate_numerical_worker(
                ev, {"worker": launch["manifest"], "launch": launch["reference"]}, name,
                group["cases"], kind="repeatability", start=collection["started_at"], end=collection["ended_at"])
            worker = validated["worker"]
            require(worker["executed_cases"] == group["cases"], "repeatability worker coverage changed")
            require(profile_configuration(worker["profile"]) == profile_configuration(bound),
                    "repeatability profile differs from fresh feasibility")
            process_id = worker["resources"]["process_identity"]
            groups.append({"id": group["id"], "cases": group["cases"], "process_id": fingerprint_configuration(process_id)})
            worker_refs.append(launch["manifest"])
            launch_refs.append(launch["reference"])
            starts.append(worker["started_at"])
            ends.append(worker["ended_at"])
            for case in worker["cases"]:
                arrays = ev.tensors(case["tensors"])
                values.append({"decoded": arrays["decoded"], "noise": arrays["noise"][0]})
                all_arrays.append(arrays)
        stacked = {key: np.stack([item[key] for item in values]) for key in ("decoded", "noise")}
        paired, reference = [], []
        for record_name in dict.fromkeys(key["record"] for key in expected):
            same = [i for i, key in enumerate(expected) if key["record"] == record_name and key["mode"] != "changed"]
            for i in same[1:]:
                paired.append(stacked["decoded"][i])
                reference.append(stacked["decoded"][same[0]])
                raw_max = max(raw_max, float(np.max(np.abs(all_arrays[i]["raw"].astype(np.float64) -
                                                           all_arrays[same[0]]["raw"].astype(np.float64)))))
                names = [key for key in all_arrays[i] if key.startswith("preprocessing.")]
                require(names, "same-backend preprocessing observations missing")
                for key in names:
                    left, right = all_arrays[same[0]][key], all_arrays[i][key]
                    require(left.shape == right.shape and left.dtype == right.dtype,
                            "repeatability preprocessing shape/dtype changed")
                    if key.endswith((".tokens", ".mask")):
                        require(np.array_equal(left, right), "same-seed categorical preprocessing differs")
                    else:
                        preprocessing_max = max(preprocessing_max, float(np.max(np.abs(
                            left.astype(np.float64) - right.astype(np.float64)))))
        stats = metrics(joint_deviations(np.stack(reference), np.stack(paired)))
        report["measurements"].append({
            "profile": name, "cases": expected, "groups": groups,
            "started_at": min(starts), "ended_at": max(ends), "workers": worker_refs, "launches": launch_refs,
            "tensors": write_tensors(ev.workspace, stacked),
            "statistics": {key: value.tolist() for key, value in stats.items()},
            "raw_max_abs": raw_max, "preprocessing_max_abs": preprocessing_max,
            "attention": bound["attention"], "device": bound["device"],
            "seed_policy": bound.get("serving_seed_policy", {"mode": "native_ambient"}),
            "replay_intervention": "seed at actual sampling boundary; distinct from deployed seed behavior",
        })
    for measured in report["measurements"]:
        basis = repeatability_basis(ev, measured, report)
        require(all(measured[key] == value for key, value in basis.items()), "repeatability reduction differs")
    return report


def prepare_tolerances(args):
    """Propose bounds from same-backend data and pinned stock semantics only."""
    import numpy as np
    from replay_upstream_parity import validate_harness
    ev = Evidence(args.workspace)
    for forbidden in ("upstream-result.json", "offline-report.json", "tolerance-agreement.json"):
        require(not (ev.workspace / forbidden).exists(), "proposal must precede comparisons/agreement")
    repeat = validate_repeatability(ev)
    require(repeat["instrument_files"] == instrument_identity(), "repeatability source became stale")
    require(load_input_lock(args.corpus, args.checkpoint) == ev.json("input-lock.json"), "current inputs changed")
    source = ev.json("harness-source.json")
    validate_harness(source["source"], source["harness"])
    # Fresh arm-free request is measured independently of replay and compared
    # only for semantic configuration, never for cross-backend model residuals.
    from policy_guard.parity_gate import operational_semantics, validate_runtime_attestation, _calibration
    attested = ev.json("arm-free-attestation.json")
    operational = next(p["observed"] for p in ev.json("profiles.json")["profiles"]
                       if p["backend"] == "lerobot" and p["purpose"] == "operational")
    sem = operational_semantics(operational)
    validate_runtime_attestation(attested["attestation"], host=attested["host"],
                                 request=attested["request"], expected_configuration=sem,
                                 now=attested["host"]["checked_at"])
    require(ev.json("profiles.json")["serving_configuration"] == sem, "serving/replay semantics differ")
    _calibration(ev)
    start = now()
    measurements = {m["profile"]: m for m in repeat["measurements"]}
    comparisons = {}
    # These floor values are explicit proposals in checkpoint percent units,
    # not approved tolerances or a fitted response to cross-backend outputs.
    floors = {"max_abs": 0.1, "mean_abs": 0.05, "bias": 0.02, "slope": 0.002}
    for name, pair in COMPARISON_PROFILES.items():
        basis = [measurements[p] for p in pair]
        thresholds = {}
        for metric, floor in floors.items():
            key = {"bias": "trace_bias_max_abs", "slope": "trace_slope_max_abs"}.get(metric, metric)
            observed = np.max([np.abs(item["statistics"][key]) for item in basis], axis=0)
            thresholds[metric] = np.maximum(observed * 4, floor).tolist()
        comparisons[name] = {
            "profiles": pair, "noise_policy": "exact" if name == "diagnostic" else "independent",
            "thresholds": {
                "preprocessing": {"atol": max(1e-6, 4 * max(m["preprocessing_max_abs"] for m in basis)), "rtol": 1e-6},
                "raw": {"atol": max(1e-3, 4 * max(m["raw_max_abs"] for m in basis)), "rtol": 1e-3},
                "decoded": thresholds,
            },
            "rationale": (
                "Prospective proposal: four times worst measured same-backend repeatability, "
                "with explicit floors max_abs=.1, mean_abs=.05, bias=.02 percent and slope=.002 percent/index. "
                "Raw floors atol=rtol=1e-3 follow pinned stock defaults. No cross-backend residuals used. "
                + ("Diagnostic full noise must match exactly." if name == "diagnostic" else
                   "Independent cross-dtype RNG: pair all 600 prescribed cases; retain individual error, bias and "
                   "slope bounds plus aggregate bounds. No claim of matched noise from equal integer seeds.")
            ),
        }
    native = measurements["native-operational"]
    proposal = {
        "schema_version": 1, **ev.identity(), "evidence_kind": "real_model", "status": "complete",
        "started_at": start, "ended_at": now(), "repeatability": ev.reference("repeatability.json"),
        "harness": source["harness"], "instrument_files": instrument_identity(), "comparisons": comparisons,
        "golden": {"atol": max(1e-5, 4 * max(native["statistics"]["max_abs"])), "rtol": 0.0,
                   "rationale": "Four times same-backend native operational worst repeatability; 1e-5 percent floor."},
        "units": ["percent"] * 5 + ["gripper_percent"], "aggregation": "trace-and-aggregate-ols-0..15",
        "arm_free_attestation": ev.reference("arm-free-attestation.json"),
        "caveats": [
            "Proposal only; explicit operator agreement remains required.",
            "Accepted forced-letterbox geometry and backbone revision do not prove historical training provenance.",
            "Historical seed_verdict not-honored remains unchanged; historical outputs are context only.",
            "Native operational FlashAttention/bf16 and LeRobot operational SDPA/bf16 are measured separately.",
            "Independent operational noise can fail prospective per-trace bridge bounds; no adaptive loosening.",
            "Five arm joints use checkpoint percent units, gripper uses percent; no implicit live-degree conversion.",
        ],
    }
    write_evidence(ev.workspace, "tolerance-proposal.json", proposal)
    validate_tolerance_proposal(Evidence(args.workspace))
    return proposal


def historical_context(args, ev):
    """Archive immutable old stochastic context without any controlled verdict."""
    import numpy as np
    from policy_guard.replay_contract import load_case, write_tensors
    lock = ev.json("input-lock.json")
    result = []
    for entry in lock["records"]:
        arrays, _ = load_case(args.corpus, lock, {"record": entry["file"], "seed": entry["seeds"][0]})
        samples = arrays["action_samples"].astype(np.float64)
        result.append({
            "record": entry["file"], "instruction": entry["instruction"],
            "record_sha256": entry["sha256"], "seed_verdict": "not-honored",
            "tensors": write_tensors(ev.workspace, {
                "ground_truth_action": arrays["ground_truth_action"],
                "historical_mean": samples.mean(axis=0), "historical_std": samples.std(axis=0),
                "historical_min": samples.min(axis=0), "historical_max": samples.max(axis=0),
            }),
        })
    return {"evidence_kind": "historical_stochastic_context", "affects_parity_verdict": False,
            "historical_seed_verdict": "not-honored", "records": result}


def model_input_dtypes(inputs, prefix=""):
    import torch
    result = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            result[prefix + key] = str(value.dtype)
        elif hasattr(value, "items"):
            result.update(model_input_dtypes(value, prefix + key + "."))
        else:
            raise PrerequisiteError("unsupported nonnumeric common input dtype")
    return result


DTYPE_NAMES = ("torch.bfloat16", "torch.float32", "torch.float64", "torch.float16",
               "torch.int64", "torch.int32", "torch.int16", "torch.int8", "torch.uint8", "torch.bool")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("feasibility", "repeatability", "replay", "full", "check", "prepare-tolerances"):
        stage = stages.add_parser(name)
        stage.add_argument("--corpus", type=Path, default=ROOT / "corpus/frozen_v1_0")
        stage.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/GR00T-N1.7-3B-SO101")
        stage.add_argument("--workspace", type=Path, required=True)
        stage.add_argument("--record", default="record_0000.npz")
        stage.add_argument("--device", default="cuda:0", choices=("cuda:0", "cpu"))
        stage.add_argument("--diagnostic-device", choices=("cuda:0", "cpu"), help="Explicit diagnostic override; operational device is unchanged")
        stage.add_argument("--stock-device", choices=("cuda:0", "cpu"), help="Explicit device for the stock two-model capacity measurement")
        stage.add_argument("--native-image", default="gr00t:latest")
        stage.add_argument("--lerobot-image", default="lerobot-policy:latest")
        stage.add_argument("--native-cache", type=Path, default=Path.home() / ".cache/huggingface")
        stage.add_argument("--worker-timeout", type=int, default=43200 if name == "full" else 900)
        if name == "check":
            stage.add_argument("--stage", dest="check_stage", required=True, choices=("repeatability", "offline"))
    worker = stages.add_parser("_worker")
    for name in ("input-lock", "schedule", "workspace", "corpus", "checkpoint"):
        worker.add_argument("--" + name, type=Path, required=True)
    for name in ("backend", "profile", "output-manifest", "image-digest", "device"):
        worker.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.stage in ("full", "check", "prepare-tolerances", "_worker"):
        try:
            if args.stage == "_worker":
                return numerical_worker(args)
            ev = Evidence(args.workspace)
            if args.stage == "full":
                result = full(ev, workers=FullWorkers(args, ev))
            elif args.stage == "prepare-tolerances":
                result = prepare_tolerances(args)
            else:
                require(load_input_lock(args.corpus, args.checkpoint) == ev.json("input-lock.json"), "current inputs differ from archive")
                result = check_offline(ev) if args.check_stage == "offline" else validate_repeatability(ev)
            print(json.dumps({"status": result["status"]}))
            return {"complete": 0, "failed": 1, "not_run": 2}[result["status"]]
        except Exception as exc:
            missing = isinstance(exc, (FileNotFoundError, PrerequisiteError))
            print(json.dumps({"status": "not_run" if missing else "failed", "message": ("not run: " if missing else "") + str(exc)}))
            return 2 if missing else 1
    if args.stage == "repeatability":
        return repeatability(args)
    args.numerical = True
    return feasibility(args)


if __name__ == "__main__":
    raise SystemExit(main())
