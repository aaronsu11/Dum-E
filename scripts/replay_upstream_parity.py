"""Unmodified pinned stock harness, strict coverage, and pre-crop observations."""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import re
import runpy
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard.parity_gate import (  # noqa: E402
    evidence, require, timestamp, validate_tolerance_agreement, validate_tolerance_proposal,
)
from policy_guard.parity_gate import (  # noqa: E402
    array_comparison, compare_raw, parse_junit, validate_launches,
    validate_observation, validate_producer_output, validate_upstream_evidence,
)


def check(workspace):
    return validate_upstream_evidence(workspace)

from policy_guard.replay_contract import (  # noqa: E402
    PrerequisiteError, capture_bytes, contained, floating_dtypes, load_input_lock,
    now, read_json, read_tensors, runtime_identity, sha256_file, to_numpy,
    write_evidence, write_tensors,
)

ROOT = Path(__file__).resolve().parents[1]
HARNESS_COMMIT = "7e241bd630a3719a56157a497ce5d08f244784f1"
PRODUCER = "tests/policies/groot/utils/dump_original_n1_7.py"
CONSUMER = "tests/policies/groot/test_groot_vs_original.py"
CASE = "test_groot_get_action_parity[new_embodiment]"
ARTIFACT = "original_n1_7_new_embodiment.npz"


def validate_harness(source, pins):
    source = Path(source).resolve()
    require(not source.is_relative_to(ROOT), "upstream harness must be outside project source")
    require(pins.get("commit") == HARNESS_COMMIT, "harness must be the immutable v0.6.1 commit")
    actual = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    require(actual == HARNESS_COMMIT, "harness checkout commit changed")
    dirty = subprocess.run(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=True).stdout
    require(not dirty.strip(), "harness checkout is not pristine")
    for role, name in (("producer", PRODUCER), ("consumer", CONSUMER)):
        path = contained(source, name)
        data = capture_bytes(path)
        committed = subprocess.run(["git", "-C", str(source), "show", f"{HARNESS_COMMIT}:{name}"],
                                   capture_output=True, check=True).stdout
        require(data == committed and hashlib.sha256(data).hexdigest() == pins[role + "_sha256"],
                f"unmodified pinned {role} source required")
    return source


def create_producer_directory(path):
    Path(path).mkdir(mode=0o700, parents=False, exist_ok=False)


def stock_argv(args, backend, profile, source, stage, artifact_ref=None):
    from replay_checkpoint_parity import worker_argv

    launch = SimpleNamespace(
        workspace=args.workspace, corpus=args.corpus, checkpoint=args.checkpoint,
        native_cache=args.native_cache, device=profile["device"],
        diagnostic_device=profile["device"], stock_device=profile["device"],
    )
    argv = worker_argv(launch, backend, "diagnostic", profile["image_digest"],
                       "unused", f"dume-stock-{stage}-{os.getpid()}")
    entry = argv.index("--entrypoint")
    # The stock lifetime holds both resident models. This is a capacity ceiling,
    # not evidence that the untouched producer actually fits or completes.
    argv[argv.index("--memory") + 1] = "48g"
    argv[argv.index("--memory-swap") + 1] = "48g"
    argv = argv[:entry] + [
        "--mount", f"type=bind,src={source},dst=/stock,readonly",
        "--workdir", "/tmp", "--env", "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
    ]
    # Native stock retains FlashAttention during its bootstrap model load.
    if backend == "native" and "--gpus" not in argv:
        argv += ["--gpus", "all"]
    bounds = args.bounds
    for key, value in {
        "GROOT_N1_7_PARITY_DIR": "/evidence/stock/producer",
        "GROOT_N1_7_LIBERO_CKPT": "/inputs/checkpoint",
        "GROOT_PARITY_DEVICE": profile["device"],
        "GROOT_PARITY_ATOL": str(bounds["atol"]), "GROOT_PARITY_RTOL": str(bounds["rtol"]),
    }.items():
        argv += ["--env", f"{key}={value}"]
    argv += ["--entrypoint", "python" if backend == "native" else "python3",
             profile["image_digest"], "/replay/scripts/replay_upstream_parity.py",
             "_worker", "--stage", stage, "--workspace", "/evidence",
             "--source", "/stock", "--checkpoint", "/inputs/checkpoint",
             "--corpus", "/inputs/corpus", "--device", profile["device"],
             "--image-digest", profile["image_digest"]]
    return argv


def execute_stock(argv, *, stdout, stderr, text, timeout):
    """Own the container lifetime even when the Docker client times out."""
    process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, text=text)
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        container = argv[argv.index("--name") + 1]
        subprocess.run(["docker", "stop", "--time", "5", container],
                       capture_output=True, timeout=20)
        process.wait(timeout=20)
        returncode = 124
    return SimpleNamespace(returncode=returncode)


def run(workspace, *, executor=None, args=None, source_validator=validate_harness, input_validator=load_input_lock, resources=None):
    ev = evidence(workspace)
    start = now()
    validate_tolerance_agreement(ev, comparison_started_at=start)  # before ANY launch
    proposal = validate_tolerance_proposal(ev)
    require(ev.test_only or (executor is None and source_validator is validate_harness and input_validator is load_input_lock and resources is None), "subprocess injection is test-only")
    from replay_checkpoint_parity import assert_instrument
    assert_instrument(ev)
    if (ev.workspace / "upstream-result.json").exists():
        raise FileExistsError("immutable upstream result exists; select a successor session")
    source_record = ev.json("harness-source.json")
    require(source_record["harness"] == proposal["harness"], "harness proposal mismatch")
    source = source_validator(source_record["source"], proposal["harness"])
    require(args is not None, "stock runtime arguments required")
    require(input_validator(args.corpus, args.checkpoint) == ev.json("input-lock.json"),
            "stock current inputs differ from agreement")
    args.workspace = ev.workspace
    args.bounds = proposal["comparisons"]["diagnostic"]["thresholds"]["raw"]
    directory = ev.workspace / "stock"
    directory.mkdir(exist_ok=True)
    create_producer_directory(directory / "producer")
    result = {
        "schema_version": 1, **ev.identity(), "evidence_kind": "test_only" if ev.test_only else "real_model",
        "started_at": start, "status": "not_run", "harness": proposal["harness"],
        "agreement": ev.reference("tolerance-agreement.json"), "seed": 42, "tag": "new_embodiment",
        "checkpoint_fingerprint": ev.json("input-lock.json")["checkpoint_fingerprint"],
        "launches": [], "prerequisite_errors": [],
    }
    profiles = {p["backend"]: p["observed"] for p in ev.json("profiles.json")["profiles"]
                if p["purpose"] == "diagnostic"}
    try:
        for stage, backend in (("producer", "native"), ("consumer", "lerobot")):
            from replay_checkpoint_parity import device_snapshot
            require(not (resources or device_snapshot)()["compute_processes"], "GPU already owned")
            # Validate the producer bytes again before stock's pickle-enabled load.
            if stage == "consumer":
                ev.bytes(result["artifact"]["path"], result["artifact"]["sha256"])
            argv = stock_argv(args, backend, profiles[backend], source, stage)
            path = directory / f"{stage}.log"
            with path.open("x") as log:
                launch_start = now()
                process = (executor or execute_stock)(argv, stdout=log, stderr=subprocess.STDOUT,
                                                       text=True, timeout=args.worker_timeout)
            result["launches"].append({"stage": stage, "argv": argv, "started_at": launch_start,
                                      "ended_at": now(), "exit_code": process.returncode})
            result[stage + "_exit"] = process.returncode
            result[stage + "_log"] = ev.reference(f"stock/{stage}.log")
            if process.returncode in (2, 124, 137):
                raise PrerequisiteError(f"stock {stage} not run: exit {process.returncode}; inspect retained log")
            require(process.returncode == 0, f"stock {stage} exited {process.returncode}")
            observation_ref = ev.reference(f"stock/{stage}-observation.json")
            result[stage + "_observation"] = observation_ref
            observation = validate_observation(ev, observation_ref, profiles[backend])
            result["left" if stage == "producer" else "right"] = observation["raw"]
            if stage == "producer":
                validate_producer_output(path.read_text(), process.returncode)
                require(sorted(p.name for p in (directory / "producer").iterdir()) == [ARTIFACT],
                        "stock producer artifact coverage differs")
                result["artifact"] = ev.reference("stock/producer/" + ARTIFACT)
                write_evidence(ev.workspace, "stock/artifact-lock.json", result["artifact"])
        result["junit"] = ev.reference("stock/junit.xml")
        result["coverage"] = ev.reference("stock/coverage.json")
        result["tests"] = parse_junit(ev.bytes("stock/junit.xml"))
        result["status"] = "complete"
    except Exception as exc:
        result["status"] = "not_run" if isinstance(exc, (FileNotFoundError, PrerequisiteError)) else "failed"
        result["prerequisite_errors"].append({"type": type(exc).__name__, "message": str(exc)})
    result["ended_at"] = now()
    # Final independent validation before publishing a passing status.
    if result["status"] == "complete":
        try:
            compare_raw(ev.workspace, result["left"], result["right"], args.bounds)
            validate_upstream_evidence(ev, result=result)
        except Exception as exc:
            result["status"] = "failed"
            result["prerequisite_errors"].append({"type": type(exc).__name__, "message": str(exc)})
    write_evidence(ev.workspace, "upstream-result.json", result)
    return result


def flatten_tensors(inputs, prefix=""):
    """Retain every numeric model input without pickle or lossy key mapping."""
    import torch
    result = {}
    require(hasattr(inputs, "items"), "model inputs must be a tensor mapping")
    for name, value in inputs.items():
        require(isinstance(name, str) and "." not in name, "ambiguous model input key")
        key = prefix + name
        if isinstance(value, torch.Tensor):
            result[key] = to_numpy(value)
        elif hasattr(value, "items"):
            result.update(flatten_tensors(value, key + "."))
        else:
            raise PrerequisiteError(f"unsupported nonnumeric model input: {key}")
    require(bool(result), "empty collated inputs")
    return result


@contextmanager
def observe_stock(workspace, backend, identity):
    """Profile a real top-level get_action call, never rebind upstream methods.

    Profiling is suspended while its callback runs. The seeded control invokes
    the same loaded model directly, restores RNG on exit, and checks raw results.
    The actual stock operation and arguments are forwarded untouched.
    """
    import torch
    import torch.nn.functional as functional
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    class Operations(TorchDispatchMode):
        def __init__(self, owner):
            self.owner = owner

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.owner.context()
            out = func(*args, **(kwargs or {}))
            self.owner.dtypes.update(floating_dtypes(args) | floating_dtypes(kwargs or {}) | floating_dtypes(out))
            return out

    class Mode(TorchFunctionMode):
        def __init__(self):
            super().__init__()
            self.dtypes, self.draws, self.sdpa = set(), [], 0
            self.autocast = self.tf32 = self.tf32_matmul = self.tf32_cudnn = False

        def context(self):
            self.autocast |= torch.is_autocast_enabled("cpu") or torch.is_autocast_enabled("cuda")
            self.tf32_matmul |= torch.backends.cuda.matmul.allow_tf32
            self.tf32_cudnn |= torch.backends.cudnn.allow_tf32
            self.tf32 = self.tf32_matmul or self.tf32_cudnn

        def __torch_function__(self, func, types, args=(), kwargs=None):
            self.context()
            out = func(*args, **(kwargs or {}))
            if func is torch.randn:
                self.draws.append(out.detach().clone())
            if func is functional.scaled_dot_product_attention:
                self.sdpa += 1
            return out

    active, observations = {}, []
    expected_class = "Gr00tN1d7" if backend == "native" else "GR00TN17"

    def profile(frame, event, result):
        if frame.f_code.co_name != "get_action":
            return
        model = frame.f_locals.get("self")
        if type(model).__name__ != expected_class:
            return
        if event == "call":
            require(not active and not observations, "stock must execute one full raw prediction")
            inputs = frame.f_locals["inputs"]
            mode = Mode()
            dispatch = Operations(mode)
            steps, backbone, input_dtypes = [], set(), set()

            def before(module, args):
                input_dtypes.update(floating_dtypes(args))

            def after(module, args, value):
                backbone.update(floating_dtypes(value))

            def step(module, args):
                steps.append(1)

            hooks = [model.backbone.register_forward_pre_hook(before),
                     model.backbone.register_forward_hook(after),
                     model.action_head.action_encoder.register_forward_pre_hook(step)]
            active.update(frame=frame, model=model, inputs=flatten_tensors(inputs),
                          rng=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                          mode=mode, dispatch=dispatch, steps=steps, backbone=backbone,
                          input_dtypes=input_dtypes, hooks=hooks)
            mode.__enter__()
            dispatch.__enter__()
        elif event == "return" and active.get("frame") is frame:
            state = dict(active)
            state["dispatch"].__exit__(None, None, None)
            state["mode"].__exit__(None, None, None)
            for hook in state["hooks"]:
                hook.remove()
            active.clear()
            require(result is not None and "action_pred" in result, "stock raw prediction absent")
            raw = result["action_pred"]
            mode = state["mode"]
            require(len(mode.draws) == 1, "stock actual single sampler draw required")
            noise = mode.draws[0]
            after_rng = torch.get_rng_state()
            after_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
            try:
                torch.set_rng_state(state["rng"])
                if state["cuda"]:
                    torch.cuda.set_rng_state_all(state["cuda"])
                with torch.inference_mode():
                    control = model.get_action(frame.f_locals["inputs"], options=frame.f_locals.get("options"))
                inert = torch.equal(raw, control["action_pred"])
            finally:
                torch.set_rng_state(after_rng)
                if after_cuda:
                    torch.cuda.set_rng_state_all(after_cuda)
            config = model.backbone.model.config
            measured = {
                "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters() if p.is_floating_point()}),
                "buffer_dtypes": sorted({str(b.dtype) for b in model.buffers() if b.is_floating_point()}),
                "input_dtypes": sorted(state["input_dtypes"]), "backbone_dtypes": sorted(state["backbone"]),
                "compute_dtypes": sorted(mode.dtypes), "flow_steps": len(state["steps"]),
                "attention": sorted({getattr(c, "_attn_implementation") for c in
                                     (config, config.text_config, config.vision_config)}),
                "eval": all(not m.training for m in model.modules()), "observer_inert": inert,
                "autocast": mode.autocast, "tf32": mode.tf32, "sdpa_calls": mode.sdpa,
                "tf32_matmul": mode.tf32_matmul, "tf32_cudnn": mode.tf32_cudnn,
                "noise_draws": len(mode.draws), "noise_shape": list(noise.shape),
                "noise_dtype": str(noise.dtype), "raw_shape": list(raw.shape), "raw_dtype": str(raw.dtype),
                "device": str(noise.device), "rng_algorithm": f"torch.default_generator.{noise.device.type}",
            }
            observations.append({
                "schema_version": 1, "status": "complete", "evidence_kind": "real_model",
                "backend": backend, **identity, "observed": measured,
                "inputs": write_tensors(workspace, state["inputs"]),
                "raw": write_tensors(workspace, {"raw": to_numpy(raw)}),
                "noise": write_tensors(workspace, {"noise": to_numpy(noise)}),
            })

    require(sys.getprofile() is None, "another Python profiler is active")
    sys.setprofile(profile)
    try:
        yield observations
    finally:
        sys.setprofile(None)
        if active:
            active["dispatch"].__exit__(None, None, None)
            active["mode"].__exit__(None, None, None)
            for hook in active["hooks"]:
                hook.remove()


def stock_worker(args):
    import torch
    backend = "native" if args.stage == "producer" else "lerobot"
    ev = evidence(args.workspace)
    validate_tolerance_agreement(ev, comparison_started_at=now())
    require(load_input_lock(args.corpus, args.checkpoint) == ev.json("input-lock.json"), "stock inputs changed")
    identity = runtime_identity(backend, args.image_digest, ev.json("input-lock.json"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    from replay_groot_native import pinned_native_cache
    scope = pinned_native_cache() if backend == "native" else nullcontext()
    with scope, observe_stock(ev.workspace, backend, identity) as observations:
        if args.stage == "producer":
            sys.argv = [str(args.source / PRODUCER), "--ckpt", str(args.checkpoint),
                        "--out-dir", str(ev.workspace / "stock/producer"), "--tags", "new_embodiment",
                        "--seed", "42", "--device", args.device]
            runpy.run_path(sys.argv[0], run_name="__main__")
            exit_code = 0
        else:
            import pytest
            ref = ev.json("stock/artifact-lock.json")
            ev.bytes(ref["path"], ref["sha256"])  # immediately before trusted stock pickle use

            class Coverage:
                def __init__(self):
                    self.collected, self.reports = [], []

                def pytest_collection_finish(self, session):
                    self.collected = [item.nodeid for item in session.items]

                def pytest_runtest_logreport(self, report):
                    self.reports.append({"nodeid": report.nodeid, "when": report.when,
                                         "outcome": report.outcome, "wasxfail": getattr(report, "wasxfail", None)})

            coverage = Coverage()
            exit_code = int(pytest.main([
                str(args.source / CONSUMER), "-q", "-s", "-c", "/dev/null", "--confcutdir=" + str(args.source / "tests/policies/groot"),
                "--rootdir=" + str(args.source), "--noconftest", "--junitxml=" + str(ev.workspace / "stock/junit.xml"),
                "-o", "cache_dir=/tmp/stock-pytest-cache",
            ], plugins=[coverage]))
            write_evidence(ev.workspace, "stock/coverage.json",
                           {"collected": coverage.collected, "reports": coverage.reports})
    require(len(observations) == 1, "stock observation unavailable or ambiguous")
    write_evidence(ev.workspace, f"stock/{args.stage}-observation.json", observations[0])
    return exit_code


def pin_harness(workspace, source):
    """Read and bind an existing external checkout; never fetch or edit it."""
    source = Path(source).resolve()
    pins = {
        "commit": HARNESS_COMMIT,
        "producer_sha256": sha256_file(contained(source, PRODUCER)),
        "consumer_sha256": sha256_file(contained(source, CONSUMER)),
    }
    validate_harness(source, pins)
    value = {"schema_version": 1, "source": str(source), "harness": pins,
             "inspected_at": now(), "status": "complete",
             "scope": "source inspection only; producer and consumer not executed"}
    write_evidence(workspace, "harness-source.json", value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "check", "pin", "_worker"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--workspace", type=Path, required=True)
        cmd.add_argument("--corpus", type=Path, default=ROOT / "corpus/frozen_v1_0")
        cmd.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/GR00T-N1.7-3B-SO101")
        cmd.add_argument("--native-cache", type=Path, default=Path.home() / ".cache/huggingface")
        cmd.add_argument("--worker-timeout", type=int, default=3600)
        if name == "pin":
            cmd.add_argument("--source", type=Path, required=True)
        if name == "_worker":
            cmd.add_argument("--stage", choices=("producer", "consumer"), required=True)
            cmd.add_argument("--source", type=Path, required=True)
            cmd.add_argument("--device", required=True)
            cmd.add_argument("--image-digest", required=True)
    args = parser.parse_args()
    try:
        if args.command == "_worker":
            return stock_worker(args)
        if args.command == "pin":
            result = pin_harness(args.workspace, args.source)
            print(__import__("json").dumps(result))
            return 0
        result = check(args.workspace) if args.command == "check" else run(args.workspace, args=args)
        print(__import__("json").dumps({"status": result["status"], "message":
                                      "not run" if result["status"] == "not_run" else result["status"]}))
        return {"complete": 0, "failed": 1, "not_run": 2}[result["status"]]
    except Exception as exc:
        missing = isinstance(exc, (FileNotFoundError, PrerequisiteError))
        print(__import__("json").dumps({"status": "not_run" if missing else "failed",
                                      "message": ("not run: " if missing else "") + str(exc)}))
        return 2 if missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
