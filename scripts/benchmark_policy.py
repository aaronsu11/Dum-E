"""Recorded-input latency/contract smoke. This command never connects robot hardware."""
import argparse
import json
from pathlib import Path
import sys
import subprocess
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from embodiment.so_arm10x.schema import JOINT_ORDER
from policy.evidence import now, sha256_file
from embodiment.so_arm10x.observation import load_observation


def http_request(port, path, data=None):
    endpoint = f"http://127.0.0.1:{port}{path}"
    request = urllib.request.Request(endpoint, data=None if data is None else json.dumps(data, allow_nan=False).encode(),
                                    headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        if response.url != endpoint:
            raise ValueError("Server redirected")
        return json.loads(response.read(1024 * 1024))


def serving_identity(container, port):
    """Bind async timing to a local loopback server, mode and measured source bytes."""
    raw = subprocess.check_output(["docker", "inspect", container], text=True, timeout=10)
    info, = json.loads(raw)
    env = dict(v.split("=", 1) for v in info["Config"].get("Env", []) if "=" in v)
    bindings = info["NetworkSettings"].get("Ports", {}).get("8080/tcp") or []
    if (not info["State"]["Running"] or
            not any(b["HostIp"] == "127.0.0.1" and b["HostPort"] == str(port) for b in bindings) or
            env.get("DUME_CHUNK_OBSERVER", "lightweight") != "lightweight" or
            env.get("DUME_PARITY_ATTESTATION_PATH") or env.get("DUME_POLICY_SEED") or
            "serve_observed_lerobot.py" not in " ".join([info["Path"], *info["Args"]])):
        raise ValueError("Async timing requires the documented loopback lightweight server with ambient RNG")
    names = ["policy/backends/lerobot/serve.py", "policy/telemetry.py", "policy/backends/lerobot/server.py"]
    probe = "import hashlib,json,pathlib; names=" + repr(names) + "; print(json.dumps({n:hashlib.sha256(pathlib.Path('/app',n).read_bytes()).hexdigest() for n in names}))"
    hashes = json.loads(subprocess.check_output(["docker", "exec", container, "python3", "-c", probe], text=True, timeout=15))
    root = Path(__file__).resolve().parents[1]
    if hashes != {name: sha256_file(root/name) for name in names}:
        raise ValueError("Measured server sources differ from checkout")
    return {"container_id": info["Id"], "image_id": info["Image"], "started_at": info["State"]["StartedAt"],
            "observer_mode": "lightweight", "source_files": hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["g05-so101", "pi05-base", "pi05-so101", "molmoact2-so101", "groot-so101", "lerobot-gr00t"])
    parser.add_argument("--observations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--port", type=int)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20265907)
    parser.add_argument("--latency-settings", action="store_true", help="LeRobot: collect at least 100 samples for existing async admission gate")
    parser.add_argument("--server-container", help="Required for local GR00T async admission measurement")
    parser.add_argument("--deployment", type=Path, help="Explicit deployment YAML; replaces --profile")
    args = parser.parse_args()
    if args.deployment:
        from policy.configuration import load_deployment, profile_for_deployment
        if args.profile is not None:
            parser.error("Use --deployment or --profile, not both")
        deployment = load_deployment(args.deployment)
        args.profile = profile_for_deployment(deployment)
        if deployment.execution == "rtc":
            parser.error("Use check_pi05_rtc_scheduler.py for RTC; this benchmark measures complete chunks")
    if not args.profile:
        parser.error("--profile or --deployment is required")

    if not 1 <= args.samples <= 1000 or not 0 <= args.warmups <= 10 or not args.instruction.strip():
        parser.error("Positive bounded sample count, 0–10 warmups and instruction required")
    if args.latency_settings and (args.profile != "lerobot-gr00t" or args.samples < 100 or not args.server_container):
        parser.error("Async admission requires lerobot-gr00t, --server-container and at least 100 samples")
    port = args.port or (8765 if args.profile == "g05-so101" else 8080 if args.profile == "lerobot-gr00t" else 8081)
    if not 1 <= port <= 65535:
        parser.error("Invalid port")
    binding = serving_identity(args.server_container, port) if args.latency_settings else None
    observations = [load_observation(p) for p in args.observations]
    policy = None
    expected = {"g05-so101": (32, 6), "pi05-base": (50, 32), "pi05-so101": (50, 6),
                "molmoact2-so101": (30, 6), "groot-so101": (16, 6), "lerobot-gr00t": (16, 6)}[args.profile]
    if args.profile != "pi05-base":
        from policy.configuration import deployment_for_profile
        from policy.factory import make_policy_backend
        deployment = deployment_for_profile(args.profile)
        kwargs = {"port": port}
        if args.profile in ("groot-so101", "lerobot-gr00t"):
            if not args.calibration:
                parser.error("GR00T requires --calibration")
            if args.profile == "groot-so101":
                kwargs["calibration_path"] = args.calibration
            else:
                from embodiment.so_arm10x.mappings.frames import to_model_frame
                for obs in observations:
                    state = to_model_frame([obs[k] for k in JOINT_ORDER], "groot-so101",
                                           calibration_path=args.calibration)
                    obs.update(zip(JOINT_ORDER, map(float, state)))
                # Serialized exchange is required for the admission benchmark.
                deployment = deployment_for_profile(args.profile, execution="async")
        policy = make_policy_backend(deployment=deployment, **kwargs)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"status": "failed", "profile": args.profile, "motor_commands_sent": 0,
              "started_at": now(), "endpoint": f"127.0.0.1:{port}",
              "inputs": {str(p): sha256_file(p) for p in args.observations},
              "seed": None if args.profile == "lerobot-gr00t" else args.seed,
              "seed_policy": "ambient" if args.profile == "lerobot-gr00t" else "fixed",
              "warmups": [], "samples": [], "scope": "Integration and timing, not task accuracy"}
    try:
        if policy is not None and hasattr(policy, "timeout_s"):
            policy.timeout_s = 180
        for index in range(args.warmups + args.samples):
            obs = observations[max(0, index - args.warmups) % len(observations)]
            started = time.perf_counter()
            if policy is None:
                from policy.checkpoints import get_profile
                from policy.backends.lerobot.transports.http import encode_image
                reply = http_request(port, "/infer", {"state": [obs[k] for k in JOINT_ORDER],
                    "front": encode_image(obs["front"]), "wrist": encode_image(obs["wrist"]),
                    "task": args.instruction, "seed": args.seed})
                if reply["health"]["profile"] != get_profile(args.profile).to_dict() or reply["health"]["status"] != "ready" or reply.get("seed") != args.seed:
                    raise ValueError("Server profile/readiness/seed mismatch")
                actions = np.asarray(reply["actions"])
                metadata = {k: reply[k] for k in ("health", "timings")}
            else:
                kw = {} if args.profile == "lerobot-gr00t" else {"seed": args.seed}
                values = policy.get_action(obs, args.instruction, **kw)
                actions = np.array([[v[k] for k in JOINT_ORDER] for v in values])
                metadata = getattr(policy, "last_metadata", None)
            elapsed = time.perf_counter() - started
            if actions.shape != expected or actions.dtype.kind not in "fiu" or not np.isfinite(actions).all():
                raise ValueError("Invalid complete model output")
            row = {"rpc_ms": elapsed * 1000, "shape": list(actions.shape), "metadata": metadata,
                   "min_per_dim": actions.min(0).tolist(), "max_per_dim": actions.max(0).tolist()}
            report["warmups" if index < args.warmups else "samples"].append(row)
            if index >= args.warmups:
                np.save(args.output / f"actions-{index-args.warmups:03d}.npy", actions, allow_pickle=False)
            print(f"{index+1}/{args.warmups+args.samples}: {elapsed*1000:.1f} ms", flush=True)
        samples = [r["rpc_ms"] / 1000 for r in report["samples"]]
        report.update(status="complete", median_ms=float(np.median(samples) * 1000), max_ms=max(samples)*1000)
        if args.latency_settings:
            if serving_identity(args.server_container, port) != binding:
                raise ValueError("Server changed during latency measurement")
            from policy.execution.asynchronous import AsyncSettings
            p99 = float(np.percentile(samples, 99, method="higher"))
            root = Path(__file__).resolve().parents[1]
            sources = ["policy/backends/lerobot/serve.py", "policy/telemetry.py",
                       "policy/backends/lerobot/serialized_backend.py", "policy/backends/lerobot/server.py"]
            latency = {"kind": "async_request_latency", "status": "complete", "sample_count": len(samples),
                       "samples_s": samples, "request_p99_s": p99, "settings": vars(AsyncSettings(p99)),
                       "observer_mode": "lightweight", "policy_device": "cuda",
                       "source_files": {name: sha256_file(root/name) for name in sources},
                       "server_binding": binding,
                       "limitations": "Empirical sequential latency; use the documented lightweight CUDA server with these sources"}
            (args.output / "latency.json").write_text(json.dumps(latency, indent=2)+"\n")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if policy is not None:
            policy.close()
        report["ended_at"] = now()
        (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
