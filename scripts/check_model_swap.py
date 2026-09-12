"""Twelve frozen observations, one per episode; one seed, no hardware imports."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from policy_lab.profiles import get_profile
from policy_lab.protocol import encode_image
from policy_guard.replay_contract import load_case, read_json, now


def request(endpoint, path, data=None):
    raw = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(endpoint + path, data=raw,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(8192).decode("utf-8", errors="replace")
        raise RuntimeError(f"Server HTTP {exc.code}: {detail}") from exc


def select_cases(lock):
    episodes = {}
    for record in lock["records"]:
        episodes.setdefault(record["episode_index"], record)
    if len(episodes) != 12:
        raise ValueError(f"Expected 12 recorded episodes, found {len(episodes)}")
    return list(episodes.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    endpoint = urlparse(args.endpoint)
    if (endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1"
            or endpoint.path not in ("", "/") or endpoint.username or endpoint.query):
        raise ValueError("Use loopback, including an SSH tunnel for a remote server")
    profile = get_profile(args.profile)
    if args.output.exists():
        raise FileExistsError("Preserve prior evidence; use a new output directory")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "corpus/phase7-trial3-20260912/input-lock.json"
    lock = read_json(lock_path)
    cases = select_cases(lock)
    report = {"profile": profile.to_dict(), "started_at": now(), "status": "running",
              "physical_motion": False, "physical_ready": False, "warmups": [], "samples": [],
              "input_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
              "evaluation": "Shape, finiteness, GPU memory and timing only; not task success or GR00T parity."}
    try:
        health = request(args.endpoint, "/health")
        if health["profile"] != profile.to_dict() or health["status"] != "ready":
            raise ValueError("Server profile/readiness mismatch")
        report["initial_health"] = health
        # Warmups are separate from the twelve single-seed observations.
        for index, record in enumerate([cases[0], cases[0], *cases]):
            seed = record["seeds"][0]
            arrays, _ = load_case(root / "corpus/frozen_v1_0", lock,
                                 {"record": record["file"], "seed": seed})
            data = {"state": arrays["state"].tolist(), "front": encode_image(arrays["video_front"]),
                    "wrist": encode_image(arrays["video_wrist"]), "task": record["instruction"], "seed": seed}
            start = time.perf_counter()
            result = request(args.endpoint, "/infer", data)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if result["health"]["profile"] != profile.to_dict() or result["physical_ready"] is not False:
                raise ValueError("Server identity/scope changed during run")
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape != (profile.horizon, profile.action_dim) or not np.isfinite(actions).all():
                raise ValueError("Invalid model output")
            row = {"record": record["file"], "episode": record["episode_index"], "seed": seed,
                   "rpc_ms": elapsed_ms, "timings": result["timings"], "shape": list(actions.shape),
                   "min_per_dim": actions.min(0).tolist(), "max_per_dim": actions.max(0).tolist(),
                   "peak_allocated_mib": result["health"]["peak_allocated_mib"]}
            if index < 2:
                report["warmups"].append(row)
            else:
                np.save(args.output / f"actions-{index-2:02d}.npy", actions, allow_pickle=False)
                report["samples"].append(row)
            print(f"{args.profile}: {index+1}/14, RPC {elapsed_ms:.1f} ms", flush=True)
        report["status"] = "smoke_pass"
        report["median_rpc_ms"] = statistics.median(row["rpc_ms"] for row in report["samples"])
        report["max_rpc_ms"] = max(row["rpc_ms"] for row in report["samples"])
        report["final_health"] = request(args.endpoint, "/health")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["ended_at"] = now()
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
