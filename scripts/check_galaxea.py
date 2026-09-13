"""G05 native backend:2 warmups+12 recorded observations, one seed each, no hardware."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from policy.galaxea.backend import GalaxeaPolicyBackend
from policy.galaxea.modalities import JOINTS
from policy_guard.replay_contract import load_case, read_json, now
from scripts.check_model_swap import select_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior evidence; use a new output directory")
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    lock_path = root / "corpus/phase7-trial3-20260912/input-lock.json"
    lock = read_json(lock_path)
    cases = select_cases(lock)
    backend = GalaxeaPolicyBackend(port=args.port)
    report = {"profile": "g05-so101", "status": "running", "started_at": now(),
              "physical_motion": False, "physical_ready": False,
              "frame_transform": "Galaxea SO101 reference; empirical joint verification pending",
              "input_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
              "warmups": [], "samples": []}
    try:
        backend._session.connect()
        report["initial_health"] = backend._session.greeting["health"]
        for index, record in enumerate([cases[0], cases[0], *cases]):
            seed = record["seeds"][0]
            arrays, _ = load_case(root / "corpus/frozen_v1_0", lock,
                                 {"record": record["file"], "seed": seed})
            observation = dict(zip(JOINTS, map(float, arrays["state"])))
            observation.update(front=arrays["video_front"], wrist=arrays["video_wrist"])
            started = time.perf_counter()
            result = backend.get_action(observation, record["instruction"], seed=seed)
            elapsed_ms = 1000 * (time.perf_counter() - started)
            actions = np.array([[row[key] for key in JOINTS] for row in result], dtype=np.float32)
            if actions.shape != (32, 6) or not np.isfinite(actions).all():
                raise ValueError("Invalid SO101 chunk")
            row = {"record": record["file"], "episode": record["episode_index"], "seed": seed,
                   "rpc_ms": elapsed_ms, "shape": list(actions.shape),
                   "min_per_dim": actions.min(0).tolist(), "max_per_dim": actions.max(0).tolist(),
                   **backend.last_metadata}
            report["warmups" if index < 2 else "samples"].append(row)
            if index >= 2:
                np.save(args.output / f"actions-{index-2:02d}.npy", actions, allow_pickle=False)
            print(f"g05-so101: {index+1}/14, full32-step RPC {elapsed_ms:.1f} ms", flush=True)
        report.update(status="smoke_pass",
                      median_rpc_ms=statistics.median(s["rpc_ms"] for s in report["samples"]),
                      max_rpc_ms=max(s["rpc_ms"] for s in report["samples"]),
                      cot_field_present=any(s.get("cot_text") for s in report["samples"]),
                      natural_language_cot_observed=any(
                          re.sub(r"<[^>]*>|[|\s]", "", s.get("cot_text") or "")
                          for s in report["samples"]))
        report["final_health"] = backend._session.request({"__health__": True})
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        backend.close()
        report["ended_at"] = now()
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
