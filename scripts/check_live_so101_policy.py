"""Check a mapped SO101 policy with live inputs; never dispatch motor targets."""
import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from policy.galaxea.modalities import JOINTS
from policy.pi05_backend import Pi05SO101PolicyBackend
from scripts.check_live_base_policy import capture
from scripts.run_integration_trial import write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=False)
    backend = Pi05SO101PolicyBackend(language_instruction="Grab a banana and put it on the plate")
    report = {"profile": backend.profile.to_dict(), "status": "failed", "chunks": [],
              "motor_commands_sent": 0, "input_mode": "live camera and read-only motor state"}
    try:
        if not backend.ping():
            raise RuntimeError("Pinned Pi0.5 SO101 server is not ready")
        backend.timeout_s = 180.
        for index in range(3):
            directory = args.workspace / f"chunk-{index + 1}"
            directory.mkdir()
            state, frames = capture(directory)
            observation = {**dict(zip(JOINTS, map(float, state))), **frames}
            started = time.monotonic()
            actions = backend.get_action(observation)
            elapsed = 1000 * (time.monotonic() - started)
            raw = np.array([[action[key] for key in JOINTS] for action in actions])
            write(directory / "actions.json", actions)
            entry = {"index": index, "rpc_ms": elapsed, "shape": list(raw.shape),
                     "state": state.tolist(), "target_min": raw.min(axis=0).tolist(),
                     "target_max": raw.max(axis=0).tolist(),
                     "max_raw_target_error": abs(raw - state).max(axis=0).tolist(),
                     "metadata": backend.last_metadata}
            report["chunks"].append(entry)
            print(f"Chunk {index + 1}: {raw.shape}, {elapsed:.1f} ms, no motor dispatch", flush=True)
        report["status"] = "mapped_live_input_inference_passed_physical_trial_pending"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        backend.close()
        write(args.workspace / "result.json", report)


if __name__ == "__main__":
    main()
