"""Three live-input Pi0.5 base chunks, with no motor-output path.

State padding and the third-camera placeholder are evaluation conventions.
No SO101 semantic action mapping or physical policy readiness is claimed.
"""
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from policy.galaxea.modalities import JOINTS
from policy_lab.profiles import get_profile
from policy_lab.protocol import encode_image
from scripts.check_model_swap import request
from scripts.run_checkpoint_sanity import StopLatch
from scripts.run_integration_trial import capture_cameras, construct, digest, write


def forbid_motor_write(*args, **kwargs):
    raise RuntimeError("Motor register writes are disabled for the base-policy input check")


def capture(workspace):
    import cv2
    controller, snapshot = construct(StopLatch())
    bus = controller.robot.bus
    # Read requests still use the serial transport. Motor register mutation,
    # torque changes and follower/controller connect are never invoked.
    bus.write = bus.sync_write = bus.enable_torque = bus.disable_torque = forbid_motor_write
    bus.write_calibration = forbid_motor_write
    bus.connect(handshake=False)
    try:
        actual = {k: asdict(v) for k, v in bus.read_calibration().items()}
        if actual != snapshot["calibration_mapping"]:
            raise ValueError("Actual motor calibration does not match the loaded file")
        cameras = capture_cameras(snapshot["controller"], workspace)
        values = bus.sync_read("Present_Position")
        state = np.array([values[k.removesuffix(".pos")] for k in JOINTS], dtype=np.float32)
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise ValueError("Invalid live state")
    frames = {role: cv2.cvtColor(cv2.imread(str(workspace / f"{role}.png")),
                                cv2.COLOR_BGR2RGB) for role in ("front", "wrist")}
    np.savez_compressed(workspace / "observation.npz", state=state, **frames)
    write(workspace / "capture.json", {
        "controller": snapshot["controller"], "calibration_path": snapshot["calibration_path"],
        "calibration_sha256": snapshot["calibration_sha256"],
        "actual_calibration": actual, "cameras": cameras,
        "motor_register_writes": 0, "arm_joint_units": "centered degrees",
        "gripper_units": "0-100 points"})
    return state, frames


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=False)
    profile = get_profile("pi05-base").to_dict()
    endpoint = "http://127.0.0.1:8081"
    report = {"profile": profile, "chunks": [], "status": "failed",
              "motor_commands_sent": 0, "physical_policy_ready": False,
              "mapping": "six raw live joints padded to 32; front/base_0, wrist/left_wrist_0; "
                         "right_wrist_0 is a zero placeholder; no SO101 normalization",
              "source_sha256": digest(__file__)}
    try:
        health = request(endpoint, "/health")
        if health["status"] != "ready" or health["profile"] != profile:
            raise ValueError("Pi0.5 readiness/profile mismatch")
        for index in range(3):
            path = args.workspace / f"chunk-{index + 1}"
            path.mkdir()
            state, frames = capture(path)
            data = {"state": state.tolist(), **{k: encode_image(v) for k, v in frames.items()},
                    "task": "Grab a banana and put it on the plate", "seed": 20265907}
            start = time.monotonic()
            reply = request(endpoint, "/infer", data)
            elapsed = (time.monotonic() - start) * 1000
            if (reply["health"]["status"] != "ready" or reply["health"]["profile"] != profile
                    or reply.get("seed") != data["seed"]):
                raise ValueError("Inference response identity/status mismatch")
            values = np.asarray(reply["actions"])
            if (values.shape != (50, 32) or values.dtype.kind not in "fiu"
                    or not np.isfinite(values).all()):
                raise ValueError("Expected finite 50x32 Pi0.5 base output")
            write(path / "response.json", reply)
            entry = {"index": index, "rpc_ms": elapsed, "shape": list(values.shape),
                     "timings": reply["timings"], "health": reply["health"]}
            report["chunks"].append(entry)
            print(json.dumps({"chunk": index + 1, "rpc_ms": elapsed,
                              "shape": list(values.shape), "motor_commands_sent": 0}), flush=True)
        report["status"] = "live_input_inference_passed_output_mapping_unverified"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write(args.workspace / "result.json", report)


if __name__ == "__main__":
    main()
