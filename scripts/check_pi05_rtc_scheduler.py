"""Exercise the physical-trial RTC scheduler over HTTP with a dummy controller."""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from policy.galaxea.modalities import JOINTS
from policy.pi05_backend import Pi05SO101PolicyBackend
from policy_guard.integration_trial import joint_limits
from policy_guard.rtc_trial import run_rtc_trial


class Stop:
    def check(self):
        pass


class DummyController:
    def __init__(self, recording):
        with np.load(recording, allow_pickle=False) as data:
            self.state = data["state"].astype(float).copy()
            self.front, self.wrist = data["front"].copy(), data["wrist"].copy()
        self.count = 0

    def get_observation(self):
        return {**dict(zip(JOINTS, self.state)), "front": self.front, "wrist": self.wrist}

    def set_target_state(self, targets):
        self.state = np.array([targets[k] for k in JOINTS], dtype=float)
        self.count += 1
        return targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    controller = DummyController(args.recording / "chunk-1/live-observation.npz")
    snapshot = json.loads((args.recording / "snapshot.json").read_text())
    policy = Pi05SO101PolicyBackend(port=8081, language_instruction="Grab a banana and put it on the plate")
    result = {"status": "failed", "actions": [], "chunks": [],
              "motor_commands_sent": 0, "controller": "dummy; no hardware imports"}
    try:
        if not policy.rtc_ping():
            raise RuntimeError("RTC server not ready")
        policy.timeout_s = 180.
        obs = controller.get_observation()
        policy.get_action(obs)
        policy.get_rtc_action(obs, np.tile(controller.state, (25, 1)),
                              epoch=0, request_id=0, delay_steps=15)
        policy.timeout_s = .75
        run_rtc_trial(controller, policy, Stop(), args.output, result,
                      controller.state.copy(), joint_limits(snapshot["calibration_mapping"]),
                      lambda: None)
        assert controller.count == 100
        result["status"] = "http_guarded_dummy_rtc_passed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result.pop("motor_targets_sent", None)
        result["dummy_targets"] = controller.count
        if len(result["actions"]) > 1:
            intervals = np.diff([a["time"] for a in result["actions"]]) * 1000
            result["interval_ms"] = {"median": float(np.median(intervals)),
                                     "max": float(intervals.max())}
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        policy.close()


if __name__ == "__main__":
    main()
