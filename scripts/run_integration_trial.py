"""One bounded mapped-policy plumbing trial. Prepare is hardware-disconnected."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import yaml
from policy.galaxea.backend import GalaxeaPolicyBackend
from policy.galaxea.modalities import JOINTS
from policy_guard.integration_trial import PROTOCOL, bounded_command, check_camera, joint_limits, probe_target
from scripts.run_checkpoint_sanity import (
    StopLatch, StopGuardedController, armed_stop, validate_loaded_calibration,
)

CONFIG = ROOT / "my-dum-e.yaml"
SOURCES = ("scripts/run_integration_trial.py", "policy_guard/integration_trial.py",
           "policy/galaxea/backend.py", "policy/galaxea/modalities.py",
           "policy/molmo_backend.py", "policy/so101_contract.py",
           "policy/pi05_backend.py", "policy_lab/pi05_so101.py",
           "policy_guard/rtc_trial.py", "policy_lab/runtime.py", "policy_lab/server.py",
           "policy/groot_trial_backend.py",
           "policy_lab/pi05-so101-manifest.json",
           "policy_lab/profiles.py", "policy_lab/protocol.py",
           "scripts/run_checkpoint_sanity.py", "embodiment/so_arm10x/controller.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def construct(stop):
    config = yaml.safe_load(CONFIG.read_text())["controller"]
    settings = {k: config[k] for k in
                ("robot_type", "robot_id", "robot_port", "wrist_cam_idx", "front_cam_idx")}
    # Explicit per-trial settings; never change the user's global configuration.
    controller = StopGuardedController(
        stop=stop, use_degrees=True,
        max_relative_target=PROTOCOL["max_tracking_error"] + 0.25, **settings)
    path = Path(controller.robot.calibration_fpath).resolve(strict=True)
    snapshot = {
        "protocol": PROTOCOL, "controller": settings, "config_sha256": digest(CONFIG),
        "effective_controller": {"use_degrees": controller.config.use_degrees,
                                 "max_relative_target": controller.config.max_relative_target},
        "calibration_path": str(path), "calibration_sha256": digest(path),
        "calibration_mapping": {k: asdict(v) for k, v in controller.robot.calibration.items()},
        "sources": {k: digest(ROOT / k) for k in SOURCES},
    }
    validate_loaded_calibration(controller, snapshot)
    return controller, snapshot


def validate_approval(workspace, snapshot):
    approval = json.loads((workspace / "approval.json").read_text())
    if not (approval.get("operator_present") is True and approval.get("approved") is True
            and approval.get("operator") == "Aaron" and approval.get("user_authorization")):
        raise ValueError("Current operator-present authorization required")
    age = (datetime.now(timezone.utc) -
           datetime.fromisoformat(approval["approved_at"])).total_seconds()
    if not 0 <= age <= 1800 or approval.get("snapshot_sha256") != digest(workspace / "snapshot.json"):
        raise ValueError("Expired or changed trial authorization")
    if snapshot != json.loads((workspace / "snapshot.json").read_text()):
        raise ValueError("Prepared trial inputs changed")


def capture_cameras(settings, workspace):
    import cv2
    result = {}
    for role in ("front", "wrist"):
        cap = cv2.VideoCapture(settings[role + "_cam_idx"], cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(15):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"{role}: camera read failed")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(str(workspace / f"{role}.png"), frame)
            result[role] = check_camera(rgb, role)
        finally:
            cap.release()
    return result


def run(workspace):
    if not sys.stdin.isatty():
        raise ValueError("Interactive stop channel required")
    if (workspace / "result.json").exists():
        raise ValueError("Trial already attempted; use a fresh workspace")
    stop = StopLatch()
    controller, snapshot = construct(stop)
    validate_approval(workspace, snapshot)
    if PROTOCOL["profile"] == "molmoact2-so101":
        from policy.molmo_backend import MolmoPolicyBackend
        policy = MolmoPolicyBackend(language_instruction=PROTOCOL["instruction"])
    elif PROTOCOL["profile"] == "pi05-so101":
        from policy.pi05_backend import Pi05SO101PolicyBackend
        policy = Pi05SO101PolicyBackend(port=PROTOCOL.get("policy_port"),
                                       language_instruction=PROTOCOL["instruction"])
    elif PROTOCOL["profile"] == "groot-so101":
        from policy.groot_trial_backend import GrootTrialBackend
        policy = GrootTrialBackend(calibration_path=snapshot["calibration_path"],
                                   language_instruction=PROTOCOL["instruction"])
    else:
        policy = GalaxeaPolicyBackend(language_instruction=PROTOCOL["instruction"])
    result = {"protocol": PROTOCOL, "status": "failed", "actions": [],
              "motor_targets_sent": False, "chunks": [], "accuracy_scored": False}
    previous = None
    with armed_stop(stop):
        try:
            result["camera_preflight"] = capture_cameras(snapshot["controller"], workspace)
            stop.check()
            # Warm model kernels before connecting/holding the robot.
            from policy_guard.replay_contract import load_case, read_json
            lock = read_json(ROOT / "corpus/phase7-trial3-20260912/input-lock.json")
            arrays, entry = load_case(ROOT / "corpus/frozen_v1_0", lock,
                                     {"record": "record_0005.npz", "seed": 20265907})
            warm_state = arrays["state"]
            if PROTOCOL["profile"] == "groot-so101":
                from policy.so101_contract import to_arm_frame
                warm_state = to_arm_frame(warm_state, "groot-so101",
                                          calibration_path=snapshot["calibration_path"])
            warm = dict(zip(JOINTS, map(float, warm_state)))
            warm.update(front=arrays["video_front"], wrist=arrays["video_wrist"])
            if PROTOCOL["profile"] in ("molmoact2-so101", "pi05-so101", "groot-so101"):
                policy.timeout_s = 180.
            policy.get_action(warm, entry["instruction"])
            if PROTOCOL["scheduler"] == "rtc":
                if not policy.rtc_ping():
                    raise ValueError("Server does not advertise the bounded-prefix RTC contract")
                # Exercise guided kernels before connecting to the arm.
                policy.get_rtc_action(warm, np.tile(warm_state, (25, 1)),
                                      epoch=0, request_id=0, delay_steps=15)
            if PROTOCOL["profile"] in ("molmoact2-so101", "pi05-so101", "groot-so101"):
                policy.timeout_s = 10.
            if PROTOCOL["scheduler"] == "rtc":
                policy.timeout_s = PROTOCOL["rtc_deadline_s"]
            stop.check()
            validate_approval(workspace, snapshot)
            controller.connect(calibrate=False)
            result["hardware_connected"] = True
            result["prearm"] = controller.last_prearm_record
            validate_loaded_calibration(controller, snapshot)
            observation = controller.get_observation()
            for role in ("front", "wrist"):
                check_camera(observation[role], role)
            origin = previous = np.array([observation[k] for k in JOINTS], dtype=float)
            result["origin"] = origin.tolist()
            limits = joint_limits(snapshot["calibration_mapping"])
            if PROTOCOL["scheduler"] == "rtc":
                from policy_guard.rtc_trial import run_rtc_trial
                def check_inputs():
                    if (digest(CONFIG) != snapshot["config_sha256"] or
                            digest(snapshot["calibration_path"]) != snapshot["calibration_sha256"]):
                        raise ValueError("Controller inputs changed during RTC trial")
                run_rtc_trial(
                    controller, policy, stop, workspace, result, origin, limits, check_inputs,
                    chunks=PROTOCOL["chunks"], period_s=PROTOCOL["period_s"],
                    deadline_s=PROTOCOL["rtc_deadline_s"])
            for chunk_index in range(PROTOCOL["chunks"] if PROTOCOL["scheduler"] == "sync" else 0):
                stop.check()
                observation = controller.get_observation()
                for role in ("front", "wrist"):
                    check_camera(observation[role], role)
                chunk_dir = workspace / f"chunk-{chunk_index + 1}"
                chunk_dir.mkdir()
                np.savez_compressed(chunk_dir / "live-observation.npz",
                                    state=np.array([observation[k] for k in JOINTS]), front=observation["front"], wrist=observation["wrist"])
                if PROTOCOL["profile"] not in ("molmoact2-so101", "pi05-so101", "groot-so101"):
                    policy._session.timeout_s = 10
                started = time.monotonic()
                if PROTOCOL.get("motion_probe"):
                    target = probe_target(origin, PROTOCOL["probe_offset_degrees"])
                    if np.any(target < limits[0]) or np.any(target > limits[1]):
                        raise ValueError("Diagnostic target outside calibrated limits")
                    actions = [dict(zip(JOINTS, map(float, target)))
                               for _ in range(PROTOCOL["actions"])]
                    metadata = {"source": "synthetic_shoulder_pan_diagnostic",
                                "model_generated": False}
                else:
                    actions = policy.get_action(observation, PROTOCOL["instruction"])
                    metadata = policy.last_metadata
                result["chunks"].append({"index": chunk_index,
                                         "chunk_rpc_ms": (time.monotonic() - started) * 1000,
                                         "model_metadata": metadata})
                if len(actions) != PROTOCOL["actions"]:
                    raise ValueError("Unexpected action horizon")
                write(chunk_dir / "raw-actions.json", actions)
                # Validate the whole chunk before its first physical target.
                for action in actions:
                    if set(action) != set(JOINTS) or not np.isfinite(list(action.values())).all():
                        raise ValueError("Invalid action group or numeric values")
                for index, action in enumerate(actions):
                    tick = time.monotonic()
                    stop.check()
                    if digest(CONFIG) != snapshot["config_sha256"] or digest(snapshot["calibration_path"]) != snapshot["calibration_sha256"]:
                        raise ValueError("Controller inputs changed during trial")
                    if not policy.ping():
                        raise RuntimeError("Policy unavailable; holding last target")
                    stop.check()
                    current = controller.get_observation()
                    for role in ("front", "wrist"):
                        check_camera(current[role], role)
                    observed = np.array([current[k] for k in JOINTS])
                    raw = np.array([action[k] for k in JOINTS])
                    command = bounded_command(
                        raw, observed, previous, origin, limits,
                        max_tracking_error=PROTOCOL.get("max_tracking_error", 0.25))
                    stop.check()
                    sent = controller.set_target_state(dict(zip(JOINTS, map(float, command))))
                    result["motor_targets_sent"] = True
                    previous = command
                    result["actions"].append({
                        "index": len(result["actions"]), "chunk_index": chunk_index,
                        "action_index": index, "time": tick, "observed": observed.tolist(),
                        "raw": raw.tolist(), "bounded": command.tolist(), "sent": sent,
                        "limited": bool(np.any(raw != command)),
                    })
                    time.sleep(max(0, PROTOCOL["period_s"] - (time.monotonic() - tick)))
            result["final_state"] = controller.get_current_state().tolist()
            final = np.asarray(result["final_state"])
            if (np.any(abs(final - origin) > PROTOCOL["max_excursion"] + 0.5)
                    or np.any(final < limits[0]) or np.any(final > limits[1])):
                raise ValueError("Final observed pose outside trial limits")
            stop.check()
            result["status"] = "awaiting_operator_observation"
        except (Exception, KeyboardInterrupt) as exc:
            stop.trip(str(exc) or type(exc).__name__)
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            # Existing guarded follower disconnect closes resources, retaining hold.
            for resource in (controller, policy):
                try:
                    resource.disconnect() if resource is controller else resource.close()
                except Exception as exc:
                    stop.trip("cleanup: " + str(exc))
            if stop.stopped:
                result["status"] = "failed"
            result.update(stop_reason=stop.reason, clamp_warnings=stop.clamp_warnings,
                          stop_events=stop.journal)
            write(workspace / "result.json", result)
    print(json.dumps({"status": result["status"], "actions_sent": len(result["actions"]),
                      "error": result.get("error")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--profile", choices=["g05-so101", "molmoact2-so101", "pi05-so101", "groot-so101"],
                        default="g05-so101")
    parser.add_argument("--scheduler", choices=["sync", "rtc"], default="sync")
    parser.add_argument("--motion-probe", action="store_true",
                        help="Synthetic 6-degree shoulder-pan diagnostic; not model motion")
    args = parser.parse_args()
    if args.scheduler == "rtc":
        if args.profile != "pi05-so101" or args.motion_probe:
            parser.error("RTC is only supported for the Pi0.5 SO101 trial")
        PROTOCOL.update(scheduler="rtc", policy_port=8081, rtc_deadline_s=0.75,
                        rtc_delay_steps=15, rtc_request_remaining=25, action_budget=100)
    if args.motion_probe and args.profile != "g05-so101":
        parser.error("Motion probe is separate from other policy trials")
    if args.profile == "molmoact2-so101":
        PROTOCOL.update(profile=args.profile, actions=30)
    elif args.profile == "pi05-so101":
        PROTOCOL.update(profile=args.profile, actions=50)
    elif args.profile == "groot-so101":
        PROTOCOL.update(profile=args.profile, actions=16)
    if args.motion_probe:
        PROTOCOL.update(profile="so101-motion-diagnostic", motion_probe=True,
                        chunks=1, actions=40, max_tracking_error=3.75, max_excursion=6.0,
                        probe_joint="shoulder_pan.pos", probe_offset_degrees=6.0,
                        instruction="Diagnostic shoulder-pan movement; no task scoring")
    if args.mode == "prepare":
        args.workspace.mkdir(parents=True, exist_ok=False)
        controller, snapshot = construct(StopLatch())
        assert not controller.robot.bus.is_connected
        write(args.workspace / "snapshot.json", snapshot)
        print("Prepared without opening hardware:", args.workspace)
    else:
        run(args.workspace)


if __name__ == "__main__":
    main()
