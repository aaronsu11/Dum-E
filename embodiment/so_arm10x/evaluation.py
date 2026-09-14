"""Legacy standalone SO101 evaluation; hardware IO remains in controller.py."""
import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import Literal
import draccus
from lerobot.utils.utils import init_logging, log_say
from embodiment.so_arm10x.controller import SO10xArmController
from utils import load_config_file

@dataclass
class EvalConfig:
    # SO-ARM10x robot configuration
    robot_type: str = "so101_follower"
    robot_id: str = "my_awesome_follower_arm"
    robot_port: str = "/dev/tty.usbmodem5A680102371"
    wrist_cam_idx: int = 0
    front_cam_idx: int = 1

    # Policy/eval parameters
    policy_host: str = "localhost"
    policy_port: int = 5555
    # Default action horizon pinned to 16 to match the trained/eval checkpoint
    # (the legacy default of 8 truncated the learned action chunk). The agent
    # path uses PickSkill(action_horizon=16).
    action_horizon: int = 16
    lang_instruction: str = "Grab a banana and put it on the plate"
    play_sounds: bool = False
    timeout: int = 60
    show_images: bool = True


@draccus.wrap()
def eval(cfg: EvalConfig):
    from policy.factory import make_policy_backend
    from policy.configuration import PolicyDeployment
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Step 1: Initialize the robot (wrapper)
    robot = SO10xArmController(
        robot_type=cfg.robot_type,
        robot_port=cfg.robot_port,
        robot_id=cfg.robot_id,
        wrist_cam_idx=cfg.wrist_cam_idx,
        front_cam_idx=cfg.front_cam_idx,
    )
    # Safe connection and initial pose handled inside the eval loop via context manager

    # get camera/state keys
    camera_keys = robot.camera_keys
    print("camera_keys:", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction

    # NOTE: for so100/so101, this should be:
    # ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    robot_state_keys = robot.robot_state_keys
    print("robot_state_keys:", robot.robot_state_keys)

    # Step 2: Initialize the policy
    policy = make_policy_backend(
        deployment=PolicyDeployment("so_arm10x", "isaac_groot", "groot", "groot-so101"),
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
        show_images=cfg.show_images,
    )
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # Step 3: Run the Eval Loop with safe connect/disconnect
    try:
        with robot.activate():
            print("Current robot state:", robot.get_current_state())
            robot.move_to_initial_pose()
            robot.move_to_ready_pose()

            while True:
                observation_dict = robot.get_observation()
                print("observation_dict", observation_dict.keys())
                action_list = policy.get_action(observation_dict, language_instruction)

                horizon = min(cfg.action_horizon, len(action_list))
                for i in range(horizon):
                    action_dict = action_list[i]
                    print("action_dict", action_dict.values())
                    robot.set_target_state(action_dict)
                    time.sleep(0.05)
    except KeyboardInterrupt:
        logging.info(
            "KeyboardInterrupt received. Disconnecting robot and exiting eval."
        )
        # Context manager handles disconnect


if __name__ == "__main__":
    # Support both: 1) YAML/JSON config file and 2) draccus CLI
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, help="Path to YAML/JSON config file")
    parser.add_argument("--instruction", type=str, help="Robot instruction")
    args, unknown = parser.parse_known_args()

    if args.config:
        cfg_root = load_config_file(args.config)
        ctrl = (
            cfg_root.get("controller", {})
            if isinstance(cfg_root.get("controller"), dict)
            else {}
        )
        cfg = EvalConfig(
            robot_type=ctrl.get("robot_type", "so101_follower"),
            robot_id=ctrl.get("robot_id", "my_awesome_follower_arm"),
            robot_port=ctrl.get("robot_port", "/dev/ttyUSB0"),
            wrist_cam_idx=int(ctrl.get("wrist_cam_idx", 0)),
            front_cam_idx=int(ctrl.get("front_cam_idx", 1)),
            policy_host=ctrl.get("policy_host", "localhost"),
            policy_port=int(ctrl.get("policy_port", 5555)),
            action_horizon=int(ctrl.get("action_horizon", 16)),
            lang_instruction=args.instruction
            or ctrl.get("lang_instruction", "Grab a banana and put it on the plate"),
            play_sounds=bool(ctrl.get("play_sounds", False)),
            timeout=int(ctrl.get("timeout", 60)),
            show_images=bool(ctrl.get("show_images", False)),
        )
        eval(cfg)
    else:
        # Fall back to draccus CLI (supports the dataclass fields directly)
        eval()
