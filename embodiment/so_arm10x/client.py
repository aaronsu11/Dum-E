"""
Unified client interfaces for interacting with the SO-ARM10x robots using the
new LeRobot `robots` API, while preserving the legacy helper methods expected by
the Dum-E agent. This module provides:

- SO10xRobot: thin wrapper around LeRobot's follower robots (SO-100/101) that
  exposes convenience methods like `move_to_initial_pose`, `move_to_remote_pose`,
  `release_at_remote_pose`, `get_current_images`, and `set_target_state`, built
  on top of the new `Robot` API.
- Gr00tRobotInferenceClient: policy client compatible with the new observation
  schema but keeping a convenient `set_lang_instruction` and stable return
  format.

A lightweight eval entrypoint is kept for manual testing with the new API.

Example usage:
```shell
python -m embodiment.so_arm10x.client \
    --robot_port /dev/tty.usbmodem5A680102371 \
    --robot_type so101_follower \
    --robot_id so101_follower_arm \
    --wrist_cam_idx 0 \
    --front_cam_idx 1 \
    --policy_host 127.0.0.1 \
    --lang_instruction "Grab a banana and put it on the plate"
```
"""

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal, Optional
from pprint import pformat

import draccus
import matplotlib.pyplot as plt
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)  # noqa: F401
from lerobot.utils.utils import init_logging, log_say

from policy.gr00t.service import ExternalRobotInferenceClient

#################################################################################


class Gr00tRobotInferenceClient:
    """Wrapper for the Isaac-GR00T inference service compatible with the new
    observation schema and Dum-E's expectations.

    - Accepts `camera_keys` and `robot_state_keys` to build observations.
    - Provides `set_lang_instruction` and stores `language_instruction`.
    - `get_action` accepts a raw observation dict and returns a list of action dicts.
    - `get_action_from_images_state` is a convenience for legacy flows.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        camera_keys: Optional[List[str]] = None,
        robot_state_keys: Optional[List[str]] = None,
        show_images: bool = False,
        language_instruction: Optional[str] = None,
    ) -> None:
        self.policy = ExternalRobotInferenceClient(host=host, port=port)
        self.camera_keys = camera_keys or ["wrist", "front"]
        self.robot_state_keys = robot_state_keys or [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]
        self.show_images = show_images
        self.language_instruction = language_instruction
        assert (
            len(self.robot_state_keys) == 6
        ), f"robot_state_keys should be size 6, but got {len(self.robot_state_keys)}"
        self.modality_keys = ["single_arm", "gripper"]

    def set_lang_instruction(self, lang_instruction: str) -> None:
        self.language_instruction = lang_instruction

    def get_action(
        self, observation_dict: Dict[str, Any], lang: Optional[str] = None
    ) -> List[Dict[str, float]]:
        # Build obs for policy
        obs_dict: Dict[str, Any] = {
            f"video.{key}": observation_dict[key] for key in self.camera_keys
        }

        if self.show_images:
            view_img({k: v for k, v in obs_dict.items() if k.startswith("video.")})

        # Pack state into arrays
        state = np.array([observation_dict[k] for k in self.robot_state_keys])
        obs_dict["state.single_arm"] = state[:5].astype(np.float64)
        obs_dict["state.gripper"] = state[5:6].astype(np.float64)
        obs_dict["annotation.human.task_description"] = (
            lang or self.language_instruction
        )

        # Add batch dim
        for k in list(obs_dict.keys()):
            if isinstance(obs_dict[k], np.ndarray):
                obs_dict[k] = obs_dict[k][np.newaxis, ...]
            else:
                obs_dict[k] = [obs_dict[k]]

        # Query policy
        action_chunk = self.policy.get_action(obs_dict)

        # Convert to list of dict[str, float]
        lerobot_actions: List[Dict[str, float]] = []
        horizon = action_chunk[f"action.{self.modality_keys[0]}"].shape[0]
        for i in range(horizon):
            concat_action = np.concatenate(
                [
                    np.atleast_1d(action_chunk[f"action.{key}"][i])
                    for key in self.modality_keys
                ],
                axis=0,
            )
            assert len(concat_action) == len(self.robot_state_keys)
            lerobot_actions.append(
                {
                    key: float(concat_action[idx])
                    for idx, key in enumerate(self.robot_state_keys)
                }
            )
        return lerobot_actions


#################################################################################


# Global figure and axis for continuous streaming
_fig = None
_ax = None


def view_img(img, overlay_img=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env.
    Continuously streams images in the same window without creating new ones.
    """
    global _fig, _ax

    if isinstance(img, dict):
        # stack the images horizontally
        img = np.concatenate([img[k] for k in img], axis=1)

    # Calculate new dimensions while maintaining aspect ratio
    h, w = img.shape[:2]
    scale = max(720 / w, 720 / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Create figure and axis only once
    if _fig is None or _ax is None:
        _fig, _ax = plt.subplots(figsize=(new_w / 100, new_h / 100))
        _ax.set_title("Camera View")
        _ax.axis("off")
        plt.ion()  # Turn on interactive mode
        plt.show(block=False)

    # Clear previous image and display new one
    _ax.clear()
    _ax.imshow(img)
    _ax.set_title("Camera View")
    _ax.axis("off")

    # Update the display
    _fig.canvas.draw()
    _fig.canvas.flush_events()
    plt.pause(0.001)  # Small pause to allow GUI to update


# ============================================================================
# Hardware wrapper built on new LeRobot API, exposing legacy helper methods
# ============================================================================


class SO10xRobot:
    """Thin wrapper for SO-100/101 follower arms using the new LeRobot API.

    Exposes convenience methods used by Dum-E while delegating to the underlying
    `Robot` implementation.
    """

    def __init__(
        self,
        robot_type: str = "so101_follower",
        robot_port: Optional[str] = None,
        robot_id: str = "my_awesome_follower_arm",
        *,
        wrist_cam_idx: int = 0,
        front_cam_idx: int = 1,
        use_degrees: bool = True,
        max_relative_target: Optional[int] = None,
    ) -> None:
        if robot_port is None:
            robot_port = os.getenv("SO_ARM_PORT")
            if not robot_port:
                raise ValueError(
                    "Robot serial port is required. Set `port` or env `SO_ARM_PORT`."
                )

        cameras = {
            "wrist": OpenCVCameraConfig(
                index_or_path=wrist_cam_idx, fps=30, width=640, height=480
            ),
            "front": OpenCVCameraConfig(
                index_or_path=front_cam_idx, fps=30, width=640, height=480
            ),
        }

        # Build the appropriate config subclass for the chosen robot type
        if robot_type == "so101_follower":
            from lerobot.robots.so101_follower import SO101FollowerConfig

            self.config = SO101FollowerConfig(
                id=robot_id,
                port=robot_port,
                cameras=cameras,
                use_degrees=use_degrees,
                max_relative_target=max_relative_target,
            )
        elif robot_type == "so100_follower":
            # Fall back to SO-100 if desired
            from lerobot.robots.so100_follower import SO100FollowerConfig  # type: ignore

            self.config = SO100FollowerConfig(
                id=robot_id,
                port=robot_port,
                cameras=cameras,
                use_degrees=use_degrees,
                max_relative_target=max_relative_target,
            )
        else:
            raise ValueError(f"Unsupported robot_type: {robot_type}")

        self.robot: Robot = make_robot_from_config(self.config)

        # Cache ordering used for vector<->dict conversions
        self._state_keys: List[str] = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

    @property
    def camera_keys(self) -> List[str]:
        return list(self.robot.cameras.keys())

    @property
    def robot_state_keys(self) -> List[str]:
        return list(self._state_keys)

    @contextmanager
    def activate(self):
        self.connect()
        try:
            yield self
        finally:
            self.disconnect()

    def connect(self, calibrate: bool = True) -> None:
        self.robot.connect(calibrate=calibrate)
        # Apply our preferred preset on connect
        self.set_so10x_robot_preset()

    def disconnect(self) -> None:
        self.robot.disconnect()

    # ------------------------ Convenience methods ------------------------
    def set_so10x_robot_preset(self) -> None:
        """Adjust controller gains to reduce shakiness. Best-effort with new API."""
        try:
            with self.robot.bus.torque_disabled():
                for motor in self.robot.bus.motors:
                    self.robot.bus.write("P_Coefficient", motor, 10)
                    self.robot.bus.write("I_Coefficient", motor, 0)
                    self.robot.bus.write("D_Coefficient", motor, 5)
        except Exception:
            # Keep silent if firmware/register names differ
            pass

    def move_to_initial_pose(self) -> None:
        # These target degrees mirror legacy behavior
        self.set_target_state(
            np.array([0.0, -102, 96.0, 76.0, -90.0, 0.0], dtype=np.float64)
        )
        time.sleep(1.0)

    def move_to_ready_pose(self) -> None:
        self.set_target_state(
            np.array([0.0, -90, 75.0, 75.0, -90.0, 0.0], dtype=np.float64)
        )
        time.sleep(1.0)

    def move_to_remote_pose(self) -> None:
        self.set_target_state(
            np.array([0.0, 0.0, 0.0, 50.0, -90.0, 60.0], dtype=np.float64)
        )
        time.sleep(1.0)

    def release_at_remote_pose(self, location: Literal["left", "right"]) -> None:
        """
        This is a pre-defined sequence of poses that the robot will move to release the item relative to the front camera.
        """
        gripper_state = float(self.get_current_state()[-1])
        random_offset = float(np.random.uniform(-5.0, 5.0))
        if location == "left":
            random_offset += 45.0
        else:
            random_offset -= 45.0

        sequence = [
            [random_offset, 0.0, 0.0, 50.0, -90.0, gripper_state],
            [random_offset, 45.0, -45.0, 50.0, -90.0, gripper_state],
            [random_offset, 45.0, -45.0, 50.0, -90.0, min(gripper_state + 10.0, 60.0)],
            [random_offset, 0.0, 0.0, 50.0, -90.0, min(gripper_state + 10.0, 60.0)],
        ]
        for state in sequence:
            self.set_target_state(np.array(state, dtype=np.float64))
            time.sleep(0.5)
        self.move_to_remote_pose()

    def get_observation(self) -> Dict[str, Any]:
        return self.robot.get_observation()

    def get_current_state(self) -> np.ndarray:
        obs = self.get_observation()
        return np.array([float(obs[k]) for k in self._state_keys], dtype=np.float64)

    def get_current_images(self) -> Dict[str, np.ndarray]:
        obs = self.get_observation()
        images: Dict[str, np.ndarray] = {}
        for cam in self.camera_keys:
            images[cam] = obs[cam]
        return images

    def set_target_state(self, target_state: Any) -> Dict[str, float]:
        """Accepts a 6-vector (np/torch) or a dict of `*.pos` keys."""
        action_dict: Dict[str, float]
        if isinstance(target_state, dict):
            action_dict = {str(k): float(v) for k, v in target_state.items()}
        else:
            # numpy / torch tensor
            if hasattr(target_state, "detach"):
                target_state = target_state.detach().cpu().numpy()
            target_state = np.asarray(target_state, dtype=np.float64).reshape(-1)
            assert target_state.shape[0] == 6, "Expected 6-dof target state"
            action_dict = {
                k: float(target_state[i]) for i, k in enumerate(self._state_keys)
            }

        sent = self.robot.send_action(action_dict)
        return {k: float(v) for k, v in sent.items()}


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
    action_horizon: int = 8
    lang_instruction: str = "Grab a banana and put it on the plate"
    play_sounds: bool = False
    timeout: int = 60
    show_images: bool = True


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Step 1: Initialize the robot (wrapper)
    robot = SO10xRobot(
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
    print("robot_state_keys:", robot_state_keys)

    # Step 2: Initialize the policy
    policy = Gr00tRobotInferenceClient(
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
    eval()
