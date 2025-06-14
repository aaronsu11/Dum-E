"""
This module provides client interfaces for interacting with the SO100 robot system.

It contains:
- SO100Robot: A class for controlling the physical SO100 robot hardware, including camera
  integration and motor control
- Gr00tRobotInferenceClient: A client for connecting to the Gr00t robot inference service
- Helper functions for visualizing robot camera feeds using matplotlib

The module handles robot initialization, camera setup, movement control, and provides
utilities for both direct hardware control and remote inference-based control.

This module can be run directly to test/evaluate a policy:

Example usage (from the root directory):
    # Test with policy inference
    python -m embodiment.so_arm10x.client --use_policy --host 0.0.0.0  --port 5555 --wrist_cam_idx 2 --front_cam_idx 0 --lang_instruction "Pick up the lego block."

    # Test with dataset playback
    python -m embodiment.so_arm10x.client --dataset_path ~/datasets/so100_pick

Command line arguments:
    --use_policy: Enable policy-based control (default: False, uses dataset playback)
    --dataset_path: Path to dataset for playback (default: ~/datasets/so100_pick)
    --host: Inference server host (default: 10.110.17.183)
    --port: Inference server port (default: 5555)
    --action_horizon: Number of actions to execute per chunk (default: 12)
    --actions_to_execute: Total number of actions to execute (default: 350)
    --wrist_cam_idx: Camera index for wrist camera (default: 0)
    --front_cam_idx: Camera index for front camera (default: 2)
    --lang_instruction: Natural language instruction for the policy (default: "Pick up the lego block")
    --record_imgs: Save camera images during execution (default: False)
"""

import time
from contextlib import contextmanager

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode
from lerobot.common.robot_devices.robots.configs import So100RobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from lerobot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
)

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
# sys.path.append(os.path.expanduser("~/Isaac-GR00T/gr00t/eval/"))
from policy.gr00t.service import ExternalRobotInferenceClient

#################################################################################


class SO100Robot:
    def __init__(
        self, calibrate=False, enable_camera=False, wrist_cam_idx=0, front_cam_idx=2
    ):
        self.config = So100RobotConfig()
        self.calibrate = calibrate
        self.enable_camera = enable_camera
        # self.cam_idx = cam_idx
        if not enable_camera:
            self.config.cameras = {}
        else:
            self.config.cameras = {
                "wrist": OpenCVCameraConfig(wrist_cam_idx, 30, 640, 480, "rgb"),
                "front": OpenCVCameraConfig(front_cam_idx, 30, 640, 480, "rgb"),
            }
        self.config.leader_arms = {}

        # remove the .cache/calibration/so100 folder
        if self.calibrate:
            import os
            import shutil

            calibration_folder = os.path.join(
                os.getcwd(), ".cache", "calibration", "so100"
            )
            print("========> Deleting calibration_folder:", calibration_folder)
            if os.path.exists(calibration_folder):
                shutil.rmtree(calibration_folder)

        # Create the robot
        self.robot = make_robot_from_config(self.config)
        self.motor_bus = self.robot.follower_arms["main"]

    @contextmanager
    def activate(self):
        try:
            self.connect()
            # self.move_to_initial_pose()
            # self.move_to_remote_pose()
            yield
        finally:
            self.disconnect()

    def connect(self):
        if self.robot.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        # Connect the arms
        self.motor_bus.connect()

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

        # Calibrate the robot
        self.robot.activate_calibration()

        self.set_so100_robot_preset()

        # Enable torque on all motors of the follower arms
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)
        # Suppress verbose position logging
        # print("robot present position:", self.motor_bus.read("Present_Position"))
        self.robot.is_connected = True

        self.wrist = self.robot.cameras["wrist"] if self.enable_camera else None
        self.front = self.robot.cameras["front"] if self.enable_camera else None
        if self.wrist is not None:
            self.wrist.connect()
        if self.front is not None:
            self.front.connect()
        # Suppress verbose connection logging
        # print("================> SO100 Robot is fully connected =================")

    def set_so100_robot_preset(self):
        # Mode=0 for Position Control
        self.motor_bus.write("Mode", 0)
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        # self.motor_bus.write("P_Coefficient", 16)
        self.motor_bus.write("P_Coefficient", 10)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.motor_bus.write("I_Coefficient", 0)
        self.motor_bus.write("D_Coefficient", 32)
        # Close the write lock so that Maximum_Acceleration gets written to EPROM address,
        # which is mandatory for Maximum_Acceleration to take effect after rebooting.
        self.motor_bus.write("Lock", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.motor_bus.write("Maximum_Acceleration", 254)
        self.motor_bus.write("Acceleration", 254)

    def move_to_initial_pose(self):
        # current_state = self.robot.capture_observation()["observation.state"]
        # print("current_state", current_state)
        # print all keys of the observation
        # print("observation keys:", self.robot.capture_observation().keys())
        target_state = torch.tensor([0.0, 190.0, 165.0, 60.0, -90.0, 50.0])
        self.robot.send_action(target_state)
        time.sleep(1)
        target_state = torch.tensor([0.0, 190.0, 177.0, 72.0, -90.0, 0.0])
        # current_state = torch.tensor([90, 90, 90, 90, -70, 30])
        self.robot.send_action(target_state)
        time.sleep(1)

    def move_to_remote_pose(self):
        target_state = torch.tensor([0.0, 90.0, 90.0, 50.0, -90.0, 1.0])
        self.robot.send_action(target_state)
        time.sleep(2)

    def release_at_remote_pose(self):
        # get the gripper state value
        gripper_state = self.get_current_state()[-1]
        # random offset to avoid stacking the same object
        random_offset = np.random.uniform(-5.0, 5.0)
        target_state = torch.tensor(
            [random_offset, 90.0, 90.0, 50.0, -90.0, gripper_state]
        )
        self.robot.send_action(target_state)
        time.sleep(1)
        target_state = torch.tensor(
            [random_offset, 60.0, 50.0, 50.0, -90.0, gripper_state]
        )
        self.robot.send_action(target_state)
        time.sleep(1)
        target_state = torch.tensor(
            [random_offset, 60.0, 50.0, 50.0, -90.0, min(gripper_state + 10, 60)]
        )
        self.robot.send_action(target_state)
        time.sleep(1)
        target_state = torch.tensor(
            [random_offset, 90.0, 90.0, 50.0, -90.0, min(gripper_state + 10, 60)]
        )
        self.robot.send_action(target_state)
        time.sleep(1)
        self.move_to_remote_pose()

    def go_home(self):
        # [ 88.0664, 156.7090, 135.6152,  83.7598, -89.1211,  16.5107]
        home_state = torch.tensor(
            [88.0664, 156.7090, 135.6152, 83.7598, -89.1211, 16.5107]
        )
        self.set_target_state(home_state)
        time.sleep(2)

    def get_observation(self):
        return self.robot.capture_observation()

    def get_current_state(self):
        return self.get_observation()["observation.state"].data.numpy()

    def get_current_images(self) -> dict[str, np.ndarray]:
        wrist_img = self.get_observation()["observation.images.wrist"].data.numpy()
        front_img = self.get_observation()["observation.images.front"].data.numpy()
        # convert bgr to rgb
        # wrist_img = cv2.cvtColor(wrist_img, cv2.COLOR_BGR2RGB)
        # front_img = cv2.cvtColor(front_img, cv2.COLOR_BGR2RGB)
        return {"wrist": wrist_img, "front": front_img}

    @staticmethod
    def interpolate_actions(actions, num_interp=5):
        """
        Linearly interpolate between each pair of actions.
        actions: np.ndarray of shape (N, D)
        num_interp: number of interpolated points between each pair
        Returns: np.ndarray of shape (N-1)*num_interp + 1, D
        """
        actions = np.asarray(actions)
        if actions.ndim == 1:
            actions = actions[:, None]  # Convert to (N, 1)
        N, D = actions.shape
        interp_actions = []
        for i in range(N - 1):
            start = actions[i]
            end = actions[i + 1]
            for alpha in np.linspace(0, 1, num_interp, endpoint=False):
                interp_actions.append((1 - alpha) * start + alpha * end)
        interp_actions.append(actions[-1])
        return np.array(interp_actions)

    @classmethod
    def interpolate_actions_with_prev_state(
        cls, prev_state, action, num_interp: int = 3
    ):
        """
        Interpolates between the previous state and the predicted actions to ensure smooth transitions.

        Args:
            prev_state: The previous state array (length 6: 5 for single arm, 1 for gripper).
            action: A dict with keys "action.single_arm" and "action.gripper".

        Returns:
            Tuple of (single_arm_interp, gripper_interp)
        """
        prev_single_arm = prev_state[:5]
        prev_gripper = prev_state[5:]

        single_arm_pred = action["action.single_arm"]
        gripper_pred = action["action.gripper"].reshape(-1, 1)

        # Interpolate between prev_state and first predicted action to avoid abrupt change
        single_arm_first = np.vstack([prev_single_arm, single_arm_pred[0]])
        gripper_first = np.vstack([prev_gripper, gripper_pred[0]])
        single_arm_interp_first = cls.interpolate_actions(
            single_arm_first, num_interp=10
        )
        gripper_interp_first = cls.interpolate_actions(gripper_first, num_interp=10)

        # Interpolate between predicted actions to smooth out the actions
        if len(single_arm_pred) > 1:
            single_arm_interp_rest = cls.interpolate_actions(
                single_arm_pred, num_interp=num_interp
            )
            gripper_interp_rest = cls.interpolate_actions(
                gripper_pred, num_interp=num_interp
            )
            # Concatenate, skipping the first point of the rest to avoid duplicate
            single_arm_interp = np.concatenate(
                [single_arm_interp_first, single_arm_interp_rest[1:]], axis=0
            )
            gripper_interp = np.concatenate(
                [gripper_interp_first, gripper_interp_rest[1:]], axis=0
            )
        else:
            single_arm_interp = single_arm_interp_first
            gripper_interp = gripper_interp_first

        return single_arm_interp, gripper_interp

    def set_target_state(self, target_state: torch.Tensor):
        self.robot.send_action(target_state)

    def enable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.ENABLED.value)

    def disable(self):
        self.motor_bus.write("Torque_Enable", TorqueMode.DISABLED.value)

    def disconnect(self):
        self.disable()
        self.robot.disconnect()
        self.robot.is_connected = False
        # Suppress verbose disconnection logging
        # print("================> SO100 Robot disconnected")

    def __del__(self):
        try:
            self.disconnect()
        except RobotDeviceNotConnectedError:
            pass  # Already disconnected, nothing to do
        except Exception:
            pass  # Optionally ignore all exceptions during shutdown


#################################################################################


class Gr00tRobotInferenceClient:
    def __init__(
        self,
        host="localhost",
        port=5555,
        language_instruction="Pick up the lego block",
    ):
        self.language_instruction = language_instruction
        # 480, 640
        self.img_size = (480, 640)
        self.policy = ExternalRobotInferenceClient(host=host, port=port)

    def get_action(self, images: dict[str, np.ndarray], state: np.ndarray):
        obs_dict = {
            "video.wrist": images["wrist"][np.newaxis, :, :, :],
            "video.front": images["front"][np.newaxis, :, :, :],
            "state.single_arm": state[:5][np.newaxis, :].astype(np.float64),
            "state.gripper": state[5:6][np.newaxis, :].astype(np.float64),
            "annotation.human.task_description": [self.language_instruction],
        }
        res = self.policy.get_action(obs_dict)
        # print("Inference query time taken", time.time() - start_time)
        return res

    def sample_action(self):
        obs_dict = {
            "video.webcam": np.zeros(
                (1, self.img_size[0], self.img_size[1], 3), dtype=np.uint8
            ),
            "state.single_arm": np.zeros((1, 5)),
            "state.gripper": np.zeros((1, 1)),
            "annotation.human.action.task_description": [self.language_instruction],
        }
        return self.policy.get_action(obs_dict)

    def set_lang_instruction(self, lang_instruction):
        self.language_instruction = lang_instruction


#################################################################################


def view_img(img, img2=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    also able to overlay the image to ensure camera view is alligned to training settings
    """
    plt.imshow(img)
    if img2 is not None:
        plt.imshow(img2, alpha=0.5)
    plt.axis("off")
    plt.pause(0.001)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


def view_images(images_dict: dict[str, np.ndarray]):
    """
    Display multiple images side by side in the same window.

    Args:
        images_dict: Dictionary of {name: image} pairs to display
    """
    num_images = len(images_dict)
    if num_images == 0:
        return

    # Use a specific figure ID to reuse the same window
    fig = plt.figure(num=1, figsize=(5 * num_images, 5), clear=True)

    # Create a grid of subplots
    for i, (name, img) in enumerate(images_dict.items()):
        ax = fig.add_subplot(1, num_images, i + 1)
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.pause(0.001)  # Non-blocking show


#################################################################################

if __name__ == "__main__":
    import argparse
    import os

    default_dataset_path = os.path.expanduser("~/datasets/so100_pick")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_policy", action="store_true"
    )  # default is to playback the provided dataset
    parser.add_argument("--dataset_path", type=str, default=default_dataset_path)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--action_horizon", type=int, default=12)
    parser.add_argument("--actions_to_execute", type=int, default=350)
    parser.add_argument("--wrist_cam_idx", type=int, default=0)
    parser.add_argument("--front_cam_idx", type=int, default=2)
    parser.add_argument(
        "--lang_instruction", type=str, default="Pick up the lego block."
    )
    parser.add_argument("--record_imgs", action="store_true")
    args = parser.parse_args()

    # print lang_instruction
    print("lang_instruction: ", args.lang_instruction)

    ACTIONS_TO_EXECUTE = args.actions_to_execute
    USE_POLICY = args.use_policy
    ACTION_HORIZON = (
        args.action_horizon
    )  # we will execute only some actions from the action_chunk of 16
    MODALITY_KEYS = ["single_arm", "gripper"]
    if USE_POLICY:
        client = Gr00tRobotInferenceClient(
            host=args.host,
            port=args.port,
            language_instruction=args.lang_instruction,
        )

        if args.record_imgs:
            # create a folder to save the images and delete all the images in the folder
            os.makedirs("eval_images", exist_ok=True)
            for file in os.listdir("eval_images"):
                os.remove(os.path.join("eval_images", file))

        robot = SO100Robot(
            calibrate=False,
            enable_camera=True,
            wrist_cam_idx=args.wrist_cam_idx,
            front_cam_idx=args.front_cam_idx,
        )
        image_count = 0
        with robot.activate():
            print(robot.get_current_state())
            robot.move_to_initial_pose()
            prev_state = robot.get_current_state()
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Executing actions"):
                images = robot.get_current_images()
                view_images(images)
                state = robot.get_current_state()
                action = client.get_action(images=images, state=state)

                # Interpolate actions for smooth transition
                single_arm_interp, gripper_interp = (
                    robot.interpolate_actions_with_prev_state(prev_state, action, 1)
                )

                start_time = time.time()

                # Use interpolated actions (skip the first, which is prev_state)
                for j in range(1, len(single_arm_interp)):
                    # single_arm_interp[j][1] += 2
                    # single_arm_interp[j][2] -= 2
                    concat_action = np.concatenate(
                        [
                            np.atleast_1d(single_arm_interp[j]),
                            np.atleast_1d(gripper_interp[j]),
                        ],
                        axis=0,
                    )
                    assert concat_action.shape == (6,), concat_action.shape
                    robot.set_target_state(torch.from_numpy(concat_action))
                    # time.sleep(0.005)

                    # Skip viewing images to speed up the execution
                    images = robot.get_current_images()
                    view_images(images)

                    if args.record_imgs:
                        # resize the image to 320x240
                        img = cv2.resize(
                            cv2.cvtColor(images["front"], cv2.COLOR_RGB2BGR),
                            (320, 240),
                        )
                        cv2.imwrite(f"eval_images/img_{image_count}.jpg", img)
                        image_count += 1

                    # 0.05*16 = 0.8 seconds
                    print("executing action", j, "time taken", time.time() - start_time)

                # Update prev_state for the next chunk
                prev_state = concat_action

                print("Action chunk execution time taken", time.time() - start_time)
            print("Done all actions")
            robot.go_home()
            print("Done home")
    else:
        # Test Dataset Source https://huggingface.co/datasets/youliangtan/so100_strawberry_grape
        dataset = LeRobotDataset(
            repo_id="",
            root=args.dataset_path,
        )

        robot = SO100Robot(
            calibrate=False,
            enable_camera=True,
            wrist_cam_idx=args.wrist_cam_idx,
            front_cam_idx=args.front_cam_idx,
        )

        with robot.activate():
            print("Run replay of the dataset")
            actions = []
            for i in tqdm(range(ACTIONS_TO_EXECUTE), desc="Loading actions"):
                action = dataset[i]["action"]
                img = dataset[i]["observation.images.front"].data.numpy()
                # original shape (3, 480, 640) for image data
                realtime_img = robot.get_current_images()

                img = img.transpose(1, 2, 0)
                view_img(img, realtime_img["front"])
                actions.append(action)
                robot.set_target_state(action)
                time.sleep(0.05)

            # plot the actions
            plt.plot(actions)
            plt.show()

            print("Done all actions")
            robot.go_home()
            print("Done home")
