"""
Synchronous, event-loop-free robot Skills for the SO-ARM10x embodiment.

A ``Skill`` is a self-contained robot operation with a uniform synchronous
``run()`` entry point. Skills orchestrate the ``SO10xArmController`` (hardware)
and the ``Gr00tRobotInferenceClient`` (policy) only -- they hold NO agent state
(no task_manager / message_broker) and have NO knowledge of the LLM framework or
the event loop.

This separation creates a clean, framework-agnostic seam: the async agent
offloads each blocking Skill into a worker thread (see ``agent.py``), and a
future ROS2 node could call ``Skill.run()`` directly without dragging in the
event loop or the LLM framework.

Design notes:
- ``Skill`` mirrors the ``IRobotController`` ABC convention in ``shared`` (class
  attributes + ``@abstractmethod``).
- ``time.sleep`` IS allowed here: Skills run inside a worker thread, never on
  the event loop.
- Skills return RAW images / state dicts. The agent's tool adapter is
  responsible for JPEG-encoding and building the ``{"status", "content": [...]}``
  response shape -- Skills stay UI/transport agnostic.

This module deliberately imports neither the event-loop library nor the LLM
framework; that property is asserted in the plan's automated check. Do not add
those imports here -- the thread offload happens in the agent adapter, not the
Skill.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

from tqdm import tqdm


class Skill(ABC):
    """Base class for synchronous robot operations.

    A Skill bundles a robot controller and a policy client and exposes a single
    synchronous ``run(**params)`` entry point returning a result dict (typically
    raw camera images). Subclasses set ``name``/``description`` class attributes.
    """

    #: Stable tool-facing name (matches the agent @tool name it backs).
    name: str = "skill"
    #: Human-readable description of the operation.
    description: str = "A robot skill"

    def __init__(self, controller, gr00t_client) -> None:
        """Store references to the hardware controller and the policy client.

        Args:
            controller: an ``SO10xArmController`` instance (hardware access).
            gr00t_client: a ``Gr00tRobotInferenceClient`` instance (policy).
        """
        self.controller = controller
        self.gr00t_client = gr00t_client

    @abstractmethod
    def run(self, **params) -> Dict[str, Any]:
        """Execute the skill synchronously and return a raw result dict."""
        raise NotImplementedError


class PickSkill(Skill):
    """Dynamic obs -> policy -> action pick loop.

    The body is moved VERBATIM from the former ``agent.pick()`` helper: it drives
    N1.7 inference and streams the resulting action chunk to the arm. The
    ``action_horizon`` default is set to 16 to match the trained/eval checkpoint
    (the legacy default of 8 truncated the learned chunk); it remains
    configurable.
    """

    name = "start_pick"
    description = "Start picking up an item and put it on the plate"

    def run(
        self,
        item: Optional[str] = None,
        actions_to_execute: int = 10,
        pose: Literal["initial", "resume"] = "resume",
        language_instruction: Optional[str] = None,
        action_horizon: int = 16,
    ) -> Dict[str, Any]:
        """Run the pick loop and return the latest camera images.

        Args:
            item: optional item label (used by the agent for its instruction).
            actions_to_execute: number of obs->policy->action iterations.
            pose: "initial" first moves to the initial+ready pose; "resume"
                continues from the current pose.
            language_instruction: instruction to condition the policy on; falls
                back to the client's stored ``language_instruction``.
            action_horizon: number of actions from each chunk to execute
                (default 16 to match the trained checkpoint).

        Returns:
            The latest camera images dict from ``get_current_images()``.
        """
        if pose == "initial":
            self.controller.move_to_initial_pose()
            self.controller.move_to_ready_pose()

        for _ in tqdm(range(actions_to_execute), desc="Executing actions"):
            # New observation -> policy -> action flow using updated interfaces
            observation_dict = self.controller.get_observation()
            action_list = self.gr00t_client.get_action(
                observation_dict,
                language_instruction or self.gr00t_client.language_instruction,
            )

            # Execute a short horizon for stability
            for action_dict in action_list[:action_horizon]:
                self.controller.set_target_state(action_dict)
                time.sleep(0.05)

        time.sleep(0.5)
        return self.controller.get_current_images()


class PlaceSkill(Skill):
    """Static release sequence: drop the grasped item at a remote location."""

    name = "place"
    description = "Place an item at a given location"

    def run(self, location: Literal["left", "right"] = "left", **_) -> Dict[str, Any]:
        """Release the item at ``location`` then return the latest images."""
        self.controller.release_at_remote_pose(location)
        return self.controller.get_current_images()


class ResetPoseSkill(Skill):
    """Static pose: reset the arm to the initial pose to clear the workspace."""

    name = "reset_pose"
    description = "Reset the robot to the initial pose to clear the workspace"

    def run(self, **_) -> Dict[str, Any]:
        """Move to the initial pose then return the latest images."""
        self.controller.move_to_initial_pose()
        return self.controller.get_current_images()
