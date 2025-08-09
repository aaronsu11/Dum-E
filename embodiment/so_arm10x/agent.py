"""
Refactored robot agent implementing the IRobotAgent interface.

This module provides an SO10x robot agent that implements the abstract
interfaces while maintaining compatibility with the existing voice assistant
streaming pattern. Key improvements include:
- Implements IRobotAgent interface for modularity
- Event publishing for real-time progress updates
- Enhanced error handling and recovery
- Task lifecycle management
- Tool registry integration

Example usage (from the root directory):
    python -m embodiment.so_arm10x.agent --port /dev/tty.usbmodem5A680102371 --wrist_cam_idx 0 --front_cam_idx 1
"""

import asyncio
import os
import time
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Literal, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from loguru import logger
from strands import Agent, tool
from strands.models.anthropic import AnthropicModel
from tqdm import tqdm

from embodiment.so_arm10x import (
    InMemoryEventPublisher,
    InMemoryTaskManager,
    InMemoryToolRegistry,
    SO10xHardwareInterface,
)
from embodiment.so_arm10x.client import Gr00tRobotInferenceClient, SO10xRobot
from interfaces import (
    Event,
    EventType,
    IEventPublisher,
    IHardwareInterface,
    IRobotAgent,
    ITaskManager,
    IToolRegistry,
    TaskStatus,
    ToolDefinition,
)
from logging_config import create_clean_callback_handler, setup_robot_logging

load_dotenv()

# Model configuration
DEFAULT_MODEL_ID = "claude-sonnet-4-20250514"


def image_to_jpeg_bytes(
    image: np.ndarray, save: bool = False, verbose: bool = False
) -> bytes:
    """Convert a numpy image (HWC, RGB or BGR) to JPEG bytes."""
    # If image is RGB, convert to BGR for OpenCV
    if image.shape[-1] == 3 and image.dtype == np.uint8:
        # Heuristic: if max value > 1, assume uint8
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image
    success, buffer = cv2.imencode(".jpg", img_bgr)
    if not success:
        raise ValueError("Could not encode image to JPEG")
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        folder = "./sample_images"
        if not os.path.exists(folder):
            os.makedirs(folder)
        image_path = f"{folder}/image_{timestamp}.jpg"
        cv2.imwrite(image_path, img_bgr)
        if verbose:
            logger.info(f"📷 Saved image: {image_path}")
    return buffer.tobytes()


def create_robot_tools(
    robot_instance: SO10xRobot, gr00t_client_instance: Gr00tRobotInferenceClient
):
    """Create robot tools that use the specific robot instance."""

    @tool
    def reset_pose():
        """Reset the robot to the initial pose to make the workspace clear and visible."""
        robot_instance.move_to_initial_pose()
        return {
            "status": "success",
            "content": [
                {"text": f"Reset the robot to the initial pose."},
            ],
        }

    @tool
    def assess_situation() -> dict:
        """Assess the situation at the current state"""
        images = robot_instance.get_current_images()
        front_image_bytes = image_to_jpeg_bytes(images["front"], verbose=False)
        wrist_image_bytes = image_to_jpeg_bytes(images["wrist"], verbose=False)

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Taken picture from the front camera and wrist camera as follows:"
                },
                {"image": {"format": "jpeg", "source": {"bytes": front_image_bytes}}},
                {"image": {"format": "jpeg", "source": {"bytes": wrist_image_bytes}}},
            ],
        }

    def pick(
        actions_to_execute: int = 10,
        pose: Literal["initial", "resume"] = "resume",
        language_instruction: Optional[str] = None,
    ) -> dict:
        if pose == "initial":
            robot_instance.move_to_initial_pose()
            robot_instance.move_to_ready_pose()

        for _ in tqdm(range(actions_to_execute), desc="Executing actions"):
            # New observation -> policy -> action flow using updated interfaces
            observation_dict = robot_instance.get_observation()
            action_list = gr00t_client_instance.get_action(
                observation_dict,
                language_instruction or gr00t_client_instance.language_instruction,
            )

            # Execute a short horizon for stability
            for action_dict in action_list:
                robot_instance.set_target_state(action_dict)
                time.sleep(0.05)

        time.sleep(0.5)
        return robot_instance.get_current_images()

    @tool
    def start_pick(item: Literal["a banana", "an apple", "an orange"]) -> dict:
        """Start picking up an item and put it on the plate"""
        language_instruction = f"Grab {item} and put it on the plate"
        gr00t_client_instance.set_lang_instruction(language_instruction)
        latest_images = pick(pose="initial", language_instruction=language_instruction)
        image_bytes = image_to_jpeg_bytes(latest_images["front"], verbose=False)

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Attempted picking up {item} with 10 steps. Current state:",
                },
                {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
            ],
        }

    @tool
    def resume_pick():
        """Resume picking up an item from a given location"""
        latest_images = pick(actions_to_execute=15, pose="resume")
        image_bytes = image_to_jpeg_bytes(latest_images["front"], verbose=False)

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Resumed pick-up with 15 steps. Current state:",
                },
                {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
            ],
        }

    @tool
    def place(location: Literal["left", "right"]):
        """Place an item at a given location"""
        robot_instance.release_at_remote_pose(location)
        latest_images = robot_instance.get_current_images()
        image_bytes = image_to_jpeg_bytes(latest_images["front"], verbose=False)

        return {
            "status": "success",
            "content": [
                {
                    "text": f"Attempted to place the item on {location}.",
                },
                {"image": {"format": "jpeg", "source": {"bytes": image_bytes}}},
            ],
        }

    return [reset_pose, assess_situation, start_pick, resume_pick, place]


class SO10xRobotAgent(IRobotAgent):
    """
    SO10x robot agent implementing the IRobotAgent interface.

    Maintains compatibility with existing voice assistant integration while
    adding enhanced features for task management, event streaming, and
    modular tool management.
    """

    def __init__(
        self,
        robot_instance: SO10xRobot,
        gr00t_client_instance: Optional[Gr00tRobotInferenceClient] = None,
        model_id: str = DEFAULT_MODEL_ID,
        region_name: str = "us-west-2",
        callback_handler: Callable = None,
        task_manager: Optional[ITaskManager] = None,
        event_publisher: Optional[IEventPublisher] = None,
        tool_registry: Optional[IToolRegistry] = None,
        hardware_interface: Optional[IHardwareInterface] = None,
    ):
        """
        Initialize the SO10x robot agent with explicit dependency injection.

        Args:
            robot_instance: Required SO100Robot instance for hardware control
            gr00t_client_instance: Optional Gr00t client instance (creates default if not provided)
            model_id: The model ID to use for the agent
            region_name: AWS region name
            callback_handler: Optional callback handler for agent events
            task_manager: Optional task manager instance
            event_publisher: Optional event publisher instance
            tool_registry: Optional tool registry instance
            hardware_interface: Optional hardware interface instance
        """
        # Required robot instance
        self._robot_instance = robot_instance
        self._gr00t_client_instance = (
            gr00t_client_instance or Gr00tRobotInferenceClient()
        )

        # Initialize interfaces with defaults if not provided (suppress verbose logging)
        self.task_manager = task_manager or InMemoryTaskManager()
        self.event_publisher = event_publisher or InMemoryEventPublisher()
        self.tool_registry = tool_registry or InMemoryToolRegistry()
        self.hardware_interface = hardware_interface or SO10xHardwareInterface(
            self._robot_instance
        )

        # Store configuration
        self.model_id = model_id
        self.region_name = region_name
        self.callback_handler = callback_handler or create_clean_callback_handler()

        # Create robot-specific tools
        self._robot_tools = create_robot_tools(
            self._robot_instance, self._gr00t_client_instance
        )

        # Initialize the underlying Strands agent
        self._strands_agent = None
        self._agent_lock = asyncio.Lock()

    async def _get_strands_agent(self) -> Agent:
        """Get or create the underlying Strands agent (lazy initialization)."""
        if self._strands_agent is None:
            async with self._agent_lock:
                if self._strands_agent is None:  # Double-check pattern
                    self._strands_agent = await self._create_strands_agent()
        return self._strands_agent

    async def _create_strands_agent(self) -> Agent:
        """Create the underlying Strands agent with tools."""
        # Create the model (same as original implementation)
        model = AnthropicModel(
            client_args={
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
            },
            max_tokens=8000,
            model_id=self.model_id,
            params={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": 6000,
                },
                "extra_headers": {
                    "anthropic-beta": "interleaved-thinking-2025-05-14",
                },
            },
        )

        # Create agent with robot-specific tools (no longer importing from robot_agent)
        agent = Agent(
            model=model,
            tools=self._robot_tools,
            system_prompt="""
            You are a robot assistant that performs pick and place tasks given a desired state.
            You have access to tools for:
            1. Assessing the current situation by taking pictures from cameras
            2. Starting to pick up a specific item
            3. Resuming pick and place operations from current pose
            4. Placing items at a given location (only when firmly grasped)

            Guidelines for task execution:
            - Always assess the situation and create a plan first
            - Examine results of each tool call during picking carefully and resume picking if target item is not yet delivered
            - If you failed to grasp an item after 2 resume attempts and there is another item in the list, switch to the other item; if there is no other item, reset the robot to the initial pose and report the failure
            - If you firmly grasped the item but failed to deliver it after 2 resume attempts, use the place tool to forcefully place the item on the plate
            - After each pick and place attempt:
              * Assess current situation very carefully against the desired state
              * Briefly describe what was accomplished
              * Determine next steps if needed
              * Continue until desired state is achieved
            - Be very concise in responses (max 15 words) and don't use special characters  
            
            Note: Colors in images may appear different due to reflections.
            """,
            callback_handler=self.callback_handler,
            trace_attributes={
                "session.id": time.strftime("%Y-%m-%d"),
                "user.id": "SO-ARM101",
                "langfuse.tags": ["GR00T-N1.5-3B"],
            },
        )

        return agent

    # IRobotAgent interface implementation

    async def arun(
        self, instruction: str, task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute instruction synchronously and return final result."""
        try:
            # Create task if not provided
            if task_id is None:
                task_id = await self.task_manager.create_task(instruction)

            await self.task_manager.update_task_status(task_id, TaskStatus.RUNNING)

            # Get the Strands agent and execute
            agent = await self._get_strands_agent()

            # Execute with robot hardware context
            with self._robot_instance.activate():
                # Warm up cameras for 1 second
                for _ in range(5):
                    self._robot_instance.get_current_images()
                    await asyncio.sleep(0.2)

                result = agent(instruction)

                # Reset to initial pose after completion
                self._robot_instance.move_to_initial_pose()

            await self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)

            return {
                "task_id": task_id,
                "status": "completed",
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            if task_id:
                await self.task_manager.update_task_status(
                    task_id, TaskStatus.FAILED, str(e)
                )

            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def astream(
        self, instruction: str, task_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute instruction with streaming progress updates.

        Maintains compatibility with existing voice assistant while adding
        enhanced progress tracking and event publishing.
        """
        try:
            # Create task if not provided
            if task_id is None:
                task_id = await self.task_manager.create_task(instruction)

            await self.task_manager.update_task_status(task_id, TaskStatus.RUNNING)

            # Publish task started event
            await self.event_publisher.publish_event(
                Event(
                    event_type=EventType.TASK_STARTED,
                    task_id=task_id,
                    timestamp=datetime.now(),
                    data={"instruction": instruction},
                )
            )

            # Get the Strands agent
            agent = await self._get_strands_agent()

            # Execute with robot hardware context and streaming
            with self._robot_instance.activate():
                # Warm up cameras with progress updates
                for i in range(5):
                    self._robot_instance.get_current_images()
                    await asyncio.sleep(0.2)

                    # Yield camera warmup progress
                    yield {
                        "task_id": task_id,
                        "type": "warmup_progress",
                        "progress": (i + 1) / 5,
                        "message": "Warming up robot cameras...",
                    }

                # Stream from the Strands agent
                # This maintains compatibility with existing voice assistant
                async for event in agent.stream_async(instruction):
                    # Enhance event with task information
                    enriched_event = {"task_id": task_id, **event}

                    # Publish streaming event
                    await self.event_publisher.publish_event(
                        Event(
                            event_type=EventType.STREAMING_DATA,
                            task_id=task_id,
                            timestamp=datetime.now(),
                            data=enriched_event,
                        )
                    )

                    yield enriched_event

                # Reset to initial pose after completion
                self._robot_instance.move_to_initial_pose()

            await self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)

            # Publish completion event
            await self.event_publisher.publish_event(
                Event(
                    event_type=EventType.TASK_COMPLETED,
                    task_id=task_id,
                    timestamp=datetime.now(),
                    data={"instruction": instruction},
                )
            )

            # Final completion message
            yield {
                "task_id": task_id,
                "type": "completion",
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Task completed successfully"}],
                },
            }

        except Exception as e:
            if task_id:
                await self.task_manager.update_task_status(
                    task_id, TaskStatus.FAILED, str(e)
                )

                # Publish failure event
                await self.event_publisher.publish_event(
                    Event(
                        event_type=EventType.TASK_FAILED,
                        task_id=task_id,
                        timestamp=datetime.now(),
                        data={"error": str(e), "instruction": instruction},
                    )
                )

            # Yield error information
            yield {
                "task_id": task_id,
                "type": "error",
                "error": str(e),
                "message": {
                    "role": "assistant",
                    "content": [{"text": f"Task failed: {str(e)}"}],
                },
            }

    async def get_available_tools(self) -> List[ToolDefinition]:
        """Get list of tools available to this agent."""
        return await self.tool_registry.list_tools()

    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status and health information."""
        try:
            is_hardware_connected = await self.hardware_interface.is_connected()
            running_tasks = await self.task_manager.list_tasks(
                status=TaskStatus.RUNNING
            )
            recent_events = await self.event_publisher.get_event_history(limit=5)

            return {
                "status": "healthy",
                "hardware_connected": is_hardware_connected,
                "running_tasks": len(running_tasks),
                "recent_events": len(recent_events),
                "timestamp": datetime.now().isoformat(),
                "model_id": self.model_id,
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


# Factory functions optimized for testability, flexibility, and resource management


def create_robot_agent(
    robot_instance: SO10xRobot,
    gr00t_client_instance: Optional[Gr00tRobotInferenceClient] = None,
    model_id: str = DEFAULT_MODEL_ID,
    region_name: str = "us-west-2",
    callback_handler: Optional[Callable] = None,
) -> SO10xRobotAgent:
    """
    Create an SO10x robot agent with explicit dependency injection.

    Requires explicit robot instance for better resource management and testability.

    Args:
        robot_instance: Required SO100Robot instance for hardware control
        gr00t_client_instance: Optional Gr00t client instance (creates default if not provided)
        model_id: The model ID to use for the agent
        region_name: AWS region name
        callback_handler: Optional callback handler for agent events
    """
    return SO10xRobotAgent(
        robot_instance=robot_instance,
        gr00t_client_instance=gr00t_client_instance,
        model_id=model_id,
        region_name=region_name,
        callback_handler=callback_handler,
    )


def create_robot_agent_with_config(
    enable_camera: bool = True,
    wrist_cam_idx: int = 2,
    front_cam_idx: int = 0,
    port: Optional[str] = None,
    model_id: str = DEFAULT_MODEL_ID,
    region_name: str = "us-west-2",
    callback_handler: Optional[Callable] = None,
) -> SO10xRobotAgent:
    """
    Create a robot agent with customizable hardware configuration.

    This creates fresh instances with specified hardware settings, ideal for
    multiple agents with different camera configurations.

    Args:
        enable_camera: Whether to enable camera support
        wrist_cam_idx: Wrist camera index
        front_cam_idx: Front camera index
        port: Serial port for the arm
        model_id: The model ID to use for the agent
        region_name: AWS region name
        callback_handler: Optional callback handler for agent events
    """
    if port is None:
        raise ValueError("`port` is required for create_robot_agent_with_config")

    robot_instance = SO10xRobot(
        enable_camera=enable_camera,
        wrist_cam_idx=wrist_cam_idx,
        front_cam_idx=front_cam_idx,
        robot_port=port,
    )
    gr00t_instance = Gr00tRobotInferenceClient()

    return SO10xRobotAgent(
        robot_instance=robot_instance,
        gr00t_client_instance=gr00t_instance,
        model_id=model_id,
        region_name=region_name,
        callback_handler=callback_handler,
    )


def create_mock_robot_agent(
    mock_robot: Optional[SO10xRobot] = None,
    mock_gr00t: Optional[Gr00tRobotInferenceClient] = None,
    model_id: str = DEFAULT_MODEL_ID,
    region_name: str = "us-west-2",
    callback_handler: Optional[Callable] = None,
) -> SO10xRobotAgent:
    """
    Create a robot agent with mock instances for testing.

    Provides easy setup for unit tests and integration tests without hardware.

    Args:
        mock_robot: Mock SO100Robot instance (creates default if not provided)
        mock_gr00t: Mock Gr00t client instance (creates default if not provided)
        model_id: The model ID to use for the agent
        region_name: AWS region name
        callback_handler: Optional callback handler for agent events
    """
    from unittest.mock import Mock

    if mock_robot is None:
        mock_robot = Mock(spec=SO10xRobot)
        mock_robot.activate.return_value.__enter__ = Mock(return_value=mock_robot)
        mock_robot.activate.return_value.__exit__ = Mock(return_value=None)
        mock_robot.get_current_images.return_value = None
        mock_robot.move_to_initial_pose.return_value = None

    if mock_gr00t is None:
        mock_gr00t = Mock(spec=Gr00tRobotInferenceClient)

    return SO10xRobotAgent(
        robot_instance=mock_robot,
        gr00t_client_instance=mock_gr00t,
        model_id=model_id,
        region_name=region_name,
        callback_handler=callback_handler,
    )


# Example usage and testing
if __name__ == "__main__":
    # Configure logging for main execution
    setup_robot_logging(log_level="INFO", include_timestamps=False)

    async def main():
        import argparse

        parser = argparse.ArgumentParser(description="Run SO-ARM10x agent")
        parser.add_argument(
            "--port",
            type=str,
            required=True,
            help="Serial port for the arm (e.g., /dev/tty.usbmodemXXXX)",
        )
        parser.add_argument("--id", type=str, default="my_awesome_follower_arm")
        parser.add_argument("--wrist_cam_idx", type=int, default=0)
        parser.add_argument("--front_cam_idx", type=int, default=1)

        parser.add_argument(
            "--policy_host",
            type=str,
            default="localhost",
            help="Host for the GR00T policy server",
        )

        parser.add_argument(
            "--instruction",
            type=str,
            help="Robot instruction (will prompt for input if not provided)",
        )
        args = parser.parse_args()

        user_query = args.instruction or input("Enter your instruction: ")

        # Create agent with provided configuration
        robot_instance = SO10xRobot(
            robot_port=args.port,
            robot_id=args.id,
            wrist_cam_idx=args.wrist_cam_idx,
            front_cam_idx=args.front_cam_idx,
        )

        gr00t_client_instance = Gr00tRobotInferenceClient(host=args.policy_host)
        so10x_agent = create_robot_agent(
            robot_instance=robot_instance, gr00t_client_instance=gr00t_client_instance
        )

        logger.info(f"🎯 Processing query: {user_query}")

        # Use the streaming interface with clean output
        async for event in so10x_agent.astream(user_query):
            event_type = event.get("type", "unknown")

            if event_type == "warmup_progress":
                progress = event.get("progress", 0)
                logger.info(f"📹 Camera warmup: {progress:.1%}")

            elif "message" in event and event["message"].get("role") == "assistant":
                for content in event["message"].get("content", []):
                    if "text" in content and not _is_signature_or_binary(
                        content["text"]
                    ):
                        logger.info(f"🤖 Agent: {content['text']}")

            elif event_type == "completion":
                logger.info("✅ Task completed successfully")
                break

            elif event_type == "error":
                logger.error(f"❌ Error: {event.get('error')}")
                break

        logger.info("🏁 Execution complete!")

    def _is_signature_or_binary(text: str) -> bool:
        """Check if text appears to be a signature or binary data."""
        if not isinstance(text, str):
            return True

        # Check for signature patterns (very long strings with base64-like chars)
        if len(text) > 100 and any(char in text for char in ["=", "+", "/"]):
            non_printable = sum(1 for c in text if ord(c) < 32 or ord(c) > 126)
            if non_printable / len(text) > 0.1:  # More than 10% non-printable
                return True

        # Check for very long strings that might be signatures/hashes
        if len(text) > 500 and " " not in text:
            return True

        return False

    asyncio.run(main())
