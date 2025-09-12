"""
Refactored robot agent implementing the IRobotAgent interface.

This module provides an SO10x robot agent that implements the abstract
interfaces while maintaining compatibility with the existing voice assistant
streaming pattern. Key improvements include:
- Implements IRobotAgent interface for modularity
- Message publishing for real-time progress updates
- Enhanced error handling and recovery
- Task lifecycle management
- Tool registry integration

Example usage (from the root directory):
python -m embodiment.so_arm10x.agent \
    --port /dev/ttyACM0 \
    --id so101_follower_arm \
    --wrist_cam_idx 0 \
    --front_cam_idx 1 \
    --policy_host localhost \
    --profile aws \
    --instruction "I want one banana and one apple on the plate"
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
from strands.models import BedrockModel
from strands.models.anthropic import AnthropicModel
from tqdm import tqdm


from embodiment.so_arm10x.controller import (
    Gr00tRobotInferenceClient,
    SO10xArmController,
)
from shared import (
    ITaskManager,
    IMessageBroker,
    IRobotAgent,
    TaskStatus,
    Message,
    MessageType,
    ToolDefinition,
)
from shared.message_broker.shm import get_shared_memory_broker_from_env
from shared.task_manager.shm import get_shared_memory_task_manager_from_env

from logging_config import create_clean_callback_handler, setup_robot_logging

load_dotenv()


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
    robot_controller: SO10xArmController,
    gr00t_client_instance: Gr00tRobotInferenceClient,
):
    """Create robot tools that use the specific robot instance."""

    @tool
    def reset_pose():
        """Reset the robot to the initial pose to make the workspace clear and visible."""
        robot_controller.move_to_initial_pose()
        return {
            "status": "success",
            "content": [
                {"text": f"Reset the robot to the initial pose."},
            ],
        }

    @tool
    def assess_situation() -> dict:
        """Assess the situation of the current state"""
        images = robot_controller.get_current_images()
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
            robot_controller.move_to_initial_pose()
            robot_controller.move_to_ready_pose()

        for _ in tqdm(range(actions_to_execute), desc="Executing actions"):
            # New observation -> policy -> action flow using updated interfaces
            observation_dict = robot_controller.get_observation()
            action_list = gr00t_client_instance.get_action(
                observation_dict,
                language_instruction or gr00t_client_instance.language_instruction,
            )

            # Execute a short horizon for stability
            for action_dict in action_list:
                robot_controller.set_target_state(action_dict)
                time.sleep(0.05)

        time.sleep(0.5)
        return robot_controller.get_current_images()

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
        robot_controller.release_at_remote_pose(location)
        latest_images = robot_controller.get_current_images()
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
        robot_controller: SO10xArmController,
        gr00t_client_instance: Gr00tRobotInferenceClient,
        task_manager: Optional[ITaskManager] = None,
        profile: Literal["default", "aws"] = "default",
        message_broker: Optional[IMessageBroker] = None,
        callback_handler: Callable = None,
    ):
        """
        Initialize the SO10x robot agent with explicit dependency injection.

        Args:
            robot_controller: Required SO100Robot instance for hardware control
            gr00t_client_instance: Required Gr00t client instance
            profile: The model provider profile
            task_manager: task manager instance
            message_broker: optional message broker instance
            callback_handler: optional callback handler for agent events
        """
        # Required controller and policy client
        self.robot_controller = robot_controller
        self.gr00t_client_instance = gr00t_client_instance

        # Initialize MCP server interfaces with defaults if not provided (suppress verbose logging)
        self.task_manager = task_manager
        self.message_broker = message_broker

        # Agent events handler for custom logging
        self.callback_handler = callback_handler or create_clean_callback_handler()

        # Create robot-specific tools
        self._robot_tools = create_robot_tools(
            self.robot_controller, self.gr00t_client_instance
        )

        # Initialize the underlying Strands agent
        self.profile = profile
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
        # Create the model based on the profile
        if self.profile == "aws":
            model = BedrockModel(
                # Use the cross-region inference profile prefix "us." with Sonnet 4
                model_id="us.anthropic.claude-sonnet-4-20250514-v1:0",
                region_name=os.getenv("AWS_REGION"),
                max_tokens=8000,
                additional_request_fields={
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": 6000,  # Minimum of 1,024
                    },
                    "anthropic_beta": ["interleaved-thinking-2025-05-14"],
                },
            )
        else:
            model = AnthropicModel(
                client_args={
                    "api_key": os.getenv("ANTHROPIC_API_KEY"),
                },
                model_id="claude-sonnet-4-20250514",
                max_tokens=8000,
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
            system_prompt="""\
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
    * Assess the current situation carefully against the desired state
    * Briefly describe what was accomplished
    * Determine next steps if needed
    * Continue or repeat until desired state is achieved
- Be very concise in responses (max 15 words) and don't use special characters  

Note: Colors in images may appear different due to reflections.""",
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
            # Create task if not provided and a task manager is available
            if self.task_manager is not None:
                if task_id is None:
                    task_id = await self.task_manager.create_task(instruction)
                await self.task_manager.update_task_status(task_id, TaskStatus.RUNNING)

            # Get the Strands agent and execute
            agent = await self._get_strands_agent()

            # Execute with robot hardware context
            with self.robot_controller.activate():
                # Warm up cameras for 1 second
                for _ in range(5):
                    self.robot_controller.get_current_images()
                    await asyncio.sleep(0.2)

                result = agent(instruction)

                # Reset to initial pose after completion
                self.robot_controller.move_to_initial_pose()

            if self.task_manager is not None and task_id is not None:
                await self.task_manager.update_task_status(
                    task_id, TaskStatus.COMPLETED
                )

            return {
                "task_id": task_id,
                "status": "completed",
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            if self.task_manager is not None and task_id is not None:
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
        enhanced progress tracking and message publishing.
        """
        try:
            # Create task if not provided and manager exists
            if self.task_manager is not None:
                if task_id is None:
                    task_id = await self.task_manager.create_task(instruction)
                await self.task_manager.update_task_status(task_id, TaskStatus.RUNNING)

            # Publish task started message
            if self.message_broker:
                await self.message_broker.publish(
                    Message(
                        message_type=MessageType.TASK_STARTED,
                        task_id=task_id,
                        timestamp=datetime.now(),
                        data={"instruction": instruction},
                    )
                )

            # Get the Strands agent
            agent = await self._get_strands_agent()

            # Execute with robot hardware context and streaming
            with self.robot_controller.activate():
                # Warm up cameras with progress updates
                for i in range(5):
                    self.robot_controller.get_current_images()
                    await asyncio.sleep(0.2)

                    # Yield camera warmup progress
                    yield {
                        "task_id": task_id,
                        "type": "warmup_progress",
                        "progress": (i + 1) / 5,
                        "message": "Warming up robot cameras...",
                    }

                # Stream from the Strands agent
                async for event in agent.stream_async(instruction):
                    # Enhance event with task information
                    enriched_event = {"task_id": task_id, **event}

                    # Publish streaming message
                    if self.message_broker:
                        await self.message_broker.publish(
                            Message(
                                message_type=MessageType.STREAMING_DATA,
                                task_id=task_id,
                                timestamp=datetime.now(),
                                data=enriched_event,
                            )
                        )

                    yield enriched_event

                # Reset to initial pose after completion
                try:
                    self.robot_controller.move_to_initial_pose()
                    time.sleep(1.0)
                except Exception as pose_error:
                    # Log the error but don't fail the task - it already completed successfully
                    logger.warning(
                        f"⚠️  Failed to reset to initial pose in time after task completion: {pose_error}"
                    )

            if self.task_manager is not None and task_id is not None:
                await self.task_manager.update_task_status(
                    task_id, TaskStatus.COMPLETED
                )

            # Publish completion message
            if self.message_broker:
                await self.message_broker.publish(
                    Message(
                        message_type=MessageType.TASK_COMPLETED,
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
            if self.task_manager is not None and task_id is not None:
                await self.task_manager.update_task_status(
                    task_id, TaskStatus.FAILED, str(e)
                )

                if self.message_broker:
                    # Publish failure message
                    await self.message_broker.publish(
                        Message(
                            message_type=MessageType.TASK_FAILED,
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
        # TODO: convert this to ToolDefinition list
        return await self._robot_tools

    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status and health information."""
        try:
            is_controller_connected = await self.robot_controller.is_connected()
            running_tasks = await self.task_manager.list_tasks(
                status=TaskStatus.RUNNING
            )
            recent_messages = await self.message_broker.get_message_history(limit=5)

            return {
                "status": "healthy",
                "controller_connected": is_controller_connected,
                "running_tasks": len(running_tasks),
                "recent_messages": len(recent_messages),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }


# Factory function optimized for testability, flexibility, and resource management
def create_robot_agent(
    robot_port: Optional[str] = None,
    robot_id: str = "my_awesome_follower_arm",
    wrist_cam_idx: int = 0,
    front_cam_idx: int = 2,
    policy_host: str = "localhost",
    profile: Literal["default", "aws"] = "default",
    callback_handler: Optional[Callable] = None,
    task_manager: Optional[ITaskManager] = None,
    message_broker: Optional[IMessageBroker] = None,
) -> SO10xRobotAgent:
    """
    Create a robot agent with customizable hardware configuration.

    This creates fresh instances with specified hardware settings, ideal for
    multiple agents with different camera configurations.

    Args:
        robot_port: Serial port for the arm
        robot_id: Robot ID
        wrist_cam_idx: Wrist camera index
        front_cam_idx: Front camera index
        policy_host: Host for the GR00T policy server
        profile: The model provider profile
        callback_handler: Optional callback handler for agent events
    """
    if robot_port is None:
        raise ValueError("`robot_port` is required for create_robot_agent")

    robot_controller = SO10xArmController(
        robot_port=robot_port,
        robot_id=robot_id,
        wrist_cam_idx=wrist_cam_idx,
        front_cam_idx=front_cam_idx,
    )
    gr00t_instance = Gr00tRobotInferenceClient(host=policy_host)

    return SO10xRobotAgent(
        robot_controller=robot_controller,
        gr00t_client_instance=gr00t_instance,
        profile=profile,
        callback_handler=callback_handler,
        task_manager=task_manager,
        message_broker=message_broker,
    )


async def _agent_worker_loop(
    agent: SO10xRobotAgent,
    task_manager: ITaskManager,
    message_broker: IMessageBroker,
    robot_id: str,
    worker_id: str,
):
    """
    Subscribe to TASK_STARTED events from the broker, claim tasks, and execute
    them with the robot agent. Designed to work with local in-memory or OTA backends.
    """
    async for msg in message_broker.subscribe(message_types=[MessageType.TASK_STARTED]):
        try:
            # Only handle tasks dispatched by the MCP server (avoid self-loop)
            if not isinstance(msg.data, dict) or msg.data.get("source") != "mcp_server":
                continue

            # Optional routing by robot_id
            target_robot = msg.data.get("robot_id")
            if target_robot and target_robot != robot_id:
                continue

            task_id = msg.task_id
            instruction = (
                msg.data.get("instruction") if isinstance(msg.data, dict) else None
            )
            if not task_id or not instruction:
                continue

            # Atomically claim the task
            claimed = await task_manager.claim_task(task_id, worker_id)
            if not claimed:
                continue

            # Execute with streaming. The agent will publish streaming updates
            # and final status via the provided message_broker.
            async for _ in agent.astream(instruction, task_id=task_id):
                # We rely on the agent to publish STREAMING_DATA
                # Optional logging can be added here if needed
                pass
        except Exception as exec_error:
            # Best-effort failure reporting
            try:
                await task_manager.update_task_status(
                    task_id, TaskStatus.FAILED, str(exec_error)
                )
            except Exception:
                pass
            try:
                if message_broker:
                    await message_broker.publish(
                        Message(
                            message_type=MessageType.TASK_FAILED,
                            task_id=task_id,
                            timestamp=datetime.now(),
                            data={"error": str(exec_error), "instruction": instruction},
                        )
                    )
            except Exception:
                pass


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
            "--profile",
            type=str,
            default="default",
            help="Model provider profile",
        )
        parser.add_argument(
            "--instruction",
            type=str,
            help="Robot instruction (will prompt for input if not provided)",
        )
        parser.add_argument(
            "--worker",
            action="store_true",
            help="Run in worker mode: consume TASK_STARTED from broker and execute",
        )

        args = parser.parse_args()

        # Configure backends: prefer shared memory when available (DUME_IPC=shm)
        task_manager = get_shared_memory_task_manager_from_env()
        message_broker = get_shared_memory_broker_from_env()

        # Create agent with injected backends
        agent = create_robot_agent(
            robot_port=args.port,
            robot_id=args.id,
            wrist_cam_idx=args.wrist_cam_idx,
            front_cam_idx=args.front_cam_idx,
            policy_host=args.policy_host,
            profile=args.profile,
            task_manager=task_manager,
            message_broker=message_broker,
        )

        if args.worker:
            import uuid

            worker_id = f"agent-worker-{uuid.uuid4()}"
            logger.info(
                f"🧩 Worker mode: namespace={args.namespace}, worker_id={worker_id}"
            )
            await _agent_worker_loop(
                agent=agent,
                task_manager=task_manager,
                message_broker=message_broker,
                robot_id=args.id,
                worker_id=worker_id,
            )
            return

        # Direct single-run mode: stream once via agent
        user_query = args.instruction or input("Enter your instruction: ")

        logger.info(f"🎯 Processing query: {user_query}")

        # Use the streaming interface with clean output
        async for event in agent.astream(user_query):
            event_type = event.get("type", "unknown")

            if event_type == "warmup_progress":
                progress = event.get("progress", 0)
                logger.info(f"📹 Camera warmup: {progress:.1%}")

            elif "message" in event and event["message"].get("role") == "assistant":
                for content in event["message"].get("content", []):
                    if "text" in content:
                        logger.info(f"🤖 Agent: {content['text']}")

            elif event_type == "completion":
                logger.info("✅ Task completed successfully")
                break

            elif event_type == "error":
                logger.error(f"❌ Error: {event.get('error')}")
                break

        logger.info("🏁 Execution complete!")

    asyncio.run(main())
