import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from shared import (
    IRobotAgent,
    ITaskManager,
    IMessageBroker,
    Message,
    MessageType,
    TaskStatus,
    ToolDefinition,
)
from shared.message_broker.shm import get_shared_memory_broker_from_env
from shared.task_manager.shm import get_shared_memory_task_manager_from_env


class MockRobotAgent(IRobotAgent):
    """
    Mock implementation of IRobotAgent for tests that simulates a short task
    with progress updates using the shared IMessageBroker/ITaskManager.
    """

    def __init__(
        self,
        task_manager: Optional[ITaskManager] = None,
        message_broker: Optional[IMessageBroker] = None,
        profile: str = "default",
    ) -> None:
        self.task_manager = task_manager
        self.message_broker = message_broker
        self.profile = profile
        self.logger = logging.getLogger("mock_agent")
        logging.basicConfig(level=logging.INFO)

    async def arun(
        self, instruction: str, task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.task_manager is not None:
            if task_id is None:
                task_id = await self.task_manager.create_task(instruction)
            await self.task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        self.logger.info(
            "arun started: task_id=%s instruction=%s", task_id, instruction
        )

        # Emit a couple of progress updates
        if self.message_broker is not None and task_id is not None:
            for i in range(3):
                await self.message_broker.publish(
                    Message(
                        message_type=MessageType.STREAMING_DATA,
                        task_id=task_id,
                        timestamp=datetime.now(),
                        data={"type": "progress", "progress": (i + 1) / 3},
                    )
                )
                await asyncio.sleep(0.01)

            await self.message_broker.publish(
                Message(
                    message_type=MessageType.TASK_COMPLETED,
                    task_id=task_id,
                    timestamp=datetime.now(),
                    data={"result": "ok"},
                )
            )

        if self.task_manager is not None and task_id is not None:
            await self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)

        return {"status": "completed", "task_id": task_id}

    async def astream(
        self, instruction: str, task_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        if self.task_manager is not None:
            if task_id is None:
                task_id = await self.task_manager.create_task(instruction)
            await self.task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        self.logger.info(
            "astream started: task_id=%s instruction=%s", task_id, instruction
        )

        # Simulate warmup
        for i in range(2):
            event = {"type": "warmup_progress", "progress": (i + 1) / 2}
            if self.message_broker is not None and task_id is not None:
                await self.message_broker.publish(
                    Message(
                        message_type=MessageType.STREAMING_DATA,
                        task_id=task_id,
                        timestamp=datetime.now(),
                        data=event,
                    )
                )
            yield event
            await asyncio.sleep(0.01)

        # Simulate a short progression of actions
        for i in range(3):
            event = {
                "message": {"role": "assistant", "content": [{"text": f"step {i+1}"}]}
            }
            if self.message_broker is not None and task_id is not None:
                await self.message_broker.publish(
                    Message(
                        message_type=MessageType.STREAMING_DATA,
                        task_id=task_id,
                        timestamp=datetime.now(),
                        data=event,
                    )
                )
            yield event
            await asyncio.sleep(0.01)

        # Completion
        if self.message_broker is not None and task_id is not None:
            await self.message_broker.publish(
                Message(
                    message_type=MessageType.TASK_COMPLETED,
                    task_id=task_id,
                    timestamp=datetime.now(),
                    data={"result": "ok"},
                )
            )

        if self.task_manager is not None and task_id is not None:
            await self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)

        yield {
            "task_id": task_id,
            "type": "completion",
            "message": {"role": "assistant", "content": [{"text": "done"}]},
        }

    async def get_available_tools(self) -> List[ToolDefinition]:
        return []

    async def get_status(self) -> Dict[str, Any]:
        return {"status": "mock", "profile": self.profile}


async def mock_agent_worker_loop(
    agent: IRobotAgent,
    task_manager: ITaskManager,
    message_broker: IMessageBroker,
    robot_id: str,
    worker_id: str,
):
    """
    Subscribe to TASK_STARTED, claim tasks, and execute via mock agent.
    Mirrors the real worker loop behavior for integration tests.
    """
    async for msg in message_broker.subscribe(message_types=[MessageType.TASK_STARTED]):
        try:
            if not isinstance(msg.data, dict) or msg.data.get("source") != "mcp_server":
                continue
            target_robot = msg.data.get("robot_id")
            if target_robot and target_robot != robot_id:
                continue
            task_id = msg.task_id
            instruction = (
                msg.data.get("instruction") if isinstance(msg.data, dict) else None
            )
            if not task_id or not instruction:
                continue
            claimed = await task_manager.claim_task(task_id, worker_id)
            if not claimed:
                continue
            async for _ in agent.astream(instruction, task_id=task_id):
                pass
        except Exception as e:
            try:
                await task_manager.update_task_status(
                    task_id, TaskStatus.FAILED, str(e)
                )
            except Exception:
                pass
            try:
                await message_broker.publish(
                    Message(
                        message_type=MessageType.TASK_FAILED,
                        task_id=task_id,
                        timestamp=datetime.now(),
                        data={"error": str(e), "instruction": instruction},
                    )
                )
            except Exception:
                pass


if __name__ == "__main__":
    import argparse
    import uuid
    import asyncio

    parser = argparse.ArgumentParser(description="Mock Robot Agent Worker")
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Run in worker mode: consume TASK_STARTED and execute",
    )
    parser.add_argument(
        "--robot_id",
        type=str,
        default="mock_robot",
        help="Robot ID for routing",
    )
    args = parser.parse_args()

    # Attach SHM backends from env
    task_manager = get_shared_memory_task_manager_from_env()
    message_broker = get_shared_memory_broker_from_env()
    if not task_manager or not message_broker:
        raise RuntimeError("Shared memory backends not configured in environment")

    agent = MockRobotAgent(task_manager=task_manager, message_broker=message_broker)

    async def _main():
        if args.worker:
            worker_id = f"mock-worker-{uuid.uuid4()}"
            await mock_agent_worker_loop(
                agent=agent,
                task_manager=task_manager,
                message_broker=message_broker,
                robot_id=args.robot_id,
                worker_id=worker_id,
            )
        else:
            # One-off run for debugging
            await agent.arun("do something simple")

    asyncio.run(_main())
