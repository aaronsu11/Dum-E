import asyncio
import os
from typing import List
from multiprocessing.managers import SharedMemoryManager

import pytest

from shared import Message, MessageType
from shared.message_broker.shm import SharedMemoryMessageBroker
from shared.task_manager.shm import SharedMemoryTaskManager
from tests.mocks import MockRobotAgent


@pytest.mark.asyncio
async def test_agent_with_shared_memory_interfaces():
    """Integration test for MockRobotAgent with SHM interfaces.

    - Creates SHM broker and task manager
    - Runs MockRobotAgent.astream to emit streaming events
    - Subscribes server-side to collect TASK_PROGRESS and TASK_COMPLETED
    - Asserts we received assistant messages and final completion
    - Confirms task status transitioned to COMPLETED with timestamps
    """

    async def run_test():
        with SharedMemoryManager() as smm:
            slot = 4096
            broker_buf = smm.ShareableList([" " * slot] * 32)
            broker_meta = smm.ShareableList([0])
            tasks_buf = smm.ShareableList([" " * slot] * 16)

            os.environ["DUME_NAMESPACE"] = "test"
            os.environ["DUME_BROKER_BUF"] = broker_buf.shm.name
            os.environ["DUME_BROKER_META"] = broker_meta.shm.name
            os.environ["DUME_TASKS_BUF"] = tasks_buf.shm.name
            os.environ["DUME_IPC"] = "shm"

            broker = SharedMemoryMessageBroker(
                namespace="test",
                buffer_name=broker_buf.shm.name,
                meta_name=broker_meta.shm.name,
                poll_interval_s=0.005,
            )
            tm = SharedMemoryTaskManager(
                namespace="test", tasks_name=tasks_buf.shm.name
            )

            # Mock agent consumes a task id and emits progress + completion
            agent = MockRobotAgent(task_manager=tm, message_broker=broker)

            task_id = await tm.create_task("dummy instruction", {"source": "unit"})

            received: List[Message] = []

            async def server_listener():
                async for m in broker.subscribe(
                    message_types=[
                        MessageType.TASK_PROGRESS,
                        MessageType.TASK_COMPLETED,
                    ],
                    task_id=task_id,
                ):
                    received.append(m)
                    if m.message_type == MessageType.TASK_COMPLETED:
                        break

            listen_task = asyncio.create_task(server_listener())

            # Have the mock agent run and stream
            async for _ in agent.astream("dummy instruction", task_id=task_id):
                pass

            await asyncio.wait_for(listen_task, timeout=1.0)

            # Validate that we saw multiple streaming data and a completion
            assert any(m.message_type == MessageType.TASK_PROGRESS for m in received)
            assert any(m.message_type == MessageType.TASK_COMPLETED for m in received)

            # Validate assistant step messages are present
            assistant_msgs = [
                m
                for m in received
                if isinstance(m.data, dict)
                and isinstance(m.data.get("message"), dict)
                and m.data["message"].get("role") == "assistant"
            ]
            assert len(assistant_msgs) >= 1

            # Task status should be completed with timestamps
            t = await tm.get_task(task_id)
            assert t is not None
            assert t.status.name.lower() == "completed"
            assert t.started_at is not None and t.completed_at is not None

    await asyncio.wait_for(run_test(), timeout=3.0)
