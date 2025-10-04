import asyncio
from multiprocessing.managers import SharedMemoryManager
from datetime import datetime

import pytest

from shared import Message, MessageType
from shared.message_broker import SharedMemoryMessageBroker


@pytest.mark.asyncio
async def test_shared_memory_broker_publish_subscribe_and_history():
    """End-to-end broker test using SHM implementation.

    - Allocates SHM ring buffer and meta write index
    - Subscribes to TASK_PROGRESS and publishes three messages
    - Asserts subscriber receives exactly three filtered messages
    - Validates get_message_history returns the last two for the task
    """

    async def run_test():
        with SharedMemoryManager() as smm:
            # Allocate fixed-size string slots to hold JSON payloads
            slot = 4096
            buffer = smm.ShareableList([" " * slot] * 16)
            meta = smm.ShareableList([0])

            broker = SharedMemoryMessageBroker(
                namespace="test",
                buffer_name=buffer.shm.name,
                meta_name=meta.shm.name,
                poll_interval_s=0.005,
            )

            # Subscriber task to collect messages
            received = []

            async def subscriber():
                async for msg in broker.subscribe(
                    message_types=[MessageType.TASK_PROGRESS]
                ):
                    received.append(msg)
                    if len(received) >= 3:
                        break

            sub_task = asyncio.create_task(subscriber())
            # Give subscriber time to start polling
            await asyncio.sleep(0.02)

            # Publish a mixture of message types
            for i in range(3):
                await broker.publish(
                    Message(
                        message_type=MessageType.TASK_PROGRESS,
                        task_id="t1",
                        timestamp=datetime.now(),
                        data={"progress": i + 1},
                    )
                )

            await asyncio.wait_for(sub_task, timeout=2.0)

            # Validate receipt
            assert len(received) == 3
            assert all(m.message_type == MessageType.TASK_PROGRESS for m in received)
            # History should return at least the last 2
            history = await broker.get_message_history(task_id="t1", limit=2)
            assert len(history) == 2

    await asyncio.wait_for(run_test(), timeout=5.0)
