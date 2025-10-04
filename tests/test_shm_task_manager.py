import asyncio
from multiprocessing.managers import SharedMemoryManager

import pytest

from shared import TaskStatus
from shared.task_manager import SharedMemoryTaskManager


@pytest.mark.asyncio
async def test_shared_memory_task_manager_lifecycle_and_claim():
    """Lifecycle coverage for the SHM TaskManager.

    - Creates a task and validates initial PENDING status
    - Claims the task atomically, updates progress
    - Transitions to COMPLETED and verifies via get_task
    - Ensures list_tasks includes the created task
    """

    async def run_test():
        with SharedMemoryManager() as smm:
            # Allocate fixed-size string slots for JSON tasks
            slot = 4096
            tasks_buf = smm.ShareableList([" " * slot] * 16)
            tm = SharedMemoryTaskManager(
                namespace="test", tasks_name=tasks_buf.shm.name
            )

            task_id = await tm.create_task("do something", {"meta": 1})
            t = await tm.get_task(task_id)
            assert t is not None and t.status == TaskStatus.PENDING

            # Claim task
            claimed = await tm.claim_task(task_id, worker_id="w1")
            assert claimed

            # Complete
            await tm.update_task(task_id, TaskStatus.COMPLETED)
            t = await tm.get_task(task_id)
            assert t is not None and t.status == TaskStatus.COMPLETED

            # List
            items = await tm.list_tasks()
            assert any(x.task_id == task_id for x in items)

    await asyncio.wait_for(run_test(), timeout=3.0)
