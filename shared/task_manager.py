import asyncio
import logging
import uuid
from datetime import datetime
from collections import deque
from typing import Any, Dict, List, Optional

from shared import ITaskManager, TaskInfo, TaskStatus

logger = logging.getLogger(__name__)


class InMemoryTaskManager(ITaskManager):
    """
    In-memory implementation of task management.

    Provides full task lifecycle management with status tracking and progress
    monitoring. Uses asyncio locks for thread safety and maintains task history.
    Suitable for single-instance deployments; for distributed systems, consider
    a database-backed implementation.
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the task manager.

        Args:
            max_history: Maximum number of completed tasks to keep in memory
        """
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = asyncio.Lock()
        self.max_history = max_history
        self._completed_tasks: deque = deque(maxlen=max_history)

    async def create_task(
        self, instruction: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new task with unique ID and initialize with PENDING status."""
        task_id = str(uuid.uuid4())

        async with self._lock:
            task_info = TaskInfo(
                task_id=task_id,
                instruction=instruction,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
                metadata=metadata or {},
            )
            self._tasks[task_id] = task_info

        logger.debug(f"Created task {task_id}: {instruction}")
        return task_id

    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Retrieve task information by ID with thread-safe access."""
        async with self._lock:
            return self._tasks.get(task_id)

    async def update_task_status(
        self, task_id: str, status: TaskStatus, error_message: Optional[str] = None
    ) -> None:
        """Update task status with automatic timestamp management."""
        async with self._lock:
            if task_id not in self._tasks:
                logger.warning(f"Attempted to update non-existent task: {task_id}")
                return

            task = self._tasks[task_id]
            old_status = task.status
            task.status = status
            task.error_message = error_message

            # Update timestamps based on status transitions
            now = datetime.now()
            if status == TaskStatus.RUNNING and old_status == TaskStatus.PENDING:
                task.started_at = now
            elif status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ]:
                task.completed_at = now
                # Move to completed tasks for history
                self._completed_tasks.append(task)

        logger.debug(
            f"Task {task_id} status changed: {old_status.value} -> {status.value}"
        )

    async def update_task_progress(
        self, task_id: str, progress: float, status_message: Optional[str] = None
    ) -> None:
        """Update task progress with validation and optional status message."""
        # Validate progress value
        progress = max(0.0, min(1.0, progress))

        async with self._lock:
            if task_id not in self._tasks:
                logger.warning(
                    f"Attempted to update progress for non-existent task: {task_id}"
                )
                return

            task = self._tasks[task_id]
            task.progress = progress

            if status_message:
                task.metadata["last_status_message"] = status_message
                task.metadata["last_update"] = datetime.now().isoformat()

        logger.debug(f"Task {task_id} progress updated: {progress:.2%}")

    async def list_tasks(
        self, status: Optional[TaskStatus] = None, limit: Optional[int] = None
    ) -> List[TaskInfo]:
        """List tasks with optional filtering and limiting."""
        async with self._lock:
            tasks = list(self._tasks.values())

            # Filter by status if specified
            if status:
                tasks = [task for task in tasks if task.status == status]

            # Sort by created_at descending (newest first)
            tasks.sort(key=lambda t: t.created_at, reverse=True)

            # Apply limit if specified
            if limit:
                tasks = tasks[:limit]

            return tasks

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if it's in a cancellable state."""
        async with self._lock:
            if task_id not in self._tasks:
                return False

            task = self._tasks[task_id]

            # Only allow cancellation of pending or running tasks
            if task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
                await self.update_task_status(task_id, TaskStatus.CANCELLED)
                return True

            return False


# TODO: Implement database-backed task manager
