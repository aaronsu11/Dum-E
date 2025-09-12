import asyncio
import fcntl
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from multiprocessing.shared_memory import ShareableList

from shared import (
    ITaskManager,
    TaskInfo,
    TaskStatus,
)


def _acquire_file_lock(path: str):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_file_lock(fd: int):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class SharedMemoryTaskManager(ITaskManager):
    """
    Cross-process task manager using SharedMemory ShareableList as a fixed-size
    store of JSON-encoded TaskInfo entries. A simple file lock serializes writers.
    """

    def __init__(self, namespace: str, tasks_name: str) -> None:
        self.namespace = namespace
        self.tasks = ShareableList(name=tasks_name)
        self.capacity = len(self.tasks)
        self.lock_file = f"/tmp/dume_{namespace}_tasks.lock"

    async def create_task(
        self, instruction: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        import uuid

        task_id = str(uuid.uuid4())
        now = datetime.now()
        task_obj = TaskInfo(
            task_id=task_id,
            instruction=instruction,
            status=TaskStatus.PENDING,
            created_at=now,
            metadata=metadata or {},
        )
        entry = self._encode_task(task_obj)

        fd = _acquire_file_lock(self.lock_file)
        try:
            # Find empty slot or overwrite oldest PENDING/COMPLETED slot
            for i in range(self.capacity):
                if not self.tasks[i].strip():
                    self.tasks[i] = entry
                    break
            else:
                # Overwrite index 0 in worst case
                self.tasks[0] = entry
        finally:
            _release_file_lock(fd)
        return task_id

    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        for i in range(self.capacity):
            if not self.tasks[i].strip():
                continue
            try:
                t = self._decode_task(self.tasks[i].rstrip())
                if t.task_id == task_id:
                    return t
            except Exception:
                continue
        return None

    async def update_task_status(
        self, task_id: str, status: TaskStatus, error_message: Optional[str] = None
    ) -> None:
        fd = _acquire_file_lock(self.lock_file)
        try:
            for i in range(self.capacity):
                if not self.tasks[i].strip():
                    continue
                try:
                    t = self._decode_task(self.tasks[i].rstrip())
                except Exception:
                    continue
                if t.task_id == task_id:
                    t.status = status
                    t.error_message = error_message
                    now = datetime.now()
                    if status == TaskStatus.RUNNING and not t.started_at:
                        t.started_at = now
                    if status in [
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                    ]:
                        t.completed_at = now
                    self.tasks[i] = self._encode_task(t)
                    return
        finally:
            _release_file_lock(fd)

    async def update_task_progress(
        self, task_id: str, progress: float, status_message: Optional[str] = None
    ) -> None:
        progress = max(0.0, min(1.0, progress))
        fd = _acquire_file_lock(self.lock_file)
        try:
            for i in range(self.capacity):
                if not self.tasks[i].strip():
                    continue
                try:
                    t = self._decode_task(self.tasks[i].rstrip())
                except Exception:
                    continue
                if t.task_id == task_id:
                    t.progress = progress
                    if status_message:
                        t.metadata["last_status_message"] = status_message
                        t.metadata["last_update"] = datetime.now().isoformat()
                    self.tasks[i] = self._encode_task(t)
                    return
        finally:
            _release_file_lock(fd)

    async def list_tasks(
        self, status: Optional[TaskStatus] = None, limit: Optional[int] = None
    ) -> List[TaskInfo]:
        result: List[TaskInfo] = []
        for i in range(self.capacity):
            if not self.tasks[i].strip():
                continue
            try:
                t = self._decode_task(self.tasks[i].rstrip())
                if status and t.status != status:
                    continue
                result.append(t)
            except Exception:
                continue
        result.sort(key=lambda t: t.created_at, reverse=True)
        if limit:
            result = result[:limit]
        return result

    async def cancel_task(self, task_id: str) -> bool:
        await self.update_task_status(task_id, TaskStatus.CANCELLED)
        return True

    async def claim_task(self, task_id: str, worker_id: str) -> bool:
        fd = _acquire_file_lock(self.lock_file)
        try:
            for i in range(self.capacity):
                if not self.tasks[i]:
                    continue
                try:
                    t = self._decode_task(self.tasks[i])
                except Exception:
                    continue
                if t.task_id == task_id and t.status == TaskStatus.PENDING:
                    t.status = TaskStatus.RUNNING
                    t.started_at = datetime.now()
                    t.metadata["worker_id"] = worker_id
                    t.metadata["claimed_at"] = datetime.now().isoformat()
                    self.tasks[i] = self._encode_task(t)
                    return True
        finally:
            _release_file_lock(fd)
        return False

    def _encode_task(self, t: TaskInfo) -> str:
        return json.dumps(
            {
                "task_id": t.task_id,
                "instruction": t.instruction,
                "status": t.status.value,
                "created_at": t.created_at.isoformat() if t.created_at else None,
                "started_at": t.started_at.isoformat() if t.started_at else None,
                "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                "error_message": t.error_message,
                "progress": t.progress,
                "metadata": t.metadata or {},
            }
        )

    def _decode_task(self, s: str) -> TaskInfo:
        obj = json.loads(s)
        return TaskInfo(
            task_id=obj["task_id"],
            instruction=obj["instruction"],
            status=TaskStatus(obj["status"]),
            created_at=(
                datetime.fromisoformat(obj["created_at"]) if obj["created_at"] else None
            ),
            started_at=(
                datetime.fromisoformat(obj["started_at"])
                if obj.get("started_at")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(obj["completed_at"])
                if obj.get("completed_at")
                else None
            ),
            error_message=obj.get("error_message"),
            progress=obj.get("progress", 0.0),
            metadata=obj.get("metadata", {}),
        )


def get_shared_memory_task_manager_from_env() -> Optional[SharedMemoryTaskManager]:
    tasks = os.getenv("DUME_TASKS_BUF")
    namespace = os.getenv("DUME_NAMESPACE", "default")
    if tasks:
        return SharedMemoryTaskManager(namespace=namespace, tasks_name=tasks)
    return None
