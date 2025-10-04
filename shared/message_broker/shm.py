import asyncio
import fcntl
import json
import os
from datetime import datetime
from typing import AsyncIterator, List, Optional

from multiprocessing.shared_memory import ShareableList

from shared import (
    IMessageBroker,
    Message,
    MessageType,
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


class SharedMemoryMessageBroker(IMessageBroker):
    """
    Simple cross-process broker using SharedMemory-backed ring buffer.

    Implementation notes:
    - payloads are JSON strings stored in a fixed-size ShareableList 'buffer'
    - a second ShareableList 'meta' holds a single integer 'write_index'
    - writers append to buffer[(write_index+1) % capacity] and then update write_index
    - subscribers poll write_index and read any new entries sequentially
    - minimal file lock is used to serialize writers (avoid interleaving updates)
    """

    def __init__(
        self,
        namespace: str,
        buffer_name: str,
        meta_name: str,
        poll_interval_s: float = 0.05,
    ) -> None:
        self.namespace = namespace
        self.buffer = ShareableList(name=buffer_name)
        self.meta = ShareableList(name=meta_name)
        self.capacity = len(self.buffer)
        self.poll_interval_s = poll_interval_s
        self.lock_file = f"/tmp/dume_{namespace}_broker.lock"
        # This process only attaches to existing SHM; avoid double-unlink at exit
        try:
            from multiprocessing import resource_tracker as _rt  # type: ignore

            try:
                n = getattr(self.buffer.shm, "_name", self.buffer.shm.name)
                _rt.unregister(n, "shared_memory")
            except Exception:
                pass
            try:
                n2 = getattr(self.meta.shm, "_name", self.meta.shm.name)
                _rt.unregister(n2, "shared_memory")
            except Exception:
                pass
        except Exception:
            pass

    async def publish(self, message: Message) -> None:
        payload = json.dumps(
            {
                "message_type": message.message_type.value,
                "task_id": message.task_id,
                "timestamp": message.timestamp.isoformat(),
                "data": message.data,
            }
        )

        # Serialize writers with file lock
        fd = _acquire_file_lock(self.lock_file)
        try:
            write_index = int(self.meta[0]) if len(self.meta) > 0 else 0
            next_index = (write_index + 1) % self.capacity
            self.buffer[next_index] = payload
            self.meta[0] = next_index
        finally:
            _release_file_lock(fd)

    async def subscribe(
        self,
        message_types: Optional[List[MessageType]] = None,
        task_id: Optional[str] = None,
    ) -> AsyncIterator[Message]:
        # Start reading from current write_index
        last_index = int(self.meta[0]) if len(self.meta) > 0 else -1
        while True:
            await asyncio.sleep(self.poll_interval_s)
            write_index = int(self.meta[0]) if len(self.meta) > 0 else last_index
            # No new messages
            if write_index == last_index:
                continue

            # Determine how many new entries to read
            idx = (
                (last_index + 1) % self.capacity
                if last_index >= 0
                else ((write_index + 1) % self.capacity)
            )
            while True:
                payload = self.buffer[idx]
                if payload:
                    try:
                        obj = json.loads(payload)
                        mtype = (
                            MessageType(obj["message_type"])
                            if "message_type" in obj
                            else None
                        )
                        if message_types and mtype not in message_types:
                            pass
                        elif task_id and obj.get("task_id") != task_id:
                            pass
                        else:
                            yield Message(
                                message_type=mtype,
                                task_id=obj.get("task_id"),
                                timestamp=(
                                    datetime.fromisoformat(obj["timestamp"])
                                    if obj.get("timestamp")
                                    else datetime.now()
                                ),
                                data=obj.get("data", {}),
                            )
                    except Exception:
                        # Ignore malformed entries
                        pass

                if idx == write_index:
                    break
                idx = (idx + 1) % self.capacity
            last_index = write_index

    async def get_message_history(
        self, task_id: Optional[str] = None, limit: Optional[int] = 100
    ) -> List[Message]:
        result: List[Message] = []
        write_index = int(self.meta[0]) if len(self.meta) > 0 else -1
        if write_index < 0:
            return result
        # Read backwards up to limit entries
        count = 0
        idx = write_index
        while count < (limit or 100) and count < self.capacity:
            payload = self.buffer[idx].rstrip()
            payload = self.buffer[idx].rstrip()
            if payload:
                try:
                    obj = json.loads(payload)
                    if task_id and obj.get("task_id") != task_id:
                        pass
                    else:
                        result.append(
                            Message(
                                message_type=MessageType(obj["message_type"]),
                                task_id=obj.get("task_id"),
                                timestamp=(
                                    datetime.fromisoformat(obj["timestamp"])
                                    if obj.get("timestamp")
                                    else datetime.now()
                                ),
                                data=obj.get("data", {}),
                            )
                        )
                except Exception:
                    pass
            count += 1
            idx = (idx - 1 + self.capacity) % self.capacity
        return result


def get_shared_memory_broker_from_env() -> Optional[SharedMemoryMessageBroker]:
    buf = os.getenv("DUME_BROKER_BUF")
    meta = os.getenv("DUME_BROKER_META")
    namespace = os.getenv("DUME_NAMESPACE", "default")
    if buf and meta:
        return SharedMemoryMessageBroker(
            namespace=namespace, buffer_name=buf, meta_name=meta
        )
    return None
