import fcntl
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from multiprocessing.shared_memory import ShareableList

from shared import IFleetManager, RobotInfo


def _acquire_file_lock(path: str):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _release_file_lock(fd: int):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class SharedMemoryFleetManager(IFleetManager):
    """
    Simple shared-memory fleet registry using a fixed-size ShareableList of JSON rows.
    Each slot stores one RobotInfo encoded as JSON, empty string means unused.
    """

    def __init__(self, namespace: str, registry_name: str) -> None:
        self.namespace = namespace
        self.registry = ShareableList(name=registry_name)
        self.capacity = len(self.registry)
        self.lock_file = f"/tmp/dume_{namespace}_fleet.lock"
        # This process only attaches to existing SHM; avoid double-unlink at exit
        try:
            from multiprocessing import resource_tracker as _rt  # type: ignore

            try:
                n = getattr(self.registry.shm, "_name", self.registry.shm.name)
                _rt.unregister(n, "shared_memory")
            except Exception:
                pass
        except Exception:
            pass

    async def register_robot(
        self,
        robot_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RobotInfo:
        now = datetime.now()
        fd = _acquire_file_lock(self.lock_file)
        try:
            # Try update existing
            for i in range(self.capacity):
                row = self.registry[i]
                if not row.strip():
                    continue
                try:
                    obj = json.loads(row.rstrip())
                    if obj.get("robot_id") == robot_id:
                        obj["name"] = name if name is not None else obj.get("name")
                        obj["metadata"] = (
                            metadata
                            if metadata is not None
                            else obj.get("metadata", {})
                        )
                        obj["last_seen"] = now.isoformat()
                        self.registry[i] = json.dumps(obj)
                        return self._decode(obj)
                except Exception:
                    continue

            # Insert into empty slot
            info = RobotInfo(
                robot_id=robot_id,
                name=name,
                enabled=True,
                registered_at=now,
                last_seen=now,
                metadata=metadata or {},
            )
            entry = self._encode(info)
            for i in range(self.capacity):
                if not self.registry[i].strip():
                    self.registry[i] = entry
                    return info
            # Overwrite index 0 if full
            self.registry[0] = entry
            return info
        finally:
            _release_file_lock(fd)

    async def list_robots(self, only_enabled: Optional[bool] = None) -> List[RobotInfo]:
        result: List[RobotInfo] = []
        for i in range(self.capacity):
            row = self.registry[i]
            if not row.strip():
                continue
            try:
                obj = json.loads(row.rstrip())
                info = self._decode(obj)
                if only_enabled is not None and info.enabled != only_enabled:
                    continue
                result.append(info)
            except Exception:
                continue
        # Sort by registered_at desc
        result.sort(key=lambda r: r.registered_at, reverse=True)
        return result

    async def get_robot(self, robot_id: str) -> Optional[RobotInfo]:
        for i in range(self.capacity):
            row = self.registry[i]
            if not row.strip():
                continue
            try:
                obj = json.loads(row.rstrip())
                if obj.get("robot_id") == robot_id:
                    return self._decode(obj)
            except Exception:
                continue
        return None

    async def set_enabled(self, robot_id: str, enabled: bool) -> bool:
        fd = _acquire_file_lock(self.lock_file)
        try:
            for i in range(self.capacity):
                row = self.registry[i]
                if not row.strip():
                    continue
                try:
                    obj = json.loads(row.rstrip())
                except Exception:
                    continue
                if obj.get("robot_id") == robot_id:
                    obj["enabled"] = bool(enabled)
                    obj["last_seen"] = datetime.now().isoformat()
                    self.registry[i] = json.dumps(obj)
                    return True
        finally:
            _release_file_lock(fd)
        return False

    async def update_robot(
        self,
        robot_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        fd = _acquire_file_lock(self.lock_file)
        try:
            for i in range(self.capacity):
                row = self.registry[i]
                if not row.strip():
                    continue
                try:
                    obj = json.loads(row.rstrip())
                except Exception:
                    continue
                if obj.get("robot_id") == robot_id:
                    if name is not None:
                        obj["name"] = name
                    if metadata is not None:
                        obj["metadata"] = metadata
                    obj["last_seen"] = datetime.now().isoformat()
                    self.registry[i] = json.dumps(obj)
                    return True
        finally:
            _release_file_lock(fd)
        return False

    def _encode(self, r: RobotInfo) -> str:
        return json.dumps(
            {
                "robot_id": r.robot_id,
                "name": r.name,
                "enabled": r.enabled,
                "registered_at": (
                    r.registered_at.isoformat() if r.registered_at else None
                ),
                "last_seen": r.last_seen.isoformat() if r.last_seen else None,
                "metadata": r.metadata or {},
            }
        )

    def _decode(self, obj: Dict[str, Any]) -> RobotInfo:
        return RobotInfo(
            robot_id=obj["robot_id"],
            name=obj.get("name"),
            enabled=bool(obj.get("enabled", True)),
            registered_at=(
                datetime.fromisoformat(obj["registered_at"])
                if obj.get("registered_at")
                else datetime.now()
            ),
            last_seen=(
                datetime.fromisoformat(obj["last_seen"])
                if obj.get("last_seen")
                else None
            ),
            metadata=obj.get("metadata", {}),
        )


def get_shared_memory_fleet_manager_from_env() -> Optional[SharedMemoryFleetManager]:
    registry = os.getenv("DUME_FLEET_BUF")
    namespace = os.getenv("DUME_NAMESPACE", "default")
    if registry:
        return SharedMemoryFleetManager(namespace=namespace, registry_name=registry)
    return None
