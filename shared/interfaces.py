'Interfaces for the Dum-E robotic system.'

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, AsyncIterator, Dict, List, Optional


logger = logging.getLogger(__name__)


from .types import TaskStatus, MessageType, RobotInfo, TaskInfo, Message, ToolDefinition

class IRobotController(ABC):
    'Interface for robot controller abstraction.'

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the robot controller."""
        pass

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to robot controller."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from robot controller."""
        pass

    @contextmanager
    def activate(self):
        self.connect()
        try:
            yield self
        finally:
            self.disconnect()

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if robot controller is connected."""
        pass

    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """Get current robot controller observation (joint positions, etc.)."""
        pass

    @abstractmethod
    def set_target_state(self, target_state: Dict[str, float]) -> Dict[str, float]:
        """Apply named targets using embodiment safety rules; return targets sent."""
        pass


class IPolicyBackend(ABC):
    'Interface for policy (inference) backend abstraction.'

    @abstractmethod
    def get_action(
        self, observation_dict: dict[str, Any], lang: str | None = None
    ) -> list[dict[str, float]]:
        'Run one inference step and return a list of per-timestep action dicts.'
        pass

    @property
    @abstractmethod
    def language_instruction(self) -> str | None:
        'The instruction the backend conditions on when ``lang`` is omitted.'
        pass

    @abstractmethod
    def set_lang_instruction(self, lang_instruction: str) -> None:
        """Set the stored language instruction."""
        pass

    @abstractmethod
    def ping(self) -> bool:
        """Check backend reachability. Returns False rather than raising."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset per-episode backend state (e.g. recreate the transport socket).

        Must be safe to call when nothing is in flight.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release backend resources. MUST be idempotent."""
        pass

    @contextmanager
    def session(self):
        """Scope a backend to one episode: ``reset()`` on entry, ``close()`` on exit.

        ``close()`` runs in a ``finally`` so the lifecycle cannot be forgotten at
        a call site, and a raising body still releases the transport.
        """
        self.reset()
        try:
            yield self
        finally:
            self.close()


class IRobotAgent(ABC):
    'Interface for robot agent implementations.'

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for the robot agent."""
        pass

    @abstractmethod
    async def arun(
        self, instruction: str, task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        'Execute a natural language instruction asynchronously.'
        pass

    @abstractmethod
    async def astream(
        self, instruction: str, task_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        'Execute instruction with streaming progress updates asynchronously.'
        pass

    @abstractmethod
    async def get_available_tools(self) -> List[ToolDefinition]:
        """Get list of tools available to this agent."""
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status and health information."""
        pass


class ITaskManager(ABC):
    'Interface for task lifecycle management.'

    @abstractmethod
    async def create_task(
        self, instruction: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new task and return its unique identifier."""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Retrieve task information by ID."""
        pass

    @abstractmethod
    async def update_task(
        self,
        task_id: str,
        status: TaskStatus,
        status_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the status of a task."""
        pass

    @abstractmethod
    async def list_tasks(
        self, status: Optional[TaskStatus] = None, limit: Optional[int] = None
    ) -> List[TaskInfo]:
        """List tasks, optionally filtered by status."""
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running or pending task."""
        pass

    @abstractmethod
    async def claim_task(self, task_id: str, worker_id: str) -> bool:
        'Atomically claim a PENDING task for execution by a worker.'
        pass


class IMessageBroker(ABC):
    'Interface for publishing and subscribing to messages during task execution.'

    @abstractmethod
    async def publish(self, message: Message) -> None:
        """Publish a message to all subscribers."""
        pass

    @abstractmethod
    async def subscribe(
        self,
        message_types: Optional[List[MessageType]] = None,
        task_id: Optional[str] = None,
    ) -> AsyncIterator[Message]:
        'Subscribe to messages with optional filtering.'
        pass

    @abstractmethod
    async def get_message_history(
        self, task_id: Optional[str] = None, limit: Optional[int] = 100
    ) -> List[Message]:
        """Get historical messages, optionally filtered by task ID."""
        pass


class IFleetManager(ABC):
    'Interface for basic fleet management operations.'

    @abstractmethod
    async def register_robot(
        self,
        robot_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RobotInfo:
        """Register or upsert a robot and return its info."""
        pass

    @abstractmethod
    async def list_robots(self, only_enabled: Optional[bool] = None) -> List[RobotInfo]:
        """List robots with optional enabled filter."""
        pass

    @abstractmethod
    async def get_robot(self, robot_id: str) -> Optional[RobotInfo]:
        """Get robot information by ID."""
        pass

    @abstractmethod
    async def set_enabled(self, robot_id: str, enabled: bool) -> bool:
        """Enable or disable a robot by ID."""
        pass

    @abstractmethod
    async def update_robot(
        self,
        robot_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update robot name and/or metadata. Returns True if updated."""
        pass




class IPolicyMapping(ABC):
    """Embodiment-owned binding of named hardware values to model coordinates.

    Model normalization stays in the checkpoint processor. A mapping must not
    open hardware, command motors, or infer units from observed numeric ranges.
    """
    @property
    @abstractmethod
    def joint_names(self) -> tuple[str, ...]: ...

    @property
    @abstractmethod
    def camera_names(self) -> tuple[str, ...]: ...

    @abstractmethod
    def to_model(self, values): ...

    @abstractmethod
    def to_arm(self, values): ...

    @abstractmethod
    def validate(self) -> None: ...

    @property
    def metadata(self) -> dict:
        return {}
