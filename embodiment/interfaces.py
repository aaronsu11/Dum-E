"""
Abstract interfaces for the Iron Man Dum-E robotic system.

This module defines the core interfaces that enable modular, loosely-coupled
components in the robotic assistant architecture. These interfaces support:
- Multiple robot agent implementations
- Shared tool management between components
- Task lifecycle management with status tracking
- Event streaming for real-time updates

The interfaces follow the dependency inversion principle, allowing high-level
modules to depend on abstractions rather than concrete implementations.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime


class TaskStatus(Enum):
    """Enumeration of possible task execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class EventType(Enum):
    """Types of events that can be published during task execution."""

    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TOOL_EXECUTED = "tool_executed"
    STATUS_UPDATE = "status_update"
    STREAMING_DATA = "streaming_data"


@dataclass
class TaskInfo:
    """Information about a task in the system."""

    task_id: str
    instruction: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Event:
    """Event data structure for streaming updates."""

    event_type: EventType
    task_id: Optional[str]
    timestamp: datetime
    data: Dict[str, Any]

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class ToolDefinition:
    """Definition of a tool that can be executed."""

    name: str
    description: str
    function: Callable
    parameters_schema: Dict[str, Any]
    requires_hardware: bool = False
    category: str = "general"


class IRobotAgent(ABC):
    """
    Interface for robot agent implementations.

    This interface defines the contract for agents that can execute natural
    language instructions, stream progress updates, and manage robotic tasks.
    Implementations might use different LLMs, reasoning strategies, or hardware
    interfaces while maintaining the same API.
    """

    @abstractmethod
    async def execute_instruction(self, instruction: str, task_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute a natural language instruction synchronously.

        Args:
            instruction: Natural language command to execute
            task_id: Optional task identifier for tracking

        Returns:
            Dict containing execution results and metadata
        """
        pass

    @abstractmethod
    async def stream_async(self, instruction: str, task_id: Optional[str] = None) -> AsyncIterator[Dict[str, Any]]:
        """
        Execute instruction with streaming progress updates.

        This method yields events during execution, compatible with Strands
        agents' streaming API. Events should include 'message' and/or 'data'
        fields for integration with voice assistant streaming.

        Args:
            instruction: Natural language command to execute
            task_id: Optional task identifier for tracking

        Yields:
            Dict with 'message', 'data', or other event information
        """
        pass

    @abstractmethod
    async def get_available_tools(self) -> List[ToolDefinition]:
        """Get list of tools available to this agent."""
        pass

    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """Get current agent status and health information."""
        pass


class IToolRegistry(ABC):
    """
    Interface for managing shared tools across the system.

    The tool registry enables tools to be shared between the voice assistant
    and robot agent, providing centralized management of available capabilities.
    """

    @abstractmethod
    async def register_tool(self, tool: ToolDefinition) -> None:
        """Register a new tool in the registry."""
        pass

    @abstractmethod
    async def unregister_tool(self, name: str) -> None:
        """Remove a tool from the registry."""
        pass

    @abstractmethod
    async def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve a specific tool by name."""
        pass

    @abstractmethod
    async def list_tools(
        self, category: Optional[str] = None, requires_hardware: Optional[bool] = None
    ) -> List[ToolDefinition]:
        """List available tools, optionally filtered by criteria."""
        pass

    @abstractmethod
    async def execute_tool(
        self, name: str, parameters: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a tool with given parameters and optional context."""
        pass


class ITaskManager(ABC):
    """
    Interface for task lifecycle management.

    Manages the creation, tracking, and lifecycle of tasks in the system.
    Supports task queuing, status updates, and persistence across restarts.
    """

    @abstractmethod
    async def create_task(self, instruction: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new task and return its unique identifier."""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[TaskInfo]:
        """Retrieve task information by ID."""
        pass

    @abstractmethod
    async def update_task_status(self, task_id: str, status: TaskStatus, error_message: Optional[str] = None) -> None:
        """Update the status of a task."""
        pass

    @abstractmethod
    async def update_task_progress(self, task_id: str, progress: float, status_message: Optional[str] = None) -> None:
        """Update task progress (0.0 to 1.0) with optional status message."""
        pass

    @abstractmethod
    async def list_tasks(self, status: Optional[TaskStatus] = None, limit: Optional[int] = None) -> List[TaskInfo]:
        """List tasks, optionally filtered by status."""
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running or pending task."""
        pass


class IEventPublisher(ABC):
    """
    Interface for publishing events during task execution.

    Enables real-time streaming of events to subscribers like the voice
    assistant, MCP clients, or monitoring systems. Events can include
    task progress, tool execution results, and streaming data.
    """

    @abstractmethod
    async def publish_event(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        pass

    @abstractmethod
    async def subscribe(
        self, event_types: Optional[List[EventType]] = None, task_id: Optional[str] = None
    ) -> AsyncIterator[Event]:
        """
        Subscribe to events with optional filtering.

        Args:
            event_types: Optional list of event types to filter
            task_id: Optional task ID to filter events for specific task

        Yields:
            Event objects matching the filter criteria
        """
        pass

    @abstractmethod
    async def get_event_history(self, task_id: Optional[str] = None, limit: Optional[int] = 100) -> List[Event]:
        """Get historical events, optionally filtered by task ID."""
        pass


class IHardwareInterface(ABC):
    """
    Interface for hardware abstraction.

    Provides a consistent interface for robot hardware operations,
    enabling different hardware implementations or mock interfaces
    for testing without changing higher-level code.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to hardware."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from hardware."""
        pass

    @abstractmethod
    async def is_connected(self) -> bool:
        """Check if hardware is connected."""
        pass

    @abstractmethod
    async def get_current_state(self) -> Dict[str, Any]:
        """Get current hardware state (joint positions, etc.)."""
        pass

    @abstractmethod
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a hardware action and return result."""
        pass

    @abstractmethod
    async def get_sensor_data(self) -> Dict[str, Any]:
        """Get current sensor readings (cameras, etc.)."""
        pass
