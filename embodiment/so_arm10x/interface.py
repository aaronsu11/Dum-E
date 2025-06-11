"""
Concrete implementations of the Iron Man Dum-E robotic system interfaces.

This module provides production-ready implementations of the abstract interfaces
defined in interfaces.py. These implementations handle:
- In-memory task management with async operations
- Event publishing with filtering and history
- Tool registry with categorization and execution
- Hardware interface wrapper for SO100Robot

These implementations are designed to be robust, thread-safe, and suitable
for production use while maintaining clean separation of concerns.
"""

import asyncio
import uuid
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Set
import logging

from ..interfaces import (
    ITaskManager,
    IEventPublisher,
    IToolRegistry,
    IHardwareInterface,
    TaskInfo,
    TaskStatus,
    Event,
    EventType,
    ToolDefinition,
)

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


class InMemoryEventPublisher(IEventPublisher):
    """
    In-memory event publishing with filtering and history.

    Manages event subscriptions and publishing with support for:
    - Event type filtering
    - Task-specific event streams
    - Event history for late subscribers
    - Automatic cleanup of old events
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the event publisher.

        Args:
            max_history: Maximum number of events to keep in history
        """
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._event_history: deque = deque(maxlen=max_history)
        self._lock = asyncio.Lock()
        self._subscriber_filters: Dict[str, Dict[str, Any]] = {}

    async def publish_event(self, event: Event) -> None:
        """Publish event to all matching subscribers and add to history."""
        async with self._lock:
            # Add to history
            self._event_history.append(event)

            # Distribute to subscribers based on their filters
            for subscriber_id, queue in self._subscribers.items():
                if self._event_matches_filter(event, subscriber_id):
                    try:
                        await queue.put(event)
                    except asyncio.QueueFull:
                        logger.warning(
                            f"Event queue full for subscriber {subscriber_id}"
                        )

        logger.debug(
            f"Published event: {event.event_type.value} for task {event.task_id}"
        )

    def _event_matches_filter(self, event: Event, subscriber_id: str) -> bool:
        """Check if event matches subscriber's filter criteria."""
        filters = self._subscriber_filters.get(subscriber_id, {})

        # Check event type filter
        if "event_types" in filters:
            if event.event_type not in filters["event_types"]:
                return False

        # Check task ID filter
        if "task_id" in filters:
            if event.task_id != filters["task_id"]:
                return False

        return True

    async def subscribe(
        self,
        event_types: Optional[List[EventType]] = None,
        task_id: Optional[str] = None,
    ) -> AsyncIterator[Event]:
        """Subscribe to filtered event stream with async iteration support."""
        subscriber_id = str(uuid.uuid4())
        queue = asyncio.Queue(maxsize=100)  # Bounded queue to prevent memory issues

        async with self._lock:
            self._subscribers[subscriber_id] = queue
            self._subscriber_filters[subscriber_id] = {}

            if event_types:
                self._subscriber_filters[subscriber_id]["event_types"] = event_types
            if task_id:
                self._subscriber_filters[subscriber_id]["task_id"] = task_id

        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            # Cleanup subscriber when iteration ends
            async with self._lock:
                self._subscribers.pop(subscriber_id, None)
                self._subscriber_filters.pop(subscriber_id, None)

    async def get_event_history(
        self, task_id: Optional[str] = None, limit: Optional[int] = 100
    ) -> List[Event]:
        """Get historical events with optional filtering."""
        async with self._lock:
            events = list(self._event_history)

            # Filter by task_id if specified
            if task_id:
                events = [event for event in events if event.task_id == task_id]

            # Sort by timestamp descending (newest first)
            events.sort(key=lambda e: e.timestamp, reverse=True)

            # Apply limit
            if limit:
                events = events[:limit]

            return events


class InMemoryToolRegistry(IToolRegistry):
    """
    In-memory tool registry with categorization and execution.

    Manages a collection of tools that can be shared between components.
    Provides categorization, filtering, and safe execution with error handling.
    Tools are stored by name and can be organized by category and hardware requirements.
    """

    def __init__(self):
        """Initialize the tool registry with empty collections."""
        self._tools: Dict[str, ToolDefinition] = {}
        self._lock = asyncio.Lock()

    async def register_tool(self, tool: ToolDefinition) -> None:
        """Register a tool with validation and conflict detection."""
        async with self._lock:
            if tool.name in self._tools:
                logger.warning(f"Overwriting existing tool: {tool.name}")

            self._tools[tool.name] = tool

        logger.info(f"Registered tool: {tool.name} (category: {tool.category})")

    async def unregister_tool(self, name: str) -> None:
        """Remove a tool from the registry."""
        async with self._lock:
            if name in self._tools:
                del self._tools[name]
                logger.info(f"Unregistered tool: {name}")
            else:
                logger.warning(f"Attempted to unregister non-existent tool: {name}")

    async def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve a specific tool by name."""
        async with self._lock:
            return self._tools.get(name)

    async def list_tools(
        self, category: Optional[str] = None, requires_hardware: Optional[bool] = None
    ) -> List[ToolDefinition]:
        """List tools with optional filtering by category and hardware requirements."""
        async with self._lock:
            tools = list(self._tools.values())

            # Filter by category if specified
            if category:
                tools = [tool for tool in tools if tool.category == category]

            # Filter by hardware requirements if specified
            if requires_hardware is not None:
                tools = [
                    tool
                    for tool in tools
                    if tool.requires_hardware == requires_hardware
                ]

            return tools

    async def execute_tool(
        self,
        name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a tool with error handling and logging."""
        async with self._lock:
            tool = self._tools.get(name)

        if not tool:
            raise ValueError(f"Tool not found: {name}")

        try:
            logger.info(f"Executing tool: {name}")

            # Execute the tool function
            # Note: This assumes tools are async; sync tools would need different handling
            if asyncio.iscoroutinefunction(tool.function):
                result = await tool.function(**parameters)
            else:
                result = tool.function(**parameters)

            logger.info(f"Tool {name} executed successfully")

            return {
                "status": "success",
                "result": result,
                "tool_name": name,
                "execution_time": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Tool {name} execution failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "tool_name": name,
                "execution_time": datetime.now().isoformat(),
            }


class SO100HardwareInterface(IHardwareInterface):
    """
    Hardware interface wrapper for SO100Robot.

    Provides async interface over the existing SO100Robot implementation,
    with connection management, error handling, and consistent state reporting.
    This adapter allows the SO100Robot to be used through the standard
    IHardwareInterface without changing existing code.
    """

    def __init__(self, so100_robot):
        """
        Initialize with existing SO100Robot instance.

        Args:
            so100_robot: Configured SO100Robot instance
        """
        self._robot = so100_robot
        self._connected = False
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Establish connection to SO100 hardware with error handling."""
        async with self._lock:
            try:
                if not self._connected:
                    # The SO100Robot connect is synchronous, so we run it in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._robot.connect)
                    self._connected = True
                    logger.info("SO100Robot connected successfully")
            except Exception as e:
                logger.error(f"Failed to connect to SO100Robot: {str(e)}")
                raise

    async def disconnect(self) -> None:
        """Disconnect from SO100 hardware."""
        async with self._lock:
            try:
                if self._connected:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self._robot.disconnect)
                    self._connected = False
                    logger.info("SO100Robot disconnected")
            except Exception as e:
                logger.error(f"Error during SO100Robot disconnect: {str(e)}")
                # Don't re-raise disconnect errors

    async def is_connected(self) -> bool:
        """Check hardware connection status."""
        return self._connected and getattr(self._robot, "is_connected", False)

    async def get_current_state(self) -> Dict[str, Any]:
        """Get current robot state including joint positions."""
        if not await self.is_connected():
            raise RuntimeError("Robot not connected")

        try:
            loop = asyncio.get_event_loop()
            state = await loop.run_in_executor(None, self._robot.get_current_state)

            return {
                "joint_positions": (
                    state.tolist() if hasattr(state, "tolist") else state
                ),
                "timestamp": datetime.now().isoformat(),
                "robot_type": "SO100",
            }
        except Exception as e:
            logger.error(f"Failed to get robot state: {str(e)}")
            raise

    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute robot action with validation and error handling."""
        if not await self.is_connected():
            raise RuntimeError("Robot not connected")

        try:
            # Extract action data based on expected format
            if "target_state" in action:
                target_state = action["target_state"]
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self._robot.set_target_state, target_state
                )

                return {
                    "status": "success",
                    "action_executed": action,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                raise ValueError("Action must contain 'target_state' key")

        except Exception as e:
            logger.error(f"Failed to execute robot action: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_sensor_data(self) -> Dict[str, Any]:
        """Get current sensor data including camera images."""
        if not await self.is_connected():
            raise RuntimeError("Robot not connected")

        try:
            loop = asyncio.get_event_loop()
            images = await loop.run_in_executor(None, self._robot.get_current_images)

            return {
                "cameras": {
                    "arm_camera": {
                        "shape": images["arm_camera"].shape,
                        "dtype": str(images["arm_camera"].dtype),
                        # Note: Not including actual image data to avoid large payloads
                        # Applications can call get_current_images directly for image data
                    },
                    "top_camera": {
                        "shape": images["top_camera"].shape,
                        "dtype": str(images["top_camera"].dtype),
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to get sensor data: {str(e)}")
            raise
