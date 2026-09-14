'Interfaces for the Dum-E robotic system.'

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Enumeration of possible task execution states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class MessageType(Enum):
    """Types of messages that can be published during task execution."""

    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_PROGRESS = "task_progress"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TOOL_EXECUTED = "tool_executed"
    STATUS_UPDATE = "status_update"
    STREAMING_DATA = "streaming_data"


@dataclass
class RobotInfo:
    """Information about a robot in the fleet."""

    robot_id: str
    name: Optional[str]
    enabled: bool
    registered_at: datetime
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TaskInfo:
    """Information about a task in the system."""

    task_id: str
    instruction: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status_message: Optional[str] = None
    error_message: Optional[str] = None
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Message:
    """Event data structure for streaming updates."""

    message_type: MessageType
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


class BackendConfig(BaseModel):
    'Backend configuration for coordinating agent/server communication.'

    # Local same-process coordination key
    namespace: str = "default"

    # AWS backend
    aws_region: Optional[str] = None
    # MQTT-based progress tracking
    mqtt_endpoint: Optional[str] = None
    mqtt_topic_prefix: Optional[str] = None
    # DynamoDB/DB-backed task coordination
    dynamodb_table: Optional[str] = None
