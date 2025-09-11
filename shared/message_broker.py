import asyncio
import logging
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, AsyncIterator

from shared import IMessageBroker, Message, MessageType

logger = logging.getLogger(__name__)


class InMemoryMessageBroker(IMessageBroker):
    """
    In-memory message publishing with filtering and history.

    Manages message subscriptions and publishing with support for:
    - Message type filtering
    - Task-specific message streams
    - Message history for late subscribers
    - Automatic cleanup of old messages
    """

    def __init__(self, max_history: int = 1000):
        """
        Initialize the message broker.

        Args:
            max_history: Maximum number of messages to keep in history
        """
        self._subscribers: Dict[str, asyncio.Queue] = {}
        self._message_history: deque = deque(maxlen=max_history)
        self._lock = asyncio.Lock()
        self._subscriber_filters: Dict[str, Dict[str, Any]] = {}

    async def publish(self, message: Message) -> None:
        """Publish message to all matching subscribers and add to history."""
        async with self._lock:
            # Add to history
            self._message_history.append(message)

            # Distribute to subscribers based on their filters
            for subscriber_id, queue in self._subscribers.items():
                if self._message_matches_filter(message, subscriber_id):
                    try:
                        await queue.put(message)
                    except asyncio.QueueFull:
                        logger.warning(
                            f"Message queue full for subscriber {subscriber_id}"
                        )

        logger.debug(
            f"Published message: {message.message_type.value} for task {message.task_id}"
        )

    def _message_matches_filter(self, message: Message, subscriber_id: str) -> bool:
        """Check if message matches subscriber's filter criteria."""
        filters = self._subscriber_filters.get(subscriber_id, {})

        # Check message type filter
        if "message_types" in filters:
            if message.message_type not in filters["message_types"]:
                return False

        # Check task ID filter
        if "task_id" in filters:
            if message.task_id != filters["task_id"]:
                return False

        return True

    async def subscribe(
        self,
        message_types: Optional[List[MessageType]] = None,
        task_id: Optional[str] = None,
    ) -> AsyncIterator[Message]:
        """Subscribe to filtered message stream with async iteration support."""
        subscriber_id = str(uuid.uuid4())
        queue = asyncio.Queue(maxsize=100)  # Bounded queue to prevent memory issues

        async with self._lock:
            self._subscribers[subscriber_id] = queue
            self._subscriber_filters[subscriber_id] = {}

            if message_types:
                self._subscriber_filters[subscriber_id]["message_types"] = message_types
            if task_id:
                self._subscriber_filters[subscriber_id]["task_id"] = task_id

        try:
            while True:
                message = await queue.get()
                yield message
        finally:
            # Cleanup subscriber when iteration ends
            async with self._lock:
                self._subscribers.pop(subscriber_id, None)
                self._subscriber_filters.pop(subscriber_id, None)

    async def get_message_history(
        self, task_id: Optional[str] = None, limit: Optional[int] = 100
    ) -> List[Message]:
        """Get historical messages with optional filtering."""
        async with self._lock:
            events = list(self._message_history)

            # Filter by task_id if specified
            if task_id:
                events = [message for message in events if message.task_id == task_id]

            # Sort by timestamp descending (newest first)
            events.sort(key=lambda m: m.timestamp, reverse=True)

            # Apply limit
            if limit:
                events = events[:limit]

            return events
