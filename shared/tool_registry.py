import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from shared import IToolRegistry, ToolDefinition

logger = logging.getLogger(__name__)


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

    async def call_tool(
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
