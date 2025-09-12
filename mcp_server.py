import asyncio
import contextlib
from typing import Optional, List, Dict, Any, Set
from datetime import datetime

import logging
from fastmcp import FastMCP, Context

# Interfaces (use abstractions to support local or OTA backends)
from shared import (
    IMessageBroker,
    ITaskManager,
    IFleetManager,
    Message,
    MessageType,
    TaskStatus,
    TaskInfo,
    RobotInfo,
)
from shared.message_broker import get_shared_memory_broker_from_env
from shared.task_manager import get_shared_memory_task_manager_from_env
from shared.fleet_manager import get_shared_memory_fleet_manager_from_env

mcp = FastMCP(
    name="Dum-E MCP Server",
    instructions="This is a MCP server for the Dum-E robot",
)

logger = logging.getLogger("mcp_server")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
logger.info("MCP server module loaded; initializing backends")

# Prefer shared-memory backends when DUME_IPC=shm and env provides shm names
_SHM_TM = get_shared_memory_task_manager_from_env()
_SHM_MB = get_shared_memory_broker_from_env()
_SHM_FLEET = get_shared_memory_fleet_manager_from_env()
if _SHM_TM and _SHM_MB:
    TASK_MANAGER: ITaskManager = _SHM_TM
    MESSAGE_BROKER: IMessageBroker = _SHM_MB
else:
    raise NotImplementedError("Only shm is supported for now")

# Fleet manager is optional; some deployments may not set DUME_FLEET_BUF
FLEET_MANAGER: Optional[IFleetManager] = _SHM_FLEET if _SHM_FLEET else None


async def _forward_event_to_ctx(event_data: dict, ctx: Context) -> None:
    """Forward a broker event payload to MCP client via ctx.report_progress."""
    # Extract human-friendly message text if present
    text_message: Optional[str] = None
    if "message" in event_data and isinstance(event_data["message"], dict):
        contents = event_data["message"].get("content", [])
        texts: List[str] = []
        for c in contents:
            if isinstance(c, dict) and "text" in c:
                texts.append(c["text"])
        if texts:
            text_message = " ".join(texts)
    # Fallbacks
    if not text_message and "type" in event_data:
        text_message = str(event_data.get("type"))
    if not text_message:
        text_message = "Processing..."

    # Progress, if provided by the event
    progress_value = event_data.get("progress")
    if isinstance(progress_value, (int, float)):
        # Normalize floats in [0,1] to percentage scale
        if isinstance(progress_value, float) and 0.0 <= progress_value <= 1.0:
            progress = int(progress_value * 100)
            total = 100
        else:
            # Treat value as absolute with unknown total; use 100 as nominal total
            progress = int(progress_value)
            total = 100
        logger.debug(
            "Forwarding progress to client: progress=%s total=%s message=%s",
            progress,
            total,
            text_message,
        )
        await ctx.report_progress(progress=progress, total=total, message=text_message)
    else:
        logger.debug("Forwarding message to client: %s", text_message)
        await ctx.report_progress(progress=0, total=100, message=text_message)


def _task_to_dict(task: TaskInfo) -> Dict[str, Any]:
    return {
        "task_id": task.task_id,
        "instruction": task.instruction,
        "status": task.status.value,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "error_message": task.error_message,
        "progress": task.progress,
        "metadata": task.metadata or {},
    }


def _message_to_dict(message: Message) -> Dict[str, Any]:
    return {
        "message_type": message.message_type.value if message.message_type else None,
        "task_id": message.task_id,
        "timestamp": message.timestamp.isoformat() if message.timestamp else None,
        "data": message.data,
    }


def _parse_task_status(status: Optional[str]) -> Optional[TaskStatus]:
    if not status:
        return None
    try:
        return TaskStatus(status.lower())
    except Exception:
        # Accept also uppercase keys like "PENDING"
        try:
            return TaskStatus[status.upper()]
        except Exception:
            return None


@mcp.tool()
async def list_tasks(status: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
    """List tasks with optional status filter. Status can be one of pending, running, completed, failed, cancelled."""
    filt = _parse_task_status(status)
    tasks = await TASK_MANAGER.list_tasks(status=filt, limit=limit)
    return {"tasks": [_task_to_dict(t) for t in tasks]}


def _robot_to_dict(info: RobotInfo) -> Dict[str, Any]:
    return {
        "robot_id": info.robot_id,
        "name": info.name,
        "enabled": info.enabled,
        "registered_at": info.registered_at.isoformat() if info.registered_at else None,
        "last_seen": info.last_seen.isoformat() if info.last_seen else None,
        "metadata": info.metadata or {},
    }


@mcp.tool()
async def register_robot(
    robot_id: str, name: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Register or upsert a robot in the fleet registry. Returns the robot info."""
    if not FLEET_MANAGER:
        return {"error": "Fleet manager not available in this deployment"}
    info = await FLEET_MANAGER.register_robot(
        robot_id=robot_id, name=name, metadata=metadata
    )
    return {"robot": _robot_to_dict(info)}


@mcp.tool()
async def list_robots(only_enabled: Optional[bool] = None) -> Dict[str, Any]:
    """List robots from the fleet registry. Optionally filter by enabled state."""
    if not FLEET_MANAGER:
        return {"error": "Fleet manager not available in this deployment"}
    items = await FLEET_MANAGER.list_robots(only_enabled=only_enabled)
    return {"robots": [_robot_to_dict(r) for r in items]}


@mcp.tool()
async def get_robot(robot_id: str) -> Dict[str, Any]:
    """Get detailed information for a robot by ID."""
    if not FLEET_MANAGER:
        return {"error": "Fleet manager not available in this deployment"}
    info = await FLEET_MANAGER.get_robot(robot_id)
    if not info:
        return {"error": f"Robot not found: {robot_id}"}
    return {"robot": _robot_to_dict(info)}


@mcp.tool()
async def set_robot_enabled(robot_id: str, enabled: bool) -> Dict[str, Any]:
    """Enable or disable a robot by ID."""
    if not FLEET_MANAGER:
        return {"error": "Fleet manager not available in this deployment"}
    ok = await FLEET_MANAGER.set_enabled(robot_id, enabled)
    return {"robot_id": robot_id, "enabled": enabled, "updated": bool(ok)}


@mcp.tool()
async def get_task_details(
    task_id: str, include_messages: bool = False, messages_limit: int = 100
) -> Dict[str, Any]:
    """Get details for a given task. Optionally include recent messages for that task."""
    task = await TASK_MANAGER.get_task(task_id)
    if not task:
        return {"error": f"Task not found: {task_id}"}
    result: Dict[str, Any] = {"task": _task_to_dict(task)}
    if include_messages:
        msgs = await MESSAGE_BROKER.get_message_history(
            task_id=task_id, limit=messages_limit
        )
        result["messages"] = [_message_to_dict(m) for m in msgs]
    return result


@mcp.tool()
async def cancel_task(task_id: str) -> Dict[str, Any]:
    """Cancel a running or pending task by ID."""
    success = await TASK_MANAGER.cancel_task(task_id)
    if success:
        # Notify subscribers of cancellation
        try:
            await MESSAGE_BROKER.publish(
                Message(
                    message_type=MessageType.STATUS_UPDATE,
                    task_id=task_id,
                    timestamp=datetime.now(),
                    data={"status": "cancelled", "source": "mcp_server"},
                )
            )
        except Exception:
            pass
    return {"task_id": task_id, "cancelled": bool(success)}


@mcp.tool()
async def list_last_messages(
    limit: int = 10,
    task_id: Optional[str] = None,
    message_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return last N broker messages, optionally filtered by task_id and message types."""
    msgs = await MESSAGE_BROKER.get_message_history(task_id=task_id, limit=limit)
    # Optional type filter
    allowed_types: Optional[Set[str]] = None
    if message_types:
        allowed_types = {t.lower() for t in message_types}

    out = []
    for m in msgs:
        mtype_val = m.message_type.value if m.message_type else None
        if allowed_types is not None and (
            mtype_val is None or mtype_val.lower() not in allowed_types
        ):
            continue
        out.append(_message_to_dict(m))
    return {"messages": out}


@mcp.tool(meta={"long_running": True})
async def execute_robot_instruction(
    instruction: str,
    robot_id: Optional[str],
    ctx: Context,
    timeout_s: int = 900,
) -> dict:
    """
    Trigger a robot task on the edge agent and stream progress to the MCP client.

    This creates a task via ITaskManager and publishes a TASK_STARTED message via
    IMessageBroker. The edge Dum-E agent should pick up the task (via its own
    subscription to the broker), execute it, and publish streaming updates.

    All updates are forwarded to the MCP client through ctx.report_progress,
    making this tool compatible with streamable HTTP clients.
    """
    # Create task record
    task_id = await TASK_MANAGER.create_task(
        instruction,
        metadata={
            "source": "mcp_server",
            "robot_id": robot_id,
            "created_at": datetime.now().isoformat(),
        },
    )
    logger.info("Created task %s for instruction: %s", task_id, instruction)

    # Announce task start so an edge agent can claim/execute it
    await MESSAGE_BROKER.publish(
        Message(
            message_type=MessageType.TASK_STARTED,
            task_id=task_id,
            timestamp=datetime.now(),
            data={
                "instruction": instruction,
                "robot_id": robot_id,
                "source": "mcp_server",
            },
        )
    )
    logger.info("Published TASK_STARTED for task %s", task_id)

    await ctx.report_progress(progress=0, total=100, message="Task dispatched to agent")

    # Stream events for this task back to the MCP client
    done = asyncio.Event()
    final_status: str = "unknown"
    final_error: Optional[str] = None

    async def _stream_updates():
        nonlocal final_status, final_error
        async for message in MESSAGE_BROKER.subscribe(
            message_types=[
                MessageType.STREAMING_DATA,
                MessageType.TASK_PROGRESS,
                MessageType.TASK_COMPLETED,
                MessageType.TASK_FAILED,
            ],
            task_id=task_id,
        ):
            if message.message_type in (
                MessageType.STREAMING_DATA,
                MessageType.TASK_PROGRESS,
            ):
                await _forward_event_to_ctx(message.data, ctx)
                continue

            if message.message_type == MessageType.TASK_COMPLETED:
                await ctx.report_progress(
                    progress=100, total=100, message="Task completed"
                )
                final_status = "completed"
                done.set()
                break

            if message.message_type == MessageType.TASK_FAILED:
                err_text = (
                    message.data.get("error")
                    if isinstance(message.data, dict)
                    else None
                )
                await ctx.report_progress(
                    progress=100, total=100, message=f"Task failed: {err_text}"
                )
                final_status = "failed"
                final_error = err_text
                done.set()
                break

    streamer = asyncio.create_task(_stream_updates())

    try:
        await asyncio.wait_for(done.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        await ctx.report_progress(
            progress=100, total=100, message="Timed out waiting for agent updates"
        )
        final_status = "timeout"
    finally:
        streamer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await streamer

    result: dict = {"task_id": task_id, "status": final_status}
    if final_error:
        result["error"] = final_error
    logger.info("Returning result for task %s: %s", task_id, result)
    return result


@mcp.tool()
async def list_namespaces() -> Dict[str, Any]:
    """Return the current namespace from environment to help clients verify scope."""
    import os

    return {"namespace": os.getenv("DUME_NAMESPACE", "default")}


if __name__ == "__main__":
    import os

    host = os.getenv("DUME_MCP_HOST", "127.0.0.1")
    try:
        port = int(os.getenv("DUME_MCP_PORT", "8000"))
    except Exception:
        port = 8000
    logger.info("Starting Dum-E MCP HTTP server on %s:%s", host, port)
    # Bind explicitly to host/port to avoid conflicts and allow tests to override
    mcp.run(transport="http", host=host, port=port)
