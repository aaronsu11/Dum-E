"""
Tests for MCP tools exposed by mcp_server.py using shared-memory backends.

These tests:
- Create ephemeral SharedMemory segments and expose them via env vars
- Start only the MCP HTTP server process (no robot agent or pipecat)
- Use FastMCP Client to invoke server tools and validate responses
- Interact with the same SHM buffers directly via SharedMemoryTaskManager and
  SharedMemoryMessageBroker to seed tasks and messages

All tests are robot-agnostic and validate only the server interfaces.
"""

import asyncio
import socket
from contextlib import closing

import pytest
import pytest_asyncio
from types import SimpleNamespace
from multiprocessing.managers import SharedMemoryManager

from fastmcp import Client

from shared import Message, MessageType, TaskStatus
from shared.message_broker.shm import SharedMemoryMessageBroker
from shared.task_manager.shm import SharedMemoryTaskManager
import dum_e


def _find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


async def _wait_for_port(host: str, port: int, timeout: float = 5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except Exception:
            if asyncio.get_event_loop().time() > deadline:
                raise
            await asyncio.sleep(0.05)


@pytest_asyncio.fixture
async def shm_env_and_server(monkeypatch):
    """Start an MCP HTTP server wired to SHM backends and yield context.

    Yields a dict containing:
    - client: FastMCP Client connected to the server
    - tm: SharedMemoryTaskManager bound to the same SHM segment
    - broker: SharedMemoryMessageBroker bound to the same SHM segments
    - port: server port
    - namespace: namespace used for SHM and server
    - cleanup: async callable to terminate the server process
    """

    namespace = "test_mcp"
    port = _find_free_port()

    with SharedMemoryManager() as smm:
        slot = 4096
        broker_buf = smm.ShareableList([" " * slot] * 32)
        broker_meta = smm.ShareableList([0])
        tasks_buf = smm.ShareableList([" " * slot] * 16)
        fleet_buf = smm.ShareableList([" " * slot] * 32)

        env = {
            "DUME_IPC": "shm",
            "DUME_NAMESPACE": namespace,
            "DUME_BROKER_BUF": broker_buf.shm.name,
            "DUME_BROKER_META": broker_meta.shm.name,
            "DUME_TASKS_BUF": tasks_buf.shm.name,
            "DUME_FLEET_BUF": fleet_buf.shm.name,
            "DUME_MCP_PORT": str(port),
            "DUME_MCP_HOST": "127.0.0.1",
        }
        for k, v in env.items():
            monkeypatch.setenv(k, v)

        # Use resolved backend config via builder to avoid strict constructor
        args = SimpleNamespace(namespace=namespace)
        backend_cfg = dum_e.build_backend_config(args, {})
        server_proc = dum_e._spawn_mcp_server(backend_cfg, env)

        try:
            await _wait_for_port("127.0.0.1", port, timeout=5.0)

            client = Client(
                f"http://127.0.0.1:{port}/mcp",
                init_timeout=3,
                timeout=5,
            )
            # Bind helpers to the same SHM segments for direct manipulation
            broker = SharedMemoryMessageBroker(
                namespace=namespace,
                buffer_name=broker_buf.shm.name,
                meta_name=broker_meta.shm.name,
                poll_interval_s=0.005,
            )
            tm = SharedMemoryTaskManager(
                namespace=namespace, tasks_name=tasks_buf.shm.name
            )

            async with client:
                yield {
                    "client": client,
                    "tm": tm,
                    "broker": broker,
                    "port": port,
                    "namespace": namespace,
                }
        finally:
            if server_proc:
                server_proc.terminate()


@pytest.mark.asyncio
async def test_list_namespaces(shm_env_and_server):
    """Ensure the server exposes the current namespace via list_namespaces."""
    client = shm_env_and_server["client"]
    ns = shm_env_and_server["namespace"]
    result = await client.call_tool("list_namespaces", {})
    assert result.data.get("namespace") == ns


@pytest.mark.asyncio
async def test_list_tasks_empty(shm_env_and_server):
    """When no tasks exist, list_tasks should return an empty list."""
    client = shm_env_and_server["client"]
    result = await client.call_tool("list_tasks", {})
    assert isinstance(result.data, dict)
    assert result.data.get("tasks") == []


@pytest.mark.asyncio
async def test_task_lifecycle_and_list_filters(shm_env_and_server):
    """Validate list_tasks with status filters and limit using SHM task manager."""
    client = shm_env_and_server["client"]
    tm: SharedMemoryTaskManager = shm_env_and_server["tm"]

    # Create three tasks with varying statuses
    t1 = await tm.create_task("inspect", {"robot_id": "r1"})
    t2 = await tm.create_task("move", {"robot_id": "r2"})
    t3 = await tm.create_task("pick", {"robot_id": "r1"})

    await tm.update_task(t2, TaskStatus.RUNNING)
    await tm.update_task(t3, TaskStatus.COMPLETED)

    # No filter
    res_all = await client.call_tool("list_tasks", {"limit": 10})
    tasks_all = res_all.data.get("tasks", [])
    assert len(tasks_all) >= 3

    # Filter by running
    res_run = await client.call_tool("list_tasks", {"status": "running"})
    run_tasks = res_run.data.get("tasks", [])
    assert all(t["status"] == "running" for t in run_tasks)
    assert any(t["task_id"] == t2 for t in run_tasks)

    # Filter by completed
    res_done = await client.call_tool("list_tasks", {"status": "completed"})
    done_tasks = res_done.data.get("tasks", [])
    assert any(t["task_id"] == t3 for t in done_tasks)


@pytest.mark.asyncio
async def test_get_task_details_with_messages(shm_env_and_server):
    """get_task_details should include task fields and optionally recent messages."""
    client = shm_env_and_server["client"]
    tm: SharedMemoryTaskManager = shm_env_and_server["tm"]
    broker: SharedMemoryMessageBroker = shm_env_and_server["broker"]

    # Create a task and publish messages against it
    task_id = await tm.create_task("status check", {"robot_id": "r9"})

    await broker.publish(
        Message(
            message_type=MessageType.STATUS_UPDATE,
            task_id=task_id,
            timestamp=None,
            data={"text": "starting", "robot_id": "r9"},
        )
    )
    await broker.publish(
        Message(
            message_type=MessageType.TASK_PROGRESS,
            task_id=task_id,
            timestamp=None,
            data={"progress": 0.5, "message": {"content": [{"text": "halfway"}]}},
        )
    )

    res = await client.call_tool(
        "get_task_details",
        {"task_id": task_id, "include_messages": True, "messages_limit": 5},
    )
    body = res.data
    assert body["task"]["task_id"] == task_id
    assert body["task"]["instruction"] == "status check"
    assert isinstance(body.get("messages"), list) and len(body["messages"]) >= 2


@pytest.mark.asyncio
async def test_cancel_task(shm_env_and_server):
    """cancel_task should flip task status to cancelled and acknowledge success."""
    client = shm_env_and_server["client"]
    tm: SharedMemoryTaskManager = shm_env_and_server["tm"]

    task_id = await tm.create_task("to-cancel", {"robot_id": "r2"})
    res = await client.call_tool("cancel_task", {"task_id": task_id})
    assert res.data.get("cancelled") is True
    # Verify via SHM task manager view
    t = await tm.get_task(task_id)
    assert t is not None and t.status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_list_last_messages_filters(shm_env_and_server):
    """list_last_messages should filter by task_id and message types when provided."""
    client = shm_env_and_server["client"]
    broker: SharedMemoryMessageBroker = shm_env_and_server["broker"]

    # Create messages for two tasks
    task_a = "task-a"
    task_b = "task-b"
    await broker.publish(
        Message(
            message_type=MessageType.TASK_PROGRESS,
            task_id=task_a,
            timestamp=None,
            data={"x": 1},
        )
    )
    await broker.publish(
        Message(
            message_type=MessageType.TASK_COMPLETED,
            task_id=task_a,
            timestamp=None,
            data={},
        )
    )
    await broker.publish(
        Message(
            message_type=MessageType.STATUS_UPDATE,
            task_id=task_b,
            timestamp=None,
            data={"y": 2},
        )
    )

    # Filter by task_id
    res_task = await client.call_tool(
        "list_last_messages", {"limit": 10, "task_id": task_a}
    )
    msgs_task = res_task.data.get("messages", [])
    assert all(m.get("task_id") == task_a for m in msgs_task)

    # Filter by message_types
    res_types = await client.call_tool(
        "list_last_messages",
        {"limit": 10, "message_types": ["task_completed"]},
    )
    msgs_types = res_types.data.get("messages", [])
    assert all(m.get("message_type") == "task_completed" for m in msgs_types)


@pytest.mark.asyncio
async def test_fleet_register_get_and_enable_disable(shm_env_and_server):
    """Register robots, fetch details, and toggle enabled state via MCP tools."""
    client = shm_env_and_server["client"]

    # Register a new robot
    res_reg = await client.call_tool(
        "register_robot",
        {"robot_id": "r-001", "name": "alpha", "metadata": {"model": "so101"}},
    )
    robot = res_reg.data.get("robot")
    assert robot["robot_id"] == "r-001"
    assert robot["enabled"] is True
    assert robot["name"] == "alpha"

    # Get robot details
    res_get = await client.call_tool("get_robot", {"robot_id": "r-001"})
    got = res_get.data.get("robot")
    assert got["robot_id"] == "r-001"
    assert got["metadata"].get("model") == "so101"

    # Disable robot
    res_dis = await client.call_tool(
        "set_robot_enabled", {"robot_id": "r-001", "enabled": False}
    )
    assert res_dis.data.get("updated") is True
    res_get2 = await client.call_tool("get_robot", {"robot_id": "r-001"})
    assert res_get2.data["robot"]["enabled"] is False

    # List robots, ensure our robot is present
    res_list = await client.call_tool("list_robots", {})
    ids = [r["robot_id"] for r in res_list.data.get("robots", [])]
    assert "r-001" in ids


@pytest.mark.asyncio
async def test_fleet_list_robots_only_enabled_filter(shm_env_and_server):
    """list_robots should filter by enabled state when provided."""
    client = shm_env_and_server["client"]

    # Register two robots and disable one
    await client.call_tool("register_robot", {"robot_id": "r-A"})
    await client.call_tool("register_robot", {"robot_id": "r-B"})
    await client.call_tool("set_robot_enabled", {"robot_id": "r-B", "enabled": False})

    res_enabled = await client.call_tool("list_robots", {"only_enabled": True})
    enabled_ids = {r["robot_id"] for r in res_enabled.data.get("robots", [])}
    assert "r-A" in enabled_ids and "r-B" not in enabled_ids

    res_disabled = await client.call_tool("list_robots", {"only_enabled": False})
    disabled_ids = {r["robot_id"] for r in res_disabled.data.get("robots", [])}
    assert "r-B" in disabled_ids


@pytest.mark.asyncio
async def test_retarget_running_task_publishes_control_without_new_task(shm_env_and_server):
    client=shm_env_and_server['client'];tm=shm_env_and_server['tm'];broker=shm_env_and_server['broker']
    task_id=await tm.create_task('banana')
    rejected=await client.call_tool('retarget_robot_instruction',{'task_id':task_id,'instruction':'apple'})
    assert rejected.data['status']=='rejected'
    await tm.update_task(task_id,TaskStatus.RUNNING)
    result=await client.call_tool('retarget_robot_instruction',{'task_id':task_id,'instruction':'apple'})
    assert result.data['status']=='requested' and result.data['task_id']==task_id
    messages=await broker.get_message_history(task_id=task_id,limit=10)
    assert any(m.message_type==MessageType.STATUS_UPDATE and m.data.get('action')=='retarget' and
               m.data.get('instruction')=='apple' for m in messages)
    assert len(await tm.list_tasks())==1


@pytest.mark.asyncio
@pytest.mark.parametrize('scenario',['failure','retarget','cancel'])
async def test_async_agent_mcp_integration(shm_env_and_server,monkeypatch,scenario):
    """Real HTTP/SHM/worker/agent/tools/scheduler; simulated model and hardware."""
    import time
    import threading
    from contextlib import suppress
    from unittest.mock import AsyncMock
    import numpy as np
    from embodiment.so_arm10x.agent import SO10xRobotAgent, _agent_worker_loop
    from embodiment.so_arm10x.async_pick import AsyncPickSkill
    from policy.lerobot.async_chunks import AsyncSettings
    from policy_guard.replay_contract import JOINT_ORDER
    import embodiment.so_arm10x.async_pick as driver
    client=shm_env_and_server['client'];tm=shm_env_and_server['tm'];broker=shm_env_and_server['broker']
    monkeypatch.setenv('DUME_ASYNC_INFERENCE','1')
    monkeypatch.setenv('DUME_ASYNC_LATENCY_PATH','test-measured-settings')
    monkeypatch.delenv('DUME_ASYNC_TRACE_DIRECTORY',raising=False)
    monkeypatch.setattr(driver,'load_settings',lambda path:AsyncSettings(.15))
    class Controller:
        id='integration-arm'
        def __init__(self):self.connected=False;self.actions=[];self.resets=[];self.threads=[]
        def touch(self):self.threads.append(threading.current_thread().name)
        def connect(self):self.touch();self.connected=True
        def disconnect(self):self.touch();self.connected=False
        def is_connected(self):return self.connected
        def get_current_images(self):self.touch();return {'front':np.zeros((4,4,3),dtype=np.uint8)}
        def get_observation(self):self.touch();return {'step':len(self.actions)}
        def move_to_initial_pose(self):self.touch();self.resets.append('initial')
        def move_to_ready_pose(self):self.touch();self.resets.append('ready')
        def set_target_state(self,target):self.touch();self.actions.append(dict(target));return dict(target)
    class Policy:
        language_instruction='banana';_handshaken=True
        def __init__(self):self._session=SimpleNamespace();self.calls=[];self.fail=False;self.closed=False
        def set_lang_instruction(self,text):self.language_instruction=text
        def set_task(self,text):self.language_instruction=text
        def get_action(self,obs,instruction):
            self.calls.append((instruction,threading.current_thread().name));time.sleep(.04)
            if self.fail:raise RuntimeError('injected inference failure')
            return [dict.fromkeys(JOINT_ORDER,.2 if instruction=='apple' else .1) for _ in range(16)]
        def close(self):self.closed=True
    controller=Controller();policy=Policy()
    agent=SO10xRobotAgent(controller,policy,task_manager=tm,message_broker=broker)
    original_run=agent.async_pick.run
    # Bound the real start_pick tool to two chunks for this test only.
    def bounded_run(*args,**kwargs):return original_run(*args,**dict(kwargs,actions_to_execute=2))
    agent.async_pick.run=bounded_run
    tools={t.tool_name:t for t in agent._robot_tools}
    class ScriptedModel:
        async def stream_async(self,instruction):
            await tools['start_pick']._tool_func(item='a banana')
            yield {'result':SimpleNamespace(message={'role':'assistant','content':[{'text':'done'}]})}
    agent._get_strands_agent=AsyncMock(return_value=ScriptedModel())
    worker=asyncio.create_task(_agent_worker_loop(agent,tm,broker,controller.id,'integration-worker'))
    execution=None
    async def until(predicate,timeout=4):
        async with asyncio.timeout(timeout):
            while not predicate():await asyncio.sleep(.005)
    try:
        await asyncio.sleep(.03)  # subscribe before MCP publishes TASK_CREATED
        execution=asyncio.create_task(client.call_tool('execute_robot_instruction',
            {'instruction':'pick banana','robot_id':controller.id,'timeout_s':8}))
        await until(lambda:len(controller.actions)>=4)
        task_id=agent._active_task_id
        assert task_id and (await tm.get_task(task_id)).status==TaskStatus.RUNNING
        # This HTTP request must complete while inference and dispatch are active.
        count=len(controller.actions);started=time.monotonic()
        details=await client.call_tool('get_task_details',{'task_id':task_id})
        assert time.monotonic()-started<1 and details.data['task']['status']=='running'
        assert any(name.startswith('policy-inference') for _,name in policy.calls)
        if scenario=='failure':policy.fail=True
        elif scenario=='retarget':
            response=await client.call_tool('retarget_robot_instruction',{'task_id':task_id,'instruction':'apple'})
            assert response.data['status']=='requested'
            await until(lambda:any(text=='apple' for text,_ in policy.calls))
        else:
            response=await client.call_tool('cancel_task',{'task_id':task_id})
            assert response.data['cancelled'] is True
        result=await execution
        messages=await broker.get_message_history(task_id=task_id,limit=32)
        task=await tm.get_task(task_id)
        assert len(await tm.list_tasks())==1
        if scenario=='retarget':
            assert task.status==TaskStatus.COMPLETED and len(controller.actions)==32
            assert not any(m.message_type==MessageType.TASK_FAILED for m in messages)
            assert any('Retargeted' in str(m.data) for m in messages)
            assert any(a[JOINT_ORDER[0]]==.2 for a in controller.actions)
            assert not policy.closed
        else:
            assert task.status==(TaskStatus.CANCELLED if scenario=='cancel' else TaskStatus.FAILED)
            assert result.data['status']==('cancelled' if scenario=='cancel' else 'failed')
            assert any(m.message_type==MessageType.TASK_FAILED for m in messages)
            assert not any(m.message_type==MessageType.TASK_COMPLETED for m in messages)
            assert controller.resets==['initial','ready'] and policy.closed
            stopped_count=len(controller.actions)
            with pytest.raises(Exception):await tools['reset_pose']._tool_func()
            await asyncio.sleep(.1)
            assert len(controller.actions)==stopped_count
        assert all(name!='MainThread' for name in controller.threads)
        await until(lambda:not controller.connected)
    finally:
        if agent.async_pick.active:agent.async_pick.stop('test cleanup')
        if execution is not None and not execution.done():execution.cancel()
        worker.cancel()
        with suppress(asyncio.CancelledError):await worker
        if execution is not None:
            with suppress(asyncio.CancelledError):await execution


@pytest.mark.asyncio
async def test_retarget_rejects_empty_missing_and_terminal_tasks(shm_env_and_server):
    client=shm_env_and_server['client'];tm=shm_env_and_server['tm'];broker=shm_env_and_server['broker']
    task_id=await tm.create_task('banana')
    await tm.update_task(task_id,TaskStatus.RUNNING)
    for target,instruction in [(task_id,'  '),('missing','apple')]:
        response=await client.call_tool('retarget_robot_instruction',{'task_id':target,'instruction':instruction})
        assert response.data['status']=='rejected'
    await tm.update_task(task_id,TaskStatus.COMPLETED)
    response=await client.call_tool('retarget_robot_instruction',{'task_id':task_id,'instruction':'apple'})
    assert response.data['status']=='rejected'
    assert not await broker.get_message_history(task_id=task_id,limit=10)
