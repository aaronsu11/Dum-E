from unittest import mock

import pytest
import sys
import asyncio
from multiprocessing.managers import SharedMemoryManager

import dum_e
from fastmcp import Client
from shared import BackendConfig


def test_spawn_mcp_server_env_injection():
    """Ensure MCP server is spawned with injected namespace and extras.

    Stubs subprocess.Popen and inspects the command/env passed by
    dum_e._spawn_mcp_server.
    """
    popen_calls = []

    class DummyPopen:
        def __init__(self, cmd, env=None):
            popen_calls.append((cmd, env))

        def wait(self):
            return 0

    with mock.patch("subprocess.Popen", DummyPopen):
        cfg = BackendConfig(namespace="unit")
        extra_env = {"DUME_IPC": "shm", "FOO": "BAR"}
        dum_e._spawn_mcp_server(cfg, extra_env)

    assert len(popen_calls) == 1
    cmd, env = popen_calls[0]
    assert cmd[0] == sys.executable and cmd[1] == "mcp_server.py"
    assert env["DUME_NAMESPACE"] == "unit"
    assert env["DUME_IPC"] == "shm"
    assert env["FOO"] == "BAR"


def test_spawn_agent_worker_mock_env_injection():
    """Ensure mock agent worker is spawned with correct CLI and env.

    Validates that dum_e chooses tests.mocks worker path when
    use_mock=True and forwards namespace and shm indicator to env.
    """
    popen_calls = []

    class DummyPopen:
        def __init__(self, cmd, env=None):
            popen_calls.append((cmd, env))

        def wait(self):
            return 0

    with mock.patch("subprocess.Popen", DummyPopen):
        cfg = BackendConfig(namespace="unit")
        agent_args = {"use_mock": True, "id": "mock_robot"}
        extra_env = {"DUME_IPC": "shm"}
        dum_e._spawn_agent_worker(cfg, agent_args, extra_env)

    assert len(popen_calls) == 1
    cmd, env = popen_calls[0]
    assert cmd[:4] == [sys.executable, "-m", "tests.mocks", "--worker"]
    assert "--robot_id" in cmd
    assert cmd[cmd.index("--robot_id") + 1] == "mock_robot"
    assert env["DUME_NAMESPACE"] == "unit"
    assert env["DUME_IPC"] == "shm"


def test_spawn_pipecat_server_env_injection():
    """Ensure Pipecat server is spawned with namespace and extra env."""
    popen_calls = []

    class DummyPopen:
        def __init__(self, cmd, env=None):
            popen_calls.append((cmd, env))

        def wait(self):
            return 0

    with mock.patch("subprocess.Popen", DummyPopen):
        cfg = BackendConfig(namespace="unit")
        extra_env = {"FOO": "BAR"}
        dum_e._spawn_pipecat_server(cfg, extra_env)

    assert len(popen_calls) == 1
    cmd, env = popen_calls[0]
    assert cmd[0] == sys.executable and cmd[1] == "pipecat_server.py"
    assert env["DUME_NAMESPACE"] == "unit"
    assert env["FOO"] == "BAR"


def test_build_backend_config_namespace_resolution(monkeypatch):
    monkeypatch.setenv("DUME_NAMESPACE", "env")

    class Args:
        namespace = None

    cfg = {}
    bc = dum_e.build_backend_config(Args, cfg)
    assert bc.namespace == "env"

    class Args2:
        namespace = "cli"

    bc2 = dum_e.build_backend_config(Args2, cfg)
    assert bc2.namespace == "cli"


@pytest.mark.asyncio
async def test_dum_e_http_progress_integration(monkeypatch):
    """End-to-end HTTP integration with FastMCP client and mock worker.

    - Spawns MCP server (HTTP) and a mock agent worker using dum_e helpers
    - Connects a FastMCP Client to the HTTP endpoint and calls execute_robot_instruction
    - Captures progress events via client's progress_handler and validates streaming
    - Terminates all processes cleanly after assertions
    """

    # Build minimal args namespace for config
    class Args:
        node = "both"
        config = None
        namespace = "test"
        port = "/dev/null"  # won't be used by spawned process
        id = None
        wrist_cam_idx = None
        front_cam_idx = None
        policy_host = None
        profile = None
        use_mock = True

    cfg = {}
    backend_cfg = dum_e.build_backend_config(Args, cfg)

    # Force SHM path
    monkeypatch.setenv("DUME_IPC", "shm")
    monkeypatch.setenv("DUME_NAMESPACE", "test")

    # Create ephemeral SHM segments and inject into env so the server defaults to SHM
    with SharedMemoryManager() as smm:
        slot = 4096
        broker_buf = smm.ShareableList([" " * slot] * 32)
        broker_meta = smm.ShareableList([0])
        tasks_buf = smm.ShareableList([" " * slot] * 16)
        fleet_buf = smm.ShareableList([" " * slot] * 32)

        shm_env = {
            "DUME_IPC": "shm",
            "DUME_NAMESPACE": "test",
            "DUME_BROKER_BUF": broker_buf.shm.name,
            "DUME_BROKER_META": broker_meta.shm.name,
            "DUME_TASKS_BUF": tasks_buf.shm.name,
            "DUME_FLEET_BUF": fleet_buf.shm.name,
        }

        # Also export to real environment so the FastMCP client-spawned server can read them
        for k, v in shm_env.items():
            monkeypatch.setenv(k, v)

        # Start the actual HTTP MCP server process (no stdio) with SHM env using dum_e helpers
        # Use alternate port to avoid collisions across concurrent tests
        shm_env_with_port = dict(shm_env)
        shm_env_with_port["DUME_MCP_PORT"] = "8010"
        server_proc = dum_e._spawn_mcp_server(backend_cfg, shm_env_with_port)
        # Start a mock agent worker that consumes TASK_STARTED and completes the task
        worker_proc = dum_e._spawn_agent_worker(
            backend_cfg,
            {"use_mock": True, "id": "mock_robot"},
            shm_env,
        )
        # Start the independent Pipecat server (no SHM dependency)
        pipecat_proc = dum_e._spawn_pipecat_server(backend_cfg)
        try:
            # Wait for port to be open
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

            await _wait_for_port("127.0.0.1", 8010, timeout=5.0)

            # Progress handler to capture MCP context progress notifications
            progress_events = []

            async def on_progress(progress, total, message):
                progress_events.append(
                    {
                        "progress": progress,
                        "total": total,
                        "message": message,
                    }
                )

            # Connect via HTTP transport URL (not stdio) with init timeout
            client = Client(
                "http://127.0.0.1:8010/mcp",
                init_timeout=3,
                timeout=5,
                progress_handler=on_progress,
            )
            async with client:
                await client.ping()
                tools = await client.list_tools()
                assert any(t.name == "execute_robot_instruction" for t in tools)
                result = await client.call_tool(
                    "execute_robot_instruction",
                    {"instruction": "pick banana", "robot_id": "mock_robot"},
                )
                assert isinstance(result.data, dict)
                assert "status" in result.data

                # Validate that progress events from the agent were propagated
                assert len(progress_events) > 0
                # Expect at least one assistant message forwarded (e.g., 'step 1')
                assert any(
                    isinstance(e.get("message"), str)
                    and e["message"].startswith("step ")
                    for e in progress_events
                )
                # We expect a completion message
                assert any(e["message"] == "Task completed" for e in progress_events)
        finally:
            if pipecat_proc:
                pipecat_proc.terminate()
            if worker_proc:
                worker_proc.terminate()
            server_proc.terminate()
