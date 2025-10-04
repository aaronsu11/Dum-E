"""
Comprehensive tests for Dum-E main application (dum_e.py).

This file consolidates all tests for the main application components including:
- Configuration resolution and priority handling
- Process spawning (MCP server, Pipecat server, agent worker)
- Environment variable injection and inheritance
- End-to-end integration testing with shared memory backends
- Error handling and edge cases

Test Design Philosophy:
- Unit tests for individual functions with mocked dependencies
- Integration tests for multi-component workflows
- Environment isolation to prevent test interference
- Comprehensive coverage of configuration priority (args > config > env)
- Hardware-independent testing using mocks and shared memory
"""

import argparse
import asyncio
import os
import subprocess
import sys
from multiprocessing.managers import SharedMemoryManager
from unittest import mock

import pytest
from fastmcp import Client

import dum_e
from shared import BackendConfig


class TestBuildBackendConfig:
    """Test backend configuration building logic with priority resolution."""

    def test_build_backend_config_defaults(self):
        """Test building config with default values when no sources provided."""
        args = argparse.Namespace()
        cfg = {}

        # Clear environment to ensure we get true defaults
        with mock.patch.dict(os.environ, {}, clear=True):
            result = dum_e.build_backend_config(args, cfg)

        assert isinstance(result, BackendConfig)
        assert result.namespace == "default"
        assert result.aws_region is None
        assert result.mqtt_endpoint is None
        assert result.mqtt_topic_prefix is None
        assert result.dynamodb_table is None

    def test_build_backend_config_from_args(self):
        """Test building config from command line arguments."""
        args = argparse.Namespace(namespace="test_namespace")
        cfg = {}

        result = dum_e.build_backend_config(args, cfg)

        assert result.namespace == "test_namespace"

    def test_build_backend_config_from_config_file(self):
        """Test building config from config file."""
        args = argparse.Namespace()
        cfg = {
            "backend": {
                "namespace": "config_namespace",
                "aws_region": "us-west-2",
                "mqtt_endpoint": "test.mqtt.com",
                "mqtt_topic_prefix": "test/prefix",
                "dynamodb_table": "test_table",
            }
        }

        result = dum_e.build_backend_config(args, cfg)

        assert result.namespace == "config_namespace"
        assert result.aws_region == "us-west-2"
        assert result.mqtt_endpoint == "test.mqtt.com"
        assert result.mqtt_topic_prefix == "test/prefix"
        assert result.dynamodb_table == "test_table"

    def test_build_backend_config_from_env(self):
        """Test building config from environment variables."""
        args = argparse.Namespace()
        cfg = {}

        with mock.patch.dict(
            os.environ,
            {
                "DUME_NAMESPACE": "env_namespace",
                "AWS_REGION": "us-east-1",
                "MQTT_ENDPOINT": "env.mqtt.com",
                "MQTT_TOPIC_PREFIX": "env/prefix",
                "DYNAMODB_TABLE": "env_table",
            },
        ):
            result = dum_e.build_backend_config(args, cfg)

            assert result.namespace == "env_namespace"
            assert result.aws_region == "us-east-1"
            assert result.mqtt_endpoint == "env.mqtt.com"
            assert result.mqtt_topic_prefix == "env/prefix"
            assert result.dynamodb_table == "env_table"

    def test_build_backend_config_priority_order(self):
        """Test that args override config file, which overrides env."""
        args = argparse.Namespace(namespace="args_namespace")
        cfg = {"backend": {"namespace": "config_namespace"}}

        with mock.patch.dict(os.environ, {"DUME_NAMESPACE": "env_namespace"}):
            result = dum_e.build_backend_config(args, cfg)

            # Args should have highest priority
            assert result.namespace == "args_namespace"

    def test_build_backend_config_invalid_backend_type(self):
        """Test handling of invalid backend config type."""
        args = argparse.Namespace()
        cfg = {"backend": "not_a_dict"}

        # Clear environment to ensure we get true defaults
        with mock.patch.dict(os.environ, {}, clear=True):
            result = dum_e.build_backend_config(args, cfg)

        # Should fall back to defaults
        assert result.namespace == "default"

    @mock.patch("dum_e.logger")
    def test_build_backend_config_logging(self, mock_logger):
        """Test that effective config is logged."""
        args = argparse.Namespace()
        cfg = {"backend": {"namespace": "test_namespace"}}

        dum_e.build_backend_config(args, cfg)

        mock_logger.info.assert_called()
        log_call = mock_logger.info.call_args[0][0]
        assert "Effective BackendConfig:" in log_call


class TestSpawnMcpServer:
    """Test MCP server spawning logic."""

    @mock.patch("subprocess.Popen")
    def test_spawn_mcp_server_basic(self, mock_popen):
        """Test basic MCP server spawning."""
        config = BackendConfig(namespace="test")

        dum_e._spawn_mcp_server(config)

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        env = call_args[1]["env"]

        assert sys.executable in cmd
        assert "mcp_server.py" in cmd
        assert env["DUME_NAMESPACE"] == "test"

    @mock.patch("subprocess.Popen")
    def test_spawn_mcp_server_with_extra_env(self, mock_popen):
        """Test MCP server spawning with extra environment variables."""
        config = BackendConfig(namespace="test")
        extra_env = {"TEST_VAR": "test_value"}

        dum_e._spawn_mcp_server(config, extra_env)

        call_args = mock_popen.call_args
        env = call_args[1]["env"]

        assert env["DUME_NAMESPACE"] == "test"
        assert env["TEST_VAR"] == "test_value"

    @mock.patch("subprocess.Popen")
    def test_spawn_mcp_server_env_inheritance(self, mock_popen):
        """Test that MCP server inherits current environment."""
        config = BackendConfig(namespace="test")

        with mock.patch.dict(os.environ, {"EXISTING_VAR": "existing_value"}):
            dum_e._spawn_mcp_server(config)

        call_args = mock_popen.call_args
        env = call_args[1]["env"]

        assert env["EXISTING_VAR"] == "existing_value"
        assert env["DUME_NAMESPACE"] == "test"


class TestSpawnPipecatServer:
    """Test Pipecat server spawning logic."""

    @mock.patch("subprocess.Popen")
    def test_spawn_pipecat_server_basic(self, mock_popen):
        """Test basic Pipecat server spawning."""
        config = BackendConfig(namespace="test")

        dum_e._spawn_pipecat_server(config)

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        env = call_args[1]["env"]

        assert sys.executable in cmd
        assert "pipecat_server.py" in cmd
        assert env["DUME_NAMESPACE"] == "test"

    @mock.patch("subprocess.Popen")
    def test_spawn_pipecat_server_env_inheritance(self, mock_popen):
        """Test that Pipecat server inherits current environment."""
        config = BackendConfig(namespace="test")

        with mock.patch.dict(os.environ, {"EXISTING_VAR": "existing_value"}):
            dum_e._spawn_pipecat_server(config)

        call_args = mock_popen.call_args
        env = call_args[1]["env"]

        assert env["EXISTING_VAR"] == "existing_value"
        assert env["DUME_NAMESPACE"] == "test"


class TestSpawnAgentWorker:
    """Test agent worker spawning logic."""

    @mock.patch("subprocess.Popen")
    def test_spawn_agent_worker_mock_mode(self, mock_popen):
        """Test spawning agent worker in mock mode."""
        config = BackendConfig(namespace="test")
        agent_args = {"use_mock": True, "id": "mock_robot"}

        dum_e._spawn_agent_worker(config, agent_args)

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        env = call_args[1]["env"]

        assert sys.executable in cmd
        assert "-m" in cmd
        assert "tests.mocks" in cmd
        assert "--worker" in cmd
        assert "--id" in cmd
        assert "mock_robot" in cmd
        assert env["DUME_NAMESPACE"] == "test"

    @mock.patch("subprocess.Popen")
    def test_spawn_agent_worker_real_mode_with_config(self, mock_popen):
        """Test spawning real agent worker with config file."""
        config = BackendConfig(namespace="test")
        agent_args = {
            "use_mock": False,
            "port": "/dev/ttyACM0",
            "id": "real_robot",
            "wrist_cam_idx": 0,
            "front_cam_idx": 1,
            "policy_host": "localhost",
        }
        config_path = "/path/to/config.yaml"

        dum_e._spawn_agent_worker(config, agent_args, config_path)

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        env = call_args[1]["env"]

        assert sys.executable in cmd
        assert "-m" in cmd
        assert "embodiment.so_arm10x.agent" in cmd
        assert "--worker" in cmd
        assert "--namespace" in cmd
        assert "test" in cmd
        assert "--config" in cmd
        assert config_path in cmd
        assert env["DUME_NAMESPACE"] == "test"

    @mock.patch("subprocess.Popen")
    def test_spawn_agent_worker_real_mode_without_config(self, mock_popen):
        """Test spawning real agent worker without config file."""
        config = BackendConfig(namespace="test")
        agent_args = {
            "use_mock": False,
            "port": "/dev/ttyACM0",
            "id": "real_robot",
            "wrist_cam_idx": 0,
            "front_cam_idx": 1,
            "policy_host": "localhost",
        }

        dum_e._spawn_agent_worker(config, agent_args)

        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        assert "--port" in cmd
        assert "/dev/ttyACM0" in cmd
        assert "--id" in cmd
        assert "real_robot" in cmd
        assert "--wrist_cam_idx" in cmd
        assert "0" in cmd
        assert "--front_cam_idx" in cmd
        assert "1" in cmd
        assert "--policy_host" in cmd
        assert "localhost" in cmd

    @mock.patch("subprocess.Popen")
    def test_spawn_agent_worker_with_extra_env(self, mock_popen):
        """Test spawning agent worker with extra environment variables."""
        config = BackendConfig(namespace="test")
        agent_args = {"use_mock": True, "id": "mock_robot"}
        extra_env = {"TEST_VAR": "test_value"}

        dum_e._spawn_agent_worker(config, agent_args, extra_env=extra_env)

        call_args = mock_popen.call_args
        env = call_args[1]["env"]

        assert env["DUME_NAMESPACE"] == "test"
        assert env["TEST_VAR"] == "test_value"

    @mock.patch("subprocess.Popen")
    def test_spawn_agent_worker_default_values(self, mock_popen):
        """Test spawning agent worker with default values."""
        config = BackendConfig(namespace="test")
        agent_args = {"use_mock": False, "port": "/dev/ttyACM0"}

        dum_e._spawn_agent_worker(config, agent_args)

        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        # Check default values are used
        assert "--id" in cmd
        assert "my_awesome_follower_arm" in cmd  # default robot_id
        assert "--wrist_cam_idx" in cmd
        assert "0" in cmd  # default wrist_cam_idx
        assert "--front_cam_idx" in cmd
        assert "1" in cmd  # default front_cam_idx
        assert "--policy_host" in cmd
        assert "localhost" in cmd  # default policy_host

    def test_spawn_agent_worker_real_mode_missing_port(self):
        """Test that real mode requires port."""
        config = BackendConfig(namespace="test")
        agent_args = {"use_mock": False}  # Missing port

        with pytest.raises(KeyError, match="port"):
            dum_e._spawn_agent_worker(config, agent_args)

    @mock.patch("subprocess.Popen")
    def test_spawn_agent_worker_env_inheritance(self, mock_popen):
        """Test that agent worker inherits current environment."""
        config = BackendConfig(namespace="test")
        agent_args = {"use_mock": True, "id": "mock_robot"}

        with mock.patch.dict(os.environ, {"EXISTING_VAR": "existing_value"}):
            dum_e._spawn_agent_worker(config, agent_args)

        call_args = mock_popen.call_args
        env = call_args[1]["env"]

        assert env["EXISTING_VAR"] == "existing_value"
        assert env["DUME_NAMESPACE"] == "test"


class TestEnvironmentInjection:
    """Test environment variable injection for all spawned processes."""

    def test_spawn_mcp_server_env_injection(self):
        """Ensure MCP server is spawned with injected namespace and extras."""
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

    def test_spawn_agent_worker_mock_env_injection(self):
        """Ensure mock agent worker is spawned with correct CLI and env."""
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
            dum_e._spawn_agent_worker(cfg, agent_args, extra_env=extra_env)

        assert len(popen_calls) == 1
        cmd, env = popen_calls[0]
        assert cmd[:4] == [sys.executable, "-m", "tests.mocks", "--worker"]
        assert "--id" in cmd
        assert cmd[cmd.index("--id") + 1] == "mock_robot"
        assert env["DUME_NAMESPACE"] == "unit"
        assert env["DUME_IPC"] == "shm"

    def test_spawn_pipecat_server_env_injection(self):
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


class TestConfigurationResolution:
    """Test configuration resolution with different priority sources."""

    def test_build_backend_config_namespace_resolution(self, monkeypatch):
        """Test namespace resolution priority: CLI args > config file > environment."""
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


class TestEndToEndIntegration:
    """Test end-to-end integration with real shared memory backends."""

    @pytest.mark.asyncio
    async def test_dum_e_http_progress_integration(self, monkeypatch):
        """End-to-end HTTP integration with FastMCP client and mock worker.

        This test:
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
                    assert any(
                        e["message"] == "Task completed" for e in progress_events
                    )
            finally:
                if pipecat_proc:
                    pipecat_proc.terminate()
                if worker_proc:
                    worker_proc.terminate()
                server_proc.terminate()
