import argparse
import re
import os
import subprocess
import sys
from typing import Any, Dict, Optional
from multiprocessing.managers import SharedMemoryManager

from loguru import logger

from shared import BackendConfig
from shared.fleet_manager import get_shared_memory_fleet_manager_from_env
from utils import load_config_file


def build_backend_config(
    args: argparse.Namespace, cfg: Dict[str, Any]
) -> BackendConfig:
    backend_cfg = cfg.get("backend", {}) if isinstance(cfg.get("backend"), dict) else {}

    model = BackendConfig.model_validate(
        {
            "namespace": (
                getattr(args, "namespace", None)
                or backend_cfg.get("namespace")
                or os.getenv("DUME_NAMESPACE")
                or "default"
            ),
            "aws_region": backend_cfg.get("aws_region") or os.getenv("AWS_REGION"),
            "mqtt_endpoint": backend_cfg.get("mqtt_endpoint")
            or os.getenv("MQTT_ENDPOINT"),
            "mqtt_topic_prefix": backend_cfg.get("mqtt_topic_prefix")
            or os.getenv("MQTT_TOPIC_PREFIX"),
            "dynamodb_table": backend_cfg.get("dynamodb_table")
            or os.getenv("DYNAMODB_TABLE"),
        }
    )

    # Log effective config in a structured way
    data = model.model_dump()
    logger.info("Effective BackendConfig: {}", data)
    return model


def _spawn_mcp_server(
    config: BackendConfig, extra_env: Optional[Dict[str, str]] = None
):
    env = os.environ.copy()
    env.update(
        {
            "DUME_NAMESPACE": config.namespace,
        }
    )
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, "mcp_server.py"]
    logger.info(f"Starting MCP server with namespace={config.namespace}")
    return subprocess.Popen(cmd, env=env)


def _spawn_pipecat_server(
    config: BackendConfig, extra_env: Optional[Dict[str, str]] = None
):
    env = os.environ.copy()
    env.update(
        {
            "DUME_NAMESPACE": config.namespace,
        }
    )
    if extra_env:
        env.update(extra_env)

    cmd = [sys.executable, "pipecat_server.py"]
    logger.info("Starting Pipecat server with namespace={}", config.namespace)
    return subprocess.Popen(cmd, env=env)


def _spawn_agent_worker(
    config: BackendConfig,
    agent_args: Dict[str, Any],
    extra_env: Optional[Dict[str, str]] = None,
):
    env = os.environ.copy()
    env.update(
        {
            "DUME_NAMESPACE": config.namespace,
        }
    )
    if extra_env:
        env.update(extra_env)

    use_mock = agent_args.get("use_mock", False)
    if use_mock:
        cmd = [
            sys.executable,
            "-m",
            "tests.mocks",
            "--worker",
            "--robot_id",
            str(agent_args.get("id", "mock_robot")),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "embodiment.so_arm10x.agent",
            "--worker",
            "--namespace",
            config.namespace,
            "--port",
            str(agent_args["port"]),
            "--id",
            str(agent_args.get("id", "my_awesome_follower_arm")),
            "--wrist_cam_idx",
            str(agent_args.get("wrist_cam_idx", 0)),
            "--front_cam_idx",
            str(agent_args.get("front_cam_idx", 1)),
            "--policy_host",
            str(agent_args.get("policy_host", "localhost")),
            "--profile",
            str(agent_args.get("profile", "default")),
        ]

    logger.info(
        "Starting agent worker with namespace={} and args={}",
        config.namespace,
        agent_args,
    )
    return subprocess.Popen(cmd, env=env)


def main():
    parser = argparse.ArgumentParser(description="Dum-E Orchestrator")
    parser.add_argument(
        "--node",
        choices=["server", "agent", "both"],
        default="both",
        help="Which components to run on this host",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to JSON/YAML config file (optional)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=os.getenv("DUME_NAMESPACE", "default"),
        help="Namespace for shared memory backends or topic/table prefixes",
    )

    # Agent overrides (optional if provided in config)
    parser.add_argument("--port", type=str, help="Robot serial port")
    parser.add_argument("--id", type=str, help="Robot ID")
    parser.add_argument("--wrist_cam_idx", type=int)
    parser.add_argument("--front_cam_idx", type=int)
    parser.add_argument("--policy_host", type=str)
    parser.add_argument("--profile", type=str)
    parser.add_argument(
        "--use_mock",
        action="store_true",
        help="Use MockRobotAgent as the worker instead of real robot agent",
    )

    args = parser.parse_args()

    cfg = load_config_file(args.config)

    # Build backend config (transport-agnostic)
    backend_config = build_backend_config(args, cfg)

    # Agent config resolution (CLI overrides config file)
    agent_cfg = cfg.get("agent", {}) if isinstance(cfg.get("agent"), dict) else {}
    ctrl_cfg = (
        cfg.get("controller", {}) if isinstance(cfg.get("controller"), dict) else {}
    )
    agent_args = {
        "port": args.port or ctrl_cfg.get("robot_port"),
        "id": args.id or ctrl_cfg.get("robot_id", "my_awesome_follower_arm"),
        "wrist_cam_idx": args.wrist_cam_idx or ctrl_cfg.get("wrist_cam_idx", 0),
        "front_cam_idx": args.front_cam_idx or ctrl_cfg.get("front_cam_idx", 1),
        "policy_host": args.policy_host or ctrl_cfg.get("policy_host", "localhost"),
        "profile": args.profile or agent_cfg.get("profile", "default"),
        "use_mock": bool(args.use_mock or agent_cfg.get("use_mock", False)),
    }

    # Validate agent port if agent is selected
    if args.node in ("agent", "both") and not agent_args.get("use_mock"):
        if not agent_args["port"]:
            parser.error("--port is required for real agent (or set in config)")

    procs = []
    server_proc = None
    agent_proc = None
    pipecat_proc = None
    use_shm = os.getenv("DUME_IPC", "shm").lower() == "shm"

    if use_shm:
        # Manager context ensures SHM segments are unlinked after children exit
        with SharedMemoryManager() as smm:
            broker_capacity = int(os.getenv("DUME_BROKER_CAP", "32"))
            tasks_capacity = int(os.getenv("DUME_TASKS_CAP", "16"))
            fleet_capacity = int(os.getenv("DUME_FLEET_CAP", "32"))
            slot_size = int(os.getenv("DUME_SHM_SLOT", "4096"))
            broker_buf = smm.ShareableList([" " * slot_size] * broker_capacity)
            broker_meta = smm.ShareableList([0])
            tasks_buf = smm.ShareableList([" " * slot_size] * tasks_capacity)
            fleet_buf = smm.ShareableList([" " * slot_size] * fleet_capacity)

            # Avoid parent resource_tracker double-clean complaints; manager will unlink
            try:
                from multiprocessing import resource_tracker as _rt  # type: ignore

                for sl in [broker_buf, broker_meta, tasks_buf, fleet_buf]:
                    try:
                        n = getattr(sl.shm, "_name", sl.shm.name)
                        _rt.unregister(n, "shared_memory")
                    except Exception:
                        pass
            except Exception:
                pass

            env_common = {
                "DUME_IPC": "shm",
                "DUME_NAMESPACE": backend_config.namespace,
                "DUME_BROKER_BUF": broker_buf.shm.name,
                "DUME_BROKER_META": broker_meta.shm.name,
                "DUME_TASKS_BUF": tasks_buf.shm.name,
                "DUME_FLEET_BUF": fleet_buf.shm.name,
            }

            # Ensure the parent process can also access these SHM-backed services
            # (needed for default fleet registration below).
            try:
                os.environ.update(env_common)
            except Exception:
                pass

            if args.node in ("server", "both"):
                server_proc = _spawn_mcp_server(backend_config, env_common)
                procs.append(server_proc)

            if args.node in ("agent", "both"):
                agent_proc = _spawn_agent_worker(backend_config, agent_args, env_common)
                procs.append(agent_proc)

                # Best-effort: register default robot and enable
                try:
                    fm = get_shared_memory_fleet_manager_from_env()
                    if fm:
                        import asyncio as _asyncio

                        rid = str(agent_args.get("id", "my_awesome_follower_arm"))
                        # Convert rid to a name without special characters
                        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", rid)
                        _asyncio.run(fm.register_robot(robot_id=rid, name=clean_name))
                        _asyncio.run(fm.set_enabled(robot_id=rid, enabled=True))
                except Exception as e:
                    logger.warning("Fleet registration failed: {}", e)

            # Spawn Pipecat independently (no SHM dependency) but only after MCP and Agent
            if args.node == "both" and server_proc and agent_proc:
                pipecat_proc = _spawn_pipecat_server(backend_config)
                procs.append(pipecat_proc)

            # Wait and ensure children stop before manager exits
            try:
                for p in procs:
                    if p:
                        p.wait()
            except KeyboardInterrupt:
                for p in procs:
                    if p and p.poll() is None:
                        p.terminate()
            finally:
                for p in procs:
                    if p and p.poll() is None:
                        p.terminate()
        return
    else:
        raise NotImplementedError("Only shm is supported for now")


if __name__ == "__main__":
    main()
