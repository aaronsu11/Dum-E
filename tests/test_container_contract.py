"""CI-runnable container contract probe + stack-isolation guard (Phase 4).

Every test here is CI-runnable with NO live policy server, NO GPU, and NO SO101
hardware. Unlike ``tests/test_gr00t_service.py`` (which mocks the ZMQ socket with
a ``MagicMock``), this module stands up the real ``scripts/mock_policy_server.py``
``zmq.REP`` server on an EPHEMERAL loopback port and connects a *real*
``ExternalRobotInferenceClient`` over actual TCP — proving the msgpack/numpy
bytes survive a real socket, i.e. the container boundary.

Surfaces covered:
- DOCK-05: the :5555 ``MsgSerializer`` wire contract is preserved across a real
  socket (get_action 2-tuple with single_arm (1,16,5) / gripper (1,16,1) float32;
  ping non-error; {"error": ...} reply -> RuntimeError).
- DOCK-03: the server GPU stack (torch/flash-attn/CUDA/tensorrt/onnxruntime-gpu)
  never leaks into the client ``pyproject.toml``, and ``requires-python`` stays
  ``>=3.12`` (the client is Py3.12; the server is Py3.10 inside the container).

No live policy server, no GPU, no hardware is required; the suite completes fast.
"""

import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
import zmq

from policy.gr00t.service import (
    ExternalRobotInferenceClient,
    MsgSerializer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make scripts/ importable so we can reuse the standalone mock server.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from mock_policy_server import serve_mock  # noqa: E402


# --- Real-socket harness ----------------------------------------------------


def _free_port() -> int:
    """Pick a currently-free loopback TCP port (CI hygiene; never hard-code 5555)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _serve_error(port: int, message: str, host: str = "127.0.0.1") -> None:
    """A one-shot zmq.REP server that always replies {"error": message}.

    Used to prove the client raises RuntimeError on a server-side error reply
    over a real socket. Serves a single request then tears down (LINGER=0).
    """
    context = zmq.Context()
    sock = context.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://{host}:{port}")
    try:
        sock.recv()
        sock.send(MsgSerializer.to_bytes({"error": message}))
    finally:
        sock.close(linger=0)
        context.term()


def _wait_for_port(port: int, host: str = "127.0.0.1", timeout: float = 5.0) -> None:
    """Block until something accepts a TCP connection on host:port (server ready)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.02)
    raise TimeoutError(f"mock server never came up on {host}:{port}")


@pytest.fixture
def mock_server():
    """Run scripts/mock_policy_server.serve_mock on an ephemeral loopback port.

    Yields the port. The server runs in a daemon thread bound to 127.0.0.1 so it
    cannot leak beyond the test host; the thread is a daemon so a stray loop
    cannot wedge interpreter shutdown. A "kill" endpoint cleanly stops the loop.
    """
    port = _free_port()
    thread = threading.Thread(
        target=serve_mock, kwargs={"port": port, "host": "127.0.0.1"}, daemon=True
    )
    thread.start()
    _wait_for_port(port)
    try:
        yield port
    finally:
        # Cleanly stop the serve loop via the contract's "kill" endpoint.
        ctx = zmq.Context()
        killer = ctx.socket(zmq.REQ)
        killer.setsockopt(zmq.LINGER, 0)
        killer.setsockopt(zmq.RCVTIMEO, 1000)
        killer.setsockopt(zmq.SNDTIMEO, 1000)
        killer.connect(f"tcp://127.0.0.1:{port}")
        try:
            killer.send(MsgSerializer.to_bytes({"endpoint": "kill"}))
            killer.recv()
        except zmq.error.ZMQError:
            pass
        finally:
            killer.close(linger=0)
            ctx.term()
        thread.join(timeout=2.0)


def _client(port: int) -> ExternalRobotInferenceClient:
    """A real client (real ZMQ REQ socket) pointed at the mock server."""
    return ExternalRobotInferenceClient(
        host="127.0.0.1", port=port, timeout_ms=5000
    )


# --- DOCK-05: wire contract survives a real socket --------------------------


def test_real_socket_get_action_returns_action_info_tuple(mock_server):
    """A real client get_action over real TCP returns the (action_chunk, info) tuple.

    Proves the msgpack/numpy bytes (single_arm (1,16,5) / gripper (1,16,1) f32)
    survive the actual socket round trip — the container boundary (DOCK-05).
    """
    client = _client(mock_server)
    try:
        result = client.get_action({"state": {}})
        assert isinstance(result, tuple) and len(result) == 2
        action_chunk, info = result
        assert set(action_chunk.keys()) == {"single_arm", "gripper"}
        assert action_chunk["single_arm"].shape == (1, 16, 5)
        assert action_chunk["gripper"].shape == (1, 16, 1)
        assert action_chunk["single_arm"].dtype.name == "float32"
        assert action_chunk["gripper"].dtype.name == "float32"
        assert isinstance(info, dict)
    finally:
        client.socket.close(linger=0)
        client.context.term()


def test_real_socket_ping_returns_truthy(mock_server):
    """ping() over a real socket returns a truthy/non-error result."""
    client = _client(mock_server)
    try:
        assert client.ping() is True
    finally:
        client.socket.close(linger=0)
        client.context.term()


def test_real_socket_error_reply_raises_runtimeerror():
    """A server replying {"error": "boom"} over a real socket -> RuntimeError("boom")."""
    port = _free_port()
    thread = threading.Thread(
        target=_serve_error, kwargs={"port": port, "message": "boom"}, daemon=True
    )
    thread.start()
    _wait_for_port(port)
    client = _client(port)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            client.call_endpoint(
                "get_action", {"observation": {}, "options": None}
            )
    finally:
        client.socket.close(linger=0)
        client.context.term()
        thread.join(timeout=2.0)


# --- DOCK-03: stack isolation (client pyproject.toml stays clean) -----------


def test_client_pyproject_has_no_server_gpu_stack():
    """The server GPU stack must never leak into the client pyproject.toml (DOCK-03)."""
    forbidden = {
        "torch",
        "flash-attn",
        "flash_attn",
        "nvidia-cuda",
        "tensorrt",
        "onnxruntime-gpu",
    }
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    leaked = sorted(f for f in forbidden if f in text)
    assert not leaked, f"server GPU stack leaked into client pyproject.toml: {leaked}"


def test_client_requires_python_stays_312():
    """The client interpreter floor stays >=3.12 (server is Py3.10 in-container)."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12"' in text
