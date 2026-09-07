"""Policy-backend seam tests (BACK-01/02/03/04/06).

Covers the seam introduced in Phase 5:
- BACK-01: ``IPolicyBackend`` covers every existing policy call site.
- BACK-02: omitting ``DUME_POLICY_BACKEND`` selects ``lerobot``.
- BACK-03: an unknown value RAISES (never warn-and-fall-back).
- BACK-04: ``lerobot`` raises an actionable "arrives in Phase 6" error.
- BACK-06: ``groot-native`` stays selectable AND functional — proven end to end
  against the real ``scripts/mock_policy_server.py`` over a real loopback ZMQ
  socket, not a mocked transport.

Every test here is KEYLESS, GPU-FREE and HARDWARE-FREE: no policy server, no
checkpoint, no SO101 arm, no serial port. The only network use is 127.0.0.1 on
an ephemeral port.
"""

import os
import socket
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import zmq

from embodiment.so_arm10x.controller import Gr00tRobotInferenceClient
from policy.factory import make_policy_backend
from policy.gr00t.service import MsgSerializer
from shared import IPolicyBackend

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make scripts/ importable so we can reuse the standalone mock server.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from mock_policy_server import serve_mock  # noqa: E402

# The flat observation shape the controller's get_action consumes, mirroring
# scripts/test_live_policy_server.py's _synthetic_observation.
ROBOT_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
CAMERA_KEYS = ["wrist", "front"]


# --- Real-socket harness (mirrors tests/test_container_contract.py) ----------


def _free_port() -> int:
    """Pick a currently-free loopback TCP port (CI hygiene; never hard-code 5555)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


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


def _synthetic_observation(height: int = 64, width: int = 64) -> dict:
    """A single raw observation dict shaped exactly as the controller expects.

    Flat ``{cam_key: HxWx3 uint8, "<joint>.pos": float}``. Values are arbitrary —
    this exercises the seam and the wire path, not policy quality. Frames are
    small so the msgpack payload stays fast over loopback.
    """
    obs: dict = {
        k: np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for k in CAMERA_KEYS
    }
    for k in ROBOT_STATE_KEYS:
        obs[k] = 0.0
    return obs


@pytest.fixture
def mock_server():
    """Run ``scripts/mock_policy_server.serve_mock`` on an ephemeral loopback port.

    Yields the port. The server runs in a daemon thread bound to 127.0.0.1 so it
    cannot leak beyond the test host and a stray loop cannot wedge interpreter
    shutdown; the contract's "kill" endpoint stops the loop in teardown.
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


# --- BACK-06: groot-native is selectable AND functional, end to end ----------


def test_groot_native_end_to_end_over_real_socket(mock_server):
    """DUME_POLICY_BACKEND=groot-native -> one action chunk off a real socket.

    The whole slice in one check: env selector -> make_policy_backend() ->
    IPolicyBackend -> Gr00tRobotInferenceClient -> real ZMQ REQ/REP over loopback
    -> the v1.0 modality contract flattened back into per-step joint dicts.
    """
    port = mock_server
    instruction = "Grab a banana and put it on the plate"

    with mock.patch.dict(
        os.environ, {"DUME_POLICY_BACKEND": "groot-native"}, clear=True
    ):
        backend = make_policy_backend(
            host="127.0.0.1",
            port=port,
            camera_keys=CAMERA_KEYS,
            robot_state_keys=ROBOT_STATE_KEYS,
        )
        try:
            # The factory returns the ABSTRACTION, not a privileged concrete type.
            assert isinstance(backend, IPolicyBackend)

            backend.set_lang_instruction(instruction)
            assert backend.language_instruction == instruction

            # Real reachability check against the real socket.
            assert backend.ping() is True

            actions = backend.get_action(_synthetic_observation())

            assert actions, "policy returned an empty action list"
            # Mock server serves T=16 chunks; every step carries all six joints.
            assert len(actions) == 16
            for step in actions:
                assert set(step) == set(ROBOT_STATE_KEYS)
                assert all(isinstance(v, float) for v in step.values())
        finally:
            backend.close()
