'Policy-backend seam tests.'

import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import zmq

from policy.backends.isaac_groot.client import Gr00tRobotInferenceClient
from policy.factory import (
    DEFAULT_POLICY_BACKEND,
    POLICY_BACKENDS,
    make_policy_backend,
)
from policy.backends.isaac_groot.service import MsgSerializer
from shared import IPolicyBackend

REPO_ROOT = Path(__file__).resolve().parents[2]

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


# --- Real-socket harness (mirrors tests/policy/test_container_contract.py) ----------


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
    'A single raw observation dict shaped exactly as the controller expects.'
    obs: dict = {
        k: np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for k in CAMERA_KEYS
    }
    for k in ROBOT_STATE_KEYS:
        obs[k] = 0.0
    return obs


@pytest.fixture
def mock_server():
    'Run ``scripts/mock_policy_server.serve_mock`` on an ephemeral loopback port.'
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


# --- groot-native is selectable AND functional, end to end -------------------


def test_groot_native_end_to_end_over_real_socket(mock_server):
    'DUME_POLICY_BACKEND=groot-native -> one action chunk off a real socket.'
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


# --- the selector is FAIL-CLOSED ---------------------------------------------


def test_unknown_backend_raises_with_value_and_allowlist():
    'An unknown value RAISES; it never warns and falls back.'
    with mock.patch.dict(
        os.environ, {"DUME_POLICY_BACKEND": "nonsense"}, clear=True
    ):
        with pytest.raises(ValueError) as excinfo:
            make_policy_backend(host="127.0.0.1")

    message = str(excinfo.value)
    assert "nonsense" in message
    for allowed in POLICY_BACKENDS:
        assert allowed in message


def test_default_backend_matches_launcher_and_example():
    "Unconfigured factory and launcher select the same native backend."
    import yaml
    assert DEFAULT_POLICY_BACKEND == "groot-native"
    config = yaml.safe_load((Path(__file__).parents[2] / "config.example.yaml").read_text())
    assert config["controller"]["policy_backend"] == DEFAULT_POLICY_BACKEND

    with mock.patch.dict(os.environ, {}, clear=True):
        assert os.getenv("DUME_POLICY_BACKEND") is None
        backend = make_policy_backend(host="127.0.0.1", port=_free_port())

    try:
        assert type(backend).__name__ == "Gr00tRobotInferenceClient"
        assert isinstance(backend, IPolicyBackend)
    finally:
        backend.close()


def test_lerobot_backend_is_constructible_and_is_an_ipolicybackend():
    "'lerobot' returns a backend, and CONSTRUCTION does not connect."
    port = _free_port()
    with mock.patch.dict(
        os.environ, {"DUME_POLICY_BACKEND": "lerobot"}, clear=True
    ):
        backend = make_policy_backend(host="127.0.0.1", port=port)

    try:
        assert isinstance(backend, IPolicyBackend)
        assert backend.ping() is False
    finally:
        backend.close()


def test_factory_module_import_pulls_no_torch_or_lerobot():
    'Lazy-import discipline: importing the factory stays torch-free.'
    probe = (
        "import sys; import policy.factory; "
        "forbidden = sorted(m for m in sys.modules "
        "if m == 'torch' or m.startswith('torch.') "
        "or m == 'grpc' or m.startswith('grpc.') "
        "or m == 'lerobot' or m.startswith('lerobot.')); "
        "print(','.join(forbidden))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    leaked = result.stdout.strip()
    assert leaked == "", f"policy.factory import leaked heavy modules: {leaked}"


def test_factory_never_builds_an_import_from_the_env_value():
    """ASVS V5 / T-05-01: the allowlist plus explicit branches is the WHOLE
    validation surface. No eval, no importlib path built from the raw env string,
    no getattr keyed by it — those would turn a config string into code
    selection."""
    source = (REPO_ROOT / "policy" / "factory.py").read_text(encoding="utf-8")
    assert "eval(" not in source
    assert "importlib" not in source
    assert "__import__" not in source
    # The allowlist is a literal tuple, not something derived at runtime.
    assert POLICY_BACKENDS == ("lerobot", "groot-native", "galaxea", "pi05-so101")


# --- IPolicyBackend edge cases: adjacency / empty / ordering -----------------


def test_language_instruction_is_readonly_property():
    """Adjacency edge: two names that are exactly equal on the ABC do not
    collide. `language_instruction` is a read-only @property backed by a field,
    so skills.py's parenthesis-free READ works while assignment is rejected —
    mutation goes through set_lang_instruction, which a backend can validate."""
    client = Gr00tRobotInferenceClient(host="127.0.0.1", port=_free_port())
    try:
        assert isinstance(
            type(client).language_instruction, property
        ), "language_instruction must be a property, not a plain attribute"
        assert client.language_instruction is None

        client.set_lang_instruction("pick up the banana")
        assert client.language_instruction == "pick up the banana"

        with pytest.raises(AttributeError):
            client.language_instruction = "assigned directly"
    finally:
        client.close()


def test_get_action_without_any_instruction_raises(mock_server):
    'Empty edge: no argument instruction AND no stored one -> raise.'
    client = Gr00tRobotInferenceClient(host="127.0.0.1", port=mock_server)
    try:
        with pytest.raises(ValueError) as excinfo:
            client.get_action(_synthetic_observation())
        assert "instruction" in str(excinfo.value).lower()

        # An explicit empty-string argument is just as unusable as no instruction.
        with pytest.raises(ValueError):
            client.get_action(_synthetic_observation(), lang="")

        # With a stored instruction the guard does NOT fire, and the value rides
        # the pinned key (double-wrapped by the T=1 then B=1 dim expansion).
        client.set_lang_instruction("pick up the banana")
        captured: dict = {}
        real_get_action = client.policy.get_action

        def _spy(observation, options=None):
            captured["observation"] = observation
            return real_get_action(observation, options)

        with mock.patch.object(client.policy, "get_action", _spy):
            actions = client.get_action(_synthetic_observation())

        assert actions
        assert captured["observation"]["language"][
            "annotation.human.task_description"
        ] == [["pick up the banana"]]
    finally:
        client.close()


def test_close_is_idempotent_and_session_closes_on_exception():
    """Ordering edge: close() twice must not raise, and session() must
    close even when its body raises — otherwise a double close in a `finally`
    would mask the original exception, and an aborted episode would leak the
    socket into the next one."""
    client = Gr00tRobotInferenceClient(host="127.0.0.1", port=_free_port())
    client.close()
    client.close()  # second call is a no-op, never a raise

    other = Gr00tRobotInferenceClient(host="127.0.0.1", port=_free_port())
    boom = RuntimeError("episode aborted")
    with pytest.raises(RuntimeError, match="episode aborted"):
        with other.session() as backend:
            assert backend is other
            raise boom

    # close() ran in session()'s finally, so a further close is still a no-op.
    assert other._closed is True
    other.close()
