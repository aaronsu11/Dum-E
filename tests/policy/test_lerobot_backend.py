'LeRobot policy backend + gRPC mock contract tests (BACK-05, BACK-07).'

import contextlib
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
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]

# Make scripts/ importable so we can reuse the standalone mock servers.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from mock_grpc_policy_server import (  # noqa: E402
    _ACTION_DIM,
    _ACTION_HORIZON,
    MOCK_MODES,
    start_mock,
)
from mock_policy_server import serve_mock as serve_zmq_mock  # noqa: E402

from lerobot.async_inference.helpers import RemotePolicyConfig  # noqa: E402
from policy.factory import make_policy_backend  # noqa: E402
from policy.backends.isaac_groot.service import MsgSerializer  # noqa: E402
from policy.backends.lerobot import features  # noqa: E402
from policy.backends.lerobot.session import LeRobotPolicySession  # noqa: E402
from shared import IPolicyBackend  # noqa: E402

# The six joints in controller order. Imported rather than restated so this
# module cannot become a second source of truth for the ordering.
ROBOT_STATE_KEYS = list(features.ROBOT_STATE_KEYS)
CAMERA_KEYS = list(features.CAMERA_KEYS)


# CONSCIOUS DUPLICATION, not an oversight: this is the THIRD copy of
# _free_port/_wait_for_port in the suite (tests/policy/test_container_contract.py and


def _free_port() -> int:
    """Pick a currently-free loopback TCP port (never hard-code 8080)."""
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


@contextlib.contextmanager
def grpc_mock(mode: str = "ok"):
    'Start the gRPC mock on an ephemeral loopback port; yield ``(port, server)``.'
    port = _free_port()
    server = start_mock(port, "127.0.0.1", mode=mode)
    try:
        _wait_for_port(port)
        yield port, server
    finally:
        server.stop(grace=0)


@contextlib.contextmanager
def lerobot_session(port: int, host: str = "127.0.0.1"):
    """Yield a ``LeRobotPolicySession`` pointed at ``host:port``, closed on exit."""
    session = LeRobotPolicySession(f"{host}:{port}")
    try:
        yield session
    finally:
        session.close()


@contextlib.contextmanager
def lerobot_backend(port: int, **kwargs):
    'Build the ``lerobot`` backend THROUGH THE FACTORY; close it on exit.'
    with mock.patch.dict(os.environ, {"DUME_POLICY_BACKEND": "lerobot"}, clear=True):
        backend = make_policy_backend(
            host="127.0.0.1",
            port=port,
            camera_keys=CAMERA_KEYS,
            robot_state_keys=ROBOT_STATE_KEYS,
            **kwargs,
        )
    try:
        yield backend
    finally:
        backend.close()


def _remote_policy_config(actions_per_chunk: int = _ACTION_HORIZON) -> RemotePolicyConfig:
    """The handshake payload, built from the SAME features builder the backend uses."""
    return RemotePolicyConfig(
        policy_type="groot",
        pretrained_name_or_path="/checkpoints/model",
        lerobot_features=features.build_lerobot_features(),
        actions_per_chunk=actions_per_chunk,
        device="cpu",
        rename_map={},
    )


def _synthetic_observation(height: int = 64, width: int = 64, state: float = 0.0) -> dict:
    'A FLAT raw observation, exactly the shape ``IRobotController.get_observation()`` produces: ``{"<joint>.pos": float}`` for six joints plus ``{cam: HxWx3 uint8}``.'
    obs: dict = {
        k: np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for k in CAMERA_KEYS
    }
    for k in ROBOT_STATE_KEYS:
        obs[k] = state
    return obs


def _session_observation(**kwargs) -> dict:
    '``_synthetic_observation`` plus the LeRobot language key.'
    obs = _synthetic_observation(**kwargs)
    obs["task"] = "pick up the banana"
    return obs


# --- BACK-05: the ordering authority ------------------------------------------


def test_state_names_ordering_is_the_controller_order():
    "The handshake's own ``names`` list IS the joint order, compared as a LIST."
    names = features.state_names(features.build_lerobot_features())

    assert names == [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    # The literal above and the module's tuple must agree; if they ever diverge
    # this test names which one moved.
    assert names == ROBOT_STATE_KEYS
    assert len(names) == _ACTION_DIM


# --- BACK-07: the gRPC mock is the exact analogue of the ZMQ mock -------------


def test_start_mock_rejects_an_unknown_mode():
    """An unknown mode raises, naming the offending value AND the allowed set."""
    assert MOCK_MODES == ("ok", "empty", "refuse")
    with pytest.raises(ValueError) as excinfo:
        start_mock(_free_port(), "127.0.0.1", mode="nonsense")
    message = str(excinfo.value)
    assert "nonsense" in message
    for allowed in MOCK_MODES:
        assert allowed in message


def test_ok_mode_session_round_trip_returns_sixteen_six_dim_zero_actions():
    """One real loopback round trip yields 16 DECODED actions of shape (6,), zeros.

    The assertions are on the decoded chunk — length, per-action shape, and
    values — because a successful RPC proves nothing on this wire.
    """
    with grpc_mock("ok") as (port, server):
        with lerobot_session(port) as session:
            session.connect(_remote_policy_config())
            timed_actions = session.infer(_session_observation())

    assert len(timed_actions) == _ACTION_HORIZON
    for ta in timed_actions:
        action = ta.get_action()
        assert tuple(action.shape) == (_ACTION_DIM,)
        assert [float(v) for v in action.tolist()] == [0.0] * _ACTION_DIM

    # The mock recorded what the CLIENT actually sent, so the handshake and the
    # observation are asserted rather than assumed.
    assert server.dume_servicer.last_specs.actions_per_chunk == _ACTION_HORIZON
    assert set(ROBOT_STATE_KEYS) <= set(server.dume_servicer.last_observation.observation)


def test_stopped_mock_makes_ready_return_false():
    """``server.stop(grace=0)`` releases the port, and ``ready()`` then returns False.

    ``ready()`` is a reachability probe: it must report unreachable rather than
    raise, or a caller cannot poll it.
    """
    port = _free_port()
    server = start_mock(port, "127.0.0.1", mode="ok")
    try:
        _wait_for_port(port)
        with lerobot_session(port) as session:
            assert session.ready() is True
    finally:
        server.stop(grace=0)

    with lerobot_session(port) as session:
        assert session.ready() is False


def test_empty_actions_payload_raises_named_error_not_eoferror():
    '``Actions(data=b"")`` -> a NAMED RuntimeError, never ``EOFError``.'
    with grpc_mock("empty") as (port, _server):
        with lerobot_session(port) as session:
            session.connect(_remote_policy_config())
            with pytest.raises(RuntimeError) as excinfo:
                session.infer(_session_observation())

    message = str(excinfo.value)
    assert "action chunk" in message.lower()
    assert "zero-length" in message.lower()
    assert "Ran out of input" not in message
    assert not isinstance(excinfo.value, EOFError)


def test_refused_handshake_raises_without_retrying():
    "A ``FAILED_PRECONDITION`` refusal surfaces the SERVER's message, in ONE attempt."
    with grpc_mock("refuse") as (port, _server):
        with lerobot_session(port) as session:
            started = time.time()
            with pytest.raises(RuntimeError) as excinfo:
                session.connect(_remote_policy_config())
            elapsed = time.time() - started

    message = str(excinfo.value)
    assert "SAFE-01/2" in message
    assert "actions_per_chunk=40" in message
    assert "FAILED_PRECONDITION" in message
    assert elapsed < 5.0, f"handshake refusal took {elapsed:.2f}s — it was retried"


def test_unreachable_grpc_server_raises_with_address_and_does_not_hang():
    'An unbound port raises a descriptive error naming the address, bounded.'
    port = _free_port()
    with lerobot_session(port) as session:
        started = time.time()
        with pytest.raises(RuntimeError) as excinfo:
            session.connect(_remote_policy_config())
        elapsed = time.time() - started

    assert f"127.0.0.1:{port}" in str(excinfo.value)
    assert elapsed < 30.0, f"unreachable server took {elapsed:.2f}s — it parked"


def test_both_mocks_run_in_one_session_on_distinct_ephemeral_ports():
    'BACK-07 adjacency edge: the ZMQ mock and the gRPC mock coexist.'
    zmq_port = _free_port()
    grpc_port = _free_port()
    assert zmq_port != grpc_port

    zmq_thread = threading.Thread(
        target=serve_zmq_mock, kwargs={"port": zmq_port, "host": "127.0.0.1"}, daemon=True
    )
    zmq_thread.start()
    _wait_for_port(zmq_port)

    grpc_server = start_mock(grpc_port, "127.0.0.1", mode="ok")
    ctx = zmq.Context()
    try:
        _wait_for_port(grpc_port)

        # gRPC leg: a decoded 16-action chunk.
        with lerobot_session(grpc_port) as session:
            session.connect(_remote_policy_config())
            assert len(session.infer(_session_observation())) == _ACTION_HORIZON

        # ZMQ leg: the v1.0 modality contract.
        assert _zmq_get_action(ctx, zmq_port)["single_arm"].shape == (1, 16, 5)

        # Stopping the gRPC server must not touch the ZMQ one.
        grpc_server.stop(grace=0)
        with lerobot_session(grpc_port) as session:
            assert session.ready() is False
        assert _zmq_get_action(ctx, zmq_port)["gripper"].shape == (1, 16, 1)
    finally:
        grpc_server.stop(grace=0)
        _zmq_kill(ctx, zmq_port)
        ctx.term()
        zmq_thread.join(timeout=2.0)


# --- BACK-05: the normalized action contract, through the factory -------------


def test_lerobot_end_to_end_over_real_grpc_socket():
    'The whole slice: env selector -> factory -> IPolicyBackend -> real socket.'
    instruction = "Grab a banana and put it on the plate"

    with grpc_mock("ok") as (port, server):
        with lerobot_backend(port) as backend:
            # The factory returns the ABSTRACTION, not a privileged concrete type.
            assert isinstance(backend, IPolicyBackend)

            backend.set_lang_instruction(instruction)
            assert backend.language_instruction == instruction

            assert backend.ping() is True

            actions = backend.get_action(_synthetic_observation())

    assert actions, "policy returned an empty action list"
    assert len(actions) == _ACTION_HORIZON
    for step in actions:
        assert list(step) == ROBOT_STATE_KEYS
        assert all(isinstance(v, float) for v in step.values())

    # The instruction must ride LeRobot's own language key, or the server
    # silently substitutes "Perform the task." — a quality loss that reads
    # downstream as checkpoint drift.
    assert server.dume_servicer.last_observation.observation["task"] == instruction
    # And the handshake carried the DERIVED ordering, not a hand-written list.
    sent = server.dume_servicer.last_specs
    assert features.state_names(sent.lerobot_features) == ROBOT_STATE_KEYS
    assert sent.actions_per_chunk == _ACTION_HORIZON
    assert sent.rename_map == {}


def test_reset_re_readies_the_server_without_a_second_handshake():
    '``reset()`` must NOT trigger a second ``SendPolicyInstructions`` (CR-01).'
    with grpc_mock("ok") as (port, server):
        servicer = server.dume_servicer
        with lerobot_backend(port) as backend:
            backend.set_lang_instruction("pick up the banana")

            # Before the first handshake there is no per-client server state to
            # flush, so the probe is skipped and this backend stays lazily
            # connecting -- IPolicyBackend.session() calls reset() on entry.
            backend.reset()
            assert servicer.ready_calls == 0
            assert servicer.handshake_calls == 0

            backend.get_action(_synthetic_observation())
            assert servicer.handshake_calls == 1
            ready_after_handshake = servicer.ready_calls
            assert ready_after_handshake >= 1

            backend.reset()

            # The reset re-armed the server's per-client state ...
            assert servicer.ready_calls == ready_after_handshake + 1, (
                "reset() must send Ready — that is the only server-side per-client "
                "state a client can flush"
            )
            # ... and did NOT pay for a second weight load.
            assert servicer.handshake_calls == 1, (
                f"reset() sent {servicer.handshake_calls} handshakes; a re-handshake "
                "is one more multi-GB weight materialization on a GPU already "
                "holding one (CR-01)"
            )

            # And the session still works, so the saving is not bought with a
            # half-dead backend.
            actions = backend.get_action(_synthetic_observation())
            assert len(actions) == _ACTION_HORIZON
            assert servicer.handshake_calls == 1


def test_reset_raises_when_the_server_cannot_be_reached():
    'A reset that could not flush server state RAISES rather than returning.'
    port = _free_port()
    server = start_mock(port, "127.0.0.1", mode="ok")
    try:
        _wait_for_port(port)
        with lerobot_backend(port) as backend:
            backend.set_lang_instruction("pick up the banana")
            backend.get_action(_synthetic_observation())
            server.stop(grace=0)
            with pytest.raises(RuntimeError, match="unreachable"):
                backend.reset()
    finally:
        server.stop(grace=0)


def test_decoded_dims_zero_to_four_are_arm_and_dim_five_is_gripper():
    'Dim i of the flat action maps to joint i — asserted, never assumed.'
    vector = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    with grpc_mock("ok") as (port, server):
        server.dume_servicer.action_vector = vector
        with lerobot_backend(port) as backend:
            backend.set_lang_instruction("pick up the banana")
            actions = backend.get_action(_synthetic_observation())

    assert len(actions) == _ACTION_HORIZON
    for step in actions:
        for index, joint in enumerate(ROBOT_STATE_KEYS):
            assert step[joint] == pytest.approx(vector[index]), (joint, index, step)

    # Stated positionally as well, so the arm/gripper split is explicit rather
    # than implied by the loop above.
    first = actions[0]
    assert [first[j] for j in ROBOT_STATE_KEYS[:5]] == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])
    assert first["gripper.pos"] == pytest.approx(5.0)


def test_get_action_without_any_instruction_raises():
    'Fail closed on a missing instruction, BEFORE any gRPC call.'
    with lerobot_backend(_free_port()) as backend:
        with pytest.raises(ValueError) as excinfo:
            backend.get_action(_synthetic_observation())
        assert "instruction" in str(excinfo.value).lower()

        # An explicit empty string is just as unusable as no instruction.
        with pytest.raises(ValueError):
            backend.get_action(_synthetic_observation(), lang="")


def test_language_instruction_is_readonly_property():
    '``language_instruction`` is a read-only property backed by a field.'
    with lerobot_backend(_free_port()) as backend:
        assert isinstance(
            type(backend).language_instruction, property
        ), "language_instruction must be a property, not a plain attribute"
        assert backend.language_instruction is None

        backend.set_lang_instruction("pick up the banana")
        assert backend.language_instruction == "pick up the banana"

        with pytest.raises(AttributeError):
            backend.language_instruction = "assigned directly"


def test_close_is_idempotent_and_session_closes_on_exception():
    """``close()`` twice is a no-op, and ``session()`` closes on a raising body.

    A double close that raised would mask the original exception in a ``finally``,
    and an aborted episode that leaked the channel would carry it into the next.
    """
    with mock.patch.dict(os.environ, {"DUME_POLICY_BACKEND": "lerobot"}, clear=True):
        backend = make_policy_backend(host="127.0.0.1", port=_free_port())
        other = make_policy_backend(host="127.0.0.1", port=_free_port())

    backend.close()
    backend.close()  # second call is a no-op, never a raise

    boom = RuntimeError("episode aborted")
    with pytest.raises(RuntimeError, match="episode aborted"):
        with other.session() as scoped:
            assert scoped is other
            raise boom

    # close() ran in session()'s finally, so a further close is still a no-op.
    other.close()


def test_show_images_true_warns_rather_than_silently_ignoring():
    '``show_images=True`` is not wired here, and says so out loud.'
    captured: list[str] = []
    handler_id = logger.add(lambda message: captured.append(str(message)), level="WARNING")
    try:
        with lerobot_backend(_free_port(), show_images=True) as backend:
            assert backend is not None
    finally:
        logger.remove(handler_id)

    joined = "\n".join(captured)
    assert "show_images" in joined
    assert "lerobot" in joined.lower()

    # And the quiet path stays quiet.
    quiet: list[str] = []
    handler_id = logger.add(lambda message: quiet.append(str(message)), level="WARNING")
    try:
        with lerobot_backend(_free_port()) as backend:
            assert backend is not None
    finally:
        logger.remove(handler_id)
    assert "show_images" not in "\n".join(quiet)


def _zmq_get_action(ctx, port: int) -> dict:
    """One get_action round trip against the ZMQ mock; returns the action_chunk."""
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 5000)
    sock.setsockopt(zmq.SNDTIMEO, 5000)
    sock.connect(f"tcp://127.0.0.1:{port}")
    try:
        sock.send(MsgSerializer.to_bytes({"endpoint": "get_action", "data": {"state": {}}}))
        action_chunk, _info = MsgSerializer.from_bytes(sock.recv())
        return action_chunk
    finally:
        sock.close(linger=0)


def _zmq_kill(ctx, port: int) -> None:
    """Stop the ZMQ mock's serve loop via the contract's ``"kill"`` endpoint."""
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
