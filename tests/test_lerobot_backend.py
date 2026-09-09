"""LeRobot policy backend + gRPC mock contract tests (BACK-05, BACK-07).

BACK-05 is the normalized action contract: a flat ``(16, 6)`` tensor off LeRobot's
async-inference wire becomes the SAME ``list[dict["<joint>.pos", float]]`` that
``groot-native`` produces from its modality dict — reindexed against the
handshake's own ``names`` list, never against a hardcoded joint order.

BACK-07 is the keyless conformance suite: every test here is KEYLESS, GPU-FREE
and HARDWARE-FREE — no policy server, no checkpoint, no SO101 arm, no serial
port. The only network use is 127.0.0.1 on an ephemeral port.

Every assertion is on the DECODED chunk, never on RPC success alone. That is not
style: ``GetActions`` can return ``services_pb2.Empty()`` from a method DECLARED
to return ``Actions`` (``policy_server.py:259-266``), which serializes to ``b''``
— so a SUCCESSFUL RPC carrying zero bytes is indistinguishable from a real reply
except by length. A test that asserted only "the RPC returned" would pass on it.
"""

import contextlib
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent

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
from policy.gr00t.service import MsgSerializer  # noqa: E402
from policy.lerobot import features  # noqa: E402
from policy.lerobot.session import LeRobotPolicySession  # noqa: E402

# The six joints in controller order. Imported rather than restated so this
# module cannot become a second source of truth for the ordering.
ROBOT_STATE_KEYS = list(features.ROBOT_STATE_KEYS)
CAMERA_KEYS = list(features.CAMERA_KEYS)


# --- Real-socket harness ------------------------------------------------------
#
# CONSCIOUS DUPLICATION, not an oversight: this is the THIRD copy of
# _free_port/_wait_for_port in the suite (tests/test_container_contract.py and
# tests/test_policy_backend.py hold the other two). There is no conftest.py in
# this repo today, and introducing one to share three six-line helpers would
# change how every existing test module resolves fixtures — a much larger blast
# radius than the duplication it removes. Recorded here so a later reader sees a
# decision rather than sloppiness.


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
    """Start the gRPC mock on an ephemeral loopback port; yield ``(port, server)``.

    Teardown is ``server.stop(grace=0)`` in a ``finally`` — the returned-handle
    equivalent of the ZMQ mock's ``"kill"`` endpoint (see ``start_mock``'s
    DELIBERATE DIVERGENCE note). A mock left running would wedge the session.
    """
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
    """A FLAT raw observation, exactly the shape ``IRobotController.get_observation()``
    produces: ``{"<joint>.pos": float}`` for six joints plus ``{cam: HxWx3 uint8}``.

    Deliberately NOT the nested ``{video, state, language}`` GR00T shape — that
    nesting belongs to the ZMQ wire; this wire's server does the packing itself
    via ``raw_observation_to_observation(...)`` plus the preprocessor.
    """
    obs: dict = {
        k: np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for k in CAMERA_KEYS
    }
    for k in ROBOT_STATE_KEYS:
        obs[k] = state
    return obs


def _session_observation(**kwargs) -> dict:
    """``_synthetic_observation`` plus the LeRobot language key.

    ``"task"`` is LeRobot's language key (``processor_groot.py:1547``); omitting
    it is not an error but silently substitutes ``"Perform the task."``, which is
    a silent degradation of policy quality. Session-level tests must supply it
    because there is no backend above them to do it.
    """
    obs = _synthetic_observation(**kwargs)
    obs["task"] = "pick up the banana"
    return obs


# --- BACK-05: the ordering authority ------------------------------------------


def test_state_names_ordering_is_the_controller_order():
    """The handshake's own ``names`` list IS the joint order, compared as a LIST.

    ORDERED comparison, never ``set(...)`` and never ``sorted(...)``: a
    permutation is precisely the failure mode this asserts against, and both of
    those would pass on one. ``build_dataset_frame`` builds the state vector by
    iterating this list (``lerobot/utils/feature_utils.py:131-132``), so a
    disagreement silently moves the arm plausibly to the wrong pose.
    """
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
    """``Actions(data=b"")`` -> a NAMED RuntimeError, never ``EOFError``.

    This reproduces upstream's REAL behaviour, not a hypothetical: ``GetActions``
    wraps its body in a blanket ``except`` and returns ``services_pb2.Empty()``
    from a method declared to return ``Actions``, which serializes to ``b''``. The
    client must refuse to unpickle zero bytes; letting it through produces
    ``EOFError: Ran out of input`` from deep inside the transport, naming nothing.
    """
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
    """A ``FAILED_PRECONDITION`` refusal surfaces the SERVER's message, in ONE attempt.

    Two claims, both load-bearing. (1) The server's own diagnosis reaches the
    operator verbatim — ``RETRYABLE_CODES`` deliberately excludes
    ``FAILED_PRECONDITION`` so a SAFE-01 refusal is not reworded into a generic
    "unreachable". (2) It is not retried: a retried handshake starts a SECOND
    concurrent multi-GB weight load rather than re-asking a finished question, so
    the elapsed-time bound is the observable proof that one attempt was made.
    """
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
    """An unbound port raises a descriptive error naming the address, bounded.

    ``wait_for_ready`` is never passed to a stub call; grpc-python defaults it to
    False, and that default is exactly what makes this true. Setting it True would
    convert an unreachable server into an indefinite park.
    """
    port = _free_port()
    with lerobot_session(port) as session:
        started = time.time()
        with pytest.raises(RuntimeError) as excinfo:
            session.connect(_remote_policy_config())
        elapsed = time.time() - started

    assert f"127.0.0.1:{port}" in str(excinfo.value)
    assert elapsed < 30.0, f"unreachable server took {elapsed:.2f}s — it parked"


def test_both_mocks_run_in_one_session_on_distinct_ephemeral_ports():
    """BACK-07 adjacency edge: the ZMQ mock and the gRPC mock coexist.

    Two mocks that collided on a port would make the parametrized contract suite
    pass or fail depending on test ORDER. Both round trips are exercised, then the
    gRPC server is stopped and the ZMQ one is proven still serving.
    """
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
