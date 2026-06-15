"""Mocked-wiring tests for the GR00T policy-client transport.

This file pins the ``policy/gr00t/service.py`` :5555 wire contract and the
ZMQ socket-lifecycle / bounded-retry hardening (Phase 3, GR-01/GR-03/GR-06).

Like ``tests/test_pipecat_server.py``, every test here is CI-runnable with NO
live policy server, NO ZMQ network I/O, and NO SO101 hardware. The ZMQ socket is
always a ``mocker.MagicMock``; clients are built via ``__new__`` so ``__init__``
never opens a real socket. The msgpack/numpy round-trip uses the real serializer
(pure in-process bytes). Source introspection (``inspect.getsource``) is used for
the one assertion that is awkward to exercise dynamically (the ``allow_pickle``
security flag on the decode path) and for the repo-wide stale-import gate.

Surfaces covered:
- msgpack ndarray round-trip (shape + float32 dtype preserved)
- ``allow_pickle=False`` on the decode path (security, T-03-SC)
- ``get_action`` request shape ``{"observation", "options": None}``
- ``(action_chunk, info)`` tuple reply
- ``b"ERROR"`` sentinel raises RuntimeError
- ``_init_socket`` closes old socket (linger=0) + sets RCVTIMEO/SNDTIMEO/LINGER
- bounded-retry-then-raise (``zmq.error.Again`` -> RuntimeError "unreachable")
- defensive validation (missing modality key -> RuntimeError, not KeyError)
- no stale ``BaseInferenceServer``/``TorchSerializer`` imports anywhere in
  ``policy/``, ``embodiment/``, ``shared/``
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import zmq

from policy.gr00t.service import (
    BaseInferenceClient,
    ExternalRobotInferenceClient,
    MsgSerializer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Helpers ----------------------------------------------------------------


def _make_client(mocker, *, host="localhost", port=5555, timeout_ms=15000,
                 api_token=None):
    """Build an ExternalRobotInferenceClient with a mocked socket/context.

    Uses ``__new__`` so the real ``__init__`` (which opens a live ZMQ socket) is
    never run. Returns ``(client, sock)`` where ``sock`` is the mocked socket.
    """
    client = ExternalRobotInferenceClient.__new__(ExternalRobotInferenceClient)
    client.host = host
    client.port = port
    client.timeout_ms = timeout_ms
    client.api_token = api_token
    client.context = mocker.MagicMock()
    sock = mocker.MagicMock()
    client.socket = sock
    return client, sock


# --- Serializer / wire contract --------------------------------------------


def test_msgserializer_ndarray_roundtrip():
    """An ndarray survives to_bytes/from_bytes with shape AND dtype intact."""
    obs = {"state": {"single_arm": np.zeros((1, 1, 5), dtype=np.float32)}}
    out = MsgSerializer.from_bytes(MsgSerializer.to_bytes(obs))
    np.testing.assert_array_equal(
        out["state"]["single_arm"], obs["state"]["single_arm"]
    )
    assert out["state"]["single_arm"].dtype == np.float32
    assert out["state"]["single_arm"].shape == (1, 1, 5)


def test_decode_uses_allow_pickle_false():
    """The decode path must call np.load with allow_pickle=False (security).

    Asserted by source introspection — the decode happens inside a msgpack
    object_hook and is not cleanly mockable. This guards T-03-SC: a malicious
    pickle in ``as_npy`` must not be able to execute code.
    """
    src = inspect.getsource(MsgSerializer.decode_custom_classes)
    assert "np.load(" in src
    assert "allow_pickle=False" in src
    # And the encode side must not silently allow object-array pickling either.
    enc = inspect.getsource(MsgSerializer.encode_custom_classes)
    assert "allow_pickle=False" in enc


def test_decode_rejects_pickled_payload():
    """A pickled-object .npy in as_npy fails to load (allow_pickle=False)."""
    from io import BytesIO

    buf = BytesIO()
    # An object-dtype array can only be saved/loaded via pickle.
    np.save(buf, np.array([{"evil": 1}], dtype=object), allow_pickle=True)
    crafted = {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    with pytest.raises(ValueError):
        MsgSerializer.decode_custom_classes(crafted)


# --- Request / reply shape --------------------------------------------------


def test_get_action_request_shape(mocker):
    """get_action sends {"endpoint":"get_action","data":{observation,options:None}}."""
    client, sock = _make_client(mocker)
    reply = MsgSerializer.to_bytes(
        [
            {
                "single_arm": np.zeros((1, 16, 5), dtype=np.float32),
                "gripper": np.zeros((1, 16, 1), dtype=np.float32),
            },
            {},
        ]
    )
    sock.recv.return_value = reply

    observation = {"state": {"single_arm": np.zeros((1, 1, 5), np.float32)}}
    client.get_action(observation)

    sent_bytes = sock.send.call_args[0][0]
    decoded = MsgSerializer.from_bytes(sent_bytes)
    assert decoded["endpoint"] == "get_action"
    assert "data" in decoded
    assert set(decoded["data"].keys()) == {"observation", "options"}
    assert decoded["data"]["options"] is None


def test_get_action_returns_action_info_tuple(mocker):
    """get_action returns a 2-tuple (action_chunk_dict, info)."""
    client, sock = _make_client(mocker)
    action_chunk = {
        "single_arm": np.zeros((1, 16, 5), dtype=np.float32),
        "gripper": np.zeros((1, 16, 1), dtype=np.float32),
    }
    sock.recv.return_value = MsgSerializer.to_bytes([action_chunk, {"meta": 1}])

    result = client.get_action({"state": {}})
    assert isinstance(result, tuple) and len(result) == 2
    chunk, info = result
    assert isinstance(chunk, dict)
    assert set(chunk.keys()) == {"single_arm", "gripper"}
    assert info == {"meta": 1}


def test_error_sentinel_raises(mocker):
    """A raw b"ERROR" reply raises RuntimeError."""
    client, sock = _make_client(mocker)
    sock.recv.return_value = b"ERROR"
    with pytest.raises(RuntimeError, match="Server error"):
        client.call_endpoint("ping", requires_input=False)


def test_error_dict_reply_raises(mocker):
    """A decoded {"error": ...} reply raises RuntimeError naming the error."""
    client, sock = _make_client(mocker)
    sock.recv.return_value = MsgSerializer.to_bytes({"error": "boom"})
    with pytest.raises(RuntimeError, match="boom"):
        client.call_endpoint("get_action", {"observation": {}, "options": None})


# --- Socket lifecycle / reconnect ------------------------------------------


def test_init_socket_closes_old_socket(mocker):
    """_init_socket closes the prior socket (linger=0) and sets timeouts/linger."""
    client = ExternalRobotInferenceClient.__new__(ExternalRobotInferenceClient)
    client.host, client.port, client.timeout_ms, client.api_token = (
        "h",
        5555,
        15000,
        None,
    )
    old_sock = mocker.MagicMock()
    new_sock = mocker.MagicMock()
    client.socket = old_sock
    client.context = mocker.MagicMock()
    client.context.socket.return_value = new_sock

    client._init_socket()

    old_sock.close.assert_called_once_with(linger=0)
    opt_args = [c.args for c in new_sock.setsockopt.call_args_list]
    opt_names = {a[0] for a in opt_args}
    assert zmq.RCVTIMEO in opt_names
    assert zmq.SNDTIMEO in opt_names
    assert zmq.LINGER in opt_names
    # RCVTIMEO/SNDTIMEO carry the configured timeout.
    assert (zmq.RCVTIMEO, 15000) in opt_args
    assert (zmq.SNDTIMEO, 15000) in opt_args
    new_sock.connect.assert_called_once_with("tcp://h:5555")


def test_call_endpoint_retries_then_raises(mocker):
    """Timeouts trigger a bounded retry then a descriptive 'unreachable' raise."""
    client, sock = _make_client(mocker, host="deadhost", port=5555)
    sock.recv.side_effect = zmq.error.Again()  # always times out
    mocker.patch.object(client, "_init_socket")

    with pytest.raises(RuntimeError, match="unreachable") as excinfo:
        client.call_endpoint(
            "get_action", {"observation": {}, "options": None}, max_retries=3
        )
    # host:port named in the message
    assert "deadhost" in str(excinfo.value)
    assert "5555" in str(excinfo.value)
    # one teardown+recreate per failed attempt
    assert client._init_socket.call_count == 3


def test_call_endpoint_recovers_after_one_timeout(mocker):
    """A transient timeout retries, then succeeds on the next attempt."""
    client, sock = _make_client(mocker)
    good = MsgSerializer.to_bytes({"status": "ok"})
    sock.recv.side_effect = [zmq.error.Again(), good]
    mocker.patch.object(client, "_init_socket")

    result = client.call_endpoint("ping", requires_input=False, max_retries=3)
    assert result == {"status": "ok"}
    assert client._init_socket.call_count == 1  # one recovery


# --- Defensive validation ---------------------------------------------------


def test_get_action_missing_key_raises_runtimeerror(mocker):
    """A reply missing a modality key raises RuntimeError, not KeyError."""
    client, sock = _make_client(mocker)
    # action chunk missing "gripper"
    chunk = {"single_arm": np.zeros((1, 16, 5), dtype=np.float32)}
    sock.recv.return_value = MsgSerializer.to_bytes([chunk, {}])
    with pytest.raises(RuntimeError, match="gripper"):
        client.get_action({"state": {}})


def test_get_action_non_dict_chunk_raises(mocker):
    """A non-dict action chunk raises a descriptive RuntimeError."""
    client, sock = _make_client(mocker)
    sock.recv.return_value = MsgSerializer.to_bytes([[1, 2, 3], {}])
    with pytest.raises(RuntimeError, match="expected a dict"):
        client.get_action({"state": {}})


def test_get_action_wrong_ndim_raises(mocker):
    """An action array with wrong ndim raises a descriptive RuntimeError."""
    client, sock = _make_client(mocker)
    chunk = {
        "single_arm": np.zeros((16, 5), dtype=np.float32),  # 2-D, not (B,T,D)
        "gripper": np.zeros((1, 16, 1), dtype=np.float32),
    }
    sock.recv.return_value = MsgSerializer.to_bytes([chunk, {}])
    with pytest.raises(RuntimeError, match="expected"):
        client.get_action({"state": {}})


def test_get_action_bad_reply_arity_raises(mocker):
    """A reply that is not a 2-element (action, info) raises RuntimeError."""
    client, sock = _make_client(mocker)
    sock.recv.return_value = MsgSerializer.to_bytes([{"single_arm": 1}])  # len 1
    with pytest.raises(RuntimeError, match="2-element|contract drift"):
        client.get_action({"state": {}})


# --- Stale-import gate (Pitfall 6) ------------------------------------------


def _iter_py_files(*subdirs):
    for sub in subdirs:
        root = REPO_ROOT / sub
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_no_stale_baseinferenceserver_or_torchserializer_imports():
    """No module under policy/, embodiment/, shared/ imports the deleted
    server-side ``BaseInferenceServer`` or ``TorchSerializer`` symbols."""
    offenders = []
    for path in _iter_py_files("policy", "embodiment", "shared"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names}
                if names & {"BaseInferenceServer", "TorchSerializer"}:
                    offenders.append(str(path.relative_to(REPO_ROOT)))
            elif isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
                if names & {"BaseInferenceServer", "TorchSerializer"}:
                    offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, f"stale server-side imports found in: {offenders}"


def test_base_client_is_subclassed_by_external_client():
    """ExternalRobotInferenceClient is the BaseInferenceClient the controller imports."""
    assert issubclass(ExternalRobotInferenceClient, BaseInferenceClient)
