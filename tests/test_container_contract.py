"""CI-runnable container contract probe + stack-isolation guard.

Every test here is CI-runnable with NO live policy server, NO GPU, and NO SO101
hardware. Unlike ``tests/test_gr00t_service.py`` (which mocks the ZMQ socket with
a ``MagicMock``), this module stands up the real ``scripts/mock_policy_server.py``
``zmq.REP`` server on an EPHEMERAL loopback port and connects a *real*
``ExternalRobotInferenceClient`` over actual TCP — proving the msgpack/numpy
bytes survive a real socket, i.e. the container boundary.

Surfaces covered:
- Wire contract: the :5555 ``MsgSerializer`` contract is preserved across a real
  socket (get_action 2-tuple with single_arm (1,16,5) / gripper (1,16,1) float32;
  ping non-error; {"error": ...} reply -> RuntimeError).
- BOTH backend legs: the same three rules — a real socket round trip returns a
  correctly-shaped chunk, ``ping()`` is truthy against a live mock, and a
  server-side error reply raises ``RuntimeError`` — are asserted on the
  ``groot-native`` ZMQ leg AND the ``lerobot`` gRPC leg, parametrized over
  ``BACKEND_LEGS`` rather than duplicated, because the two legs' return SHAPES
  differ by design while the RULE is shared.
- Dependency isolation: the client's declared ``lerobot`` extras stay
  within an allowlist with ``feetech`` required, the pin stays exact, no
  server-only GPU or model package is a DIRECT dependency, the resolved closure
  from ``uv.lock`` carries no server-only package, and ``requires-python`` stays
  ``>=3.12``. The client is Py3.12. The two policy servers differ and the
  distinction matters: the Isaac-GR00T container is Py3.10, while the newer
  ``lerobot-policy`` container is Py3.12 because ``lerobot==0.6.1`` declares
  ``Requires-Python: >=3.12``.
  Six negative tests drive the helpers with synthetic input to prove the guard
  actually fails when a violation is injected.

Scope — this guard is a LOCAL pytest gate, and no CI workflow is created.
Recorded as a decision so a later reader does not read the absent workflow as an
oversight. The reasoning: this is one developer on one machine with the arm
attached to it, so CI's value (which scales with contributors and machines) is
small here; the guard runs in the suite before every commit; the word "CI" in
the requirement's own text names this contract-test file rather than a hosted
service; and the realistic GPU-dependency leak arrives with the
``lerobot[groot]`` extra, where a local ``pytest`` run catches it. The
requirement is therefore satisfied at the pytest level.

What this guard deliberately does NOT assert: that ``torch`` or the nvidia CUDA
wheels are absent from the RESOLVED environment. ``torch`` is an unconditional
base dependency of ``lerobot`` and the CUDA wheels resolve with it, so such an
assertion is unsatisfiable by construction and its inevitable remedy is to weaken
the guard. See ``FORBIDDEN_RESOLVED_PACKAGES``.

No live policy server, no GPU, no hardware is required; the suite completes fast.
"""

import os
import re
import socket
import sys
import threading
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import numpy as np
import pytest
import zmq

from policy.gr00t.service import (
    ExternalRobotInferenceClient,
    MsgSerializer,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make scripts/ importable so we can reuse the standalone mock servers.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from mock_grpc_policy_server import start_mock as start_grpc_mock  # noqa: E402
from mock_policy_server import serve_mock  # noqa: E402

from policy.lerobot import features  # noqa: E402


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


def _kill_zmq_mock(port: int) -> None:
    """Stop scripts/mock_policy_server's serve loop via the contract's "kill" endpoint."""
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


def _client(port: int) -> ExternalRobotInferenceClient:
    """A real client (real ZMQ REQ socket) pointed at the mock server."""
    return ExternalRobotInferenceClient(
        host="127.0.0.1", port=port, timeout_ms=5000
    )


# --- The two backend legs ----------------------------------------------------
#
# The same three RULES hold on both backends over a real loopback socket: a round
# trip returns a correctly-shaped chunk, ping() is truthy against a live mock, and
# a server-side error reply raises RuntimeError. The ASSERTION BODIES cannot be
# shared, because the two legs' return shapes differ BY DESIGN:
#
#   groot-native : the ZMQ transport returns an (action_chunk, info) tuple of
#                  modality arrays {single_arm: (1,16,5), gripper: (1,16,1)}
#   lerobot      : the backend returns list[dict["<joint>.pos", float]], len 16
#
# So the parametrization carries a THIRD element, ``assert_chunk``, holding each
# leg's own shape assertion. Forcing one body onto both legs would either weaken
# the incumbent groot-native assertions or fabricate a shape the lerobot leg does
# not produce. The groot-native assertions below are the incumbent contract,
# copied unchanged.


class _ZmqLegMock:
    """The groot-native leg's mock: a daemon-threaded zmq.REP server."""

    def __init__(self, port: int) -> None:
        self._port = port
        self._thread = threading.Thread(
            target=serve_mock, kwargs={"port": port, "host": "127.0.0.1"}, daemon=True
        )
        self._thread.start()
        _wait_for_port(port)

    def stop(self) -> None:
        _kill_zmq_mock(self._port)
        self._thread.join(timeout=2.0)


class _GrpcLegMock:
    """The lerobot leg's mock: a real grpc.Server, stopped by its own handle."""

    def __init__(self, port: int, mode: str = "ok") -> None:
        self._server = start_grpc_mock(port, "127.0.0.1", mode=mode)
        _wait_for_port(port)

    def stop(self) -> None:
        self._server.stop(grace=0)


class _ZmqLegClient:
    """A real ExternalRobotInferenceClient over a real ZMQ REQ socket."""

    def __init__(self, port: int) -> None:
        self._client = _client(port)

    def get_chunk(self) -> Any:
        return self._client.get_action({"state": {}})

    def ping(self) -> bool:
        return self._client.ping()

    def close(self) -> None:
        self._client.socket.close(linger=0)
        self._client.context.term()


class _LerobotLegClient:
    """The real LeRobotPolicyBackend over a real gRPC channel."""

    def __init__(self, port: int) -> None:
        from policy.lerobot.backend import LeRobotPolicyBackend

        # Scrub the DUME_LEROBOT_* variables for the duration of construction:
        # the backend reads them, and an operator's exported value would otherwise
        # change the handshake this suite asserts on. patch.dict restores them.
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in (
                "DUME_LEROBOT_POLICY_PORT",
                "DUME_LEROBOT_POLICY_TYPE",
                "DUME_LEROBOT_CHECKPOINT_PATH",
                "DUME_LEROBOT_ACTIONS_PER_CHUNK",
                "DUME_LEROBOT_POLICY_DEVICE",
            ):
                os.environ.pop(name, None)
            self._backend = LeRobotPolicyBackend(host="127.0.0.1", port=port)
        self._backend.set_lang_instruction("pick up the banana")

    def get_chunk(self) -> Any:
        return self._backend.get_action(_lerobot_observation())

    def ping(self) -> bool:
        return self._backend.ping()

    def close(self) -> None:
        self._backend.close()


def _lerobot_observation(height: int = 64, width: int = 64) -> dict:
    """The FLAT observation shape IRobotController.get_observation() produces."""
    obs: dict = {
        cam: np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for cam in features.CAMERA_KEYS
    }
    for joint in features.ROBOT_STATE_KEYS:
        obs[joint] = 0.0
    return obs


def _assert_groot_native_chunk(result: Any) -> None:
    """The INCUMBENT groot-native assertions, unchanged."""
    assert isinstance(result, tuple) and len(result) == 2
    action_chunk, info = result
    assert set(action_chunk.keys()) == {"single_arm", "gripper"}
    assert action_chunk["single_arm"].shape == (1, 16, 5)
    assert action_chunk["gripper"].shape == (1, 16, 1)
    assert action_chunk["single_arm"].dtype.name == "float32"
    assert action_chunk["gripper"].dtype.name == "float32"
    assert isinstance(info, dict)


def _assert_lerobot_chunk(result: Any) -> None:
    """The lerobot leg's normalized contract: 16 dicts of six named floats."""
    assert isinstance(result, list)
    assert len(result) == 16
    expected = list(features.ROBOT_STATE_KEYS)
    for step in result:
        # Ordered comparison: a permuted dict with the right keys passes a set
        # comparison and still commands the wrong joints.
        assert list(step) == expected
        assert all(isinstance(value, float) for value in step.values())


def _start_zmq_error_server(port: int) -> Any:
    """A one-shot zmq.REP server replying {"error": "boom"}; returns a stop handle."""
    thread = threading.Thread(
        target=_serve_error, kwargs={"port": port, "message": "boom"}, daemon=True
    )
    thread.start()
    _wait_for_port(port)

    class _Handle:
        def stop(self) -> None:
            thread.join(timeout=2.0)

    return _Handle()


def _provoke_groot_native_error(port: int) -> None:
    """Make the call that must raise on the groot-native leg."""
    client = _client(port)
    try:
        client.call_endpoint("get_action", {"observation": {}, "options": None})
    finally:
        client.socket.close(linger=0)
        client.context.term()


def _provoke_lerobot_error(port: int) -> None:
    """Make the call that must raise on the lerobot leg (handshake or chunk)."""
    client = _LerobotLegClient(port)
    try:
        client.get_chunk()
    finally:
        client.close()


@dataclass(frozen=True)
class BackendErrorCase:
    """One server-side error SHAPE for one leg."""

    id: str
    start: Callable[[int], Any]
    provoke: Callable[[int], None]
    match: str


@dataclass(frozen=True)
class BackendLeg:
    """One backend's real-socket harness plus its own chunk assertion."""

    id: str
    mock_factory: Callable[[int], Any]
    client_factory: Callable[[int], Any]
    assert_chunk: Callable[[Any], None]
    error_cases: tuple[BackendErrorCase, ...]


BACKEND_LEGS = (
    BackendLeg(
        id="groot-native",
        mock_factory=_ZmqLegMock,
        client_factory=_ZmqLegClient,
        assert_chunk=_assert_groot_native_chunk,
        error_cases=(
            BackendErrorCase(
                id="error-reply",
                start=_start_zmq_error_server,
                provoke=_provoke_groot_native_error,
                match="boom",
            ),
        ),
    ),
    BackendLeg(
        id="lerobot",
        mock_factory=_GrpcLegMock,
        client_factory=_LerobotLegClient,
        assert_chunk=_assert_lerobot_chunk,
        # TWO error shapes, not one: this wire can fail in two distinct ways that
        # a single case would conflate. "refuse" is a FAILED_PRECONDITION abort
        # (the shape a SAFE-01 guard refusal takes, deliberately non-retryable);
        # "empty" is a SUCCESSFUL RPC carrying zero-length Actions.data, which is
        # what upstream actually emits when GetActions swallows an exception.
        error_cases=(
            BackendErrorCase(
                id="refused-handshake",
                start=lambda port: _GrpcLegMock(port, mode="refuse"),
                provoke=_provoke_lerobot_error,
                match="SAFE-01/2",
            ),
            BackendErrorCase(
                id="empty-actions",
                start=lambda port: _GrpcLegMock(port, mode="empty"),
                provoke=_provoke_lerobot_error,
                # NOT merely "any RuntimeError": the empty case must be named, or
                # this assertion would also pass on an EOFError-shaped failure.
                match="ZERO-LENGTH action chunk",
            ),
        ),
    ),
)

#: Flattened (leg, case) pairs, DERIVED from BACKEND_LEGS so the parametrize ids
#: carry each leg's id and a per-shape suffix.
BACKEND_ERROR_CASES = tuple(
    (leg, case) for leg in BACKEND_LEGS for case in leg.error_cases
)


# --- Wire contract survives a real socket, on BOTH legs ----------------------


@pytest.mark.parametrize("leg", BACKEND_LEGS, ids=lambda leg: leg.id)
def test_real_socket_get_action_returns_the_leg_contract(leg):
    """A real client over real TCP returns THIS leg's chunk contract.

    Proves the bytes survive an actual socket round trip — the container
    boundary — for both backends. The assertion is on the DECODED chunk, never on
    the call merely returning: the lerobot wire can deliver a SUCCESSFUL RPC
    carrying zero bytes, which a "did it return" check would pass.
    """
    port = _free_port()
    server = leg.mock_factory(port)
    try:
        client = leg.client_factory(port)
        try:
            leg.assert_chunk(client.get_chunk())
        finally:
            client.close()
    finally:
        server.stop()


@pytest.mark.parametrize("leg", BACKEND_LEGS, ids=lambda leg: leg.id)
def test_real_socket_ping_returns_truthy(leg):
    """ping() over a real socket returns True against a live mock, on both legs."""
    port = _free_port()
    server = leg.mock_factory(port)
    try:
        client = leg.client_factory(port)
        try:
            assert client.ping() is True
        finally:
            client.close()
    finally:
        server.stop()


@pytest.mark.parametrize(
    "leg,case", BACKEND_ERROR_CASES, ids=lambda item: getattr(item, "id", "")
)
def test_real_socket_error_reply_raises_runtimeerror(leg, case):
    """A server-side error reply over a real socket raises RuntimeError, on both legs.

    The match is per-shape rather than a bare ``RuntimeError``: on the lerobot leg
    the two shapes are a non-retryable FAILED_PRECONDITION refusal and a
    zero-length action payload, and accepting any RuntimeError would let the
    zero-length guard be replaced by an EOFError without the suite noticing.
    """
    port = _free_port()
    server = case.start(port)
    try:
        with pytest.raises(RuntimeError, match=re.escape(case.match)):
            case.provoke(port)
    finally:
        server.stop()


def test_both_legs_bind_distinct_ephemeral_ports_in_one_session():
    """BACK-07 adjacency edge, at the suite level: the two mocks coexist.

    Two mocks that collided on a port would make the parametrized suite above
    pass or fail depending on test ORDER — the worst kind of green. Both legs are
    started in one test, both are exercised, one is stopped, and the other is
    proven to still answer.
    """
    ports = {leg.id: _free_port() for leg in BACKEND_LEGS}
    assert len(set(ports.values())) == len(BACKEND_LEGS), ports

    servers = {leg.id: leg.mock_factory(ports[leg.id]) for leg in BACKEND_LEGS}
    try:
        for leg in BACKEND_LEGS:
            client = leg.client_factory(ports[leg.id])
            try:
                leg.assert_chunk(client.get_chunk())
            finally:
                client.close()

        # Stop the lerobot leg; the groot-native leg must be untouched.
        servers["lerobot"].stop()
        groot = BACKEND_LEGS[0]
        client = groot.client_factory(ports[groot.id])
        try:
            assert client.ping() is True
            groot.assert_chunk(client.get_chunk())
        finally:
            client.close()
    finally:
        for server in servers.values():
            server.stop()


# --- Dependency isolation guard ---------------------------------------------

# RECORDED SUPPLY-CHAIN GATE — the two packages the ``async`` extra introduces.
#
# Widening the allowlist below to include ``async`` pulls exactly two names into
# the client closure, and BOTH were reviewed and approved by a human before the
# single ``uv add`` ran. Recorded here, at the guard that lets them in, so the
# approval is greppable from the code rather than buried in a chat log:
#
#   * ``grpcio``   — the gRPC project's official PyPI distribution
#     (https://pypi.org/project/grpcio/, homepage https://grpc.io). GENUINELY
#     NEW to the closure. Declared by the installed wheel's own metadata as
#     ``grpcio<2.0.0,>=1.73.1; extra == "grpcio-dep"``
#     (lerobot-0.6.1.dist-info/METADATA:81) — upstream's range, not one this
#     project chose — and named by lerobot's own fail-closed import guard:
#     ``import lerobot.transport`` raises "'grpcio' is required but not
#     installed. Install it with: pip install 'lerobot[grpcio-dep]'".
#   * ``protobuf`` — Google's Protocol Buffers runtime
#     (https://pypi.org/project/protobuf/). Declared as
#     ``protobuf<8.0.0,>=6.31.1; extra == "grpcio-dep"`` (METADATA:82) and
#     ALREADY RESOLVED at 6.33.6, so it is not actually entering the
#     environment — only being re-declared through an extra.
#
# An automated legitimacy audit returned [SUS] for both. Both verdicts were
# ACCEPTED AS FALSE POSITIVES of a ``too-new`` + ``unknown-downloads`` recency
# heuristic in which ``publishedAt`` is the most recent release date of a
# long-established project rather than the project's age. Neither name was
# invented by a researcher: both are reached transitively via
# ``lerobot[async] -> lerobot[grpcio-dep]`` (METADATA:227). The version decision
# that actually matters — ``lerobot==0.6.1`` — is the unchanged incumbent pin.
#
# ``async`` was chosen over the leaner ``grpcio-dep`` deliberately: it is the
# extra lerobot's own ImportError names, and it is the one this comment
# predicted. ``matplotlib`` arrives with it (via ``lerobot[matplotlib-dep]``)
# and trips nothing — ``pyproject.toml`` already declares ``matplotlib``
# directly, and the name is in neither ``FORBIDDEN_DIRECT_DEPENDENCIES`` nor
# ``FORBIDDEN_RESOLVED_PACKAGES``.

# An ALLOWLIST, not a denylist: an extra that is not named here fails by
# default, so a new heavyweight extra cannot slip in unnoticed. Phase 6 (the
# LeRobot policy container + gRPC backend) IS the "later phase" this comment
# predicted: it widens the set to include lerobot's ``async`` extra, because
# ``lerobot.async_inference`` fails closed without ``grpcio``
# (``lerobot/async_inference/__init__.py`` raises "'grpcio' is required but not
# installed. Install it with: pip install 'lerobot[async]'"), and the gRPC
# policy session cannot exist without it. That widening landed as a deliberate
# edit here in the same commit as the ``pyproject.toml`` change — not as a
# surprise failure — behind the human-approved supply-chain gate recorded above.
LEROBOT_EXTRAS_ALLOWLIST = frozenset({"feetech", "async"})

# Server-only GPU and model packages that must never be DIRECT client
# declarations. This is the ONLY place torch and the nvidia-cuda prefix belong:
# both are legitimately present in the RESOLVED closure (torch is an
# unconditional base dependency of lerobot, and the nvidia CUDA wheels come with
# it), so they are meaningful only as a statement about what this project itself
# declares.
FORBIDDEN_DIRECT_DEPENDENCIES = frozenset(
    {
        "torch",
        "torchvision",
        "transformers",
        "flash-attn",
        "tensorrt",
        "onnxruntime-gpu",
    }
)
NVIDIA_CUDA_PREFIX = "nvidia-cuda"

# Genuinely server-only packages that must not appear anywhere in the resolved
# closure.
#
# torch, torchvision and every nvidia-* wheel are DELIBERATELY EXCLUDED. torch is
# an unconditional base dependency of lerobot and roughly a dozen nvidia CUDA
# wheels resolve transitively with it, so asserting their absence from the
# resolved closure is unsatisfiable BY CONSTRUCTION — and the inevitable "fix"
# for a permanently red assertion is to gut the whole guard. Do not add them
# here to make the guard look stricter.
#
# diffusers was ADDED HERE by the same commit that bumps lerobot
# from 0.3.3 to 0.6.1 — because that bump is what makes the assertion satisfiable.
# diffusers 0.38.0 was a member of the resolved closure under the incumbent 0.3.3
# pin and is not a base dependency of 0.6.1 (it lives only in the groot extra), so
# this entry would have FAILED before that commit and passes after it. That
# fail-first property is the non-vacuity proof for the whole resolved-closure
# assertion: an entry that has never been red is an entry that has never been
# tested.
#
# `av` is DELIBERATELY NOT here, and it is the one name in the bump's
# dataset-reader group that survives. av 16.1.0 is required by aiortc
# (av<17.0.0,>=14.0.0), which pipecat-ai pulls through its webrtc extra for the
# voice transport — verified with `uv tree --invert --package av`. Its presence is
# nothing to do with lerobot's dataset extra, so asserting its absence would be
# unsatisfiable-by-construction in the same way a resolved-`torch` assertion would
# be, and its only available remedy would be dropping WebRTC voice support.
FORBIDDEN_RESOLVED_PACKAGES = frozenset(
    {
        "flash-attn",
        "tensorrt",
        "onnxruntime-gpu",
        "decord",
        "dm-tree",
        "peft",
        "timm",
        "diffusers",
    }
)

# name, optional [extras], then the version specifier. Environment markers are
# stripped before matching.
_REQUIREMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*"
    r"(?:\[(?P<extras>[^\]]*)\])?\s*"
    r"(?P<spec>.*)$"
)


def normalize_dist_name(name: str) -> str:
    """PEP 503 normalization: lowercase, runs of -, _ and . collapse to one -.

    Without this, ``Flash_Attn``, ``flash.attn`` and ``flash-attn`` read as three
    different distributions and only one spelling gets caught.
    """
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def _split_requirement(requirement: str) -> tuple[str, frozenset[str], str]:
    """Parse one requirement string into (normalized name, extras, specifier)."""
    text = requirement.split(";", 1)[0].strip()
    match = _REQUIREMENT_RE.match(text)
    if not match:
        raise ValueError(f"Unparseable requirement: {requirement!r}")
    raw_extras = match.group("extras") or ""
    extras = frozenset(
        normalize_dist_name(part) for part in raw_extras.split(",") if part.strip()
    )
    return normalize_dist_name(match.group("name")), extras, match.group("spec").strip()


def client_dependencies() -> list[str]:
    """The client's DIRECT dependency list, parsed from pyproject.toml.

    Structural parsing via tomllib — never a substring search over the file text,
    which is what let ``lerobot[groot]`` pass the previous guard.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        manifest = tomllib.load(handle)
    return list(manifest["project"]["dependencies"])


def parse_lerobot_requirement(dependencies) -> tuple[frozenset[str], str]:
    """Return the lerobot requirement's (extras, version specifier).

    Fails loudly when no lerobot requirement is present: silently returning an
    empty extras set would make the extras allowlist vacuously satisfiable.
    """
    for requirement in dependencies:
        name, extras, spec = _split_requirement(requirement)
        if name == "lerobot":
            return extras, spec
    raise ValueError(
        f"no lerobot requirement found in {list(dependencies)!r} — the extras "
        f"allowlist cannot be checked"
    )


def forbidden_direct_dependencies(dependencies) -> list[str]:
    """Normalized names in the DIRECT dependency list that are server-only."""
    leaked = set()
    for requirement in dependencies:
        name, _extras, _spec = _split_requirement(requirement)
        if name in FORBIDDEN_DIRECT_DEPENDENCIES or name.startswith(NVIDIA_CUDA_PREFIX):
            leaked.add(name)
    return sorted(leaked)


def read_lock_text(path) -> str:
    """Read uv.lock, raising FileNotFoundError rather than reporting clean.

    An absent lockfile means the resolved closure is unknown, which is not the
    same as known-clean; treating it as clean would be a vacuous pass.
    """
    lock_path = Path(path)
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"lockfile not found at {lock_path} — the resolved closure cannot be "
            f"checked, so the dependency-isolation guard cannot pass"
        )
    return lock_path.read_text(encoding="utf-8")


def resolved_package_names(lock_text: str) -> list[str]:
    """Normalized package names declared by the lockfile's [[package]] blocks."""
    names = []
    in_package = False
    for line in lock_text.splitlines():
        stripped = line.strip()
        if stripped == "[[package]]":
            in_package = True
            continue
        if in_package and stripped.startswith("name = "):
            raw = stripped.split("=", 1)[1].strip().strip('"')
            names.append(normalize_dist_name(raw))
            in_package = False
        elif stripped.startswith("["):
            in_package = False
    return names


def forbidden_resolved_packages(lock_text: str) -> list[str]:
    """Normalized server-only packages found in the resolved closure.

    Raises when the text declares no packages at all: an empty closure is not a
    clean closure, and reporting it clean would be exactly the vacuous pass this
    guard exists to prevent.
    """
    names = resolved_package_names(lock_text)
    if not names:
        raise ValueError(
            "the lockfile text declares no [[package]] entries — this is not a "
            "resolved closure, and reporting it clean would be a vacuous pass"
        )
    return sorted(set(names) & FORBIDDEN_RESOLVED_PACKAGES)


# --- Positive assertions against the real manifest and lockfile -------------


def test_client_lerobot_extras_within_allowlist():
    """The lerobot extras set is an ALLOWLIST subset, and feetech is required.

    An allowlist, not a denylist: a new heavyweight extra fails by default. This
    is the assertion that catches ``lerobot[groot]``, which pulls transformers,
    peft, diffusers, timm, dm-tree and an x86-only decoder on top of the torch
    that is already present transitively.
    """
    extras, _spec = parse_lerobot_requirement(client_dependencies())
    assert extras <= LEROBOT_EXTRAS_ALLOWLIST, (
        f"lerobot extras {sorted(extras)} exceed the allowlist "
        f"{sorted(LEROBOT_EXTRAS_ALLOWLIST)}"
    )
    assert "feetech" in extras, "the feetech extra is required for the motor bus"


def test_client_lerobot_pin_is_exact():
    """The lerobot requirement uses an exact == pin with a concrete version.

    Wire payloads are version-coupled once the LeRobot policy backend is wired,
    so an exact pin is the precondition for the client/server lockstep test.
    """
    _extras, spec = parse_lerobot_requirement(client_dependencies())
    assert spec.startswith("=="), f"lerobot pin must be exact, got {spec!r}"
    assert spec[2:].strip(), f"lerobot pin declares no concrete version: {spec!r}"


def test_client_has_no_direct_gpu_or_model_dependencies():
    """No DIRECT dependency is a server-only GPU or model package.

    Parsed, not substring-matched: parsing kills both the false pass on
    ``lerobot[groot]`` and the false fail on a package whose name merely contains
    a forbidden name as a substring. This is the only place torch and the
    nvidia-cuda prefix belong.
    """
    leaked = forbidden_direct_dependencies(client_dependencies())
    assert leaked == [], f"server GPU stack declared as a direct dependency: {leaked}"


def test_client_lock_has_no_server_only_packages():
    """The resolved closure from uv.lock contains no genuinely server-only package."""
    leaked = forbidden_resolved_packages(read_lock_text(REPO_ROOT / "uv.lock"))
    assert leaked == [], f"server-only packages in the resolved closure: {leaked}"


def test_client_lock_no_longer_resolves_diffusers():
    """diffusers has left the resolved closure, and the denylist says so.

    This is the guard's own non-vacuity proof, asserted against the REAL lockfile
    rather than a synthetic one. diffusers 0.38.0 was in the closure under the
    incumbent ``lerobot[feetech]==0.3.3`` pin; it is not a base dependency of
    0.6.1 (only the groot extra pulls it). So this test would have failed before
    the bump commit and passes after it — an entry with a demonstrated fail-first
    property, not a name added because it was already absent.

    A future regression here means something re-introduced the model stack, most
    likely the groot extra. The correct response is to find what pulled it in, not
    to remove diffusers from ``FORBIDDEN_RESOLVED_PACKAGES``.
    """
    resolved = resolved_package_names(read_lock_text(REPO_ROOT / "uv.lock"))
    assert "diffusers" not in resolved, (
        "diffusers is back in the resolved closure — investigate which dependency "
        "reintroduced it (the lerobot groot extra is the likely cause); do not "
        "remove it from the denylist"
    )
    # And the denylist carries it, so the closure check above is what enforces it.
    assert "diffusers" in FORBIDDEN_RESOLVED_PACKAGES


def test_client_lock_no_longer_resolves_the_dataset_reader_stack():
    """The dataset readers 0.6.0 moved behind the ``dataset`` extra are gone.

    lerobot 0.6.0 stopped bundling dataset dependencies; five of the six readers
    that were present only as a side effect of the 0.3.3 pin leave with the bump.
    Asserted against the real lockfile so a later ``lerobot[dataset]`` addition is
    caught as the closure enlargement it is.

    ``av`` is deliberately excluded from this list: it is required by aiortc for
    the WebRTC voice transport (``uv tree --invert --package av``), so it never
    depended on the lerobot pin and its presence is not a dataset-extra leak.
    """
    resolved = set(resolved_package_names(read_lock_text(REPO_ROOT / "uv.lock")))
    readers = {"torchcodec", "pandas", "pyarrow", "datasets", "jsonlines"}
    assert readers & resolved == set(), (
        f"dataset-reading packages back in the resolved closure: "
        f"{sorted(readers & resolved)} — check whether the dataset extra was added"
    )


# --- Negative tests: the guard has teeth (roadmap criterion 5) ---------------
#
# Each drives a helper with SYNTHETIC input. No test mutates the real
# pyproject.toml or uv.lock.


def test_guard_detects_torch_injected_as_direct_dependency():
    """A deliberately injected DIRECT torch declaration is reported.

    "Deliberately injected" means a direct declaration or a selected extra —
    never transitive base-dependency presence, which is unavoidable and would
    make the assertion unsatisfiable.
    """
    injected = ["fastapi>=0.129.0", "torch>=2.7", "lerobot[feetech]==0.6.1"]
    assert forbidden_direct_dependencies(injected) == ["torch"]

    for package in ("torchvision==0.26.0", "transformers>=4.57", "tensorrt", "onnxruntime-gpu"):
        name = normalize_dist_name(package.split("=")[0].split(">")[0].split("<")[0])
        assert forbidden_direct_dependencies([package]) == [name]

    # The nvidia-cuda family is caught by prefix, as a DIRECT declaration only.
    assert forbidden_direct_dependencies(["nvidia-cuda-runtime==13.0.96"]) == [
        "nvidia-cuda-runtime"
    ]

    # A package whose name merely CONTAINS a forbidden name is not a false fail.
    assert forbidden_direct_dependencies(["torchmetrics>=1.0", "pytorch-nowhere"]) == []


def test_guard_detects_lerobot_groot_extra():
    """A lerobot requirement carrying the groot extra is reported as a violation.

    This is the exact false pass the previous substring guard exhibited: the
    string ``lerobot[feetech,groot]==0.6.1`` contains none of that guard's six
    forbidden substrings.
    """
    extras, spec = parse_lerobot_requirement(["lerobot[feetech,groot]==0.6.1"])
    assert "groot" in extras
    assert not extras <= LEROBOT_EXTRAS_ALLOWLIST, (
        "the groot extra must fall outside the allowlist"
    )
    assert spec == "==0.6.1"

    # And the substring guard's blind spot is real: no forbidden name appears.
    assert forbidden_direct_dependencies(["lerobot[feetech,groot]==0.6.1"]) == []

    # A non-exact pin is rejected too.
    _extras, loose = parse_lerobot_requirement(["lerobot[feetech]>=0.6.1"])
    assert not loose.startswith("==")


def test_guard_detects_forbidden_package_in_resolved_lock():
    """A server-only package in a synthetic resolved closure is reported."""
    lock_text = (
        '[[package]]\nname = "fastapi"\nversion = "0.129.0"\n\n'
        '[[package]]\nname = "flash-attn"\nversion = "2.7.0"\n\n'
        '[[package]]\nname = "peft"\nversion = "0.17.0"\n'
    )
    assert forbidden_resolved_packages(lock_text) == ["flash-attn", "peft"]

    # torch, torchvision and the nvidia wheels are DELIBERATELY absent from the
    # resolved-closure denylist — they resolve legitimately, so asserting their
    # absence would be unsatisfiable by construction.
    legitimate = (
        '[[package]]\nname = "torch"\nversion = "2.11.0"\n\n'
        '[[package]]\nname = "torchvision"\nversion = "0.26.0"\n\n'
        '[[package]]\nname = "nvidia-cuda-runtime"\nversion = "13.0.96"\n'
    )
    assert forbidden_resolved_packages(legitimate) == []
    assert "torch" not in FORBIDDEN_RESOLVED_PACKAGES
    assert "torchvision" not in FORBIDDEN_RESOLVED_PACKAGES
    assert not any(p.startswith("nvidia-") for p in FORBIDDEN_RESOLVED_PACKAGES)

    # diffusers IS in the set, added by the same commit that
    # bumped lerobot to 0.6.1 and thereby made the assertion satisfiable. Before
    # that commit this entry was red; that is the point of it.
    assert "diffusers" in FORBIDDEN_RESOLVED_PACKAGES
    injected_diffusers = '[[package]]\nname = "diffusers"\nversion = "0.38.0"\n'
    assert forbidden_resolved_packages(injected_diffusers) == ["diffusers"]

    # av is NOT in the set: it resolves legitimately via aiortc <- pipecat-ai's
    # webrtc extra, so asserting its absence would be unsatisfiable.
    assert "av" not in FORBIDDEN_RESOLVED_PACKAGES


def test_guard_normalizes_distribution_names_per_pep503():
    """Flash_Attn, flash.attn and flash-attn are all the same distribution."""
    assert (
        normalize_dist_name("Flash_Attn")
        == normalize_dist_name("flash.attn")
        == normalize_dist_name("flash-attn")
        == "flash-attn"
    )
    # Runs of separators collapse to a single hyphen.
    assert normalize_dist_name("dm___tree") == "dm-tree"
    assert normalize_dist_name("ONNXRuntime-.GPU") == "onnxruntime-gpu"

    # All three spellings are DETECTED, in both surfaces.
    for spelling in ("Flash_Attn", "flash.attn", "flash-attn"):
        assert forbidden_direct_dependencies([f"{spelling}>=2.7"]) == ["flash-attn"]
        lock_text = f'[[package]]\nname = "{spelling}"\nversion = "2.7.0"\n'
        assert forbidden_resolved_packages(lock_text) == ["flash-attn"]


def test_guard_fails_loudly_when_lockfile_absent():
    """An absent lockfile raises rather than reporting an empty clean list."""
    missing = REPO_ROOT / "uv.lock.does-not-exist"
    assert not missing.exists()
    with pytest.raises(FileNotFoundError):
        read_lock_text(missing)

    # A lockfile that declares no packages at all cannot be a resolved closure;
    # reporting it clean would be a vacuous pass.
    for empty in ("", "\n\n", "[manifest]\nmembers = []\n"):
        with pytest.raises(ValueError):
            forbidden_resolved_packages(empty)


def test_guard_rejects_requirement_declaring_no_extras():
    """A lerobot requirement declaring no extras fails the feetech assertion."""
    extras, spec = parse_lerobot_requirement(["lerobot==0.6.1"])
    assert extras == frozenset()
    assert "feetech" not in extras, "no-extras must fail the feetech-required check"
    assert extras <= LEROBOT_EXTRAS_ALLOWLIST, "the empty set is trivially a subset"
    assert spec == "==0.6.1"

    # An empty bracket group is also no extras, not a parse error.
    empty_brackets, _spec = parse_lerobot_requirement(["lerobot[]==0.6.1"])
    assert empty_brackets == frozenset()

    # A dependency list with no lerobot requirement at all fails loudly rather
    # than silently reporting a clean, empty extras set.
    with pytest.raises(ValueError):
        parse_lerobot_requirement(["fastapi>=0.129.0", "loguru>=0.7.3"])


def test_client_requires_python_stays_312():
    """The client interpreter floor stays >=3.12.

    The floor is load-bearing and unchanged. What needed clarifying is WHICH
    container is which: the Isaac-GR00T inference container is Py3.10, while the
    newer ``lerobot-policy`` container is Py3.12 because
    ``lerobot-0.6.1.dist-info/METADATA`` declares ``Requires-Python: >=3.12``. So
    "the server is Py3.10" is true only of the older of the two, and the client's
    ``>=3.12`` floor is what lets it import ``lerobot.transport`` at all.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12"' in text


def _load_container_entrypoint():
    """Load the container's entrypoint module from source, without Docker.

    The Dockerfile copies only ``docker/lerobot-policy/*.py`` into the image, so
    there is no package to import from the host tree. Loading the file directly
    is what lets the preflight harness's exit contract be pinned by a keyless CI
    test instead of resting on a manual ``docker run`` observation.
    """
    import importlib.util

    path = REPO_ROOT / "docker" / "lerobot-policy" / "entrypoint.py"
    spec = importlib.util.spec_from_file_location("_dume_container_entrypoint", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_container_server():
    """Load the container's ``server.py`` from source, without Docker.

    Importable from the client venv: ``server.py``'s imports are all ``lerobot``,
    ``torch``, ``grpc`` and ``policy_guard``, every one of which the Dum-E venv
    already carries (``transformers`` is reached only transitively through
    ``lerobot``, never named here). That is what makes the SAFE-01 post-load call
    site's fail-closed behaviour pinnable by a keyless test instead of resting on a
    live ``docker run``.

    Unlike ``_load_container_entrypoint`` this loads no ``sys.path`` mutation of its
    own — ``server.py`` performs none.
    """
    import importlib.util

    path = REPO_ROOT / "docker" / "lerobot-policy" / "server.py"
    spec = importlib.util.spec_from_file_location("_dume_container_server", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD_WRAPPER = REPO_ROOT / "scripts" / "build_lerobot_policy_image.sh"


def _run_build_wrapper_sandboxed(tmp_path, dotenv_body: str):
    """Run the build wrapper in a throwaway tree with ``docker`` stubbed out.

    Executes the REAL script, so the dotenv handling under test is the shipped one.
    ``docker`` is replaced by a shim that echoes its argv and exits 0, which is what
    makes the ``--build-arg`` values observable: they are the pins the script actually
    handed the build, not a restatement of its own variables.

    Nothing here touches the real repo, the real ``.env`` or the real Docker daemon.
    """
    import shutil
    import subprocess

    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    (root / "docker" / "lerobot-policy").mkdir(parents=True)
    shutil.copy(BUILD_WRAPPER, root / "scripts" / BUILD_WRAPPER.name)
    (root / "docker" / "lerobot-policy" / "Dockerfile").write_text("FROM scratch\n")
    (root / ".env").write_text(dotenv_body)

    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    docker = fakebin / "docker"
    docker.write_text('#!/bin/sh\necho "[docker] $*"\nexit 0\n')
    docker.chmod(0o755)

    env = dict(os.environ)
    env.pop("HF_TOKEN", None)
    env.pop("DUME_NO_DOTENV", None)
    env["PATH"] = f"{fakebin}:{env['PATH']}"
    return subprocess.run(
        ["bash", f"scripts/{BUILD_WRAPPER.name}"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_dotenv_cannot_replace_the_build_wrappers_reproducibility_pins(tmp_path):
    """A ``.env`` entry cannot silently replace ``LEROBOT_PIN``/``BACKBONE_REVISION``.

    WR-08. The wrapper assigned its five pins and THEN ran ``set -a; . .env; set +a``,
    which executes .env in the current shell and exports everything it assigns — so any
    of those five names appearing in .env overwrote the pin, unconditionally and with no
    message. The post-build assertion could not catch it, because it compares the image
    against ``$BACKBONE_REVISION``: the SAME overridden variable. The "reproducibility
    anchor" was self-referential once .env was in play, and the image would still be
    tagged ``lerobot-policy`` while carrying a different backbone revision than the
    script declares.

    Asserted against the pins the script actually handed ``docker build``, and against
    the value it handed the in-image assertion — the two places the override would show
    up — rather than against the script's own echo of its variables.
    """
    hostile = (
        "HF_TOKEN=hf_sandbox_not_a_real_token\n"
        "LEROBOT_PIN=9.9.9\n"
        "BACKBONE_REVISION=deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\n"
        "IMAGE_TAG=totally-different\n"
    )
    result = _run_build_wrapper_sandboxed(tmp_path, hostile)
    combined = result.stdout + result.stderr

    source = BUILD_WRAPPER.read_text(encoding="utf-8")
    pinned_revision = re.search(r'^BACKBONE_REVISION="([0-9a-f]+)"', source, re.M).group(1)
    pinned_lerobot = re.search(r'^LEROBOT_PIN="([^"]+)"', source, re.M).group(1)

    assert f"--build-arg LEROBOT_PIN={pinned_lerobot}" in combined, combined
    assert f"--build-arg BACKBONE_REVISION={pinned_revision}" in combined, combined
    assert f"EXPECT_BACKBONE_REVISION={pinned_revision}" in combined, (
        "the in-image assertion was handed the .env's revision, so it validated the "
        f"override instead of catching it:\n{combined}"
    )
    for override in ("9.9.9", "deadbeef", "totally-different"):
        assert override not in combined, (
            f"the .env value {override!r} reached the build:\n{combined}"
        )

    # The correct behaviour must be VISIBLE. "The .env entry is ignored" is otherwise
    # as silent as "the .env entry wins" was, and an operator who put a revision in
    # .env expecting it to take effect could not tell which happened. Names only --
    # never values, and never the token.
    assert "WARNING: .env names build pin(s)" in result.stderr, result.stderr
    for named in ("LEROBOT_PIN", "BACKBONE_REVISION", "IMAGE_TAG"):
        assert named in result.stderr, result.stderr
    assert "hf_sandbox_not_a_real_token" not in combined, "the token was echoed"


def test_the_build_wrapper_still_reads_hf_token_from_dotenv(tmp_path):
    """Hardening the dotenv path did not break the thing it exists for.

    The token still has to come out of ``.env`` — this repo's convention is that
    credentials live there, not in the shell — and a wrapper that stopped reading it
    would fail closed on every normal invocation.
    """
    result = _run_build_wrapper_sandboxed(tmp_path, "HF_TOKEN=hf_sandbox_not_a_real_token\n")
    combined = result.stdout + result.stderr

    assert "HF_TOKEN is not set" not in combined, (
        f"the wrapper no longer reads HF_TOKEN from .env:\n{combined}"
    )
    assert "--secret id=hf_token,env=HF_TOKEN" in combined, combined
    # A token still reaches Docker ONLY as a BuildKit secret, never a --build-arg
    # (which would persist into the image history).
    assert "--build-arg HF_TOKEN" not in combined, combined
    assert "hf_sandbox_not_a_real_token" not in combined, "the token was echoed"
    # And no pin warning on a normal .env.
    assert "names build pin(s)" not in result.stderr, result.stderr


def test_the_build_wrapper_fails_closed_without_a_token(tmp_path):
    """No token anywhere still refuses to build, naming the gated repo."""
    result = _run_build_wrapper_sandboxed(tmp_path, "SOMETHING_ELSE=x\n")
    assert result.returncode != 0
    assert "HF_TOKEN is not set" in result.stderr
    assert "[docker] build" not in result.stdout, "it started a build without a token"


def test_container_camera_and_frame_geometry_match_the_client_handshake():
    """The server's declared geometry IS the client's, asserted ACROSS the boundary.

    WR-06. ``docker/lerobot-policy/server.py`` states the requirement — its
    ``CAMERA_KEYS``/``FRAME_*``/``STATE_DIM``/``ACTION_DIM`` "must agree with
    ``policy/lerobot/features.py``" — but nothing asserted it, and the values cannot
    be imported: the Dockerfile copies ``policy_guard/`` and
    ``docker/lerobot-policy/*.py`` into the image, not ``policy/``. So they are pinned
    copies, and this is the keyless cross-check that keeps them from drifting (the
    same idiom ``scripts/dump_preprocessed_image.py`` uses for
    ``SERVING_LETTER_BOX_TRANSFORM``).

    **The two halves fail differently, and this test exists for the quiet one.** The
    client's ``lerobot_features`` decides which ``observation.images.<cam>`` keys
    arrive; ``config.input_features`` decides which are looked up AND what they are
    resized to. A camera-NAME disagreement is a loud ``KeyError`` in
    ``prepare_raw_observation``. A frame-SIZE disagreement is silent:
    ``raw_observation_to_observation`` -> ``prepare_raw_observation`` *resizes* every
    incoming frame to the declared shape (``helpers.py:165-168``), which is blocker
    3's mechanism — so a drift reintroduces the aspect-ratio corruption the module
    says it prevents, with correct shapes end to end. It is shape-invisible
    downstream too, because the forced pad squares every input. Nothing else in the
    suite would notice.
    """
    server_module = _load_container_server()
    server = server_module.DumEGrootPolicyServer

    assert tuple(server.CAMERA_KEYS) == tuple(features.CAMERA_KEYS), (
        f"server CAMERA_KEYS {tuple(server.CAMERA_KEYS)} != client "
        f"{tuple(features.CAMERA_KEYS)}; a name disagreement is a KeyError in "
        "prepare_raw_observation"
    )
    assert (server.FRAME_HEIGHT, server.FRAME_WIDTH) == (
        features.FRAME_HEIGHT,
        features.FRAME_WIDTH,
    ), (
        f"server frame geometry ({server.FRAME_HEIGHT}, {server.FRAME_WIDTH}) != client "
        f"({features.FRAME_HEIGHT}, {features.FRAME_WIDTH}). This one is SILENT: the "
        "server resizes every incoming frame to its own declared shape, so the aspect "
        "ratio is corrupted with correct shapes end to end"
    )
    assert server.STATE_DIM == server.ACTION_DIM == len(features.ROBOT_STATE_KEYS) == 6

    # And the client's own handshake payload really declares that geometry, so the
    # comparison is against the bytes on the wire rather than two constants that
    # happen to match.
    built = features.build_lerobot_features()
    for cam in server.CAMERA_KEYS:
        key = f"observation.images.{cam}"
        assert key in built, f"{key} is not in the handshake features: {sorted(built)}"
        assert tuple(built[key]["shape"])[:2] == (server.FRAME_HEIGHT, server.FRAME_WIDTH)


class _AbortRaised(Exception):
    """What a real ``ServicerContext.abort`` does: terminate by raising."""


class _FakeContext:
    """A ``grpc.ServicerContext`` stand-in that records the abort it was given.

    ``abort_returns`` inverts the one behaviour the refusal path used to depend on
    implicitly, so "control leaves the handler" can be asserted rather than assumed.
    """

    def __init__(self, abort_returns: bool = False) -> None:
        self.abort_returns = abort_returns
        self.aborted: tuple[Any, str] | None = None

    def peer(self) -> str:
        return "ipv4:127.0.0.1:0"

    def abort(self, code, details):
        self.aborted = (code, details)
        if self.abort_returns:
            return None
        raise _AbortRaised(details)


def _handshake_request(server_module, actions_per_chunk: int = 16):
    """A pickled ``RemotePolicyConfig`` in the shape the wire carries it."""
    import pickle

    from lerobot.async_inference.helpers import RemotePolicyConfig

    class _Request:
        pass

    request = _Request()
    request.data = pickle.dumps(
        RemotePolicyConfig(
            policy_type="groot",
            pretrained_name_or_path=str(REAL_CHECKPOINT_FOR_SERVER),
            lerobot_features=features.build_lerobot_features(),
            actions_per_chunk=actions_per_chunk,
            device="cpu",
            rename_map={},
        )
    )
    return request


REAL_CHECKPOINT_FOR_SERVER = REPO_ROOT / "checkpoints" / "GR00T-N1.7-3B-SO101"


class _PackStep:
    """Stands in for ``GrootN17PackInputsStep``: the ``state_dropout_prob`` marker."""

    state_dropout_prob = 0.2
    training = False


class _EncodeStep:
    """Stands in for ``GrootN17VLMEncodeStep`` with the serving pad FORCED on."""

    def __init__(self, letter_box_transform: bool = True) -> None:
        self.letter_box_transform = letter_box_transform
        self.training = False


class GrootN17ActionDecodeStep:  # noqa: N801 - the NAME is the assertion
    """Stands in for upstream's relative-aware decode step.

    Deliberately NOT underscore-prefixed: SAFE-01/3 compares
    ``type(step).__name__`` against ``EXPECTED_DECODE_STEP``, so the class name IS
    the thing under test. The legacy ``GrootActionUnpackUnnormalizeStep`` is what a
    real refusal here would name. ``env_action_dim`` is the marker
    ``_decode_step_type`` locates it by.
    """

    env_action_dim = 6


class _StubPipeline:
    def __init__(self, steps=()):
        self.steps = tuple(steps)
        self.name = "stand-in"


def _guard_passing_pipelines():
    """Pipeline stubs whose shape the SAFE-01 guard ACCEPTS against the real checkpoint.

    A harness that could only ever produce a refusal would make every
    ``"SAFE-01 guard: PASS" not in log`` assertion below vacuous, so the default is
    the passing shape and ``test_..._logs_pass_on_a_conforming_pipeline`` is the
    positive control that proves it.
    """
    return _StubPipeline([_PackStep(), _EncodeStep()]), _StubPipeline(
        [GrootN17ActionDecodeStep()]
    )


def _armed_handshake_server(server_module, monkeypatch):
    """A ``DumEGrootPolicyServer`` whose weight load and pipelines are stubbed out.

    Stubs exactly the three things a 12.6 GB load would otherwise require — the
    model materialization, the processor build and the dtype histogram — and NOTHING
    on the guard path: every SAFE-01 field except the four the pipelines carry is
    read from the REAL checkpoint's sidecars. The handler body under test is the
    real one.
    """

    class _StubConfig:
        embodiment_tag = "new_embodiment"
        model_params_fp32 = False
        input_features: dict = {}
        output_features: dict = {}

        def __init__(self):
            self.base_model_path = str(REAL_CHECKPOINT_FOR_SERVER)

    class _StubPolicy:
        def __init__(self):
            self.config = _StubConfig()

        def to(self, device):
            return self

    monkeypatch.setattr(
        server_module.DumEGrootPolicy,
        "from_pretrained",
        classmethod(lambda cls, path, config=None: _StubPolicy()),
    )
    monkeypatch.setattr(
        server_module, "make_pre_post_processors", lambda *a, **k: _guard_passing_pipelines()
    )
    monkeypatch.setattr(server_module, "parameter_dtype_histogram", lambda module: {})

    class _Logger:
        def __init__(self):
            self.lines: list[str] = []

        def _record(self, fmt, *args):
            self.lines.append(fmt % args if args else str(fmt))

        info = _record
        warning = _record
        error = _record

    server = object.__new__(server_module.DumEGrootPolicyServer)
    # `running` is a read-only property over upstream's shutdown_event
    # (policy_server.py), and only Ready() clears it. A cleared event is what the
    # client's mandatory Ready-before-SendPolicyInstructions ordering produces, so
    # this is the state the handler under test actually runs in.
    server.shutdown_event = threading.Event()
    assert server.running is True
    server.logger = _Logger()
    server.policy = None
    server.preprocessor = None
    server.postprocessor = None
    return server


def test_safe01_post_load_guard_drops_the_policy_on_a_non_valueerror_failure(monkeypatch):
    """An ``AttributeError`` from ``snapshot_from_loaded`` REFUSES and drops the policy.

    This is CR-02, and it is the discriminating case: the call site used to catch
    ``except ValueError`` only. ``assert_groot_serving_contract`` raises only
    ``ValueError``, but ``snapshot_from_loaded``'s own docstring advertises that it
    fails **at attribute-access time** — i.e. ``AttributeError`` — and
    ``infer_groot_n1_7_action_horizon`` can raise ``KeyError`` on a reshaped
    sidecar. Under the narrow clause none of those reached the handler, so
    ``self.policy`` stayed bound to a loaded, un-validated policy and the exception
    escaped as ``UNKNOWN``; a client that ignored that and called
    ``SendObservations``/``GetActions`` anyway would have been served from it.

    A test that only proves the ``ValueError`` path still works cannot close this —
    that path was never broken. So the failure is injected at a REAL read site by
    deleting the attribute ``snapshot_from_loaded`` reads first.
    """
    server_module = _load_container_server()
    if not REAL_CHECKPOINT_FOR_SERVER.is_dir():
        pytest.fail(
            f"checkpoint not found at {REAL_CHECKPOINT_FOR_SERVER} — a skip here would "
            "be a silent pass on the fail-closed claim"
        )
    server = _armed_handshake_server(server_module, monkeypatch)
    context = _FakeContext()

    # THE INJECTION, at a real read site inside snapshot_from_loaded: a pack step
    # that carries `state_dropout_prob` (so `_require_step` locates it and does NOT
    # raise its own ValueError) but no `training`. `groot_guard.py:524` then does
    # `bool(pack_step.training)` and raises AttributeError — the module's documented
    # "breaks LOUDLY, at attribute-access time" behaviour, produced by the guard
    # itself rather than hand-thrown. This is exactly the shape a pinned-lerobot
    # pipeline reshape takes: the marker survives, the field next to it moves.
    class _ReshapedPackStep:
        state_dropout_prob = 0.2
        # `training` deliberately ABSENT.

    monkeypatch.setattr(
        server_module,
        "make_pre_post_processors",
        lambda *a, **k: (
            _StubPipeline([_ReshapedPackStep(), _EncodeStep()]),
            _StubPipeline([GrootN17ActionDecodeStep()]),
        ),
    )

    with pytest.raises(_AbortRaised):
        server_module.DumEGrootPolicyServer.SendPolicyInstructions(
            server, _handshake_request(server_module), context
        )

    # 1. It refused, with FAILED_PRECONDITION and the exception TYPE named, so an
    #    operator can tell "the guard said no" from "the guard could not run".
    assert context.aborted is not None, "the handler did not abort"
    code, details = context.aborted
    assert code is server_module.grpc.StatusCode.FAILED_PRECONDITION
    assert "AttributeError" in details, details

    # 2. It dropped the un-validated policy AND the pipelines. This is the assertion
    #    the narrow `except ValueError` failed: under it, `server.policy` was still
    #    the loaded object here.
    assert server.policy is None, (
        "an un-validated policy is still bound after a guard failure the narrow "
        "`except ValueError` clause did not catch — SendObservations/GetActions "
        "would serve from it"
    )
    assert server.preprocessor is None
    assert server.postprocessor is None

    # 3. And it logged the refusal, not a pass.
    log = "\n".join(server.logger.lines)
    assert "SAFE-01 guard: REFUSED" in log, log
    assert "AttributeError" in log, log
    assert "SAFE-01 guard: PASS" not in log, log


def test_safe01_post_load_guard_still_refuses_a_valueerror_violation(monkeypatch):
    """Broadening the catch did not replace the ``ValueError`` message path.

    The guard's own refusals must keep arriving with their ``SAFE-01/N`` identifier
    intact, because that identifier is what the live suite greps for and what tells
    an operator which assertion fired.
    """
    server_module = _load_container_server()
    server = _armed_handshake_server(server_module, monkeypatch)
    context = _FakeContext()

    monkeypatch.setattr(
        server_module,
        "snapshot_from_loaded",
        lambda *a, **k: (_ for _ in ()).throw(
            ValueError("SAFE-01/2 configured actions_per_chunk=40 disagrees with 16")
        ),
    )

    with pytest.raises(_AbortRaised):
        server_module.DumEGrootPolicyServer.SendPolicyInstructions(
            server, _handshake_request(server_module), context
        )

    _code, details = context.aborted
    assert "SAFE-01/2" in details, details
    assert "ValueError" in details, details
    assert server.policy is None
    log = "\n".join(server.logger.lines)
    assert "SAFE-01 guard: REFUSED" in log
    assert "SAFE-01 guard: PASS" not in log


def test_safe01_post_load_guard_logs_pass_on_a_conforming_pipeline(monkeypatch):
    """THE POSITIVE CONTROL for the three refusal tests above.

    They assert ``"SAFE-01 guard: PASS" not in log``. If this harness could never
    produce that line, all three would be vacuous. So: with pipeline stubs whose
    markers and flags conform — a pack step carrying ``state_dropout_prob`` and
    ``training=False``, an encode step with the serving pad FORCED on, and a decode
    step named ``GrootN17ActionDecodeStep`` — the real handler body runs the real
    guard against the REAL checkpoint's sidecars and passes.

    Note what this does NOT stub: the SAFE-01 assertions themselves, the snapshot
    builder, or any of the thirteen checkpoint-derived fields. Only the weight load
    and the processor build are stubbed.
    """
    server_module = _load_container_server()
    server = _armed_handshake_server(server_module, monkeypatch)
    context = _FakeContext()

    reply = server_module.DumEGrootPolicyServer.SendPolicyInstructions(
        server, _handshake_request(server_module), context
    )

    assert context.aborted is None, f"a conforming pipeline was refused: {context.aborted}"
    assert reply is not None
    log = "\n".join(server.logger.lines)
    assert "SAFE-01 guard: PASS" in log, log
    assert "SAFE-01 guard: REFUSED" not in log, log
    # The served letterbox value is on the PASS line specifically because it is the
    # one value the checkpoint's own configuration contradicts.
    assert "served_letter_box_transform=True" in log, log
    assert "checkpoint_letter_box_transform=False" in log, log
    assert server.policy is not None


def test_a_returning_abort_cannot_produce_a_safe01_pass_line(monkeypatch):
    """A refusal never reaches the ``SAFE-01 guard: PASS`` log, even if abort returns.

    WR-04. ``_refuse`` was called as a bare statement and control left the handler
    only because ``context.abort`` happens to raise. If that ever stopped being true
    — an ``aio`` servicer context, a test double, an upstream change — execution fell
    through to the unconditional ``SAFE-01 guard: PASS`` line and
    ``return services_pb2.Empty()``: a PASS log and an OK handshake reply for a
    handshake the guard REFUSED. That is the precise inverse of T-06-14, and
    ``tests/test_lerobot_serving_live.py`` greps that exact line as positive
    evidence. (It would also reference the unbound ``snapshot`` local, masking the
    real diagnosis behind an ``UnboundLocalError``.)

    The property is asserted with the inversion actually injected — an ``abort``
    that RETURNS — rather than by reading the code, because "abort raises" is the
    assumption under test.

    The refusal is injected at ``assert_groot_serving_contract``, NOT at
    ``snapshot_from_loaded``, and the difference is the whole point: a
    snapshot-build failure leaves ``snapshot`` unbound, so the fall-through would
    die on ``UnboundLocalError`` before the PASS line and the hazard would be
    invisible. A guard REFUSAL — the common case, e.g. SAFE-01/2's wrong horizon —
    leaves ``snapshot`` bound with real values, so the fall-through logs a fully
    populated, entirely false PASS line and returns OK.
    """
    server_module = _load_container_server()
    server = _armed_handshake_server(server_module, monkeypatch)
    context = _FakeContext(abort_returns=True)

    monkeypatch.setattr(
        server_module,
        "assert_groot_serving_contract",
        lambda snapshot: (_ for _ in ()).throw(
            ValueError("SAFE-01/2 configured actions_per_chunk=40 disagrees with 16")
        ),
    )

    # It must not return an OK reply. Any raise is acceptable; a return is not.
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - the TYPE is not the contract
        server_module.DumEGrootPolicyServer.SendPolicyInstructions(
            server, _handshake_request(server_module), context
        )
    assert not isinstance(excinfo.value, AssertionError), excinfo.value

    assert context.aborted is not None, "the refusal never reached context.abort"
    log = "\n".join(server.logger.lines)
    assert "SAFE-01 guard: REFUSED" in log, log
    assert "SAFE-01 guard: PASS" not in log, (
        "a handshake the guard REFUSED produced a 'SAFE-01 guard: PASS' log line — "
        f"the live suite greps that line as positive evidence:\n{log}"
    )
    assert server.policy is None


def test_preflight_refuses_when_a_check_never_runs():
    """A vanished preflight check FAILS; it is not silently absent (T-06-37).

    This is the repudiation gap the Phase 6 security audit found. The harness is
    a ported copy whose original gates on ``passed == len(self.results)`` — every
    *recorded* check. Under that contract, deleting a check from
    ``run_preflight`` records nothing for it, the remaining checks all pass, and
    the container starts on an incomplete preflight, indistinguishable from one
    that genuinely verified everything.

    Both halves are pinned, because only asserting the happy path is how this
    regressed in the first place: all-ran-and-passed exits 0, and
    fewer-ran-but-all-passed exits non-zero.
    """
    entrypoint = _load_container_entrypoint()

    # All expected checks ran and passed -> serve.
    complete = entrypoint.Checks(3)
    for i in range(3):
        complete.start(f"check {i}")
        complete.ok(f"name-{i}", "fine")
    assert complete.report() == 0

    # One check never ran. Every RECORDED check passed, so the original contract
    # would return 0 here -- that is precisely the hole.
    truncated = entrypoint.Checks(3)
    for i in range(2):
        truncated.start(f"check {i}")
        truncated.ok(f"name-{i}", "fine")
    assert truncated.report() != 0, (
        "a preflight missing a check must refuse to serve; every recorded check "
        "passing is not evidence that every expected check ran"
    )

    # A recorded failure still fails, so the new clause did not replace the old one.
    failing = entrypoint.Checks(2)
    failing.start("check 0")
    failing.ok("name-0", "fine")
    failing.start("check 1")
    failing.fail("name-1", "nope")
    assert failing.report() != 0


def test_preflight_numbered_lines_track_total_checks():
    """``TOTAL_CHECKS`` is the single source of the ``[n/N]`` prefixes.

    Guards against the stale-literal drift the audit looked for: the count was
    raised 5 -> 6 when SAFE-01 was armed, and the prefixes must follow the
    constant rather than a hardcoded number.
    """
    entrypoint = _load_container_entrypoint()
    source = (REPO_ROOT / "docker" / "lerobot-policy" / "entrypoint.py").read_text(
        encoding="utf-8"
    )

    # run_preflight() must open exactly TOTAL_CHECKS numbered checks.
    assert source.count("checks.start(") == entrypoint.TOTAL_CHECKS, (
        f"run_preflight() opens {source.count('checks.start(')} checks but "
        f"TOTAL_CHECKS is {entrypoint.TOTAL_CHECKS}"
    )

    # The prefix is formatted from self.total, never a literal.
    assert 'f"\\n[{self.index}/{self.total}]' in source
