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
- Stack isolation: the server GPU stack (torch/flash-attn/CUDA/tensorrt/onnxruntime-gpu)
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


# --- Wire contract survives a real socket -----------------------------------


def test_real_socket_get_action_returns_action_info_tuple(mock_server):
    """A real client get_action over real TCP returns the (action_chunk, info) tuple.

    Proves the msgpack/numpy bytes (single_arm (1,16,5) / gripper (1,16,1) f32)
    survive the actual socket round trip — the container boundary.
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


# --- Dependency isolation guard (SAFE-04) -----------------------------------
#
# RED phase (plan 05-03 Task 3): the helpers below are stubs. The positive and
# negative tests that follow MUST fail until the GREEN phase implements them.

_NOT_YET = "RED phase (plan 05-03 Task 3): not implemented yet"


def normalize_dist_name(name: str) -> str:
    raise NotImplementedError(_NOT_YET)


def parse_lerobot_requirement(dependencies):
    raise NotImplementedError(_NOT_YET)


def forbidden_direct_dependencies(dependencies):
    raise NotImplementedError(_NOT_YET)


def forbidden_resolved_packages(lock_text: str):
    raise NotImplementedError(_NOT_YET)


def read_lock_text(path):
    raise NotImplementedError(_NOT_YET)


def client_dependencies():
    raise NotImplementedError(_NOT_YET)


LEROBOT_EXTRAS_ALLOWLIST = frozenset()
FORBIDDEN_DIRECT_DEPENDENCIES = frozenset()
FORBIDDEN_RESOLVED_PACKAGES = frozenset()
NVIDIA_CUDA_PREFIX = "nvidia-cuda"


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

    Wire payloads are version-coupled from Phase 6 onward, so an exact pin is the
    precondition for the Phase 6 lockstep test.
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

    # diffusers is NOT in the set yet: it is in the current closure via the
    # incumbent lerobot pin and only leaves when the bump lands (plan 05-04).
    assert "diffusers" not in FORBIDDEN_RESOLVED_PACKAGES


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
    """The client interpreter floor stays >=3.12 (server is Py3.10 in-container)."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12"' in text
