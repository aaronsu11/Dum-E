#!/usr/bin/env python3
"""Standalone hardware-free mock LeRobot async-inference server (gRPC on :8080).

Speaks the EXACT four-method ``transport.AsyncInference`` contract that
``lerobot.async_inference.policy_server`` serves, with NO GPU, NO model, and NO
checkpoint. This is the gRPC sibling of ``scripts/mock_policy_server.py`` (the
``zmq.REP`` mock for the ``groot-native`` backend). Reusable two ways:

  * imported by ``tests/test_lerobot_backend.py`` and
    ``tests/test_container_contract.py`` via ``start_mock(port)`` for an
    in-process real-socket round trip (the container-boundary probe);
  * run standalone by an operator to exercise a client against a fake server.

It mirrors the bytes on the wire (the FIXED transport boundary), not policy
quality: ``GetActions`` always returns zeros of the correct shape.

Usage:
    uv run python scripts/mock_grpc_policy_server.py                 # bind 127.0.0.1:8080
    uv run python scripts/mock_grpc_policy_server.py --port 8081     # custom port
    uv run python scripts/mock_grpc_policy_server.py --mode empty    # inject a failure

Contract served (see policy/lerobot/session.py's PINNED WIRE CONTRACT block):
    Ready(Empty)                        -> Empty
    SendPolicyInstructions(PolicySetup) -> Empty
        PolicySetup.data = pickle.dumps(RemotePolicyConfig)
    SendObservations(stream Observation) -> Empty
        Observation.data = pickle.dumps(TimedObservation), split across messages
        by send_bytes_in_chunks' TRANSFER_BEGIN/MIDDLE/END state machine
    GetActions(Empty)                   -> Actions
        Actions.data = pickle.dumps(list[TimedAction]), 16 actions of shape (6,)

Injectable failure modes (``MOCK_MODES``) exist so the client's guards are
PROVABLE rather than merely written:
    "ok"      -> the contract above
    "empty"   -> GetActions returns Actions(data=b"") (the zero-length path)
    "refuse"  -> SendPolicyInstructions aborts FAILED_PRECONDITION (the
                 non-retryable path a SAFE-01 guard refusal takes)
"""

import argparse
import os
import pickle  # nosec B403 - the wire is pickle by upstream design; see policy/lerobot/session.py
import sys
from concurrent import futures

import grpc
import torch

# Ensure the repo root is importable regardless of CWD (the scripts/ dir would
# otherwise shadow the repo root on sys.path[0]).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lerobot.async_inference.helpers import TimedAction  # noqa: E402
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402

# N1.7 action horizon for the SO101 fruit-picking checkpoint (T=16). Kept as a
# named module constant, matching the ZMQ sibling's ``_ACTION_HORIZON``.
_ACTION_HORIZON = 16

# Six joints: dims 0:5 are the ``single_arm`` group, dim 5 is ``gripper``.
_ACTION_DIM = 6

#: The injectable behaviours. An unknown value is rejected by ``start_mock``.
MOCK_MODES = ("ok", "empty", "refuse")


class MockAsyncInferenceServicer(services_pb2_grpc.AsyncInferenceServicer):
    """The four-method wire, and nothing else.

    Records the last handshake and the last observation on the instance so a
    test can assert what the CLIENT sent without reaching for a global. The
    started server exposes this object as ``server.dume_servicer``.
    """

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.last_specs = None
        self.last_observation = None
        # Injection point for a NON-UNIFORM action vector, so a test can prove
        # the client's flat->named reindex maps dim i to joint i rather than
        # merely producing a six-wide dict. None means "zeros", the default
        # stance of both mocks. Deliberately NOT a fourth MOCK_MODES entry: the
        # modes describe wire-level failure shapes, and this is a payload value.
        self.action_vector: list[float] | None = None

    def Ready(self, request, context):  # noqa: N802 - upstream's generated name
        """``Empty -> Empty``. Upstream also clears its shutdown event here."""
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """``PolicySetup -> Empty``. ``request.data`` is a pickled RemotePolicyConfig."""
        if self.mode == "refuse":
            # The shape a SAFE-01 guard refusal takes on the wire: an ABORT with
            # a specific, actionable message. FAILED_PRECONDITION is deliberately
            # absent from the client's RETRYABLE_CODES, so this must surface in
            # ONE attempt with the server's own text intact — three retries would
            # bury the diagnosis under a generic "unreachable".
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "SAFE-01/2 configured actions_per_chunk=40 disagrees with the "
                "checkpoint's delta_indices (16). D-11 config drift.",
            )
        # Recorded so a test can assert what the CLIENT sent (policy_type,
        # checkpoint path, actions_per_chunk, the lerobot_features ordering)
        # rather than assuming it.
        self.last_specs = pickle.loads(request.data)  # nosec B301 - see module docstring
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """``stream Observation -> Empty``, reassembling the chunked payload.

        The client sends via ``send_bytes_in_chunks``, so the server MUST consume
        every message in the iterator: a client-streaming handler that returns
        early leaves the client blocked on an unread stream.
        """
        buffer = bytearray()
        for item in request_iterator:
            if item.transfer_state == services_pb2.TRANSFER_BEGIN:
                buffer = bytearray(item.data)
            else:
                buffer.extend(item.data)
        self.last_observation = pickle.loads(bytes(buffer))  # nosec B301
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """``Empty -> Actions{data: pickled list[TimedAction]}``, zeros by default."""
        if self.mode == "empty":
            # NOT a hypothetical. Upstream's GetActions wraps its whole body in a
            # blanket `except Exception` and returns services_pb2.Empty() from a
            # method DECLARED to return Actions (policy_server.py:214-266). protobuf
            # serializes that mismatch to b'', so the client sees a SUCCESSFUL RPC
            # carrying zero bytes. This mode is what proves the client's
            # zero-length check has teeth instead of an EOFError out of pickle.
            return services_pb2.Actions(data=b"")

        vector = self.action_vector
        actions = [
            TimedAction(
                # Keyword arguments, NOT positional: TimedAction inherits
                # TimedData, so the field order is (timestamp, timestep, action).
                # Constructing it positionally in the wrong order would produce a
                # mock that passes while the real server's replies fail.
                timestamp=float(i),
                timestep=i,
                action=(
                    torch.zeros(_ACTION_DIM)
                    if vector is None
                    else torch.tensor(vector, dtype=torch.float32)
                ),
            )
            for i in range(_ACTION_HORIZON)
        ]
        return services_pb2.Actions(data=pickle.dumps(actions))


def start_mock(port: int, host: str = "127.0.0.1", mode: str = "ok") -> grpc.Server:
    """Start a mock AsyncInference server on ``host:port`` and RETURN the handle.

    DELIBERATE DIVERGENCE from ``scripts/mock_policy_server.py``: the ZMQ
    sibling is stopped by a ``"kill"`` endpoint because a ``zmq.REP`` loop has no
    other exit, so its teardown had to become part of its wire. This mock returns
    the ``grpc.Server`` and callers stop it with ``server.stop(grace=0)``, which
    is gRPC's first-class equivalent of the ZMQ mock's ``LINGER=0`` teardown. Do
    NOT "restore parity" by adding a ``"kill"``-style sentinel RPC: that would add
    a FIFTH method to the four-method wire this phase exists to pin, and the mock
    would stop being the contract.

    Args:
        port: TCP port to bind. Tests pass an ephemeral port; never hard-code.
        host: Bind host. Defaults to loopback (see ``main``'s ``--host``).
        mode: One of ``MOCK_MODES``.

    Returns:
        The started ``grpc.Server``, with the servicer instance attached as
        ``server.dume_servicer``.

    Raises:
        ValueError: on an unknown ``mode``, naming the value and the allowed set
            (the ``policy/factory.py`` fail-closed idiom).
    """
    if mode not in MOCK_MODES:
        raise ValueError(
            f"mode={mode!r} is not a known mock mode; allowed values are "
            f"{', '.join(MOCK_MODES)}."
        )

    servicer = MockAsyncInferenceServicer(mode=mode)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(servicer, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    # Attached rather than returned as a tuple so the stop handle stays the
    # primary return value and the fixture teardown reads as one call.
    server.dume_servicer = servicer
    return server


def serve_mock(port: int, host: str = "127.0.0.1", mode: str = "ok") -> None:
    """Blocking wrapper: start, then wait for termination.

    Keeps the operator-facing two-positional-argument shape of
    ``scripts/mock_policy_server.py``'s ``serve_mock(port, host)``. The
    ``try/finally`` means a ``KeyboardInterrupt`` cannot leave a bound port.
    """
    server = start_mock(port, host=host, mode=mode)
    try:
        server.wait_for_termination()
    finally:
        server.stop(grace=0)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # DELIBERATE DIVERGENCE from the ZMQ sibling's ``*`` default: this wire is
    # pickle in BOTH directions by upstream design, so anyone who can reach the
    # port can execute arbitrary code in the peer process. Loopback is the ONLY
    # mitigation this transport has, so the standalone default must not bind
    # every interface.
    ap.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host/interface (default: 127.0.0.1 = loopback only).",
    )
    ap.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080).")
    ap.add_argument(
        "--mode",
        default="ok",
        choices=MOCK_MODES,
        help="Injected behaviour (default: ok).",
    )
    args = ap.parse_args()

    print(
        f"[mock_grpc_policy_server] serving transport.AsyncInference on "
        f"{args.host}:{args.port} mode={args.mode} (Ctrl-C to stop)",
        file=sys.stderr,
    )
    try:
        serve_mock(args.port, host=args.host, mode=args.mode)
    except KeyboardInterrupt:
        print("\n[mock_grpc_policy_server] stopped.", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
