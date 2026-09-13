#!/usr/bin/env python3
'Standalone hardware-free mock LeRobot async-inference server (gRPC on :8080).'

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
    'The four-method wire, and nothing else.'

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.last_specs = None
        self.last_observation = None
        # Per-method CALL COUNTS, not just the last payload. The two handshake
        # halves cost wildly different things on the real server — ``Ready`` flushes
        self.ready_calls = 0
        self.handshake_calls = 0
        # Injection point for a NON-UNIFORM action vector, so a test can prove
        # the client's flat->named reindex maps dim i to joint i rather than
        # merely producing a six-wide dict. None means "zeros", the default
        # stance of both mocks. Deliberately NOT a fourth MOCK_MODES entry: the
        # modes describe wire-level failure shapes, and this is a payload value.
        self.action_vector: list[float] | None = None

    def Ready(self, request, context):  # noqa: N802 - upstream's generated name
        """``Empty -> Empty``. Upstream also clears its shutdown event here."""
        self.ready_calls += 1
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """``PolicySetup -> Empty``. ``request.data`` is a pickled RemotePolicyConfig."""
        # Counted BEFORE the refuse branch: on the real server the weight load
        # happens inside this handler, so "the handler was entered" is the fact that
        # costs VRAM, whether or not it goes on to succeed.
        self.handshake_calls += 1
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
        '``stream Observation -> Empty``, reassembling the chunked payload.'
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
    'Start a mock AsyncInference server on ``host:port`` and RETURN the handle.'
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
    'Blocking wrapper: start, then wait for termination.'
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
