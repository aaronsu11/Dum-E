'LeRobot async-inference gRPC session (client side only).'

import pickle  # nosec B403 - the wire is pickle by upstream design; see DELIBERATE DIVERGENCE
import time

import grpc
from loguru import logger

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

#: Per-call gRPC deadline, in seconds. Matches ``policy/backends/isaac_groot/service.py``'s
#: ``timeout_ms=15000`` exactly (D-10) so the two backends fail on the same
#: budget and an operator does not have to remember which is which.
DEADLINE_S: float = 15.0

#: Bounded attempt count. Matches ``policy/backends/isaac_groot/service.py``'s ``max_retries=3``
#: exactly (D-10). Worst-case total wait is ``DEADLINE_S * MAX_ATTEMPTS`` = ~45s.
MAX_ATTEMPTS: int = 3

#: The HANDSHAKE deadline, which is deliberately NOT ``DEADLINE_S``, and is used
#: for exactly one call: ``SendPolicyInstructions``.
HANDSHAKE_DEADLINE_S: float = 900.0

#: Attempts for the handshake: exactly ONE, for the same reason. A retried
#: handshake does not re-ask a finished question, it starts a second full weight
#: load alongside the first. If the handshake fails, the operator needs the
#: server's own message and a fresh container — not another 12.6 GB allocation on
#: a GPU that is already holding one.
HANDSHAKE_MAX_ATTEMPTS: int = 1

#: The ONLY statuses worth retrying: a server that is not up yet, and a call that
#: outran its deadline. Everything else is a decision the server already made.
RETRYABLE_CODES: tuple[grpc.StatusCode, ...] = (
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
)


class LeRobotPolicySession:
    "A thin, synchronous gRPC session over LeRobot's async-inference service."

    def __init__(
        self,
        address: str,
        deadline_s: float = DEADLINE_S,
        max_attempts: int = MAX_ATTEMPTS,
        handshake_deadline_s: float = HANDSHAKE_DEADLINE_S,
        handshake_max_attempts: int = HANDSHAKE_MAX_ATTEMPTS,
    ) -> None:
        """Open a channel to ``address`` (``"host:port"``). Does not call the server."""
        self._address = address
        self._deadline_s = deadline_s
        self._max_attempts = max_attempts
        self._handshake_deadline_s = handshake_deadline_s
        self._handshake_max_attempts = handshake_max_attempts
        # grpc_channel_options rather than a hand-built options list: it sets the
        # 4 MB send/receive message limits our two 480x640 uint8 frames (~1.84 MB
        self._channel = grpc.insecure_channel(
            address, options=grpc_channel_options(enable_retries=False)
        )
        self._stub = services_pb2_grpc.AsyncInferenceStub(self._channel)
        self._timestep = 0
        self._closed = False

    @property
    def address(self) -> str:
        """The ``host:port`` this session talks to."""
        return self._address

    def connect(self, specs: RemotePolicyConfig) -> None:
        'Perform the two-step handshake: ``Ready``, then ``SendPolicyInstructions``.'
        self.probe_ready_or_raise()
        payload = pickle.dumps(specs)
        # HANDSHAKE_DEADLINE_S / HANDSHAKE_MAX_ATTEMPTS, NOT the per-inference
        # budget: the weight load happens inside this request handler, so a 15s
        # deadline with 3 retries would start three overlapping ~12.6 GB loads and
        # OOM the card. See those constants for the full reasoning.
        self._call(
            lambda: self._stub.SendPolicyInstructions(
                services_pb2.PolicySetup(data=payload), timeout=self._handshake_deadline_s
            ),
            max_attempts=self._handshake_max_attempts,
        )
        logger.info(
            "LeRobot policy handshake sent to {} | policy_type={} | path={} | "
            "actions_per_chunk={} | device={}",
            self._address,
            specs.policy_type,
            specs.pretrained_name_or_path,
            specs.actions_per_chunk,
            specs.device,
        )

    def probe_ready_or_raise(self) -> None:
        'Send ``Ready`` and NOTHING else — so no weight load can be triggered.'
        self._call(lambda: self._stub.Ready(services_pb2.Empty(), timeout=self._deadline_s))

    def ready(self) -> bool:
        'Return True if the server answered ``Ready``, False on any gRPC error.'
        try:
            self.probe_ready_or_raise()
            return True
        except (grpc.RpcError, RuntimeError):
            return False

    def infer(self, raw_observation: dict) -> list:
        'Send one observation and return the decoded action chunk.'
        obs = TimedObservation(
            timestamp=time.time(),
            timestep=self._timestep,
            observation=raw_observation,
            # must_go=True on EVERY observation is MANDATORY. Two independent
            # server-side filters otherwise drop it silently
            must_go=True,
        )
        # Monotonic timestep: the predicted-timesteps filter above is keyed on it,
        # so reusing a value silently drops the observation.
        self._timestep += 1

        self._call(
            lambda: self._stub.SendObservations(
                send_bytes_in_chunks(pickle.dumps(obs), services_pb2.Observation),
                timeout=self._deadline_s,
            )
        )
        reply = self._call(
            lambda: self._stub.GetActions(services_pb2.Empty(), timeout=self._deadline_s)
        )

        if not reply.data:
            # NOT a hypothetical, and NOT an "empty result": GetActions wraps its
            # whole body in a blanket `except Exception` and returns
            raise RuntimeError(
                f"LeRobot policy server at {self._address} returned a ZERO-LENGTH action "
                f"chunk for timestep {obs.timestep}. This arrives as a SUCCESSFUL RPC with "
                "zero-length data, so it is indistinguishable on the wire from a real reply "
                "except by length. Two causes, both server-side: (1) the server timed out "
                "waiting on its observation queue (obs_queue_timeout defaults to 2s), or "
                "(2) it raised during inference and swallowed the exception — check the "
                "container log for 'Error in StreamActions'. Refusing to unpickle zero bytes "
                "into a bare EOFError."
            )

        return pickle.loads(reply.data)  # nosec B301 - see DELIBERATE DIVERGENCE

    def close(self) -> None:
        """Close the channel. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self._channel.close()
        except Exception:  # noqa: BLE001 - best-effort teardown must never raise
            pass

    def configure_requests(self, *, deadline_s, max_attempts=1):
        """Set inference deadlines after warmup; handshake budgets stay separate."""
        import math
        if not math.isfinite(deadline_s) or deadline_s <= 0 or type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("Positive finite deadline and retry count required")
        self._deadline_s = deadline_s
        self._max_attempts = max_attempts

    def _call(self, thunk, max_attempts: int | None = None):
        'Bounded retry on transient statuses, immediate raise on anything else.'
        attempts = self._max_attempts if max_attempts is None else max_attempts
        last: grpc.RpcError | None = None
        for _attempt in range(attempts):
            try:
                return thunk()
            except grpc.RpcError as exc:
                if exc.code() not in RETRYABLE_CODES:
                    # The server made a decision (FAILED_PRECONDITION from the
                    # SAFE-01 guard, INVALID_ARGUMENT, INTERNAL, ...). Surface
                    # ITS message; retrying cannot change the answer.
                    raise RuntimeError(
                        f"LeRobot policy server at {self._address} rejected the call with "
                        f"gRPC status {exc.code()}: {exc.details()!r}. Not retried — a "
                        "non-transient status is the server's own diagnosis, and three "
                        "retries would only bury it under a generic 'unreachable'."
                    ) from exc
                last = exc

        raise RuntimeError(
            f"LeRobot policy server unreachable at {self._address} after "
            f"{attempts} attempts (last gRPC status: "
            f"{last.code() if last is not None else 'unknown'})"
        ) from last
