"""LeRobot async-inference gRPC session (client side only).

This module is the Dum-E client half of LeRobot 0.6.1's
``transport.AsyncInference`` gRPC contract. The inference *server* runs inside
the ``lerobot-policy`` container (``docker/lerobot-policy/``); the Dum-E venv
never loads policy weights in-process. Unlike ``policy/gr00t/service.py`` — which
deliberately mirrors the bytes on the wire and refuses to import the server
package — this module DOES import ``lerobot.transport``, because the generated
stubs and the chunked-transfer helpers are the wire, and reimplementing them
would guarantee drift.

==================== WHY THIS SEAM (D-09) ====================
Three candidate seams existed; this module is candidate 1, composition against
lerobot's generated stubs behind a thin session object. Recorded here so a later
reader does not re-open it:

* Candidate 2, ``lerobot.async_inference.robot_client.RobotClient``, is
  DISQUALIFIED: its ``__init__`` calls ``make_robot_from_config`` +
  ``robot.connect()``, i.e. it takes ownership of the robot and the serial bus.
  PROJECT.md forbids that outright and ASY-03 restates it — nothing built in this
  phase may take robot ownership. ``embodiment/so_arm10x/controller.py`` keeps it.
* Candidate 3, our own minimal unary servicer in the container, is DISQUALIFIED:
  the wire proven here would then not be the wire Phase 8's async inference uses,
  re-entangling the plumbing and async axes that the 5->6->7->8 chain exists to
  keep separately bisectable.

Upstream symbols touched, all import-and-call only (no patching, no vendoring):
``lerobot.transport.services_pb2_grpc.AsyncInferenceStub``,
``lerobot.transport.services_pb2.{Empty, PolicySetup, Observation, Actions}``,
``lerobot.transport.utils.{send_bytes_in_chunks, grpc_channel_options}``,
``lerobot.async_inference.helpers.{RemotePolicyConfig, TimedObservation}``.

==================== PINNED transport.AsyncInference WIRE CONTRACT ====================
This client MUST mirror ``lerobot.transport.services_pb2_grpc`` @ ``0.6.1``. Any
change here is a contract change and must break a test in
``tests/test_lerobot_upstream_surface.py``.

* ``Ready``                  ``Empty       -> Empty``   (unary)
* ``SendPolicyInstructions`` ``PolicySetup -> Empty``   (unary)
* ``SendObservations``       ``Observation -> Empty``   (CLIENT-streaming)
* ``GetActions``             ``Empty       -> Actions`` (unary)

* Serialization: ``pickle`` in BOTH directions.
  - ``PolicySetup.data``  = ``pickle.dumps(RemotePolicyConfig)``
  - ``Observation.data``  = ``pickle.dumps(TimedObservation)``, split across
    messages by ``send_bytes_in_chunks`` (the TRANSFER_BEGIN/MIDDLE/END state
    machine the server's ``receive_bytes_in_chunks`` requires — a wrong first
    state raises ``ValueError`` server-side).
  - ``Actions.data``      = ``pickle.dumps(list[TimedAction])``.
* Handshake ORDER is part of the contract: ``Ready`` then
  ``SendPolicyInstructions``. See ``connect``.
* Every ``TimedObservation`` carries ``must_go=True``. See ``infer``.
* ZERO-LENGTH ``Actions.data`` is a SERVER-SIDE FAILURE, not an empty result. See
  ``infer``.

==================== DELIBERATE DIVERGENCE: pickle, not allow_pickle=False ====================
``policy/gr00t/service.py`` hardens its wire with
``np.load(..., allow_pickle=False)`` and says so — a hostile server there cannot
smuggle executable code into a reply. THIS WIRE CANNOT BE HARDENED THAT WAY. It
is ``pickle`` in both directions by upstream's design (``policy_server.py:125``
server-side handshake, ``:183`` server-side observations, ``:236`` client-bound
actions; upstream marks all three ``# nosec``), so anyone who can reach the port
can execute arbitrary code in this process.

The sole mitigation is the loopback publish spec: the container is published
``-p 127.0.0.1:8080:8080``, never ``--network host``, and that spec is asserted
by a test (plan 06-05) rather than left resting on a default. Do NOT "restore
parity" with ``policy/gr00t/service.py`` by attempting a restricted unpickler
here: the payload is arbitrary dataclasses holding torch tensors, so a safe
unpickler would break the wire rather than harden it.

==================== TRANSPORT HARDENING ====================
* Every stub call passes an explicit ``timeout=self._deadline_s`` and nothing
  else. ``wait_for_ready`` is NEVER passed — grpc-python defaults it to False,
  and that default is exactly what makes "an unreachable server raises instead of
  hanging" true. Setting it True would silently convert an unreachable server
  into an indefinite park.
* ``_call`` retries a bounded number of times (``MAX_ATTEMPTS``, default 3) on
  TRANSIENT statuses only, then raises a descriptive ``RuntimeError`` naming
  ``host:port``. Worst-case total wait is ``DEADLINE_S * MAX_ATTEMPTS`` (~45s),
  matching ``policy/gr00t/service.py``'s budget exactly and comfortably inside
  the agent's long-running ``function_call_timeout_secs``.
* A NON-transient status is raised immediately with the server's own
  ``code()``/``details()`` — never retried, never reworded into "unreachable".
* ``infer`` validates ``len(reply.data)`` before ``pickle.loads`` and raises a
  descriptive ``RuntimeError`` instead of letting a bare ``EOFError`` surface.
"""

import pickle  # nosec B403 - the wire is pickle by upstream design; see DELIBERATE DIVERGENCE
import time

import grpc
from loguru import logger

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks

#: Per-call gRPC deadline, in seconds. Matches ``policy/gr00t/service.py``'s
#: ``timeout_ms=15000`` exactly (D-10) so the two backends fail on the same
#: budget and an operator does not have to remember which is which.
DEADLINE_S: float = 15.0

#: Bounded attempt count. Matches ``policy/gr00t/service.py``'s ``max_retries=3``
#: exactly (D-10). Worst-case total wait is ``DEADLINE_S * MAX_ATTEMPTS`` = ~45s.
MAX_ATTEMPTS: int = 3

#: The HANDSHAKE deadline, which is deliberately NOT ``DEADLINE_S``, and is used
#: for exactly one call: ``SendPolicyInstructions``.
#:
#: D-10 pins the per-inference budget to ``groot-native``'s 15s/3-attempt shape,
#: and ``Ready``/``SendObservations``/``GetActions`` all keep it. But
#: ``SendPolicyInstructions`` is not an inference call: upstream loads the policy
#: INSIDE that request handler — ``GrootPolicy.from_pretrained(...)`` on EVERY
#: handshake (``policy_server.py:151``) — so the call blocks for one full ~12.6 GB
#: weight load, minutes of wall clock on an RTX 3060. A 15s deadline on it is not
#: a strict budget, it is a LOOP: the deadline expires while the server is still
#: loading, ``DEADLINE_EXCEEDED`` is (correctly) retryable, and because gRPC
#: server handlers are not cancelled by a client-side deadline, each retry starts
#: ANOTHER concurrent load on a 4-worker thread pool. Three overlapping loads of a
#: 3B-parameter model cannot fit in 12288 MiB, so the 15s deadline manufactures
#: the exact OOM it looks like it is protecting against.
#:
#: 900s is a fail-loud ceiling, not an expectation: a handshake that has not
#: returned in 15 minutes is wedged, not slow.
HANDSHAKE_DEADLINE_S: float = 900.0

#: Attempts for the handshake: exactly ONE, for the same reason. A retried
#: handshake does not re-ask a finished question, it starts a second full weight
#: load alongside the first. If the handshake fails, the operator needs the
#: server's own message and a fresh container — not another 12.6 GB allocation on
#: a GPU that is already holding one.
HANDSHAKE_MAX_ATTEMPTS: int = 1

#: The ONLY statuses worth retrying: a server that is not up yet, and a call that
#: outran its deadline. Everything else is a decision the server already made.
#:
#: This matters concretely and is not defensive boilerplate: the SAFE-01 guard
#: (plan 06-03) aborts a violating handshake with ``FAILED_PRECONDITION`` and a
#: specific message naming which assertion failed. Retrying that three times
#: would burn the whole budget and then bury the server's own diagnosis under a
#: generic "unreachable" message — the operator would debug the network instead
#: of the checkpoint.
RETRYABLE_CODES: tuple[grpc.StatusCode, ...] = (
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
)


class LeRobotPolicySession:
    """A thin, synchronous gRPC session over LeRobot's async-inference service.

    Every member is deliberately SYNCHRONOUS. Concurrency is handled at the
    ``@tool`` boundary in ``embodiment/so_arm10x/agent.py`` via
    ``await asyncio.to_thread(sync_method, *args)``; an ``async def`` here would
    break that offload pattern and every existing call site.

    This class owns the channel and the wire only. The ``IPolicyBackend``
    adapter — instruction storage, the flat->named action reindex, ``reset()`` —
    is a separate object (``policy/lerobot/backend.py``, plan 06-04) so that the
    transport can be tested without the contract and vice versa.
    """

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
        # pickled) live inside, and it is the same helper the upstream client
        # uses, so the two ends cannot disagree about limits.
        #
        # enable_retries=False: we own the bounded retry so the budget matches
        # groot-native exactly (D-10), instead of upstream's maxAttempts=5 with
        # exponential backoff. Two retry layers would multiply, not add.
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
        """Perform the two-step handshake: ``Ready``, then ``SendPolicyInstructions``.

        The ORDER IS MANDATORY, not stylistic. ``SendPolicyInstructions`` returns
        early — accepting the message and doing nothing — when ``not
        self.running`` (``policy_server.py:119-121``), and ``shutdown_event`` is
        cleared ONLY by ``Ready`` (``policy_server.py:108-114``). Reversing the
        order therefore gets the handshake silently ignored: the server answers
        OK, no policy is ever loaded, and the first ``GetActions`` fails with a
        zero-length payload instead of naming the real cause.

        Note this call is EXPENSIVE: upstream calls
        ``GrootPolicy.from_pretrained(...)`` on every handshake rather than once
        per container lifetime, so each ``connect()`` is one full ~12.6 GB weight
        load (minutes of wall clock on an RTX 3060).
        """
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
        """Send ``Ready`` and NOTHING else — so no weight load can be triggered.

        This is the CHEAP half of :meth:`connect`, split out because the two halves
        cost wildly different things and only one of them is idempotent in practice:

        * ``Ready`` calls upstream's ``_reset_server()``
          (``policy_server.py:107-113``), which clears ``observation_queue`` and
          ``_predicted_timesteps`` and leaves the loaded ``policy``,
          ``preprocessor`` and ``postprocessor`` untouched. That is the ONLY
          per-client server state a client can flush.
        * ``SendPolicyInstructions`` is what reloads the weights
          (``policy_server.py:151``) — one full multi-GB materialization per call.

        RAISES rather than returning a bool, deliberately unlike :meth:`ready`.
        The caller that needs this is ``LeRobotPolicyBackend.reset()`` at an
        episode boundary, and a reset that quietly did nothing is exactly how
        stale server-side per-client state survives into the next episode.

        Raises:
            RuntimeError: on a non-transient gRPC status, or an unreachable server
                after ``MAX_ATTEMPTS`` attempts.
        """
        self._call(lambda: self._stub.Ready(services_pb2.Empty(), timeout=self._deadline_s))

    def ready(self) -> bool:
        """Return True if the server answered ``Ready``, False on any gRPC error.

        Mirrors ``BaseInferenceClient.ping``'s contract: a reachability probe
        returns a bool rather than raising, so a caller can poll it. Delegates to
        :meth:`probe_ready_or_raise` so there is ONE ``Ready`` call site here, and
        the polling and fail-loud contracts cannot drift apart.
        """
        try:
            self.probe_ready_or_raise()
            return True
        except (grpc.RpcError, RuntimeError):
            return False

    def infer(self, raw_observation: dict) -> list:
        """Send one observation and return the decoded action chunk.

        Args:
            raw_observation: The flat robot observation — six ``"<joint>.pos"``
                floats, one ``(H, W, 3)`` uint8 frame per camera key, and a
                ``"task"`` string. ``"task"`` is the LeRobot-side language key
                (``processor_groot.py:1547 language_key = "task"``); omitting it
                is NOT an error but silently substitutes the default prompt
                ``"Perform the task."``
                (``lerobot/policies/groot/utils.py:239-241``), which is a silent
                degradation of policy quality, so callers must supply it.

        Returns:
            ``list[TimedAction]`` — for this checkpoint, 16 actions of shape (6,).

        Raises:
            RuntimeError: on a zero-length payload, a non-transient gRPC status,
                or an unreachable server after ``MAX_ATTEMPTS`` attempts.
        """
        obs = TimedObservation(
            timestamp=time.time(),
            timestep=self._timestep,
            observation=raw_observation,
            # must_go=True on EVERY observation is MANDATORY. Two independent
            # server-side filters otherwise drop it silently
            # (policy_server.py:268-310): one skips a timestep already predicted,
            # and observations_similar skips anything within an L2 joint-space
            # tolerance of the previous observation -- with atol=1 BY DEFAULT
            # (helpers.py:281-283), which in joint space means a deliberate
            # replay of the same observation is exactly the case it filters.
            # _enqueue_observation short-circuits on obs.must_go.
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
            # services_pb2.Empty() from a method DECLARED to return Actions
            # (policy_server.py:259-266). protobuf serializes that mismatch to
            # b'', so the client sees a SUCCESSFUL RPC carrying zero bytes.
            # Unpickling it would raise `EOFError: Ran out of input` from deep
            # inside the transport, naming nothing useful.
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

    def _call(self, thunk, max_attempts: int | None = None):
        """Bounded retry on transient statuses, immediate raise on anything else.

        No sleep between attempts: the per-call deadline already paces the loop,
        exactly as ``policy/gr00t/service.py``'s socket timeout does.

        Args:
            thunk: The stub call to make. It carries its own ``timeout=``.
            max_attempts: Override the attempt budget. ``connect`` passes 1 —
                retrying a handshake starts a second concurrent weight load rather
                than re-asking a finished question.
        """
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
