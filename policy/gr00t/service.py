# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GR00T policy-client transport (client side only).

This module is the Dum-E client half of the Isaac-GR00T ``n1.7-release`` ZMQ
wire contract. The inference *server* runs upstream as-is
(``run_gr00t_server.py`` / ``gr00t.policy.server_client.PolicyServer`` on
``tcp://host:5555``); Dum-E never runs the server in-process, so this module
intentionally does NOT depend on the server-only ``gr00t`` package. It mirrors
the *bytes on the wire* — not the upstream import graph — so it can run inside
the lightweight client ``uv`` environment where ``gr00t`` is not installed.

==================== PINNED :5555 WIRE CONTRACT (D-04) ====================
This client MUST mirror upstream ``gr00t/policy/server_client.py`` @
``n1.7-release`` byte-for-byte on the wire. Any change here is a contract
change and must break a test in ``tests/test_gr00t_service.py``.

* Serialization: msgpack with a custom ndarray hook.
  - encode ndarray -> ``{"__ndarray_class__": True, "as_npy": np.save(BytesIO, arr, allow_pickle=False)}``
  - decode ``{"__ndarray_class__": ...}`` -> ``np.load(BytesIO(as_npy), allow_pickle=False)``
    (``allow_pickle=False`` closes the arbitrary-code-execution hole — a hostile
    server cannot smuggle a pickle in ``as_npy``; matches upstream.)
* Request shape: ``{"endpoint": <name>}``; if the endpoint requires input,
  add ``"data": <dict>``; if an api_token is configured, add ``"api_token"``.
* ``get_action`` request data: ``{"observation": <obs>, "options": <opts|None>}``.
* Reply: msgpack-encoded list ``[action_chunk, info]`` decoded and returned as a
  ``(action_chunk, info)`` tuple. ``action_chunk`` is a dict keyed by modality
  (``single_arm`` / ``gripper``) with arrays of shape ``(B=1, T, D)``.
* Error sentinel: a raw ``b"ERROR"`` reply, or a decoded ``{"error": ...}`` dict,
  raises ``RuntimeError``.

==================== TRANSPORT HARDENING ====================
* ``_init_socket`` closes the previous REQ socket with ``LINGER=0`` before
  recreating it, and applies ``RCVTIMEO``/``SNDTIMEO``/``LINGER=0`` BEFORE
  ``connect`` so a recv against a dead server cannot block forever (D-05).
* ``call_endpoint`` retries a bounded number of times (``max_retries``,
  default 3), tearing down + recreating the strict-FSM REQ socket on each
  ``zmq.error.ZMQError`` (which includes ``zmq.error.Again`` timeouts), then
  raises a descriptive ``RuntimeError`` naming ``host:port`` (D-06). Worst-case
  total wait is ``timeout_ms * max_retries`` (~45s with the 15s default),
  comfortably inside Phase 1's long-running ``function_call_timeout_secs``.
* ``get_action`` defensively validates the decoded action chunk (type / required
  modality keys / ndim) and raises a descriptive ``RuntimeError`` instead of
  letting a bare ``KeyError``/``IndexError`` surface deep in the robot action
  loop (D-04).
"""

from io import BytesIO
from typing import Any

import msgpack
import numpy as np
import zmq


class MsgSerializer:
    """msgpack (de)serializer mirroring upstream GR00T ``MsgSerializer``.

    Preserves the exact upstream ndarray encoding so the bytes are identical to
    what ``gr00t/policy/server_client.py`` produces/consumes. The
    ``__ModalityConfig_class__`` decode branch is kept for wire compatibility,
    but — because the client does not depend on the server-only ``gr00t``
    package — a ``ModalityConfig`` payload is decoded to the plain dict under
    ``as_json`` rather than a ``ModalityConfig`` instance. The client's
    ``get_action`` path never receives ``ModalityConfig`` objects, so this is
    inert in practice and only matters if ``get_modality_config`` were called.
    """

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            # Client has no server-side ModalityConfig class; return the raw
            # json dict. (get_action never carries this; kept for contract parity.)
            return obj["as_json"]
        if "__ndarray_class__" in obj:
            # allow_pickle=False: a malicious pickle embedded in `as_npy` cannot
            # execute code. Matches upstream server_client.py (security fix, T-03-SC).
            return np.load(BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = BytesIO()
            # allow_pickle=False on save too — never serialize object arrays via pickle.
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class BaseInferenceClient:
    """ZMQ REQ transport to the upstream GR00T policy server.

    Owns the strict-FSM REQ socket lifecycle (close + recreate on error) and the
    bounded-retry-then-raise behaviour. Subclasses add typed endpoint helpers.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token
        self._init_socket()

    def _init_socket(self):
        """(Re)initialize the REQ socket, closing the previous one first.

        A ZMQ REQ socket is a strict send->recv lock-step state machine; after a
        timeout/error it is unusable and MUST be discarded. Close the previous
        socket with ``LINGER=0`` (discard queued msgs, return immediately — never
        block flushing to a dead peer), then create a fresh socket and apply the
        recv/send timeouts and ``LINGER=0`` BEFORE ``connect`` so the very next
        ``recv`` is already bounded (D-05 / Pitfall 3/4).
        """
        old = getattr(self, "socket", None)
        if old is not None:
            old.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except (zmq.error.ZMQError, RuntimeError):
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """Kill the server."""
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self,
        endpoint: str,
        data: dict | None = None,
        requires_input: bool = True,
        max_retries: int = 3,
    ) -> Any:
        """Call an endpoint on the server with bounded retry-then-raise.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
            max_retries: How many times to (re)try before raising. Each attempt
                is bounded by ``RCVTIMEO``/``SNDTIMEO`` (``timeout_ms``), so the
                worst-case total wait is ``timeout_ms * max_retries``. No sleep
                between attempts — the socket timeout already paces the loop.

        Raises:
            RuntimeError: on a server-side error reply, an ``b"ERROR"`` sentinel,
                or after ``max_retries`` transport failures (unreachable server).
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token

        last_err: Exception | None = None
        for _attempt in range(max_retries):
            try:
                self.socket.send(MsgSerializer.to_bytes(request))
                message = self.socket.recv()
            except zmq.error.ZMQError as exc:  # includes zmq.error.Again (timeout)
                # REQ socket is now mid-FSM and unusable; close + recreate before
                # the next attempt (strict-REQ teardown, D-05).
                last_err = exc
                self._init_socket()
                continue

            if message == b"ERROR":
                raise RuntimeError(
                    "Server error. Make sure we are running the correct policy server."
                )
            response = MsgSerializer.from_bytes(message)
            if isinstance(response, dict) and "error" in response:
                raise RuntimeError(f"Server error: {response['error']}")
            return response

        raise RuntimeError(
            f"Policy server unreachable at {self.host}:{self.port} "
            f"after {max_retries} attempts"
        ) from last_err

    def __del__(self):
        """Best-effort cleanup of socket + context on destruction."""
        socket = getattr(self, "socket", None)
        if socket is not None:
            try:
                socket.close(linger=0)
            except Exception:
                pass
        context = getattr(self, "context", None)
        if context is not None:
            try:
                context.term()
            except Exception:
                pass


class ExternalRobotInferenceClient(BaseInferenceClient):
    """Client for the upstream GR00T policy server's ``get_action`` endpoint.

    Mirrors upstream ``PolicyClient._get_action``: sends
    ``{"observation": observation, "options": options}`` and returns the
    ``(action_chunk, info)`` tuple — with defensive validation of the action
    chunk before it reaches the robot action loop (D-04).
    """

    #: Modality keys the SO101 fruit-picking checkpoint returns in the action chunk.
    DEFAULT_MODALITY_KEYS = ("single_arm", "gripper")

    @staticmethod
    def _validate_action_chunk(action_chunk: Any, modality_keys) -> None:
        """Raise a descriptive RuntimeError on contract drift in the action chunk.

        Guards against a malformed/version-mismatched server reply producing a
        bare ``KeyError``/``IndexError`` deep in the action loop (D-04 / ASVS V5).
        """
        if not isinstance(action_chunk, dict):
            raise RuntimeError(
                f"Policy returned {type(action_chunk).__name__}, expected a dict of "
                f"action keys. Wire contract drift — check the server is N1.7 and "
                f"msgpack-encoded."
            )
        for key in modality_keys:
            if key not in action_chunk:
                raise RuntimeError(
                    f"Action chunk missing '{key}' (got {list(action_chunk.keys())}). "
                    f"Embodiment-tag/modality mismatch — server must run NEW_EMBODIMENT."
                )
            arr = action_chunk[key]
            if getattr(arr, "ndim", None) != 3:
                raise RuntimeError(
                    f"Action '{key}' has shape {getattr(arr, 'shape', None)}, "
                    f"expected (B=1, T, D)."
                )

    def get_action(
        self,
        observation: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get an action chunk from the server.

        Sends ``{"observation": observation, "options": options}`` and returns
        ``(action_chunk, info)``. The exact observation schema is defined by the
        policy/checkpoint modality config (see controller).
        """
        response = self.call_endpoint(
            "get_action", {"observation": observation, "options": options}
        )
        # Upstream replies with a list [action_chunk, info]; msgpack decodes it to
        # a list — convert to the documented (action_chunk, info) tuple.
        if not isinstance(response, (list, tuple)) or len(response) != 2:
            raise RuntimeError(
                f"Policy reply has shape {type(response).__name__} "
                f"(len {len(response) if hasattr(response, '__len__') else 'n/a'}), "
                f"expected a 2-element (action_chunk, info). Wire contract drift."
            )
        action_chunk, info = response[0], response[1]
        self._validate_action_chunk(action_chunk, self.DEFAULT_MODALITY_KEYS)
        return action_chunk, info
