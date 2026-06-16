#!/usr/bin/env python3
"""Standalone hardware-free mock GR00T policy server (zmq.REP echo on :5555).

Speaks the EXACT ``MsgSerializer`` :5555 wire contract that the real Isaac-GR00T
n1.7-release inference server emits, with NO GPU, NO model, and NO checkpoint.
Reusable two ways:

  * imported by ``tests/test_container_contract.py`` via ``serve_mock(port)`` for
    an in-process real-socket round trip (the container-boundary probe, DOCK-05);
  * run standalone by an operator to exercise a client against a fake server.

It mirrors the bytes on the wire (the FIXED Phase 3 boundary), not policy
quality: ``get_action`` always returns zeros of the correct shape/dtype.

Usage:
    uv run python scripts/mock_policy_server.py                 # bind 0.0.0.0:5555
    uv run python scripts/mock_policy_server.py --port 5556     # custom port
    # then, in another shell:
    uv run python scripts/test_live_policy_server.py --port 5556 --skip-unreachable

Contract served (see policy/gr00t/service.py):
    request  {"endpoint": <name>, "data"?: {...}, "api_token"?: <str>}
    ping        -> {"status": "ok", "message": "Server is running"}
    get_action  -> [action_chunk, info]
                   action_chunk = {"single_arm": (1,16,5) f32, "gripper": (1,16,1) f32}
                   info = {}
    <other>     -> {"error": "Unknown endpoint: <ep>"}
"""

import argparse
import os
import sys

import numpy as np
import zmq

# Ensure the repo root is importable regardless of CWD (the scripts/ dir would
# otherwise shadow the repo root on sys.path[0]).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mirror the wire contract from Dum-E's client-side serializer — NOT the
# server-only upstream ``gr00t`` package (not installed in the client uv env).
from policy.gr00t.service import MsgSerializer  # noqa: E402

# N1.7 action horizon for the SO101 fruit-picking checkpoint (T=16).
_ACTION_HORIZON = 16


def _mock_action_reply() -> list:
    """Return the [action_chunk, info] reply matching the :5555 get_action contract.

    action_chunk is keyed by SO101 modality with (B=1, T=16, D) float32 arrays;
    info is an (empty) metadata dict. Values are zeros — this exercises the wire
    path/decode shapes, not policy quality.
    """
    action_chunk = {
        "single_arm": np.zeros((1, _ACTION_HORIZON, 5), dtype=np.float32),
        "gripper": np.zeros((1, _ACTION_HORIZON, 1), dtype=np.float32),
    }
    info: dict = {}
    return [action_chunk, info]


def serve_mock(port: int, host: str = "*") -> None:
    """Bind a zmq.REP socket on ``tcp://{host}:{port}`` and serve the contract.

    Loops forever decoding each request via ``MsgSerializer.from_bytes`` and
    replying with the matching contract bytes. Designed to run either in the
    foreground (operator) or in a daemon thread (the contract test); the caller
    owns the thread lifecycle. The zmq context/socket are torn down with
    LINGER=0 on any exit so a test thread cannot hang the interpreter.
    """
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://{host}:{port}")
    try:
        while True:
            raw = socket.recv()
            req = MsgSerializer.from_bytes(raw)
            endpoint = req.get("endpoint", "get_action") if isinstance(req, dict) else "get_action"

            if endpoint == "ping":
                reply: object = {"status": "ok", "message": "Server is running"}
            elif endpoint == "kill":
                # Acknowledge then stop serving so a manual operator can shut us down.
                socket.send(MsgSerializer.to_bytes({"status": "ok", "message": "killing"}))
                break
            elif endpoint == "get_action":
                reply = _mock_action_reply()
            else:
                reply = {"error": f"Unknown endpoint: {endpoint}"}

            socket.send(MsgSerializer.to_bytes(reply))
    finally:
        socket.close(linger=0)
        context.term()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="*",
                    help="Bind host/interface (default: * = all interfaces).")
    ap.add_argument("--port", type=int, default=5555,
                    help="Bind port (default: 5555).")
    args = ap.parse_args()

    print(f"[mock_policy_server] serving MsgSerializer contract on tcp://{args.host}:{args.port}"
          " (Ctrl-C to stop)", file=sys.stderr)
    try:
        serve_mock(args.port, host=args.host)
    except KeyboardInterrupt:
        print("\n[mock_policy_server] stopped.", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
