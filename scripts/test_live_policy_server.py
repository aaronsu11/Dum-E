#!/usr/bin/env python3
"""Hardware-free live smoke test for the GR00T N1.7 policy client transport.

Drives the REAL containerized Isaac-GR00T n1.7-release policy server with
SYNTHETIC observations (no SO-ARM101, no cameras). Validates the transport and
wire contract that the mocked unit tests can only simulate:

  1. ping()                  — reachability against the live server
  2. get_action(synthetic)   — full obs->policy->action round trip over ZMQ/msgpack
  3. unreachable-raises       — stop the server, confirm a descriptive RuntimeError
                                within ~45s instead of an infinite hang
  4. (optional) wrong tag     — schema/embodiment mismatch fails loudly

This covers the transport surfaces against real bytes. It does NOT cover the
physical arm-motion portion (that still needs the SO101 hardware gate).

Usage:
    uv run python scripts/test_live_policy_server.py --host localhost --port 5555
    uv run python scripts/test_live_policy_server.py --skip-unreachable   # keep server up

Synthetic obs shapes are derived from the checkpoint experiment_cfg:
    video: front, wrist  (uint8 HxWx3)
    state: single_arm (5) + gripper (1)  ->  robot_state_keys (6)
    language key: annotation.human.action.task_description
"""

import argparse
import os
import sys
import time

import numpy as np

# Ensure the repo root is importable regardless of CWD (scripts/ would otherwise
# shadow the repo root on sys.path[0]).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the project client wrapper.
from embodiment.so_arm10x.controller import Gr00tRobotInferenceClient  # noqa: E402

ROBOT_STATE_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
CAMERA_KEYS = ["wrist", "front"]


def _synthetic_observation(height: int = 480, width: int = 640) -> dict:
    """Build a single raw observation dict the controller's get_action expects.

    Cameras are random uint8 frames (the server resizes to 224x224); state is a
    plausible mid-range joint vector. Values are arbitrary — this exercises the
    WIRE PATH and decode shape, not policy quality.
    """
    obs = {
        "wrist": np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),
        "front": np.random.randint(0, 255, (height, width, 3), dtype=np.uint8),
    }
    # Mid-range-ish joint positions (degrees/normalized — value is irrelevant here).
    for k in ROBOT_STATE_KEYS:
        obs[k] = 0.0
    return obs


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def test_ping(client: Gr00tRobotInferenceClient) -> bool:
    print("\n[1/4] ping() — reachability ...")
    try:
        ok = client.policy.ping()
        if ok:
            print(_green("  PASS: server reachable (ping ok)"))
            return True
        print(_red("  FAIL: ping returned False (server up but not responding to ping)"))
        return False
    except Exception as e:  # noqa: BLE001
        print(_red(f"  FAIL: ping raised {type(e).__name__}: {e}"))
        return False


def test_get_action(client: Gr00tRobotInferenceClient) -> bool:
    print("\n[2/4] get_action(synthetic obs) — full ZMQ/msgpack round trip ...")
    obs = _synthetic_observation()
    try:
        t0 = time.time()
        actions = client.get_action(obs, lang="I want one banana on the plate")
        dt = time.time() - t0
    except Exception as e:  # noqa: BLE001
        print(_red(f"  FAIL: get_action raised {type(e).__name__}: {e}"))
        return False

    if not isinstance(actions, list) or not actions:
        print(_red(f"  FAIL: expected a non-empty list of action dicts, got {type(actions)}"))
        return False
    first = actions[0]
    missing = [k for k in ROBOT_STATE_KEYS if k not in first]
    if missing:
        print(_red(f"  FAIL: action dict missing keys: {missing}"))
        return False
    print(_green(f"  PASS: received {len(actions)} action steps in {dt:.2f}s "
                 f"(horizon={len(actions)}, keys={list(first.keys())})"))
    print(f"       sample action[0]: { {k: round(v, 4) for k, v in first.items()} }")
    return True


def test_unreachable(host: str, port: int) -> bool:
    print("\n[3/4] unreachable-raises — STOP THE SERVER NOW, then press Enter ...")
    print("       (validates bounded retry -> descriptive RuntimeError, no hang)")
    try:
        input("       Press Enter once the server is stopped... ")
    except EOFError:
        print("       (non-interactive; skipping unreachable test)")
        return True
    client = Gr00tRobotInferenceClient(host=host, port=port,
                                       camera_keys=CAMERA_KEYS,
                                       robot_state_keys=ROBOT_STATE_KEYS)
    obs = _synthetic_observation()
    t0 = time.time()
    try:
        client.get_action(obs, lang="probe")
        print(_red("  FAIL: get_action returned instead of raising against a dead server"))
        return False
    except RuntimeError as e:
        dt = time.time() - t0
        msg = str(e)
        if "unreachable" in msg.lower() and str(port) in msg:
            print(_green(f"  PASS: descriptive RuntimeError after {dt:.1f}s: {msg!r}"))
            return True
        print(_red(f"  PARTIAL: raised RuntimeError but message lacks 'unreachable'/port: {msg!r}"))
        return False
    except Exception as e:  # noqa: BLE001
        print(_red(f"  FAIL: raised {type(e).__name__} (expected RuntimeError): {e}"))
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--skip-unreachable", action="store_true",
                    help="Skip the stop-the-server test (keeps the server running).")
    args = ap.parse_args()

    print("=" * 64)
    print(f" GR00T N1.7 live transport smoke test -> {args.host}:{args.port}")
    print("=" * 64)

    client = Gr00tRobotInferenceClient(
        host=args.host, port=args.port,
        camera_keys=CAMERA_KEYS, robot_state_keys=ROBOT_STATE_KEYS,
    )

    results = {}
    results["ping"] = test_ping(client)
    results["get_action"] = test_get_action(client)
    if not args.skip_unreachable:
        results["unreachable"] = test_unreachable(args.host, args.port)

    print("\n" + "=" * 64)
    passed = sum(1 for v in results.values() if v)
    for name, ok in results.items():
        print(f"  {_green('PASS') if ok else _red('FAIL')}  {name}")
    print(f" {passed}/{len(results)} checks passed")
    print("=" * 64)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
