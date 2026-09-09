#!/usr/bin/env python3
"""Container entrypoint: refuse-to-start preflight, then serve the Dum-E subclass.

Deliberately LOUD. A preflight that shrugs at a missing mount is a silent pass,
which is the exact failure this instrument exists to prevent — because
``configuration_groot.py:382-383`` DEFAULTS ``base_model_path`` to the hub model
``nvidia/GR00T-N1.7-3B`` when it is unset. A wrong or absent bind-mount would
therefore serve BASE weights instead of the SO101 fine-tune while every log line
said the run was fine, and the arm would move confidently to the wrong place. So
a failed check prints ``FAIL:`` naming the path and returns non-zero WITHOUT
starting the server; it never falls through.

This file is intentionally MINIMAL-BUT-REAL for the tracer. Plan 06-06 expands
the two checks below into a five-check numbered ``Checks`` harness with
``--preflight-only``, adding: the config-level horizon assertion, the pinned
backbone-snapshot assertion, and a forced ``encode_step.proc`` build so an
``HF_HUB_OFFLINE=1`` cache miss fails at STARTUP rather than on the first
inference request. Do not pre-empt that expansion here.

==================== WHY NOT upstream's serve() ====================
``lerobot.async_inference.policy_server.serve`` is ``@draccus.wrap()``-decorated
and hardcodes ``PolicyServer(cfg)`` (``policy_server.py:413-435``), so it cannot
be parameterized to serve a subclass. The six gRPC lines in ``main`` below
therefore duplicate ``serve()``'s body DELIBERATELY — composition where
parameterization is unavailable, not a fork of upstream logic.
``tests/test_lerobot_upstream_surface.py`` (plan 06-06) pins the symbols so an
upstream rename breaks the suite rather than the robot.
"""

import argparse
import os
import sys
from concurrent import futures
from pathlib import Path

import grpc

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.policies.groot.configuration_groot import is_raw_groot_n1_7_checkpoint
from lerobot.transport import services_pb2_grpc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import DumEGrootPolicyServer  # noqa: E402

#: Where the fine-tuned checkpoint is bind-mounted read-only at run time. It is
#: NOT baked into the image (D-07), so the mount path is part of the run
#: contract — which is what makes the preflight below mandatory.
DEFAULT_CHECKPOINT_PATH = "/checkpoints/model"


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def run_preflight(checkpoint_path: str, backbone_revision: str) -> int:
    """Two loud checks. Returns 0 to proceed, 1 to refuse to start.

    Args:
        checkpoint_path: The in-container checkpoint directory.
        backbone_revision: The pinned backbone revision SHA, reported for the
            record. Plan 06-06 turns this into an ASSERTION against the cached
            snapshot directory on disk; here it is logged so the value a given
            container run is carrying is visible in ``docker logs``.
    """
    print(f"\n[1/2] checkpoint mount present at {checkpoint_path!r} ...")
    config_json = Path(checkpoint_path) / "config.json"
    if not config_json.is_file():
        print(
            _red(
                f"  FAIL: {config_json} does not exist. The checkpoint bind-mount is "
                f"missing or wrong. Refusing to start: with base_model_path unset, "
                f"configuration_groot.py:382-383 falls back to the hub model "
                f"'nvidia/GR00T-N1.7-3B', so the server would silently serve BASE "
                f"weights instead of the SO101 fine-tune. Mount with "
                f"-v <host-checkpoint-dir>:{checkpoint_path}:ro"
            )
        )
        return 1
    print(_green(f"  PASS: {config_json} exists"))

    print(f"\n[2/2] raw-GR00T-N1.7 checkpoint discriminator for {checkpoint_path!r} ...")
    if not is_raw_groot_n1_7_checkpoint(checkpoint_path):
        print(
            _red(
                f"  FAIL: is_raw_groot_n1_7_checkpoint({checkpoint_path!r}) is False. "
                f"This is not the raw GR00T trainer layout, so checkpoint_assets would "
                f"be None and EVERY checkpoint-derived setting — the relative-action "
                f"decoder, the per-timestep statistics, the image geometry — would "
                f"silently fall back to LeRobot defaults. Refusing to serve."
            )
        )
        return 1
    print(_green(f"  PASS: is_raw_groot_n1_7_checkpoint({checkpoint_path!r}) is True"))

    print(f"\nPreflight PASSED (2/2). Pinned backbone revision: {backbone_revision or '<unset>'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # argparse, NOT draccus: draccus owns upstream's CLI surface for
    # PolicyServerConfig, and this entrypoint adds arguments (checkpoint path,
    # backbone revision) that config does not carry.
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104 - see comment below; loopback is enforced by the publish spec
        help=(
            "Bind address INSIDE the container. Deliberately 0.0.0.0, overriding "
            "PolicyServerConfig's host='localhost' default (configs.py:56): a "
            "container-internal loopback bind is unreachable from the host, so that "
            "default is simply wrong here. The loopback GUARANTEE lives in the "
            "host-side publish spec -p 127.0.0.1:8080:8080, never in this bind "
            "address, and --network host is rejected outright."
        ),
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to bind inside the container.")
    parser.add_argument(
        "--checkpoint-path",
        default=DEFAULT_CHECKPOINT_PATH,
        help="In-container path of the read-only fine-tuned checkpoint bind-mount.",
    )
    parser.add_argument(
        "--backbone-revision",
        default=os.environ.get("DUME_BACKBONE_REVISION", ""),
        help="Pinned nvidia/Cosmos-Reason2-2B revision SHA baked into the image.",
    )
    args = parser.parse_args()

    if run_preflight(args.checkpoint_path, args.backbone_revision) != 0:
        return 1

    cfg = PolicyServerConfig(host=args.host, port=args.port)
    policy_server = DumEGrootPolicyServer(cfg)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info(f"DumEGrootPolicyServer started on {cfg.host}:{cfg.port}")
    server.start()
    server.wait_for_termination()
    policy_server.logger.info("Server terminated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
