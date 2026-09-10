#!/usr/bin/env python3
"""Container entrypoint: five-check refuse-to-start preflight, then serve the Dum-E subclass.

Deliberately LOUD, and it **never skips quietly**. A preflight that shrugs at a
missing mount is a silent pass, which is the exact failure this instrument exists
to prevent — because ``configuration_groot.py:382-383`` DEFAULTS
``base_model_path`` to the hub model ``nvidia/GR00T-N1.7-3B`` when it is unset. A
wrong or absent bind-mount would therefore serve BASE weights instead of the
SO101 fine-tune while every log line said the run was fine, and the arm would
move confidently to the wrong place.

So an absent or wrong checkpoint mount, a non-raw checkpoint, an action horizon
that is not the checkpoint's real 16, a missing or mismatched pinned backbone
snapshot (an EMPTY ``--backbone-revision`` included), and a processor build that
cannot reach its ``HF_HUB_OFFLINE=1`` cache ALL print a red ``FAIL:`` naming the
observed value and refuse to serve. "Refuse to serve" is literal, not a log
line: ``main`` returns BEFORE the gRPC server is constructed and before
``add_insecure_port``, so a failed check can never reach a listening socket.

``--preflight-only`` runs exactly those checks and returns their verdict without
constructing the server, so an operator can exercise the refusal path without
starting anything. It calls the SAME ``run_preflight`` body the serving path
calls — there is deliberately no second copy of the checks, because a parallel
copy would make the flag prove nothing about the real startup.

None of the five checks loads policy weights: every one is config-only or
processor-only, so the preflight costs no 12.6 GB weight load and no GPU memory.

==================== WHY NOT upstream's serve() ====================
``lerobot.async_inference.policy_server.serve`` is ``@draccus.wrap()``-decorated
and hardcodes ``PolicyServer(cfg)`` (``policy_server.py:413-435``), so it cannot
be parameterized to serve a subclass. The six gRPC lines in ``main`` below
therefore duplicate ``serve()``'s body DELIBERATELY — composition where
parameterization is unavailable, not a fork of upstream logic.
``tests/test_lerobot_upstream_surface.py`` pins the symbols so an upstream rename
breaks the suite rather than the robot.
"""

import argparse
import os
import sys
from concurrent import futures
from pathlib import Path

import grpc
from huggingface_hub.constants import HF_HUB_CACHE

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.policies import make_pre_post_processors
from lerobot.policies.groot.configuration_groot import (
    GrootConfig,
    infer_groot_n1_7_action_horizon,
    is_raw_groot_n1_7_checkpoint,
)
from lerobot.policies.groot.processor_groot import GrootN17VLMEncodeStep
from lerobot.transport import services_pb2_grpc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# EXPECTED_HORIZON / EXPECTED_TAG are imported rather than restated: server.py is
# the one place in the image that owns those two numbers, and check 3 below must
# assert the value the SERVER will actually load with.
from server import (  # noqa: E402
    EXPECTED_HORIZON,
    EXPECTED_TAG,
    DumEGrootPolicyServer,
    fixup_policy_features,
)

#: Where the fine-tuned checkpoint is bind-mounted read-only at run time. It is
#: NOT baked into the image (D-07), so the mount path is part of the run
#: contract — which is what makes the preflight below mandatory.
DEFAULT_CHECKPOINT_PATH = "/checkpoints/model"

#: Sidecars whose absence means the mount is not a GR00T trainer output. All
#: three are read by the load path: ``config.json`` (the model config and the
#: raw-checkpoint discriminator), ``processor_config.json`` (the relative-action
#: and image-geometry recipe) and ``statistics.json`` (the per-timestep
#: normalization percentiles).
REQUIRED_CHECKPOINT_FILES = ("config.json", "processor_config.json", "statistics.json")

#: The VLM backbone the checkpoint's processor loads, and its Hugging Face hub
#: cache directory name. The name is hardcoded at upstream's call site
#: (``processor_groot.py:1259``), never read from the checkpoint, so this string
#: is the only thing that can be asserted against.
BACKBONE_MODEL = "nvidia/Cosmos-Reason2-2B"
BACKBONE_CACHE_DIRNAME = "models--nvidia--Cosmos-Reason2-2B"

#: How many numbered checks the preflight prints. It is stated once here so the
#: ``[n/TOTAL]`` prefixes and the harness agree; plan 06-03 raises it to 6 when
#: it fills the SAFE-01 call site marked below.
TOTAL_CHECKS = 5


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


class Checks:
    """Numbered-check PASS/FAIL harness with a ``passed == total`` exit contract.

    PORTED COPY of ``scripts/capture_frozen_corpus.py:144-173`` (the ``Checks``
    class, plus the ``_green``/``_red`` helpers above it). It is a copy rather
    than an import for a mechanical reason, not a stylistic one: the Dockerfile
    copies ONLY ``docker/lerobot-policy/*.py`` into the image
    (``COPY docker/lerobot-policy/*.py /app/docker/lerobot-policy/``), so
    ``scripts/`` does not exist inside the container and there is nothing to
    import from. Keep the two in sync by hand if the host-side harness changes
    shape; the contract that matters is ``report()`` returning non-zero unless
    every recorded check passed.
    """

    def __init__(self, total: int) -> None:
        self.total = total
        self.index = 0
        self.results: dict[str, bool] = {}

    def start(self, description: str) -> None:
        self.index += 1
        print(f"\n[{self.index}/{self.total}] {description} ...")

    def ok(self, name: str, message: str) -> bool:
        print(_green(f"  PASS: {message}"))
        self.results[name] = True
        return True

    def fail(self, name: str, message: str) -> bool:
        print(_red(f"  FAIL: {message}"))
        self.results[name] = False
        return False

    def report(self) -> int:
        print("\n" + "=" * 72)
        passed = sum(1 for ok in self.results.values() if ok)
        for name, ok in self.results.items():
            print(f"  {_green('PASS') if ok else _red('FAIL')}  {name}")
        print(f" {passed}/{len(self.results)} checks passed")
        print("=" * 72)
        return 0 if passed == len(self.results) else 1


def _cached_backbone_revisions() -> tuple[Path, list[str]]:
    """The pinned backbone's snapshots directory and the revisions inside it."""
    snapshots_dir = Path(HF_HUB_CACHE) / BACKBONE_CACHE_DIRNAME / "snapshots"
    if not snapshots_dir.is_dir():
        return snapshots_dir, []
    return snapshots_dir, sorted(entry.name for entry in snapshots_dir.iterdir() if entry.is_dir())


def run_preflight(checkpoint_path: str, backbone_revision: str) -> int:
    """Five loud checks. Returns 0 to proceed, non-zero to refuse to start.

    Short-circuits on the first FAIL: the later checks read the artifacts the
    earlier ones prove exist, so continuing past a failure would replace a
    precise refusal with a traceback about a consequence.

    Args:
        checkpoint_path: The in-container checkpoint directory (the bind-mount).
        backbone_revision: The pinned ``nvidia/Cosmos-Reason2-2B`` revision SHA
            baked into the image. ASSERTED by check 4 — an empty value FAILs.
    """
    checks = Checks(TOTAL_CHECKS)

    # ---- 1. the checkpoint bind-mount is present and is a trainer output ----
    checks.start(f"checkpoint mount present at {checkpoint_path!r}")
    missing = [
        str(Path(checkpoint_path) / name)
        for name in REQUIRED_CHECKPOINT_FILES
        if not (Path(checkpoint_path) / name).is_file()
    ]
    if missing:
        checks.fail(
            "checkpoint_mount",
            f"missing {missing} under {checkpoint_path!r}. The checkpoint bind-mount is "
            f"absent or wrong. Refusing to start: with base_model_path unset, "
            f"configuration_groot.py:382-383 falls back to the hub model "
            f"'nvidia/GR00T-N1.7-3B', so the server would silently serve BASE weights "
            f"instead of the SO101 fine-tune under healthy-looking logs. Mount with "
            f"-v <host-checkpoint-dir>:{checkpoint_path}:ro",
        )
        return checks.report()
    checks.ok(
        "checkpoint_mount",
        f"{list(REQUIRED_CHECKPOINT_FILES)} all present under {checkpoint_path!r}",
    )

    # ---- 2. raw-GR00T-N1.7 discriminator ----
    checks.start(f"raw-GR00T-N1.7 checkpoint discriminator for {checkpoint_path!r}")
    if not is_raw_groot_n1_7_checkpoint(checkpoint_path):
        checks.fail(
            "raw_checkpoint",
            f"is_raw_groot_n1_7_checkpoint({checkpoint_path!r}) is False. This is not the "
            f"raw GR00T trainer layout, so checkpoint_assets would be None and EVERY "
            f"checkpoint-derived setting — the relative-action decoder, the per-timestep "
            f"statistics, the image geometry — would silently fall back to LeRobot "
            f"defaults. Refusing to serve.",
        )
        return checks.report()
    checks.ok("raw_checkpoint", f"is_raw_groot_n1_7_checkpoint({checkpoint_path!r}) is True")

    # ---- 3. the action horizon is the checkpoint's real 16, at CONFIG level ----
    # The tag is passed EXPLICITLY: tag inference returns None for this
    # checkpoint (it carries nine tags, and only 'new_embodiment' has 16
    # delta_indices), so an implicit call would report None and prove nothing.
    # This is a CONFIG-level assertion on purpose — the emitted chunk length is
    # forced to 16 by three independent truncations regardless of whether the
    # configuration is right, so it corroborates and never evidences.
    checks.start(
        f"action horizon for {checkpoint_path!r} at embodiment_tag={EXPECTED_TAG!r} "
        f"is {EXPECTED_HORIZON}"
    )
    observed_horizon = infer_groot_n1_7_action_horizon(checkpoint_path, EXPECTED_TAG)
    if observed_horizon != EXPECTED_HORIZON:
        checks.fail(
            "action_horizon",
            f"infer_groot_n1_7_action_horizon({checkpoint_path!r}, {EXPECTED_TAG!r}) is "
            f"{observed_horizon!r}, not {EXPECTED_HORIZON}. Refusing to serve — a wrong "
            f"horizon decodes the chunk against the wrong per-timestep statistics. Both "
            f"nearby 40s are traps and neither is the answer: GrootConfig's own "
            f"action_horizon DEFAULT is 40, and this checkpoint's config.json ALSO says "
            f"action_horizon: 40, while the load-bearing value is the 16 delta_indices "
            f"on the {EXPECTED_TAG!r} embodiment tag. A None here means the tag did not "
            f"resolve at all.",
        )
        return checks.report()
    checks.ok(
        "action_horizon",
        f"infer_groot_n1_7_action_horizon({checkpoint_path!r}, {EXPECTED_TAG!r}) is "
        f"{observed_horizon}",
    )

    # --- SAFE-01 guard call site (plan 06-03) ---
    # Plan 06-03 adds the config-only GrootGuardSnapshot build plus the
    # assert_groot_serving_contract call here, and the harness then renumbers to
    # [n/6] (raise TOTAL_CHECKS). Deliberately NOT stubbed with a fake guard
    # function: a placeholder comment is the whole contract.

    # ---- 4. the image carries the PINNED backbone snapshot ----
    checks.start(
        f"pinned {BACKBONE_MODEL} snapshot present in the image's HF hub cache "
        f"({backbone_revision or '<unset>'})"
    )
    snapshots_dir, found_revisions = _cached_backbone_revisions()
    pinned_snapshot = snapshots_dir / backbone_revision if backbone_revision else None
    if not backbone_revision:
        checks.fail(
            "backbone_snapshot",
            f"--backbone-revision is EMPTY, so there is no revision to assert against "
            f"{snapshots_dir} (revisions found on disk: {found_revisions}). Refusing to "
            f"serve an UNPINNED build rather than skipping this check: "
            f"_build_n1_7_processor accepts no `revision` argument "
            f"(processor_groot.py:1369-1381), so the image's cache contents plus "
            f"HF_HUB_OFFLINE=1 are the ONLY available enforcement (D-08) and an empty "
            f"pin disables all of it. This assertion is DRIFT protection only — it "
            f"proves the image carries the revision the build pinned, and makes no claim "
            f"that the pin is the revision the checkpoint was trained against (no local "
            f"artifact records one). Set DUME_BACKBONE_REVISION or pass "
            f"--backbone-revision.",
        )
        return checks.report()
    if pinned_snapshot is None or not pinned_snapshot.is_dir() or not any(pinned_snapshot.iterdir()):
        checks.fail(
            "backbone_snapshot",
            f"asserted revision {backbone_revision!r} is not a non-empty snapshot "
            f"directory under {snapshots_dir}; revisions actually found: "
            f"{found_revisions}. Refusing to serve: the image's backbone layer is empty "
            f"or carries a DIFFERENT revision than this run claims, and "
            f"_build_n1_7_processor accepts no `revision` argument "
            f"(processor_groot.py:1369-1381), so cache contents plus HF_HUB_OFFLINE=1 "
            f"are the ONLY available enforcement (D-08). Rebuild with "
            f"scripts/build_lerobot_policy_image.sh. This assertion is DRIFT protection "
            f"only: it does not claim the pinned SHA is the revision the checkpoint was "
            f"trained against — no local artifact records one.",
        )
        return checks.report()
    checks.ok(
        "backbone_snapshot",
        f"{pinned_snapshot} exists and is non-empty (revisions on disk: {found_revisions})",
    )

    # ---- 5. force the LAZY processor build, so an offline cache miss fails HERE ----
    # Deliberately LAST: it is the final gate, run only after the cheap
    # config-level checks above have established that the mount is what it
    # claims. GrootN17VLMEncodeStep.proc is a lazily-built property
    # (processor_groot.py:2069-2073), so without this the three backbone
    # from-pretrained loads happen on the FIRST GetActions — where
    # policy_server.py's blanket `except Exception -> Empty()` turns the failure
    # into a successful RPC carrying zero bytes. Config-only and CPU-only: a
    # tokenizer plus two image/video processors, no model weights.
    checks.start("forced processor build (HF_HUB_OFFLINE cache reachable)")
    try:
        config = GrootConfig(base_model_path=checkpoint_path, embodiment_tag=EXPECTED_TAG)
        fixup_policy_features(
            config,
            camera_keys=DumEGrootPolicyServer.CAMERA_KEYS,
            height=DumEGrootPolicyServer.FRAME_HEIGHT,
            width=DumEGrootPolicyServer.FRAME_WIDTH,
            state_dim=DumEGrootPolicyServer.STATE_DIM,
            action_dim=DumEGrootPolicyServer.ACTION_DIM,
        )
        preprocessor, _ = make_pre_post_processors(config, pretrained_path=checkpoint_path)
        encode_steps = [
            step for step in preprocessor.steps if isinstance(step, GrootN17VLMEncodeStep)
        ]
        if not encode_steps:
            raise RuntimeError(
                "the built preprocessor contains no GrootN17VLMEncodeStep "
                f"(steps: {[type(step).__name__ for step in preprocessor.steps]})"
            )
        processor = encode_steps[0].proc
    except Exception as exc:  # noqa: BLE001 - any failure here must refuse to serve
        checks.fail(
            "processor_build",
            f"forcing the lazy GrootN17VLMEncodeStep.proc build raised "
            f"{type(exc).__name__}: {exc}. Refusing to serve. Under HF_HUB_OFFLINE=1 this "
            f"means the image's backbone layer is EMPTY or the pin MOVED — the assets "
            f"cannot be resolved from the cache, so every inference would fail. Left to "
            f"run time it would surface inside the first GetActions, where "
            f"policy_server.py's blanket `except Exception -> Empty()` returns a "
            f"SUCCESSFUL RPC carrying zero bytes and hides the cause entirely. Rebuild "
            f"with scripts/build_lerobot_policy_image.sh.",
        )
        return checks.report()
    checks.ok(
        "processor_build",
        f"GrootN17VLMEncodeStep.proc built offline: {type(processor).__name__}",
    )

    print(f"\nPreflight PASSED ({TOTAL_CHECKS}/{TOTAL_CHECKS}).")
    return checks.report()


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
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Run the numbered preflight and exit with its verdict WITHOUT constructing "
            "the gRPC server. Exercises the same run_preflight body the serving path "
            "calls, so the refusal path can be tested without starting a server."
        ),
    )
    args = parser.parse_args()

    # ONE call site for both paths. --preflight-only must not grow a second copy
    # of the checks: a parallel copy could pass while the real startup failed.
    preflight_status = run_preflight(args.checkpoint_path, args.backbone_revision)
    if args.preflight_only:
        return preflight_status
    if preflight_status != 0:
        # Refuse to SERVE, not merely to log: return before the gRPC server is
        # constructed and before add_insecure_port, so nothing ever listens.
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
