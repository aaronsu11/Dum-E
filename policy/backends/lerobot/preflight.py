#!/usr/bin/env python3
'Container entrypoint: six-check refuse-to-start preflight, then serve the Dum-E subclass.'

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

# EXPECTED_HORIZON / EXPECTED_TAG are IMPORTED, never restated as literals. They
# come from ``policy.backends.lerobot.models.groot`` — the same module check 6 below calls, the
from policy.backends.lerobot.models.groot import (
    EXPECTED_HORIZON,
    EXPECTED_TAG,
    assert_groot_serving_contract,
    snapshot_from_checkpoint_dir,
)

from policy.backends.lerobot.server import DumEGrootPolicyServer, fixup_policy_features  # noqa: E402

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
#: ``[n/TOTAL]`` prefixes and the harness agree. Raised 5 -> 6 by plan 06-03 when
#: it filled the SAFE-01 call site below.
TOTAL_CHECKS = 6


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


class Checks:
    'Numbered-check PASS/FAIL harness with a ``passed == total`` exit contract.'

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
        # A check that never ran is reported distinctly from a check that failed:
        # the two demand different operator responses (a code/build defect vs a
        missing = self.total - len(self.results)
        if missing > 0 and passed == len(self.results):
            print(
                _red(
                    f"  FAIL: {missing} of {self.total} preflight check(s) never ran. "
                    "Every recorded check passed, so this is not a bad mount or a "
                    "drifted revision — a check was removed from run_preflight() or "
                    "TOTAL_CHECKS disagrees with it. Refusing to serve on an "
                    "incomplete preflight."
                )
            )
        print("=" * 72)
        return 0 if passed == len(self.results) == self.total else 1


def _cached_backbone_revisions() -> tuple[Path, list[str]]:
    """The pinned backbone's snapshots directory and the revisions inside it."""
    snapshots_dir = Path(HF_HUB_CACHE) / BACKBONE_CACHE_DIRNAME / "snapshots"
    if not snapshots_dir.is_dir():
        return snapshots_dir, []
    return snapshots_dir, sorted(entry.name for entry in snapshots_dir.iterdir() if entry.is_dir())


def run_preflight(checkpoint_path: str, backbone_revision: str) -> int:
    'Six loud checks. Returns 0 to proceed, non-zero to refuse to start.'
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

    # The tag is passed EXPLICITLY: tag inference returns None for this
    # checkpoint (it carries nine tags, and only 'new_embodiment' has 16
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

    # Plan 06-01 left a placeholder comment here; plan 06-03 filled it. This is
    # D-04 option 3's PREFLIGHT site, and it is the reason SAFE-01's "refuses to
    checks.start(f"SAFE-01 serving contract for {checkpoint_path!r} (config-only)")
    try:
        snapshot = snapshot_from_checkpoint_dir(checkpoint_path, EXPECTED_HORIZON)
        assert_groot_serving_contract(snapshot)
    except ValueError as exc:
        checks.fail(
            "safe01_serving_contract",
            f"{exc} [SAFE-01 preflight, config-only, against {checkpoint_path!r}]",
        )
        return checks.report()
    checks.ok(
        "safe01_serving_contract",
        f"SAFE-01/1..5 hold config-only: embodiment_tag="
        f"{snapshot.embodiment_tag!r}, checkpoint_horizon={snapshot.checkpoint_horizon}, "
        f"use_relative_action={snapshot.use_relative_action}, "
        f"use_percentiles={snapshot.use_percentiles}, "
        f"stats_non_empty={snapshot.stats_non_empty}, "
        f"crop_fraction={snapshot.crop_fraction}, "
        f"shortest_image_edge={snapshot.shortest_image_edge}, "
        f"letter_box_transform={snapshot.letter_box_transform}, "
        # The camera-view ORDER the checkpoint declares. On the line because it is
        # config-visible AND because a mismatch is silent downstream: upstream falls
        # back to alphabetical order with one logging.warning, so an operator needs to
        # be able to read the accepted layout out of `docker logs`.
        f"video_modality_keys={snapshot.video_modality_keys} "
        f"(decode_step_type, served_letter_box_transform and the two training flags "
        f"are NOT validated here — post-load site only)",
    )

    # ---- 5. the image carries the PINNED backbone snapshot ----
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

    # Deliberately LAST: it is the final gate, run only after the cheap
    # config-level checks above have established that the mount is what it
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


def positive_cpu_threads(value: str) -> int:
    try:
        threads = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("CPU threads must be a positive integer") from exc
    if threads < 1:
        raise argparse.ArgumentTypeError("CPU threads must be a positive integer")
    return threads


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
    parser.add_argument(
        "--cpu-threads", type=positive_cpu_threads,
        default=os.environ.get("DUME_POLICY_CPU_THREADS", "1"),
        help="PyTorch CPU intra-op threads for image preparation (default: 1; "
             "environment: DUME_POLICY_CPU_THREADS). Model inference remains on CUDA.",
    )
    args = parser.parse_args()

    # Apply before processor construction and before gRPC worker threads start.
    # The RTX 3060 benchmark found substantial overhead with the 20-thread
    # host default; one thread retained exact outputs for the measured input.
    import torch
    torch.set_num_threads(args.cpu_threads)
    print(f"PyTorch CPU threads: intra_op={torch.get_num_threads()} "
          f"inter_op={torch.get_num_interop_threads()}", flush=True)

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
