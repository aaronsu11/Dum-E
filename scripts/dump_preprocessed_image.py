#!/usr/bin/env python3
"""Settle PAR-05 offline: what geometry does THIS checkpoint's image recipe produce?

**Recorded verdict, up front: a 480x640x3 uint8 frame becomes exactly
``(256, 340, 3)`` uint8** under the checkpoint's own recipe. The effective
pipeline is *resize-shortest-edge-to-256 -> center-crop-95% ->
resize-shortest-edge-to-256*.

This probe needs **no GPU, no weights, no network, no Hugging Face token and no
hardware**. It calls the module-level pure function
``lerobot.policies.groot.processor_groot._transform_n1_7_image_for_vlm_albumentations``
directly — no model load, no ``PolicyServer`` subclass, no monkeypatch — which is
what makes PAR-05 reachable at all under this phase's compose-only posture.

It is deliberately LOUD. A shape that does not match the pinned expectation
**FAILS** with a non-zero exit; it is never printed as an interesting
observation. A probe that reports a mismatch and exits 0 would launder a real
preprocessing shift into a passing record, and PAR-05 sits in this phase
precisely because Phase 7's parity numbers are meaningless until the geometry is
settled.

Nothing here reimplements a resize or a crop. Upstream documents its cv2
``INTER_AREA`` resize and floored center-crop as needing to stay bit-exact
(``processor_groot.py:1394-1401``), so a hand-rolled copy would *guarantee* the
PAR-05 mismatch this probe exists to rule out.

Every ``.npz`` this script reads is loaded with ``allow_pickle=False`` passed
EXPLICITLY, the same choice ``policy/gr00t/service.py`` makes on the wire and
``scripts/verify_frozen_corpus.py`` makes on the corpus. NOTE, deliberately:
``numpy.savez_compressed`` accepts **no** ``allow_pickle`` parameter — every
keyword it receives is treated as an array to store, so passing one would write
an array literally named ``allow_pickle`` (``scripts/capture_frozen_corpus.py``
records the same constraint). The write side is therefore proven instead of
asserted: :func:`write_dump` re-loads what it just wrote with
``allow_pickle=False``, so a pickled object array could not survive the write.

Usage:
    uv run python scripts/dump_preprocessed_image.py
    uv run python scripts/dump_preprocessed_image.py --outdir outputs/par05 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure the repo root is importable regardless of CWD (the scripts/ dir would
# otherwise shadow the repo root on sys.path[0]).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO_ROOT = Path(__file__).resolve().parent.parent

#: The checkpoint whose recipe is the subject of the PAR-05 verdict.
CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "GR00T-N1.7-3B-SO101"

#: The sidecar that is the SOURCE OF TRUTH for every recipe value below.
PROCESSOR_CONFIG = CHECKPOINT_DIR / "processor_config.json"


# --- The checkpoint's recipe, pinned ------------------------------------------
#
# Every constant here is transcribed from
# ``checkpoints/GR00T-N1.7-3B-SO101/processor_config.json`` -> ``processor_kwargs``.
# Check 1 in main() re-reads that file and FAILS on any disagreement, so these
# cannot silently drift away from the checkpoint on disk.

#: processor_config.json: processor_kwargs.letter_box_transform.
#: False -> this checkpoint does NOT take the letterbox branch. The branch is
#: real and reachable (it yields (256, 256, 3)); it is simply not this
#: checkpoint's path, which is half of why the ROADMAP's framing is stale.
LETTER_BOX_TRANSFORM = False

#: processor_config.json: processor_kwargs.crop_fraction. Because this is SET
#: (not None), it takes precedence over ``image_crop_size`` — see below.
CROP_FRACTION = 0.95

#: processor_config.json: processor_kwargs.image_crop_size.
#: **PROVABLY INERT on this checkpoint.** ``processor_groot.py:1453-1454`` reads
#: it only inside ``if crop_fraction is None and image_crop_size is not None``,
#: and ``crop_fraction`` is 0.95 here — so [230, 230] is dead configuration.
#: Measured, not assumed: ``crop_size_inert`` in the manifest compares the
#: [230, 230] output against a [999, 999] output with ``numpy.array_equal``.
#: Recorded so a future reader does not tune a value that does nothing.
IMAGE_CROP_SIZE = [230, 230]

#: processor_config.json: processor_kwargs.image_target_size. Only supplies
#: ``target_h`` as the fallback resize edge; it is NOT the output shape.
IMAGE_TARGET_SIZE = [256, 256]

#: processor_config.json: processor_kwargs.shortest_image_edge. This is the edge
#: both resize passes target, and it is why the output HEIGHT is 256.
SHORTEST_IMAGE_EDGE = 256

#: processor_config.json: processor_kwargs.use_albumentations.
#: True selects the cv2/numpy transform this probe calls. It is NOT a keyword
#: argument of ``_transform_n1_7_image_for_vlm_albumentations`` — it is the flag
#: upstream branches on to reach that function at all (the torch path returns
#: (3, 256, 256) instead). Pinned here because a checkpoint that flipped it
#: would invalidate the whole verdict.
USE_ALBUMENTATIONS = True

#: The keys of the six recipe values, in the order check 1 reports them.
RECIPE_KEYS = (
    "letter_box_transform",
    "crop_fraction",
    "image_crop_size",
    "image_target_size",
    "shortest_image_edge",
    "use_albumentations",
)


# --- Expected geometry, pinned as exact integers ------------------------------

#: **THE PAR-05 VERDICT.** A 480x640x3 uint8 frame through the checkpoint's own
#: recipe. Exact integers, no tolerance: the pipeline is a deterministic
#: crop-then-resize, so an approximate match would let a real preprocessing
#: shift pass.
#:
#: The ROADMAP's "341x256-crop vs 256x256-letterbox" framing is STALE, and
#: REQUIREMENTS.md:50 already records it as a misread. Both halves of that
#: framing are real but neither is the answer: 256x341 is the INTERMEDIATE after
#: the first resize-shortest-edge of a 480x640 frame, and 256x256 is the
#: LETTERBOX branch this checkpoint does not take.
EXPECTED_SHAPE = (256, 340, 3)

#: What the C-3 corruption produces instead. The reason a shape-only
#: cross-container comparison is genuinely discriminating here rather than
#: merely convenient: the corruption CHANGES the shape.
EXPECTED_CORRUPTED_SHAPE = (256, 256, 3)

#: The source frame size, taken from ``scripts/test_live_policy_server.py``'s
#: ``_synthetic_observation(height=480, width=640)`` rather than invented — the
#: (256, 340, 3) verdict is measured AT 480x640 and does not generalize to
#: another aspect ratio.
SOURCE_HEIGHT = 480
SOURCE_WIDTH = 640

#: The edge ``from_pretrained``'s placeholder feature squares every camera to.
PLACEHOLDER_EDGE = 224

#: The four case names recorded in the manifest and dumped to the .npz.
CASE_NAMES = ("checkpoint_recipe", "placeholder_corrupted", "square_256", "letterbox")


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


class Checks:
    """Numbered-check PASS/FAIL harness with a ``passed == total`` exit contract.

    Shape copied from ``scripts/capture_frozen_corpus.py:144-173``.
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


# --- The recipe, read from the checkpoint -------------------------------------


def pinned_recipe() -> dict[str, Any]:
    """The six recipe values as this module pins them."""
    return {
        "letter_box_transform": LETTER_BOX_TRANSFORM,
        "crop_fraction": CROP_FRACTION,
        "image_crop_size": IMAGE_CROP_SIZE,
        "image_target_size": IMAGE_TARGET_SIZE,
        "shortest_image_edge": SHORTEST_IMAGE_EDGE,
        "use_albumentations": USE_ALBUMENTATIONS,
    }


def checkpoint_recipe(path: Path = PROCESSOR_CONFIG) -> dict[str, Any]:
    """The six recipe values as the checkpoint on disk declares them.

    Raises (never returns a defaulted dict) when the file or any of the six keys
    is absent: a silently defaulted geometry value is exactly the drift this
    probe exists to catch.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint processor config not found at {path} — the PAR-05 verdict "
            "rests on its six geometry values and cannot be asserted without it"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    kwargs = payload.get("processor_kwargs")
    if not isinstance(kwargs, dict):
        raise ValueError(f"{path} carries no processor_kwargs object")
    missing = [key for key in RECIPE_KEYS if key not in kwargs]
    if missing:
        raise ValueError(
            f"{path} is missing processor_kwargs keys {missing} — upstream coerces "
            "absent geometry keys to False/None without comment, so a defaulted "
            "value would pass a check it should fail"
        )
    return {key: kwargs[key] for key in RECIPE_KEYS}


# --- The transforms ----------------------------------------------------------


def synthetic_frame(seed: int = 0) -> np.ndarray:
    """A deterministic 480x640x3 uint8 frame.

    Seeded (``RandomState``, not the global RNG) so the dump is byte-reproducible
    and so a cross-container comparison can be made against *the same bytes*
    rather than two independently random frames.
    """
    return np.random.RandomState(seed).randint(
        0, 255, (SOURCE_HEIGHT, SOURCE_WIDTH, 3), dtype=np.uint8
    )


def transform_checkpoint_recipe(frame: np.ndarray, **overrides: Any) -> np.ndarray:
    """Run the checkpoint's own image recipe over ``frame``.

    Calls upstream's module-level pure function directly. ``overrides`` lets the
    same helper serve the inert-crop-size case (``image_crop_size=[999, 999]``)
    and the letterbox case (``letter_box_transform=True``) without a second copy
    of the call.

    ``use_albumentations`` is deliberately NOT forwarded: it is not a parameter
    of this function, it is the flag upstream branches on to select it.
    """
    from lerobot.policies.groot.processor_groot import (
        _transform_n1_7_image_for_vlm_albumentations,
    )

    kwargs: dict[str, Any] = {
        "image_crop_size": IMAGE_CROP_SIZE,
        "image_target_size": IMAGE_TARGET_SIZE,
        "shortest_image_edge": SHORTEST_IMAGE_EDGE,
        "crop_fraction": CROP_FRACTION,
        "letter_box_transform": LETTER_BOX_TRANSFORM,
    }
    kwargs.update(overrides)
    return _transform_n1_7_image_for_vlm_albumentations(frame, **kwargs)


def transform_after_placeholder_resize(frame: np.ndarray) -> np.ndarray:
    """Reproduce the C-3 corruption, then run the checkpoint recipe on it.

    This is not a hypothetical. ``from_pretrained`` injects a SINGLE
    ``observation.images.camera`` placeholder feature at ``(3, 224, 224)``
    (``modeling_groot.py:255-261``), and ``prepare_raw_observation`` resizes
    every camera to that placeholder's shape (``helpers.py:165-168``) *before*
    the checkpoint's own geometry runs — so a single-camera observation is
    silently squared to 224x224 and its aspect ratio is destroyed.

    The resize is performed by ``resize_robot_observation_image``, the very
    function ``prepare_raw_observation`` calls, over a uint8 HWC tensor built the
    same way (``torch.tensor(lerobot_obs[key])``) — not a hand-written resize.

    Why this corruption is CATCHABLE by a shape check: it lands on
    ``(256, 256, 3)``, not ``(256, 340, 3)``.
    """
    import torch
    from lerobot.async_inference.helpers import resize_robot_observation_image

    squared = resize_robot_observation_image(
        torch.tensor(frame), (3, PLACEHOLDER_EDGE, PLACEHOLDER_EDGE)
    )
    # (C, H, W) back to the HxWx3 uint8 frame the checkpoint transform accepts.
    hwc = np.ascontiguousarray(squared.permute(1, 2, 0).to(torch.uint8).numpy())
    return transform_checkpoint_recipe(hwc)


# --- Dump writing ------------------------------------------------------------


def write_dump(
    outdir: Path, arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> tuple[Path, Path]:
    """Write ``dumps.npz`` + ``manifest.json`` under ``outdir``.

    Mirrors ``scripts/capture_frozen_corpus.py:695-745``: ``np.savez_compressed``
    for the arrays, ``json.dumps(..., indent=2, sort_keys=True)`` for the
    manifest.

    The manifest records the recipe values, the seed, and every observed shape as
    DATA rather than prose, because the cross-container PAR-05 comparison must
    compare RECORDED facts rather than re-derive them in a second place (and a
    re-derivation is a second source of truth that can disagree).

    ``savez_compressed`` takes no ``allow_pickle`` argument — any keyword it
    receives becomes an array in the archive — so the write is PROVEN instead:
    the file is immediately re-loaded with ``allow_pickle=False``, which fails
    outright on a pickled object array.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    npz_path = outdir / "dumps.npz"
    np.savez_compressed(npz_path, **arrays)

    # Proof, not assertion: a pickled object array cannot survive this reload.
    with np.load(npz_path, allow_pickle=False) as verify:
        for name in arrays:
            _ = verify[name]

    manifest_path = outdir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return npz_path, manifest_path


def _shape_of(array: np.ndarray) -> list[int]:
    return [int(n) for n in array.shape]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="outputs/par05",
        help="Destination for dumps.npz + manifest.json (default: outputs/par05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the synthetic 480x640 frame (default: 0)",
    )
    args = parser.parse_args()

    checks = Checks(total=6)

    # --- 1: the pinned recipe still equals the checkpoint on disk ------------
    checks.start("Pinned recipe constants match the checkpoint's processor_config.json")
    try:
        on_disk = checkpoint_recipe()
    except (FileNotFoundError, ValueError) as exc:
        checks.fail("recipe_matches_checkpoint", str(exc))
        return checks.report()
    pinned = pinned_recipe()
    disagreements = [
        f"{key}: pinned {pinned[key]!r} != checkpoint {on_disk[key]!r}"
        for key in RECIPE_KEYS
        if pinned[key] != on_disk[key]
    ]
    if disagreements:
        checks.fail(
            "recipe_matches_checkpoint",
            "the pinned recipe drifted from the checkpoint — the whole PAR-05 verdict "
            f"rests on these six values: {'; '.join(disagreements)}",
        )
        return checks.report()
    checks.ok(
        "recipe_matches_checkpoint",
        "six values agree: "
        + ", ".join(f"{key}={on_disk[key]!r}" for key in RECIPE_KEYS),
    )

    frame = synthetic_frame(args.seed)
    square = np.random.RandomState(args.seed + 1).randint(
        0, 255, (SHORTEST_IMAGE_EDGE, SHORTEST_IMAGE_EDGE, 3), dtype=np.uint8
    )

    outputs: dict[str, np.ndarray] = {
        "checkpoint_recipe": transform_checkpoint_recipe(frame),
        "placeholder_corrupted": transform_after_placeholder_resize(frame),
        "square_256": transform_checkpoint_recipe(square),
        "letterbox": transform_checkpoint_recipe(frame, letter_box_transform=True),
    }

    # --- 2: the verdict itself ----------------------------------------------
    checks.start(f"480x{SOURCE_WIDTH} through the checkpoint recipe is {EXPECTED_SHAPE}")
    got = outputs["checkpoint_recipe"]
    if tuple(got.shape) != EXPECTED_SHAPE or got.dtype != np.uint8:
        checks.fail(
            "checkpoint_recipe_shape",
            f"observed {tuple(got.shape)} {got.dtype}, expected {EXPECTED_SHAPE} uint8 "
            "— the recipe drifted from the checkpoint, or upstream's crop/resize "
            "arithmetic changed",
        )
    else:
        checks.ok(
            "checkpoint_recipe_shape",
            f"observed {tuple(got.shape)} {got.dtype} (exact integers, no tolerance)",
        )

    # --- 3: the corruption it must be distinguished from --------------------
    checks.start(f"The C-3 placeholder pre-resize lands on {EXPECTED_CORRUPTED_SHAPE}")
    corrupted = outputs["placeholder_corrupted"]
    if tuple(corrupted.shape) != EXPECTED_CORRUPTED_SHAPE or corrupted.dtype != np.uint8:
        checks.fail(
            "corrupted_shape",
            f"observed {tuple(corrupted.shape)} {corrupted.dtype}, expected "
            f"{EXPECTED_CORRUPTED_SHAPE} uint8",
        )
    else:
        checks.ok(
            "corrupted_shape",
            f"observed {tuple(corrupted.shape)} {corrupted.dtype}",
        )

    # --- 4: the corruption is discriminable --------------------------------
    checks.start("The correct output and the corrupted output are NOT equal")
    if tuple(got.shape) == tuple(corrupted.shape) and np.array_equal(got, corrupted):
        checks.fail(
            "corruption_discriminable",
            "the two outputs are identical — a shape-only cross-container comparison "
            "would NOT be discriminating, which invalidates the stated rationale for "
            "using one",
        )
    else:
        checks.ok(
            "corruption_discriminable",
            f"{tuple(got.shape)} != {tuple(corrupted.shape)}; a shape check catches it",
        )

    # --- 5: image_crop_size is inert, by measurement -----------------------
    checks.start("image_crop_size is inert while crop_fraction is set")
    widened = transform_checkpoint_recipe(frame, image_crop_size=[999, 999])
    crop_size_inert = bool(
        tuple(widened.shape) == tuple(got.shape) and np.array_equal(got, widened)
    )
    if not crop_size_inert:
        checks.fail(
            "crop_size_inert",
            f"[230, 230] and [999, 999] produced different output ({tuple(got.shape)} "
            f"vs {tuple(widened.shape)}) — crop_fraction is no longer taking "
            "precedence (processor_groot.py:1453-1454)",
        )
    else:
        checks.ok(
            "crop_size_inert",
            "[230, 230] and [999, 999] outputs are byte-identical: [230, 230] is dead "
            "configuration on this checkpoint",
        )

    # --- 6: replay-identical under the serving configuration ---------------
    checks.start("Two invocations on the same frame are byte-identical")
    replay = transform_checkpoint_recipe(frame)
    if not np.array_equal(got, replay):
        checks.fail(
            "replay_identical",
            "two invocations differ — the serving path should take the deterministic "
            "CENTER crop, not the train-time random crop",
        )
    else:
        checks.ok(
            "replay_identical",
            "exact array equality across two invocations (center crop, not random crop)",
        )

    manifest: dict[str, Any] = {
        "verdict": {
            "source_shape": [SOURCE_HEIGHT, SOURCE_WIDTH, 3],
            "expected_shape": list(EXPECTED_SHAPE),
            "expected_corrupted_shape": list(EXPECTED_CORRUPTED_SHAPE),
            "effective_pipeline": (
                "resize-shortest-edge-to-256 -> center-crop-95% -> "
                "resize-shortest-edge-to-256"
            ),
            "stale_roadmap_framing": (
                "'341x256-crop vs 256x256-letterbox' is a misread (REQUIREMENTS.md:50): "
                "256x341 is the intermediate after the first resize-shortest-edge, and "
                "256x256 is the letterbox branch this checkpoint does not take"
            ),
        },
        "recipe": on_disk,
        "recipe_source": str(PROCESSOR_CONFIG.relative_to(REPO_ROOT)),
        "seed": int(args.seed),
        "cases": {
            name: {
                "shape": _shape_of(outputs[name]),
                "dtype": str(outputs[name].dtype),
            }
            for name in CASE_NAMES
        },
        "crop_size_inert": crop_size_inert,
        "replay_identical": bool(np.array_equal(got, replay)),
    }
    manifest["cases"]["checkpoint_recipe"]["input_shape"] = [
        SOURCE_HEIGHT,
        SOURCE_WIDTH,
        3,
    ]
    manifest["cases"]["placeholder_corrupted"]["input_shape"] = [
        PLACEHOLDER_EDGE,
        PLACEHOLDER_EDGE,
        3,
    ]
    manifest["cases"]["square_256"]["input_shape"] = [
        SHORTEST_IMAGE_EDGE,
        SHORTEST_IMAGE_EDGE,
        3,
    ]
    manifest["cases"]["letterbox"]["input_shape"] = [SOURCE_HEIGHT, SOURCE_WIDTH, 3]

    npz_path, manifest_path = write_dump(Path(args.outdir), outputs, manifest)
    print(f"\nWrote {npz_path}")
    print(f"Wrote {manifest_path}")

    return checks.report()


if __name__ == "__main__":
    sys.exit(main())
