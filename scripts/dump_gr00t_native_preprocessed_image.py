#!/usr/bin/env python3
"""The GR00T-NATIVE half of PAR-05: what geometry does Isaac-GR00T produce?

**Recorded verdict, up front: the two backends now AGREE — and the agreement was
engineered, not discovered.** On the SAME ``RandomState(0)`` 480x640x3 uint8 frame
and the SAME six recipe values from
``checkpoints/GR00T-N1.7-3B-SO101/processor_config.json``:

=================================================  ==================  =======
Isaac-GR00T ``image_augmentations`` (here, eval)    ``(256, 256, 3)``   —
Dum-E's LeRobot **SERVING** path (pad FORCED on)    ``(256, 256, 3)``   MATCH
LeRobot's UNPATCHED transform (pad gated off)       ``(256, 340, 3)``   history
=================================================  ==================  =======

The match is byte-exact, not merely shape-exact: both hash to
:data:`GR00T_NATIVE_EVAL_OUTPUT_SHA256`.

**HISTORY, kept because the fix only makes sense against it.** This probe's first
run recorded a MISMATCH: LeRobot ``(256, 340, 3)`` vs Isaac ``(256, 256, 3)``.
That reading is still true of the UNPATCHED upstream transform
(:data:`LEROBOT_UNPATCHED_OUTPUT_SHA256`, and
:data:`GEOMETRY_MATCHES_LEROBOT_UNPATCHED` is still ``False``). What changed is
Dum-E's serving path, which now forces the pad on.

**Where the divergence came from** (both sides read, not guessed):

- LeRobot gates the letterbox pad on the flag: ``if letter_box_transform:`` wraps
  the ``cv2.copyMakeBorder`` call (``processor_groot.py:1423-1433``), and this
  checkpoint sets ``letter_box_transform: false``, so no pad happened and the
  480x640 aspect ratio survived to the output.
- Isaac-GR00T applies ``LetterBoxPad()`` **unconditionally** as step 1 of both the
  train and eval albumentations pipelines
  (``gr00t/model/gr00t_n1d7/image_augmentations.py:420-487``), and its
  ``Gr00tN1d7Processor`` documents ``letter_box_transform`` under
  ``# Backward-compat params (stored but not actively used)``
  (``processing_gr00t_n1d7.py:171-172, 198``). The frame is padded to 640x640
  first, so the output is square.

So it was never an interpolation or rounding difference that shape-only fidelity
might have been hiding — it was a whole pipeline stage that one backend ran and the
other skipped, changing the aspect ratio of every non-square camera frame.

**And the cause was isolated to exactly that one stage, by measurement.** Isaac's
eval output hashes to :data:`GR00T_NATIVE_EVAL_OUTPUT_SHA256`, and LeRobot's own
transform with the pad forced ON hashes to the **same** value — **byte-identical
across two Python versions (3.10 vs 3.12), two numpy majors (1.26.4 vs 2.2.6) and
two OpenCV builds (4.11.0 vs 4.13.0)**. Every other stage — the ``INTER_AREA``
resizes, the floored 95% center crop — therefore agrees bit-for-bit between the
backends, which is what made a one-stage fix sufficient. The bit-exactness is
recorded as an *observation*, not promoted to a contract: §1's cross-container
fidelity decision stays **shape-only**, because a future OpenCV build is still
entitled to move a pixel.

**WHICH SIDE WAS MADE TO MATCH, AND ON WHAT INFERENCE.** The operator's decision
was to match Isaac: these weights were trained by Isaac's code, so the geometry
they learned on is Isaac's padded square, and LeRobot's flag-honouring behaviour —
which reads as more correct in isolation — was the deviation from training-time
behaviour. **That is an INFERENCE, accepted knowingly.** "Training used Isaac's
geometry" follows from "Isaac trained the checkpoint"; the actual training recipe
was **not read**, and no local artifact records it. If Phase 7's parity work
disappoints, this assumption is the first thing to re-examine. It is not settled
fact and must not be written up as one.

This module is deliberately built so the state cannot rot in either direction:
check 9 FAILS if Isaac's geometry stops matching the recorded LeRobot SERVING
shape, and also if it starts matching the recorded UNPATCHED shape — a stale
"settled match" is a silent regression exactly as a stale "known mismatch" is a
false alarm.

**Posture, mirroring ``scripts/dump_preprocessed_image.py``:** no GPU, no weights,
no network (run it with ``--network none``), no Hugging Face token, no hardware
and no live server. It calls the module-level pure builder
``build_image_transformations_albumentations`` directly and never constructs
``Gr00tN1d7Processor`` — whose ``__init__`` would call ``build_processor`` and pull
the Cosmos backbone. It never starts ``gr00t-server``.

Nothing here reimplements a resize or a crop on either side. A hand-rolled copy
would *manufacture* the mismatch this probe exists to characterise.

**One source of truth for the input.** The frame and the six recipe values come
from ``scripts/dump_preprocessed_image.py`` — imported, not re-typed — so the two
containers cannot drift into comparing different bytes. That the bytes really are
identical across the two interpreters is **proven, not assumed**: ``--expect-frame-sha256``
is mandatory and check 3 fails without it (the containers run numpy 1.26.4 on
Python 3.10 and numpy 2.2.6 on Python 3.12; ``RandomState`` is the frozen legacy
API, and this check is what turns that documented guarantee into a measurement).

**Image provenance is a CHECK, not a claim.** ``gr00t:latest`` carries no
``.git``, no ``gr00t.__version__``, no revision label and no build arg, so its
Isaac-GR00T revision cannot be read off the image. ``--expect-package-digest`` is
therefore mandatory too: check 1 recomputes a digest over every ``.py`` in the
installed ``gr00t`` package and fails if it disagrees, which is what lets the
recorded verdict name a revision instead of naming an unidentified image.

Usage — host side, compute the two expectations from the pinned clone::

    uv run python scripts/dump_gr00t_native_preprocessed_image.py --emit-expectations \
        --package-dir /path/to/Isaac-GR00T/gr00t

Usage — inside the image, run the dump (both expectations required)::

    docker run --rm --network none \
      -v "$PWD/scripts:/probe/scripts:ro" \
      -v "$PWD/checkpoints/GR00T-N1.7-3B-SO101/processor_config.json:/probe/processor_config.json:ro" \
      --entrypoint python gr00t:latest /probe/scripts/dump_gr00t_native_preprocessed_image.py \
        --recipe-json /probe/processor_config.json \
        --expect-frame-sha256 <hex> --expect-package-digest <hex>

``--outdir`` is supported but defaults to writing nothing: under ``docker run`` it
would leave root-owned files in the host tree. The numbers that matter are
transcribed into ``docs/LEROBOT-SERVING-VERDICTS.md`` and pinned by
``tests/test_par05_image_geometry.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Ensure this script's own directory is importable so the LeRobot-side probe's
# constants can be shared verbatim (same idiom as that probe's sys.path fix-up).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dump_preprocessed_image import (  # noqa: E402
    CROP_FRACTION,
    EXPECTED_SHAPE,
    IMAGE_CROP_SIZE,
    IMAGE_TARGET_SIZE,
    RECIPE_KEYS,
    SERVING_EXPECTED_SHAPE,
    SERVING_LETTER_BOX_TRANSFORM,
    SHORTEST_IMAGE_EDGE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    Checks,
    checkpoint_recipe,
    pinned_recipe,
    synthetic_frame,
)

# --- The measured verdict, pinned as exact integers ---------------------------

#: **THE GR00T-NATIVE MEASUREMENT.** The same 480x640x3 uint8 frame through
#: Isaac-GR00T's own eval pipeline, HWC as albumentations returns it. Exact
#: integers, no tolerance — the pipeline is a deterministic pad/resize/crop.
GR00T_NATIVE_EXPECTED_SHAPE = (256, 256, 3)

#: The same output as the real serving call site returns it: ``apply_with_replay``
#: hands back a CHW ``torch.uint8`` tensor (``image_augmentations.py:38-40``), so
#: the layouts differ as well as the geometry. Recorded separately so a reader
#: cannot mistake the layout difference for the geometry difference.
GR00T_NATIVE_EXPECTED_CHW_SHAPE = (3, 256, 256)

#: **True, and that is the finding.** Isaac-GR00T's ``(256, 256, 3)`` IS Dum-E's
#: LeRobot SERVING geometry, because the serving path forces the letterbox pad on
#: (``policy_guard.groot_guard.SERVING_LETTER_BOX_TRANSFORM``). Check 9 fails if a
#: future measurement stops matching, so a stale "settled match" cannot hide a
#: silent regression.
GEOMETRY_MATCHES_LEROBOT_SERVING = True

#: **False, and that is the HISTORY — kept, not deleted.** Isaac's geometry does not
#: match what LeRobot's UNPATCHED transform produces for this checkpoint's own
#: ``letter_box_transform: false``. This is the mismatch this probe originally
#: recorded and the reason the serving path overrides the flag; check 9 fails if it
#: ever becomes True, because then the override would be a no-op and the whole
#: justification would need re-recording rather than silently rotting.
GEOMETRY_MATCHES_LEROBOT_UNPATCHED = False

#: The Isaac-GR00T revision the verdict is recorded against — the pin enforced by
#: ``scripts/build_gr00t_image.sh`` (tag ``n1.7-release``).
GR00T_PIN = "23ace64f17aa5015259b8609d371eb61a357c776"

#: The image the measurement was taken in, by ID. ``gr00t:latest`` is a local tag
#: with no registry digest, so the ID is the only stable identifier it has.
GR00T_IMAGE_ID = "sha256:e263056fffe7a60a7f48b6309a8b8f2fb3ea9f8f2afa9c94a0105ed5b7d2eeaf"

#: :func:`package_py_digest` over the 64 ``.py`` files of the ``gr00t`` package —
#: identical in ``gr00t:latest`` and in the clone at :data:`GR00T_PIN`. This
#: content match is what stands in for the revision label the image does not
#: carry.
GR00T_PACKAGE_DIGEST = "b18c8578c077cddf02705b80da815e5d838752cab9f391c19e4edca7a6e74a40"

#: The number of ``.py`` files the digest covers, recorded so a truncated tree
#: cannot match by covering fewer files.
GR00T_PACKAGE_PY_COUNT = 64

#: sha256 of Isaac-GR00T's eval output bytes for the seed-0 frame, measured in
#: ``gr00t:latest``. **Dum-E's LeRobot SERVING path produces this same digest**,
#: which is both the cross-backend agreement and what isolated the whole original
#: divergence to the pad gating rather than to interpolation or rounding.
#: ``test_lerobot_serving_path_reproduces_the_gr00t_native_bytes`` pins it
#: keylessly. If a host OpenCV change ever moves it, that is a real within-LeRobot
#: geometry change and must be RE-MEASURED in the image, never loosened to a
#: tolerance.
GR00T_NATIVE_EVAL_OUTPUT_SHA256 = (
    "c30150ec8d9d7ccb648aade0588aed2d18a356ab7f984a510dc57bb6c485927f"
)

#: sha256 of the UNPATCHED LeRobot transform's output bytes (pad OFF, as this
#: checkpoint DECLARES it) for the same frame. This is the HISTORY: the artifact
#: the serving path used to produce and no longer does. Recorded alongside the
#: value above so the two stay visibly different artifacts, and so "the fix
#: changed something" is an assertion rather than a claim.
LEROBOT_UNPATCHED_OUTPUT_SHA256 = (
    "e8e4939bac10afa7cced1edb739656e14deb1acbbd509b80dfcfd7810fd2ce5a"
)

#: sha256 of the shared seed-0 480x640x3 uint8 input frame, identical in both
#: interpreters — the precondition that makes the comparison a comparison.
SHARED_INPUT_FRAME_SHA256 = (
    "70eecaa5a18341fcf3e9d22091d94d92e9d4c5aef7e839ae94ff3d11fd6612ee"
)

#: The case names recorded in the manifest.
CASE_NAMES = ("gr00t_native_eval", "gr00t_native_eval_chw", "gr00t_native_train")


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


# --- Provenance ---------------------------------------------------------------


def package_py_digest(package_dir: Path) -> tuple[str, int]:
    """A single digest over every ``.py`` file in a ``gr00t`` package tree.

    Returns ``(digest, file_count)``. The listing is ``"<sha256>  <relpath>"``
    lines sorted as byte strings (so it does not depend on the caller's locale,
    which is exactly how a first attempt at this comparison went wrong), joined
    with ``\\n`` and newline-terminated. ``relpath`` is relative to the package's
    PARENT, so it starts ``gr00t/`` on both sides and the digest is comparable
    between a git clone and an installed copy at a different absolute path.

    Deliberately content-based: the image carries no ``.git``, no
    ``__version__``, no OCI revision label and no recorded build arg, so hashing
    what is actually installed is the only proof of revision available.
    """
    package_dir = package_dir.resolve()
    parent = package_dir.parent
    lines: list[str] = []
    for path in package_dir.rglob("*.py"):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(parent).as_posix()}")
    lines.sort()
    payload = "\n".join(lines) + "\n" if lines else ""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), len(lines)


def installed_package_dir() -> Path:
    """The directory of the importable ``gr00t`` package, or a loud failure."""
    import gr00t  # noqa: PLC0415 — import kept local so the host can run --emit-expectations

    init = getattr(gr00t, "__file__", None)
    if not init:
        raise RuntimeError("the imported gr00t package has no __file__ to locate")
    return Path(init).resolve().parent


def version_markers() -> dict[str, Any]:
    """Every revision marker the image *might* have carried, and whether it did.

    Recorded as data rather than prose so the verdict can state precisely which
    proofs were unavailable instead of implying the pin was simply assumed.
    """
    import gr00t  # noqa: PLC0415

    package_dir = installed_package_dir()
    return {
        "gr00t.__version__": getattr(gr00t, "__version__", None),
        "package_parent_has_dot_git": (package_dir.parent / ".git").exists(),
        "python": sys.version.split()[0],
    }


# --- The GR00T-native transforms ---------------------------------------------


def build_native_transforms(recipe: dict[str, Any]) -> tuple[Any, Any]:
    """Isaac-GR00T's own ``(train, eval)`` albumentations transforms.

    Calls the module-level builder ``Gr00tN1d7Processor.__init__`` itself calls
    when ``use_albumentations`` is true (``processing_gr00t_n1d7.py:230-241``),
    with the checkpoint's own values. Deliberately NOT via the processor class:
    its ``__init__`` calls ``build_processor(model_name, ...)``, which would pull
    the Cosmos backbone and make PAR-05 unreachable under this phase's posture.

    ``letter_box_transform`` is NOT forwarded, because the builder takes no such
    parameter — that is the divergence, not an omission (see the module
    docstring).
    """
    from gr00t.model.gr00t_n1d7.image_augmentations import (  # noqa: PLC0415
        build_image_transformations_albumentations,
    )

    return build_image_transformations_albumentations(
        recipe["image_target_size"],
        recipe["image_crop_size"],
        None,  # random_rotation_angle — absent from this checkpoint's kwargs
        None,  # color_jitter_params — absent from this checkpoint's kwargs
        recipe["shortest_image_edge"],
        recipe["crop_fraction"],
    )


def transform_native_eval(transform: Any, frame: np.ndarray) -> np.ndarray:
    """The eval (deterministic, center-crop) pipeline, HWC as cv2 produces it."""
    return transform(image=frame)["image"]


def transform_native_eval_serving_call_site(transform: Any, frame: np.ndarray) -> Any:
    """The eval pipeline through the call the serving path actually makes.

    ``Gr00tN1d7Processor`` feeds PIL images to ``apply_with_replay``
    (``processing_gr00t_n1d7.py:405-408``), which returns CHW ``torch.uint8``.
    Exercised so the recorded geometry is the geometry the SERVER would see, not
    only the geometry a bare Compose call produces.
    """
    from PIL import Image  # noqa: PLC0415
    from gr00t.model.gr00t_n1d7.image_augmentations import apply_with_replay  # noqa: PLC0415

    tensors, _replay = apply_with_replay(transform, [Image.fromarray(frame)])
    return tensors[0]


# --- Reporting ---------------------------------------------------------------


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _shape_of(array: Any) -> list[int]:
    return [int(n) for n in array.shape]


def emit_expectations(package_dir: Path, seed: int) -> int:
    """Print the two mandatory expectations, computed from the pinned clone.

    Run on the HOST against the Isaac-GR00T clone at :data:`GR00T_PIN`; feed the
    output into the in-container invocation. Both sides compute them with the
    functions in this module, so the algorithm cannot differ between them.
    """
    digest, count = package_py_digest(package_dir)
    frame_sha = _sha256(synthetic_frame(seed))
    print(json.dumps(
        {
            "package_dir": str(package_dir),
            "expect_package_digest": digest,
            "package_py_count": count,
            "expect_frame_sha256": frame_sha,
            "seed": seed,
        },
        indent=2,
        sort_keys=True,
    ))
    print(
        "\n--expect-package-digest "
        f"{digest} --expect-frame-sha256 {frame_sha}",
        file=sys.stderr,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe-json",
        default="/probe/processor_config.json",
        help="The checkpoint's processor_config.json (source of truth for the six values)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the synthetic 480x640 frame (default: 0)"
    )
    parser.add_argument(
        "--expect-frame-sha256",
        default=None,
        help="REQUIRED: sha256 of the LeRobot-side frame bytes (see --emit-expectations)",
    )
    parser.add_argument(
        "--expect-package-digest",
        default=None,
        help="REQUIRED: package_py_digest of the gr00t package at the pinned revision",
    )
    parser.add_argument(
        "--package-dir",
        default=None,
        help="gr00t package directory (default: the importable one)",
    )
    parser.add_argument(
        "--emit-expectations",
        action="store_true",
        help="Host mode: print the two expectations for --package-dir and exit",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Optional: write dumps.npz + manifest.json here (default: write nothing)",
    )
    args = parser.parse_args()

    if args.emit_expectations:
        if not args.package_dir:
            print(
                _red("--emit-expectations needs --package-dir <clone>/gr00t"),
                file=sys.stderr,
            )
            return 2
        return emit_expectations(Path(args.package_dir), args.seed)

    checks = Checks(total=9)

    # --- 1: the image under test is the pinned revision, by content ----------
    checks.start(
        "The installed gr00t package matches the pinned Isaac-GR00T revision by content"
    )
    try:
        package_dir = Path(args.package_dir) if args.package_dir else installed_package_dir()
        digest, count = package_py_digest(package_dir)
        markers = version_markers()
    except Exception as exc:  # noqa: BLE001 — any failure here is a provenance failure
        checks.fail("package_digest_matches_pin", f"could not hash the gr00t package: {exc}")
        return checks.report()
    if not args.expect_package_digest:
        checks.fail(
            "package_digest_matches_pin",
            f"--expect-package-digest was not supplied. Observed digest {digest} over {count} "
            ".py files, but this image carries no .git, no gr00t.__version__ and no revision "
            "label, so an unchecked digest proves nothing about which revision produced it — "
            "and a verdict recorded against an unidentified image is what Phase 7 must not "
            "inherit. Compute the expectation with --emit-expectations against the clone at "
            f"{GR00T_PIN}",
        )
    elif digest != args.expect_package_digest:
        checks.fail(
            "package_digest_matches_pin",
            f"observed {digest} over {count} .py files, expected {args.expect_package_digest} — "
            "this is NOT the pinned revision's preprocessing code, so the geometry it produces "
            "cannot be attributed to it",
        )
    else:
        checks.ok(
            "package_digest_matches_pin",
            f"{count} .py files digest to {digest}, matching the clone at {GR00T_PIN}; "
            f"markers absent as expected ({markers})",
        )

    # --- 2: the recipe is the checkpoint's, not this file's ------------------
    checks.start("The six recipe values read from the checkpoint match the pinned recipe")
    try:
        on_disk = checkpoint_recipe(Path(args.recipe_json))
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
            "the pinned recipe drifted from the checkpoint, so the two backends would be "
            f"compared under different settings: {'; '.join(disagreements)}",
        )
        return checks.report()
    checks.ok(
        "recipe_matches_checkpoint",
        "six values agree: " + ", ".join(f"{key}={on_disk[key]!r}" for key in RECIPE_KEYS),
    )

    # --- 3: the two backends really see the same bytes ----------------------
    checks.start("The synthetic frame is byte-identical to the LeRobot-side frame")
    frame = synthetic_frame(args.seed)
    frame_sha = _sha256(frame)
    if not args.expect_frame_sha256:
        checks.fail(
            "same_input_bytes",
            f"--expect-frame-sha256 was not supplied; observed {frame_sha}. Without it the "
            "comparison rests on two independently generated frames, which is not a "
            "comparison at all",
        )
    elif frame_sha != args.expect_frame_sha256:
        checks.fail(
            "same_input_bytes",
            f"observed {frame_sha}, expected {args.expect_frame_sha256} — the two interpreters "
            "are NOT generating the same frame (numpy RandomState stream drift), so any shape "
            "comparison between them is meaningless",
        )
    else:
        checks.ok("same_input_bytes", f"sha256 {frame_sha} on a {frame.shape} {frame.dtype} frame")

    train_transform, eval_transform = build_native_transforms(on_disk)

    outputs: dict[str, Any] = {
        "gr00t_native_eval": transform_native_eval(eval_transform, frame),
        "gr00t_native_eval_chw": transform_native_eval_serving_call_site(eval_transform, frame),
        "gr00t_native_train": transform_native_eval(train_transform, frame),
    }

    # --- 4: the measurement -------------------------------------------------
    checks.start(
        f"{SOURCE_HEIGHT}x{SOURCE_WIDTH} through Isaac-GR00T's eval pipeline is "
        f"{GR00T_NATIVE_EXPECTED_SHAPE}"
    )
    native = outputs["gr00t_native_eval"]
    if tuple(native.shape) != GR00T_NATIVE_EXPECTED_SHAPE or native.dtype != np.uint8:
        checks.fail(
            "gr00t_native_shape",
            f"observed {tuple(native.shape)} {native.dtype}, expected "
            f"{GR00T_NATIVE_EXPECTED_SHAPE} uint8 — re-measure BOTH backends and re-record the "
            "verdict; the recorded divergence is stale",
        )
    else:
        checks.ok(
            "gr00t_native_shape",
            f"observed {tuple(native.shape)} {native.dtype} (exact integers, no tolerance)",
        )

    # --- 5: the same geometry through the real serving call site ------------
    checks.start(
        f"apply_with_replay (the serving call site) returns {GR00T_NATIVE_EXPECTED_CHW_SHAPE}"
    )
    chw = outputs["gr00t_native_eval_chw"]
    if tuple(chw.shape) != GR00T_NATIVE_EXPECTED_CHW_SHAPE:
        checks.fail(
            "serving_call_site_shape",
            f"observed {tuple(chw.shape)}, expected {GR00T_NATIVE_EXPECTED_CHW_SHAPE} — the bare "
            "Compose call and the call the server makes disagree, so neither can be reported as "
            "the server's geometry",
        )
    else:
        checks.ok(
            "serving_call_site_shape",
            f"observed {tuple(chw.shape)} {chw.dtype}: same H/W as the HWC case, CHW layout",
        )

    # --- 6: the train branch does not explain the divergence ----------------
    checks.start("The train pipeline yields the same H/W, so the eval/train branch is not the cause")
    train_out = outputs["gr00t_native_train"]
    if tuple(train_out.shape)[:2] != GR00T_NATIVE_EXPECTED_SHAPE[:2]:
        checks.fail(
            "train_branch_same_geometry",
            f"train pipeline gives {tuple(train_out.shape)} vs eval "
            f"{tuple(native.shape)} — picking a branch changes the geometry, so the recorded "
            "comparison must name which branch it measured and why",
        )
    else:
        checks.ok(
            "train_branch_same_geometry",
            f"train {tuple(train_out.shape)} shares H/W with eval {tuple(native.shape)}: the "
            "random-vs-center crop changes WHERE, not HOW BIG, so the divergence cannot be "
            "explained away as a branch mistake",
        )

    # --- 7: the eval pipeline is deterministic ------------------------------
    checks.start("Two eval invocations on the same frame are byte-identical")
    replay = transform_native_eval(eval_transform, frame)
    if not np.array_equal(native, replay):
        checks.fail(
            "eval_replay_identical",
            "two eval invocations differ — the eval Compose should take the deterministic "
            "FractionalCenterCrop, not the train pipeline's FractionalRandomCrop",
        )
    else:
        checks.ok(
            "eval_replay_identical",
            "exact array equality across two invocations (center crop, not random crop)",
        )

    # --- 8: the bytes, and what they isolate --------------------------------
    checks.start(
        "The eval output bytes match the recorded digest (the same digest Dum-E's LeRobot "
        "SERVING path produces)"
    )
    native_sha = _sha256(native)
    if native_sha != GR00T_NATIVE_EVAL_OUTPUT_SHA256:
        checks.fail(
            "eval_output_bytes_recorded",
            f"observed {native_sha}, recorded {GR00T_NATIVE_EVAL_OUTPUT_SHA256} — the "
            "byte-level cross-backend agreement no longer holds. RE-MEASURE both backends "
            "and re-record; do not relax this to a tolerance",
        )
    else:
        checks.ok(
            "eval_output_bytes_recorded",
            f"{native_sha}: byte-identical to the LeRobot SERVING path's output "
            "across cv2 4.11.0/4.13.0, numpy 1.26/2.2 and Python 3.10/3.12, so every stage "
            "agrees bit-for-bit once the pad is forced on",
        )

    # --- 9: the recorded state, pinned in BOTH directions -------------------
    checks.start(
        f"The recorded cross-backend verdict still holds (serving match expected: "
        f"{GEOMETRY_MATCHES_LEROBOT_SERVING}; unpatched match expected: "
        f"{GEOMETRY_MATCHES_LEROBOT_UNPATCHED})"
    )
    observed_match = tuple(native.shape) == tuple(SERVING_EXPECTED_SHAPE)
    observed_unpatched_match = tuple(native.shape) == tuple(EXPECTED_SHAPE)
    if (
        observed_match != GEOMETRY_MATCHES_LEROBOT_SERVING
        or observed_unpatched_match != GEOMETRY_MATCHES_LEROBOT_UNPATCHED
    ):
        checks.fail(
            "recorded_divergence_holds",
            f"Isaac-GR00T {tuple(native.shape)} vs LeRobot serving "
            f"{tuple(SERVING_EXPECTED_SHAPE)} (match {observed_match}, recorded "
            f"{GEOMETRY_MATCHES_LEROBOT_SERVING}) and vs LeRobot unpatched "
            f"{tuple(EXPECTED_SHAPE)} (match {observed_unpatched_match}, recorded "
            f"{GEOMETRY_MATCHES_LEROBOT_UNPATCHED}). Re-record the verdict in "
            "docs/LEROBOT-SERVING-VERDICTS.md — a stale 'settled match' is a silent "
            "regression and a stale 'known mismatch' is a false alarm",
        )
    else:
        checks.ok(
            "recorded_divergence_holds",
            f"Isaac-GR00T {tuple(native.shape)} MATCHES the LeRobot serving geometry "
            f"{tuple(SERVING_EXPECTED_SHAPE)} (letter_box_transform forced to "
            f"{SERVING_LETTER_BOX_TRANSFORM}) and still DIFFERS from the unpatched "
            f"{tuple(EXPECTED_SHAPE)}, exactly as recorded",
        )

    if not observed_match:
        print(
            _red(
                "\n"
                + "!" * 72
                + "\n!! PAR-05 CROSS-BACKEND GEOMETRY MISMATCH HAS RETURNED\n"
                f"!!   LeRobot serving (recorded) : {tuple(SERVING_EXPECTED_SHAPE)}\n"
                f"!!   Isaac-GR00T   (measured)   : {tuple(native.shape)}\n"
                "!! Same frame bytes, same six recipe values. The forced letterbox pad\n"
                "!! in docker/lerobot-policy/server.py is what made these agree; check\n"
                "!! whether it still lands (SAFE-01/5) before touching anything else.\n"
                "!! Phase 7's numerical parity comparison is NOT meaningful while they\n"
                "!! disagree -- the two backends would see different pixels.\n"
                + "!" * 72
            )
        )

    manifest: dict[str, Any] = {
        "verdict": {
            "source_shape": [SOURCE_HEIGHT, SOURCE_WIDTH, 3],
            "lerobot_serving_shape": list(SERVING_EXPECTED_SHAPE),
            "lerobot_unpatched_shape": list(EXPECTED_SHAPE),
            "gr00t_native_shape": _shape_of(native),
            "geometry_match_serving": bool(observed_match),
            "recorded_geometry_match_serving": GEOMETRY_MATCHES_LEROBOT_SERVING,
            "geometry_match_unpatched": bool(observed_unpatched_match),
            "recorded_geometry_match_unpatched": GEOMETRY_MATCHES_LEROBOT_UNPATCHED,
            "gr00t_native_effective_pipeline": (
                "letterbox-pad-to-square (UNCONDITIONAL) -> resize-shortest-edge-to-256 -> "
                "center-crop-95% -> resize-shortest-edge-to-256"
            ),
            "original_divergence_cause": (
                "Isaac-GR00T's build_image_transformations_albumentations puts LetterBoxPad() "
                "first in both pipelines and treats letter_box_transform as a stored-but-unused "
                "backward-compat param; LeRobot's _transform_n1_7_image_for_vlm_albumentations "
                "gates the pad on that flag, which this checkpoint sets false"
            ),
            "resolution": (
                "Dum-E's serving path FORCES letter_box_transform=True at the config seam in "
                "docker/lerobot-policy/server.py (one definition: "
                "policy_guard.groot_guard.serving_preprocessor_overrides), so the served "
                "geometry is Isaac's padded square. SAFE-01/5 reads the effective value off "
                "the built step and refuses the handshake if the override did not land."
            ),
            "inference_boundary": (
                "Matching Isaac rests on an INFERENCE accepted knowingly: 'training used "
                "Isaac's geometry' follows from 'Isaac trained this checkpoint'. The training "
                "recipe was NOT read and no local artifact records it. Re-examine this first "
                "if Phase 7 parity disappoints; do not present it as settled fact."
            ),
        },
        "provenance": {
            "pin": GR00T_PIN,
            "image_id": GR00T_IMAGE_ID,
            "package_dir": str(package_dir),
            "package_py_digest": digest,
            "package_py_count": count,
            "package_digest_expectation": args.expect_package_digest,
            "version_markers": markers,
        },
        "recipe": on_disk,
        "recipe_source": str(args.recipe_json),
        "seed": int(args.seed),
        "input_frame_sha256": frame_sha,
        "cases": {
            name: {
                "shape": _shape_of(outputs[name]),
                "dtype": str(outputs[name].dtype),
                "input_shape": [SOURCE_HEIGHT, SOURCE_WIDTH, 3],
            }
            for name in CASE_NAMES
        },
        "eval_output_sha256": native_sha,
        "eval_output_sha256_recorded": GR00T_NATIVE_EVAL_OUTPUT_SHA256,
        "lerobot_unpatched_output_sha256_recorded": LEROBOT_UNPATCHED_OUTPUT_SHA256,
        "eval_replay_identical": bool(np.array_equal(native, replay)),
        "environment": {
            "numpy": np.__version__,
            "cv2": __import__("cv2").__version__,
            "albumentations": __import__("albumentations").__version__,
        },
    }

    if args.outdir:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        arrays = {
            "gr00t_native_eval": np.ascontiguousarray(native),
            "gr00t_native_train": np.ascontiguousarray(train_out),
        }
        npz_path = outdir / "dumps.npz"
        np.savez_compressed(npz_path, **arrays)
        # Proven, not asserted: a pickled object array cannot survive this reload.
        # (savez_compressed takes no allow_pickle argument — every keyword it
        # receives would be stored as an array, see the LeRobot-side probe.)
        with np.load(npz_path, allow_pickle=False) as verify:
            for name in arrays:
                _ = verify[name]
        (outdir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nWrote {npz_path}")
        print(f"Wrote {outdir / 'manifest.json'}")

    print("\n--- manifest ---")
    print(json.dumps(manifest, indent=2, sort_keys=True))

    return checks.report()


if __name__ == "__main__":
    sys.exit(main())
