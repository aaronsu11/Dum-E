"""Keyless geometry gate pinning PAR-05's verdict as exact integers.

**The verdict, up front:** this checkpoint's effective image geometry is
*resize-shortest-edge-to-256 -> center-crop-95% -> resize-shortest-edge-to-256*,
which turns a 480x640x3 uint8 frame into exactly **``(256, 340, 3)``** uint8. It
is a resolution, not a note: the numbers are integers produced by a deterministic
crop-then-resize, so every assertion here is exact equality with no tolerance,
no ``pytest.approx`` and no ``numpy.allclose``. A recorded verdict that hedged
would let a real preprocessing shift pass in Phase 7, where this verdict is what
makes the parity numbers mean anything. See ``docs/LEROBOT-SERVING-VERDICTS.md``
for the resolution these tests defend.

**The ROADMAP's framing is stale, and both of its halves are documented here
rather than omitted.** It asks to settle "341x256-crop vs 256x256-letterbox
empirically"; REQUIREMENTS.md:50 already records that as a misread, and the real
answer is neither option. 256x341 is a real *intermediate* — the first
resize-shortest-edge of a 480x640 frame — and 256x256 is the *letterbox* branch
this checkpoint does not take (``letter_box_transform: false``).
``test_letterbox_branch_is_256x256x3_and_this_checkpoint_does_not_take_it`` and
``test_square_256_input_is_256x256x3`` exist to DOCUMENT those two negatives;
neither is ever the verdict's evidence.

``image_crop_size: [230, 230]`` is **provably inert** on this checkpoint, and
that is proven here by MEASUREMENT — a byte-identical output against
``[999, 999]`` — not by citing upstream's source. It is consulted only when
``crop_fraction is None`` (``processor_groot.py:1453-1454``), so it is dead
configuration a future reader must not waste time tuning.

Every test is hermetic: it reads only the checkpoint's ``processor_config.json``
and the pinned constants in ``scripts/dump_preprocessed_image.py``, and needs no
GPU, no weights, no network, no Hugging Face token and no hardware. **Nothing
here may skip** — an absent checkpoint ``processor_config.json`` calls
``pytest.fail`` naming the path, because a geometry test that skips is a silent
pass on the one fact Phase 7's parity numbers rest on.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make scripts/ importable so the tests and the probe share ONE source of truth
# for the pinned numbers (same idiom as tests/test_units_verdict.py:32-40).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from dump_preprocessed_image import (  # noqa: E402
    CROP_FRACTION,
    EXPECTED_CORRUPTED_SHAPE,
    EXPECTED_SHAPE,
    IMAGE_CROP_SIZE,
    IMAGE_TARGET_SIZE,
    LETTER_BOX_TRANSFORM,
    PROCESSOR_CONFIG,
    RECIPE_KEYS,
    SHORTEST_IMAGE_EDGE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    USE_ALBUMENTATIONS,
    checkpoint_recipe,
    pinned_recipe,
    synthetic_frame,
    transform_after_placeholder_resize,
    transform_checkpoint_recipe,
)

# The four shape assertions below spell their expected tuples as LOCAL LITERALS
# on purpose, and that is not a duplication to "clean up": the literal IS the
# thing under test. Asserting `== EXPECTED_SHAPE` alone would pass for whatever
# value that constant happened to hold, which is exactly the drift this module
# exists to catch. Every OTHER number here is imported, never restated.


def _on_disk_recipe() -> dict:
    """The checkpoint's six geometry values, or a loud failure naming the path."""
    if not PROCESSOR_CONFIG.is_file():
        pytest.fail(
            f"checkpoint processor config not found at {PROCESSOR_CONFIG} — PAR-05's "
            "geometry verdict cannot be asserted against the real recipe, and a skip "
            "here would be a silent pass on the fact Phase 7's parity numbers rest on"
        )
    return checkpoint_recipe(PROCESSOR_CONFIG)


# --- The four distinct shape cases (the boundary edge) ------------------------


def test_checkpoint_recipe_on_480x640_is_256x340x3():
    """THE VERDICT: 480x640 through the checkpoint's own recipe is (256, 340, 3).

    Exact tuple equality on integers, dtype uint8. No tolerance anywhere: the
    transform is a deterministic cv2 INTER_AREA resize plus a floored center
    crop, so an approximate pass would hide a genuine geometry change.
    """
    out = transform_checkpoint_recipe(synthetic_frame(0))
    assert tuple(out.shape) == (256, 340, 3)
    assert out.dtype == np.uint8
    # And the shared constant the probe, the manifest and the verdict doc all
    # read is pinned to the same integers.
    assert EXPECTED_SHAPE == (256, 340, 3)


def test_placeholder_pre_resize_corruption_is_256x256x3_and_differs():
    """The C-3 corruption lands on (256, 256, 3), NOT (256, 340, 3).

    ``from_pretrained`` injects a single ``observation.images.camera``
    placeholder at ``(3, 224, 224)`` and ``prepare_raw_observation`` resizes
    every camera to it, so a single-camera observation is silently squared to
    224x224 before the checkpoint's geometry runs.

    This test carries the assertion that makes a shape-only cross-container
    comparison genuinely DISCRIMINATING rather than merely convenient: the
    corruption changes the shape, so a shape check catches it.
    """
    corrupted = transform_after_placeholder_resize(synthetic_frame(0))
    assert tuple(corrupted.shape) == (256, 256, 3)
    assert corrupted.dtype == np.uint8
    assert EXPECTED_CORRUPTED_SHAPE == (256, 256, 3)
    # The load-bearing half: corrupted and correct are distinguishable by shape.
    assert tuple(corrupted.shape) != tuple(EXPECTED_SHAPE)


def test_square_256_input_is_256x256x3():
    """DOCUMENTS the stale framing's other half: output width tracks input aspect.

    A square input gives a square output, so ``(256, 256, 3)`` is reachable
    without the letterbox branch — which is why "256x256" alone identifies
    nothing. This is the "one step either side" case for the aspect-ratio
    threshold, and it is never the verdict's evidence.
    """
    square = np.random.RandomState(1).randint(
        0, 255, (SHORTEST_IMAGE_EDGE, SHORTEST_IMAGE_EDGE, 3), dtype=np.uint8
    )
    out = transform_checkpoint_recipe(square)
    assert tuple(out.shape) == (256, 256, 3)
    assert out.dtype == np.uint8


def test_letterbox_branch_is_256x256x3_and_this_checkpoint_does_not_take_it():
    """DOCUMENTS the ROADMAP's "256x256-letterbox" half, and proves it is not ours.

    The letterbox branch is real and reachable — forcing it on the same 480x640
    frame gives ``(256, 256, 3)`` — but this checkpoint's own
    ``processor_config.json`` sets ``letter_box_transform: false``, so the
    serving path never enters it. Recorded as a named test rather than as prose,
    so the negative cannot be "simplified" away later. This test is never the
    verdict's evidence; it exists to close off an alternative explanation.
    """
    out = transform_checkpoint_recipe(synthetic_frame(0), letter_box_transform=True)
    assert tuple(out.shape) == (256, 256, 3)
    assert out.dtype == np.uint8
    # The checkpoint on disk does NOT take that branch.
    assert _on_disk_recipe()["letter_box_transform"] is False
    assert LETTER_BOX_TRANSFORM is False


# --- The provably-inert config value -----------------------------------------


def test_image_crop_size_is_inert_when_crop_fraction_is_set():
    """``image_crop_size: [230, 230]`` is dead configuration — proven, not cited.

    ``processor_groot.py:1453-1454`` consults it only inside
    ``if crop_fraction is None and image_crop_size is not None``, and this
    checkpoint sets ``crop_fraction: 0.95``. Substituting an absurd
    ``[999, 999]`` therefore produces a BYTE-IDENTICAL output. Asserted by
    measurement rather than by reading the docs, and recorded so a future reader
    does not tune a value that does nothing.
    """
    frame = synthetic_frame(0)
    baseline = transform_checkpoint_recipe(frame)
    widened = transform_checkpoint_recipe(frame, image_crop_size=[999, 999])
    assert tuple(baseline.shape) == tuple(widened.shape)
    assert np.array_equal(baseline, widened)
    # The precondition the inertness depends on, asserted rather than assumed.
    assert CROP_FRACTION is not None
    assert _on_disk_recipe()["image_crop_size"] == IMAGE_CROP_SIZE


# --- The replay-identical property -------------------------------------------


def test_transform_replays_identically_under_the_serving_configuration():
    """Two invocations on one frame are byte-identical: exact, never a tolerance.

    This is the KEYLESS half of criterion 5's determinism claim, and only that
    half. The serving configuration takes the deterministic CENTER-crop branch:
    the train-time random crop is gated on ``self.training and
    torch.is_grad_enabled()``, and both are false on the serving path
    (``training`` is a constructor kwarg set from ``dataset_meta``, which
    ``policy_server`` never passes, and ``predict_action_chunk`` is decorated
    ``@torch.no_grad()``).

    **This is NOT a claim about the policy's flow-matching sampler**, which draws
    initial noise from the ambient torch RNG over ``num_inference_timesteps: 4``.
    That half is measured in-container by plan 06-03 and must be reported as
    "deterministic under an in-process seed". Nothing here may be cited as
    evidence that the server honours a seed — ``RemotePolicyConfig`` has no seed
    field, and Phase 5 recorded ``seed_verdict: not-honored`` for the sibling
    GR00T-native server.
    """
    frame = synthetic_frame(0)
    first = transform_checkpoint_recipe(frame)
    second = transform_checkpoint_recipe(frame)
    assert np.array_equal(first, second)


# --- The drift catcher -------------------------------------------------------


def test_recipe_constants_match_the_checkpoint_processor_config():
    """The pinned recipe still equals the checkpoint on disk, value by value.

    The whole verdict rests on these six values, so a checkpoint swap that
    changes one of them must go RED here rather than silently producing a
    different shape somewhere downstream. On an absent file this calls
    ``pytest.fail`` naming the path — never ``pytest.skip``.
    """
    on_disk = _on_disk_recipe()
    pinned = pinned_recipe()
    assert set(on_disk) == set(RECIPE_KEYS)
    for key in RECIPE_KEYS:
        assert on_disk[key] == pinned[key], (
            f"{key}: checkpoint has {on_disk[key]!r} but "
            f"scripts/dump_preprocessed_image.py pins {pinned[key]!r}"
        )
    # Named individually too, so a failure report says WHICH value moved.
    assert on_disk["letter_box_transform"] == LETTER_BOX_TRANSFORM
    assert on_disk["crop_fraction"] == CROP_FRACTION
    assert on_disk["image_crop_size"] == IMAGE_CROP_SIZE
    assert on_disk["image_target_size"] == IMAGE_TARGET_SIZE
    assert on_disk["shortest_image_edge"] == SHORTEST_IMAGE_EDGE
    assert on_disk["use_albumentations"] == USE_ALBUMENTATIONS
    # The frame size the verdict was measured AT is part of the contract: the
    # (256, 340, 3) result does not generalize to another aspect ratio.
    assert (SOURCE_HEIGHT, SOURCE_WIDTH) == (480, 640)
