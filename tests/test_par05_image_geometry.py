"""Keyless geometry gate pinning PAR-05's verdict as exact integers.

**TWO verdicts, and keeping them apart is what this module is for.** Both are
measured, both are pinned, and collapsing them would erase the reason the serving
path is configured the way it is:

* **The SERVING path** — what Dum-E's ``lerobot-policy`` server actually feeds the
  VLM — is *letterbox-pad-to-square (FORCED) -> resize-shortest-edge-to-256 ->
  center-crop-95% -> resize-shortest-edge-to-256*, turning a 480x640x3 uint8 frame
  into exactly **``(256, 256, 3)``** uint8. **This is the verdict Phase 7 must
  use.**
* **The UNPATCHED upstream transform**, run as this checkpoint's own
  ``letter_box_transform: false`` configures it, still gives **``(256, 340, 3)``**.
  Kept and asserted, because it is what makes the forced pad a real change rather
  than a no-op.

Every assertion is exact equality with no tolerance, no ``pytest.approx`` and no
``numpy.allclose``: the numbers are integers produced by a deterministic
crop-then-resize, and a verdict that hedged would let a real preprocessing shift
pass in Phase 7. See ``docs/LEROBOT-SERVING-VERDICTS.md`` for the resolution these
tests defend.

**Why the serving path overrides the checkpoint's own flag — and the inference that
rests on.** Isaac-GR00T trained these weights and applies ``LetterBoxPad()``
unconditionally, treating ``letter_box_transform`` as a stored-but-unused
backward-compat parameter; LeRobot honours the flag, so it produced a geometry Isaac
never emitted. Forcing the pad on makes the two outputs BYTE-identical. **The
operator's decision to match Isaac rests on an inference accepted knowingly:**
"training used Isaac's geometry" follows from "Isaac trained the checkpoint" — the
actual training recipe was **not read**, and no local artifact records it. If
Phase 7's parity work disappoints, that is the first assumption to re-examine. It is
not settled fact.

**The ROADMAP's framing is stale, and both of its halves are documented here
rather than omitted.** It asks to settle "341x256-crop vs 256x256-letterbox
empirically"; REQUIREMENTS.md:50 already records that as a misread. 256x341 is a
real *intermediate* — the first resize-shortest-edge of a 480x640 frame — and
256x256 was the *letterbox* branch this checkpoint's config declines, which the
serving path now takes anyway for the reason above.
``test_letterbox_branch_is_256x256x3_and_this_checkpoint_declares_it_off`` and
``test_square_256_input_is_256x256x3`` exist to DOCUMENT those two negatives;
neither is ever the verdict's evidence.

``image_crop_size: [230, 230]`` is **provably inert** on this checkpoint, and
that is proven here by MEASUREMENT — a byte-identical output against
``[999, 999]`` — not by citing upstream's source. It is consulted only when
``crop_fraction is None`` (``processor_groot.py:1453-1454``), so it is dead
configuration a future reader must not waste time tuning.

**The cross-backend comparison PAR-05 asks for now AGREES, and the agreement was
engineered rather than discovered.** Isaac-GR00T produces ``(256, 256, 3)`` on the
same frame bytes under the same six recipe values; the serving path reproduces it
byte-for-byte. The recorded MISMATCH against the unpatched transform is kept as
history, pinned in both directions, so neither reading can rot: a stale "settled
match" is a silent regression exactly as a stale "known mismatch" is a false alarm.

**Honest limit on what CI can re-run, stated rather than papered over.** The
GR00T-native *measurement* needs the 42.8 GB ``gr00t`` image and cannot run here.
What runs here instead is not a placeholder: the LeRobot serving pipeline is BUILT
for real and its geometry read off the constructed step, reproducing the
GR00T-native digest locally. What no test in this repo asserts is Isaac's
*unconditional* pad — that half is carried by the in-container probe (9/9 checks,
image ID and revision verified by content hash) and by the source citation in its
docstring. There is deliberately **no** test that imports ``gr00t``: it would skip
everywhere, and a test that can never run is worse than this note.

Every test is hermetic: it reads the checkpoint's ``processor_config.json``, the
pinned constants in ``scripts/dump_preprocessed_image.py`` and
``scripts/dump_gr00t_native_preprocessed_image.py``, and — for the serving-path
cases — builds the real processor pipeline, which is weight-free, GPU-free and
network-free because the encode step's Qwen processor is lazy and never touched.
**Nothing here may skip** — an absent checkpoint ``processor_config.json`` calls
``pytest.fail`` naming the path, because a geometry test that skips is a silent pass
on the one fact Phase 7's parity numbers rest on.
"""

import ast
import hashlib
import inspect
import sys
import textwrap
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
    SERVING_EXPECTED_SHAPE,
    SERVING_GEOMETRY_FIELDS,
    SERVING_LETTER_BOX_TRANSFORM,
    SERVING_OUTPUT_SHA256,
    SHORTEST_IMAGE_EDGE,
    SOURCE_HEIGHT,
    SOURCE_WIDTH,
    CAMERA_KEYS as PROBE_CAMERA_KEYS,
    Checks,
    USE_ALBUMENTATIONS,
    build_serving_preprocessor,
    checkpoint_recipe,
    pinned_recipe,
    placeholder_squared,
    serving_geometry_settings,
    synthetic_frame,
    transform_after_placeholder_resize,
    transform_as_served,
    transform_checkpoint_recipe,
)
from dump_gr00t_native_preprocessed_image import (  # noqa: E402
    GEOMETRY_MATCHES_LEROBOT_SERVING,
    GEOMETRY_MATCHES_LEROBOT_UNPATCHED,
    GR00T_IMAGE_ID,
    GR00T_NATIVE_EXPECTED_CHW_SHAPE,
    GR00T_NATIVE_EXPECTED_SHAPE,
    GR00T_NATIVE_EVAL_OUTPUT_SHA256,
    GR00T_PACKAGE_DIGEST,
    GR00T_PACKAGE_PY_COUNT,
    GR00T_PIN,
    LEROBOT_UNPATCHED_OUTPUT_SHA256,
    SHARED_INPUT_FRAME_SHA256,
)

#: The built serving pipeline's geometry settings, resolved ONCE per session.
#: Building it is cheap (no weights, no network, lazy Qwen processor) but not free,
#: and every serving-path test must measure the SAME pipeline the others do.
_CACHED_SERVED_SETTINGS: dict | None = None


def served_settings() -> dict:
    """The five geometry settings read off the REAL built serving pipeline.

    Read off the constructed ``GrootN17VLMEncodeStep``, never restated: a
    "serving path" number derived from this file's own opinion of the settings would
    prove nothing about what the server does.
    """
    global _CACHED_SERVED_SETTINGS
    if _CACHED_SERVED_SETTINGS is None:
        if not PROCESSOR_CONFIG.is_file():
            pytest.fail(
                f"checkpoint processor config not found at {PROCESSOR_CONFIG} — the "
                "serving pipeline cannot be built, and a skip here would be a silent pass "
                "on the geometry the policy actually sees"
            )
        _CACHED_SERVED_SETTINGS = serving_geometry_settings(build_serving_preprocessor())
    return _CACHED_SERVED_SETTINGS

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


def test_serving_path_on_480x640_is_256x256x3():
    """**THE SERVING VERDICT**: 480x640 through the SERVED pipeline is (256, 256, 3).

    Measured on the pipeline the ``lerobot-policy`` server actually builds — the five
    geometry settings are read off its constructed ``GrootN17VLMEncodeStep`` and
    handed to the same transform that step calls — so this is what the VLM sees, not
    a restatement of an intention.

    Exact tuple equality on integers, dtype uint8. No tolerance anywhere: the
    transform is a deterministic cv2 INTER_AREA resize plus a floored center crop, so
    an approximate pass would hide a genuine geometry change.
    """
    out = transform_as_served(synthetic_frame(0), served_settings())
    assert tuple(out.shape) == (256, 256, 3)
    assert out.dtype == np.uint8
    assert SERVING_EXPECTED_SHAPE == (256, 256, 3)


def test_serving_pipeline_carries_the_forced_letterbox_pad():
    """The override LANDED on the step that will transform frames.

    This is the load-bearing wiring assertion on the keyless side: the shape verdict
    above is only the serving path's verdict if the pad is genuinely forced on the
    built pipeline. The checkpoint DECLARES the pad off, so reading ``True`` here can
    only have come from the override
    (``policy_guard.groot_guard.serving_preprocessor_overrides``, applied in
    ``docker/lerobot-policy/server.py``) — it cannot be inherited from the
    configuration by accident.

    The other four settings are asserted to be the CHECKPOINT's, so the override is
    proven surgical: exactly one stage changed.
    """
    served = served_settings()
    assert set(served) == set(SERVING_GEOMETRY_FIELDS)
    assert served["letter_box_transform"] is True
    assert SERVING_LETTER_BOX_TRANSFORM is True
    # The checkpoint says the opposite, which is what makes the line above evidence.
    assert _on_disk_recipe()["letter_box_transform"] is False
    # Surgical: only the pad moved.
    assert served["crop_fraction"] == CROP_FRACTION
    assert served["shortest_image_edge"] == SHORTEST_IMAGE_EDGE
    assert served["image_crop_size"] == IMAGE_CROP_SIZE
    assert served["image_target_size"] == IMAGE_TARGET_SIZE


def test_probe_refuses_to_report_a_verdict_when_a_check_never_runs():
    """A vanished probe check FAILS; it is not silently absent (WR-07 / T-06-37).

    This is the MIRROR of ``test_preflight_refuses_when_a_check_never_runs``. The
    T-06-37 fix landed in ``docker/lerobot-policy/entrypoint.py``'s ``Checks`` copy
    (commit ``fb5ae11``) but not in this one — the copy whose output IS the committed
    geometry verdict: ``docs/LEROBOT-SERVING-VERDICTS.md`` cites "9/9 checks PASS,
    exit 0" for BOTH probes, and ``scripts/dump_gr00t_native_preprocessed_image.py``
    imports this very class. Under the unfixed contract, deleting a check from
    ``main()`` printed ``8/8 checks passed`` and exited 0.

    Both halves are pinned, because asserting only the happy path is how this
    regressed in the first place.
    """
    complete = Checks(3)
    for i in range(3):
        complete.start(f"check {i}")
        complete.ok(f"name-{i}", "fine")
    assert complete.report() == 0

    # One check never ran. Every RECORDED check passed, so the original contract
    # returned 0 here -- that is precisely the hole.
    truncated = Checks(3)
    for i in range(2):
        truncated.start(f"check {i}")
        truncated.ok(f"name-{i}", "fine")
    assert truncated.report() != 0, (
        "a probe missing a check must refuse to report a verdict; every recorded "
        "check passing is not evidence that every expected check ran"
    )

    # A recorded failure still fails, so the new clause did not replace the old one.
    failing = Checks(2)
    failing.start("check 0")
    failing.ok("name-0", "fine")
    failing.start("check 1")
    failing.fail("name-1", "nope")
    assert failing.report() != 0


def test_both_probe_harnesses_share_one_checks_contract():
    """The two probes' ``9/9`` verdicts come from ONE class, and its contract matches
    the container entrypoint's.

    ``dump_gr00t_native_preprocessed_image.py`` imports ``Checks`` from
    ``dump_preprocessed_image``, so there is one definition behind both recorded
    verdicts. The entrypoint holds a genuinely separate copy (``scripts/`` does not
    exist inside the image), and the two must agree on the exit contract — the
    ``fb5ae11`` fix landing in one and not the other is what WR-07 reports.
    """
    import dump_gr00t_native_preprocessed_image as native  # noqa: PLC0415

    assert native.Checks is Checks

    entrypoint_source = (
        REPO_ROOT / "docker" / "lerobot-policy" / "entrypoint.py"
    ).read_text(encoding="utf-8")
    probe_source = (REPO_ROOT / "scripts" / "dump_preprocessed_image.py").read_text(
        encoding="utf-8"
    )
    contract = "return 0 if passed == len(self.results) == self.total else 1"
    assert entrypoint_source.count(contract) == 1, "the entrypoint's exit contract moved"
    assert probe_source.count(contract) == 1, (
        "the probe harness no longer carries the same exit contract as the container "
        "entrypoint's copy; a check deleted from main() would report a verdict"
    )


def test_probe_geometry_matches_the_client_handshake_definition():
    """The probe's THIRD copy of the camera keys and frame size tracks the definition.

    WR-06. ``policy/lerobot/features.py`` is the definition; this module and
    ``docker/lerobot-policy/server.py`` each hold a pinned copy, for the same reason
    :data:`SERVING_LETTER_BOX_TRANSFORM` is a copy — this module is imported inside the
    ``gr00t:latest`` container by ``scripts/dump_gr00t_native_preprocessed_image.py``,
    where ``policy`` and ``lerobot`` do not exist, so a module-scope import would make
    the cross-backend probe unrunnable.

    Why it matters HERE specifically: the probe builds the real serving preprocessor
    from ``config.input_features``, and those features carry the camera NAMES and the
    frame SIZE. A drift would make the measured verdict describe a pipeline the server
    does not run, while every number in the manifest still looked self-consistent.
    """
    from policy.lerobot import features

    assert (SOURCE_HEIGHT, SOURCE_WIDTH) == (features.FRAME_HEIGHT, features.FRAME_WIDTH)
    assert set(PROBE_CAMERA_KEYS) == set(features.CAMERA_KEYS), (
        f"the probe builds a pipeline for {tuple(PROBE_CAMERA_KEYS)} but the handshake "
        f"declares {tuple(features.CAMERA_KEYS)}"
    )


def test_serving_letterbox_constant_matches_the_guards_definition():
    """The probe's pinned copy and the guard's DEFINITION are one value.

    ``scripts/dump_preprocessed_image.py`` pins ``SERVING_LETTER_BOX_TRANSFORM`` as a
    COPY rather than importing it, because that module is also imported inside the
    ``gr00t:latest`` container where ``policy_guard`` (and the ``lerobot`` it imports
    at module scope) do not exist. This test is what stops the copy drifting from the
    definition the server actually injects.
    """
    from policy_guard.groot_guard import (
        SERVING_LETTER_BOX_TRANSFORM as GUARD_SERVING_LETTER_BOX_TRANSFORM,
        VLM_ENCODE_STEP_KEY,
        serving_preprocessor_overrides,
    )

    assert SERVING_LETTER_BOX_TRANSFORM == GUARD_SERVING_LETTER_BOX_TRANSFORM
    # And the override fragment the server merges really carries that value under a
    # key that matches a real step of the built pipeline.
    overrides = serving_preprocessor_overrides()
    assert overrides == {VLM_ENCODE_STEP_KEY: {"letter_box_transform": True}}
    step_keys = {
        getattr(type(step), "_registry_name", None) or type(step).__name__
        for step in build_serving_preprocessor().steps
    }
    assert VLM_ENCODE_STEP_KEY in step_keys, (
        f"the override key {VLM_ENCODE_STEP_KEY!r} matches no step of the built "
        f"pipeline (available: {sorted(step_keys)}), so the forced pad would be dropped"
    )


def test_unpatched_transform_on_480x640_is_256x340x3():
    """HISTORY, pinned: the UNPATCHED transform still gives (256, 340, 3).

    Run exactly as this checkpoint configures it (``letter_box_transform: false``).
    Kept and asserted for two reasons: it is what makes the forced pad a real change
    rather than a no-op, and a silent move in upstream's resize/crop arithmetic must
    still go red somewhere.

    This is NOT the serving path's geometry — see
    ``test_serving_path_on_480x640_is_256x256x3``.
    """
    out = transform_checkpoint_recipe(synthetic_frame(0))
    assert tuple(out.shape) == (256, 340, 3)
    assert out.dtype == np.uint8
    # And the shared constant the probe, the manifest and the verdict doc all
    # read is pinned to the same integers.
    assert EXPECTED_SHAPE == (256, 340, 3)
    # The two geometries are genuinely different artifacts, not two names for one.
    assert tuple(out.shape) != tuple(SERVING_EXPECTED_SHAPE)
    assert not np.array_equal(out, transform_as_served(synthetic_frame(0), served_settings()))


def test_placeholder_pre_resize_corruption_is_256x256x3_and_differs_only_unpatched():
    """The C-3 corruption, and the HONEST cost of the forced pad.

    ``from_pretrained`` injects a single ``observation.images.camera``
    placeholder at ``(3, 224, 224)`` and ``prepare_raw_observation`` resizes
    every camera to it, so a single-camera observation is silently squared to
    224x224 before the checkpoint's geometry runs.

    Under the UNPATCHED transform that corruption CHANGES the shape, which is what
    made a shape-only cross-container comparison discriminating rather than merely
    convenient — asserted below.

    **On the SERVING path it no longer does, and that is asserted rather than quietly
    dropped.** The forced pad squares every input, so the corrupted and correct
    serving outputs share a shape and differ only in pixels. What carries C-3 now is
    PREVENTION, not detection: ``fixup_policy_features`` sets ``input_features``
    before ``from_pretrained``, so the placeholder branch — guarded by
    ``config is None`` (``modeling_groot.py:247-261``) — never runs at all. A future
    reader must not inherit "a shape check catches C-3" as a live reassurance.
    """
    frame = synthetic_frame(0)
    corrupted = transform_after_placeholder_resize(frame)
    assert tuple(corrupted.shape) == (256, 256, 3)
    assert corrupted.dtype == np.uint8
    assert EXPECTED_CORRUPTED_SHAPE == (256, 256, 3)
    # Unpatched: corrupted and correct are distinguishable BY SHAPE.
    assert tuple(corrupted.shape) != tuple(EXPECTED_SHAPE)

    # Serving: same shape, different bytes. Both halves asserted — "same shape" alone
    # could be satisfied by the two outputs being identical, which would mean C-3 was
    # undetectable by ANY comparison rather than merely by a shape check.
    served = served_settings()
    corrupted_served = transform_as_served(placeholder_squared(frame), served)
    correct_served = transform_as_served(frame, served)
    assert tuple(corrupted_served.shape) == tuple(correct_served.shape) == (256, 256, 3)
    assert not np.array_equal(corrupted_served, correct_served)


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


def test_letterbox_branch_is_256x256x3_and_this_checkpoint_declares_it_off():
    """DOCUMENTS the ROADMAP's "256x256-letterbox" half, and who takes that branch.

    The letterbox branch is real and reachable — forcing it on the same 480x640
    frame gives ``(256, 256, 3)``. This checkpoint's own ``processor_config.json``
    DECLARES ``letter_box_transform: false``, so the unpatched transform never enters
    it; **the serving path enters it anyway**, by override, because Isaac's code took
    it when it trained these weights. Recorded as a named test rather than as prose,
    so the distinction cannot be "simplified" away later.
    """
    out = transform_checkpoint_recipe(synthetic_frame(0), letter_box_transform=True)
    assert tuple(out.shape) == (256, 256, 3)
    assert out.dtype == np.uint8
    # The checkpoint on disk DECLARES the branch off ...
    assert _on_disk_recipe()["letter_box_transform"] is False
    assert LETTER_BOX_TRANSFORM is False
    # ... and forcing it is exactly what the serving path does, byte for byte.
    assert np.array_equal(out, transform_as_served(synthetic_frame(0), served_settings()))


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
    # And on the geometry that actually serves, which is the one Phase 7 measures.
    served = served_settings()
    assert np.array_equal(transform_as_served(frame, served), transform_as_served(frame, served))


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
    # The frame size the verdict was measured AT is part of the contract: neither
    # result generalizes to another aspect ratio.
    assert (SOURCE_HEIGHT, SOURCE_WIDTH) == (480, 640)


# --- The cross-backend comparison: AGREEMENT, with the mismatch kept as history ---
#
# PAR-05 asks for a comparison "between backends", and this is that comparison's
# LeRobot-side anchor. The GR00T-native number was measured in `gr00t:latest`
# (image ID pinned below, Isaac-GR00T revision proven by a 64-file content
# digest) by `scripts/dump_gr00t_native_preprocessed_image.py`, 9/9 checks. The
# SERVING path matches it byte-for-byte; the UNPATCHED transform still does not.
# These tests pin both directions, so neither reading can decay: not the agreement
# into an unnoticed regression, and not the historical mismatch into a stale alarm.


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def test_gr00t_native_geometry_is_256x256x3_and_the_serving_path_matches_it():
    """THE AGREEMENT: Isaac-GR00T (256, 256, 3) == the LeRobot SERVING geometry.

    Local literals again, for the same reason the cases above use them: the literal
    IS the recorded finding, and asserting only against the imported constant would
    pass for whatever that constant drifted to.

    Both directions are pinned. The serving path MATCHES, which is what makes
    Phase 7's numerical parity comparison meaningful at all — the two backends now
    see the same pixels. The UNPATCHED transform still does NOT match, which is the
    historical mismatch and the reason the serving path overrides the flag.
    """
    assert GR00T_NATIVE_EXPECTED_SHAPE == (256, 256, 3)
    assert SERVING_EXPECTED_SHAPE == (256, 256, 3)
    assert GR00T_NATIVE_EXPECTED_SHAPE == SERVING_EXPECTED_SHAPE
    assert GEOMETRY_MATCHES_LEROBOT_SERVING is True

    # History, not deleted: the unpatched transform diverges, and that is recorded
    # state rather than something implied by two constants merely differing.
    assert EXPECTED_SHAPE == (256, 340, 3)
    assert GR00T_NATIVE_EXPECTED_SHAPE != EXPECTED_SHAPE
    assert GEOMETRY_MATCHES_LEROBOT_UNPATCHED is False

    # The serving call site's CHW layout carries the same H/W, so the layout
    # difference cannot be mistaken for a geometry difference.
    assert GR00T_NATIVE_EXPECTED_CHW_SHAPE == (3, 256, 256)
    assert GR00T_NATIVE_EXPECTED_CHW_SHAPE[1:] == GR00T_NATIVE_EXPECTED_SHAPE[:2]


def test_lerobot_reproduces_the_gr00t_native_shape_only_with_the_pad_forced_on():
    """EXECUTED locally: LeRobot matches Isaac only BECAUSE the letterbox pad is forced.

    The load-bearing half of the diagnosis, and the reason this file can pin a
    cross-container measurement without the 42.8 GB image: the SERVED pipeline
    reproduces the GR00T-native shape, while the checkpoint's own configuration
    (pad off) does not. That is what identifies the pad gating — not interpolation,
    not the crop fraction, not ``image_crop_size`` — as the whole of the original
    divergence, and it is also what shows the agreement is DUE to the override rather
    than coincidental.
    """
    frame = synthetic_frame(0)
    served = transform_as_served(frame, served_settings())
    as_declared = transform_checkpoint_recipe(frame)
    assert tuple(served.shape) == GR00T_NATIVE_EXPECTED_SHAPE
    assert tuple(as_declared.shape) != GR00T_NATIVE_EXPECTED_SHAPE
    assert tuple(as_declared.shape) == (256, 340, 3)


def test_lerobot_serving_path_reproduces_the_gr00t_native_bytes():
    """The digests, exactly: Isaac's output and the SERVING path's output are equal.

    Byte-level, no tolerance. Measured across Python 3.10 vs 3.12, numpy 1.26.4
    vs 2.2.6 and OpenCV 4.11.0 vs 4.13.0.

    **This does not upgrade the cross-container fidelity contract.**
    ``docs/LEROBOT-SERVING-VERDICTS.md`` §1 keeps that leg at *shape only*,
    because a future OpenCV build may legitimately move a pixel. What is asserted
    here is the WITHIN-LeRobot exact-equality contract that section already
    states — the expected digest simply happens to be the value measured in the
    other container, which is what makes the recorded cross-container number
    verifiable here rather than merely quoted. If a host OpenCV upgrade turns
    this red, the response is to RE-MEASURE both backends and re-record, never to
    replace this with a tolerance.

    Two digests are asserted, not one: the unpatched output is a genuinely different
    artifact, so "the fix changed something" is measured rather than assumed.
    """
    frame = synthetic_frame(0)
    # The precondition of the whole comparison: both containers hashed the same
    # input bytes. Asserted here too, so a numpy RandomState stream change could
    # not silently invalidate the cross-container claim.
    assert _sha256(frame) == SHARED_INPUT_FRAME_SHA256
    assert _sha256(transform_as_served(frame, served_settings())) == (
        GR00T_NATIVE_EVAL_OUTPUT_SHA256
    )
    # The probe's own pinned copy of that digest agrees — two independent literals,
    # so the equality is an assertion rather than a definition.
    assert SERVING_OUTPUT_SHA256 == GR00T_NATIVE_EVAL_OUTPUT_SHA256
    # And the unpatched output is a different artifact, not a second name for it.
    assert _sha256(transform_checkpoint_recipe(frame)) == LEROBOT_UNPATCHED_OUTPUT_SHA256
    assert LEROBOT_UNPATCHED_OUTPUT_SHA256 != GR00T_NATIVE_EVAL_OUTPUT_SHA256


def test_lerobot_gates_the_letterbox_pad_on_the_flag_this_checkpoint_sets_false():
    """The MECHANISM of the divergence on the LeRobot side, proven at AST level.

    Isaac-GR00T puts ``LetterBoxPad()`` unconditionally at the head of both its
    albumentations pipelines and documents ``letter_box_transform`` as a
    "backward-compat param (stored but not actively used)". LeRobot instead wraps
    its ``cv2.copyMakeBorder`` in ``if letter_box_transform:``.

    **This gating is still the upstream behaviour and is still asserted here** — the
    fix does not remove it, it feeds it ``True``. That distinction is the point: had
    the pad been hoisted out of the branch upstream, the override would become a
    silent no-op that happened to look right, and this test would go red first.

    Asserted structurally rather than by grep: **every** ``copyMakeBorder`` call
    in the function must sit inside that conditional.
    """
    from lerobot.policies.groot.processor_groot import (
        _transform_n1_7_image_for_vlm_albumentations,
    )

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(_transform_n1_7_image_for_vlm_albumentations))
    )

    def pad_calls(node) -> int:
        return sum(
            1
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "copyMakeBorder"
        )

    gated = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "letter_box_transform"
    ]
    assert len(gated) == 1, "expected exactly one `if letter_box_transform:` branch"
    assert pad_calls(tree) >= 1, "no copyMakeBorder call found — the pad moved or was removed"
    # Every pad call in the function is inside the flag's branch.
    assert pad_calls(gated[0]) == pad_calls(tree)
    # And the checkpoint really does set that flag false, which is why the serving
    # path has to override it rather than inherit it.
    assert _on_disk_recipe()["letter_box_transform"] is False
    assert served_settings()["letter_box_transform"] is True


def test_the_gr00t_native_measurement_records_which_image_it_came_from():
    """Provenance is part of the verdict, not a footnote.

    ``gr00t:latest`` carries no ``.git``, no ``gr00t.__version__``, no OCI
    revision label and no recorded build arg, so the Isaac-GR00T revision cannot
    be read off the image. It was established by CONTENT instead: all
    ``GR00T_PACKAGE_PY_COUNT`` ``.py`` files of the installed ``gr00t`` package
    digest to ``GR00T_PACKAGE_DIGEST``, identical to the clone at ``GR00T_PIN``
    (the commit ``scripts/build_gr00t_image.sh`` enforces). These constants are
    pinned so the recorded measurement can never be re-attributed to a different
    image by editing prose.
    """
    assert GR00T_PIN == "23ace64f17aa5015259b8609d371eb61a357c776"
    assert GR00T_IMAGE_ID.startswith("sha256:")
    assert len(GR00T_IMAGE_ID.removeprefix("sha256:")) == 64
    assert len(GR00T_PACKAGE_DIGEST) == 64
    assert GR00T_PACKAGE_PY_COUNT == 64
    # The build wrapper's pin and the verdict's pin are ONE value, not two that
    # can drift: read it back off the wrapper rather than trusting the copy.
    wrapper = (REPO_ROOT / "scripts" / "build_gr00t_image.sh").read_text(encoding="utf-8")
    assert f'PIN="{GR00T_PIN}"' in wrapper
