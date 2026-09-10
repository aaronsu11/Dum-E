#!/usr/bin/env python3
"""Settle PAR-05 offline: what geometry does THIS checkpoint's image recipe produce?

**TWO verdicts, and keeping them apart is the whole point of this module.** Both
are true, both are measured, and collapsing them loses the reason the serving path
is configured the way it is:

============================================  ==================  =========================
What                                          A 480x640x3 frame   Constant
becomes
============================================  ==================  =========================
The UNPATCHED upstream transform, run as       ``(256, 340, 3)``   :data:`EXPECTED_SHAPE`
this checkpoint configures it
(``letter_box_transform: false``)
**Dum-E's SERVING path**, with the letterbox   ``(256, 256, 3)``   :data:`SERVING_EXPECTED_SHAPE`
pad FORCED on
============================================  ==================  =========================

The unpatched pipeline is *resize-shortest-edge-to-256 -> center-crop-95% ->
resize-shortest-edge-to-256*. The serving pipeline is that, with
*letterbox-pad-to-square* in front of it.

**Why the serving path overrides the checkpoint's own flag.** Isaac-GR00T, which
trained these weights, applies ``LetterBoxPad()`` UNCONDITIONALLY and treats
``letter_box_transform`` as a stored-but-unused backward-compat parameter, so the
geometry the weights were trained on is the padded square. LeRobot honours the flag
and therefore produced a geometry Isaac never emitted. Forcing the pad on makes
LeRobot's output BYTE-IDENTICAL to Isaac's
(:data:`SERVING_OUTPUT_SHA256`), which is what isolates the whole divergence to
that one stage. Full record: ``docs/LEROBOT-SERVING-VERDICTS.md``.

**The inference this rests on, stated because it was accepted knowingly and never
proven:** "the weights were trained on Isaac's padded square" is INFERRED from
"Isaac trained this checkpoint". The training recipe itself was **not read** — no
local artifact records it. If Phase 7's parity work disappoints, this is the first
assumption to re-examine. It is not a settled fact.

This probe needs **no GPU, no weights, no network, no Hugging Face token and no
hardware**. It calls the module-level pure function
``lerobot.policies.groot.processor_groot._transform_n1_7_image_for_vlm_albumentations``
directly — no model load, no ``PolicyServer`` subclass, no monkeypatch — which is
what makes PAR-05 reachable at all under this phase's compose-only posture. The
serving-path measurement goes one step further and BUILDS the real preprocessor
pipeline (``make_pre_post_processors``, with the server's own override fragment),
then reads the five geometry settings off the constructed ``GrootN17VLMEncodeStep``
and calls the same transform the step itself calls — so the number reported is the
served configuration's, not a restatement of it. That build is still weight-free
and network-free: the step's Qwen processor is lazy and is never touched.

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
import hashlib
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
#: False -> the checkpoint DECLARES no letterbox, and the unpatched upstream
#: transform honours that. Kept pinned as a checkpoint FACT (check 1 re-reads it),
#: and deliberately NOT the value the serving path uses — see
#: :data:`SERVING_LETTER_BOX_TRANSFORM`.
LETTER_BOX_TRANSFORM = False

#: **What Dum-E's SERVING path forces instead**, overriding the flag above so the
#: model sees the padded square Isaac's code produced at training time.
#:
#: This is a PINNED COPY, not the definition. The definition is
#: ``policy_guard.groot_guard.SERVING_LETTER_BOX_TRANSFORM``, which
#: ``docker/lerobot-policy/server.py`` injects at the config seam;
#: ``test_serving_letterbox_constant_matches_the_guards_definition`` asserts the two
#: agree on every suite run. It is copied rather than imported because this module is
#: ALSO imported inside the ``gr00t:latest`` container by
#: ``scripts/dump_gr00t_native_preprocessed_image.py``, where ``policy_guard`` (and
#: the ``lerobot`` it imports at module scope) do not exist — an import here would
#: make the cross-backend probe unrunnable.
SERVING_LETTER_BOX_TRANSFORM = True

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

#: **What the UNPATCHED upstream transform produces**, run exactly as this
#: checkpoint configures it (pad off). A 480x640x3 uint8 frame. Exact integers, no
#: tolerance: the pipeline is a deterministic crop-then-resize, so an approximate
#: match would let a real preprocessing shift pass.
#:
#: **This is no longer the serving path's output** — see
#: :data:`SERVING_EXPECTED_SHAPE`. It is kept, pinned and asserted because it is
#: what makes the forced pad a real change rather than a no-op, and because a
#: silent move in upstream's resize/crop arithmetic must still go red somewhere.
#:
#: The ROADMAP's "341x256-crop vs 256x256-letterbox" framing is STALE, and
#: REQUIREMENTS.md:50 already records it as a misread. Both halves of that
#: framing are real but neither described the answer at the time: 256x341 is the
#: INTERMEDIATE after the first resize-shortest-edge of a 480x640 frame, and
#: 256x256 was the letterbox branch this checkpoint's own config declines — which
#: the serving path now takes anyway, for the reason in the module docstring.
EXPECTED_SHAPE = (256, 340, 3)

#: **THE SERVING-PATH VERDICT, and the one Phase 7 must use.** The same frame
#: through the pipeline Dum-E's ``lerobot-policy`` server actually builds, with the
#: letterbox pad forced on. Byte-identical to Isaac-GR00T's own eval output
#: (:data:`SERVING_OUTPUT_SHA256`), which is the whole point of forcing it.
SERVING_EXPECTED_SHAPE = (256, 256, 3)

#: sha256 of the SERVING path's output bytes for the seed-0 frame. Equal, by
#: measurement, to Isaac-GR00T's ``GR00T_NATIVE_EVAL_OUTPUT_SHA256`` — recorded as
#: an independent literal here so that equality stays an ASSERTION rather than
#: becoming a definition (two names for one constant could not disagree, and a
#: comparison that cannot fail is not a comparison).
SERVING_OUTPUT_SHA256 = "c30150ec8d9d7ccb648aade0588aed2d18a356ab7f984a510dc57bb6c485927f"

#: What the C-3 corruption produces under the UNPATCHED transform. On that path the
#: corruption CHANGES the shape, which is what made a shape-only cross-container
#: comparison discriminating rather than merely convenient.
#:
#: **On the SERVING path it no longer does, and that is recorded rather than
#: quietly dropped.** The forced pad squares every input, so a 224x224-corrupted
#: frame and a correct 480x640 frame both emerge as ``(256, 256, 3)`` and differ
#: only in pixels (check 8 measures exactly that). What carries C-3 now is
#: PREVENTION, not detection: ``fixup_policy_features`` sets ``input_features``
#: before ``from_pretrained``, so the ``(3, 224, 224)`` placeholder branch — guarded
#: by ``config is None`` (``modeling_groot.py:247-261``) — never runs at all.
EXPECTED_CORRUPTED_SHAPE = (256, 256, 3)

#: The source frame size, taken from ``scripts/test_live_policy_server.py``'s
#: ``_synthetic_observation(height=480, width=640)`` rather than invented — BOTH
#: verdicts are measured AT 480x640 and neither generalizes to another aspect ratio.
#: The unpatched ``(256, 340, 3)`` obviously depends on it, since the output width
#: tracks the input aspect. The serving ``(256, 256, 3)`` is square for ANY input,
#: which makes it look aspect-independent and is exactly why the frame size is pinned
#: here: the padded square's CONTENT — how much of the frame is image and how much is
#: zero padding — still depends entirely on 480x640, and that is what the VLM sees.
#:
#: **PINNED COPIES of ``policy/lerobot/features.py``'s ``FRAME_HEIGHT``/``FRAME_WIDTH``**,
#: for the same reason :data:`SERVING_LETTER_BOX_TRANSFORM` above is a copy: this module is
#: also imported inside the ``gr00t:latest`` container by
#: ``scripts/dump_gr00t_native_preprocessed_image.py``, where ``policy`` and the ``lerobot``
#: it imports do not exist, so a module-scope import here would make the cross-backend probe
#: unrunnable. ``test_probe_geometry_matches_the_client_handshake_definition`` asserts the
#: copies against the definition on every suite run.
SOURCE_HEIGHT = 480
SOURCE_WIDTH = 640

#: The cameras the SERVING pipeline is built for. A THIRD pinned copy of
#: ``policy/lerobot/features.py``'s ``CAMERA_KEYS`` (``docker/lerobot-policy/server.py``
#: holds the second), named rather than inlined so the same keyless cross-check can reach
#: it. Order is irrelevant to THIS probe — it only decides which ``input_features`` keys
#: exist so the built pipeline is the served one — but the NAMES are not: a rename would
#: build a pipeline for cameras the server does not serve.
CAMERA_KEYS: tuple[str, ...] = ("wrist", "front")

#: The edge ``from_pretrained``'s placeholder feature squares every camera to.
PLACEHOLDER_EDGE = 224

#: The five case names recorded in the manifest and dumped to the .npz.
#: ``serving_path`` is the one that matters for Phase 7; ``checkpoint_recipe`` is the
#: unpatched baseline kept so the difference between them stays visible.
CASE_NAMES = (
    "checkpoint_recipe",
    "serving_path",
    "placeholder_corrupted",
    "square_256",
    "letterbox",
)


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


class Checks:
    """Numbered-check PASS/FAIL harness with a ``passed == total`` exit contract.

    Shape copied from ``scripts/capture_frozen_corpus.py:144-173``, and carrying the
    SAME repudiation fix as ``docker/lerobot-policy/entrypoint.py``'s copy (T-06-37,
    commit ``fb5ae11``). The two contracts are deliberately identical; do not resync
    either back to the original.

    The hole the fix closes: gating on ``passed == len(self.results)`` counts every
    *recorded* check, so a check DELETED from ``main()`` records nothing, the rest all
    pass, and the probe prints ``8/8 checks passed`` and exits 0 — indistinguishable
    from a probe that genuinely measured everything. That matters more here than in the
    entrypoint, because ``docs/LEROBOT-SERVING-VERDICTS.md`` cites "**9/9** checks
    PASS, exit 0" as the evidence behind the PAR-05 geometry verdict, and this class is
    what produces that line for BOTH probes (``dump_gr00t_native_preprocessed_image.py``
    imports it).

    ``report()`` therefore additionally requires ``len(self.results) == self.total``.
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
        # A check that never RAN is reported distinctly from a check that FAILED: the
        # two demand different responses (a code defect in main() vs a real geometry
        # drift), and conflating them is the repudiation failure T-06-37 names.
        # Early-return failure paths legitimately record fewer than `total`, but they
        # already carry a FAIL and so exit non-zero via the `passed` clause below.
        missing = self.total - len(self.results)
        if missing > 0 and passed == len(self.results):
            print(
                _red(
                    f"  FAIL: {missing} of {self.total} check(s) never ran. Every "
                    "recorded check passed, so this is not a geometry drift — a check "
                    "was removed from main() or `total` disagrees with it. Refusing to "
                    "report a verdict from an incomplete probe."
                )
            )
        print("=" * 72)
        return 0 if passed == len(self.results) == self.total else 1


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


#: The five geometry settings the encode step forwards to the transform it calls
#: (``processor_groot.py:2109-2118``). Read off the BUILT step by
#: :func:`serving_geometry_settings` rather than restated, so a measurement of the
#: "served" geometry cannot silently become a measurement of this file's opinion.
SERVING_GEOMETRY_FIELDS = (
    "image_crop_size",
    "image_target_size",
    "shortest_image_edge",
    "crop_fraction",
    "letter_box_transform",
)


def build_serving_preprocessor(checkpoint_dir: Path = CHECKPOINT_DIR) -> Any:
    """Build the REAL preprocessor pipeline the ``lerobot-policy`` server builds.

    Same public builder, same three override fragments — including the geometry
    override imported from ``policy_guard.groot_guard``, so this measures the
    server's configuration rather than a copy of it. Still weight-free, GPU-free and
    network-free: ``GrootN17VLMEncodeStep``'s Qwen processor is lazy (``_proc`` is
    ``init=False``, built only on first ``.proc`` access) and nothing here touches it.

    ``policy_guard`` is imported INSIDE the function on purpose. This module is also
    imported inside the ``gr00t:latest`` container by
    ``scripts/dump_gr00t_native_preprocessed_image.py``, where neither
    ``policy_guard`` nor ``lerobot`` exists; a module-scope import would make the
    cross-backend probe unrunnable.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

    from policy_guard.groot_guard import EXPECTED_TAG, serving_preprocessor_overrides

    config = GrootConfig(
        base_model_path=str(checkpoint_dir),
        embodiment_tag=EXPECTED_TAG,
        model_params_fp32=False,
    )
    # The same feature fixup ``docker/lerobot-policy/server.py`` applies before the
    # load: without it ``validate_features`` inserts a 132-wide action feature and the
    # pipeline is not the served one.
    config.input_features = {
        f"{OBS_IMAGES}.{cam}": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, SOURCE_HEIGHT, SOURCE_WIDTH)
        )
        for cam in CAMERA_KEYS
    }
    config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
    config.device = "cpu"

    overrides: dict[str, Any] = {
        "device_processor": {"device": "cpu"},
        "rename_observations_processor": {"rename_map": {}},
    }
    overrides.update(serving_preprocessor_overrides())
    preprocessor, _postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint_dir),
        preprocessor_overrides=overrides,
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return preprocessor


def serving_encode_step(preprocessor: Any) -> Any:
    """The built pipeline's image step, located by the marker the guard locates it by.

    Raises rather than returning ``None``: a "served geometry" measured off a step
    that could not be found would be a fabricated number.
    """
    candidates = [
        step for step in preprocessor.steps if hasattr(step, "letter_box_transform")
    ]
    if not candidates:
        raise RuntimeError(
            "no step of the built preprocessor carries `letter_box_transform`, so the "
            "SERVED image geometry cannot be read off the pipeline that will run it. "
            "The pinned lerobot release reshaped the processor pipeline; fix this "
            "against the new shape rather than measuring a stand-in."
        )
    return candidates[-1]


def serving_geometry_settings(preprocessor: Any) -> dict[str, Any]:
    """The five geometry settings, READ OFF the built step (never restated)."""
    step = serving_encode_step(preprocessor)
    return {name: getattr(step, name) for name in SERVING_GEOMETRY_FIELDS}


def transform_as_served(frame: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    """Run ``frame`` through the transform with the SERVED settings.

    ``settings`` comes from :func:`serving_geometry_settings`, i.e. off the built
    step, and this calls the very function the step calls with the very arguments it
    forwards (``processor_groot.py:2109-2118``). So this is the served pipeline's
    geometry, not a reimplementation of it — and nothing here re-derives a resize or a
    crop, which upstream documents as needing to stay bit-exact
    (``processor_groot.py:1394-1401``).

    ``crop_position`` is deliberately not passed: it defaults to ``None``, the
    deterministic CENTER crop, which is what the serving path takes
    (``training`` is False and ``predict_action_chunk`` is ``@torch.no_grad()``).
    """
    from lerobot.policies.groot.processor_groot import (
        _transform_n1_7_image_for_vlm_albumentations,
    )

    return _transform_n1_7_image_for_vlm_albumentations(frame, **settings)


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

    Under the UNPATCHED transform this corruption is CATCHABLE by a shape check: it
    lands on ``(256, 256, 3)``, not ``(256, 340, 3)``. Under the SERVING path it is
    NOT — the forced pad squares every input, so correct and corrupted frames share a
    shape and differ only in pixels. Check 7 measures both halves of that; C-3 is
    carried by prevention (``fixup_policy_features``), not by this shape.
    """
    return transform_checkpoint_recipe(placeholder_squared(frame))


def placeholder_squared(frame: np.ndarray) -> np.ndarray:
    """The C-3 pre-resize ALONE: a 480x640 frame squared to 224x224, HWC uint8.

    Split out from :func:`transform_after_placeholder_resize` so the same corruption
    can be fed to either geometry — the unpatched transform (where it changes the
    output shape) or the serving pipeline (where it no longer does). One
    implementation, so the two comparisons cannot diverge in how they corrupt.
    """
    import torch
    from lerobot.async_inference.helpers import resize_robot_observation_image

    squared = resize_robot_observation_image(
        torch.tensor(frame), (3, PLACEHOLDER_EDGE, PLACEHOLDER_EDGE)
    )
    # (C, H, W) back to the HxWx3 uint8 frame the checkpoint transform accepts.
    return np.ascontiguousarray(squared.permute(1, 2, 0).to(torch.uint8).numpy())


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


def _sha256(array: np.ndarray) -> str:
    """sha256 over an array's raw bytes. Same helper the GR00T-native probe uses, so
    the two containers hash identically rather than nearly-identically."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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

    checks = Checks(total=9)

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

    # The SERVING pipeline, built for real, with its geometry read off the step that
    # would transform frames. Weight-free and network-free (the step's Qwen processor
    # is lazy); this is what makes the "serving path" numbers below measurements of
    # the server's configuration rather than of this file's opinion.
    preprocessor = build_serving_preprocessor()
    served = serving_geometry_settings(preprocessor)

    outputs: dict[str, np.ndarray] = {
        "checkpoint_recipe": transform_checkpoint_recipe(frame),
        "serving_path": transform_as_served(frame, served),
        "placeholder_corrupted": transform_after_placeholder_resize(frame),
        "square_256": transform_checkpoint_recipe(square),
        "letterbox": transform_checkpoint_recipe(frame, letter_box_transform=True),
    }

    # --- 2: the UNPATCHED baseline ------------------------------------------
    checks.start(
        f"480x{SOURCE_WIDTH} through the UNPATCHED transform, as the checkpoint "
        f"configures it, is {EXPECTED_SHAPE}"
    )
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
            f"observed {tuple(got.shape)} {got.dtype} (exact integers, no tolerance); "
            "this is NOT the serving path's geometry — see check 4",
        )

    # --- 3: the override reached the step that will run ---------------------
    checks.start(
        "The built serving pipeline's image step carries the FORCED letterbox pad, and "
        "its other four geometry settings are the checkpoint's"
    )
    expected_served = {
        "image_crop_size": IMAGE_CROP_SIZE,
        "image_target_size": IMAGE_TARGET_SIZE,
        "shortest_image_edge": SHORTEST_IMAGE_EDGE,
        "crop_fraction": CROP_FRACTION,
        "letter_box_transform": SERVING_LETTER_BOX_TRANSFORM,
    }
    served_disagreements = [
        f"{key}: built step has {served[key]!r}, expected {expected_served[key]!r}"
        for key in SERVING_GEOMETRY_FIELDS
        if served[key] != expected_served[key]
    ]
    if served_disagreements:
        checks.fail(
            "serving_override_landed",
            "the serving pipeline is not configured as the server configures it, so every "
            "'serving path' number below would describe something else. "
            + "; ".join(served_disagreements)
            + ". If letter_box_transform is False, the override in "
            "docker/lerobot-policy/server.py did not land (most likely the step's registry "
            "name moved) and the server would feed the VLM an unpadded frame",
        )
    else:
        checks.ok(
            "serving_override_landed",
            "read off the built step: "
            + ", ".join(f"{key}={served[key]!r}" for key in SERVING_GEOMETRY_FIELDS)
            + f" — the checkpoint DECLARES letter_box_transform={on_disk['letter_box_transform']!r} "
            "and the serving path overrides it on purpose",
        )

    # --- 4: THE SERVING-PATH VERDICT, shape and bytes -----------------------
    checks.start(
        f"480x{SOURCE_WIDTH} through the SERVING pipeline is {SERVING_EXPECTED_SHAPE} and "
        f"hashes to Isaac-GR00T's own eval digest"
    )
    serving = outputs["serving_path"]
    serving_sha = _sha256(serving)
    if tuple(serving.shape) != SERVING_EXPECTED_SHAPE or serving.dtype != np.uint8:
        checks.fail(
            "serving_path_shape",
            f"observed {tuple(serving.shape)} {serving.dtype}, expected "
            f"{SERVING_EXPECTED_SHAPE} uint8 — the served geometry is NOT the padded square "
            "Isaac's code produced at training time",
        )
    elif serving_sha != SERVING_OUTPUT_SHA256:
        checks.fail(
            "serving_path_shape",
            f"shape is right but the BYTES are not: observed {serving_sha}, recorded "
            f"{SERVING_OUTPUT_SHA256} (which is Isaac-GR00T's own eval output digest). The "
            "cross-backend agreement no longer holds byte-for-byte. RE-MEASURE both "
            "backends and re-record; do not relax this to a tolerance",
        )
    else:
        checks.ok(
            "serving_path_shape",
            f"observed {tuple(serving.shape)} {serving.dtype}, sha256 {serving_sha}: "
            f"byte-identical to Isaac-GR00T's eval output, and a different artifact from "
            f"the unpatched {tuple(got.shape)}",
        )

    # --- 5: the corruption it must be distinguished from --------------------
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

    # --- 6: the corruption is discriminable ON THE UNPATCHED PATH ----------
    checks.start(
        "The UNPATCHED output and the corrupted output are NOT equal (the historical "
        "shape-only rationale)"
    )
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
            f"{tuple(got.shape)} != {tuple(corrupted.shape)}; a shape check catches it on "
            "the unpatched path — check 7 records that it does NOT on the serving path",
        )

    # --- 7: the honest consequence of the forced pad -----------------------
    # Recorded as a MEASUREMENT rather than as prose, because it is a real cost of
    # the operator's decision and the kind of fact a later reader would otherwise
    # inherit as a false reassurance: the C-3 corruption becomes shape-INVISIBLE once
    # every input is padded square. What carries C-3 now is prevention
    # (fixup_policy_features runs before from_pretrained, so the placeholder branch
    # never executes), not this shape.
    checks.start(
        "On the SERVING path the C-3 corruption shares the correct output's SHAPE but "
        "differs in BYTES (recorded cost of the forced pad)"
    )
    corrupted_served = transform_as_served(placeholder_squared(frame), served)
    shapes_equal = tuple(corrupted_served.shape) == tuple(serving.shape)
    bytes_equal = shapes_equal and np.array_equal(corrupted_served, serving)
    serving_corruption_shape_invisible = bool(shapes_equal and not bytes_equal)
    if bytes_equal:
        checks.fail(
            "serving_corruption_shape_invisible",
            "the corrupted and correct SERVING outputs are byte-identical, which would "
            "mean the C-3 pre-resize is undetectable by ANY comparison rather than merely "
            "by a shape check — re-examine fixup_policy_features before serving",
        )
    elif not shapes_equal:
        checks.fail(
            "serving_corruption_shape_invisible",
            f"the corrupted serving output is {tuple(corrupted_served.shape)} against "
            f"{tuple(serving.shape)} for the correct one. That is BETTER than recorded, not "
            "worse — but the recorded cost of the forced pad is now stale and "
            "docs/LEROBOT-SERVING-VERDICTS.md must be re-recorded rather than left claiming "
            "a weakness that no longer exists",
        )
    else:
        checks.ok(
            "serving_corruption_shape_invisible",
            f"both are {tuple(serving.shape)} and they differ in bytes: the forced pad "
            "squares every input, so C-3 is no longer catchable by a shape check and is "
            "carried by PREVENTION (fixup_policy_features) instead",
        )

    # --- 8: image_crop_size is inert, by measurement -----------------------
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

    # --- 9: replay-identical, on BOTH geometries ---------------------------
    checks.start("Two invocations on the same frame are byte-identical (unpatched AND served)")
    replay = transform_checkpoint_recipe(frame)
    serving_replay = transform_as_served(frame, served)
    replay_identical = bool(np.array_equal(got, replay))
    serving_replay_identical = bool(np.array_equal(serving, serving_replay))
    if not (replay_identical and serving_replay_identical):
        checks.fail(
            "replay_identical",
            f"two invocations differ (unpatched identical={replay_identical}, served "
            f"identical={serving_replay_identical}) — the serving path should take the "
            "deterministic CENTER crop, not the train-time random crop",
        )
    else:
        checks.ok(
            "replay_identical",
            "exact array equality across two invocations of each geometry (center crop, "
            "not random crop)",
        )

    manifest: dict[str, Any] = {
        "verdict": {
            "source_shape": [SOURCE_HEIGHT, SOURCE_WIDTH, 3],
            "expected_shape": list(EXPECTED_SHAPE),
            "serving_expected_shape": list(SERVING_EXPECTED_SHAPE),
            "expected_corrupted_shape": list(EXPECTED_CORRUPTED_SHAPE),
            "unpatched_effective_pipeline": (
                "resize-shortest-edge-to-256 -> center-crop-95% -> "
                "resize-shortest-edge-to-256"
            ),
            "serving_effective_pipeline": (
                "letterbox-pad-to-square (FORCED) -> resize-shortest-edge-to-256 -> "
                "center-crop-95% -> resize-shortest-edge-to-256"
            ),
            "serving_letter_box_transform": SERVING_LETTER_BOX_TRANSFORM,
            "checkpoint_letter_box_transform": on_disk["letter_box_transform"],
            "why_the_pad_is_forced": (
                "Isaac-GR00T trained these weights and applies LetterBoxPad() "
                "unconditionally (image_augmentations.py:420-487), treating "
                "letter_box_transform as a stored-but-unused backward-compat param "
                "(processing_gr00t_n1d7.py:171-172, 198). LeRobot honours the flag, so "
                "it produced a geometry Isaac never emitted. Forcing the pad makes the "
                "outputs byte-identical."
            ),
            "inference_boundary": (
                "'the weights were trained on Isaac's padded square' is INFERRED from "
                "'Isaac trained this checkpoint'. The training recipe was NOT read and no "
                "local artifact records it. Re-examine this first if Phase 7 parity "
                "disappoints."
            ),
            "stale_roadmap_framing": (
                "'341x256-crop vs 256x256-letterbox' is a misread (REQUIREMENTS.md:50): "
                "256x341 is the intermediate after the first resize-shortest-edge, and "
                "256x256 was the letterbox branch this checkpoint's config declines — "
                "which the serving path now takes anyway"
            ),
        },
        "recipe": on_disk,
        "recipe_source": str(PROCESSOR_CONFIG.relative_to(REPO_ROOT)),
        "served_geometry_settings": {
            key: served[key] for key in SERVING_GEOMETRY_FIELDS
        },
        "seed": int(args.seed),
        "cases": {
            name: {
                "shape": _shape_of(outputs[name]),
                "dtype": str(outputs[name].dtype),
            }
            for name in CASE_NAMES
        },
        "serving_output_sha256": serving_sha,
        "serving_output_sha256_recorded": SERVING_OUTPUT_SHA256,
        "serving_corruption_shape_invisible": serving_corruption_shape_invisible,
        "crop_size_inert": crop_size_inert,
        "replay_identical": replay_identical,
        "serving_replay_identical": serving_replay_identical,
    }
    manifest["cases"]["checkpoint_recipe"]["input_shape"] = [
        SOURCE_HEIGHT,
        SOURCE_WIDTH,
        3,
    ]
    manifest["cases"]["serving_path"]["input_shape"] = [
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
