"""SAFE-01: the ONE implementation of the GR00T serving contract's five assertions.

This module is a pure validator. It builds a frozen :class:`GrootGuardSnapshot` of the
facts SAFE-01 turns on, and :func:`assert_groot_serving_contract` raises a named,
specific ``ValueError`` naming the failing assertion and the observed value. There is
**one implementation and two call sites**, not two code paths:

1. ``docker/lerobot-policy/entrypoint.py``'s preflight calls
   :func:`snapshot_from_checkpoint_dir` — config-only, no weights, no GPU, no network —
   so a missing or wrong checkpoint bind-mount is caught *before* 12.6 GB of shards are
   read. That is what makes "refuses to **start** with a named, specific error" literally
   true rather than "refuses on the first request".
2. The ``PolicyServer`` subclass's post-``super()`` hook calls
   :func:`snapshot_from_loaded` — the only place that sees the object which will actually
   run inference, because the policy load happens inside a request handler.

Plan 06-03 wires both sites. Nothing here imports the arm, opens a serial port, or
commands a motor.

DELIBERATE DIVERGENCE (location)
--------------------------------
PATTERNS.md proposed ``shared/groot_guard.py``. This module deliberately lives OUTSIDE
``shared/`` because importing anything from ``shared`` executes ``shared/__init__.py``,
which imports ``pydantic`` at module scope (``shared/__init__.py:23``) — a client-side
dependency the policy container has no reason to carry. Do NOT "restore parity" by moving
this module into ``shared/``; ``policy_guard/__init__.py`` is deliberately zero bytes for
the same reason.

WHY ``GrootConfig.normalization_mapping`` IS NOT READ HERE
---------------------------------------------------------
The roadmap names "normalization has fallen back to identity" as a SAFE-01 condition. The
originally designated mechanism for detecting it is a guaranteed FALSE ALARM: that mapping
is IDENTITY for ``VISUAL``, ``STATE`` and ``ACTION`` **by design** on every healthy launch.
Upstream says so at ``configuration_groot.py:258-269``:

    GR00T normalizes state/action internally in its processor steps (min/max with
    q01/q99 percentiles, per embodiment), and the Qwen3-VL backbone's image processor
    handles image normalization. The policy therefore does NOT use LeRobot's
    NormalizerProcessorStep/UnnormalizerProcessorStep, so this mapping is intentionally
    IDENTITY for every feature and is not consulted by make_groot_pre_post_processors.

The intent survives; the mechanism is replaced. The three DISCRIMINATING identity-fallback
signals, all asserted below under ``SAFE-01/4`` and ``SAFE-01/3``, are:

* ``decode_step_type != EXPECTED_DECODE_STEP`` — the legacy ``GrootActionUnpackUnnormalizeStep``
  is installed only when the checkpoint's stats are unusable, and it collapses ``(B,T,D)``
  chunks to a single timestep (``processor_groot.py:2459-2463``). Its presence in a live
  pipeline IS the identity-normalization tell.
* ``stats_non_empty is False`` — an empty stats table makes the decoder return normalized
  ``[-1, 1]`` actions while every log line looks healthy.
* ``use_percentiles is False`` — this checkpoint requires q01/q99.

That negative is DOCUMENTED, not silently omitted:
``tests/test_groot_guard.py::test_normalization_mapping_is_identity_by_design_and_is_not_a_discriminator``
asserts the field is identity on the real checkpoint and that this module reads no attribute
of that name. It is never the guard's evidence.

THE WRONG KNOBS
---------------
``GrootConfig.use_relative_actions`` (PLURAL, ``configuration_groot.py:330-336``) and
``GrootConfig.relative_exclude_joints`` are NOT the knobs that decide relative-action
decoding for this checkpoint. Verified against the real checkpoint:
``GrootConfig(base_model_path=CK).use_relative_actions is False`` while the checkpoint's
``processor_config.json`` carries ``use_relative_action: True`` (SINGULAR), and it is the
CHECKPOINT value the inference path reads (``processor_groot.py:190``, and the
native-vs-fallback branch at ``processor_groot.py:1271-1273``). This guard therefore reads
the singular checkpoint value. Setting the plural config flag could only install the generic
``RelativeActionsProcessorStep`` fallback, whose own warning says it normalizes relative
deltas with ABSOLUTE action stats (``processor_groot.py:1275-1282``) — strictly worse than
the native path, and not what SAFE-01/3 is asking about.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.policies.groot.configuration_groot import (
    GrootConfig,
    infer_groot_n1_7_action_horizon,
    is_raw_groot_n1_7_checkpoint,
)

# Upstream's OWN sidecar reader. Called rather than reimplemented, deliberately: the
# question SAFE-01/1 asks is "would LeRobot resolve checkpoint_assets for this path", and
# the only non-lying answer is the one LeRobot's reader gives. It is a private symbol, so
# it is also a pinned surface — a rename fails loudly at import time, which is the D-03
# design intent (a monkeypatch would have failed silently instead).
from lerobot.policies.groot.processor_groot import (
    _load_n1_7_checkpoint_processor_assets,  # noqa: PLC2701
)

#: The horizon this checkpoint actually decodes, from ``new_embodiment.action.delta_indices``
#: (``[0..15]`` -> 16). 40 is the WELL-LIT WRONG PATH and it is lit twice: ``GrootConfig``'s
#: own default is 40 (``configuration_groot.py:512``) AND the checkpoint's own ``config.json``
#: says ``action_horizon: 40``. Both are traps; neither is the decoded horizon.
EXPECTED_HORIZON = 16

#: The one embodiment tag whose ``delta_indices`` are 16. This checkpoint's
#: ``processor_config.json`` carries NINE modality configs and the other eight all carry 40,
#: so the tag is a load-bearing input rather than a label. Tag INFERENCE cannot rescue a
#: wrong value here: ``infer_groot_n1_7_embodiment_tag`` returns ``None`` for this checkpoint
#: (it only infers when exactly one modality config exists — ``configuration_groot.py:145-163``),
#: so the value comes from ``GrootConfig.embodiment_tag``'s default. Every statistics lookup
#: and the horizon read are both keyed by it.
EXPECTED_TAG = "new_embodiment"

#: ``processor_kwargs.crop_fraction``. Set, so ``image_crop_size: [230, 230]`` is INERT —
#: upstream consults the crop size only when ``crop_fraction is None``
#: (``processor_groot.py:1453-1454``).
EXPECTED_CROP_FRACTION = 0.95

#: ``processor_kwargs.shortest_image_edge``. The effective geometry is
#: resize-shortest-edge-to-256 -> center-crop-95% -> resize-shortest-edge-to-256, giving
#: (256, 340, 3) for a 480x640 frame. 224 is the placeholder-path corruption this catches.
EXPECTED_SHORTEST_IMAGE_EDGE = 256

#: The relative-aware decode step. Its alternative, ``GrootActionUnpackUnnormalizeStep``, is
#: installed only when the checkpoint's stats are unusable and is deliberately stubbed for
#: native relative actions upstream (``processor_groot.py:2459-2463``).
EXPECTED_DECODE_STEP = "GrootN17ActionDecodeStep"

#: Returned by :func:`snapshot_from_loaded` when the postprocessor exposes no decode step at
#: all, so ``SAFE-01/3`` fires with a named error instead of an ``IndexError`` escaping.
NO_DECODE_STEP = "<no decode step found>"

#: The hub default ``base_model_path`` resolves to when it is unset
#: (``configuration_groot.py:382-383``) — i.e. the BASE weights, not our fine-tune.
_HUB_DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.7-3B"

#: The three sidecars every field below is derived from. A snapshot whose fields silently
#: defaulted because one of these was absent would PASS the guard and turn SAFE-01 into a
#: vacuous pass (threat T-06-08), so their absence raises instead.
_REQUIRED_SIDECARS = ("config.json", "processor_config.json", "statistics.json")

#: The ``processor_kwargs`` keys this guard reads. Required, not optional: a missing key is
#: raised on rather than defaulted, for the same T-06-08 reason.
_REQUIRED_PROCESSOR_KWARGS = (
    "use_relative_action",
    "use_percentiles",
    "letter_box_transform",
    "crop_fraction",
    "shortest_image_edge",
    "use_albumentations",
)


@dataclass(frozen=True)
class GrootGuardSnapshot:
    """The facts SAFE-01 turns on, decoupled from how they were obtained.

    Frozen for two reasons: a call site cannot mutate a snapshot between building it and
    asserting on it, and tests mutate exactly ONE field via :func:`dataclasses.replace`
    against a snapshot built from the REAL checkpoint — which is what makes each violation
    test a fail-first proof against the real config shape rather than against a fabricated
    fixture that might misrepresent it.
    """

    base_model_path: str | None
    is_raw_checkpoint: bool
    assets_present: bool
    embodiment_tag: str
    checkpoint_horizon: int | None
    configured_actions_per_chunk: int
    use_relative_action: bool
    decode_step_type: str
    stats_non_empty: bool
    use_percentiles: bool
    letter_box_transform: bool
    crop_fraction: float | None
    shortest_image_edge: int | None
    use_albumentations: bool
    preprocessor_training: bool
    encode_step_training: bool


def assert_groot_serving_contract(snapshot: GrootGuardSnapshot) -> None:
    """Raise ``ValueError`` unless every SAFE-01 condition holds.

    Returns ``None`` on success and raises on failure — deliberately NOT a bool and
    deliberately not a log line. A caller that forgets to check a returned bool, or an
    operator who misses a warning in a container log, is exactly the silent failure this
    guard exists to remove.

    Every message starts with its own identifier (``SAFE-01/1`` .. ``SAFE-01/5``) followed
    by a short condition name, embeds the OBSERVED value, and closes by saying what the
    guard is refusing to do and why — the ``policy/factory.py:57-62`` idiom.
    """
    # --- SAFE-01/1: the resolved path really is our raw fine-tune -------------
    if not snapshot.is_raw_checkpoint:
        raise ValueError(
            f"SAFE-01/1 base_model_path: is_raw_groot_n1_7_checkpoint("
            f"{snapshot.base_model_path!r}) is False. The checkpoint bind-mount is missing "
            f"or wrong, or the path resolved to the hub default "
            f"{_HUB_DEFAULT_BASE_MODEL!r} (configuration_groot.py:382-383). Refusing to "
            f"serve base weights in place of the fine-tuned SO101 checkpoint."
        )
    if not snapshot.assets_present:
        raise ValueError(
            f"SAFE-01/1 checkpoint_assets: assets_present={snapshot.assets_present!r} for "
            f"{snapshot.base_model_path!r}, so LeRobot resolved no checkpoint sidecars. "
            f"Every checkpoint-derived setting — percentiles, relative decoding, the stats "
            f"table and the image geometry — would silently fall back to LeRobot defaults. "
            f"Refusing to serve a checkpoint whose own configuration is not being read."
        )

    # --- SAFE-01/2: the horizon is 16 at CONFIG level ------------------------
    # Deliberately NOT a check on the emitted chunk length: three independent truncations
    # force 16 regardless of configuration (T=16, T=40 and T=50 all decode to (1,16,6)), so
    # an emitted 16 proves nothing. See test_chunk_length_sixteen_is_not_horizon_evidence.
    if snapshot.embodiment_tag != EXPECTED_TAG:
        raise ValueError(
            f"SAFE-01/2 embodiment_tag is {snapshot.embodiment_tag!r}, expected "
            f"{EXPECTED_TAG!r}. This checkpoint carries 9 embodiment tags and only "
            f"{EXPECTED_TAG!r} has {EXPECTED_HORIZON} delta_indices; the other 8 have 40. "
            f"Every statistics lookup and the horizon read are keyed by this tag, so a "
            f"wrong tag silently selects another embodiment's normalization. Refusing to "
            f"decode actions against the wrong embodiment's statistics."
        )
    if snapshot.checkpoint_horizon != EXPECTED_HORIZON:
        raise ValueError(
            f"SAFE-01/2 horizon: the checkpoint's delta_indices give "
            f"{snapshot.checkpoint_horizon}, expected {EXPECTED_HORIZON}. GrootConfig's own "
            f"default is 40 and the checkpoint's config.json says action_horizon: 40 — both "
            f"are traps, and neither is the horizon this checkpoint decodes. Refusing to "
            f"serve a horizon the per-timestep relative statistics do not cover."
        )
    if snapshot.configured_actions_per_chunk != snapshot.checkpoint_horizon:
        raise ValueError(
            f"SAFE-01/2 configured actions_per_chunk="
            f"{snapshot.configured_actions_per_chunk} disagrees with the checkpoint's "
            f"delta_indices ({snapshot.checkpoint_horizon}). The horizon is configurable "
            f"(D-11), so it can be configured wrong; refusing to obey a configured horizon "
            f"the checkpoint cannot decode."
        )

    # --- SAFE-01/3: relative-action decoding is on and native ----------------
    if not snapshot.use_relative_action:
        raise ValueError(
            f"SAFE-01/3 use_relative_action is {snapshot.use_relative_action!r} in the "
            f"checkpoint's processor_config.json. This checkpoint is relative-arm / "
            f"absolute-gripper; reading its output as a flat absolute 6-vector inflates "
            f"commanded motion 1.83x-3.01x and the arm snaps, silently. Refusing to command "
            f"an arm from mis-decoded actions."
        )
    if snapshot.decode_step_type != EXPECTED_DECODE_STEP:
        raise ValueError(
            f"SAFE-01/3 postprocessor decode step is {snapshot.decode_step_type}, expected "
            f"{EXPECTED_DECODE_STEP}. GrootActionUnpackUnnormalizeStep means the "
            f"relative-aware decoder was NOT installed (processor_groot.py:1297-1320) — it "
            f"is reached only when the checkpoint's stats are unusable, and it is "
            f"deliberately stubbed for native relative actions. Refusing to serve with the "
            f"identity-normalizing decoder."
        )

    # --- SAFE-01/4: normalization has NOT fallen back to identity ------------
    # The discriminating signals, not GrootConfig.normalization_mapping — that field is
    # IDENTITY by design here and is not consulted upstream (see the module docstring).
    if not snapshot.stats_non_empty:
        raise ValueError(
            f"SAFE-01/4 checkpoint stats are empty (stats_non_empty="
            f"{snapshot.stats_non_empty!r}) for embodiment tag "
            f"{snapshot.embodiment_tag!r}: the decoder would return normalized [-1, 1] "
            f"actions while every log line looked healthy. Refusing to serve normalized "
            f"actions as if they were joint targets."
        )
    if not snapshot.use_percentiles:
        raise ValueError(
            f"SAFE-01/4 use_percentiles is {snapshot.use_percentiles!r}; this checkpoint "
            f"normalizes with q01/q99 percentiles (use_mean_std: False), so min/max "
            f"normalization would rescale every action. Refusing to unnormalize with the "
            f"wrong statistic."
        )

    # --- SAFE-01/5: image geometry and eval-mode determinism -----------------
    if snapshot.letter_box_transform:
        raise ValueError(
            f"SAFE-01/5 letter_box_transform is {snapshot.letter_box_transform!r}; this "
            f"checkpoint requires False (crop-then-resize, not letterbox). The letterbox "
            f"branch yields a 256x256 frame where this checkpoint's recipe yields 256x340, "
            f"so the model would see a geometry it was not trained on. Refusing to serve "
            f"the wrong image geometry."
        )
    if (
        snapshot.crop_fraction != EXPECTED_CROP_FRACTION
        or snapshot.shortest_image_edge != EXPECTED_SHORTEST_IMAGE_EDGE
        or not snapshot.use_albumentations
    ):
        raise ValueError(
            f"SAFE-01/5 image geometry: crop_fraction={snapshot.crop_fraction} "
            f"(want {EXPECTED_CROP_FRACTION}), "
            f"shortest_image_edge={snapshot.shortest_image_edge} "
            f"(want {EXPECTED_SHORTEST_IMAGE_EDGE}), "
            f"use_albumentations={snapshot.use_albumentations} (want True). The effective "
            f"recipe is resize-shortest-edge-to-256 -> center-crop-95% -> "
            f"resize-shortest-edge-to-256; any of these three moving changes what the VLM "
            f"sees. Refusing to serve a preprocessing path the checkpoint was not trained "
            f"with."
        )
    if snapshot.preprocessor_training or snapshot.encode_step_training:
        raise ValueError(
            f"SAFE-01/5 processor is in TRAINING mode: "
            f"preprocessor_training={snapshot.preprocessor_training!r}, "
            f"encode_step_training={snapshot.encode_step_training!r}. Training mode enables "
            f"Isaac's train-time random crop and this checkpoint's state dropout "
            f"(state_dropout_prob: 0.2), so identical observations would produce different "
            f"chunks. Refusing to serve a non-deterministic preprocessing path."
        )

    return None


def snapshot_from_checkpoint_dir(
    checkpoint_path: str | Path,
    configured_actions_per_chunk: int,
) -> GrootGuardSnapshot:
    """Build a snapshot from a checkpoint directory alone. No weights, no GPU, no network.

    This is the container entrypoint's preflight builder: it answers "is the bind-mounted
    checkpoint the one we think it is, configured the way we think it is" before any shard
    is read.

    Raises:
        ValueError: if the directory or any of the three sidecar JSONs is absent, or a
            required ``processor_kwargs`` key is missing. It never returns a snapshot with
            silently defaulted fields — that snapshot would pass the guard (T-06-08).
    """
    path = Path(checkpoint_path).expanduser()
    if not path.is_dir():
        raise ValueError(
            f"SAFE-01 preflight: checkpoint directory {str(path)!r} does not exist or is "
            "not a directory. Refusing to build a snapshot whose fields would default to "
            "LeRobot's own defaults, because such a snapshot would PASS the guard."
        )
    for sidecar in _REQUIRED_SIDECARS:
        if not (path / sidecar).is_file():
            raise ValueError(
                f"SAFE-01 preflight: required checkpoint sidecar {str(path / sidecar)!r} is "
                f"absent. Every SAFE-01 field is derived from {', '.join(_REQUIRED_SIDECARS)}; "
                "refusing to build a partially-defaulted snapshot, which would be a vacuous "
                "pass."
            )

    config = GrootConfig(base_model_path=str(path))
    common = _snapshot_common(config, path, configured_actions_per_chunk)
    return GrootGuardSnapshot(
        # The config-only site does NOT validate these three. There is no processor object
        # on this path, so a decode step and two training flags simply do not exist yet;
        # they are the post-load site's job. They are set to their passing values on
        # purpose, and pretending otherwise would make this preflight look stronger than it
        # is. If you need them checked, you need snapshot_from_loaded.
        decode_step_type=EXPECTED_DECODE_STEP,
        preprocessor_training=False,
        encode_step_training=False,
        **common,
    )


def snapshot_from_loaded(
    config: Any,
    preprocessor: Any,
    postprocessor: Any,
    configured_actions_per_chunk: int,
) -> GrootGuardSnapshot:
    """Build a snapshot from the objects that will actually run inference.

    Every attribute read here is a place a lerobot version bump can break — and it breaks
    LOUDLY, at attribute-access time, naming what it looked for. That is the D-03 design
    intent: composition over a pinned upstream fails at the seam, where a monkeypatch would
    have failed silently three layers down inside a request handler.

    Args:
        config: the resolved ``GrootConfig`` off the loaded policy.
        preprocessor: the built ``PolicyProcessorPipeline`` (input side).
        postprocessor: the built ``PolicyProcessorPipeline`` (output side).
        configured_actions_per_chunk: the operator-supplied horizon (D-11).
    """
    base_model_path = config.base_model_path
    if not is_raw_groot_n1_7_checkpoint(base_model_path):
        # Deliberately an early return, not a raise: the caller's next line is the guard,
        # and SAFE-01/1's named message ("the bind-mount is missing or wrong, or the path
        # resolved to the hub default") is strictly more useful to an operator than an
        # IOError from reading sidecars that are not there. Nothing here can be mistaken
        # for a pass — the guard raises on the very first check.
        return GrootGuardSnapshot(
            base_model_path=base_model_path,
            is_raw_checkpoint=False,
            assets_present=False,
            embodiment_tag=getattr(config, "embodiment_tag", ""),
            checkpoint_horizon=None,
            configured_actions_per_chunk=configured_actions_per_chunk,
            use_relative_action=False,
            decode_step_type=NO_DECODE_STEP,
            stats_non_empty=False,
            use_percentiles=False,
            letter_box_transform=False,
            crop_fraction=None,
            shortest_image_edge=None,
            use_albumentations=False,
            preprocessor_training=False,
            encode_step_training=False,
        )

    common = _snapshot_common(
        config, Path(base_model_path).expanduser(), configured_actions_per_chunk
    )
    pack_step = _require_step(preprocessor, "state_dropout_prob", "GrootN17PackInputsStep")
    encode_step = _require_step(preprocessor, "letter_box_transform", "GrootN17VLMEncodeStep")
    return GrootGuardSnapshot(
        decode_step_type=_decode_step_type(postprocessor),
        # `training` is a make_*_processors kwarg set from dataset_meta
        # (`processor_groot.py:1225, 1266` — `training=dataset_meta is not None`), NOT a
        # torch module flag, and policy_server passes no dataset_meta, so it is False on the
        # serving path. If a future lerobot release moves that flag, this is where it breaks.
        preprocessor_training=bool(pack_step.training),
        encode_step_training=bool(encode_step.training),
        **common,
    )


def _snapshot_common(
    config: Any, checkpoint_path: Path, configured_actions_per_chunk: int
) -> dict[str, Any]:
    """The fields both builders derive identically, so they cannot drift apart."""
    embodiment_tag = config.embodiment_tag
    assets = _load_n1_7_checkpoint_processor_assets(config)
    processor_kwargs = _read_processor_kwargs(checkpoint_path)
    return {
        "base_model_path": config.base_model_path,
        "is_raw_checkpoint": is_raw_groot_n1_7_checkpoint(config.base_model_path),
        "assets_present": assets is not None,
        "embodiment_tag": embodiment_tag,
        "checkpoint_horizon": infer_groot_n1_7_action_horizon(checkpoint_path, embodiment_tag),
        "configured_actions_per_chunk": configured_actions_per_chunk,
        "use_relative_action": bool(processor_kwargs["use_relative_action"]),
        # `assets.raw_stats` IS `statistics.json[embodiment_tag]` as upstream resolves it
        # (`processor_groot.py:177-180`), so this reads the same table the decoder will use
        # rather than a second, independently-parsed copy of the file.
        "stats_non_empty": bool(assets is not None and assets.raw_stats),
        "use_percentiles": bool(processor_kwargs["use_percentiles"]),
        "letter_box_transform": bool(processor_kwargs["letter_box_transform"]),
        "crop_fraction": processor_kwargs["crop_fraction"],
        "shortest_image_edge": processor_kwargs["shortest_image_edge"],
        "use_albumentations": bool(processor_kwargs["use_albumentations"]),
    }


def _read_processor_kwargs(checkpoint_path: Path) -> dict[str, Any]:
    """Read the ``processor_kwargs`` keys SAFE-01 needs, raising on any that is absent."""
    processor_config_path = checkpoint_path / "processor_config.json"
    if not processor_config_path.is_file():
        raise ValueError(
            f"SAFE-01: {str(processor_config_path)!r} is absent, so the checkpoint's own "
            "geometry and relative-action settings cannot be read. Refusing to default them."
        )
    with processor_config_path.open() as handle:
        processor_config = json.load(handle)
    processor_kwargs = processor_config.get("processor_kwargs")
    if not isinstance(processor_kwargs, dict):
        raise ValueError(
            f"SAFE-01: {str(processor_config_path)!r} has no 'processor_kwargs' mapping "
            f"(got {type(processor_kwargs).__name__}). Refusing to default every "
            "checkpoint-derived setting."
        )
    missing = [key for key in _REQUIRED_PROCESSOR_KWARGS if key not in processor_kwargs]
    if missing:
        raise ValueError(
            f"SAFE-01: {str(processor_config_path)!r} is missing required processor_kwargs "
            f"{missing}. Refusing to default them: a defaulted field would pass the guard."
        )
    return processor_kwargs


def _decode_step_type(postprocessor: Any) -> str:
    """Class name of the postprocessor's action-decode step, or :data:`NO_DECODE_STEP`.

    ``env_action_dim`` is the action-decode surface: both candidate steps declare it
    (``GrootN17ActionDecodeStep`` at ``processor_groot.py:2325``, the legacy
    ``GrootActionUnpackUnnormalizeStep`` at ``:2464``), and none of the surrounding
    device/absolute-action steps do. The LAST such step is taken, because it is the one
    whose output leaves the pipeline.
    """
    steps = getattr(postprocessor, "steps", None) or ()
    candidates = [step for step in steps if hasattr(step, "env_action_dim")]
    if not candidates:
        return NO_DECODE_STEP
    return type(candidates[-1]).__name__


def _require_step(pipeline: Any, marker_attribute: str, expected_class: str) -> Any:
    """Return the last pipeline step carrying ``marker_attribute``, or raise loudly."""
    steps = getattr(pipeline, "steps", None) or ()
    candidates = [step for step in steps if hasattr(step, marker_attribute)]
    if not candidates:
        raise ValueError(
            f"SAFE-01: no step in {getattr(pipeline, 'name', pipeline)!r} carries "
            f"{marker_attribute!r}, so {expected_class} could not be located. The pinned "
            "lerobot release changed the processor pipeline's shape; fix the guard against "
            "the new shape rather than skipping the check."
        )
    return candidates[-1]
