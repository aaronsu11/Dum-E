"SAFE-01: the ONE implementation of the GR00T serving contract's five assertions."

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
EXPECTED_TAG = "new_embodiment"

#: ``processor_kwargs.crop_fraction``. Set, so ``image_crop_size: [230, 230]`` is INERT —
#: upstream consults the crop size only when ``crop_fraction is None``
#: (``processor_groot.py:1453-1454``).
EXPECTED_CROP_FRACTION = 0.95

#: ``processor_kwargs.shortest_image_edge``. On the SERVING path the effective geometry is
#: letterbox-pad-to-square -> resize-shortest-edge-to-256 -> center-crop-95% ->
EXPECTED_SHORTEST_IMAGE_EDGE = 256

#: ``processor_kwargs.letter_box_transform`` as THIS checkpoint DECLARES it. Asserted as a
#: drift catcher, not as a correctness claim about the pad: every recorded geometry number —
#: the two shapes, both output digests — is measured against a checkpoint declaring False, so
#: a checkpoint that declared True would invalidate the recorded verdict even though its
#: geometry would still come out square. See :data:`SERVING_LETTER_BOX_TRANSFORM`.
EXPECTED_CHECKPOINT_LETTER_BOX_TRANSFORM = False

#: **What the SERVING path must actually DO, and it is the opposite of the flag above.** The
#: pad is FORCED ON so the model sees the padded square Isaac's code produced at training
SERVING_LETTER_BOX_TRANSFORM = True

#: ``processor_kwargs.modality_configs[EXPECTED_TAG]["video"]["modality_keys"]`` — **the value
#: that decides WHICH CAMERA LANDS IN WHICH VIEW SLOT.** It is not a label: upstream's
EXPECTED_VIDEO_MODALITY_KEYS: tuple[str, ...] = ("front", "wrist")

#: Registry name of the pipeline step that applies the image geometry
#: (``@ProcessorStepRegistry.register(name=...)`` on ``GrootN17VLMEncodeStep``,
VLM_ENCODE_STEP_KEY = "groot_n1_7_vlm_encode_v1"

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
    # The camera-view ORDER authority. Required for the same T-06-08 reason as the rest: an
    # absent modality_configs must raise here rather than let ``video_modality_keys`` be
    # derived from nothing. See EXPECTED_VIDEO_MODALITY_KEYS.
    "modality_configs",
)


@dataclass(frozen=True)
class GrootGuardSnapshot:
    'The facts SAFE-01 turns on, decoupled from how they were obtained.'

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
    served_letter_box_transform: bool
    crop_fraction: float | None
    shortest_image_edge: int | None
    use_albumentations: bool
    video_modality_keys: tuple[str, ...]
    preprocessor_training: bool
    encode_step_training: bool


def serving_preprocessor_overrides() -> dict[str, dict[str, Any]]:
    'The image-geometry override the serving path MUST apply, as ONE definition.'
    return {VLM_ENCODE_STEP_KEY: {"letter_box_transform": SERVING_LETTER_BOX_TRANSFORM}}


def assert_groot_serving_contract(snapshot: GrootGuardSnapshot) -> None:
    'Raise ``ValueError`` unless every SAFE-01 condition holds.'
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
    if snapshot.letter_box_transform != EXPECTED_CHECKPOINT_LETTER_BOX_TRANSFORM:
        raise ValueError(
            f"SAFE-01/5 the checkpoint DECLARES letter_box_transform="
            f"{snapshot.letter_box_transform!r}, expected "
            f"{EXPECTED_CHECKPOINT_LETTER_BOX_TRANSFORM!r}. Every recorded geometry number — "
            f"both output shapes and both output digests — was measured against a checkpoint "
            f"declaring {EXPECTED_CHECKPOINT_LETTER_BOX_TRANSFORM!r}, and the serving path's "
            f"forced pad was justified against that reading. A checkpoint that declares "
            f"otherwise has a different image recipe, so the recorded verdict no longer "
            f"describes it. Refusing to serve against a geometry verdict that was measured "
            f"on a different recipe; RE-MEASURE both backends and re-record."
        )
    if snapshot.served_letter_box_transform != SERVING_LETTER_BOX_TRANSFORM:
        raise ValueError(
            f"SAFE-01/5 the SERVED image pipeline has letter_box_transform="
            f"{snapshot.served_letter_box_transform!r}, expected "
            f"{SERVING_LETTER_BOX_TRANSFORM!r}: the letterbox pad was NOT forced onto the "
            f"step that will actually run, so a 480x640 frame reaches the VLM as "
            f"(256, 340, 3) instead of the padded square (256, 256, 3) Isaac's code produced "
            f"when it trained these weights. The override in "
            f"docker/lerobot-policy/server.py did not land — most likely the step's registry "
            f"name moved. Refusing to serve a geometry the weights were not trained on."
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
            f"serving recipe is letterbox-pad-to-square -> resize-shortest-edge-to-256 -> "
            f"center-crop-95% -> resize-shortest-edge-to-256; any of these three moving "
            f"changes what the VLM sees. Refusing to serve a preprocessing path the "
            f"checkpoint was not trained with."
        )
    if tuple(snapshot.video_modality_keys) != EXPECTED_VIDEO_MODALITY_KEYS:
        raise ValueError(
            f"SAFE-01/5 video modality keys are {tuple(snapshot.video_modality_keys)!r}, expected "
            f"{EXPECTED_VIDEO_MODALITY_KEYS!r}. These are the checkpoint's OWN declaration of "
            f"which camera lands in which view slot: _ordered_image_keys matches them against the "
            f"observation.images.<cam> keys the handshake declares, IN THIS ORDER "
            f"(processor_groot.py:1563-1598). An unmatched key does NOT raise — upstream emits a "
            f"single logging.warning, once, and falls back to feeding all cameras in ALPHABETICAL "
            f"order; a partial match silently feeds FEWER views. Shapes, chunk length and every "
            f"other SAFE-01 field stay correct throughout, so the model receives the wrist frame "
            f"where it expects the front frame and the arm moves plausibly to the wrong place. "
            f"Refusing to serve a camera layout the checkpoint was not trained with."
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
    'Build a snapshot from a checkpoint directory alone. No weights, no GPU, no network.'
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
        # The config-only site does NOT validate these FOUR. There is no processor object
        # on this path, so a decode step, two training flags and the SERVED letterbox value
        decode_step_type=EXPECTED_DECODE_STEP,
        served_letter_box_transform=SERVING_LETTER_BOX_TRANSFORM,
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
    'Build a snapshot from the objects that will actually run inference.'
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
            served_letter_box_transform=False,
            crop_fraction=None,
            shortest_image_edge=None,
            use_albumentations=False,
            video_modality_keys=(),
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
        # The EFFECTIVE letterbox value, read off the step that will actually transform
        # frames — never re-derived from the checkpoint, which declares the opposite. This is
        served_letter_box_transform=bool(encode_step.letter_box_transform),
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
        "video_modality_keys": _video_modality_keys(processor_kwargs, embodiment_tag),
    }


def _video_modality_keys(processor_kwargs: dict[str, Any], embodiment_tag: str) -> tuple[str, ...]:
    "The checkpoint's declared camera-view ORDER for ``embodiment_tag``."
    modality_configs = processor_kwargs.get("modality_configs")
    if not isinstance(modality_configs, dict):
        return ()
    video = (modality_configs.get(embodiment_tag) or {}).get("video")
    if not isinstance(video, dict):
        return ()
    keys = video.get("modality_keys")
    if not isinstance(keys, (list, tuple)):
        return ()
    return tuple(str(key) for key in keys)


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
    "Class name of the postprocessor's action-decode step, or :data:`NO_DECODE_STEP`."
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
