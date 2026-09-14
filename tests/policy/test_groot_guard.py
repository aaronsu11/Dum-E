'Hermetic SAFE-01 gate for the GR00T serving contract.'

import ast
import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Import the pinned values rather than restating them: the guard and its tests must share
# ONE source of truth, or a drift in one silently satisfies the other.
from policy.backends.lerobot.models.groot import (  # noqa: E402
    EXPECTED_CHECKPOINT_LETTER_BOX_TRANSFORM,
    EXPECTED_CROP_FRACTION,
    EXPECTED_DECODE_STEP,
    EXPECTED_HORIZON,
    EXPECTED_SHORTEST_IMAGE_EDGE,
    EXPECTED_TAG,
    EXPECTED_VIDEO_MODALITY_KEYS,
    NO_DECODE_STEP,
    SERVING_LETTER_BOX_TRANSFORM,
    VLM_ENCODE_STEP_KEY,
    GrootGuardSnapshot,
    assert_groot_serving_contract,
    serving_preprocessor_overrides,
    snapshot_from_checkpoint_dir,
    snapshot_from_loaded,
)

from policy.backends.lerobot import features  # noqa: E402

REAL_CHECKPOINT = REPO_ROOT / "tests" / "fixtures" / "groot-so101"

GUARD_SOURCE = REPO_ROOT / "policy" / "backends" / "lerobot" / "models" / "groot.py"

#: The fourteen single-field mutations of the REAL snapshot, one per assertion the guard
#: makes, shared by the individual violation tests' intent and by the programmatic
#: message-consistency gate below so the two cannot drift apart.
VIOLATION_MUTATIONS: tuple[tuple[str, object], ...] = (
    ("is_raw_checkpoint", False),
    ("assets_present", False),
    ("embodiment_tag", "oxe_droid_relative_eef_relative_joint"),
    ("checkpoint_horizon", 40),
    ("configured_actions_per_chunk", 40),
    ("use_relative_action", False),
    ("decode_step_type", "GrootActionUnpackUnnormalizeStep"),
    ("stats_non_empty", False),
    ("use_percentiles", False),
    ("letter_box_transform", True),
    # The forced pad NOT landing on the served pipeline. This is the mutation that
    # keeps SAFE-01/5's updated expectation from being a rubber stamp: the assertion
    # was changed from "the pad must be off" to "the pad must be forced on", and an
    # assertion that has only ever been GREEN on the right value has never been tested.
    ("served_letter_box_transform", False),
    ("crop_fraction", None),
    # The camera-view ORDER the checkpoint declares. LIBERO-style keys are the
    # realistic wrong value: they match no served camera, so upstream degrades to
    # ALPHABETICAL order with one logging.warning and the model gets the wrong view in
    # the wrong slot while every shape check passes.
    ("video_modality_keys", ("image", "wrist_image")),
    ("preprocessor_training", True),
)

_CACHED_SNAPSHOT: GrootGuardSnapshot | None = None


def real_snapshot() -> GrootGuardSnapshot:
    'The config-only snapshot of the REAL checkpoint, built once and cached.'
    global _CACHED_SNAPSHOT
    if _CACHED_SNAPSHOT is None:
        if not REAL_CHECKPOINT.is_dir():
            pytest.fail(
                f"checkpoint not found at {REAL_CHECKPOINT} — the SAFE-01 guard cannot be "
                "calibrated against the real config shape, and a skip here would be a "
                "silent pass"
            )
        _CACHED_SNAPSHOT = snapshot_from_checkpoint_dir(
            REAL_CHECKPOINT, configured_actions_per_chunk=EXPECTED_HORIZON
        )
    return _CACHED_SNAPSHOT


def guard_attribute_names() -> set[str]:
    """Every attribute name the guard module READS, by AST — not by substring.

    AST-based on purpose: the module's docstring explains ``normalization_mapping`` at
    length, so a substring check would be permanently red for the wrong reason.
    """
    tree = ast.parse(GUARD_SOURCE.read_text())
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def test_real_checkpoint_snapshot_passes_the_guard():
    "The REAL checkpoint's config-only snapshot passes, and reports the pinned facts."
    snapshot = real_snapshot()

    assert assert_groot_serving_contract(snapshot) is None

    assert snapshot.is_raw_checkpoint is True
    assert snapshot.assets_present is True
    assert snapshot.embodiment_tag == EXPECTED_TAG == "new_embodiment"
    assert snapshot.checkpoint_horizon == EXPECTED_HORIZON == 16
    assert snapshot.use_relative_action is True
    assert snapshot.use_percentiles is True
    assert snapshot.stats_non_empty is True
    # The checkpoint DECLARES the letterbox off; the serving path forces it ON. Both
    # are asserted, because the guard's whole job at SAFE-01/5 is that they disagree in
    # exactly this direction. A config-only snapshot cannot observe the served value, so
    # it is set to the passing value by construction and documented as such — the live
    # proof is tests/policy/test_lerobot_serving_live.py.
    assert snapshot.letter_box_transform is EXPECTED_CHECKPOINT_LETTER_BOX_TRANSFORM is False
    assert snapshot.served_letter_box_transform is SERVING_LETTER_BOX_TRANSFORM is True
    assert snapshot.crop_fraction == EXPECTED_CROP_FRACTION == 0.95
    assert snapshot.shortest_image_edge == EXPECTED_SHORTEST_IMAGE_EDGE == 256
    assert snapshot.use_albumentations is True
    # The checkpoint's OWN camera-view order, which is NOT features.CAMERA_KEYS'
    # ("wrist", "front") -- the checkpoint decides the slots, the client only decides
    # which feature keys exist.
    assert snapshot.video_modality_keys == EXPECTED_VIDEO_MODALITY_KEYS == ("front", "wrist")


# Each drives the pure guard with a single-field ``dataclasses.replace`` mutation of the
# snapshot built from the REAL checkpoint. No test mutates the checkpoint on disk, and no


def test_violation_1a_non_raw_checkpoint_raises():
    bad = dataclasses.replace(real_snapshot(), is_raw_checkpoint=False)
    with pytest.raises(ValueError, match=r"^SAFE-01/1 ") as excinfo:
        assert_groot_serving_contract(bad)
    assert str(bad.base_model_path) in str(excinfo.value)
    assert "nvidia/GR00T-N1.7-3B" in str(excinfo.value)


def test_violation_1b_absent_checkpoint_assets_raises():
    bad = dataclasses.replace(real_snapshot(), assets_present=False)
    with pytest.raises(ValueError, match=r"^SAFE-01/1 ") as excinfo:
        assert_groot_serving_contract(bad)
    assert "False" in str(excinfo.value)
    assert str(bad.base_model_path) in str(excinfo.value)


def test_violation_2a_wrong_embodiment_tag_raises():
    wrong_tag = "oxe_droid_relative_eef_relative_joint"
    bad = dataclasses.replace(real_snapshot(), embodiment_tag=wrong_tag)
    with pytest.raises(ValueError, match=r"^SAFE-01/2 ") as excinfo:
        assert_groot_serving_contract(bad)
    assert wrong_tag in str(excinfo.value)
    assert "16" in str(excinfo.value)


def test_violation_2b_horizon_forty_raises():
    bad = dataclasses.replace(real_snapshot(), checkpoint_horizon=40)
    with pytest.raises(ValueError, match=r"^SAFE-01/2 ") as excinfo:
        assert_groot_serving_contract(bad)
    assert "40" in str(excinfo.value)
    assert "16" in str(excinfo.value)


def test_violation_2c_configured_actions_per_chunk_disagrees_with_delta_indices_raises():
    bad = dataclasses.replace(real_snapshot(), configured_actions_per_chunk=40)
    with pytest.raises(ValueError, match=r"^SAFE-01/2 ") as excinfo:
        assert_groot_serving_contract(bad)
    assert "40" in str(excinfo.value)
    assert "16" in str(excinfo.value)


def test_violation_3a_relative_action_decoding_off_raises():
    bad = dataclasses.replace(real_snapshot(), use_relative_action=False)
    with pytest.raises(ValueError, match=r"^SAFE-01/3 ") as excinfo:
        assert_groot_serving_contract(bad)
    assert "False" in str(excinfo.value)
    assert "1.83" in str(excinfo.value)


def test_violation_3b_legacy_decode_step_raises():
    legacy = "GrootActionUnpackUnnormalizeStep"
    bad = dataclasses.replace(real_snapshot(), decode_step_type=legacy)
    with pytest.raises(ValueError, match=r"^SAFE-01/3 ") as excinfo:
        assert_groot_serving_contract(bad)
    assert legacy in str(excinfo.value)
    assert EXPECTED_DECODE_STEP in str(excinfo.value)


def test_violation_4a_empty_statistics_raises():
    bad = dataclasses.replace(real_snapshot(), stats_non_empty=False)
    with pytest.raises(ValueError, match=r"^SAFE-01/4 ") as excinfo:
        assert_groot_serving_contract(bad)
    assert "False" in str(excinfo.value)
    assert "[-1, 1]" in str(excinfo.value)


def test_violation_4b_percentiles_off_raises():
    bad = dataclasses.replace(real_snapshot(), use_percentiles=False)
    with pytest.raises(ValueError, match=r"^SAFE-01/4 ") as excinfo:
        assert_groot_serving_contract(bad)
    assert "False" in str(excinfo.value)
    assert "q01" in str(excinfo.value)


def test_violation_5a_checkpoint_declaring_letterbox_on_raises():
    'A checkpoint that DECLARES the pad on invalidates the recorded verdict.'
    bad = dataclasses.replace(real_snapshot(), letter_box_transform=True)
    with pytest.raises(ValueError, match=r"^SAFE-01/5 ") as excinfo:
        assert_groot_serving_contract(bad)
    message = str(excinfo.value)
    assert "True" in message
    assert "DECLARES" in message
    assert "RE-MEASURE" in message


def test_violation_5a2_served_letterbox_off_raises():
    'THE TEETH OF THE UPDATED ASSERTION: the pad NOT forced is a refusal.'
    bad = dataclasses.replace(real_snapshot(), served_letter_box_transform=False)
    with pytest.raises(ValueError, match=r"^SAFE-01/5 ") as excinfo:
        assert_groot_serving_contract(bad)
    message = str(excinfo.value)
    assert "False" in message
    assert "SERVED" in message
    # The message must name BOTH geometries, so an operator learns what the server was
    # about to feed the VLM and what it should have fed it.
    assert "(256, 340, 3)" in message
    assert "(256, 256, 3)" in message


def test_violation_5b_wrong_crop_fraction_or_shortest_edge_raises():
    """All three geometry knobs discriminate independently, and each prints all three."""
    for field_name, value in (
        ("crop_fraction", None),
        ("shortest_image_edge", 224),
        ("use_albumentations", False),
    ):
        bad = dataclasses.replace(real_snapshot(), **{field_name: value})
        with pytest.raises(ValueError, match=r"^SAFE-01/5 ") as excinfo:
            assert_groot_serving_contract(bad)
        message = str(excinfo.value)
        assert str(value) in message, (field_name, message)
        # One message prints all three observed values, so an operator never has to
        # re-run to learn which of the three moved.
        assert "crop_fraction=" in message
        assert "shortest_image_edge=" in message
        assert "use_albumentations=" in message


def test_violation_5d_wrong_video_modality_keys_raises():
    'The camera-view ORDER is guarded, not merely documented (WR-05).'
    for wrong in (
        ("image", "wrist_image"),
        ("wrist", "front"),
        ("front",),
        (),
    ):
        bad = dataclasses.replace(real_snapshot(), video_modality_keys=wrong)
        with pytest.raises(ValueError, match=r"^SAFE-01/5 ") as excinfo:
            assert_groot_serving_contract(bad)
        message = str(excinfo.value)
        assert str(wrong) in message, (wrong, message)
        assert str(EXPECTED_VIDEO_MODALITY_KEYS) in message, (wrong, message)
        # The message must state the SILENT mechanism, or a reader concludes a
        # mismatch would have raised somewhere downstream.
        assert "ALPHABETICAL" in message, message
        assert "logging.warning" in message, message

    # A PERMUTATION is not merely "a different tuple": ("wrist", "front") is
    # features.CAMERA_KEYS itself, which is the value a future reader is most likely to
    # assume is authoritative. It must be refused.
    assert tuple(features.CAMERA_KEYS) != EXPECTED_VIDEO_MODALITY_KEYS, (
        "features.CAMERA_KEYS and the checkpoint's modality_keys are no longer "
        "different orderings, so this test no longer proves the checkpoint is the "
        "ordering authority"
    )


def test_video_modality_keys_come_from_the_checkpoint_not_from_camera_keys():
    "The guard reads the CHECKPOINT's declaration, never the client's CAMERA_KEYS."
    processor_kwargs = json.loads((REAL_CHECKPOINT / "processor_config.json").read_text())[
        "processor_kwargs"
    ]
    on_disk = tuple(processor_kwargs["modality_configs"][EXPECTED_TAG]["video"]["modality_keys"])

    assert on_disk == EXPECTED_VIDEO_MODALITY_KEYS
    assert real_snapshot().video_modality_keys == on_disk
    assert set(on_disk) == set(features.CAMERA_KEYS), (
        "the checkpoint names cameras the client does not serve (or vice versa); "
        "_ordered_image_keys would then feed fewer views than the weights expect"
    )
    assert on_disk != tuple(features.CAMERA_KEYS), (
        "the two tuples now agree on ORDER, so this test can no longer show which of "
        "them the guard read"
    )


def test_violation_5c_processor_in_training_mode_raises():
    for field_name in ("preprocessor_training", "encode_step_training"):
        bad = dataclasses.replace(real_snapshot(), **{field_name: True})
        with pytest.raises(ValueError, match=r"^SAFE-01/5 ") as excinfo:
            assert_groot_serving_contract(bad)
        message = str(excinfo.value)
        assert "True" in message
        assert "random crop" in message
        assert "state dropout" in message


def test_snapshot_from_checkpoint_dir_raises_on_missing_config_json(tmp_path):
    """A directory without the sidecars raises, naming the path — never a defaulted snapshot.

    A snapshot whose fields silently defaulted would PASS the guard, which is threat
    T-06-08 and the exact vacuous-pass failure class this phase exists to remove.
    """
    empty = tmp_path / "not-a-checkpoint"
    empty.mkdir()
    with pytest.raises(ValueError) as excinfo:
        snapshot_from_checkpoint_dir(empty, configured_actions_per_chunk=EXPECTED_HORIZON)
    assert str(empty / "config.json") in str(excinfo.value)

    absent = tmp_path / "does-not-exist"
    with pytest.raises(ValueError) as excinfo:
        snapshot_from_checkpoint_dir(absent, configured_actions_per_chunk=EXPECTED_HORIZON)
    assert str(absent) in str(excinfo.value)


def test_snapshot_from_loaded_never_leaves_decode_step_type_empty():
    '``SAFE-01/3`` must fire on a decode-step-less pipeline, never an ``IndexError``.'
    from lerobot.policies.groot.configuration_groot import GrootConfig

    class Pipeline:
        def __init__(self, steps, name="stand-in"):
            self.steps = steps
            self.name = name

    class PackLike:
        state_dropout_prob = 0.2
        training = False

    class EncodeLike:
        # True, not False: this stands in for the SERVED pipeline, whose pad the
        # server forces on. A stand-in carrying the checkpoint's declared False
        # would be a stand-in for a pipeline the server never builds, and it would
        # make the healthy assertion below fail for the right reason at the wrong
        # place (SAFE-01/5's served-value check, which test_violation_5a2 owns).
        letter_box_transform = True
        training = False

    class GrootN17ActionDecodeStep:  # name is the assertion
        env_action_dim = 6

    class DeviceLike:
        pass

    config = GrootConfig(base_model_path=str(REAL_CHECKPOINT))
    preprocessor = Pipeline([PackLike(), EncodeLike()])

    healthy = snapshot_from_loaded(
        config,
        preprocessor,
        Pipeline([GrootN17ActionDecodeStep(), DeviceLike()]),
        configured_actions_per_chunk=EXPECTED_HORIZON,
    )
    assert healthy.decode_step_type == EXPECTED_DECODE_STEP
    assert assert_groot_serving_contract(healthy) is None

    no_decode_step = snapshot_from_loaded(
        config,
        preprocessor,
        Pipeline([DeviceLike()]),
        configured_actions_per_chunk=EXPECTED_HORIZON,
    )
    assert no_decode_step.decode_step_type == NO_DECODE_STEP
    assert no_decode_step.decode_step_type != ""
    with pytest.raises(ValueError, match=r"^SAFE-01/3 "):
        assert_groot_serving_contract(no_decode_step)

    # A base_model_path that is not a raw N1.7 checkpoint (here: unset, so it resolves to
    # the hub default) returns early rather than reading sidecars that are not there, and
    # the guard reports SAFE-01/1 — the message an operator can act on.
    unset_path = snapshot_from_loaded(
        GrootConfig(base_model_path=None),
        preprocessor,
        Pipeline([GrootN17ActionDecodeStep()]),
        configured_actions_per_chunk=EXPECTED_HORIZON,
    )
    assert unset_path.decode_step_type != ""
    with pytest.raises(ValueError, match=r"^SAFE-01/1 "):
        assert_groot_serving_contract(unset_path)


def test_snapshot_from_loaded_reads_the_served_letterbox_off_the_encode_step():
    '``served_letter_box_transform`` comes from the PIPELINE, not the checkpoint.'
    from lerobot.policies.groot.configuration_groot import GrootConfig

    class Pipeline:
        def __init__(self, steps, name="stand-in"):
            self.steps = steps
            self.name = name

    class PackLike:
        state_dropout_prob = 0.2
        training = False

    class EncodeLike:
        def __init__(self, letter_box_transform):
            self.letter_box_transform = letter_box_transform
            self.training = False

    class GrootN17ActionDecodeStep:  # name is the assertion
        env_action_dim = 6

    config = GrootConfig(base_model_path=str(REAL_CHECKPOINT))
    postprocessor = Pipeline([GrootN17ActionDecodeStep()])

    forced = snapshot_from_loaded(
        config,
        Pipeline([PackLike(), EncodeLike(letter_box_transform=True)]),
        postprocessor,
        configured_actions_per_chunk=EXPECTED_HORIZON,
    )
    assert forced.served_letter_box_transform is True
    # The checkpoint still declares the opposite — that is what makes the line above
    # evidence that the pipeline, not the sidecar, was read.
    assert forced.letter_box_transform is False
    assert assert_groot_serving_contract(forced) is None

    dropped = snapshot_from_loaded(
        config,
        Pipeline([PackLike(), EncodeLike(letter_box_transform=False)]),
        postprocessor,
        configured_actions_per_chunk=EXPECTED_HORIZON,
    )
    assert dropped.served_letter_box_transform is False
    with pytest.raises(ValueError, match=r"^SAFE-01/5 "):
        assert_groot_serving_contract(dropped)


def test_serving_preprocessor_overrides_is_a_single_definition():
    "The override fragment the server merges is one value, shaped for upstream's seam."
    from dataclasses import fields

    from lerobot.policies.groot.processor_groot import GrootN17VLMEncodeStep

    assert serving_preprocessor_overrides() == {
        VLM_ENCODE_STEP_KEY: {"letter_box_transform": SERVING_LETTER_BOX_TRANSFORM}
    }
    assert getattr(GrootN17VLMEncodeStep, "_registry_name", None) == VLM_ENCODE_STEP_KEY, (
        "the registry name moved, so the override key would match no step and "
        "_apply_groot_step_overrides would raise KeyError at every handshake"
    )
    init_fields = {f.name for f in fields(GrootN17VLMEncodeStep) if f.init}
    assert "letter_box_transform" in init_fields, (
        "letter_box_transform is no longer an init field of GrootN17VLMEncodeStep, so "
        "the override would raise TypeError; re-derive the injection against the new shape "
        "rather than reaching for a monkeypatch"
    )


# --- Documented negatives: mechanisms proven non-discriminating ---------------


def test_normalization_mapping_is_identity_by_design_and_is_not_a_discriminator():
    "DOCUMENTS a negative. It is never the verdict's evidence."
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.configs.types import NormalizationMode

    if not REAL_CHECKPOINT.is_dir():
        pytest.fail(f"checkpoint not found at {REAL_CHECKPOINT} — a skip here would be a silent pass")

    mapping = GrootConfig(base_model_path=str(REAL_CHECKPOINT)).normalization_mapping
    assert set(mapping) == {"VISUAL", "STATE", "ACTION"}
    for feature_type, mode in mapping.items():
        assert mode is NormalizationMode.IDENTITY, (feature_type, mode)

    # Identity here is HEALTHY: the same snapshot passes the guard.
    assert assert_groot_serving_contract(real_snapshot()) is None

    # And the guard reads no attribute of that name — AST, so the module docstring's
    # lengthy explanation of the field cannot make this red.
    assert "normalization_mapping" not in guard_attribute_names()


def test_chunk_length_sixteen_is_not_horizon_evidence():
    'DOCUMENTS why the guard asserts CONFIG-level horizon values, never a chunk length.'
    from lerobot.lerobot_types import TransitionKey  # NOT lerobot.processor.core
    from lerobot.policies.groot.processor_groot import GrootN17ActionDecodeStep

    if not REAL_CHECKPOINT.is_dir():
        pytest.fail(f"checkpoint not found at {REAL_CHECKPOINT} — a skip here would be a silent pass")

    processor_kwargs = json.loads((REAL_CHECKPOINT / "processor_config.json").read_text())[
        "processor_kwargs"
    ]
    modality_config = processor_kwargs["modality_configs"][EXPECTED_TAG]
    raw_stats = json.loads((REAL_CHECKPOINT / "statistics.json").read_text())[EXPECTED_TAG]

    class Anchor:
        """Stands in for GrootN17PackInputsStep's cached raw state."""

        def get_cached_raw_state(self):
            return {
                "single_arm": np.zeros((1, 5), np.float32),
                "gripper": np.zeros((1, 1), np.float32),
            }

    decode_step = GrootN17ActionDecodeStep(
        env_action_dim=6,
        raw_stats=raw_stats,
        modality_config=modality_config,
        use_percentiles=bool(processor_kwargs["use_percentiles"]),
        use_relative_action=bool(processor_kwargs["use_relative_action"]),
        pack_step=Anchor(),
    )

    for input_horizon in (16, 40, 50):
        decoded = decode_step({TransitionKey.ACTION: torch.zeros((1, input_horizon, 6))})
        assert tuple(decoded[TransitionKey.ACTION].shape) == (1, EXPECTED_HORIZON, 6), input_horizon


# --- The guard's own consistency gate ----------------------------------------


def test_every_message_names_its_assertion_id_and_the_observed_value():
    'Every raised message starts with ``SAFE-01/N `` and embeds the observed value.'
    assert len(VIOLATION_MUTATIONS) == 14

    for field_name, value in VIOLATION_MUTATIONS:
        bad = dataclasses.replace(real_snapshot(), **{field_name: value})
        with pytest.raises(ValueError) as excinfo:
            assert_groot_serving_contract(bad)
        message = str(excinfo.value)
        assert re.match(r"^SAFE-01/[1-5] ", message), (field_name, message)
        assert str(value) in message, (field_name, message)
