"""Hermetic SAFE-01 gate for the GR00T serving contract.

Defends the five conditions :func:`policy_guard.groot_guard.assert_groot_serving_contract`
asserts: the resolved ``base_model_path`` really is our raw fine-tune, the horizon is 16 at
CONFIG level, relative-action decoding is on and native, normalization has not fallen back
to identity, and the image geometry plus eval-mode determinism are the ones this checkpoint
was trained with. It satisfies criterion 2's *"mis-flagging the launch is a test, not a
hypothetical"* KEYLESS — one snapshot built config-only from the REAL checkpoint, then one
single-field mutation per violation — instead of five ~6 GB weight loads.

**The originally designated mechanism is proven non-discriminating here, not reused.** The
roadmap names "normalization has fallen back to identity" as a SAFE-01 condition, and the
field that name points at, ``GrootConfig.normalization_mapping``, is IDENTITY for
``VISUAL``, ``STATE`` and ``ACTION`` **by design** — upstream states at
``configuration_groot.py:258-269`` that GR00T normalizes internally in its processor steps
and that the mapping "is not consulted by make_groot_pre_post_processors". An assertion
written against it would fail on every healthy launch.
``test_normalization_mapping_is_identity_by_design_and_is_not_a_discriminator`` exists to
DOCUMENT that negative — it is never the verdict's evidence. The three DISCRIMINATING
signals that carry the condition instead are the decode step's class
(``GrootN17ActionDecodeStep``, not the legacy ``GrootActionUnpackUnnormalizeStep``), a
non-empty checkpoint stats table, and ``use_percentiles``.

The same is true of the horizon: ``test_chunk_length_sixteen_is_not_horizon_evidence``
documents that an emitted chunk length of 16 proves nothing, because three independent
truncations force 16 regardless of configuration.

**SAFE-01/5's letterbox expectation was INVERTED, and the inversion is proven RED.** The
guard used to require ``letter_box_transform`` False. It now requires two different things
of two different values: the CHECKPOINT must still declare ``False`` (a drift catcher — every
recorded geometry number was measured against that declaration), and the SERVED pipeline must
carry ``True``, because Isaac-GR00T padded to a square when it trained these weights and
LeRobot's flag-honouring path fed the model a geometry Isaac never emitted. Changing a guard's
expected value is exactly the edit that can silently defang it, so
``test_violation_5a2_served_letterbox_off_raises`` and
``test_snapshot_from_loaded_reads_the_served_letterbox_off_the_encode_step`` prove the new
assertion fires on a wrong geometry rather than merely passing on the right one. **The
"trained on the padded square" half is an INFERENCE accepted knowingly** — it follows from
"Isaac trained this checkpoint", the training recipe was never read, and no local artifact
records it.

Every test is hermetic: it reads only the real checkpoint's sidecar JSONs and the pinned
constants in ``policy_guard/groot_guard.py``, and needs no serial port, no camera, no
network, no GPU and no weights. **Nothing here may skip** — a skipped guard test is a
silent pass on the one fact this phase turns on. An absent checkpoint directory calls
``pytest.fail`` naming the path.
"""

import ast
import dataclasses
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import the pinned values rather than restating them: the guard and its tests must share
# ONE source of truth, or a drift in one silently satisfies the other.
from policy_guard.groot_guard import (  # noqa: E402
    EXPECTED_CHECKPOINT_LETTER_BOX_TRANSFORM,
    EXPECTED_CROP_FRACTION,
    EXPECTED_DECODE_STEP,
    EXPECTED_HORIZON,
    EXPECTED_SHORTEST_IMAGE_EDGE,
    EXPECTED_TAG,
    NO_DECODE_STEP,
    SERVING_LETTER_BOX_TRANSFORM,
    VLM_ENCODE_STEP_KEY,
    GrootGuardSnapshot,
    assert_groot_serving_contract,
    serving_preprocessor_overrides,
    snapshot_from_checkpoint_dir,
    snapshot_from_loaded,
)

REAL_CHECKPOINT = REPO_ROOT / "checkpoints" / "GR00T-N1.7-3B-SO101"

GUARD_SOURCE = REPO_ROOT / "policy_guard" / "groot_guard.py"

#: The thirteen single-field mutations of the REAL snapshot, one per assertion the guard
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
    ("preprocessor_training", True),
)

_CACHED_SNAPSHOT: GrootGuardSnapshot | None = None


def real_snapshot() -> GrootGuardSnapshot:
    """The config-only snapshot of the REAL checkpoint, built once and cached.

    Fails loudly rather than skipping when the checkpoint is absent: the whole point of
    building against the real directory is D-05's calibration requirement, and a skip here
    would be a silent pass.
    """
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
    """The REAL checkpoint's config-only snapshot passes, and reports the pinned facts.

    This is D-05's calibration leg. Asserting the field VALUES as well as the verdict is
    what prevents a fixture that misrepresents the real config shape from passing here
    while the live guard never fires.
    """
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
    # proof is tests/test_lerobot_serving_live.py.
    assert snapshot.letter_box_transform is EXPECTED_CHECKPOINT_LETTER_BOX_TRANSFORM is False
    assert snapshot.served_letter_box_transform is SERVING_LETTER_BOX_TRANSFORM is True
    assert snapshot.crop_fraction == EXPECTED_CROP_FRACTION == 0.95
    assert snapshot.shortest_image_edge == EXPECTED_SHORTEST_IMAGE_EDGE == 256
    assert snapshot.use_albumentations is True


# --- Negative tests: the guard has teeth -------------------------------------
#
# Each drives the pure guard with a single-field ``dataclasses.replace`` mutation of the
# snapshot built from the REAL checkpoint. No test mutates the checkpoint on disk, and no
# test fabricates a snapshot from scratch: an assertion that has never been red is an
# assertion that has never been tested, and a fabricated fixture cannot prove redness
# against the real config shape.


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
    """A checkpoint that DECLARES the pad on invalidates the recorded verdict.

    Not a correctness claim about the pad — the geometry would still come out square.
    It is a DRIFT claim: every recorded geometry number (both shapes, both digests, the
    cross-backend agreement) was measured against a checkpoint declaring False, and the
    forced-pad decision was justified against that reading. A checkpoint declaring
    otherwise has a different image recipe, so the verdict must be re-measured rather
    than assumed to carry over.
    """
    bad = dataclasses.replace(real_snapshot(), letter_box_transform=True)
    with pytest.raises(ValueError, match=r"^SAFE-01/5 ") as excinfo:
        assert_groot_serving_contract(bad)
    message = str(excinfo.value)
    assert "True" in message
    assert "DECLARES" in message
    assert "RE-MEASURE" in message


def test_violation_5a2_served_letterbox_off_raises():
    """THE TEETH OF THE UPDATED ASSERTION: the pad NOT forced is a refusal.

    SAFE-01/5's letterbox expectation was inverted by this fix — from "the pad must be
    off" to "the pad must be forced on for the served pipeline". Proving the new
    expectation goes GREEN on the right value proves nothing; this test proves it goes
    RED on the wrong one, against a snapshot built from the REAL checkpoint with exactly
    one field mutated.

    The failure this catches is specific and silent: if the override in
    ``docker/lerobot-policy/server.py`` stopped landing (a moved step registry name, a
    dropped kwarg), the server would happily serve ``(256, 340, 3)`` — a geometry the
    weights never saw — with every log line looking healthy.
    """
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
    """``SAFE-01/3`` must fire on a decode-step-less pipeline, never an ``IndexError``.

    Plan 06-03 owns the real wiring, so the loaded builder is exercised here with
    minimal stand-ins carrying only the marker attributes it locates steps by. That is
    enough to prove the contract this plan owes 06-03: ``decode_step_type`` is a non-empty
    string on EVERY path, including the no-candidate path and the non-raw-path early
    return, so a missing decode step surfaces as a named SAFE-01/3 error rather than as an
    index error escaping from inside a request handler.
    """
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
    """``served_letter_box_transform`` comes from the PIPELINE, not the checkpoint.

    The distinction is the whole mechanism: the checkpoint declares ``false``, so if
    this field were derived from ``processor_config.json`` it would read ``False`` on a
    correctly-serving server and the guard would refuse every healthy handshake. It must
    be read off the built step, and it must track that step rather than a constant.

    Both directions are driven, against the SAME real config, so neither answer can be
    a coincidence: a step with the pad forced ON passes, and a step with the pad OFF —
    i.e. the override silently not landing — is REFUSED by SAFE-01/5.
    """
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
    """The override fragment the server merges is one value, shaped for upstream's seam.

    Asserted rather than assumed because it is the seam the whole fix rides on: the key
    must be the step's REGISTRY name (upstream's ``from_pretrained`` matcher accepts only
    registry names), and the field must be a real ``init`` field of the step, or the
    override would raise at handshake time instead of applying.
    """
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
    """DOCUMENTS a negative. It is never the verdict's evidence.

    The roadmap names "normalization has fallen back to identity" as a SAFE-01 condition,
    and ``GrootConfig.normalization_mapping`` is the field that name points at. It is
    IDENTITY for every feature type on a HEALTHY launch of this checkpoint, and upstream
    states at ``configuration_groot.py:258-269`` that it "is not consulted by
    make_groot_pre_post_processors". So an assertion written against it is a guaranteed
    false alarm, and this test's job is to record that rather than let a later reader
    "restore" the missing check.

    The three DISCRIMINATING signals that carry the condition instead, all asserted by the
    guard: the decode step's class (``GrootN17ActionDecodeStep``, not the legacy
    ``GrootActionUnpackUnnormalizeStep``, which is reached only when the checkpoint's stats
    are unusable), a non-empty checkpoint stats table, and ``use_percentiles``.
    """
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
    """DOCUMENTS why the guard asserts CONFIG-level horizon values, never a chunk length.

    Three independent truncations force an output length of 16 regardless of the input
    horizon (``modeling_groot.py:308-324``, ``policy_server.py:328``, and the decode step's
    own ``valid_horizon`` truncation). So observing a 16-step chunk proves nothing about
    whether the horizon is configured correctly, and a future reader must NOT "simplify"
    SAFE-01/2 into a chunk-length check.

    Keyless and weightless: the decode step is constructed directly from the checkpoint's
    ``statistics.json`` and ``modality_configs``, with a stand-in for
    ``GrootN17PackInputsStep``'s cached raw state.
    """
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
    """Every raised message starts with ``SAFE-01/N `` and embeds the observed value.

    A message that omits the observed value forces an operator into the source to learn
    what actually went wrong, which is exactly what the ``policy/factory.py:57-62`` idiom
    exists to prevent. Driving all thirteen mutations from one table also means an emptied or
    reworded message goes red here even if its own violation test only matched the prefix.
    """
    assert len(VIOLATION_MUTATIONS) == 13

    for field_name, value in VIOLATION_MUTATIONS:
        bad = dataclasses.replace(real_snapshot(), **{field_name: value})
        with pytest.raises(ValueError) as excinfo:
            assert_groot_serving_contract(bad)
        message = str(excinfo.value)
        assert re.match(r"^SAFE-01/[1-5] ", message), (field_name, message)
        assert str(value) in message, (field_name, message)
