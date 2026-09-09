"""Hermetic SAFE-01 gate for the GR00T serving contract.

Defends the five conditions :mod:`policy_guard.groot_guard` asserts, and satisfies
criterion 2's "mis-flagging the launch is a test, not a hypothetical" KEYLESS — one
snapshot built config-only from the real checkpoint, then one single-field mutation per
violation, instead of five ~6 GB weight loads.

Every test is hermetic: it reads only the real checkpoint's sidecar JSONs and the pinned
constants in ``policy_guard/groot_guard.py``, and needs no serial port, no camera, no
network, no GPU and no weights. **Nothing here may skip** — a skipped guard test is a
silent pass on the one fact this phase turns on. An absent checkpoint directory calls
``pytest.fail`` naming the path.
"""

import dataclasses
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import the pinned values rather than restating them: the guard and its tests must share
# ONE source of truth, or a drift in one silently satisfies the other.
from policy_guard.groot_guard import (  # noqa: E402
    EXPECTED_CROP_FRACTION,
    EXPECTED_DECODE_STEP,
    EXPECTED_HORIZON,
    EXPECTED_SHORTEST_IMAGE_EDGE,
    EXPECTED_TAG,
    GrootGuardSnapshot,
    assert_groot_serving_contract,
    snapshot_from_checkpoint_dir,
)

REAL_CHECKPOINT = REPO_ROOT / "checkpoints" / "GR00T-N1.7-3B-SO101"

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
    assert snapshot.letter_box_transform is False
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


def test_violation_5a_letterbox_transform_on_raises():
    bad = dataclasses.replace(real_snapshot(), letter_box_transform=True)
    with pytest.raises(ValueError, match=r"^SAFE-01/5 ") as excinfo:
        assert_groot_serving_contract(bad)
    assert "True" in str(excinfo.value)


def test_violation_5b_wrong_crop_fraction_or_shortest_edge_raises():
    """All three geometry knobs are independently discriminating, and each prints all three."""
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
