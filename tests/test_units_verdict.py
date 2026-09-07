"""Hermetic normalization-units gate for PAR-04 / PAR-06 (roadmap criteria 2, 2a).

Settles, offline and mechanically, which ``MotorNormMode`` the GR00T checkpoint
was trained in. The verdict is ``RANGE_M100_100`` (equivalently
``use_degrees=False``), and it rests on three independent arguments — the
plus-or-minus 100.0 clip fingerprint, the ``elbow_flex`` degrees falsification,
and the ``wrist_roll`` dataset cross-check. See ``docs/UNITS-VERDICT.md`` for
the resolution these tests defend.

**PAR-04's designated mechanism is proven non-discriminating here, not reused.**
PAR-04 specifies "assert the observed ready-pose state falls inside the
checkpoint's ``state`` q01/q99 envelope". That assertion returns PASS under BOTH
candidate conventions on all five arm joints, so it decides nothing.
``test_ready_pose_envelope_passes_under_both_conventions_and_is_not_a_discriminator``
exists to DOCUMENT that negative — it is never the verdict's evidence. The
envelope check is retained only against the ``initial`` pose, whose
``shoulder_lift`` of -102 does discriminate.

Every test is hermetic: it reads only the pinned constants in
``scripts/pose_sweep_units_probe.py`` and needs no serial port, no camera, no
network and no local data artifact. **Nothing here may skip** — a skipped units
test is a silent pass on the one fact this whole phase turns on.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make scripts/ importable so the tests and the harness share ONE source of
# truth for the pinned numbers (same idiom as tests/test_container_contract.py).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from pose_sweep_units_probe import (  # noqa: E402
    CALIBRATION_TICK_RANGES,
    CHECKPOINT_ACTION_STATS,
    CHECKPOINT_STATE_STATS,
    CLIP_FINGERPRINT_TOL,
    DATASET_EPISODE0_STATE_STATS,
    DEG_PER_PCT_PINNED,
    DUME_POSES,
    GRIPPER_NORM_MODE,
    JOINT_NAMES,
    MAX_RES,
    clip_fingerprint_count,
    deg_per_pct_table,
    degrees_reachable_range,
    degrees_to_percent,
    envelope_contains,
    norm_mode_for,
    normalize_degrees,
    normalize_m100_100,
    normalize_v033,
    normalize_v061,
)

ARM_JOINTS = JOINT_NAMES[:5]
NORM_MODES = ("RANGE_M100_100", "RANGE_0_100", "DEGREES")


# --- Argument 1: the +/-100.0 clip fingerprint (primary, assumption-free) ----


def test_checkpoint_state_space_shows_range_m100_100_clip_fingerprint():
    """Exact +/-100.0 at a recorded bound is the RANGE_M100_100 clip signature.

    RANGE_M100_100 maps the CLAMPED raw tick onto [-100, +100], so its endpoints
    are exactly +/-100.0 by construction. The DEGREES branch has no clamp and
    therefore no such boundary at all. This is the primary written verdict
    because it assumes nothing about which calibration was in use at training
    time.
    """
    count = clip_fingerprint_count()
    assert count >= 5, (
        f"expected at least 5 saturated (joint, bound) pairs as the "
        f"RANGE_M100_100 clip fingerprint, found {count} — the units verdict "
        f"may have changed"
    )

    # The fingerprint is exact, not rounded: equality holds at 1e-9.
    saturated = [
        (group_name, bound, joint_index, value)
        for group_name, group in (
            ("state", CHECKPOINT_STATE_STATS),
            ("action", CHECKPOINT_ACTION_STATS),
        )
        for bound in ("min", "max")
        for joint_index, value in enumerate(group["single_arm"][bound])
        if abs(abs(value) - 100.0) < CLIP_FINGERPRINT_TOL
    ]
    assert len(saturated) == count
    for _group, _bound, _index, value in saturated:
        assert abs(value) == 100.0

    # The mechanism behind the fingerprint: past the calibrated span the clamped
    # branch pins at exactly 100.0 while the unclamped DEGREES branch keeps
    # growing. That asymmetry is why only one convention can produce the bound.
    cal = CALIBRATION_TICK_RANGES["elbow_flex"]
    lo, hi = cal["range_min"], cal["range_max"]
    at_bound = normalize_m100_100(hi, lo, hi)
    past_bound = normalize_m100_100(hi + 200, lo, hi)
    assert at_bound == 100.0
    assert past_bound == 100.0, "RANGE_M100_100 must clamp, producing the bound"
    assert normalize_degrees(hi + 200, lo, hi) > normalize_degrees(hi, lo, hi), (
        "the DEGREES branch must be unclamped — the absence of the clamp is the "
        "entire basis of the clip fingerprint"
    )


# --- Argument 2: the elbow_flex degrees falsification ------------------------


def test_degrees_formula_cannot_reach_recorded_elbow_flex_maximum():
    """elbow_flex's full mechanical span is +/-96.35 deg, but the checkpoint records 100.0.

    Under RANGE_M100_100 a recorded 100.0 is simply the joint driven to its
    calibrated limit. Under DEGREES it is unreachable, because the joint's tick
    span converts to at most 96.35 degrees.
    """
    index = JOINT_NAMES.index("elbow_flex")
    low_deg, high_deg = degrees_reachable_range("elbow_flex")

    assert round(high_deg, 2) == 96.35
    assert round(low_deg, 2) == -96.35

    recorded_max = CHECKPOINT_STATE_STATS["single_arm"]["max"][index]
    recorded_q99 = CHECKPOINT_STATE_STATS["single_arm"]["q99"][index]
    assert recorded_max == 100.0
    assert recorded_q99 == 100.0
    assert recorded_max > high_deg, (
        f"recorded elbow_flex max {recorded_max} must exceed the reachable "
        f"degrees range {high_deg} for the DEGREES hypothesis to be falsified"
    )

    # Every other arm joint IS reachable in degrees — elbow_flex is the single
    # falsifier, so the argument cannot be an artifact of a wholesale mismatch.
    for joint in ARM_JOINTS:
        if joint == "elbow_flex":
            continue
        joint_index = JOINT_NAMES.index(joint)
        _low, high = degrees_reachable_range(joint)
        assert abs(CHECKPOINT_STATE_STATS["single_arm"]["q99"][joint_index]) <= high


# --- Argument 3: the wrist_roll cross-check against the dataset -------------


def test_wrist_roll_percent_conversion_lands_inside_dataset_band():
    """Dum-E's hardcoded -90.0 is -53.64 percent, inside the dataset's band.

    The dataset's episode-0 wrist_roll band is [-58.952, -50.564] with a
    standard deviation of 2.885. Read as percent, Dum-E's pose lands inside it.
    Read as degrees, -90.0 is more than 10 standard deviations outside — so the
    two hypotheses give different answers, which is what makes this evidence.
    """
    index = JOINT_NAMES.index("wrist_roll")
    as_degrees = DUME_POSES["ready"][index]
    assert as_degrees == -90.0

    as_percent = degrees_to_percent(DUME_POSES["ready"][:5])[index]
    assert round(as_percent, 2) == -53.64

    band_low = DATASET_EPISODE0_STATE_STATS["min"][index]
    band_high = DATASET_EPISODE0_STATE_STATS["max"][index]
    mean = DATASET_EPISODE0_STATE_STATS["mean"][index]
    std = DATASET_EPISODE0_STATE_STATS["std"][index]
    assert (band_low, band_high) == (-58.952, -50.564)
    assert std == 2.885

    assert band_low <= as_percent <= band_high, (
        f"the percent reading {as_percent} must land inside the dataset band "
        f"[{band_low}, {band_high}]"
    )
    assert not (band_low <= as_degrees <= band_high)
    assert abs(as_degrees - mean) / std > 10.0, (
        "the degrees reading must be more than 10 standard deviations outside "
        "the band, otherwise this cross-check does not discriminate"
    )
    # Every pose hardcodes the same wrist_roll, so the cross-check covers all of them.
    for pose in DUME_POSES.values():
        assert pose[index] == -90.0


# --- PAR-04's own mechanism: proven non-discriminating, not reused -----------


def test_ready_pose_envelope_passes_under_both_conventions_and_is_not_a_discriminator():
    """The ready pose is inside the envelope under BOTH conventions, on all five joints.

    This test exists to record a NEGATIVE. PAR-04's literal assertion returns
    PASS regardless of which convention is true, so it can never be cited as
    the units verdict's evidence. Documenting the silent-pass hazard is the
    point; if this test ever starts failing, the envelope or the poses moved and
    the verdict must be re-derived rather than re-asserted.
    """
    as_degrees = DUME_POSES["ready"][:5]
    as_percent = degrees_to_percent(as_degrees)

    inside_as_degrees = envelope_contains(as_degrees)
    inside_as_percent = envelope_contains(as_percent)

    assert inside_as_degrees == [True] * 5, (
        f"ready pose read as degrees: {inside_as_degrees} — PAR-04's assertion "
        f"is expected to PASS here"
    )
    assert inside_as_percent == [True] * 5, (
        f"ready pose read as percent: {inside_as_percent} — PAR-04's assertion "
        f"is expected to PASS here too, which is exactly the problem"
    )
    assert inside_as_degrees == inside_as_percent, (
        "the ready-pose envelope check must read identically under both "
        "hypotheses; that is what makes it inadmissible as the verdict"
    )
    # The two readings genuinely differ numerically — the envelope check is blind
    # to a difference that is really there.
    assert as_degrees != as_percent


def test_initial_pose_envelope_discriminates_between_conventions():
    """The initial pose's shoulder_lift of -102 is outside as degrees, inside as percent.

    This is the ONLY envelope check the verdict retains, and it is a boundary
    case: -102 sits just past the q01 of -99.743 as degrees, and just inside it
    at -98.33 as percent. One step either side of the boundary behaves
    differently — unlike the ready pose.
    """
    index = JOINT_NAMES.index("shoulder_lift")
    as_degrees = DUME_POSES["initial"][:5]
    as_percent = degrees_to_percent(as_degrees)

    assert as_degrees[index] == -102.0
    assert round(as_percent[index], 2) == -98.33

    inside_as_degrees = envelope_contains(as_degrees)
    inside_as_percent = envelope_contains(as_percent)

    assert inside_as_degrees[index] is False, (
        "read as degrees, shoulder_lift -102 must fall OUTSIDE the checkpoint "
        "state q01/q99 envelope"
    )
    assert inside_as_percent[index] is True, (
        "read as percent, shoulder_lift -98.33 must fall INSIDE the envelope"
    )
    assert inside_as_degrees != inside_as_percent, (
        "the initial pose must discriminate, otherwise no envelope check does"
    )

    # The boundary itself: q01 sits strictly between the two readings.
    q01 = CHECKPOINT_STATE_STATS["single_arm"]["q01"][index]
    assert q01 == -99.7432632446289
    assert as_degrees[index] < q01 < as_percent[index]

    # Every other joint of the initial pose is inside under both readings, so
    # shoulder_lift is doing the discriminating on its own.
    for other in range(5):
        if other == index:
            continue
        assert inside_as_degrees[other] is True
        assert inside_as_percent[other] is True


# --- Criterion 2a part (a): the upgrade delta is IDENTITY --------------------


def test_upgrade_normalization_delta_is_identity_across_swept_ticks():
    """The 0.3.3 and 0.6.1 normalization formulas agree bit-for-bit.

    The upstream ``_normalize`` body is byte-identical between the two versions
    (the cited evidence). This sweep is the mechanical re-check that the two
    independently transcribed implementations in the harness agree across every
    raw tick in the servo's domain, in all three modes — including ticks outside
    the calibrated span, where the clamped and unclamped branches diverge from
    each other but must not diverge between versions.
    """
    compared = 0
    for joint, cal in CALIBRATION_TICK_RANGES.items():
        low, high = cal["range_min"], cal["range_max"]
        drive_mode = cal["drive_mode"]
        ticks = set(range(0, MAX_RES + 1, 7))
        ticks.update({low, high, low - 1, high + 1, low + 1, high - 1})
        for tick in sorted(ticks):
            for mode in NORM_MODES:
                from_v033 = normalize_v033(tick, low, high, drive_mode, mode)
                from_v061 = normalize_v061(tick, low, high, drive_mode, mode)
                assert from_v033 == from_v061, (
                    f"{joint} tick={tick} mode={mode}: 0.3.3 gave {from_v033!r} "
                    f"but 0.6.1 gave {from_v061!r} — the upgrade delta is NOT "
                    f"identity and every downstream parity claim is void"
                )
                compared += 1
    assert compared > 5000, f"sweep too thin to be evidence: {compared} comparisons"


# --- Criterion 2a part (b): the real seam is a pure per-joint SCALE ----------


def test_deg_per_pct_table_matches_pinned_values():
    """The degrees-to-percent seam is a pure per-joint scale with pinned magnitudes.

    No sign flip and no additive offset: every motor has drive_mode 0, so both
    modes share the midpoint ``(range_min + range_max) / 2`` and the ratio is a
    single multiplicative factor per joint.
    """
    table = deg_per_pct_table()

    assert [round(table[joint], 5) for joint in ARM_JOINTS] == DEG_PER_PCT_PINNED
    assert DEG_PER_PCT_PINNED == [1.16527, 1.03736, 0.96352, 1.00791, 1.6778]
    assert MAX_RES == 4095, "the servo resolution constant is 4096 minus one"

    # The gripper is excluded: it is RANGE_0_100 under both settings, so there is
    # no degrees-to-percent scale for it at all.
    assert "gripper" not in table
    assert set(table) == set(ARM_JOINTS)

    # Every factor is positive (no sign flip) and none is unity (a real seam).
    assert all(factor > 0.0 for factor in table.values())

    # The pinned extremes: wrist_flex diverges least, wrist_roll most.
    divergence = {joint: (factor - 1.0) * 100.0 for joint, factor in table.items()}
    assert min(divergence, key=lambda joint: abs(divergence[joint])) == "wrist_flex"
    assert max(divergence, key=lambda joint: abs(divergence[joint])) == "wrist_roll"
    assert round(divergence["wrist_flex"], 2) == 0.79
    assert round(divergence["wrist_roll"], 2) == 67.78

    # The scale is exactly (span * 360) / (MAX_RES * 200), in float64, unrounded.
    for joint in ARM_JOINTS:
        cal = CALIBRATION_TICK_RANGES[joint]
        span = cal["range_max"] - cal["range_min"]
        assert table[joint] == (span * 360) / (MAX_RES * 200)


def test_gripper_needs_no_conversion_under_either_mode():
    """The gripper is RANGE_0_100 whether use_degrees is True or False.

    Upstream hardcodes the gripper's norm mode instead of gating it on
    ``use_degrees``, so the gripper's reported value is identical under both
    settings and needs no conversion in either direction.
    """
    assert GRIPPER_NORM_MODE == "RANGE_0_100"

    for use_degrees in (True, False):
        assert norm_mode_for("gripper", use_degrees) == "RANGE_0_100"

    # The arm joints DO switch, which is what makes the gripper's invariance a
    # real observation rather than a property of the helper.
    for joint in ARM_JOINTS:
        assert norm_mode_for(joint, True) == "DEGREES"
        assert norm_mode_for(joint, False) == "RANGE_M100_100"

    cal = CALIBRATION_TICK_RANGES["gripper"]
    low, high = cal["range_min"], cal["range_max"]
    for tick in (low, low + 1, 2500, 3000, high - 1, high):
        as_use_degrees_true = normalize_v061(
            tick, low, high, cal["drive_mode"], norm_mode_for("gripper", True)
        )
        as_use_degrees_false = normalize_v061(
            tick, low, high, cal["drive_mode"], norm_mode_for("gripper", False)
        )
        assert as_use_degrees_true == as_use_degrees_false
        assert 0.0 <= as_use_degrees_true <= 100.0

    assert "gripper" not in deg_per_pct_table()
