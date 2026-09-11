"""Hermetic normalization-units gate for the joint-value convention.

Settles, offline and mechanically, which ``MotorNormMode`` the GR00T checkpoint
was trained in. The verdict is ``RANGE_M100_100`` (equivalently
``use_degrees=False``), and it rests on three independent arguments — the
plus-or-minus 100.0 clip fingerprint, the ``elbow_flex`` degrees falsification,
and the ``wrist_roll`` dataset cross-check. See ``docs/UNITS-VERDICT.md`` for
the resolution these tests defend.

**The originally designated mechanism is proven non-discriminating here, not
reused.** It specified "assert the observed ready-pose state falls inside the
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

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make scripts/ importable so the tests and the harness share ONE source of
# truth for the pinned numbers (same idiom as tests/test_container_contract.py).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import pose_sweep_units_probe as probe  # noqa: E402
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
    MEASURED_SCALE_TOL,
    assert_pose_reachable,
    calibration_bounds,
    clamp_demo_targets,
    clip_fingerprint_count,
    compare_pre_and_post_upgrade,
    deg_per_pct_table,
    degrees_reachable_range,
    degrees_to_percent,
    envelope_contains,
    envelope_row,
    interpolate_steps,
    measured_deg_per_pct,
    norm_mode_for,
    normalize_degrees,
    normalize_m100_100,
    normalize_v033,
    normalize_v061,
    tick_for_degrees,
    within_calibrated_ticks,
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
    """elbow_flex's full mechanical span is +/-96.57 deg, but the checkpoint records 100.0.

    Under RANGE_M100_100 a recorded 100.0 is simply the joint driven to its
    calibrated limit. Under DEGREES it is unreachable, because the joint's tick
    span converts to at most 96.57 degrees.
    """
    index = JOINT_NAMES.index("elbow_flex")
    low_deg, high_deg = degrees_reachable_range("elbow_flex")

    assert round(high_deg, 2) == 96.57
    assert round(low_deg, 2) == -96.57

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


def test_current_wrist_roll_conversion_no_longer_supports_historical_band():
    """Current calibration maps -90 degrees to -50 percent, outside the old band.

    The historical calibration gave -53.64 percent. Its cross-check remains
    historical evidence; the current arithmetic must not claim to reproduce it.
    """
    index = JOINT_NAMES.index("wrist_roll")
    as_degrees = DUME_POSES["ready"][index]
    assert as_degrees == -90.0

    as_percent = degrees_to_percent(DUME_POSES["ready"][:5])[index]
    assert round(as_percent, 2) == -50.0

    band_low = DATASET_EPISODE0_STATE_STATS["min"][index]
    band_high = DATASET_EPISODE0_STATE_STATS["max"][index]
    mean = DATASET_EPISODE0_STATE_STATS["mean"][index]
    std = DATASET_EPISODE0_STATE_STATS["std"][index]
    assert (band_low, band_high) == (-58.952, -50.564)
    assert std == 2.885

    assert not (band_low <= as_percent <= band_high), (
        f"the current percent reading {as_percent} no longer lands inside the historical band "
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


# --- the originally designated mechanism: non-discriminating, not reused -----


def test_ready_pose_envelope_passes_under_both_conventions_and_is_not_a_discriminator():
    """The ready pose is inside the envelope under BOTH conventions, on all five joints.

    This test exists to record a NEGATIVE. The ready-pose envelope assertion returns
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
        f"ready pose read as degrees: {inside_as_degrees} — the envelope assertion "
        f"is expected to PASS here"
    )
    assert inside_as_percent == [True] * 5, (
        f"ready pose read as percent: {inside_as_percent} — the envelope assertion "
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
    at -98.83 as percent. One step either side of the boundary behaves
    differently — unlike the ready pose.
    """
    index = JOINT_NAMES.index("shoulder_lift")
    as_degrees = DUME_POSES["initial"][:5]
    as_percent = degrees_to_percent(as_degrees)

    assert as_degrees[index] == -102.0
    assert round(as_percent[index], 2) == -98.83

    inside_as_degrees = envelope_contains(as_degrees)
    inside_as_percent = envelope_contains(as_percent)

    assert inside_as_degrees[index] is False, (
        "read as degrees, shoulder_lift -102 must fall OUTSIDE the checkpoint "
        "state q01/q99 envelope"
    )
    assert inside_as_percent[index] is True, (
        "read as percent, shoulder_lift -98.83 must fall INSIDE the envelope"
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
    assert DEG_PER_PCT_PINNED == [1.13934, 1.03209, 0.96571, 1.00396, 1.8]
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
    assert round(divergence["wrist_flex"], 2) == 0.4
    assert round(divergence["wrist_roll"], 2) == 80.0

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


# --- The live half's pure helpers: hermetic, so CI covers them arm-free -------
#
# Plan 05-06 ran the hardware half against a real bus. The MEASUREMENTS live in
# docs/UNITS-VERDICT.md's "## Live confirmation" section and in the gitignored
# corpus output; what belongs in an always-running gate is the arithmetic those
# measurements were derived through, plus the safety guards that decided which
# commands were allowed to reach the arm. Nothing here opens a serial port.


def test_tick_for_degrees_inverts_normalize_degrees():
    """The commanded-tick predictor must invert the reported-degrees formula.

    This is a safety-critical inverse, not a convenience: it is what
    ``assert_pose_reachable`` uses to refuse a command that would drive a joint
    past a mechanical stop.
    """
    for joint in ARM_JOINTS:
        cal = CALIBRATION_TICK_RANGES[joint]
        low, high = float(cal["range_min"]), float(cal["range_max"])
        for tick in (low, (low + high) / 2, high, low + 17.0):
            degrees = normalize_degrees(tick, low, high)
            assert abs(tick_for_degrees(degrees, low, high) - tick) < 1e-9

        assert within_calibrated_ticks(low, low, high)
        assert within_calibrated_ticks(high, low, high)
        assert not within_calibrated_ticks(low - 1, low, high)
        assert not within_calibrated_ticks(high + 1, low, high)


def test_assert_pose_reachable_accepts_dume_poses_and_rejects_a_pose_past_a_stop():
    """Upstream's DEGREES un-normalization has NO clamp, so this guard is required.

    ``_unnormalize``'s DEGREES branch is ``int(val * max_res / 360 + mid)``
    (motors_bus.py :904-907) — unbounded. A commanded degree value outside a
    joint's calibrated span therefore becomes an out-of-range tick and drives the
    servo into a stop. Every one of Dum-E's own poses must pass; a value past
    ``elbow_flex``'s +/-96.57 span must not.

    The calibration is passed in explicitly. On the live path the caller hands it
    ``bus.calibration``; here it is the pinned table, which is what makes the
    +/-96.57 span in this test's arithmetic well-defined.
    """
    for pose in DUME_POSES.values():
        assert_pose_reachable(list(pose), CALIBRATION_TICK_RANGES)

    low, high = degrees_reachable_range("elbow_flex")
    assert round(high, 2) == 96.57

    unreachable = list(DUME_POSES["initial"])
    unreachable[2] = high + 5.0
    try:
        assert_pose_reachable(unreachable, CALIBRATION_TICK_RANGES)
    except ValueError as exc:
        assert "elbow_flex" in str(exc)
        assert "outside" in str(exc)
    else:  # pragma: no cover - the guard must not silently accept it
        raise AssertionError("assert_pose_reachable accepted a pose past a stop")


def test_assert_pose_reachable_requires_a_calibration_rather_than_defaulting():
    """The guard must not fall back to the pinned snapshot when none is supplied.

    The pinned table is wider than the live calibration on three joints, so a
    default would let the guard approve a target the bus maps past a mechanical
    stop — and the DEGREES unnormalize branch has no upstream clamp to catch it.
    A missing calibration has to be a loud programming error.
    """
    with pytest.raises(TypeError):
        assert_pose_reachable(list(DUME_POSES["initial"]))


def test_assert_pose_reachable_rejects_a_pose_the_live_span_no_longer_reaches():
    """A pose inside the pinned span but outside the LIVE span must be refused.

    This is the failure the guard existed to prevent and could not: the pinned
    ``shoulder_pan`` span is 2651 ticks against a live 2592, so a commanded value
    the snapshot calls reachable can land past the live calibrated stop. Using a
    narrowed calibration here reproduces that relationship without needing an arm.
    """
    narrowed = {
        joint: dict(entry) for joint, entry in CALIBRATION_TICK_RANGES.items()
    }
    narrowed["shoulder_pan"].update(range_min=1500, range_max=2600)

    pose = list(DUME_POSES["remote"])
    pose[0] = 60.0  # inside the pinned +/-116.53 span, outside the narrowed one

    assert_pose_reachable(pose, CALIBRATION_TICK_RANGES)
    with pytest.raises(ValueError, match="shoulder_pan"):
        assert_pose_reachable(pose, narrowed)


def test_a_failed_drift_check_refuses_the_whole_arm_half(monkeypatch, capsys):
    """A FAILED drift check must stop the arm half, not merely set the exit code.

    The arm half opens the bus and, with ``--pose-sequence`` / ``--demo-clamp``,
    commands motion — against targets validated with the very constants that just
    failed. This is the fail-open the drift check existed to prevent, so it is
    pinned here rather than rediscovered on hardware. Nothing in this test touches
    a serial port: the three arm-half entry points are replaced with recorders that
    fail the test if they are ever reached.
    """
    reached: list[str] = []

    monkeypatch.setattr(probe, "check_clip_fingerprint", lambda index: True)
    monkeypatch.setattr(probe, "check_degrees_falsification", lambda index: True)
    monkeypatch.setattr(probe, "check_scale_and_cross_check", lambda index: True)
    monkeypatch.setattr(
        probe, "check_envelope_discrimination", lambda index, poses: True
    )
    monkeypatch.setattr(probe, "check_pinned_constants", lambda index, args: False)
    monkeypatch.setattr(
        probe,
        "check_hardware_raw_tick",
        lambda index, args: reached.append("hardware_raw_tick") or True,
    )
    monkeypatch.setattr(
        probe,
        "check_live_pose_sweep",
        lambda index, args, poses: reached.append("live_pose_sweep") or True,
    )
    monkeypatch.setattr(
        probe,
        "check_clamp_demo",
        lambda index, args: reached.append("clamp_demo") or True,
    )
    monkeypatch.setattr(
        sys, "argv", ["probe", "--pose-sequence", "initial", "--demo-clamp"]
    )

    exit_code = probe.main()

    assert reached == [], f"the arm half must not run after a drift failure: {reached}"
    assert exit_code == 1
    assert "REFUSED" in capsys.readouterr().out


def test_a_passing_drift_check_still_lets_the_arm_half_run(monkeypatch):
    """The refusal must be conditional, not a blanket disable of the arm half.

    Without this, a guard that always refused would pass the test above while
    silently removing the probe's whole reason to exist.
    """
    reached: list[str] = []

    monkeypatch.setattr(probe, "check_clip_fingerprint", lambda index: True)
    monkeypatch.setattr(probe, "check_degrees_falsification", lambda index: True)
    monkeypatch.setattr(probe, "check_scale_and_cross_check", lambda index: True)
    monkeypatch.setattr(
        probe, "check_envelope_discrimination", lambda index, poses: True
    )
    monkeypatch.setattr(probe, "check_pinned_constants", lambda index, args: True)
    monkeypatch.setattr(
        probe,
        "check_hardware_raw_tick",
        lambda index, args: reached.append("hardware_raw_tick") or True,
    )
    monkeypatch.setattr(
        probe,
        "check_live_pose_sweep",
        lambda index, args, poses: reached.append("live_pose_sweep") or True,
    )
    monkeypatch.setattr(
        probe,
        "check_clamp_demo",
        lambda index, args: reached.append("clamp_demo") or True,
    )
    monkeypatch.setattr(
        sys, "argv", ["probe", "--pose-sequence", "initial", "--demo-clamp"]
    )

    exit_code = probe.main()

    assert reached == ["hardware_raw_tick", "live_pose_sweep", "clamp_demo"]
    assert exit_code == 0


def test_clamp_demo_targets_exceed_the_clamp_and_clip_to_exactly_it():
    """The demonstration delta must be above the clamp, and clip to the clamp."""
    requested, clipped = clamp_demo_targets(-90.0, 160.0, 1.5)
    assert requested == -90.0 + 240.0
    assert clipped == -90.0 + 160.0
    assert abs(requested - (-90.0)) > 160.0
    assert abs(clipped - (-90.0)) == 160.0

    # A multiple at or below 1.0 proves nothing and must be refused outright.
    for bad_multiple in (1.0, 0.5, 0.0):
        try:
            clamp_demo_targets(0.0, 160.0, bad_multiple)
        except ValueError as exc:
            assert "EXCEED" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"accepted multiple={bad_multiple}")

    for bad_clamp in (0.0, -1.0, float("inf")):
        try:
            clamp_demo_targets(0.0, bad_clamp, 1.5)
        except ValueError as exc:
            assert "positive finite" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"accepted clamp={bad_clamp}")


def test_interpolate_steps_lands_exactly_on_target_without_exceeding_max_step():
    """Slow motion has to come from small increments, not from a servo register.

    ``configure_motors`` sets Maximum_Acceleration/Acceleration to 254 — the
    servo's maximum — so the only lever on speed is the size of each commanded
    increment. The final step must land EXACTLY on the target: an interpolation
    that stops short would leave the arm somewhere the measurement did not name.
    """
    present = [0.0, -102.0, 96.0, 76.0, -90.0, 0.0]
    target = [0.0, 0.0, 0.0, 50.0, -90.0, 60.0]
    steps = interpolate_steps(present, target, 4.0)

    assert steps[-1] == target
    assert len(steps) == 26  # ceil(102 / 4)

    previous = present
    for step in steps:
        assert max(abs(a - b) for a, b in zip(step, previous)) <= 4.0 + 1e-9
        previous = step

    # An already-satisfied move is one no-op step, never zero steps.
    assert interpolate_steps(target, target, 4.0) == [target]

    for bad in (0.0, -1.0):
        try:
            interpolate_steps(present, target, bad)
        except ValueError as exc:
            assert "positive" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"accepted max_step={bad}")


def test_measured_deg_per_pct_recovers_the_derived_table_from_one_tick():
    """The ratio of the two readings of ONE tick IS the per-joint scale factor.

    This is the arithmetic behind the live measurement that discharged research
    assumption A2: reading the same register once as degrees and once as percent
    yields both sides of the seam from a single physical pose, and their ratio is
    the scale. Here both sides are computed from a synthetic tick, which proves
    the method; the live run supplied the hardware numbers.
    """
    table = deg_per_pct_table()
    for joint in ARM_JOINTS:
        cal = CALIBRATION_TICK_RANGES[joint]
        low, high = float(cal["range_min"]), float(cal["range_max"])
        for tick in (low, low + 100.0, high):
            degrees = normalize_degrees(tick, low, high)
            percent = normalize_m100_100(tick, low, high, cal["drive_mode"])
            measured = measured_deg_per_pct(degrees, percent)
            assert measured is not None
            assert abs(measured - table[joint]) < MEASURED_SCALE_TOL

    # At the calibrated midpoint both readings are zero and the ratio is
    # meaningless. It must report "not measurable", never a fabricated number.
    cal = CALIBRATION_TICK_RANGES["wrist_roll"]
    midpoint = (cal["range_min"] + cal["range_max"]) / 2
    assert measured_deg_per_pct(
        normalize_degrees(midpoint, cal["range_min"], cal["range_max"]), 0.0
    ) is None


def test_envelope_row_discriminates_at_initial_and_does_not_at_ready():
    """The row-level restatement of the ready-pose silent-pass hazard.

    ``initial``'s ``shoulder_lift`` is the one joint whose envelope membership
    differs between the conventions; every ``ready`` joint reads the same under
    both, so no ``ready`` row is admissible as the verdict's evidence.
    """
    initial_deg = DUME_POSES["initial"][:5]
    initial_pct = degrees_to_percent(initial_deg)
    rows = [
        envelope_row(joint, initial_deg[index], initial_pct[index])
        for index, joint in enumerate(ARM_JOINTS)
    ]
    discriminating = [row for row in rows if row["discriminates"]]
    assert [row["joint"] for row in discriminating] == ["shoulder_lift"]
    assert discriminating[0]["percent_inside"] is True
    assert discriminating[0]["degrees_inside"] is False
    assert round(discriminating[0]["as_degrees"], 2) == -102.0
    assert round(discriminating[0]["as_percent"], 2) == -98.83

    ready_deg = DUME_POSES["ready"][:5]
    ready_pct = degrees_to_percent(ready_deg)
    for index, joint in enumerate(ARM_JOINTS):
        row = envelope_row(joint, ready_deg[index], ready_pct[index])
        assert row["discriminates"] is False
        assert row["degrees_inside"] and row["percent_inside"]

    # The gripper has no arm-envelope row: its stats live in a separate group.
    try:
        envelope_row("gripper", 0.0, 0.0)
    except ValueError as exc:
        assert "arm joints" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("envelope_row accepted the gripper")


def test_compare_pre_and_post_upgrade_is_exact_identity_on_synthetic_ticks():
    """The before/after comparison must be an EXACT identity, not merely <0.5.

    Roadmap criterion 2 budgets 0.5 per joint, but the two versions' ``_normalize``
    bodies are byte-identical, so the honest expectation is bit-for-bit equality.
    Asserting the identity here is what makes a nonzero live difference readable
    as a calibration problem rather than as an upgrade artefact.
    """
    calibration = {
        joint: dict(cal) for joint, cal in CALIBRATION_TICK_RANGES.items()
    }
    raw = {joint: cal["range_min"] + 137 for joint, cal in CALIBRATION_TICK_RANGES.items()}

    for use_degrees in (True, False):
        reported = {
            joint: normalize_v061(
                tick,
                calibration[joint]["range_min"],
                calibration[joint]["range_max"],
                calibration[joint]["drive_mode"],
                norm_mode_for(joint, use_degrees),
            )
            for joint, tick in raw.items()
        }
        result = compare_pre_and_post_upgrade(raw, reported, calibration, use_degrees)
        assert set(result) == set(JOINT_NAMES)
        for joint, row in result.items():
            assert row["formula_delta"] == 0.0
            assert row["diff"] == 0.0
            assert row["pre_upgrade_v033"] == row["post_upgrade_v061"]
            assert row["norm_mode"] == norm_mode_for(joint, use_degrees)


def test_calibration_bounds_reads_a_dataclass_and_a_plain_dict_alike():
    """Live code hands over MotorCalibration dataclasses; the tests hand over dicts.

    Both shapes must work, so the same derivation functions the hardware run used
    are the ones CI exercises — not a parallel reimplementation.
    """
    class _Entry:
        range_min, range_max, drive_mode = 898, 3090, 0

    from_dataclass = calibration_bounds({"elbow_flex": _Entry()}, "elbow_flex")
    from_dict = calibration_bounds(
        {"elbow_flex": {"range_min": 898, "range_max": 3090, "drive_mode": 0}}, "elbow_flex"
    )
    assert from_dataclass == from_dict == (898.0, 3090.0, 0)

    # A dict without drive_mode defaults to 0 rather than raising: every motor on
    # this arm has drive_mode 0, which is why both conventions share a midpoint.
    assert calibration_bounds(
        {"gripper": {"range_min": 2045, "range_max": 3486}}, "gripper"
    ) == (2045.0, 3486.0, 0)


@pytest.fixture
def derivation_inputs(tmp_path):
    import json
    calibration = tmp_path / 'calibration.json'
    statistics = tmp_path / 'statistics.json'
    calibration.write_text(json.dumps(CALIBRATION_TICK_RANGES))
    statistics.write_text(json.dumps({'new_embodiment': {
        'state': CHECKPOINT_STATE_STATS, 'action': CHECKPOINT_ACTION_STATS}}))
    return calibration, statistics


def test_derivation_requires_local_inputs(derivation_inputs):
    calibration, statistics = derivation_inputs
    calibration.unlink()
    ok, notes = probe.check_pinned_constants_against_local_artifacts(
        str(statistics), str(calibration))
    assert not ok, 'missing calibration must not satisfy the Phase 7 prerequisite'


def test_derivation_current_snapshot_and_refuses_one_tick_drift(derivation_inputs):
    import json
    calibration, statistics = derivation_inputs
    assert hasattr(probe, 'derive_current_calibration'), 'offline derivation API is required'
    record = probe.derive_current_calibration(calibration, statistics)
    assert record['status'] == 'passed'
    assert record['kind'] == 'offline_arithmetic'
    assert record['calibration']['sha256']
    assert record['reset_targets']['initial']['reachable'] is True
    assert record['scale_deg_per_pct']['wrist_roll'] == 1.8
    changed = json.loads(calibration.read_text())
    changed['elbow_flex']['range_max'] += 1
    calibration.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='DRIFT'):
        probe.derive_current_calibration(calibration, statistics)


@pytest.mark.parametrize('flags', [[], ['--skip-hardware', '--demo-clamp'],
                                  ['--skip-hardware', '--pose-sequence', 'initial']])
def test_derivation_cli_refuses_motion_modes(flags, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['probe', '--write-derivation', str(tmp_path/'out.json'), *flags])
    with pytest.raises(SystemExit) as exc:
        probe.main()
    assert exc.value.code == 2


def test_current_resolver_cannot_fall_back_to_stale_robot_copy(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv('HF_LEROBOT_CALIBRATION', str(tmp_path))
    stale = tmp_path/'robots/so101_follower'/f'{probe.DEFAULT_ROBOT_ID}.json'
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps(CALIBRATION_TICK_RANGES))
    current = probe.resolve_calibration_path()
    assert current == tmp_path/'robots/so_follower'/f'{probe.DEFAULT_ROBOT_ID}.json'
    assert not current.exists()
    assert not probe.check_pinned_constants_against_local_artifacts(
        calibration_path=str(current))[0]


def test_derivation_missing_statistics_is_not_run(derivation_inputs):
    calibration, statistics = derivation_inputs
    statistics.unlink()
    with pytest.raises(FileNotFoundError):
        probe.derive_current_calibration(calibration, statistics)
    assert not probe.check_pinned_constants_against_local_artifacts(
        str(statistics), str(calibration))[0]


def test_derivation_publication_is_immutable_and_hashes_exact_inputs(derivation_inputs, tmp_path):
    import hashlib
    import json
    calibration, statistics = derivation_inputs
    record = probe.derive_current_calibration(calibration, statistics)
    assert record['calibration']['sha256'] == hashlib.sha256(calibration.read_bytes()).hexdigest()
    assert record['statistics']['sha256'] == hashlib.sha256(statistics.read_bytes()).hexdigest()
    assert record['at_limit_predictions']['elbow_flex']['tick'] == 3100
    assert record['at_limit_predictions']['elbow_flex']['degrees'] == pytest.approx(96.57142857142857)
    assert not record['wrist_roll_cross_check']['percent_inside_band']
    target = tmp_path/'evidence/calibration.json'
    probe.write_derivation(target, record)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        probe.write_derivation(target, {**record, 'status': 'failed'})
    assert target.read_bytes() == before
    assert json.loads(before)['kind'] == 'offline_arithmetic'


def test_derivation_cli_rejects_stale_explicit_copy(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['probe', '--skip-hardware', '--calibration',
        str(tmp_path/'stale.json'), '--write-derivation', str(tmp_path/'evidence.json')])
    with pytest.raises(SystemExit) as exc:
        probe.main()
    assert exc.value.code == 2
