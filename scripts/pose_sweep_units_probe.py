#!/usr/bin/env python3
"""Arm-free normalization-units discriminators for PAR-04 / PAR-06.

Settles offline which ``MotorNormMode`` the GR00T checkpoint was trained in, and
derives the real per-joint convention seam. The verdict is ``RANGE_M100_100``
(equivalently ``use_degrees=False``); ``docs/UNITS-VERDICT.md`` is the standing
resolution this harness defends, and ``tests/test_units_verdict.py`` is the
always-running gate over the same numbers.

Four arm-free discriminators, in order of strength:

  1. ``clip_fingerprint_count()`` — the plus-or-minus 100.0 clip signature in the
     checkpoint statistics. The PRIMARY written verdict: it assumes nothing about
     which calibration was in use at training time.
  2. ``degrees_reachable_range()`` — the ``elbow_flex`` falsification: the joint's
     physical span is plus-or-minus 96.35 degrees, but the checkpoint records
     100.0, which the DEGREES convention cannot produce.
  3. ``deg_per_pct_table()`` — the per-joint degrees-to-percent scale, plus the
     ``wrist_roll`` cross-check against the training dataset's recorded band.
  4. ``envelope_contains()`` — run against the INITIAL pose. Never the ready
     pose: the ready pose is inside the envelope under BOTH conventions, so
     PAR-04's literal assertion decides nothing. The harness prints that as an
     explicit non-discrimination note rather than citing it as evidence.

The hardware half (``units_verdict()`` / ``active_mode_from_raw_tick()``,
``live_pose_sweep()``, ``compare_pre_and_post_upgrade()``, ``demo_clamp()``) needs
the arm: ``--skip-hardware`` opens no serial port and runs the four arm-free
discriminators only. Plan 05-06 ran the hardware half behind the hardware-attach
gate, which is what converted the per-joint scale magnitudes from exact
arithmetic on verified inputs into an actual measurement. See the
``## Live confirmation`` section of ``docs/UNITS-VERDICT.md`` for the values.

TWO CONVENTIONS ARE MEASURED, DELIBERATELY. The bus reports whatever
``use_degrees`` selects, and Dum-E runs ``use_degrees=True`` (DEGREES) while the
CHECKPOINT was trained in RANGE_M100_100. Those are answers to two different
questions, so the sweep probes the bus twice — once at the running configuration
and once at ``use_degrees=False`` — and decides the CHECKPOINT question from the
envelope discrimination at the ``initial`` pose plus the ``elbow_flex``
at-mechanical-limit fingerprint. Motion is only ever commanded in the running
DEGREES configuration, because the four hardcoded pose vectors are degrees.

Usage:
    uv run python scripts/pose_sweep_units_probe.py --skip-hardware
    uv run python scripts/pose_sweep_units_probe.py --port /dev/ttyACM0
    uv run python scripts/pose_sweep_units_probe.py --pose-sequence initial,ready,remote
    uv run python scripts/pose_sweep_units_probe.py --demo-clamp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Pinned reference constants ---------------------------------------------

# Provenance: lerobot 0.6.1 wheel motors/motors_bus.py `_normalize` —
# `max_res = self.model_resolution_table[...] - 1` and sts3215 resolves to 4096,
# so the divisor is 4095, NOT 4096. Deriving this off by one is the documented trap.
MAX_RES = 4095

# Provenance: RESEARCH `units_verdict()` reference implementation — a candidate
# normalization matches the bus-reported value within 1e-3.
MODE_MATCH_TOL = 1e-3

# Provenance: RESEARCH clip-fingerprint reference implementation — a recorded
# bound counts as saturated when |value| equals 100.0 within 1e-9.
CLIP_FINGERPRINT_TOL = 1e-9

# Provenance: lerobot so_follower.py motor order; index 5 (gripper) is the
# non-arm joint every arm-joint table excludes.
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Provenance: lerobot 0.6.1 wheel so_follower.py:59 —
# `Motor(6, "sts3215", MotorNormMode.RANGE_0_100)` is hardcoded for the gripper
# and is NOT gated on `use_degrees`.
GRIPPER_NORM_MODE = "RANGE_0_100"

# Provenance: checkpoints/GR00T-N1.7-3B-SO101/statistics.json, group
# `new_embodiment`, key `state`. Full stored precision (float32 widened).
CHECKPOINT_STATE_STATS = {
    "single_arm": {
        "min": [
            -84.64447784423828,
            -100.0,
            -93.23892211914062,
            -5.65541410446167,
            -99.68569946289062,
        ],
        "max": [
            61.10689926147461,
            69.19126892089844,
            100.0,
            100.0,
            -5.447878360748291,
        ],
        "mean": [
            -10.548956871032715,
            -7.485757827758789,
            17.880617141723633,
            60.12580490112305,
            -56.27016067504883,
        ],
        "std": [
            24.712890625,
            40.22310256958008,
            39.03307342529297,
            20.067169189453125,
            16.543886184692383,
        ],
        "q01": [
            -63.76702117919922,
            -99.7432632446289,
            -53.390987396240234,
            12.669881820678711,
            -99.63302612304688,
        ],
        "q99": [
            44.04852294921875,
            54.04542049407958,
            100.0,
            98.61120529174802,
            -14.248297691345215,
        ],
    },
    "gripper": {
        "min": [0.0],
        "max": [57.952972412109375],
        "mean": [18.37102699279785],
        "std": [13.891586303710938],
        "q01": [0.27397260069847107],
        "q99": [47.7869987487793],
    },
}

# Provenance: same file, key `action`.
CHECKPOINT_ACTION_STATS = {
    "single_arm": {
        "min": [
            -85.03553771972656,
            -100.0,
            -95.62642669677734,
            -7.267951011657715,
            -100.0,
        ],
        "max": [
            62.09495162963867,
            68.39863586425781,
            100.0,
            100.0,
            -4.973822116851807,
        ],
        "mean": [
            -10.727166175842285,
            -8.480545997619629,
            14.531797409057617,
            59.80523681640625,
            -56.273399353027344,
        ],
        "std": [
            24.671228408813477,
            38.306697845458984,
            38.723289489746094,
            20.474946975708008,
            16.538484573364258,
        ],
        "q01": [
            -64.0104751586914,
            -99.65928649902344,
            -55.812877044677734,
            11.052861213684082,
            -99.895263671875,
        ],
        "q99": [
            43.634110794067375,
            51.700679779052734,
            100.0,
            99.12434387207031,
            -13.956533432006836,
        ],
    },
    "gripper": {
        "min": [0.0],
        "max": [58.36802673339844],
        "mean": [16.96166229248047],
        "std": [14.029448509216309],
        "q01": [0.1663893461227417],
        "q99": [47.87677001953125],
    },
}

# Provenance: the LeRobot follower calibration JSON under the
# HF_LEROBOT_CALIBRATION root — `robots/<robot class name>/<robot_id>.json`, i.e.
# the file `resolve_calibration_path()` derives, as it stood when this verdict was
# reasoned from it. Raw encoder ticks. Every motor has drive_mode 0, which is why
# both candidate normalization modes share the same midpoint and the delta carries
# no sign flip.
#
# These are a SNAPSHOT, not a live read. `check_pinned_constants_against_local_
# artifacts()` compares them against the file on disk and FAILS on any difference,
# because a recalibration invalidates the derived scale table rather than merely
# shifting it — the verdict has to be re-derived, not re-asserted.
CALIBRATION_TICK_RANGES = {
    "shoulder_pan": {"id": 1, "drive_mode": 0, "range_min": 792, "range_max": 3443},
    "shoulder_lift": {"id": 2, "drive_mode": 0, "range_min": 851, "range_max": 3211},
    "elbow_flex": {"id": 3, "drive_mode": 0, "range_min": 898, "range_max": 3090},
    "wrist_flex": {"id": 4, "drive_mode": 0, "range_min": 926, "range_max": 3219},
    "wrist_roll": {"id": 5, "drive_mode": 0, "range_min": 148, "range_max": 3965},
    "gripper": {"id": 6, "drive_mode": 0, "range_min": 2045, "range_max": 3486},
}

# Provenance: aaronsu11/so101_fruit meta/episodes_stats.jsonl, first line
# (episode 0, count 261), `observation.state`, rounded to 3dp as recorded in the
# phase research. Six entries (five arm joints + gripper). NOT locally
# re-verifiable offline: the dataset is remote, so the drift check below cannot
# cover this constant.
DATASET_EPISODE0_STATE_STATS = {
    "min": [-29.955, -99.744, -15.43, 34.989, -58.952, 0.274],
    "max": [10.136, 35.918, 99.727, 79.521, -50.564, 34.247],
    "mean": [-7.891, -8.014, 17.205, 57.977, -56.201, 17.037],
    "std": [15.182, 49.319, 42.02, 15.405, 2.885, 11.968],
}

# Provenance: embodiment/so_arm10x/controller.py:357, 363, 369 and the
# release_at_remote_pose sequence at :385-388. The comment above the first
# vector states "These target degrees mirror legacy behavior", i.e. the literals
# encode the unit convention implicitly. `release_lift` is the sequence's
# distinct arm vector; its shoulder_pan is a runtime random offset (pinned 0.0
# here) and its gripper is read from the live arm (pinned 0.0 here).
DUME_POSES = {
    "initial": [0.0, -102.0, 96.0, 76.0, -90.0, 0.0],
    "ready": [0.0, -90.0, 75.0, 75.0, -90.0, 0.0],
    "remote": [0.0, 0.0, 0.0, 50.0, -90.0, 60.0],
    "release_lift": [0.0, 45.0, -45.0, 50.0, -90.0, 0.0],
}

# Provenance: (range_max - range_min) * 360 / (MAX_RES * 200) per arm joint,
# computed in float64 from CALIBRATION_TICK_RANGES and rounded to 5dp.
DEG_PER_PCT_PINNED = [1.16527, 1.03736, 0.96352, 1.00791, 1.6778]

ARM_JOINTS = JOINT_NAMES[:5]

# Default LeRobot robot identity. Both are overridable on the command line;
# neither is ever an absolute path.
#
# DEFAULT_ROBOT_TYPE is Dum-E's OWN robot_type string, used to construct the
# controller. It is NOT the directory the calibration lives under — see
# LEROBOT_ROBOT_CLASS_NAME.
DEFAULT_ROBOT_TYPE = "so101_follower"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"

# The robot CLASS's `name`, which is the segment lerobot derives the calibration
# directory from. 0.6.1 consolidated both follower classes into `SOFollower`,
# whose `name` is "so_follower", so the directory moved even though `robot_type`
# did not. Using DEFAULT_ROBOT_TYPE here reads the stale pre-0.6.x copy that the
# bus no longer loads, which is a drift check that cannot fail. Pass
# `--calibration` to override the derivation entirely.
LEROBOT_ROBOT_CLASS_NAME = "so_follower"

# Repo-relative location of the (gitignored) checkpoint statistics.
STATISTICS_RELPATH = Path("checkpoints") / "GR00T-N1.7-3B-SO101" / "statistics.json"

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Live-measurement constants (the hardware half) -------------------------

# Provenance: roadmap criterion 2 — "the same joint vector to under 0.5 degrees
# per joint before and after the upgrade". Because the 0.3.3 and 0.6.1 formulas
# are byte-identical, the honest expectation is EXACT equality; 0.5 is the
# criterion's own budget, kept as the assertion so the recorded number is
# comparable to what the roadmap asked for.
BEFORE_AFTER_TOL = 0.5

# A measured per-joint scale counts as agreeing with the offline derived table
# when it matches to 5 decimal places — the precision the derived table is
# recorded at (DEG_PER_PCT_PINNED). Deliberately NOT loosened: the derived value
# is exact arithmetic, so a real disagreement would be gross, not marginal.
MEASURED_SCALE_TOL = 5e-6

# Motion shaping for parking the arm. Upstream `configure_motors` sets
# Maximum_Acceleration/Acceleration to 254 (as fast as the servo goes), so slow
# motion has to come from small commanded increments rather than from a register:
# each interpolation step moves at most this many units, and every step still
# passes through the SAFE-02 clamp.
PARK_MAX_STEP = 4.0
PARK_STEP_DELAY_S = 0.04
PARK_SETTLE_S = 1.5

# Clamp demonstration. `wrist_roll` is the ONLY joint with more than
# 2 x DEFAULT_MAX_RELATIVE_TARGET of calibrated travel (+/-167.78 deg against
# +/-116.53 for the next widest), so it is the only joint on which a delta above
# the clamp can be requested without the UNCLAMPED target lying past a mechanical
# stop. That property is what makes the demonstration safe: even a clamp that
# failed to engage would command a physically reachable pose. 1.5x is a modest
# multiple — the point is to cross the threshold, not to stress the arm.
CLAMP_DEMO_JOINT = "wrist_roll"
CLAMP_DEMO_MULTIPLE = 1.5

# The pose the clamp demonstration is commanded FROM, and the pose the arm is
# parked at before every disconnect: `initial`, i.e. the target of
# `SO10xArmController.move_to_initial_pose()`. Operator-designated as this arm's
# safe testing pose, and it is also the right choice for the disconnect because
# `disable_torque_on_disconnect` defaults to True upstream, so the arm goes limp
# when the probe exits and `initial` is a folded pose it can be left in safely.
CLAMP_DEMO_START_POSE = "initial"
REST_POSE = "initial"

# Raw numeric output goes here (gitignored), never into a tracked repo path.
CORPUS_DIRNAME = "corpus"

# Retry count for every live bus read. Upstream defaults `num_retry=0`, which
# turns one dropped Feetech packet into an apparent value mismatch — and a value
# mismatch is precisely what this probe measures.
LIVE_READ_RETRIES = 3


# --- Normalization branches, exactly as upstream defines them ----------------


def norm_mode_for(joint: str, use_degrees: bool) -> str:
    """The MotorNormMode a joint runs in, given the ``use_degrees`` setting.

    The gripper is hardcoded ``RANGE_0_100`` upstream and is NOT gated on
    ``use_degrees`` (so_follower.py:59), which is why its value needs no
    conversion in either direction. The five arm joints do switch.
    """
    if joint not in CALIBRATION_TICK_RANGES:
        raise ValueError(f"Unknown joint: {joint!r}; expected one of {JOINT_NAMES}")
    if joint == "gripper":
        return GRIPPER_NORM_MODE
    return "DEGREES" if use_degrees else "RANGE_M100_100"


def normalize_m100_100(tick: float, lo: float, hi: float, drive_mode: int = 0) -> float:
    """RANGE_M100_100: the CLAMPED tick mapped onto a 200-wide span, minus 100.

    The clamp is what produces exactly plus-or-minus 100.0 at the calibrated
    endpoints — the clip fingerprint the units verdict rests on.
    """
    if hi == lo:
        raise ValueError(f"Invalid calibration: range_min == range_max == {lo}")
    bounded_val = min(hi, max(lo, tick))
    norm = (((bounded_val - lo) / (hi - lo)) * 200) - 100
    return -norm if drive_mode else norm


def normalize_0_100(tick: float, lo: float, hi: float, drive_mode: int = 0) -> float:
    """RANGE_0_100: the CLAMPED tick scaled onto 0..100. The gripper's mode."""
    if hi == lo:
        raise ValueError(f"Invalid calibration: range_min == range_max == {lo}")
    bounded_val = min(hi, max(lo, tick))
    norm = ((bounded_val - lo) / (hi - lo)) * 100
    return 100 - norm if drive_mode else norm


def normalize_degrees(tick: float, lo: float, hi: float) -> float:
    """DEGREES: (tick - midpoint) * 360 / MAX_RES, with NO clamp.

    The absence of a clamp is implemented faithfully rather than defensively:
    it is precisely why this branch has no plus-or-minus 100 boundary and
    therefore cannot produce the checkpoint's saturated bounds.
    """
    mid = (lo + hi) / 2
    return (tick - mid) * 360 / MAX_RES


def _normalize_dispatch(
    tick: float, lo: float, hi: float, drive_mode: int, norm_mode: str
) -> float:
    """The `_normalize` body's three-branch dispatch, transcribed literally."""
    if hi == lo:
        raise ValueError(f"Invalid calibration: range_min == range_max == {lo}")
    bounded_val = min(hi, max(lo, tick))
    if norm_mode == "RANGE_M100_100":
        norm = (((bounded_val - lo) / (hi - lo)) * 200) - 100
        return -norm if drive_mode else norm
    if norm_mode == "RANGE_0_100":
        norm = ((bounded_val - lo) / (hi - lo)) * 100
        return 100 - norm if drive_mode else norm
    if norm_mode == "DEGREES":
        mid = (lo + hi) / 2
        max_res = MAX_RES
        return (tick - mid) * 360 / max_res
    raise NotImplementedError(f"Unknown MotorNormMode: {norm_mode!r}")


def normalize_v033(
    tick: float, lo: float, hi: float, drive_mode: int, norm_mode: str
) -> float:
    """Normalization as lerobot 0.3.3 defines it.

    Provenance: lerobot 0.3.3, motors/motors_bus.py `MotorsBus._normalize`.
    """
    return _normalize_dispatch(tick, lo, hi, drive_mode, norm_mode)


def normalize_v061(
    tick: float, lo: float, hi: float, drive_mode: int, norm_mode: str
) -> float:
    """Normalization as lerobot 0.6.1 defines it.

    Provenance: lerobot 0.6.1 wheel, motors/motors_bus.py `MotorsBus._normalize`.
    The body is byte-identical to 0.3.3's — that byte-identity is the CITED
    evidence that the upgrade delta is identity, and the swept-tick test is the
    mechanical re-check of it.
    """
    return _normalize_dispatch(tick, lo, hi, drive_mode, norm_mode)


# --- Discriminator 1: the +/-100.0 clip fingerprint (primary) ----------------


def clip_fingerprint_count() -> int:
    """Count (joint, bound) pairs whose magnitude is exactly 100.0.

    Scans the checkpoint's ``state`` and ``action`` ``single_arm`` min/max
    arrays. RANGE_M100_100 produces exactly plus-or-minus 100.0 at a clamped
    endpoint by construction; the unclamped DEGREES branch has no such boundary.
    """
    count = 0
    for group in (CHECKPOINT_STATE_STATS, CHECKPOINT_ACTION_STATS):
        for bound in ("min", "max"):
            for value in group["single_arm"][bound]:
                if abs(abs(value) - 100.0) < CLIP_FINGERPRINT_TOL:
                    count += 1
    return count


def saturated_bounds() -> list[tuple[str, str, str, float]]:
    """The saturated (group, bound, joint, value) tuples behind the fingerprint."""
    found = []
    for group_name, group in (
        ("state", CHECKPOINT_STATE_STATS),
        ("action", CHECKPOINT_ACTION_STATS),
    ):
        for bound in ("min", "max"):
            for index, value in enumerate(group["single_arm"][bound]):
                if abs(abs(value) - 100.0) < CLIP_FINGERPRINT_TOL:
                    found.append((group_name, bound, JOINT_NAMES[index], value))
    return found


# --- Discriminator 2: the elbow_flex degrees falsification -------------------


def degrees_reachable_range(joint: str) -> tuple[float, float]:
    """The (min, max) degrees this joint can physically reach under DEGREES.

    Computed from the joint's calibrated tick span. ``elbow_flex`` reaches only
    plus-or-minus 96.35 degrees, yet the checkpoint records 100.0 — unreachable
    under the DEGREES convention, ordinary under RANGE_M100_100.
    """
    if joint not in CALIBRATION_TICK_RANGES:
        raise ValueError(f"Unknown joint: {joint!r}; expected one of {JOINT_NAMES}")
    cal = CALIBRATION_TICK_RANGES[joint]
    lo, hi = cal["range_min"], cal["range_max"]
    return (normalize_degrees(lo, lo, hi), normalize_degrees(hi, lo, hi))


# --- Discriminator 3: the per-joint scale and the wrist_roll cross-check -----


def deg_per_pct_table() -> dict[str, float]:
    """Degrees per percent, per ARM joint: (span * 360) / (MAX_RES * 200).

    Both formulas are affine in the raw tick and — because every motor has drive
    mode 0 — share the same midpoint, so the seam between them is a PURE
    multiplicative scale: no sign flip, no additive offset. The gripper is
    excluded: it is RANGE_0_100 under both settings, so it has no such scale.
    """
    table: dict[str, float] = {}
    for joint in ARM_JOINTS:
        cal = CALIBRATION_TICK_RANGES[joint]
        span = cal["range_max"] - cal["range_min"]
        table[joint] = (span * 360) / (MAX_RES * 200)
    return table


def pct_per_deg_table() -> dict[str, float]:
    """The inverse of :func:`deg_per_pct_table` — percent per degree, per joint."""
    return {joint: 1.0 / factor for joint, factor in deg_per_pct_table().items()}


def degrees_to_percent(vector_deg: list[float]) -> list[float]:
    """Convert a 5-joint degrees vector into the RANGE_M100_100 convention.

    DERIVED BY ARITHMETIC, NOT MEASURED. Exact given the verified formulas and
    the verified calibration file, but unconfirmed on hardware — a hypothesis for
    the raw-tick probe (plan 05-06) to confirm. It also depends on the current
    calibration file, so recalibration changes it.
    """
    if len(vector_deg) != 5:
        raise ValueError(
            f"Expected a 5-joint arm vector (gripper excluded), got {len(vector_deg)}"
        )
    table = deg_per_pct_table()
    return [value / table[joint] for value, joint in zip(vector_deg, ARM_JOINTS)]


def percent_to_degrees(vector_pct: list[float]) -> list[float]:
    """Convert a 5-joint RANGE_M100_100 vector into degrees. Same caveat as above."""
    if len(vector_pct) != 5:
        raise ValueError(
            f"Expected a 5-joint arm vector (gripper excluded), got {len(vector_pct)}"
        )
    table = deg_per_pct_table()
    return [value * table[joint] for value, joint in zip(vector_pct, ARM_JOINTS)]


def wrist_roll_cross_check() -> dict:
    """The wrist_roll cross-check against the training dataset's episode-0 band."""
    index = JOINT_NAMES.index("wrist_roll")
    as_degrees = DUME_POSES["ready"][index]
    as_percent = degrees_to_percent(DUME_POSES["ready"][:5])[index]
    band_low = DATASET_EPISODE0_STATE_STATS["min"][index]
    band_high = DATASET_EPISODE0_STATE_STATS["max"][index]
    mean = DATASET_EPISODE0_STATE_STATS["mean"][index]
    std = DATASET_EPISODE0_STATE_STATS["std"][index]
    return {
        "as_degrees": as_degrees,
        "as_percent": as_percent,
        "band": (band_low, band_high),
        "percent_inside_band": band_low <= as_percent <= band_high,
        "degrees_inside_band": band_low <= as_degrees <= band_high,
        "degrees_sigma_outside": abs(as_degrees - mean) / std,
    }


# --- Discriminator 4: the envelope check, against the INITIAL pose -----------


def envelope_contains(vector: list[float], group: str = "state") -> list[bool]:
    """Per-joint: is each value inside the checkpoint's q01/q99 envelope?

    Run this against the INITIAL pose. Against the ready pose it returns all-True
    under both candidate conventions and is therefore not admissible as the units
    verdict's evidence (PAR-04's silent-pass hazard).
    """
    groups = {"state": CHECKPOINT_STATE_STATS, "action": CHECKPOINT_ACTION_STATS}
    if group not in groups:
        raise ValueError(f"Unknown group: {group!r}; expected one of {sorted(groups)}")
    if len(vector) != 5:
        raise ValueError(
            f"Expected a 5-joint arm vector (gripper excluded), got {len(vector)}"
        )
    stats = groups[group]["single_arm"]
    return [
        bool(stats["q01"][index] <= value <= stats["q99"][index])
        for index, value in enumerate(vector)
    ]


# --- The hardware half: shipped here, RUN in plan 05-06 ----------------------


def active_mode_from_raw_tick(
    tick: float, reported: float, lo: float, hi: float, drive_mode: int = 0
) -> list[str]:
    """Which candidate normalization actually produced ``reported`` for ``tick``.

    Proof rather than inference: recompute all three candidates from the
    calibration file and report which matches the value the bus reported, within
    ``MODE_MATCH_TOL``. Pure arithmetic — unit-testable against synthetic ticks
    with no arm attached.
    """
    candidates = {
        "RANGE_M100_100": normalize_m100_100(tick, lo, hi, drive_mode),
        "RANGE_0_100": normalize_0_100(tick, lo, hi, drive_mode),
        "DEGREES": normalize_degrees(tick, lo, hi),
    }
    return [
        mode
        for mode, value in candidates.items()
        if abs(value - reported) < MODE_MATCH_TOL
    ]


def units_verdict(bus, calibration) -> dict:
    """Read the same register twice — raw then normalized — and report the mode.

    REQUIRES THE ARM. Plan 05-03 ships this unrun; plan 05-06 runs it behind the
    hardware-attach gate. The ``normalize`` keyword is present and keyword-only in
    BOTH lerobot 0.3.3 and 0.6.1, so this identical probe runs on either side of
    the version bump — which is what makes the before/after comparison meaningful.
    """
    raw = bus.sync_read("Present_Position", normalize=False, num_retry=LIVE_READ_RETRIES)
    got = bus.sync_read("Present_Position", normalize=True, num_retry=LIVE_READ_RETRIES)

    verdict = {}
    for motor, tick in raw.items():
        lo, hi, drive_mode = calibration_bounds(calibration, motor)
        matched = active_mode_from_raw_tick(tick, got[motor], lo, hi, drive_mode)
        verdict[motor] = {
            "raw_tick": tick,
            "reported": got[motor],
            "candidates": {
                "RANGE_M100_100": normalize_m100_100(tick, lo, hi, drive_mode),
                "RANGE_0_100": normalize_0_100(tick, lo, hi, drive_mode),
                "DEGREES": normalize_degrees(tick, lo, hi),
            },
            "active_mode": matched[0] if len(matched) == 1 else matched,
            "deg_per_pct": (hi - lo) * 360 / (MAX_RES * 200),
        }
    return verdict


# --- Pure helpers for the live half: no hardware, unit-testable in CI --------


def tick_for_degrees(degrees: float, lo: float, hi: float) -> float:
    """Invert :func:`normalize_degrees` — the raw tick a degree value commands.

    Needed as a PRE-FLIGHT SAFETY CHECK, not as a convenience. Upstream's
    ``_unnormalize`` DEGREES branch is ``int(val * max_res / 360 + mid)`` with no
    clamp of any kind (``motors_bus.py`` :904-907), so a commanded degree value
    outside the joint's calibrated span produces an out-of-range tick and drives
    the servo into a mechanical stop. Every commanded target in this harness is
    checked through here first.
    """
    mid = (lo + hi) / 2
    return mid + degrees * MAX_RES / 360


def within_calibrated_ticks(tick: float, lo: float, hi: float) -> bool:
    """Is ``tick`` inside the joint's calibrated (i.e. physically recorded) span?"""
    return lo <= tick <= hi


def measured_deg_per_pct(reported_degrees: float, computed_percent: float) -> float | None:
    """The per-joint degrees-to-percent factor MEASURED from one raw tick read.

    Reading the same register once with normalization off and once with it on
    gives both sides of the seam from a single physical pose: the bus reports the
    DEGREES value, and the RANGE_M100_100 value is recomputed from the same tick.
    Their ratio IS the scale factor — a measurement, not arithmetic on the
    calibration file, which is what discharges research assumption A2.

    Returns ``None`` when the percent reading is at or near zero (the joint is at
    its calibrated midpoint), where the ratio is numerically meaningless. A
    ``None`` is reported as "not measurable at this pose", never silently skipped:
    the sweep visits several poses precisely so every joint is measurable at one
    of them.
    """
    if abs(computed_percent) < 1e-6:
        return None
    return reported_degrees / computed_percent


def envelope_row(joint: str, degrees_value: float, percent_value: float, group: str = "state") -> dict:
    """Per-joint envelope membership under BOTH candidate conventions.

    ``discriminates`` is the load-bearing field: when it is False the joint's
    reading is inside (or outside) the envelope under both conventions and
    therefore decides nothing — PAR-04's documented silent-pass hazard. Only rows
    where it is True are admissible as evidence of the checkpoint's convention.
    """
    groups = {"state": CHECKPOINT_STATE_STATS, "action": CHECKPOINT_ACTION_STATS}
    if group not in groups:
        raise ValueError(f"Unknown group: {group!r}; expected one of {sorted(groups)}")
    if joint not in ARM_JOINTS:
        raise ValueError(f"Envelope rows cover the arm joints {ARM_JOINTS}; got {joint!r}")
    stats = groups[group]["single_arm"]
    index = ARM_JOINTS.index(joint)
    q01, q99 = stats["q01"][index], stats["q99"][index]
    inside_deg = bool(q01 <= degrees_value <= q99)
    inside_pct = bool(q01 <= percent_value <= q99)
    return {
        "joint": joint,
        "q01": q01,
        "q99": q99,
        "as_degrees": degrees_value,
        "degrees_inside": inside_deg,
        "as_percent": percent_value,
        "percent_inside": inside_pct,
        "discriminates": inside_deg != inside_pct,
    }


def clamp_demo_targets(present: float, clamp: float, multiple: float = CLAMP_DEMO_MULTIPLE) -> tuple[float, float]:
    """``(requested, expected_clipped)`` for the oversized-delta demonstration.

    The requested delta is ``multiple * clamp`` so it is unambiguously above the
    threshold; upstream clips the delta to exactly ``clamp``, so the expected
    clipped target is ``present + clamp`` with the same sign.
    """
    if not (multiple > 1.0):
        raise ValueError(
            f"The demonstration delta must EXCEED the clamp or nothing is proven; "
            f"got multiple={multiple!r}"
        )
    if not (clamp > 0.0) or not math.isfinite(clamp):
        raise ValueError(f"clamp must be a positive finite float; got {clamp!r}")
    return present + clamp * multiple, present + clamp


def interpolate_steps(
    present: list[float], target: list[float], max_step: float = PARK_MAX_STEP
) -> list[list[float]]:
    """Break one large pose change into small increments, ending exactly on target.

    Upstream sets the servos' acceleration registers to their maximum during
    ``configure()``, so "move slowly" cannot be expressed as a register value —
    it has to be expressed as small commanded increments. Every increment still
    passes through the SAFE-02 clamp; this only ensures none of them needs to.
    """
    if len(present) != len(target):
        raise ValueError(f"Vector length mismatch: {len(present)} != {len(target)}")
    if not (max_step > 0.0):
        raise ValueError(f"max_step must be positive; got {max_step!r}")
    span = max((abs(t - p) for p, t in zip(present, target)), default=0.0)
    count = max(1, math.ceil(span / max_step))
    return [
        [p + (t - p) * (index / count) for p, t in zip(present, target)]
        for index in range(1, count + 1)
    ]


def assert_pose_reachable(vector: list[float], calibration) -> None:
    """Refuse a commanded arm vector whose DEGREES target lands past a stop.

    Applies to the five arm joints only: the gripper is ``RANGE_0_100``, whose
    ``_unnormalize`` branch IS clamped upstream, so it cannot be driven out of
    range by an out-of-band value.

    ``calibration`` is REQUIRED and must be the calibration the BUS is using
    (``bus.calibration``), not :data:`CALIBRATION_TICK_RANGES`. The pinned table
    is a snapshot, and where it is wider than the live span — it is on
    ``shoulder_pan``, ``shoulder_lift`` and ``wrist_flex`` after the mid-phase
    recalibration — a guard computing against it would approve a target the bus
    then maps outside the calibrated span. Since the DEGREES ``_unnormalize``
    branch has no upstream clamp, that is the one input whose staleness makes this
    guard wrong in the unsafe direction. There is deliberately no default: an
    omitted calibration must be a ``TypeError``, never a silent fallback to the
    pinned table.
    """
    offenders = []
    for value, joint in zip(vector[:5], ARM_JOINTS):
        lo, hi, _drive_mode = calibration_bounds(calibration, joint)
        tick = tick_for_degrees(value, lo, hi)
        if not within_calibrated_ticks(tick, lo, hi):
            offenders.append(f"{joint}={value:.3f}deg -> tick {tick:.1f} outside [{lo:.0f}, {hi:.0f}]")
    if offenders:
        raise ValueError(
            "Refusing to command a pose that would drive a joint past its "
            f"calibrated span (the DEGREES unnormalize branch has no clamp): "
            f"{'; '.join(offenders)}"
        )


def compare_pre_and_post_upgrade(
    raw_ticks: dict, reported: dict, calibration: dict, use_degrees: bool
) -> dict:
    """Per-joint |pre-upgrade computed value - post-upgrade reported value|.

    The pre-upgrade value is recomputed from the SAME raw ticks with the 0.3.3
    formula. That is exact rather than approximate: plan 05-03 proved the two
    versions' ``_normalize`` bodies byte-identical by sweeping every raw tick
    across each joint's calibrated span, so this is a genuine before-and-after
    comparison taken from one physical pose and one tick read — strictly less
    noisy than two separate live runs, which would add servo read noise and
    re-parking error on top of any real delta. See the corrected-mechanism note
    in ``docs/UNITS-VERDICT.md``.
    """
    result = {}
    for motor, tick in raw_ticks.items():
        lo, hi, drive_mode = calibration_bounds(calibration, motor)
        mode = norm_mode_for(motor, use_degrees)
        pre = normalize_v033(tick, lo, hi, drive_mode, mode)
        post = normalize_v061(tick, lo, hi, drive_mode, mode)
        result[motor] = {
            "raw_tick": tick,
            "norm_mode": mode,
            "pre_upgrade_v033": pre,
            "post_upgrade_v061": post,
            "bus_reported": reported[motor],
            "formula_delta": post - pre,
            "diff": abs(pre - reported[motor]),
        }
    return result


def calibration_bounds(calibration: dict, motor: str) -> tuple[float, float, int]:
    """``(range_min, range_max, drive_mode)`` from a dataclass OR a plain dict.

    Live code hands us ``MotorCalibration`` dataclasses; the hermetic tests hand
    us plain dicts. Both are supported so the same functions are exercised in CI.
    """
    entry = calibration[motor]
    lo = getattr(entry, "range_min", None)
    hi = getattr(entry, "range_max", None)
    drive_mode = getattr(entry, "drive_mode", None)
    if lo is None or hi is None:
        lo, hi = entry["range_min"], entry["range_max"]
        drive_mode = entry.get("drive_mode", 0)
    return float(lo), float(hi), int(drive_mode or 0)


# --- Local data artifacts: resolved, never hardcoded ------------------------


def resolve_calibration_path(
    explicit: str | None = None,
    robot_name: str = LEROBOT_ROBOT_CLASS_NAME,
    robot_id: str = DEFAULT_ROBOT_ID,
) -> Path:
    """Locate the calibration JSON ``lerobot`` ACTUALLY loads. Never hardcoded.

    Delegates to ``controller.resolve_calibration_file()`` rather than rebuilding
    the path here. Two independent resolvers is how this drifted: this one used to
    compose the directory from Dum-E's ``robot_type`` (``"so101_follower"``), while
    the bus reads the directory named after the robot CLASS's ``name``
    (``"so_follower"``) — so the drift check compared the pinned constants against
    the stale pre-0.6.x copy and reported "no drift" while every joint had moved.
    The controller's resolver is the single source of truth for that derivation,
    and it also handles the ``HF_LEROBOT_CALIBRATION`` / ``HF_LEROBOT_HOME`` /
    ``HF_HOME`` search order.

    The import is lazy so the arm-free discriminators still run without paying for
    the LeRobot stack unless a calibration path actually has to be derived.
    """
    if explicit:
        return Path(explicit).expanduser()
    _repo_on_path()
    from embodiment.so_arm10x.controller import resolve_calibration_file

    return resolve_calibration_file(robot_name, robot_id)


def resolve_statistics_path(explicit: str | None = None) -> Path:
    """Locate the checkpoint statistics JSON. Never a hardcoded absolute path.

    Search order: the explicit flag, ``DUME_CHECKPOINT_STATISTICS``, then the
    repo-relative default walked up through the repo root's ancestors — the last
    step is what lets a linked git worktree find the gitignored checkpoint that
    lives in the main checkout.
    """
    if explicit:
        return Path(explicit).expanduser()
    from_env = os.environ.get("DUME_CHECKPOINT_STATISTICS")
    if from_env:
        return Path(from_env).expanduser()
    for base in [REPO_ROOT, *REPO_ROOT.parents]:
        candidate = base / STATISTICS_RELPATH
        if candidate.is_file():
            return candidate
    return REPO_ROOT / STATISTICS_RELPATH


def check_pinned_constants_against_local_artifacts(
    statistics_path: str | None = None,
    calibration_path: str | None = None,
    robot_name: str = LEROBOT_ROBOT_CLASS_NAME,
    robot_id: str = DEFAULT_ROBOT_ID,
) -> tuple[bool, list[str]]:
    """Compare the pinned constants against the real local data artifacts.

    Returns ``(ok, notes)``. Drift in a RESOLVED artifact is a FAILURE: it means
    the local statistics or calibration no longer match the numbers this verdict
    was reasoned from, and the verdict must be re-derived rather than re-asserted.
    An UNRESOLVABLE artifact cannot drift, so it is not a failure — but it is
    reported loudly, naming the path searched, because an unchecked constant is
    exactly what this check exists to surface.
    """
    notes: list[str] = []
    ok = True

    stats_file = resolve_statistics_path(statistics_path)
    if stats_file.is_file():
        recorded = json.loads(stats_file.read_text(encoding="utf-8"))["new_embodiment"]
        for key, pinned in (
            ("state", CHECKPOINT_STATE_STATS),
            ("action", CHECKPOINT_ACTION_STATS),
        ):
            for subgroup, arrays in pinned.items():
                for stat, values in arrays.items():
                    actual = recorded[key][subgroup][stat]
                    if list(actual) != list(values):
                        ok = False
                        notes.append(
                            f"DRIFT {key}.{subgroup}.{stat}: pinned {values} != "
                            f"recorded {actual}"
                        )
        if ok:
            notes.append(f"checkpoint statistics match pinned constants ({stats_file})")
    else:
        notes.append(
            f"UNVERIFIED: checkpoint statistics not resolvable at {stats_file} — "
            f"set --statistics or DUME_CHECKPOINT_STATISTICS to check for drift"
        )

    cal_file = resolve_calibration_path(calibration_path, robot_name, robot_id)
    if cal_file.is_file():
        recorded_cal = json.loads(cal_file.read_text(encoding="utf-8"))
        drifted = False
        for joint, pinned in CALIBRATION_TICK_RANGES.items():
            actual = recorded_cal.get(joint)
            if actual is None:
                ok, drifted = False, True
                notes.append(f"DRIFT calibration: joint {joint} absent from {cal_file}")
                continue
            for field, value in pinned.items():
                if actual.get(field) != value:
                    ok, drifted = False, True
                    notes.append(
                        f"DRIFT calibration {joint}.{field}: pinned {value} != "
                        f"recorded {actual.get(field)}"
                    )
        if not drifted:
            notes.append(f"calibration tick ranges match pinned constants ({cal_file})")
    else:
        notes.append(
            f"UNVERIFIED: calibration not resolvable at {cal_file} — set "
            f"--calibration or HF_LEROBOT_CALIBRATION to check for drift"
        )

    notes.append(
        "UNVERIFIED: DATASET_EPISODE0_STATE_STATS is remote "
        "(aaronsu11/so101_fruit meta/episodes_stats.jsonl) and is not checked offline"
    )
    return ok, notes


# --- The live session: read-only first, then commanded motion ----------------
#
# Ordering is a safety property, not a style choice. Every tick read, the PID
# read-back and the calibration checksum happen BEFORE the first command that
# moves a joint, so a bus or calibration fault surfaces while the arm is
# stationary.


# Dum-E's state-vector key order, matching `SO10xArmController._state_keys`.
STATE_KEYS = [f"{joint}.pos" for joint in JOINT_NAMES]


def _repo_on_path() -> None:
    """Put the repo root on ``sys.path`` so ``embodiment.*`` imports resolve.

    Running this file as a script puts ``scripts/`` on ``sys.path[0]``, not the
    repo root, so the controller import fails without this. Discovered by running
    the hardware half for the first time (plan 05-06).
    """
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def resolve_live_controller_settings(args: argparse.Namespace) -> dict:
    """Serial port, robot identity and camera indices for the live session.

    Resolution order per field: the command-line flag, then the Dum-E YAML config
    (``DUME_CONFIG`` or ``my-dum-e.yaml`` at the repo root), then the value baked
    into this module. ``config.example.yaml`` is deliberately NOT consulted: it is
    a template, and its camera indices are not this host's devices — a template
    value must never silently become a device selection.

    The camera indices matter even though this probe never looks at an image:
    ``SO10xArmController`` always constructs two cameras and ``connect()``
    connects them, so a wrong index fails the whole connect. On the host this was
    first run against, the two real capture nodes are 0 and 2 — indices 1 and 3
    are V4L2 metadata nodes that cannot be opened at all.
    """
    config: dict = {}
    explicit = os.environ.get("DUME_CONFIG")
    candidate = Path(explicit).expanduser() if explicit else REPO_ROOT / "my-dum-e.yaml"
    if candidate.is_file():
        import yaml  # lazy: the arm-free path needs no YAML parser

        loaded = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        block = loaded.get("controller")
        config = block if isinstance(block, dict) else {}

    def pick(flag, key, fallback):
        if flag is not None:
            return flag
        if key in config and config[key] is not None:
            return config[key]
        return fallback

    return {
        "config_file": str(candidate) if config else None,
        "robot_type": pick(args.robot_type, "robot_type", DEFAULT_ROBOT_TYPE),
        "robot_id": pick(args.robot_id, "robot_id", DEFAULT_ROBOT_ID),
        "robot_port": pick(args.port, "robot_port", None),
        "wrist_cam_idx": int(pick(args.wrist_cam_idx, "wrist_cam_idx", 0)),
        "front_cam_idx": int(pick(args.front_cam_idx, "front_cam_idx", 1)),
    }


def prearm_goal_to_present(bus) -> dict:
    """Write ``Goal_Position <- Present_Position`` while torque is still OFF.

    THIS IS A SAFETY PRECONDITION FOR CONNECTING, discovered by reading the live
    registers before touching anything (plan 05-06). Upstream's ``configure()``
    runs inside ``bus.torque_disabled()``, whose exit calls ``enable_torque()``,
    and ``enable_torque()`` writes ``Torque_Enable`` and ``Lock`` only — it does
    NOT synchronise ``Goal_Position`` to the present position
    (``feetech.py`` :302-305). On this arm, with torque off after a power cycle,
    every motor's ``Goal_Position`` register reads **0**: connecting without
    pre-arming would therefore command all six joints to raw tick 0 the instant
    torque came back, a slam of up to 3090 ticks on ``elbow_flex``.

    Pre-arming with torque disabled cannot itself move the arm, and it makes the
    torque-enable a hold rather than a move. The dangerous path is never
    exercised, so this function does not prove the slam would happen — it
    prevents it.

    Returns a record of what was found and written. Skips (and says so) when
    torque is already enabled, where ``Goal_Position`` is live and overwriting it
    would be the very command this exists to avoid.
    """
    bus.connect()
    try:
        torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=LIVE_READ_RETRIES)
        present = bus.sync_read("Present_Position", normalize=False, num_retry=LIVE_READ_RETRIES)
        goal_before = bus.sync_read("Goal_Position", normalize=False, num_retry=LIVE_READ_RETRIES)
        record = {
            "torque_enable_before": dict(torque),
            "present_ticks": dict(present),
            "goal_ticks_before": dict(goal_before),
            "worst_pending_jump_ticks": max(
                (abs(goal_before[m] - present[m]) for m in present), default=0
            ),
        }
        if any(value for value in torque.values()):
            record.update(prearmed=False, reason="torque already enabled — Goal_Position is live")
            return record
        for motor, tick in present.items():
            bus.write("Goal_Position", motor, int(tick), normalize=False, num_retry=LIVE_READ_RETRIES)
        goal_after = bus.sync_read("Goal_Position", normalize=False, num_retry=LIVE_READ_RETRIES)
        mismatched = {
            motor: (present[motor], goal_after[motor])
            for motor in present
            if abs(goal_after[motor] - present[motor]) > 1
        }
        if mismatched:
            raise RuntimeError(
                "Goal_Position pre-arm did not take, so enabling torque would "
                f"command a jump. Refusing to connect. present vs goal: {mismatched}"
            )
        record.update(prearmed=True, reason="written and verified", goal_ticks_after=dict(goal_after))
        return record
    finally:
        # disable_torque=False: leave the torque state EXACTLY as found.
        bus.disconnect(False)


def open_controller(args: argparse.Namespace, *, use_degrees=None):
    """Construct, pre-arm and connect the controller. Returns ``(controller, info)``.

    ``connect()`` performs the calibration-file assertion and the PID read-back,
    so ``info`` carries both as the live evidence LR-04 and D-05 ask for.
    """
    _repo_on_path()
    from embodiment.so_arm10x.controller import SO10xArmController

    settings = resolve_live_controller_settings(args)
    if not settings["robot_port"]:
        raise ValueError(
            "No serial port: pass --port, set SO_ARM_PORT, or name robot_port in "
            "the controller block of my-dum-e.yaml"
        )
    kwargs = dict(
        robot_type=settings["robot_type"],
        robot_port=settings["robot_port"],
        robot_id=settings["robot_id"],
        wrist_cam_idx=settings["wrist_cam_idx"],
        front_cam_idx=settings["front_cam_idx"],
    )
    if use_degrees is not None:
        kwargs["use_degrees"] = use_degrees
    controller = SO10xArmController(**kwargs)

    prearm = prearm_goal_to_present(controller.robot.bus)
    controller.connect()

    calibration_path, calibration_sha256 = controller._assert_calibration_loaded()
    pid_readback = controller._assert_pid_landed()
    info = {
        "settings": settings,
        "prearm": prearm,
        "calibration_path": str(calibration_path),
        "calibration_path_redacted": _redact_home(calibration_path),
        "calibration_sha256": calibration_sha256,
        "pid_readback": pid_readback,
        "use_degrees": bool(getattr(controller.config, "use_degrees", True)),
        "max_relative_target": getattr(controller.config, "max_relative_target", None),
    }
    return controller, info


def _redact_home(path) -> str:
    """``~``-relative form of a path, so no absolute home directory is committed."""
    text = str(path)
    home = str(Path.home())
    return "~" + text[len(home) :] if text.startswith(home) else text


def close_controller(controller, *, keep_torque: bool = False) -> None:
    """Disconnect, optionally leaving torque enabled so the arm holds position."""
    if keep_torque:
        controller.config.disable_torque_on_disconnect = False
    controller.disconnect()


def read_joint_vector(controller) -> list[float]:
    """The 6-joint vector straight off the bus — no camera read, no observation."""
    got = controller.robot.bus.sync_read(
        "Present_Position", normalize=True, num_retry=LIVE_READ_RETRIES
    )
    return [float(got[joint]) for joint in JOINT_NAMES]


def park_slowly(controller, target: list[float], *, label: str = "") -> list[tuple]:
    """Move to ``target`` in small increments. Returns any clamp reports observed.

    Refuses outright if the target is not physically reachable, then interpolates
    so no single commanded step approaches the clamp. Any clamp firing HERE is a
    finding, not an expectation: the demonstration in :func:`demo_clamp` is where
    the clamp is supposed to engage.
    """
    _repo_on_path()
    from embodiment.so_arm10x.controller import diff_clamped_joints

    assert_pose_reachable(target, controller.robot.bus.calibration)
    present = read_joint_vector(controller)
    clamped: list[tuple] = []
    for step_vector in interpolate_steps(present, list(target), PARK_MAX_STEP):
        action = {key: float(value) for key, value in zip(STATE_KEYS, step_vector)}
        sent = controller.set_target_state(action)
        clamped.extend(diff_clamped_joints(action, sent))
        time.sleep(PARK_STEP_DELAY_S)
    time.sleep(PARK_SETTLE_S)
    if clamped and label:
        print(_red(f"       NOTE: the clamp fired while parking at {label!r}: {clamped}"))
    return clamped


def measure_pose(controller, pose_name: str, commanded: list[float] | None) -> dict:
    """Read the same registers twice — raw then normalized — and derive everything.

    Produces, per joint: the raw tick, the value the upgraded bus reports, all
    three recomputed candidates, which candidate actually matched (the active-mode
    PROOF), the pre-upgrade computed value and its difference from the reported
    one, and the measured degrees-to-percent factor. Plus the envelope membership
    of the five arm joints under both candidate conventions.
    """
    bus = controller.robot.bus
    calibration = bus.calibration
    use_degrees = bool(getattr(controller.config, "use_degrees", True))

    raw = bus.sync_read("Present_Position", normalize=False, num_retry=LIVE_READ_RETRIES)
    reported = bus.sync_read("Present_Position", normalize=True, num_retry=LIVE_READ_RETRIES)

    # The two-sided measurement: flip the bus's own norm modes in place and read
    # the SAME physical pose again through the SAME upstream `_normalize`. No
    # disconnect, no torque cycle, no motion between the reads — which is what
    # makes the ratio of the two readings a hardware measurement of the seam
    # rather than arithmetic on the calibration file. Restored in `finally`.
    from lerobot.motors import MotorNormMode

    original_modes = {name: motor.norm_mode for name, motor in bus.motors.items()}
    try:
        for name, motor in bus.motors.items():
            if name != "gripper":
                motor.norm_mode = MotorNormMode.RANGE_M100_100
        reported_percent = bus.sync_read(
            "Present_Position", normalize=True, num_retry=LIVE_READ_RETRIES
        )
    finally:
        for name, motor in bus.motors.items():
            motor.norm_mode = original_modes[name]
    restored = {name: bus.motors[name].norm_mode for name in bus.motors}
    if restored != original_modes:
        raise RuntimeError(
            f"Bus normalization modes were not restored: {restored} != {original_modes}"
        )

    before_after = compare_pre_and_post_upgrade(raw, reported, calibration, use_degrees)

    joints: dict[str, dict] = {}
    for motor, tick in raw.items():
        lo, hi, drive_mode = calibration_bounds(calibration, motor)
        candidates = {
            "RANGE_M100_100": normalize_m100_100(tick, lo, hi, drive_mode),
            "RANGE_0_100": normalize_0_100(tick, lo, hi, drive_mode),
            "DEGREES": normalize_degrees(tick, lo, hi),
        }
        matched = active_mode_from_raw_tick(tick, reported[motor], lo, hi, drive_mode)
        joints[motor] = {
            "raw_tick": tick,
            "reported_running_config": reported[motor],
            "reported_percent_config": reported_percent[motor],
            "candidates": candidates,
            "matched_candidates": matched,
            "active_mode": matched[0] if len(matched) == 1 else matched,
            "at_calibrated_limit": bool(tick <= lo or tick >= hi),
            "derived_deg_per_pct": (hi - lo) * 360 / (MAX_RES * 200),
            "measured_deg_per_pct_one_sided": measured_deg_per_pct(
                reported[motor], candidates["RANGE_M100_100"]
            ),
            "measured_deg_per_pct_two_sided": (
                None
                if motor == "gripper"
                else measured_deg_per_pct(reported[motor], reported_percent[motor])
            ),
            **{
                key: value
                for key, value in before_after[motor].items()
                if key in ("pre_upgrade_v033", "post_upgrade_v061", "formula_delta", "diff")
            },
        }

    envelope = [
        envelope_row(
            joint,
            degrees_value=joints[joint]["reported_running_config"]
            if use_degrees
            else joints[joint]["candidates"]["DEGREES"],
            percent_value=joints[joint]["reported_percent_config"],
        )
        for joint in ARM_JOINTS
    ]

    return {
        "pose": pose_name,
        "commanded_degrees": commanded,
        "use_degrees": use_degrees,
        "joints": joints,
        "before_after_diff": {motor: row["diff"] for motor, row in before_after.items()},
        "envelope_initial_convention_check": envelope,
    }


def live_pose_sweep(args: argparse.Namespace, pose_names: list[str]) -> dict:
    """Park at each fixed pose in turn and take the full measurement at each.

    Read-only first: the connect performed by :func:`open_controller` does the
    calibration assertion and the PID read-back, and the first measurement is
    taken at the pose the arm was already in, all before any commanded motion.
    """
    unknown = [name for name in pose_names if name not in DUME_POSES]
    if unknown:
        raise ValueError(f"Unknown pose(s) {unknown}; known: {sorted(DUME_POSES)}")

    controller, info = open_controller(args)
    poses: list[dict] = []
    parking_clamps: dict[str, list] = {}
    reset_to_initial: dict = {}
    try:
        as_found = measure_pose(controller, "as-found (no motion commanded)", None)
        for name in pose_names:
            target = list(DUME_POSES[name])
            print(f"       parking at {name!r} = {target} ...")
            parking_clamps[name] = [list(entry) for entry in park_slowly(controller, target, label=name)]
            poses.append(measure_pose(controller, name, target))
        reset_to_initial = observe_reset_to_initial(controller)
    finally:
        close_controller(controller)

    return {
        "kind": "pose_sweep",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "connect": info,
        "pid_readback": info["pid_readback"],
        "calibration_path": info["calibration_path_redacted"],
        "calibration_sha256": info["calibration_sha256"],
        "as_found": as_found,
        "poses": poses,
        "parking_clamps": parking_clamps,
        "reset_to_initial": reset_to_initial,
        "reset_to_initial_worst_case_deltas": reset_to_initial_deltas(pose_names),
    }


def reset_to_initial_deltas(pose_names: list[str]) -> dict:
    """Per-joint single-step delta a reset-to-initial would command from each pose.

    Plan 05-05 flagged the reset-to-initial-pose from an extreme policy pose as the
    likeliest LEGITIMATE clamp trigger, with a worst-case ``elbow_flex`` delta near
    190 against a clamp of 160. ``move_to_initial_pose()`` issues ONE unsmoothed
    command, so the delta it presents is the full pose difference — this records
    that difference for each pose actually visited, so Phase 7 can compare a real
    clamp warning against a number rather than against a recollection.
    """
    initial = DUME_POSES["initial"]
    rows = {}
    for name in pose_names:
        pose = DUME_POSES[name]
        rows[name] = {
            joint: abs(initial[index] - pose[index]) for index, joint in enumerate(JOINT_NAMES)
        }
        rows[name]["max"] = max(rows[name].values())
    return rows


def observe_reset_to_initial(controller) -> dict:
    """Call the production ``move_to_initial_pose()`` and record what it did.

    Deliberately the real helper rather than an interpolated park: this is the
    single-command reset the production pick loop performs, so it is the honest
    place to observe whether a nominal reset trips the SAFE-02 clamp. It is also
    the safe pose to leave the arm in before the torque drops at disconnect.

    The clamp signal is read off Dum-E's own loguru stream, which is the surface
    SAFE-02 requires it on — not off a return value the helper discards.
    """
    _repo_on_path()
    from loguru import logger

    from embodiment.so_arm10x.controller import CLAMP_WARNING_TEXT

    before = read_joint_vector(controller)
    target = list(DUME_POSES["initial"])
    assert_pose_reachable(target, controller.robot.bus.calibration)
    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(message.record["message"]), level="WARNING")
    try:
        print("       resetting with the production move_to_initial_pose() ...")
        controller.move_to_initial_pose()
    finally:
        logger.remove(sink_id)
    time.sleep(PARK_SETTLE_S)
    settled = read_joint_vector(controller)
    commanded_delta = {
        joint: target[index] - before[index] for index, joint in enumerate(JOINT_NAMES)
    }
    return {
        "before": before,
        "commanded": target,
        "settled": settled,
        "commanded_delta": commanded_delta,
        "worst_commanded_delta": max(abs(value) for value in commanded_delta.values()),
        "clamp_warnings": [line for line in captured if CLAMP_WARNING_TEXT in line],
        "clamp_fired": any(CLAMP_WARNING_TEXT in line for line in captured),
    }


def demo_clamp(args: argparse.Namespace) -> dict:
    """Command ONE joint a per-step delta above the clamp, and prove it clipped.

    Three independent assertions, two of them from data rather than from a log:
    the returned action differs from the requested one on that joint; the returned
    delta equals the clamp within upstream's own divergence threshold; and the
    warning reached Dum-E's loguru stream. Whether upstream's bridged root-logger
    warning also arrived is recorded either way.

    Safety: the joint is chosen so that even the UNCLAMPED requested target is
    physically reachable (see :data:`CLAMP_DEMO_JOINT`), and both the requested
    and the expected clipped target are checked against the calibrated tick span
    before anything is sent.
    """
    _repo_on_path()
    from loguru import logger

    from embodiment.so_arm10x.controller import (
        CLAMP_DIVERGENCE_THRESHOLD,
        CLAMP_WARNING_TEXT,
        diff_clamped_joints,
    )
    from utils import install_stdlib_to_loguru_bridge

    # The bridge is what carries upstream's own root-logger warning into loguru;
    # installing it here is what lets a single sink see BOTH emitters.
    install_stdlib_to_loguru_bridge()
    captured: list[str] = []
    sink_id = logger.add(lambda message: captured.append(message.record["message"]), level="WARNING")

    controller, info = open_controller(args)
    try:
        clamp = controller.config.max_relative_target
        if not isinstance(clamp, float):
            raise ValueError(
                f"The clamp must be a live float for this demonstration; got {clamp!r}"
            )
        print(f"       parking at {CLAMP_DEMO_START_POSE!r} (the designated safe testing pose) ...")
        park_slowly(controller, list(DUME_POSES[CLAMP_DEMO_START_POSE]), label=CLAMP_DEMO_START_POSE)

        present_vector = read_joint_vector(controller)
        index = JOINT_NAMES.index(CLAMP_DEMO_JOINT)
        present = present_vector[index]
        requested_value, expected_clipped = clamp_demo_targets(present, clamp)

        # The LIVE calibration, never the pinned snapshot: this is the span the
        # bus will map the commanded degrees through, and the DEGREES unnormalize
        # branch has no upstream clamp to catch a target derived from a stale one.
        lo, hi, _drive_mode = calibration_bounds(
            controller.robot.bus.calibration, CLAMP_DEMO_JOINT
        )
        for label, value in (("requested", requested_value), ("clipped", expected_clipped)):
            tick = tick_for_degrees(value, lo, hi)
            if not within_calibrated_ticks(tick, lo, hi):
                raise ValueError(
                    f"Refusing the demonstration: the {label} target {value:.3f} on "
                    f"{CLAMP_DEMO_JOINT} maps to tick {tick:.1f}, outside the calibrated "
                    f"span [{lo:.0f}, {hi:.0f}]. Even an unclamped command must stay "
                    f"physically reachable."
                )

        requested = {key: float(value) for key, value in zip(STATE_KEYS, present_vector)}
        requested[f"{CLAMP_DEMO_JOINT}.pos"] = requested_value
        print(
            f"       commanding {CLAMP_DEMO_JOINT}: present={present:.3f} "
            f"requested={requested_value:.3f} (delta {clamp * CLAMP_DEMO_MULTIPLE:.1f} "
            f"= {CLAMP_DEMO_MULTIPLE}x the clamp of {clamp}) ..."
        )
        captured.clear()
        sent = controller.set_target_state(requested)
        time.sleep(PARK_SETTLE_S)
        settled = read_joint_vector(controller)

        clamped = diff_clamped_joints(requested, sent)
        sent_value = sent[f"{CLAMP_DEMO_JOINT}.pos"]
        dume_warnings = [
            line
            for line in captured
            if CLAMP_WARNING_TEXT in line and "max_relative_target=" in line
        ]
        upstream_warnings = [
            line
            for line in captured
            if CLAMP_WARNING_TEXT in line and "original goal_pos" in line
        ]

        record = {
            "kind": "clamp_demo",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "connect": info,
            "pid_readback": info["pid_readback"],
            "calibration_path": info["calibration_path_redacted"],
            "calibration_sha256": info["calibration_sha256"],
            "clamp": clamp,
            "multiple": CLAMP_DEMO_MULTIPLE,
            "joint": CLAMP_DEMO_JOINT,
            "start_pose": CLAMP_DEMO_START_POSE,
            "present": present,
            "requested": requested_value,
            "expected_clipped": expected_clipped,
            "returned": sent_value,
            "returned_delta": sent_value - present,
            "settled": settled[index],
            "divergence_threshold": CLAMP_DIVERGENCE_THRESHOLD,
            "returned_differs_from_requested": abs(sent_value - requested_value)
            > CLAMP_DIVERGENCE_THRESHOLD,
            "returned_delta_equals_clamp": abs(abs(sent_value - present) - clamp)
            <= CLAMP_DIVERGENCE_THRESHOLD,
            "clamped_joints": [list(entry) for entry in clamped],
            "dume_loguru_warning": dume_warnings[0] if dume_warnings else None,
            "upstream_bridged_warning": upstream_warnings[0] if upstream_warnings else None,
            "all_captured_warnings": list(captured),
        }

        print(f"       returning to the rest pose {REST_POSE!r} ...")
        park_slowly(controller, list(DUME_POSES[REST_POSE]), label=REST_POSE)
        return record
    finally:
        close_controller(controller)
        logger.remove(sink_id)


def write_corpus_results(payload: dict) -> Path:
    """Write raw numeric output under the gitignored corpus directory.

    Never into a tracked repo path: the committed record is the live confirmation
    section of ``docs/UNITS-VERDICT.md``, which carries the values a reader needs.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = REPO_ROOT / CORPUS_DIRNAME / f"pose_sweep_{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "results.json"
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


# --- Numbered-check harness -------------------------------------------------


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def check_clip_fingerprint(index: str) -> bool:
    print(f"\n[{index}] clip fingerprint — the primary, assumption-free verdict ...")
    count = clip_fingerprint_count()
    for group, bound, joint, value in saturated_bounds():
        print(f"       {group}.single_arm.{bound}[{joint}] = {value}")
    if count < 5:
        print(_red(f"  FAIL: only {count} saturated bounds (need >= 5)"))
        return False
    print(
        _green(
            f"  PASS: {count} bounds at exactly +/-100.0 within {CLIP_FINGERPRINT_TOL}"
            f" -> RANGE_M100_100 (the DEGREES branch has no clamp, so no such bound)"
        )
    )
    return True


def check_degrees_falsification(index: str) -> bool:
    print(f"\n[{index}] elbow_flex falsification — DEGREES cannot reach 100.0 ...")
    low, high = degrees_reachable_range("elbow_flex")
    joint_index = JOINT_NAMES.index("elbow_flex")
    recorded = CHECKPOINT_STATE_STATS["single_arm"]["max"][joint_index]
    print(f"       reachable under DEGREES: [{low:.2f}, {high:.2f}] deg")
    print(f"       checkpoint state.single_arm.max[elbow_flex] = {recorded}")
    if recorded <= high:
        print(_red(f"  FAIL: {recorded} is reachable in degrees; no falsification"))
        return False
    print(
        _green(
            f"  PASS: {recorded} exceeds the physical span of +/-{high:.2f} deg — "
            f"unreachable under DEGREES, ordinary under RANGE_M100_100"
        )
    )
    return True


def check_scale_and_cross_check(index: str) -> bool:
    print(f"\n[{index}] per-joint scale + wrist_roll dataset cross-check ...")
    table = deg_per_pct_table()
    inverse = pct_per_deg_table()
    for joint in ARM_JOINTS:
        print(
            f"       {joint:<14} deg_per_pct={table[joint]:.5f} "
            f"pct_per_deg={inverse[joint]:.5f} "
            f"swap_error={(table[joint] - 1.0) * 100:+.2f}%"
        )
    print("       gripper        excluded — RANGE_0_100 under both modes")
    rounded = [round(table[joint], 5) for joint in ARM_JOINTS]
    if rounded != DEG_PER_PCT_PINNED:
        print(_red(f"  FAIL: scale table {rounded} != pinned {DEG_PER_PCT_PINNED}"))
        return False

    cross = wrist_roll_cross_check()
    print(
        f"       wrist_roll: {cross['as_degrees']} deg -> "
        f"{cross['as_percent']:.2f}% ; dataset band {cross['band']}"
    )
    if not cross["percent_inside_band"] or cross["degrees_inside_band"]:
        print(_red("  FAIL: the cross-check does not discriminate"))
        return False
    print(
        _green(
            f"  PASS: scale table matches pinned values; the percent reading lands "
            f"inside the dataset band while the degrees reading is "
            f"{cross['degrees_sigma_outside']:.1f} sigma outside it"
        )
    )
    print(
        "       NOTE: these magnitudes are exact ARITHMETIC on verified inputs, "
        "NOT a hardware measurement — the raw-tick probe (plan 05-06) confirms them"
    )
    return True


def check_envelope_discrimination(index: str, poses: list[str]) -> bool:
    print(f"\n[{index}] envelope check — against the INITIAL pose, not the ready pose ...")
    ok = True
    for pose in poses:
        if pose not in DUME_POSES:
            print(_red(f"  FAIL: unknown pose {pose!r}; known: {sorted(DUME_POSES)}"))
            return False
        as_degrees = DUME_POSES[pose][:5]
        as_percent = degrees_to_percent(as_degrees)
        inside_deg = envelope_contains(as_degrees)
        inside_pct = envelope_contains(as_percent)
        discriminates = inside_deg != inside_pct
        print(f"       pose {pose!r}:")
        for i, joint in enumerate(ARM_JOINTS):
            print(
                f"         {joint:<14} {as_degrees[i]:>8.2f} deg "
                f"{'in ' if inside_deg[i] else 'OUT'} | "
                f"{as_percent[i]:>8.2f} %   {'in ' if inside_pct[i] else 'OUT'}"
            )
        if pose == "initial":
            if not discriminates:
                print(_red("  FAIL: the initial pose must discriminate"))
                ok = False
        elif discriminates:
            print(_red(f"  FAIL: pose {pose!r} unexpectedly discriminates"))
            ok = False
        else:
            print(
                "       NOTE: this pose reads IDENTICALLY under both conventions, so "
                "PAR-04's literal assertion on it decides nothing — recorded as a "
                "documented negative, never as the verdict's evidence"
            )
    if ok:
        print(
            _green(
                "  PASS: the initial pose's shoulder_lift discriminates "
                "(-102 outside as degrees, -98.33 inside as percent)"
            )
        )
    return ok


def check_pinned_constants(index: str, args: argparse.Namespace) -> bool:
    print(f"\n[{index}] pinned-constants drift against local data artifacts ...")
    ok, notes = check_pinned_constants_against_local_artifacts(
        statistics_path=args.statistics,
        calibration_path=args.calibration,
        # `--robot-id` defaults to None so the YAML config can supply it for the
        # LIVE session; the offline drift check needs the module fallback when
        # neither is given.
        #
        # `--robot-type` is deliberately NOT threaded in here: the calibration
        # directory is named after the robot CLASS, not after Dum-E's
        # `robot_type`, so passing `robot_type` is exactly the substitution that
        # made this check read a file the bus does not load. `--calibration` is
        # the escape hatch when the derivation is wrong.
        robot_id=args.robot_id or DEFAULT_ROBOT_ID,
    )
    for note in notes:
        print(f"       {note}")
    if not ok:
        print(
            _red(
                "  FAIL: a resolved artifact drifted from the pinned constants — "
                "the units verdict must be RE-DERIVED before proceeding"
            )
        )
        return False
    print(_green("  PASS: no drift in any resolvable artifact"))
    return True


def check_hardware_raw_tick(index: str, args: argparse.Namespace) -> bool:
    """The arm-required half, read-only. Never reached under --skip-hardware.

    Reports whichever normalization mode the constructed controller is ACTUALLY
    running — it does not presuppose one, which is the whole point of the probe.
    Commands no motion: the connect pre-arms the goal registers so the
    torque-enable is a hold (see :func:`prearm_goal_to_present`).
    """
    print(f"\n[{index}] raw-tick round-trip probe — REQUIRES THE ARM (read-only) ...")
    try:
        controller, info = open_controller(args)
    except Exception as exc:  # noqa: BLE001
        print(_red(f"  FAIL: cannot open the arm: {type(exc).__name__}: {exc}"))
        return False
    try:
        result = units_verdict(controller.robot.bus, controller.robot.bus.calibration)
    except Exception as exc:  # noqa: BLE001
        print(_red(f"  FAIL: probe raised {type(exc).__name__}: {exc}"))
        return False
    finally:
        try:
            close_controller(controller)
        except Exception:  # noqa: BLE001
            pass
    print(f"       calibration {info['calibration_path_redacted']}")
    print(f"       sha256      {info['calibration_sha256']}")
    print(f"       use_degrees={info['use_degrees']} max_relative_target={info['max_relative_target']}")
    for motor, row in result.items():
        print(
            f"       {motor:<14} tick={row['raw_tick']} reported={row['reported']} "
            f"active_mode={row['active_mode']}"
        )
    modes = {
        motor
        for motor, row in result.items()
        if not isinstance(row["active_mode"], str)
    }
    if modes:
        print(_red(f"  FAIL: ambiguous active mode for {sorted(modes)}"))
        return False
    print(_green("  PASS: the active normalization mode is proven per joint"))
    return True


def _summarize_live_sweep(results: dict) -> dict:
    """Derive the verdict fields from the measured per-pose records.

    Kept separate from the printing so the derivation is testable without an arm.

    TWO DIFFERENT QUESTIONS, ANSWERED SEPARATELY AND ON PURPOSE:

    * ``active_mode.running_config`` — which mode THE BUS is producing. Fixed by
      ``use_degrees``, which Dum-E deliberately keeps at ``True``. This is a fact
      about the configuration, not about the checkpoint.
    * ``checkpoint_convention_verdict`` — which convention THE CHECKPOINT was
      trained in. Decided only by rows that DISCRIMINATE: the ``initial`` pose's
      envelope membership, and ``elbow_flex`` sitting at its mechanical limit
      where percent reads exactly +/-100.0 (the value the checkpoint records) and
      degrees cannot.
    """
    arm_modes_running: set = set()
    arm_modes_percent: set = set()
    ambiguous: list[str] = []
    scale_disagreements: list[str] = []
    before_after_failures: list[str] = []
    discriminating: list[dict] = []
    limit_fingerprints: list[dict] = []

    for pose in results["poses"]:
        for motor, row in pose["joints"].items():
            if not isinstance(row["active_mode"], str):
                ambiguous.append(f"{pose['pose']}.{motor}={row['matched_candidates']}")
            elif motor in ARM_JOINTS:
                arm_modes_running.add(row["active_mode"])
            if motor in ARM_JOINTS:
                if (
                    abs(row["candidates"]["RANGE_M100_100"] - row["reported_percent_config"])
                    < MODE_MATCH_TOL
                ):
                    arm_modes_percent.add("RANGE_M100_100")
                else:
                    arm_modes_percent.add(f"UNMATCHED@{pose['pose']}.{motor}")
                measured = row["measured_deg_per_pct_two_sided"]
                if measured is not None and abs(measured - row["derived_deg_per_pct"]) > MEASURED_SCALE_TOL:
                    scale_disagreements.append(
                        f"{pose['pose']}.{motor}: measured {measured:.6f} != derived "
                        f"{row['derived_deg_per_pct']:.6f}"
                    )
                if row["at_calibrated_limit"]:
                    limit_fingerprints.append(
                        {
                            "pose": pose["pose"],
                            "joint": motor,
                            "raw_tick": row["raw_tick"],
                            "as_degrees": row["candidates"]["DEGREES"],
                            "as_percent": row["candidates"]["RANGE_M100_100"],
                        }
                    )
        for motor, diff in pose["before_after_diff"].items():
            if not abs(diff) < BEFORE_AFTER_TOL:
                before_after_failures.append(f"{pose['pose']}.{motor}={diff}")
        if pose["pose"] == "initial":
            discriminating = [
                row for row in pose["envelope_initial_convention_check"] if row["discriminates"]
            ]

    percent_wins = bool(discriminating) and all(
        row["percent_inside"] and not row["degrees_inside"] for row in discriminating
    )
    degrees_wins = bool(discriminating) and all(
        row["degrees_inside"] and not row["percent_inside"] for row in discriminating
    )
    verdict = "RANGE_M100_100" if percent_wins else ("DEGREES" if degrees_wins else "INDETERMINATE")

    return {
        "active_mode": {
            "running_config_use_degrees_true": (
                sorted(arm_modes_running)[0] if len(arm_modes_running) == 1 else sorted(arm_modes_running)
            ),
            "percent_config_use_degrees_false": (
                sorted(arm_modes_percent)[0] if len(arm_modes_percent) == 1 else sorted(arm_modes_percent)
            ),
        },
        "active_mode_running": (
            sorted(arm_modes_running)[0] if len(arm_modes_running) == 1 else sorted(arm_modes_running)
        ),
        "ambiguous_joints": ambiguous,
        "scale_disagreements": scale_disagreements,
        "before_after_failures": before_after_failures,
        "initial_pose_discriminating_rows": discriminating,
        "calibrated_limit_fingerprints": limit_fingerprints,
        "checkpoint_convention_verdict": verdict,
        "checkpoint_convention_agreement": "CONFIRM" if verdict == "RANGE_M100_100" else "CONTRADICT",
        "offline_verdict": "RANGE_M100_100",
    }


def check_live_pose_sweep(index: str, args: argparse.Namespace, pose_names: list[str]) -> bool:
    """Park at 3+ fixed poses and settle the units question on hardware."""
    print(f"\n[{index}] live pose sweep — COMMANDS MOTION on {len(pose_names)} poses ...")
    _repo_on_path()
    from embodiment.so_arm10x.controller import DUME_PID

    try:
        results = live_pose_sweep(args, pose_names)
    except Exception as exc:  # noqa: BLE001
        print(_red(f"  FAIL: the sweep raised {type(exc).__name__}: {exc}"))
        return False

    summary = _summarize_live_sweep(results)
    results.update(summary)
    path = write_corpus_results(results)

    print(f"       calibration {results['calibration_path']}")
    print(f"       sha256      {results['calibration_sha256']}")
    print(f"       raw output  {path.relative_to(REPO_ROOT)}")
    print(
        f"       pre-arm: {results['connect']['prearm']['reason']}; worst pending jump was "
        f"{results['connect']['prearm']['worst_pending_jump_ticks']} ticks"
    )
    for pose in results["poses"]:
        print(f"       pose {pose['pose']!r}:")
        for motor in JOINT_NAMES:
            row = pose["joints"][motor]
            measured = row["measured_deg_per_pct_two_sided"]
            print(
                f"         {motor:<14} tick={row['raw_tick']:>4} "
                f"reported={row['reported_running_config']:>9.4f} "
                f"pct={row['reported_percent_config']:>9.4f} "
                f"pre={row['pre_upgrade_v033']:>9.4f} diff={row['diff']:.2e} "
                f"mode={row['active_mode']:<14} "
                f"scale={'n/a' if measured is None else f'{measured:.5f}'}"
            )

    ok = True
    if len(results["poses"]) < 3:
        print(_red(f"  FAIL: only {len(results['poses'])} poses swept (need >= 3)"))
        ok = False
    if summary["ambiguous_joints"]:
        print(_red(f"  FAIL: ambiguous active mode (0 or >1 candidate matched): {summary['ambiguous_joints']}"))
        ok = False
    if summary["before_after_failures"]:
        print(_red(f"  FAIL: before/after difference reached {BEFORE_AFTER_TOL}: {summary['before_after_failures']}"))
        ok = False
    if summary["scale_disagreements"]:
        print(_red(f"  FAIL: measured scale disagrees with the derived table: {summary['scale_disagreements']}"))
        ok = False
    if results["pid_readback"] != {motor: dict(DUME_PID) for motor in results["pid_readback"]}:
        print(_red(f"  FAIL: PID read-back is not the Dum-E preset: {results['pid_readback']}"))
        ok = False
    if not summary["initial_pose_discriminating_rows"]:
        print(_red("  FAIL: the initial pose produced no discriminating joint — the live "
                   "measurement cannot decide the checkpoint's convention"))
        ok = False
    if summary["checkpoint_convention_agreement"] != "CONFIRM":
        print(
            _red(
                f"  FAIL: the live measurement says the checkpoint convention is "
                f"{summary['checkpoint_convention_verdict']}, CONTRADICTING the offline "
                f"verdict RANGE_M100_100. Record the contradiction with both provenances "
                f"and STOP — do not widen a tolerance and do not reconcile the two."
            )
        )
        ok = False

    print("       initial-pose envelope discrimination (the rows that decide):")
    for row in summary["initial_pose_discriminating_rows"]:
        print(
            f"         {row['joint']:<14} q01={row['q01']:.4f} q99={row['q99']:.4f} | "
            f"{row['as_degrees']:>9.4f} deg {'in ' if row['degrees_inside'] else 'OUT'} | "
            f"{row['as_percent']:>9.4f} %   {'in ' if row['percent_inside'] else 'OUT'}"
        )
    for row in summary["calibrated_limit_fingerprints"]:
        print(
            f"       at-limit fingerprint: {row['joint']} at tick {row['raw_tick']} reads "
            f"{row['as_degrees']:.4f} deg / {row['as_percent']:.4f} % — the checkpoint records "
            f"100.0, which only the percent convention can produce"
        )
    print(
        f"       active mode: running config -> {summary['active_mode']['running_config_use_degrees_true']}; "
        f"percent config -> {summary['active_mode']['percent_config_use_degrees_false']}"
    )
    if ok:
        print(
            _green(
                f"  PASS: exactly one candidate matched every joint at every pose; the "
                f"before/after difference is under {BEFORE_AFTER_TOL} everywhere; the "
                f"measured scale reproduces the derived table; PID reads back as the preset; "
                f"and the checkpoint convention measures as {summary['checkpoint_convention_verdict']}, "
                f"{summary['checkpoint_convention_agreement']}ing the offline verdict"
            )
        )
    return ok


def check_clamp_demo(index: str, args: argparse.Namespace) -> bool:
    """Command an oversized per-step delta and prove the clamp clipped it."""
    print(f"\n[{index}] clamp demonstration — COMMANDS MOTION on one joint ...")
    try:
        record = demo_clamp(args)
    except Exception as exc:  # noqa: BLE001
        print(_red(f"  FAIL: the demonstration raised {type(exc).__name__}: {exc}"))
        return False

    path = write_corpus_results(record)
    print(f"       raw output  {path.relative_to(REPO_ROOT)}")
    print(
        f"       {record['joint']}: present={record['present']:.4f} "
        f"requested={record['requested']:.4f} returned={record['returned']:.4f} "
        f"settled={record['settled']:.4f}"
    )
    print(
        f"       clamp={record['clamp']} returned_delta={record['returned_delta']:.4f} "
        f"expected_clipped={record['expected_clipped']:.4f}"
    )
    print(f"       Dum-E loguru warning : {record['dume_loguru_warning']}")
    print(f"       upstream bridged     : {record['upstream_bridged_warning']}")

    ok = True
    if not record["returned_differs_from_requested"]:
        print(_red("  FAIL: the returned action equalled the requested action — the clamp did "
                   "not engage, so it is not configured on the live path"))
        ok = False
    if not record["returned_delta_equals_clamp"]:
        print(_red(f"  FAIL: the returned delta {record['returned_delta']:.4f} does not equal the "
                   f"clamp {record['clamp']} within {record['divergence_threshold']}"))
        ok = False
    if not record["clamped_joints"]:
        print(_red("  FAIL: diff_clamped_joints reported no clamped joint"))
        ok = False
    if not record["dume_loguru_warning"]:
        print(_red("  FAIL: no clamp warning was captured on Dum-E's loguru stream — the clamp "
                   "fired but is still not surfaced, which is the swallowed-warning failure "
                   "SAFE-02 targets"))
        ok = False
    if not record["upstream_bridged_warning"]:
        print("       NOTE: upstream's own root-logger warning did NOT arrive through the "
              "bridge; Dum-E's own warning did, so the signal is not lost. Recorded either way.")
    if ok:
        print(
            _green(
                f"  PASS: a {record['multiple']}x-clamp delta on {record['joint']} was clipped to "
                f"exactly the clamp on real hardware, and the warning reached Dum-E's own stream"
            )
        )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hardware",
        action="store_true",
        help="Run only the arm-free discriminators (no serial port is opened).",
    )
    parser.add_argument("--port", default=os.getenv("SO_ARM_PORT"))
    # These four default to None so the Dum-E YAML config can supply them; see
    # `resolve_live_controller_settings`. Their module-level fallbacks are only
    # reached when neither a flag nor a config file names them.
    parser.add_argument("--robot-id", default=None)
    parser.add_argument("--robot-type", default=None)
    parser.add_argument(
        "--wrist-cam-idx",
        type=int,
        default=None,
        help="OpenCV index of the wrist camera (else the config file, else 0).",
    )
    parser.add_argument(
        "--front-cam-idx",
        type=int,
        default=None,
        help="OpenCV index of the front camera (else the config file, else 1).",
    )
    parser.add_argument(
        "--pose-sequence",
        default=None,
        help=(
            "Comma-separated fixed poses for the LIVE sweep, e.g. 'initial,ready,remote'. "
            "REQUIRES THE ARM and COMMANDS MOTION."
        ),
    )
    parser.add_argument(
        "--demo-clamp",
        action="store_true",
        help=(
            "Command one deliberately oversized per-step joint delta and prove the "
            "SAFE-02 clamp clipped it. REQUIRES THE ARM and COMMANDS MOTION."
        ),
    )
    parser.add_argument(
        "--statistics",
        default=None,
        help="Path to the checkpoint statistics.json (else env / repo-relative).",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Path to the follower calibration JSON (else HF_LEROBOT_CALIBRATION).",
    )
    parser.add_argument(
        "--poses",
        default="initial,ready",
        help="Comma-separated poses for the envelope check (default: initial,ready).",
    )
    args = parser.parse_args()

    poses = [p.strip() for p in args.poses.split(",") if p.strip()]
    sweep_poses = (
        [p.strip() for p in args.pose_sequence.split(",") if p.strip()]
        if args.pose_sequence
        else []
    )
    # A live flag implies hardware; --skip-hardware wins outright so the offline
    # gate can never be turned into a motion command by a stray flag.
    if args.skip_hardware:
        sweep_poses, args.demo_clamp = [], False
    total = 5 + (0 if args.skip_hardware else 1) + bool(sweep_poses) + bool(args.demo_clamp)

    print("=" * 72)
    print(" Normalization-units probe (PAR-04 / PAR-06) — verdict: RANGE_M100_100")
    print("=" * 72)

    results: dict[str, bool] = {}
    results["clip_fingerprint"] = check_clip_fingerprint(f"1/{total}")
    results["degrees_falsification"] = check_degrees_falsification(f"2/{total}")
    results["scale_and_cross_check"] = check_scale_and_cross_check(f"3/{total}")
    results["envelope_discrimination"] = check_envelope_discrimination(
        f"4/{total}", poses
    )
    results["pinned_constants"] = check_pinned_constants(f"5/{total}", args)
    next_index = 6
    if args.skip_hardware:
        print(
            "\n[not run] raw-tick round-trip probe, live pose sweep and clamp "
            "demonstration — all three need the arm (--skip-hardware). The per-joint "
            "scale above is a MEASUREMENT taken on this arm; see the '## Live "
            "confirmation' section of docs/UNITS-VERDICT.md for the numbers."
        )
    elif not results["pinned_constants"]:
        # A FAILED drift check must STOP the arm half, not merely colour the exit
        # code. The pinned constants are the input the per-joint scale table is
        # derived from, and the arm half opens the bus, enables torque and (with
        # --pose-sequence / --demo-clamp) commands motion. Proceeding would measure
        # a stack the pinned table no longer describes, and would validate every
        # commanded target against a calibration the bus is not using. Neither of
        # these checks is recorded as PASS or FAIL: they did not run.
        print(
            _red(
                "\n[REFUSED] raw-tick round-trip probe, live pose sweep and clamp "
                "demonstration — the pinned constants drifted from the local "
                "artifacts, so nothing derived from them may be measured or "
                "commanded against this arm. Re-derive the constants first, or pass "
                "--calibration if the derived path is wrong."
            )
        )
    else:
        results["hardware_raw_tick"] = check_hardware_raw_tick(f"{next_index}/{total}", args)
        next_index += 1
        if sweep_poses:
            results["live_pose_sweep"] = check_live_pose_sweep(
                f"{next_index}/{total}", args, sweep_poses
            )
            next_index += 1
        if args.demo_clamp:
            results["clamp_demo"] = check_clamp_demo(f"{next_index}/{total}", args)
            next_index += 1

    print("\n" + "=" * 72)
    passed = sum(1 for ok in results.values() if ok)
    for name, ok in results.items():
        print(f"  {_green('PASS') if ok else _red('FAIL')}  {name}")
    print(f" {passed}/{len(results)} checks passed")
    print("=" * 72)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
