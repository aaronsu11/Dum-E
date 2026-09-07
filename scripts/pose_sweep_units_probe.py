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

The hardware half (``units_verdict()`` / ``active_mode_from_raw_tick()``) is
shipped here but NOT run by this plan: ``--skip-hardware`` opens no serial port
and needs no arm. Plan 05-06 runs it behind the hardware-attach gate, which is
what converts the per-joint scale magnitudes from exact arithmetic on verified
inputs into an actual measurement.

Usage:
    uv run python scripts/pose_sweep_units_probe.py --skip-hardware
    uv run python scripts/pose_sweep_units_probe.py --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
# HF_LEROBOT_CALIBRATION root (robots/<robot_type>/<robot_id>.json). Raw encoder
# ticks. Every motor has drive_mode 0, which is why both candidate normalization
# modes share the same midpoint and the delta carries no sign flip.
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

# Default LeRobot robot identity used to derive the calibration path. Both are
# overridable on the command line; neither is ever an absolute path.
DEFAULT_ROBOT_TYPE = "so101_follower"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"

# Repo-relative location of the (gitignored) checkpoint statistics.
STATISTICS_RELPATH = Path("checkpoints") / "GR00T-N1.7-3B-SO101" / "statistics.json"

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    raw = bus.sync_read("Present_Position", normalize=False, num_retry=2)
    got = bus.sync_read("Present_Position", normalize=True, num_retry=2)

    verdict = {}
    for motor, tick in raw.items():
        entry = calibration[motor]
        lo = getattr(entry, "range_min", None)
        hi = getattr(entry, "range_max", None)
        drive_mode = getattr(entry, "drive_mode", 0)
        if lo is None or hi is None:  # a plain dict, not a calibration dataclass
            lo, hi = entry["range_min"], entry["range_max"]
            drive_mode = entry.get("drive_mode", 0)
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


# --- Local data artifacts: resolved, never hardcoded ------------------------


def _hf_lerobot_home() -> Path:
    """The LeRobot cache root, from the documented env vars with upstream defaults."""
    explicit = os.environ.get("HF_LEROBOT_HOME")
    if explicit:
        return Path(explicit).expanduser()
    hf_cache = os.environ.get("HF_HOME")
    base = Path(hf_cache).expanduser() if hf_cache else Path.home() / ".cache" / "huggingface"
    return base / "lerobot"


def resolve_calibration_path(
    explicit: str | None = None,
    robot_type: str = DEFAULT_ROBOT_TYPE,
    robot_id: str = DEFAULT_ROBOT_ID,
) -> Path:
    """Locate the follower calibration JSON. Never a hardcoded absolute path."""
    if explicit:
        return Path(explicit).expanduser()
    env_root = os.environ.get("HF_LEROBOT_CALIBRATION")
    root = Path(env_root).expanduser() if env_root else _hf_lerobot_home() / "calibration"
    return root / "robots" / robot_type / f"{robot_id}.json"


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
    robot_type: str = DEFAULT_ROBOT_TYPE,
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

    cal_file = resolve_calibration_path(calibration_path, robot_type, robot_id)
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
        robot_type=args.robot_type,
        robot_id=args.robot_id,
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
    """The arm-required half. Never reached under --skip-hardware.

    Reports whichever normalization mode the constructed controller is ACTUALLY
    running — it does not presuppose one, which is the whole point of the probe.
    """
    print(f"\n[{index}] raw-tick round-trip probe — REQUIRES THE ARM ...")
    if not args.port:
        print(_red("  FAIL: --port is required for the hardware probe"))
        return False
    try:
        # Imported here so --skip-hardware opens no serial port and pulls in no
        # part of the lerobot robot stack at all.
        from embodiment.so_arm10x.controller import SO10xArmController
    except Exception as exc:  # noqa: BLE001
        print(_red(f"  FAIL: cannot import the controller: {type(exc).__name__}: {exc}"))
        return False
    controller = SO10xArmController(
        robot_type=args.robot_type, robot_port=args.port, robot_id=args.robot_id
    )
    try:
        controller.connect()
        result = units_verdict(controller.robot.bus, controller.robot.bus.calibration)
    except Exception as exc:  # noqa: BLE001
        print(_red(f"  FAIL: probe raised {type(exc).__name__}: {exc}"))
        return False
    finally:
        try:
            controller.disconnect()
        except Exception:  # noqa: BLE001
            pass
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-hardware",
        action="store_true",
        help="Run only the arm-free discriminators (no serial port is opened).",
    )
    parser.add_argument("--port", default=os.getenv("SO_ARM_PORT"))
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--robot-type", default=DEFAULT_ROBOT_TYPE)
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
    total = 5 if args.skip_hardware else 6

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
    if not args.skip_hardware:
        results["hardware_raw_tick"] = check_hardware_raw_tick(f"6/{total}", args)
    else:
        print(
            "\n[not run] raw-tick round-trip probe — deferred to plan 05-06 behind "
            "the hardware-attach gate (--skip-hardware). The per-joint scale above "
            "stays a HYPOTHESIS until that probe measures it."
        )

    print("\n" + "=" * 72)
    passed = sum(1 for ok in results.values() if ok)
    for name, ok in results.items():
        print(f"  {_green('PASS') if ok else _red('FAIL')}  {name}")
    print(f" {passed}/{len(results)} checks passed")
    print("=" * 72)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
