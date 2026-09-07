#!/usr/bin/env python3
"""Arm-free normalization-units discriminators for PAR-04 / PAR-06.

SKELETON (RED phase of plan 05-03 Task 1). Constants are pinned; the
discriminators are not implemented yet. `tests/test_units_verdict.py` drives
this module and MUST fail until the GREEN phase lands the arithmetic.
"""

from __future__ import annotations

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

_NOT_YET = "RED phase (plan 05-03 Task 1): not implemented yet"


# --- Discriminators (stubs) --------------------------------------------------


def norm_mode_for(joint: str, use_degrees: bool) -> str:
    raise NotImplementedError(_NOT_YET)


def normalize_m100_100(tick: float, lo: float, hi: float, drive_mode: int = 0) -> float:
    raise NotImplementedError(_NOT_YET)


def normalize_0_100(tick: float, lo: float, hi: float, drive_mode: int = 0) -> float:
    raise NotImplementedError(_NOT_YET)


def normalize_degrees(tick: float, lo: float, hi: float) -> float:
    raise NotImplementedError(_NOT_YET)


def normalize_v033(
    tick: float, lo: float, hi: float, drive_mode: int, norm_mode: str
) -> float:
    raise NotImplementedError(_NOT_YET)


def normalize_v061(
    tick: float, lo: float, hi: float, drive_mode: int, norm_mode: str
) -> float:
    raise NotImplementedError(_NOT_YET)


def deg_per_pct_table() -> dict[str, float]:
    raise NotImplementedError(_NOT_YET)


def pct_per_deg_table() -> dict[str, float]:
    raise NotImplementedError(_NOT_YET)


def degrees_reachable_range(joint: str) -> tuple[float, float]:
    raise NotImplementedError(_NOT_YET)


def degrees_to_percent(vector_deg: list[float]) -> list[float]:
    raise NotImplementedError(_NOT_YET)


def clip_fingerprint_count() -> int:
    raise NotImplementedError(_NOT_YET)


def envelope_contains(vector: list[float], group: str = "state") -> list[bool]:
    raise NotImplementedError(_NOT_YET)


def active_mode_from_raw_tick(
    tick: float, reported: float, lo: float, hi: float, drive_mode: int = 0
) -> list[str]:
    raise NotImplementedError(_NOT_YET)


def units_verdict(bus, calibration) -> dict:
    raise NotImplementedError(_NOT_YET)


def check_pinned_constants_against_local_artifacts(
    statistics_path=None, calibration_path=None
):
    raise NotImplementedError(_NOT_YET)


def main() -> int:
    raise NotImplementedError(_NOT_YET)


if __name__ == "__main__":
    import sys

    sys.exit(main())
