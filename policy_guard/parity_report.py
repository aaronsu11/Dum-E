"""Strict numerical instruments. Arithmetic never authorizes an experiment.

Only explicitly injected Evidence(test_only=True) accepts fabricated witnesses.
The CLI uses the real-model boundary and a prior subject-matching agreement.
"""

from __future__ import annotations

import numpy as np

from policy_guard.parity_gate import (
    COMPARISON_PROFILES, PREPROCESSING_KEYS, evidence, require, timestamp,
    validate_tolerance_agreement, validate_tolerance_proposal,
)
from policy_guard.replay_contract import (
    CAMERA_ORDER, JOINT_ORDER, canonical, fingerprint_configuration,
    validate_schedule, write_tensors,
)


def validate_coverage(cases, expected):
    require(bool(expected) and cases == expected, "complete ordered case coverage required")
    keys = [canonical(key) for key in cases]
    require(len(set(keys)) == len(keys), "duplicate case identity")


def _finite(values):
    values = np.asarray(values)
    require(values.dtype.kind in "biuf" and values.size > 0 and np.isfinite(values).all(),
            "nonempty finite numeric arrays required")
    return values


def joint_deviations(left, right):
    left, right = _finite(left), _finite(right)
    require(left.shape == right.shape and left.ndim == 3 and left.shape[1:] == (16, 6),
            "full decoded trace shape must be (case,16,6)")
    return right.astype(np.float64) - left.astype(np.float64)


def signed_bias(delta):
    delta = _finite(delta)
    require(delta.ndim == 3 and delta.shape[1:] == (16, 6), "full decoded delta required")
    return delta.astype(np.float64).mean(axis=1)


def chunk_index_slopes(delta):
    delta = _finite(delta)
    require(delta.ndim == 3 and delta.shape[1:] == (16, 6), "full decoded delta required")
    indices = np.arange(16, dtype=np.float64) - 7.5
    return np.einsum("ctj,t->cj", delta.astype(np.float64), indices) / np.dot(indices, indices)


def metrics(delta):
    """Same float64 definitions consumed independently by the release validator."""
    delta = _finite(delta).astype(np.float64)
    bias, slopes = signed_bias(delta), chunk_index_slopes(delta)
    return {
        "max_abs": np.abs(delta).max(axis=(0, 1)),
        "mean_abs": np.abs(delta).mean(axis=(0, 1)),
        "bias": bias.mean(axis=0), "slope": slopes.mean(axis=0),
        "per_index_bias": delta.mean(axis=0),
        "trace_bias_max_abs": np.abs(bias).max(axis=0),
        "trace_slope_max_abs": np.abs(slopes).max(axis=0),
    }


def array_comparison(left, right, bounds, *, exact=False):
    left, right = _finite(left), _finite(right)
    require(left.shape == right.shape, "unequal full tensor shape")
    require(left.dtype == right.dtype, "unequal serialized tensor dtype")
    if exact or left.dtype.kind in "biu":
        return bool(np.array_equal(left, right))
    delta = np.abs(right.astype(np.float64) - left.astype(np.float64))
    return bool(np.all(delta <= bounds["atol"] + bounds["rtol"] * np.abs(left.astype(np.float64))))


def _rows(ev, left, right, keys, field, bounds):
    require(len(left[field]) == len(right[field]) == len(keys), f"partial {field} coverage")
    rows, failures, checked = [], [], {}
    for index, (key, lref, rref) in enumerate(zip(keys, left[field], right[field], strict=True)):
        row = {"key": key, "left": lref, "right": rref}
        rows.append(row)
        cache = fingerprint_configuration([lref, rref])
        if cache not in checked:
            lvalues, rvalues = ev.tensors(lref), ev.tensors(rref)
            require(set(lvalues) == set(rvalues), f"{field}: different input keys")
            if field in ("preprocessing", "common_inputs"):
                require(set(lvalues) == PREPROCESSING_KEYS, f"{field}: incomplete preprocessing witness")
            else:
                require(bool(lvalues), "empty full collated input witness")
            passed = True
            for name in lvalues:
                if name in ("tokens", "mask"):
                    require(lvalues[name].dtype.kind in "biu", "tokens/masks must be exact integers")
                passed &= array_comparison(lvalues[name], rvalues[name], bounds,
                                           exact=field in ("common_inputs", "common_collated"))
            checked[cache] = passed
        if not checked[cache]:
            failures.append(f"{field}[{index}]")
    return rows, failures


def _bundle(ev, bundle, profile, schedule):
    validate_coverage(bundle["cases"], schedule)
    require(bundle["profile"] == profile, "comparison profile mismatch")
    require(bundle["input_fingerprint"] == ev.identity()["input_fingerprint"], "input identity mismatch")
    require(bundle["joint_order"] == list(JOINT_ORDER), "joint ordering mismatch")
    require(bundle["camera_order"] == list(CAMERA_ORDER), "camera ordering mismatch")
    arrays = ev.tensors(bundle["tensors"])
    require(set(arrays) == {"raw", "noise", "decoded"}, "raw/noise/decoded witnesses required")
    for name, shape in (("raw", (600, 40, 132)), ("noise", (600, 40, 132)), ("decoded", (600, 16, 6))):
        require(arrays[name].shape == shape and arrays[name].dtype == np.float32,
                f"full {name} shape/dtype required")
        _finite(arrays[name])
    return arrays


def compare_tiers(workspace, name, left, right, *, started_at, ended_at, matrix_reference=None):
    """Emit discriminating metrics even on numerical failure; structural faults raise."""
    ev = evidence(workspace)
    agreement = validate_tolerance_agreement(ev, comparison_started_at=started_at)
    proposal = validate_tolerance_proposal(ev)
    require(timestamp(started_at) <= timestamp(ended_at), "comparison chronology reversed")
    require(name in COMPARISON_PROFILES, "unknown comparison")
    rule = proposal["comparisons"][name]
    profiles = {p["backend"] + "-" + p["purpose"]: p["observed"] for p in ev.json("profiles.json")["profiles"] if p["purpose"] != "stock-capacity"}
    if name == "diagnostic":
        for key in ("device", "rng_algorithm", "noise_dtype", "noise_shape"):
            require(profiles[rule["profiles"][0]][key] == profiles[rule["profiles"][1]][key], "diagnostic actual sampler controls differ")
    schedule = ev.json("input-lock.json")["schedule"]
    validate_schedule(ev.json("input-lock.json"), schedule, "replay")
    la = _bundle(ev, left, rule["profiles"][0], schedule)
    ra = _bundle(ev, right, rule["profiles"][1], schedule)
    rows, failures = {}, []
    for field in ("preprocessing", "independent_collated", "common_inputs", "common_collated"):
        rows[field], issues = _rows(ev, left, right, schedule, field, rule["thresholds"]["preprocessing"])
        failures.extend(issues)
    if not array_comparison(la["raw"], ra["raw"], rule["thresholds"]["raw"]):
        failures.append("raw")
    noise_equal = bool(np.array_equal(la["noise"], ra["noise"]))
    if rule["noise_policy"] == "exact" and not noise_equal:
        failures.append("noise")
    delta = joint_deviations(la["decoded"], ra["decoded"])
    actual = metrics(delta)
    for metric in ("max_abs", "mean_abs", "bias", "slope"):
        if np.any(np.abs(actual[metric]) > rule["thresholds"]["decoded"][metric]):
            failures.append("decoded." + metric)
    for metric, bound in (("trace_bias_max_abs", "bias"), ("trace_slope_max_abs", "slope")):
        if np.any(actual[metric] > rule["thresholds"]["decoded"][bound]):
            failures.append("decoded.trace_" + bound)
    matrix_values = {
        "delta": delta, "trace_bias": signed_bias(delta), "trace_slope": chunk_index_slopes(delta),
        "trace_max_abs": np.abs(delta).max(axis=1), "trace_mean_abs": np.abs(delta).mean(axis=1),
    }
    if matrix_reference is None:
        matrix = write_tensors(ev.workspace, matrix_values)
    else:
        saved = ev.tensors(matrix_reference)
        require(set(saved) == set(matrix_values) and all(np.array_equal(saved[k], v) for k, v in matrix_values.items()), "archived signed matrix differs")
        matrix = matrix_reference
    return {
        "name": name, "profiles": rule["profiles"], "cases": schedule,
        "started_at": started_at, "ended_at": ended_at,
        "agreement": ev.reference("tolerance-agreement.json"),
        "agreement_decided_at": agreement["decided_at"],
        "joint_order": list(JOINT_ORDER), "camera_order": list(CAMERA_ORDER),
        "left": left["tensors"], "right": right["tensors"], **rows,
        "metrics": {key: value.tolist() for key, value in actual.items()},
        "matrix": matrix, "units": proposal["units"],
        "slope_units": [unit + "/action_index" for unit in proposal["units"]],
        "axes": ["record_seed_repeat", "chunk_index_0..15", "joint"],
        "delta_definition": "right-minus-left", "aggregation": proposal["aggregation"],
        "thresholds": rule["thresholds"], "passed": not failures, "failures": failures,
        "noise": {"policy": rule["noise_policy"], "equal": noise_equal,
                  "matched_seed_proves_equal_noise": False,
                  "profiles": [{key: profiles[p][key] for key in ("device", "rng_algorithm", "noise_dtype", "noise_shape")} for p in rule["profiles"]],
                  "treatment": rule["rationale"]},
        "caveats": proposal["caveats"],
    }
