"""Reduce a frozen 12-case GPU sample from preserved operational captures.

This is a scoped milestone check, not the historical exhaustive release gate.
It performs arithmetic only: no model loading, inference, or hardware access.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from policy_guard.parity_gate import (
    Evidence, array_comparison, chunk_index_slopes, instrument_identity,
    joint_deviations, metrics, require, signed_bias, timestamp,
    validate_numerical_worker, validate_tolerance_agreement,
)
from policy_guard.replay_contract import now, write_evidence, write_tensors

SCOPE = "milestone-scope.json"
REPORT = "milestone-report.json"
SELECTION = "middle stored observation per episode; first locked seed"


def selected_cases(lock):
    episodes = {}
    for record in lock["records"]:
        episodes.setdefault(record["episode_index"], []).append(record)
    require(len(episodes) == 12, "exactly twelve recorded episodes required")
    chosen = []
    for episode, records in sorted(episodes.items()):
        records = sorted(records, key=lambda r: (r["frame_index"], r["file"]))
        record = records[len(records) // 2]
        chosen.append({"record": record["file"], "seed": record["seeds"][0]})
    require(len({k["record"] for k in chosen}) == 12, "duplicate selected record")
    return chosen


def frozen_scope(ev):
    saved = ev.json(SCOPE)
    require(saved["schema"] == "dume.checkpoint-milestone-scope.v1", "scope schema mismatch")
    require(saved["authorization"]["selected_user_text"] == "12 observations, one seed", "scope authorization mismatch")
    require(saved["input_lock"] == ev.reference("input-lock.json"), "input lock changed")
    require(saved["original_proposal"] == ev.reference("tolerance-proposal.json"), "proposal changed")
    require(saved["original_tolerance_agreement"] == ev.reference("tolerance-agreement.json"), "agreement changed")
    cases = [{"record": row["record"], "seed": row["seed"]} for row in saved["selected"]]
    require(cases == selected_cases(ev.json("input-lock.json")), "frozen selection differs")
    proposal = ev.json(saved["original_proposal"])
    return {
        "created_at": saved["recorded_at"], "cases": cases,
        "agreement": saved["original_tolerance_agreement"],
        "proposal": saved["original_proposal"], "proofs": saved["sources"],
        "rule": proposal["comparisons"]["operational"],
        "limitations": ["One seed per selected observation; not stochastic coverage",
                        "Independent preprocessing; no common-input isolation rerun",
                        "No exhaustive precision bridges or full-corpus verdict",
                        "No golden or hardware approval"],
    }


def selected_capture(workspace, scope, name, end):
    ev = Evidence(workspace)
    proof = scope["proofs"][name]
    launch = ev.json(proof["launch"])
    validate_tolerance_agreement(ev, comparison_started_at=launch["started_at"])
    result = validate_numerical_worker(
        ev, proof, name, ev.json("input-lock.json")["schedule"], kind="replay",
        start=launch["started_at"], end=end,
    )
    profile = result["worker"]["profile"]
    require(profile["device"].startswith("cuda"), "GPU capture required")
    require(profile["purpose"] == "operational", "operational capture required")
    require(profile["backbone_dtypes"] == ["torch.bfloat16"], "BF16 backbone required")
    keys = result["worker"]["executed_cases"]
    selected = [result["captures"][keys.index(key)] for key in scope["cases"]]
    # Retain only selected arrays. The source validator checks the original full
    # launch/case provenance; it does not run inference or compare other cases.
    return selected, profile


def compare_arrays(left, right, rule):
    require(len(left) == len(right) == 12, "exactly twelve paired captures required")
    failures, inputs = [], []
    bounds = rule["thresholds"]
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        row = {"index": index, "differences": [], "schema_differences": [],
               "captured_dtype_codes": {
                   "native": {k: int(v) for k, v in a.items() if k.startswith("dtype.")},
                   "lerobot": {k: int(v) for k, v in b.items() if k.startswith("dtype.")},
               }}
        for prefix in ("preprocessing.", "collated."):
            ak, bk = {k for k in a if k.startswith(prefix)}, {k for k in b if k.startswith(prefix)}
            require(ak and bk, "missing captured inputs")
            if ak != bk:
                row["schema_differences"].append({"prefix": prefix,
                    "native_only": sorted(ak - bk), "lerobot_only": sorted(bk - ak)})
            for key in sorted(ak & bk):
                if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
                    row["schema_differences"].append({"key": key,
                        "native_shape": list(a[key].shape), "lerobot_shape": list(b[key].shape),
                        "native_dtype": str(a[key].dtype), "lerobot_dtype": str(b[key].dtype)})
                elif not array_comparison(a[key], b[key], bounds["preprocessing"]):
                    row["differences"].append(key)
        if row["differences"] or row["schema_differences"]:
            failures.append(f"inputs[{index}]")
        inputs.append(row)
    stacked = [{key: np.stack([row[key].squeeze(0) if key != "decoded" else row[key]
                              for row in side]) for key in ("raw", "noise", "decoded")}
               for side in (left, right)]
    a, b = stacked
    for side in stacked:
        for key, shape in (("raw", (12, 40, 132)), ("noise", (12, 40, 132)), ("decoded", (12, 16, 6))):
            require(side[key].shape == shape and np.isfinite(side[key]).all(), "invalid full captured " + key)
    if not array_comparison(a["raw"], b["raw"], bounds["raw"]):
        failures.append("raw")
    equal_noise = bool(np.array_equal(a["noise"], b["noise"]))
    if rule["noise_policy"] == "exact" and not equal_noise:
        failures.append("noise")
    delta = joint_deviations(a["decoded"], b["decoded"])
    stats = metrics(delta)
    for key in ("max_abs", "mean_abs", "bias", "slope"):
        if np.any(np.abs(stats[key]) > bounds["decoded"][key]):
            failures.append("decoded." + key)
    for key, bound in (("trace_bias_max_abs", "bias"), ("trace_slope_max_abs", "slope")):
        if np.any(stats[key] > bounds["decoded"][bound]):
            failures.append("decoded." + key)
    return {
        "passed": not failures, "failures": failures, "inputs": inputs,
        "metrics": {k: v.tolist() for k, v in stats.items()},
        "raw_max_abs": float(np.abs(b["raw"].astype(np.float64) - a["raw"]).max()),
        "noise": {"policy": rule["noise_policy"], "equal": equal_noise,
                  "matched_seed_proves_equal_noise": False},
    }, {"delta": delta, "trace_bias": signed_bias(delta), "trace_slope": chunk_index_slopes(delta)}


def run(workspace):
    ev = Evidence(workspace)
    require(not (ev.workspace / REPORT).exists(), "immutable report already exists")
    scope = frozen_scope(ev)
    start = now()
    require(timestamp(scope["created_at"]) < timestamp(start), "sample must precede comparison")
    sides, profiles = [], []
    for name in ("native-operational", "lerobot-operational"):
        arrays, profile = selected_capture(workspace, scope, name, start)
        sides.append(arrays)
        profiles.append(profile)
        gc.collect()
    result, matrix = compare_arrays(*sides, scope["rule"])
    report = {
        "schema_version": 1, "kind": "milestone_gpu_12", "evidence_kind": "real_model_reduction",
        "started_at": start, "ended_at": now(), "status": "complete" if result["passed"] else "failed",
        "scope": ev.reference(SCOPE), "cases": scope["cases"], "case_count": 12,
        "agreement": scope["agreement"], "proofs": scope["proofs"],
        "reducer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_instrument": instrument_identity(), "profiles": profiles,
        "thresholds": scope["rule"]["thresholds"], "joint_order": ev.json("input-lock.json")["joint_order"],
        "units": ev.json(scope["proposal"])["units"], "delta_definition": "lerobot-minus-native",
        "matrix": write_tensors(ev.workspace, matrix), "limitations": scope["limitations"],
        "inference_runs_added": 0, "old_exhaustive_gate_passed": False,
        "release_ready": False, **result,
    }
    write_evidence(ev.workspace, REPORT, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.workspace)
    print(json.dumps({k: result[k] for k in ("status", "case_count", "passed", "failures", "raw_max_abs", "noise", "metrics")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
