"""One explicit continuation of the September 11 attestation-only latency fix.

Historical captures and decisions retain their bytes. This validates a source
transition and one real fixed-seed serving witness; it never grants approval.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import os
from pathlib import Path

import numpy as np

from policy_guard import parity_gate as gate
from policy_guard.replay_contract import capture_bytes, load_case, now, write_evidence

ROOT = Path(__file__).resolve().parents[1]
RECORD = "instrumentation-transition.json"
SERVER = "docker/lerobot-policy/server.py"
ACCEPTANCE = "policy_guard/milestone_acceptance.py"
GOLDEN = "policy_guard/milestone_golden.py"
THIS = "policy_guard/instrumentation_transition.py"
BEFORE_SERVER = "06e5ae691924aa612c0560813cf668f09e2ab13913403234458d5c8ffeb1c847"
INSTRUCTION = "Let's fix this and rerun from trial 1"
BASELINE = {
    "session.json": "58a7b09ef93b2e19821804fe2b8ac137ab7d879103e50f2a4fbbf1719eaba5c1",
    "input-lock.json": "5f3d5ec3354df6c474280aed10fd3fa818e570cc8375337b61199265f4bf4da4",
    "calibration.json": "1a7234939b2783cda2aeb55cfb0e3e229e3d676c69aaba0d6743d1d8cc16fd6c",
    "profiles.json": "bb648734a46486a473076dcc4ab85113a3b42a3ffbd5b6eebf759a8d42f1be81",
    "milestone-scope.json": "fae6835e8df81b453e2bbc19fe9cdbf64982af25d3569982ea4ce2d782172ba2",
    "milestone-report.json": "975b291c46dcbb30e8f7cf097e32dcab3e805f88a32fd314900c937fca5c2af5",
    "milestone-criteria-authorization.json": "58c5ac9d8ca0b7f28b25aaa021e938c6c7f2b17f734cb177aaad7d4950bbdeb0",
    "milestone-acceptance.json": "7ec01b1dcf9605914fa81035df137aa2d26ff41a4a5d19720dde1e215346d3fd",
    "tolerance-proposal.json": "a8794bcaff088ac1e9638872337ecb336ce6f3156ce5f6f470cfe63c3e18d8cf",
    "tolerance-agreement.json": "1b79b1b28d8ee1d9aff349635a8aed677af6617f674eb2a2316ff5e6c214ceea",
    "milestone-golden-candidate.json": "36431859e9d9b23923254b610842928b3815be08f816c78868dca6ec40b780a7",
    "milestone-golden-replay.json": "2c28a4876778e74b8a297e79f6c0a630c6ae9a5534231c7cd2f055f762448185",
    "live-run.json": "2b69ae0de303fdda0ecf65f5e43e439d5cc40ea229abc642d1ee3db544e4bba4",
    "live-approval.json": "447bd2add6595f098c1f49288abef1c19a50619f2aa481f822ecd66521c99256",
}
# Exact reviewed ASTs, not permission to replace arbitrary attestor code.
AST_CHANGES = {
    "file_metadata": "a6a468fba696d8a633675cff87e236e826fd24ebe6b070ebe62658f0c763514f",
    "capture_serving_metadata": "74efb4fc764940f6e484bcee6e352ffdd193706bdc237939ee4a4732b6e500bb",
    "ServingAttestor.__init__": "b2fb98db13140c58762ef487acce78ba9668875ea355a5f6d9428baaf0c06671",
    "ServingAttestor.complete": "ebac1d6beee637943ee0ffd42bc9331ead1703187d0efad0373376f7e55734a6",
    "DumEGrootPolicyServer._predict_action_chunk": "ce202d9833cc6a397d379be20eec8a305d784a11718f813a42153dd9a032c035",
}


def present(workspace):
    return os.path.lexists(gate.evidence(workspace).workspace / RECORD)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def _ast(node):
    return ast.dump(node, include_attributes=False)


def validate_server_change(before, after):
    gate.require(sha(before) == BEFORE_SERVER and before != after, "wrong server transition baseline")
    old, new = ast.parse(before), ast.parse(after)
    old_top = {n.name: n for n in old.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    seen, body = set(), []
    for node in new.body:
        name = getattr(node, "name", None)
        if name in AST_CHANGES:
            gate.require(sha(_ast(node).encode()) == AST_CHANGES[name], "unreviewed metadata helper: " + name)
            gate.require(name not in old_top, "metadata helper replaces existing code")
            seen.add(name)
            continue
        if isinstance(node, ast.ClassDef) and name in old_top:
            methods = {n.name: n for n in old_top[name].body if isinstance(n, ast.FunctionDef)}
            for index, method in enumerate(node.body):
                key = name + "." + getattr(method, "name", "")
                if key in AST_CHANGES:
                    gate.require(sha(_ast(method).encode()) == AST_CHANGES[key], "unreviewed attestor/timing change: " + key)
                    gate.require(method.name in methods, "missing baseline attestor method")
                    node.body[index] = methods[method.name]
                    seen.add(key)
        body.append(node)
    new.body = body
    gate.require(seen == set(AST_CHANGES) and _ast(old) == _ast(new),
                 "model/inference AST changed outside the exact instrumentation fix")


def current_sources():
    return {name: sha(capture_bytes(ROOT / name)) for name in (SERVER, ACCEPTANCE, GOLDEN, THIS)}


def projected_profile(profile, server_sha):
    result = copy.deepcopy(profile)
    gate.require(result["owned_source_files"][SERVER] == BEFORE_SERVER, "profile server baseline changed")
    result["owned_source_files"][SERVER] = server_sha
    result["owned_source_fingerprint"] = gate.fingerprint_configuration(result["owned_source_files"])
    return result


def projected_profiles(profiles, server_sha):
    """Historical measured profiles stay intact; only serving source identity advances."""
    result = copy.deepcopy(profiles)
    observed = next(p["observed"] for p in profiles["profiles"]
                    if p["backend"] == "lerobot" and p["purpose"] == "operational")
    old = gate.operational_semantics(observed)
    gate.require(old == profiles["serving_configuration"], "baseline serving profile differs")
    result["serving_configuration"] = gate.operational_semantics(projected_profile(observed, server_sha))
    return result


def validate_smoke(ev, ref, baseline, server_sha):
    smoke = ev.json(ref)
    gate.require(smoke["schema_version"] == 1 and smoke["status"] == "complete"
                 and smoke["evidence_kind"] == "real_model", "real completed serving smoke required")
    case = ev.json("milestone-report.json")["cases"][0]
    gate.require(smoke["case"] == case and smoke["server_sha256"] == server_sha,
                 "smoke must use first selected case and changed server")
    proof = ev.json("milestone-report.json")["proofs"]["lerobot-operational"]
    worker = ev.json(proof["worker"])
    row = worker["cases"][worker["executed_cases"].index(case)]
    gate.require(smoke["baseline_case"] == row["evidence"], "smoke baseline capture changed")
    saved = ev.json(row["evidence"])
    gate.require(all(saved.get(k) == v for k, v in row.items() if k != "evidence"),
                 "smoke baseline case differs from manifest")
    old = ev.tensors(row["tensors"])["decoded"]
    decoded = ev.tensors(smoke["decoded"])
    gate.require(set(decoded) == {"decoded"} and decoded["decoded"].shape == old.shape == (16, 6)
                 and decoded["decoded"].dtype == old.dtype == np.float32
                 and np.isfinite(decoded["decoded"]).all() and np.array_equal(decoded["decoded"], old),
                 "fixed-seed full output differs from accepted capture")
    arrays, entry = load_case(Path(smoke["corpus"]), ev.json("input-lock.json"), case)
    observation = {name: float(value) for name, value in zip(gate.JOINT_ORDER, arrays["state"], strict=True)}
    observation.update(front=arrays["video_front"], wrist=arrays["video_wrist"], task=entry["instruction"])
    runtime = smoke["runtime"]
    request = runtime["request"]
    gate.require(request["observation_sha256"] == gate.observation_fingerprint(observation)
                 and request["output_sha256"] == gate.array_fingerprint(decoded["decoded"]),
                 "smoke request/input/output binding changed")
    expected = copy.deepcopy(ev.json("profiles.json")["serving_configuration"])
    expected["seed_policy"] = {"mode": "fixed", "seed": case["seed"]}
    gate.validate_runtime_attestation(runtime["attestation"], host=runtime["host"], request=request,
                                     expected_configuration=expected, now=smoke["ended_at"])
    gate.require(gate.timestamp(baseline.json("live-run.json")["ended_at"]) <
                 gate.timestamp(runtime["attestation"]["loaded_at"]) <=
                 gate.timestamp(smoke["started_at"]) <= gate.timestamp(request["started_at"]) <=
                 gate.timestamp(request["completed_at"]) <= gate.timestamp(smoke["ended_at"]),
                 "smoke must follow interrupted run and fresh changed-server load")
    return smoke


def validate(workspace, *, record=None):
    ev = gate.evidence(workspace)
    if record is None and RECORD in ev._validated:
        return ev._validated[RECORD]
    value = ev.json(RECORD) if record is None else record
    gate.require(value["schema_version"] == 1 and value["kind"] == "attestation_latency_only"
                 and value["status"] == "complete" and value["user_instruction"] == INSTRUCTION,
                 "explicit instrumentation-only restart required")
    gate.require(value["grants_approval"] is False, "transition cannot grant approval")
    baseline = gate.Evidence(Path(value["baseline_workspace"]))
    gate.require(baseline.workspace != ev.workspace, "fresh restart workspace required")
    for name, digest in BASELINE.items():
        gate.require(baseline.reference(name)["sha256"] == digest, "preserved baseline changed: " + name)
        if name not in ("profiles.json", "live-run.json", "live-approval.json"):
            gate.require(ev.reference(name)["sha256"] == digest, "historical evidence relabelled: " + name)
    sources = current_sources()
    gate.require(value["source_files"] == sources, "transition source bytes changed")
    before = ev.bytes(value["before_server"]["path"], value["before_server"]["sha256"])
    after = ev.bytes(value["after_server"]["path"], value["after_server"]["sha256"])
    gate.require(sha(after) == sources[SERVER], "changed server archive differs from current bytes")
    validate_server_change(before, after)
    gate.require(ev.json("profiles.json") == projected_profiles(baseline.json("profiles.json"), sources[SERVER]),
                 "profile changed beyond serving instrumentation identity")
    smoke = validate_smoke(ev, value["smoke"], baseline, sources[SERVER])
    gate.require(gate.timestamp(smoke["ended_at"]) <= gate.timestamp(value["ended_at"]),
                 "transition predates equivalence smoke")
    prior_sources = baseline.json("milestone-golden-candidate.json")["source_files"]
    for name, digest in prior_sources.items():
        if name not in (SERVER, ACCEPTANCE, GOLDEN):
            gate.require(sha(capture_bytes(ROOT / name)) == digest, "non-instrumentation source changed: " + name)
    result = {**value, "baseline_sources": prior_sources}
    if record is None:
        ev._validated[RECORD] = result
    return result


def require_source(ev, name, expected):
    if not present(ev):
        gate.require(sha(capture_bytes(ROOT / name)) == expected, "inference/capture source changed: " + name)
        return
    transition = validate(ev)
    gate.require(transition["baseline_sources"].get(name) == expected, "unbound historical source: " + name)


def serving_profile(ev, profile):
    return projected_profile(profile, validate(ev)["source_files"][SERVER]) if present(ev) else profile


def historical_sources(ev, current):
    return validate(ev)["baseline_sources"] if present(ev) else current


def historical_workspace(ev):
    return Path(validate(ev)["baseline_workspace"]).resolve() if present(ev) else ev.workspace


def release_fields(ev):
    if not present(ev):
        return {}
    value = validate(ev)
    return {"instrumentation_transition": ev.reference(RECORD), "ended_at": value["ended_at"],
            "continuation": "operator-requested restart from trial 1; predecessor remains unscored"}


def record_transition(workspace, *, baseline_workspace, before_server, after_server, smoke, user_instruction):
    """Parent supplies archived source refs and the completed real smoke witness."""
    ev = gate.evidence(workspace)
    value = {"schema_version": 1, "kind": "attestation_latency_only", "status": "complete",
             "baseline_workspace": str(Path(baseline_workspace).resolve()),
             "before_server": before_server, "after_server": after_server, "smoke": smoke,
             "source_files": current_sources(), "user_instruction": user_instruction,
             "grants_approval": False, "ended_at": now()}
    validate(ev, record=value)
    return write_evidence(ev.workspace, RECORD, value)


def prepare_restart_view(workspace, *, baseline_workspace, before_server):
    """Import immutable evidence; permit only a pre-existing runtime-live directory.

    Before-server bytes come from the parent's archive (e.g. git show
    0850e97:docker/lerobot-policy/server.py). No Git or model runs here.
    """
    import errno
    import shutil

    baseline = gate.Evidence(baseline_workspace)
    target = Path(workspace).resolve()
    gate.require(target != baseline.workspace and not target.is_relative_to(baseline.workspace),
                 "restart workspace must be separate from baseline")
    for name, digest in BASELINE.items():
        gate.require(baseline.reference(name)["sha256"] == digest, "preserved baseline changed: " + name)
    after = capture_bytes(ROOT / SERVER)
    validate_server_change(before_server, after)
    def diagnostic(path):
        return path.name.endswith(".json") and path.name.startswith(("serving-loaded-", "serving-attestation", "latency-"))

    if target.exists():
        gate.require(target.is_dir() and all(not p.is_symlink() and
                                            ((p.name in {"runtime-live", "runtime-smoke"} and p.is_dir()) or (diagnostic(p) and p.is_file()))
                                            for p in target.iterdir()),
                     "restart workspace already contains evidence; preserve it")
    else:
        target.mkdir(parents=True, exist_ok=False)
    excluded = {"profiles.json", "live-run.json", "live-approval.json", "release-review.json",
                "final-regression.json", "closeout.json", RECORD}
    for source in baseline.workspace.rglob("*"):
        relative = source.relative_to(baseline.workspace)
        if (relative.parts[0] in {"runtime", "runtime-live", "runtime-smoke", "preflights", "instrumentation"}
                or str(relative) in excluded or (len(relative.parts) == 1 and diagnostic(source))):
            continue
        gate.require(not source.is_symlink(), "restart import requires regular immutable evidence")
        if source.is_dir():
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            with source.open("rb") as src, destination.open("xb") as dst:
                shutil.copyfileobj(src, dst)
    write_evidence(target, "profiles.json", projected_profiles(baseline.json("profiles.json"), sha(after)))
    folder = target / "instrumentation"
    folder.mkdir(exist_ok=False)
    for name, data in (("before-server.py", before_server), ("after-server.py", after)):
        with (folder / name).open("xb") as stream:
            stream.write(data)
    ev = gate.Evidence(target)
    setup = {"baseline_workspace": str(baseline.workspace),
             "before_server": ev.reference("instrumentation/before-server.py"),
             "after_server": ev.reference("instrumentation/after-server.py")}
    write_evidence(target, "instrumentation/setup.json", setup)
    return setup


def collect_smoke(workspace, *, corpus, attestation, container,
                  endpoint="127.0.0.1:8080", checkpoint_mount="/checkpoints/model"):
    """Parent-invoked arm-free request to an already loaded fixed-seed server."""
    from policy.lerobot.session import LeRobotPolicySession
    from policy_guard.replay_contract import read_json, write_tensors

    ev = gate.Evidence(workspace)
    gate.require(endpoint.startswith("127.0.0.1:"), "smoke requires loopback server")
    gate.require(not (ev.workspace / "instrumentation/smoke.json").exists()
                 and not present(ev), "immutable smoke or transition already exists")
    setup = ev.json("instrumentation/setup.json")
    case = ev.json("milestone-report.json")["cases"][0]
    source = ev.json("milestone-report.json")["proofs"]["lerobot-operational"]
    worker = ev.json(source["worker"])
    original = worker["cases"][worker["executed_cases"].index(case)]
    arrays, entry = load_case(Path(corpus), ev.json("input-lock.json"), case)
    observation = {name: float(value) for name, value in zip(gate.JOINT_ORDER, arrays["state"], strict=True)}
    observation.update(front=arrays["video_front"], wrist=arrays["video_wrist"], task=entry["instruction"])
    expected = copy.deepcopy(ev.json("profiles.json")["serving_configuration"])
    expected["seed_policy"] = {"mode": "fixed", "seed": case["seed"]}
    loaded = read_json(capture_bytes(Path(attestation)))
    gate.require(loaded["status"] in ("loaded", "complete"), "loaded server required; smoke never loads a model")
    if loaded["status"] == "complete":
        gate.require(loaded["semantic_configuration"] == expected, "fixed-seed smoke serving configuration differs")
    started = now()
    client = LeRobotPolicySession(endpoint, max_attempts=1, handshake_max_attempts=1)
    try:
        client.probe_ready_or_raise()
        actions = client.infer(observation)
        gate.require(len(actions) == 16, "smoke must return exactly sixteen actions")
        decoded = np.stack([item.get_action().detach().cpu().float().numpy() for item in actions])
        observed = read_json(capture_bytes(Path(attestation)))
        request = {**observed["request"], "observation_sha256": gate.observation_fingerprint(observation),
                   "output_sha256": gate.array_fingerprint(decoded), "decoded_shape": list(decoded.shape),
                   "timestamp": actions[0].get_timestamp(), "timestep": actions[0].get_timestep()}
        host = gate.collect_runtime_host(observed, container, endpoint, checkpoint_mount)
    finally:
        client.close()
    smoke = {"schema_version": 1, "status": "complete", "evidence_kind": "real_model",
             "started_at": started, "ended_at": now(), "case": case,
             "server_sha256": sha(capture_bytes(ROOT / SERVER)), "baseline_case": original["evidence"],
             "corpus": str(Path(corpus).resolve()), "decoded": write_tensors(ev.workspace, {"decoded": decoded}),
             "runtime": {"attestation": observed, "host": host, "request": request}}
    reference = write_evidence(ev.workspace, "instrumentation/smoke.json", smoke)
    validate_smoke(ev, reference, gate.Evidence(setup["baseline_workspace"]), smoke["server_sha256"])
    transition = record_transition(ev, **setup, smoke=reference, user_instruction=INSTRUCTION)
    return {"smoke": reference, "transition": transition, "case": case, "exact_output_match": True}


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--baseline-workspace", type=Path, required=True)
    prepare.add_argument("--before-server", type=Path, required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--workspace", type=Path, required=True)
    smoke.add_argument("--corpus", type=Path, required=True)
    smoke.add_argument("--attestation", type=Path, required=True)
    smoke.add_argument("--container", required=True)
    smoke.add_argument("--endpoint", default="127.0.0.1:8080")
    smoke.add_argument("--checkpoint-mount", default="/checkpoints/model")
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare":
        args["before_server"] = capture_bytes(args["before_server"])
        result = prepare_restart_view(**args)
    else:
        result = collect_smoke(**args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
