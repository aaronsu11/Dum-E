"""Hermetic native lifecycle evidence. No models, GPUs, hardware or real decisions."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import hashlib
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from policy_guard.replay_contract import (
    ReplayManifest, execute_cases, fingerprint_configuration as digest,
    read_json, write_evidence, write_tensors,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_parity_gate import fixture_workspace, rewrite, review, ts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
NATIVE_PATH = "Gr00tPolicy.get_action -> upstream _get_action -> client joint mapping"


def cli():
    assert importlib.util.find_spec("replay_native_golden"), (
        "Native candidate/review/promotion/verification must be executable commands"
    )
    return importlib.import_module("replay_native_golden")


def api():
    assert importlib.util.find_spec("policy_guard.golden"), (
        "Approved native references must have an immutable validating lifecycle"
    )
    return importlib.import_module("policy_guard.golden")


def setup(workspace, *, session="test-session"):
    fixture_workspace(workspace)
    rewrite(workspace, "session.json", lambda value: value.update(session_id=session))
    def configure(profiles):
        profiles["session"] = session
        native = profiles["profiles"][2]["observed"]
        native.update(
            path=NATIVE_PATH, parameter_dtypes=["torch.bfloat16"],
            compute_dtypes=["torch.bfloat16", "torch.float32"],
            noise_dtype="torch.bfloat16", device="cuda:0",
            rng_algorithm="torch.default_generator.cuda",
            effective_configuration={"processor": {"test_only": True}},
        )
    rewrite(workspace, "profiles.json", configure)
    profiles_digest = digest(read_json(workspace / "profiles.json"))
    for name in ("repeatability.json", "tolerance-proposal.json", "calibration.json"):
        rewrite(workspace, name, lambda value: value.update(profiles_fingerprint=profiles_digest, session=session))
    repeat_ref = {"path": "repeatability.json", "sha256": hashlib.sha256((workspace / "repeatability.json").read_bytes()).hexdigest()}
    rewrite(workspace, "tolerance-proposal.json", lambda value: value.update(repeatability=repeat_ref))
    review(workspace, "tolerances", 10)
    return read_json(workspace / "input-lock.json")


class HermeticWorker:
    """Explicit test injection at the same boundary as the isolated worker."""

    def __init__(self, *, offset=0, purpose="operational", fail_at=None, second=None):
        self.offset, self.purpose, self.fail_at = offset, purpose, fail_at
        self.calls = []
        self.second = second

    def __call__(self, args, cases, schedule_file, suffix):
        self.calls.append(copy.deepcopy(cases))
        workspace = args.workspace
        lock = read_json(workspace / "input-lock.json")
        profile = copy.deepcopy(read_json(workspace / "profiles.json")["profiles"][2]["observed"])
        profile["purpose"] = self.purpose
        report = ReplayManifest(
            session=read_json(workspace / "session.json")["session_id"], stage="operational",
            input_fingerprint=lock["fingerprint"], expected_cases=cases,
            started_at=ts(self.second + 1 if self.second is not None else (21 if "candidate" in suffix else 41)), evidence_kind="test_only",
        )
        seen = []
        def trace(key):
            if self.fail_at == len(seen):
                raise RuntimeError("TEST ONLY interrupted inference")
            seen.append(key)
            index = int(key["record"][7:11])
            value = np.float32(index / 100 + key["seed"] / 1000)
            # Distinct record/seed outputs, plus the complete action mapping.
            from replay_groot_native import NativeReplay
            class PolicyFixture:
                def get_action(self, observation):
                    assert observation["language"]["annotation.human.task_description"] == [["banana"]]
                    return {
                        "single_arm": np.full((1, 16, 5), value + self_offset, np.float32),
                        "gripper": np.full((1, 16, 1), value + 1 + self_offset, np.float32),
                    }, {}
            self_offset = self.offset
            adapter = object.__new__(NativeReplay)
            adapter.purpose, adapter.policy = "operational", PolicyFixture()
            arrays = {
                "state": np.arange(6, dtype=np.float32),
                "video_front": np.zeros((2, 2, 3), np.uint8),
                "video_wrist": np.ones((2, 2, 3), np.uint8),
            }
            tensors = {
                "raw": np.full((1, 40, 132), value + self.offset, np.float32),
                "noise": np.full((1, 40, 132), value, np.float32),
                "decoded": adapter.predict(arrays, lock["records"][index]),
                "rng_before": np.array([key["seed"]], np.uint8),
                "rng_after": np.array([key["seed"] + 1], np.uint8),
            }
            return profile, tensors, lock["records"][index]
        try:
            execute_cases(report, workspace, lock, cases, trace, "native-operational" + suffix)
            report.status = "complete"
        except Exception as exc:
            report.status = "failed"
            report.prerequisite_errors.append({"message": str(exc)})
        report.ended_at = ts(self.second + 2 if self.second is not None else (22 if "candidate" in suffix else 42))
        ref = write_evidence(workspace, f"workers/native-operational{suffix}.json", asdict(report))
        return {"status": report.status, "exit_code": 0 if report.status == "complete" else 1,
                "manifest": ref, "evidence_kind": "test_only"}, asdict(report)


def snapshot(args):
    lock = read_json(args.workspace / "input-lock.json")
    return {
        "input_fingerprint": lock["fingerprint"],
        "source_files": {"TEST_ONLY.py": digest("test source")},
        "calibration_sha256": digest("test calibration"),
        "image_digest": read_json(args.workspace / "profiles.json")["profiles"][2]["observed"]["image_digest"],
    }


def run(workspace, command, *, worker=None, second=20, prompt=None, **kwargs):
    return cli().main(
        [command, "--workspace", str(workspace), *kwargs.pop("extra", [])],
        worker=worker, probe=kwargs.pop("probe", snapshot), test_only=True,
        clock=lambda: ts(second), prompt=prompt, **kwargs,
    )


def generate(workspace, worker=None, **kwargs):
    return run(workspace, "generate", worker=worker or HermeticWorker(),
               extra=["--reason", "TEST ONLY initial intentional baseline"], **kwargs)


def approve(workspace, answers=None, second=30):
    answers = iter(answers or ["TEST ONLY Operator", "TEST ONLY intentional review", "approve"])
    candidate = api().Evidence(workspace, test_only=True).reference("golden-candidate.json")
    return run(workspace, "review", second=second, prompt=lambda _: next(answers),
               extra=["--candidate-sha256", candidate["sha256"], *previous_args(workspace)])


def promote(workspace, second=31):
    candidate = api().Evidence(workspace, test_only=True).reference("golden-candidate.json")
    return run(workspace, "promote", second=second, extra=["--candidate-sha256", candidate["sha256"], *previous_args(workspace)])


def lifecycle(workspace):
    setup(workspace)
    assert generate(workspace) == 0
    assert approve(workspace) == 0
    assert promote(workspace) == 0


def test_command_worker_review_promotion_verification_lifecycle(tmp_path):
    lock = setup(tmp_path)
    worker = HermeticWorker()
    assert generate(tmp_path, worker) == 0
    assert worker.calls == [lock["schedule"]]
    assert not (tmp_path / "golden-approval.json").exists()
    assert not (tmp_path / "golden-manifest.json").exists()
    assert approve(tmp_path) == 0
    assert promote(tmp_path) == 0
    before = {name: (tmp_path / name).read_bytes() for name in
              ("golden-candidate.json", "golden-approval.json", "golden-manifest.json")}
    replay = HermeticWorker()
    assert run(tmp_path, "verify", worker=replay, second=40, extra=["--stage", "pre-live"]) == 0
    assert replay.calls == [lock["schedule"]]
    assert run(tmp_path, "check", second=50, extra=["--stage", "pre-live"]) == 0
    assert before == {name: (tmp_path / name).read_bytes() for name in before}
    decision = read_json(tmp_path / "golden-approval.json")
    assert decision["decision_type"] == "golden" and decision["evidence_kind"] == "test_only"
    assert (tmp_path / f"decisions/{digest(decision)}.json").read_bytes() == before["golden-approval.json"]
    manifest = api().validate_golden(api().Evidence(tmp_path, test_only=True))
    assert manifest["candidate"]["sha256"] == decision["subject_digest"]
    with pytest.raises(ValueError, match="test_only|real_model"):
        api().validate_golden(tmp_path)


@pytest.mark.parametrize("problem", ["diagnostic", "missing-checkpoint", "unapproved", "overwrite",
                                    "candidate-tamper", "fabricated-decision", "drift"])
def test_lifecycle_fails_closed(tmp_path, problem):
    setup(tmp_path)
    if problem == "diagnostic":
        assert generate(tmp_path, HermeticWorker(purpose="diagnostic")) == 1
        assert not (tmp_path / "golden-manifest.json").exists()
        return
    if problem == "missing-checkpoint":
        def missing(args):
            raise FileNotFoundError("TEST ONLY missing checkpoint")
        worker = HermeticWorker()
        assert generate(tmp_path, worker, probe=missing) == 2
        assert not worker.calls
        return
    assert generate(tmp_path) == 0
    if problem == "unapproved":
        assert promote(tmp_path) == 2
        assert run(tmp_path, "verify", worker=HermeticWorker(),
                   extra=["--stage", "pre-live"]) == 2
        assert not (tmp_path / "golden-manifest.json").exists()
        return
    assert approve(tmp_path) == 0
    if problem == "candidate-tamper":
        rewrite(tmp_path, "golden-candidate.json", lambda c: c.update(reason="tampered"))
        assert promote(tmp_path) == 1
        return
    if problem == "fabricated-decision":
        rewrite(tmp_path, "golden-approval.json", lambda c: c.update(operator="fabricated"))
        assert promote(tmp_path) == 1
        return
    assert promote(tmp_path) == 0
    before = (tmp_path / "golden-manifest.json").read_bytes()
    if problem == "overwrite":
        assert promote(tmp_path) == 1
        assert generate(tmp_path) == 1
    else:
        assert run(tmp_path, "verify", worker=HermeticWorker(offset=1), second=40,
                   extra=["--stage", "pre-live"]) == 1
        assert read_json(tmp_path / "golden-replay.json")["status"] == "failed"
    assert (tmp_path / "golden-manifest.json").read_bytes() == before


def test_lifecycle_review_rejects_wrong_digest_and_noninteractive_input(tmp_path):
    setup(tmp_path)
    assert generate(tmp_path) == 0
    assert run(tmp_path, "review", second=30, extra=["--candidate-sha256", "0" * 64]) == 1
    ref = api().Evidence(tmp_path, test_only=True).reference("golden-candidate.json")
    assert run(tmp_path, "review", second=30, extra=["--candidate-sha256", ref["sha256"]]) == 2
    assert not (tmp_path / "golden-approval.json").exists()


@pytest.fixture(scope="module")
def approved_template(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("TEST_ONLY_native_approved")
    lifecycle(workspace)
    assert run(workspace, "verify", worker=HermeticWorker(), second=40,
               extra=["--stage", "pre-live"]) == 0
    return workspace


@pytest.fixture
def approved_copy(approved_template, tmp_path):
    shutil.copytree(approved_template, tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_review_actual_operational_attention_is_not_diagnostic_sdpa():
    from test_parity_gate import observed
    profile = observed("native", "operational")
    profile.update(path=NATIVE_PATH, parameter_dtypes=["torch.bfloat16"],
                   attention=["flash_attention_2"], sdpa_calls=132,
                   effective_configuration={"processor": {"test_only": True}})
    # Mirrors the preserved measured profile; this must not change model attention.
    api().validate_native_profile(profile)


@pytest.mark.parametrize("external", [False, True])
def test_review_actual_calibration_format_and_reference(tmp_path, external):
    from scripts import pose_sweep_units_probe as probe
    from policy_guard.replay_contract import canonical
    calibration_path, statistics_path = tmp_path / "calibration-input.json", tmp_path / "statistics.json"
    calibration_path.write_bytes(canonical(probe.CALIBRATION_TICK_RANGES))
    statistics_path.write_bytes(canonical({"new_embodiment": {
        "state": probe.CHECKPOINT_STATE_STATS, "action": probe.CHECKPOINT_ACTION_STATS}}))
    record = probe.derive_current_calibration(calibration_path, statistics_path)
    record.update(session_id="TEST_ONLY_prior", started_at=ts(1), ended_at=ts(2))
    prior = tmp_path / "prior"
    ref = write_evidence(prior if external else tmp_path, "calibration.json", record)
    session = {"schema_version": 1, "session_id": "TEST_ONLY_prior"}
    if external:
        session.update(session_id="TEST_ONLY_successor", calibration_reference={
            "path": str(prior / ref["path"]), "sha256": ref["sha256"]})
    write_evidence(tmp_path, "session.json", session)
    assert callable(getattr(api(), "calibration_identity", None)), "native freshness must use canonical Plan02 resolver"
    assert api().calibration_identity(api().Evidence(tmp_path))["sha256"] == record["calibration"]["sha256"]
    calibration_path.write_bytes(canonical({}))
    with pytest.raises(ValueError, match="changed"):
        api().calibration_identity(api().Evidence(tmp_path))


def test_review_check_cannot_accept_replay_without_its_worker(approved_copy):
    candidate = read_json(approved_copy / "golden-candidate.json")
    def forge(record):
        record["tensors"] = candidate["tensors"]
        record.pop("worker")
    rewrite(approved_copy, "golden-replay.json", forge)
    assert run(approved_copy, "check", second=50, extra=["--stage", "pre-live"]) == 1


@pytest.mark.parametrize("field", ["source_files", "calibration_sha256", "image_digest", "input_fingerprint"])
def test_check_invalidates_changed_current_identity(approved_copy, field):
    def changed(args):
        value = snapshot(args)
        value[field] = {"TEST_ONLY.py": digest("new source")} if field == "source_files" else digest("new")
        return value
    assert run(approved_copy, "check", second=50, probe=changed, extra=["--stage", "pre-live"]) == 1


def test_changed_source_still_replays_all_cases_and_retains_failure(approved_copy):
    worker = HermeticWorker(second=60)
    def changed(args):
        value = snapshot(args)
        value["source_files"]["TEST_ONLY.py"] = digest("new source")
        return value
    # A new stage lets the original passing replay remain immutable.
    ev = api().Evidence(approved_copy, test_only=True)
    write_evidence(approved_copy, "live-run.json", {
        **ev.identity(), "schema_version": 1, "evidence_kind": "test_only", "status": "complete",
        "started_at": ts(45), "ended_at": ts(55), "TEST_ONLY": True,
    })
    assert run(approved_copy, "verify", second=60, probe=changed, worker=worker,
               extra=["--stage", "final"]) == 1
    assert len(worker.calls) == 1 and len(worker.calls[0]) == 600
    final = read_json(approved_copy / "final-regression.json")
    assert final["status"] == "failed" and final["identity_changed"] is True
    assert final["comparison"]["passed"] is True


@pytest.mark.parametrize("problem", ["missing-live", "live-after-replay"])
def test_final_requires_exact_prior_live_evidence(approved_copy, problem):
    if problem == "live-after-replay":
        ev = api().Evidence(approved_copy, test_only=True)
        write_evidence(approved_copy, "live-run.json", {
            **ev.identity(), "schema_version": 1, "evidence_kind": "test_only", "status": "complete",
            "started_at": ts(45), "ended_at": ts(70), "TEST_ONLY": True,
        })
    worker = HermeticWorker(second=60)
    result = run(approved_copy, "verify", second=60, worker=worker, extra=["--stage", "final"])
    assert result == (2 if problem == "missing-live" else 1)
    assert not worker.calls


def test_final_links_live_evidence_and_is_immutable(approved_copy):
    ev = api().Evidence(approved_copy, test_only=True)
    live_ref = write_evidence(approved_copy, "live-run.json", {
        **ev.identity(), "schema_version": 1, "evidence_kind": "test_only", "status": "complete",
        "started_at": ts(45), "ended_at": ts(55), "TEST_ONLY": True,
    })
    assert run(approved_copy, "verify", second=60, worker=HermeticWorker(second=60),
               extra=["--stage", "final"]) == 0
    final = read_json(approved_copy / "final-regression.json")
    assert final["live_run"] == live_ref and len(final["cases"]) == 600
    assert run(approved_copy, "check", second=80, extra=["--stage", "final"]) == 0
    assert run(approved_copy, "verify", second=90, worker=HermeticWorker(second=90),
               extra=["--stage", "final"]) == 1


@pytest.mark.parametrize("corruption", ["599", "duplicate", "reordered", "raw-cropped", "decoded-cropped",
                                        "noise-drift", "nan", "path-escape", "missing-case", "rng-missing"])
def test_full_coverage_and_replay_witness_cannot_be_forged(approved_copy, corruption):
    ev = api().Evidence(approved_copy, test_only=True)
    replay = ev.json("golden-replay.json")
    if corruption in ("599", "duplicate", "reordered"):
        if corruption == "599":
            replay["cases"].pop()
        elif corruption == "duplicate":
            replay["cases"][-1] = replay["cases"][0]
        else:
            replay["cases"].reverse()
    elif corruption in ("raw-cropped", "decoded-cropped", "noise-drift", "nan"):
        arrays = {k: v.copy() for k, v in ev.tensors(replay["tensors"]).items()}
        if corruption == "raw-cropped":
            arrays["raw"] = arrays["raw"][:, :16, :]
        elif corruption == "decoded-cropped":
            arrays["decoded"] = arrays["decoded"][:, :, :5]
        elif corruption == "noise-drift":
            arrays["noise"][0, 0, 0] += 1
        else:
            arrays["decoded"][0, 0, 0] = np.nan
        if corruption == "nan":
            path = approved_copy / "bad.npz"
            np.savez(path, **arrays)
            replay["tensors"] = {"path": "bad.npz", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()}}
        else:
            replay["tensors"] = write_tensors(approved_copy, arrays)
    elif corruption == "path-escape":
        replay["worker"]["manifest"]["path"] = "../outside.json"
    else:
        worker = ev.json(replay["worker"]["manifest"])
        if corruption == "missing-case":
            worker["cases"].pop()
        else:
            worker["cases"][0]["tensors"]["arrays"].pop("rng_before")
        replay["worker"]["manifest"] = write_evidence(approved_copy, "bad-worker.json", worker)
    rewrite(approved_copy, "golden-replay.json", lambda r: r.update(replay))
    assert run(approved_copy, "check", second=50, extra=["--stage", "pre-live"]) == 1


def test_interrupted_candidate_retains_partial_archive_and_cannot_promote(tmp_path):
    setup(tmp_path)
    assert generate(tmp_path, HermeticWorker(fail_at=1)) == 1
    worker = read_json(tmp_path / "workers/native-operational-golden-candidate.json")
    assert len(worker["cases"]) == 1 and len(worker["failure_ledger"]) == 1
    assert approve(tmp_path) == 1
    assert not (tmp_path / "golden-manifest.json").exists()


def test_production_cli_missing_environment_never_passes(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/replay_native_golden.py"),
                             "verify", "--workspace", str(tmp_path), "--stage", "pre-live"],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2 and json.loads(result.stdout)["status"] == "not_run"
    assert list(tmp_path.iterdir()) == []


def test_injected_worker_cannot_create_real_evidence(tmp_path):
    worker = HermeticWorker()
    assert cli().main(["generate", "--workspace", str(tmp_path), "--reason", "test"],
                      worker=worker) == 1
    assert not worker.calls



def previous_args(workspace):
    ev = api().Evidence(workspace, test_only=True)
    candidate = ev.json("golden-candidate.json")
    if candidate.get("previous") is None:
        return []
    return ["--previous-sha256", ev.json(candidate["previous"])["manifest"]["sha256"]]


def test_reviewed_replacement_preserves_both_generations(approved_template, tmp_path):
    setup(tmp_path, session="TEST_ONLY_successor")
    old_path = approved_template / "golden-manifest.json"
    old_bytes = old_path.read_bytes()
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    worker = HermeticWorker(second=60, offset=0.1)
    assert run(tmp_path, "generate", worker=worker, second=60, extra=[
        "--reason", "TEST ONLY intentional model change", "--previous-manifest", str(old_path),
        "--previous-sha256", old_sha]) == 0
    candidate = read_json(tmp_path / "golden-candidate.json")
    assert candidate["previous_comparison"]["passed"] is False
    assert not (tmp_path / "golden-manifest.json").exists()
    assert approve(tmp_path, second=70) == 0
    assert promote(tmp_path, second=71) == 0
    manifest = api().validate_golden(api().Evidence(tmp_path, test_only=True))
    ev = api().Evidence(tmp_path, test_only=True)
    previous = ev.json(manifest["previous"])
    assert (tmp_path / previous["manifest"]["path"]).read_bytes() == old_bytes
    assert old_path.read_bytes() == old_bytes
    decision = read_json(tmp_path / "golden-approval.json")
    assert manifest["previous"] in decision["evidence_reviewed"]
    # Retaining only an approved status without its historical archive cannot pass.
    prior_approval = tmp_path / previous["approval"]["path"]
    prior_approval.write_text("{}")
    with pytest.raises((ValueError, KeyError)):
        api().validate_golden(api().Evidence(tmp_path, test_only=True))


def test_replacement_requires_explicit_previous_digest(approved_template, tmp_path):
    setup(tmp_path, session="TEST_ONLY_successor")
    worker = HermeticWorker(second=60)
    assert run(tmp_path, "generate", worker=worker, second=60, extra=[
        "--reason", "TEST ONLY intentional change", "--previous-manifest",
        str(approved_template / "golden-manifest.json"), "--previous-sha256", "0" * 64]) == 1
    assert not worker.calls


def test_rejected_review_is_retained_and_cannot_promote(tmp_path):
    setup(tmp_path)
    assert generate(tmp_path) == 0
    assert approve(tmp_path, ["TEST ONLY Operator", "Reject unintended drift", "reject"]) == 1
    assert promote(tmp_path) == 1
    assert read_json(tmp_path / "golden-approval.json")["decision"] == "rejected"
    assert not (tmp_path / "golden-manifest.json").exists()
