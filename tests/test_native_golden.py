"""Hermetic native lifecycle evidence. No models, GPUs, hardware or real decisions."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import hashlib
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


def setup(workspace):
    fixture_workspace(workspace)
    def configure(profiles):
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
        rewrite(workspace, name, lambda value: value.update(profiles_fingerprint=profiles_digest))
    repeat_ref = {"path": "repeatability.json", "sha256": hashlib.sha256((workspace / "repeatability.json").read_bytes()).hexdigest()}
    rewrite(workspace, "tolerance-proposal.json", lambda value: value.update(repeatability=repeat_ref))
    review(workspace, "tolerances", 10)
    return read_json(workspace / "input-lock.json")


class HermeticWorker:
    """Explicit test injection at the same boundary as the isolated worker."""

    def __init__(self, *, offset=0, purpose="operational", fail_at=None):
        self.offset, self.purpose, self.fail_at = offset, purpose, fail_at
        self.calls = []

    def __call__(self, args, cases, schedule_file, suffix):
        self.calls.append(copy.deepcopy(cases))
        workspace = args.workspace
        lock = read_json(workspace / "input-lock.json")
        profile = copy.deepcopy(read_json(workspace / "profiles.json")["profiles"][2]["observed"])
        profile["purpose"] = self.purpose
        report = ReplayManifest(
            session="test-session", stage="operational",
            input_fingerprint=lock["fingerprint"], expected_cases=cases,
            started_at=ts(21 if "candidate" in suffix else 41), evidence_kind="test_only",
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
        report.ended_at = ts(22 if "candidate" in suffix else 42)
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


def approve(workspace, answers=None):
    answers = iter(answers or ["TEST ONLY Operator", "TEST ONLY intentional review", "approve"])
    candidate = api().Evidence(workspace, test_only=True).reference("golden-candidate.json")
    return run(workspace, "review", second=30, prompt=lambda _: next(answers),
               extra=["--candidate-sha256", candidate["sha256"]])


def promote(workspace):
    candidate = api().Evidence(workspace, test_only=True).reference("golden-candidate.json")
    return run(workspace, "promote", second=31, extra=["--candidate-sha256", candidate["sha256"]])


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
