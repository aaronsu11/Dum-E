"""Regression protection for deployment composition and package boundaries."""
import ast
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from policy.configuration import PolicyDeployment, deployment_for_profile, load_deployment
from policy.factory import make_policy_backend

ROOT = Path(__file__).resolve().parents[2]


def test_runtimes_and_schedulers_do_not_import_embodiments():
    for directory in ("policy/backends", "policy/execution"):
        for path in (ROOT / directory).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("embodiment"), path
                elif isinstance(node, ast.Import):
                    assert not any(n.name.startswith("embodiment") for n in node.names), path
    assert not list((ROOT / "docker").rglob("*.py"))


@pytest.mark.parametrize("path", sorted((ROOT / "configs/deployments").glob("*.yaml")))
def test_checked_in_deployments_have_supported_capabilities(path):
    assert load_deployment(path).validate()


@pytest.mark.parametrize("deployment", [
    PolicyDeployment("galaxea_r1_pro", "galaxea", "g05", "g05-so101"),
    PolicyDeployment("so_arm10x", "lerobot", "pi05", "pi05-base", transport="http"),
    PolicyDeployment("so_arm10x", "lerobot", "molmoact2", "molmoact2-so101", "rtc", "http"),
    PolicyDeployment("so_arm10x", "galaxea", "g05", "g05-so101", "async"),
    PolicyDeployment("so_arm10x", "unknown", "groot", "groot-so101"),
])
def test_invalid_deployment_fails_before_client_construction(deployment):
    with pytest.raises(ValueError):
        make_policy_backend(deployment=deployment)


def test_deployment_typo_never_falls_back(tmp_path):
    path = tmp_path / "deployment.yaml"
    path.write_text("embodiment: so_arm10x\nbackned: lerobot\n")
    with pytest.raises(ValueError):
        load_deployment(path)


def test_http_client_accepts_an_injected_mapping_without_so101_names(monkeypatch):
    from policy.backends.lerobot.http_client import HTTPPolicyBackend
    from shared import IPolicyMapping

    class Mapping(IPolicyMapping):
        joint_names = tuple(f"axis_{i}" for i in range(6))
        camera_names = ("front", "wrist")

        def validate(self): pass
        def to_model(self, values): return np.asarray(values) + 10
        def to_arm(self, values): return np.asarray(values) - 10

    backend = HTTPPolicyBackend(profile="molmoact2-so101", mapping=Mapping())
    health = {"status": "ready", "fault": None, "profile": backend.profile.to_dict()}

    def request(path, data=None):
        if path == "/health":
            return health
        assert data["state"] == [10, 11, 12, 13, 14, 15]
        return {"health": health, "seed": data["seed"], "timings": {},
                "actions": [data["state"]] * 30}

    monkeypatch.setattr(backend, "_request", request)
    obs = {f"axis_{i}": i for i in range(6)}
    obs.update(front=np.zeros((480, 640, 3), np.uint8),
               wrist=np.ones((480, 640, 3), np.uint8))
    actions = backend.get_action(obs, "test mapping")
    assert len(actions) == 30
    assert actions[0] == {f"axis_{i}": float(i) for i in range(6)}


def test_native_client_import_does_not_load_hardware_stack():
    code = """import sys
from policy.backends.isaac_groot.client import Gr00tRobotInferenceClient
assert 'embodiment.so_arm10x.controller' not in sys.modules
assert 'lerobot.robots.so_follower' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_rtc_selection_is_explicit_and_keeps_prefix_api(monkeypatch):
    monkeypatch.delenv("DUME_ASYNC_INFERENCE", raising=False)
    backend = make_policy_backend(
        deployment=deployment_for_profile("pi05-so101", execution="rtc"))
    assert callable(backend.get_rtc_action)
    assert backend.mapping.joint_names[-1] == "gripper.pos"
