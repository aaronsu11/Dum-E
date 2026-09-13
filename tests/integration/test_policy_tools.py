"""Portable recordings and trial authorization fail closed without opening hardware."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import numpy as np
import pytest
from embodiment.so_arm10x.observation import load_observation
from embodiment.so_arm10x import trial
from scripts.benchmark_policy import serving_identity


def test_recorded_observation_requires_named_numeric_rgb_arrays(tmp_path):
    path = tmp_path / "obs.npz"
    state = np.arange(6, dtype=np.float32)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    np.savez(path, state=state, front=image, wrist=image)
    loaded = load_observation(path)
    assert loaded["elbow_flex.pos"] == 2 and loaded["front"].dtype == np.uint8
    np.savez(path, state=state, front=image, wrist=np.zeros((3, 480, 640), np.uint8))
    with pytest.raises(ValueError, match="RGB"):
        load_observation(path)
    np.savez(path, state=np.array([float("nan")] * 6), front=image, wrist=image)
    with pytest.raises(ValueError):
        load_observation(path)
    np.savez(path, state=state, front=image, wrist=image, extra=np.zeros(1))
    with pytest.raises(ValueError):
        load_observation(path)


@pytest.mark.parametrize("change", ["absent", "expired", "future", "different_snapshot", "different_config", "unnamed"])
def test_approval_binds_present_operator_and_prepared_inputs(tmp_path, change):
    snapshot = {"protocol": {"chunks": 3}, "config_sha256": "known"}
    (tmp_path / "snapshot.json").write_text(json.dumps(snapshot))
    approval = {"operator_present": True, "approved": True, "operator": "Another operator",
                "user_authorization": "YES", "approved_at": datetime.now(timezone.utc).isoformat(),
                "snapshot_sha256": trial.digest(tmp_path / "snapshot.json")}
    (tmp_path / "approval.json").write_text(json.dumps(approval))
    trial.validate_approval(tmp_path, snapshot)
    if change == "absent": approval["operator_present"] = False
    if change == "unnamed": approval["operator"] = " "
    if change == "expired": approval["approved_at"] = (datetime.now(timezone.utc)-timedelta(hours=1)).isoformat()
    if change == "future": approval["approved_at"] = (datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()
    if change == "different_snapshot": approval["snapshot_sha256"] = "changed"
    if change == "different_config": snapshot["config_sha256"] = "changed"
    (tmp_path / "approval.json").write_text(json.dumps(approval))
    with pytest.raises(ValueError):
        trial.validate_approval(tmp_path, snapshot)


def test_latency_admission_refuses_unverified_observer_or_public_binding(monkeypatch):
    from scripts import benchmark_policy as bench
    info = {"State": {"Running": True}, "Config": {"Env": ["DUME_CHUNK_OBSERVER=off"]},
            "NetworkSettings": {"Ports": {"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "8080"}]}},
            "Path": "python3", "Args": ["/app/scripts/serve_observed_lerobot.py"]}
    monkeypatch.setattr(bench.subprocess, "check_output", lambda *a, **k: json.dumps([info]))
    with pytest.raises(ValueError, match="lightweight"):
        serving_identity("test-server", 8080)
    info["Config"]["Env"] = ["DUME_CHUNK_OBSERVER=lightweight"]
    info["NetworkSettings"]["Ports"]["8080/tcp"][0]["HostIp"] = "0.0.0.0"
    with pytest.raises(ValueError, match="loopback"):
        serving_identity("test-server", 8080)


def test_trial_snapshot_source_list_has_no_archived_helpers():
    for source in trial.SOURCES:
        assert (trial.ROOT / source).is_file(), source
    assert "embodiment/so_arm10x/safety.py" in trial.SOURCES
    assert "policy/backends/lerobot/http_client.py" in trial.SOURCES
