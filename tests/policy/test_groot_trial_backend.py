import json
import numpy as np
import pytest

from embodiment.so_arm10x.mappings.galaxea import JOINTS
from policy.factory import make_policy_backend
from policy.configuration import deployment_for_profile
from functools import partial
GrootTrialBackend = partial(make_policy_backend, deployment=deployment_for_profile("groot-so101"))
from policy.backends.lerobot.transports.http import decode_request


def make_backend(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        key.removesuffix(".pos"): {"range_min": 900 + index * 50, "range_max": 3100 - index * 50}
        for index, key in enumerate(JOINTS)}))
    return GrootTrialBackend(calibration_path=path, language_instruction="Pick banana")


def test_calibrated_roundtrip_native_horizon_and_camera_roles(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    # The upper calibrated endpoint is +100 in checkpoint space; gripper is identity.
    state = backend.mapping.scales * np.array([100, -100, 50, -50, 25, 40])
    obs = dict(zip(JOINTS, state))
    obs.update(front=np.full((480, 640, 3), 20, np.uint8),
               wrist=np.full((480, 640, 3), 180, np.uint8))
    health = {"status": "ready", "fault": None, "profile": backend.profile.to_dict()}
    def request(path, data=None):
        if path == "/health":
            return health
        normalized, front, wrist, task, seed = decode_request(data)
        np.testing.assert_allclose(normalized, [100, -100, 50, -50, 25, 40], atol=1e-5)
        assert np.all(front == 20) and np.all(wrist == 180)
        return {"health": health, "actions": np.tile(normalized, (16, 1)).tolist(),
                "seed": seed, "timings": {}, "physical_ready": False}
    monkeypatch.setattr(backend, "_request", request)
    actions = backend.get_action(obs)
    assert len(actions) == 16
    np.testing.assert_allclose(list(actions[-1].values()), state, atol=1e-5)
    assert backend.last_metadata["calibration_sha256"] == backend.mapping.calibration_sha256


def test_changed_calibration_and_async_refused(tmp_path, monkeypatch):
    backend = make_backend(tmp_path)
    backend.mapping.calibration_path.write_text("{}")
    with pytest.raises(ValueError, match="calibration changed"):
        backend.get_action({})
    monkeypatch.setenv("DUME_ASYNC_INFERENCE", "1")
    with pytest.raises(ValueError, match="conflicts"):
        GrootTrialBackend(calibration_path=backend.mapping.calibration_path)
