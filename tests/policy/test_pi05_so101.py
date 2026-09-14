import numpy as np
import pytest

from embodiment.so_arm10x.mappings.galaxea import JOINTS
from policy.factory import make_policy_backend
from policy.configuration import deployment_for_profile
from functools import partial
Pi05SO101PolicyBackend = partial(make_policy_backend, deployment=deployment_for_profile("pi05-so101"))
from embodiment.so_arm10x.mappings.frames import to_arm_frame, to_model_frame
from policy.backends.lerobot.models.pi05 import JOINT_NAMES, validate_config, verify_checkpoint
from policy.backends.lerobot.transports.http import decode_request


def config():
    return {
        "type": "pi05", "action_feature_names": JOINT_NAMES.copy(),
        "use_relative_actions": False, "chunk_size": 50, "n_action_steps": 50,
        "empty_cameras": 0, "image_resolution": [224, 224],
        "input_features": {
            "observation.state": {"shape": [6]},
            "observation.images.wrist_left": {"shape": [3, 480, 640]},
            "observation.images.desk_view": {"shape": [3, 600, 800]}},
        "output_features": {"action": {"shape": [6]}},
        "normalization_mapping": {"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
    }


def test_exact_named_absolute_joint_contract_and_unmapped_base():
    validate_config(config())
    assert JOINT_NAMES == list(JOINTS)
    values = np.array([15, -90, 85, 40, -80, 25], dtype=np.float32)
    np.testing.assert_array_equal(to_model_frame(values, "pi05-so101"), values)
    np.testing.assert_array_equal(to_arm_frame(values, "pi05-so101"), values)
    with pytest.raises(ValueError, match="no verified"):
        to_arm_frame(np.zeros(6), "pi05-base")


@pytest.mark.parametrize("change", [
    lambda c: c.update(action_feature_names=list(reversed(JOINT_NAMES))),
    lambda c: c.update(use_relative_actions=True),
    lambda c: c.update(empty_cameras=1),
    lambda c: c["output_features"]["action"].update(shape=[32]),
    lambda c: c["normalization_mapping"].update(ACTION="IDENTITY"),
])
def test_wrong_output_semantics_refused(change):
    value = config()
    change(value)
    with pytest.raises(ValueError):
        validate_config(value)


def test_missing_or_wrong_artifacts_refused(tmp_path):
    with pytest.raises(ValueError, match="artifact"):
        verify_checkpoint(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="wrong-size"):
        verify_checkpoint(tmp_path)


def test_bridge_preserves_degrees_cameras_and_native_50_step_horizon(monkeypatch):
    backend = Pi05SO101PolicyBackend(language_instruction="Pick banana")
    values = [15., -90., 85., 40., -80., 25.]
    obs = dict(zip(JOINTS, values))
    obs.update(front=np.full((480, 640, 3), 20, np.uint8),
               wrist=np.full((480, 640, 3), 180, np.uint8))
    health = {"status": "ready", "fault": None, "profile": backend.profile.to_dict()}
    def request(path, data=None):
        if path == "/health":
            return health
        state, front, wrist, task, seed = decode_request(data)
        np.testing.assert_array_equal(state, values)
        assert np.all(front == 20) and np.all(wrist == 180)
        return {"actions": np.tile(state, (50, 1)).tolist(), "health": health,
                "seed": seed, "timings": {}, "physical_ready": False}
    monkeypatch.setattr(backend, "_request", request)
    actions = backend.get_action(obs)
    assert len(actions) == 50 and actions[0] == dict(zip(JOINTS, values))
    monkeypatch.setenv("DUME_ASYNC_INFERENCE", "1")
    with pytest.raises(ValueError, match="conflicts"):
        Pi05SO101PolicyBackend()


def test_factory_selection_is_explicit_and_uses_common_configuration(monkeypatch):
    from policy.factory import make_policy_backend
    monkeypatch.setenv("DUME_POLICY_BACKEND", "pi05-so101")
    backend = make_policy_backend(host="localhost", port=18081,
                                  camera_keys=["front", "wrist"],
                                  robot_state_keys=list(JOINTS), show_images=False)
    from policy.backends.lerobot.pi05_client import Pi05SO101PolicyBackend as Backend
    assert isinstance(backend, Backend)
    assert backend.endpoint == "http://127.0.0.1:18081"
    for kwargs in ({"host": "example.com"}, {"camera_keys": ["front"]},
                   {"robot_state_keys": ["joint1"]}):
        with pytest.raises(ValueError):
            make_policy_backend(**kwargs)
