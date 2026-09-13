import numpy as np
import pytest

from policy.galaxea.modalities import JOINTS
from policy.molmo_backend import MolmoPolicyBackend
from policy_lab.protocol import decode_request


def observation():
    obs = dict(zip(JOINTS, [10., -20., 30., -40., 50., 60.]))
    obs.update(front=np.full((480, 640, 3), 40, np.uint8),
               wrist=np.full((480, 640, 3), 180, np.uint8))
    return obs


def install_server(monkeypatch, backend, change=None):
    health = {"status": "ready", "fault": None, "profile": backend.profile.to_dict()}
    seen = []

    def request(path, data=None):
        if path == "/health":
            return health
        state, front, wrist, task, seed = decode_request(data)
        seen.append((state, front, wrist))
        reply = {"health": health, "actions": np.tile(state, (30, 1)).tolist(),
                 "seed": seed, "timings": {}, "physical_ready": False}
        if change:
            change(reply)
        return reply
    monkeypatch.setattr(backend, "_request", request)
    return seen


def test_named_frame_camera_roles_and_full_horizon(monkeypatch):
    backend = MolmoPolicyBackend(language_instruction="Pick banana")
    seen = install_server(monkeypatch, backend)
    actions = backend.get_action(observation())
    np.testing.assert_array_equal(seen[0][0], [10, 110, 120, -40, 50, 60])
    assert np.all(seen[0][1] == 40) and np.all(seen[0][2] == 180)
    assert len(actions) == 30
    for action in actions:
        assert action == {k: observation()[k] for k in JOINTS}
    assert backend.last_metadata["server_physical_ready"] is False


@pytest.mark.parametrize("change", [
    lambda r: r.update(actions=[[0.] * 6] * 29),
    lambda r: r.update(actions=[[float("nan")] * 6] * 30),
    lambda r: r.update(seed=0),
    lambda r: r["health"]["profile"].update(revision="wrong"),
    lambda r: r["health"].update(status="failed"),
])
def test_invalid_reply_refused_before_actions_return(monkeypatch, change):
    backend = MolmoPolicyBackend(language_instruction="Pick banana")
    install_server(monkeypatch, backend, change)
    with pytest.raises(ValueError):
        backend.get_action(observation())


def test_missing_camera_async_and_unreachable_server_refused(monkeypatch):
    backend = MolmoPolicyBackend(language_instruction="Pick banana")
    obs = observation()
    del obs["wrist"]
    with pytest.raises(KeyError):
        backend.get_action(obs)
    def unavailable(*args, **kwargs):
        raise TimeoutError("server unreachable")
    monkeypatch.setattr(backend, "_request", unavailable)
    assert not backend.ping()
    with pytest.raises(TimeoutError):
        backend.get_action(observation())
    monkeypatch.setenv("DUME_ASYNC_INFERENCE", "1")
    with pytest.raises(ValueError, match="synchronous"):
        MolmoPolicyBackend()
