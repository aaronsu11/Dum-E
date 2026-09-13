import threading
from contextlib import contextmanager
import msgpack
import numpy as np
import pytest
from websockets.sync.server import serve

from policy.factory import make_policy_backend
from policy.galaxea.codec import packb, unpackb
from policy.galaxea.modalities import (
    JOINTS, CHECKPOINT_REVISION, arm_to_model, model_to_arm,
    make_observation, decode_action, validate_codec_presence,
)
from policy.galaxea.session import GalaxeaSession


def observation():
    values = [10., -30., 25., 50., -12., 20.]
    return {**dict(zip(JOINTS, values)),
            "front": np.full((480, 640, 3), 9, np.uint8),
            "wrist": np.full((480, 640, 3), 19, np.uint8)}


def test_joint_frame_and_camera_roles():
    obs = observation()
    native = make_observation(obs, "pick banana", seed=123)
    assert native["state"]["right_arm"].tolist() == [10, 120, 115, 50, -12, 20]
    assert np.array_equal(model_to_arm(native["state"]["right_arm"]), list(obs.values())[:6])
    assert np.all(native["images"]["exterior"] == 9)
    assert np.all(native["images"]["wrist_right"] == 19)
    assert not native["images"]["wrist_left"].any()
    del obs["front"]
    with pytest.raises(KeyError):
        make_observation(obs, "pick banana")


def test_reject_wrong_action_group():
    validate_codec_presence({"left_control", "left_gripper", "right_gripper"})
    for absent in (set(), {"right_control"}, {"left_control", "right_gripper"}):
        with pytest.raises(ValueError):
            validate_codec_presence(absent)
    for action in ({}, {"left_arm": np.zeros(6)}, {"right_control": np.zeros(9)},
                   {"right_arm": np.zeros(6), "left_arm": np.zeros(6)},
                   {"right_arm": np.zeros(27)}, {"right_arm": [float("nan")] * 6}):
        with pytest.raises(ValueError):
            decode_action(action)


def test_wire_roundtrip_and_unsafe_array_rejection():
    raw = make_observation(observation(), "banana")
    decoded = unpackb(packb(raw))
    assert np.array_equal(decoded["images"]["exterior"], raw["images"]["exterior"])
    for dtype, shape, data in [("O", [1], b"\0"*8), ("f4", [10**10], b""),
                               ("f4", [-1], b""), ("f4", [6], b"")]:
        bad = msgpack.packb({"__ndarray__": True, "dtype": dtype, "shape": shape, "data": data})
        with pytest.raises(ValueError):
            unpackb(bad)


@contextmanager
def fake_server(*, fail=False, revision=CHECKPOINT_REVISION):
    seen = []
    health = {"profile": "g05-so101", "checkpoint_revision": revision, "status": "ready"}

    def handler(ws):
        ws.send(packb({"action_steps": 32, "health": health}))
        index = 0
        state = None
        for message in ws:
            raw = unpackb(message)
            if raw == {"__health__": True}:
                ws.send(packb(health))
            elif raw == {"__reset__": True}:
                index = 0
                seen.append("reset")
                ws.send(packb({"__reset__": True}))
            elif fail:
                ws.send(packb({"error": {"code": 500, "message": "controlled failure"}}))
            else:
                if index == 0:
                    state = raw["state"]["right_arm"]
                    seen.append(raw)
                else:
                    assert raw == {}
                ws.send(packb({"action": {"right_arm": state.copy()}, "need_obs": index == 31}))
                index += 1

    server = serve(handler, "127.0.0.1", 0, max_size=16 * 1024**2)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.socket.getsockname()[1], seen
    finally:
        server.shutdown()
        thread.join(5)


def test_real_websocket_factory_full_chunk_reset_and_retarget(monkeypatch):
    monkeypatch.setenv("DUME_POLICY_BACKEND", "galaxea")
    monkeypatch.setenv("DUME_ASYNC_INFERENCE", "0")
    with fake_server() as (port, seen):
        backend = make_policy_backend(port=port, language_instruction="banana")
        try:
            assert backend.ping()
            actions = backend.get_action(observation(), seed=123)
            assert len(actions) == 32
            assert actions[0] == {key: observation()[key] for key in JOINTS}
            backend.set_lang_instruction("apple")
            assert backend.language_instruction == "apple"
            backend.get_action(observation(), seed=123)
            requests = [v for v in seen if isinstance(v, dict)]
            assert [v["task"] for v in requests] == ["banana", "apple"]
            assert requests[0]["_dume_seed"] == 123
        finally:
            backend.close()
            backend.close()


def test_server_error_closes_session_and_identity_mismatch_refuses():
    with fake_server(fail=True) as (port, _):
        session = GalaxeaSession(f"ws://127.0.0.1:{port}")
        with pytest.raises(RuntimeError, match="controlled failure"):
            session.request(make_observation(observation(), "banana"))
        assert session.socket is None
    with fake_server(revision="wrong") as (port, _):
        session = GalaxeaSession(f"ws://127.0.0.1:{port}")
        with pytest.raises(ValueError, match="identity"):
            session.connect()
        assert session.socket is None


def test_async_mode_is_explicitly_refused(monkeypatch):
    monkeypatch.setenv("DUME_POLICY_BACKEND", "galaxea")
    monkeypatch.setenv("DUME_ASYNC_INFERENCE", "1")
    with pytest.raises(ValueError, match="async"):
        make_policy_backend()
