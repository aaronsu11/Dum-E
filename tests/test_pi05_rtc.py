import threading
import time

import numpy as np
import pytest

from policy.galaxea.modalities import JOINTS
from policy.pi05_backend import Pi05SO101PolicyBackend
from policy_guard.rtc_trial import BoundedRTCQueue, accept_reply, run_rtc_trial
from policy_lab.protocol import decode_rtc_request, prefix_digest


ORIGIN = np.array([0., 0., 0., 0., 0., 50.])
LIMITS = (np.array([-180.] * 5 + [0.]), np.array([180.] * 5 + [100.]))


def test_guidance_contains_bounded_queue_and_replacement_preserves_slew():
    queue = BoundedRTCQueue(ORIGIN, LIMITS)
    raw = np.full((50, 6), 1000.)
    queue.install(raw, ORIGIN)
    for _ in range(25):
        queue.consume(ORIGIN)
    prefix = queue.prefix()
    assert np.max(abs(prefix - ORIGIN)) <= 3.75
    assert not np.array_equal(prefix, raw[:25])
    before = queue.previous.copy()
    queue.install(-raw, ORIGIN, elapsed=9)
    assert len(queue.items) == 41
    after = queue.consume(ORIGIN)["command"]
    assert np.max(abs(after - before)) <= 0.25


def test_entire_chunk_checked_atomically_and_changed_feedback_stops():
    queue = BoundedRTCQueue(ORIGIN, LIMITS)
    raw = np.tile(ORIGIN, (50, 1))
    queue.install(raw, ORIGIN)
    bad = raw.copy()
    bad[-1, 0] = np.nan
    with pytest.raises(ValueError, match="complete"):
        queue.install(bad, ORIGIN)
    assert len(queue.items) == 50
    shifted = ORIGIN.copy()
    shifted[0] = 4.
    with pytest.raises(RuntimeError, match="no longer safe"):
        queue.consume(shifted)
    assert len(queue.items) == 50


@pytest.mark.parametrize("now,epoch,elapsed", [(1.75, 4, 9), (1.2, 5, 9), (1.2, 4, 25)])
def test_late_completed_replies_old_epochs_and_exhausted_prefixes_rejected(now, epoch, elapsed):
    with pytest.raises(RuntimeError):
        accept_reply({"epoch": 4, "started": 1., "deadline": .75}, now, epoch, elapsed)


def test_rtc_transport_binds_identity_and_exact_bounded_prefix(monkeypatch):
    backend = Pi05SO101PolicyBackend(port=8081, language_instruction="Pick banana")
    obs = dict(zip(JOINTS, ORIGIN))
    obs.update(front=np.full((480, 640, 3), 20, np.uint8),
               wrist=np.full((480, 640, 3), 180, np.uint8))
    health = {"status": "ready", "fault": None, "profile": backend.profile.to_dict(),
              "rtc_contract": "pi05-bounded-prefix-v1"}
    prefix = np.tile(ORIGIN, (25, 1))
    wrong = False
    def request(path, data=None, **kwargs):
        if path == "/health":
            assert kwargs["timeout_s"] == .1
            return health
        _, rtc = decode_rtc_request(data)
        assert rtc["prefix_arm"] == prefix.tolist()
        return {"health": health, "seed": 20265907, "timings": {},
                "actions": np.tile(ORIGIN, (50, 1)).tolist(),
                "rtc": {"epoch": 3, "request_id": 6 if wrong else 5,
                        "delay_steps": 15, "prefix_sha256": prefix_digest(prefix)}}
    monkeypatch.setattr(backend, "_request", request)
    assert backend.rtc_ping()
    backend.get_rtc_action(obs, prefix, epoch=3, request_id=5, delay_steps=15)
    wrong = True
    with pytest.raises(ValueError, match="Stale"):
        backend.get_rtc_action(obs, prefix, epoch=3, request_id=5, delay_steps=15)


class Stop:
    def __init__(self):
        self.stopped = False
    def check(self):
        if self.stopped:
            raise RuntimeError("operator stop")


class Controller:
    def __init__(self):
        self.state = ORIGIN.copy()
        self.sent = []
        self.frame = np.tile(np.arange(640, dtype=np.uint8), (480, 1))[..., None].repeat(3, 2)
    def get_observation(self):
        return dict(zip(JOINTS, self.state), front=self.frame, wrist=self.frame)
    def set_target_state(self, target):
        self.state = np.array([target[k] for k in JOINTS])
        self.sent.append(self.state.copy())
        return target


def test_stop_during_inference_prevents_all_late_dispatch(tmp_path):
    stop = Stop()
    controller = Controller()
    class Policy:
        language_instruction = "Pick banana"
        def rtc_ping(self):
            return True
        def get_rtc_action(self, *args, **kwargs):
            stop.stopped = True
            time.sleep(.02)
            return np.tile(ORIGIN, (50, 1)), {}
    result = {"actions": [], "chunks": []}
    with pytest.raises(RuntimeError, match="operator stop"):
        run_rtc_trial(controller, Policy(), stop, tmp_path, result, ORIGIN,
                      LIMITS, lambda: None)
    time.sleep(.03)
    assert controller.sent == []


def test_health_check_does_not_wait_for_inference_lock(monkeypatch):
    backend = Pi05SO101PolicyBackend(port=8081)
    health = {"status": "ready", "fault": None, "profile": backend.profile.to_dict(),
              "rtc_contract": "pi05-bounded-prefix-v1"}
    monkeypatch.setattr(backend, "_request", lambda *a, **kw: health)
    finished = threading.Event()
    with backend._lock:
        thread = threading.Thread(target=lambda: (backend.rtc_ping(), finished.set()))
        thread.start()
        assert finished.wait(.5)
    thread.join()


def test_controller_target_change_stops_before_next_dispatch(tmp_path):
    controller = Controller()
    send = controller.set_target_state
    def changed(target):
        result = send(target).copy()
        result[JOINTS[0]] += .1
        return result
    controller.set_target_state = changed
    class Policy:
        language_instruction = "Pick banana"
        def rtc_ping(self):
            return True
        def get_rtc_action(self, *args, **kwargs):
            return np.tile(ORIGIN, (50, 1)), {}
    with pytest.raises(RuntimeError, match="Controller changed"):
        run_rtc_trial(controller, Policy(), Stop(), tmp_path,
                      {"actions": [], "chunks": []}, ORIGIN, LIMITS, lambda: None)
    assert len(controller.sent) == 1


def test_reply_expiring_during_installation_never_dispatches(tmp_path, monkeypatch):
    controller = Controller()
    original = BoundedRTCQueue.install
    def slow_install(self, *args, **kwargs):
        original(self, *args, **kwargs)
        time.sleep(.06)
    monkeypatch.setattr(BoundedRTCQueue, "install", slow_install)
    class Policy:
        language_instruction = "Pick banana"
        def rtc_ping(self):
            return True
        def get_rtc_action(self, *args, **kwargs):
            return np.tile(ORIGIN, (50, 1)), {}
    with pytest.raises(RuntimeError, match="deadline"):
        run_rtc_trial(controller, Policy(), Stop(), tmp_path,
                      {"actions": [], "chunks": []}, ORIGIN, LIMITS,
                      lambda: None, chunks=1, deadline_s=.05)
    assert controller.sent == []


@pytest.mark.parametrize("failure", ["health", "inference", "instruction"])
def test_failure_while_playing_never_installs_late_replacement(tmp_path, failure):
    controller, stop = Controller(), Stop()
    class Policy:
        language_instruction = "Pick banana"
        calls = 0
        def rtc_ping(self):
            return not (failure == "health" and self.calls > 1)
        def get_rtc_action(self, obs, prefix, **kwargs):
            self.calls += 1
            if self.calls > 1:
                if failure == "instruction":
                    self.language_instruction = "Pick apple"
                time.sleep(.02)
                if failure == "inference":
                    raise RuntimeError("server disconnected")
            return np.tile(ORIGIN + .1, (50, 1)), {}
    result = {"actions": [], "chunks": []}
    with pytest.raises(RuntimeError):
        run_rtc_trial(controller, Policy(), stop, tmp_path, result, ORIGIN,
                      LIMITS, lambda: None, period_s=.002)
    sent = len(controller.sent)
    time.sleep(.03)
    assert len(controller.sent) == sent
    assert 25 <= sent < 50
    assert len(result["chunks"]) == 1
