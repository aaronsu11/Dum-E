"""Hermetic Plan07-06 evidence and real inherited bus paths; no device is opened."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import test_parity_gate as fixtures
from policy_guard import parity_gate as gate
from policy_guard.replay_contract import JOINT_ORDER, read_json


def runner():
    assert importlib.util.find_spec("run_checkpoint_sanity"), (
        "Approved checkpoint trials need the executable runner and raw stop boundary"
    )
    return importlib.import_module("run_checkpoint_sanity")


class Clock:
    def __init__(self, second=33):
        self.second = second

    def __call__(self):
        self.second += 1
        return fixtures.ts(self.second)


class Runtime:
    def __init__(self, clock, current):
        self.clock, self.value = clock, current

    def current(self):
        return self.value

    def collect(self):
        second = self.clock.second + 1
        self.value = fixtures.runtime(second, self.value["attestation"]["instance"]["load_id"])
        # Keep instance identity stable unless the test explicitly restarts it.
        self.value["attestation"]["instance"] = self.instance
        self.value["host"].update(self.instance)
        self.clock.second = second + 2
        return self.value

    @property
    def instance(self):
        return self._instance

    @instance.setter
    def instance(self, value):
        self._instance = value


def fake_bus(r, stop, *, interrupt=None, retry=False):
    """Real inherited write/enable/retry/context algorithms with fake SDK I/O."""
    from lerobot.motors import Motor, MotorNormMode

    events = []

    class Port:
        is_open = False

        def closePort(self):
            self.is_open = False
            events.append(("close",))

    class Packet:
        def writeTxRx(self, port, motor, address, length, data):
            events.append(("raw", motor, address, length, list(data)))
            if interrupt:
                interrupt(events, stop)
            return (-1 if retry else 0), 0

        def getTxRxResult(self, result):
            return str(result)

        def getRxPacketError(self, result):
            return str(result)

    class Sync:
        def __init__(self):
            self.params = {}

        def clearParam(self):
            self.params.clear()

        def addParam(self, motor, data):
            self.params[motor] = data
            return True

        def txPacket(self):
            events.append(("sync", self.start_address, self.data_length, dict(self.params)))
            if interrupt:
                interrupt(events, stop)
            return -1 if retry else 0

    class Bus(r.StopGuardedBus):
        def connect(self, handshake=True):
            self.stop.check()
            events.append(("bus-connect",))
            self.port_handler.is_open = True

        def disconnect(self, disable_torque=True):
            assert disable_torque is False, "cleanup must preserve torque as found"
            self.port_handler.closePort()

        def sync_read(self, name, *args, **kwargs):
            events.append(("read", name, kwargs))
            if name == "Torque_Enable":
                return {m: 0 for m in self.motors}
            goals = [e for e in events if e[0] == "raw" and e[2] == 42]
            result = {m: (100 + i if name == "Present_Position" or goals else 0)
                      for i, m in enumerate(self.motors)}
            if name == "Goal_Position" and goals and interrupt:
                interrupt(events, self.stop)
            return result

        @property
        def is_calibrated(self):
            return True

        def read(self, name, motor, **kwargs):
            from embodiment.so_arm10x.controller import DUME_PID
            return DUME_PID[name]

    bus = Bus("FAKE-NO-DEVICE", {
        name.removesuffix(".pos"): Motor(i + 1, "sts3215", MotorNormMode.DEGREES)
        for i, name in enumerate(JOINT_ORDER)
    }, stop=stop)
    bus.port_handler = Port()
    bus.packet_handler = r.StopGuardedPacketHandler(Packet(), stop)
    bus.sync_writer = r.StopGuardedSyncWriter(Sync(), stop)
    return bus, events


@pytest.mark.parametrize("where", ["prearm-write", "prearm-readback", "enable", "write-retry", "sync-retry", "cleanup"])
def test_tracer_raw_stop_bypasses(where):
    r = runner()
    stop = r.StopLatch()

    def interrupt(events, latch):
        raw = [e for e in events if e[0] == "raw"]
        if ((where == "prearm-write" and len(raw) == 1)
                or (where == "prearm-readback" and events[-1][:2] == ("read", "Goal_Position"))
                or (where == "enable" and raw and raw[-1][2] == 40)
                or where in ("write-retry", "sync-retry")):
            latch.trip(where)

    bus, events = fake_bus(r, stop, interrupt=interrupt, retry="retry" in where)
    bus.connect()
    with pytest.raises(r.SafetyStop):
        if where.startswith("prearm"):
            from embodiment.so_arm10x.controller import SO10xArmController

            class Follower(r.StopGuardedFollower):
                def __init__(self, config):
                    self.config, self.stop, self.bus, self.cameras = config, stop, bus, {}

                def configure(self):
                    events.append(("follower-configure",))
                    super().configure()

            controller = SO10xArmController(robot_port="FAKE", robot_factory=Follower)
            controller.connect(calibrate=False)
        elif where == "enable":
            bus.enable_torque(num_retry=3)
        elif where == "write-retry":
            bus.write("Goal_Position", "shoulder_pan", 100, normalize=False, num_retry=3)
        elif where == "sync-retry":
            bus.sync_write("Goal_Position", {"shoulder_pan": 100}, normalize=False, num_retry=3)
        else:
            with bus.torque_disabled():
                stop.trip("inside torque_disabled")
    raw = [e for e in events if e[0] in ("raw", "sync")]
    assert len(raw) == (6 if where == "prearm-readback" else 12 if where == "cleanup" else 1)
    assert not any(e[0] == "follower-configure" for e in events)
    for operation in (lambda: bus.write("Lock", "shoulder_pan", 1),
                      lambda: bus.enable_torque(),
                      lambda: bus._disable_torque(1, "sts3215"),
                      lambda: bus.sync_write("Goal_Position", 100)):
        with pytest.raises(r.SafetyStop):
            operation()
    assert len([e for e in events if e[0] in ("raw", "sync")]) == len(raw)
    if where in ("prearm-write", "enable", "write-retry", "sync-retry"):
        assert stop.events[0]["in_flight"], "already-dispatched operation must be recorded"


def test_tracer_nonstopped_prearm_readback_and_sdk_forwarding():
    r = runner()
    stop = r.StopLatch()
    bus, events = fake_bus(r, stop)
    from embodiment.so_arm10x.controller import SO10xArmController
    controller = SO10xArmController(
        robot_port="FAKE", robot_factory=lambda config: SimpleNamespace(bus=bus, cameras={}),
    )
    record = controller._prearm_goal_to_present()
    assert record["prearmed"] and record["goal_ticks_after"] == record["present_ticks"]
    assert [e[1] for e in events if e[0] == "read"] == [
        "Torque_Enable", "Present_Position", "Goal_Position", "Goal_Position",
    ]
    assert all(e[2] == {"normalize": False, "num_retry": 3} for e in events if e[0] == "read")
    bus.connect()
    bus.sync_write("Goal_Position", {"shoulder_pan": 123}, normalize=False, num_retry=2)
    assert events[-1] == ("sync", 42, 2, {1: [123, 0]})
    bus.enable_torque("shoulder_pan", num_retry=2)
    assert [e[2:] for e in events if e[0] == "raw"][-2:] == [(40, 1, [1]), (55, 1, [1])]


def test_tracer_default_factory_and_completed_config_preserved(monkeypatch):
    from embodiment.so_arm10x import controller as cm
    assert "robot_factory" in inspect.signature(cm.SO10xArmController).parameters, (
        "Controller requires optional construction seam without changing its default factory"
    )
    configs = []
    sentinel = object()
    monkeypatch.setattr(cm, "make_robot_from_config", lambda config: configs.append(config) or sentinel)
    default = cm.SO10xArmController(robot_port="FAKE")
    injected = cm.SO10xArmController(robot_port="FAKE", robot_factory=lambda config: configs.append(config) or sentinel)
    assert default.robot is injected.robot is sentinel
    assert configs[0] == configs[1]
    assert configs[0].use_degrees is True and configs[0].max_relative_target == 160.0
    assert (configs[0].position_p_coefficient, configs[0].position_i_coefficient,
            configs[0].position_d_coefficient) == (10, 0, 5)


def test_tracer_follower_construction_preserves_unconnected_collaborators(tmp_path):
    r = runner()
    from lerobot.robots.so_follower import SO101FollowerConfig
    config = SO101FollowerConfig(id="fake", port="FAKE", cameras={}, calibration_dir=tmp_path)
    follower = r.StopGuardedFollower(config, stop=r.StopLatch())
    assert isinstance(follower.bus, r.StopGuardedBus)
    assert follower.bus.port == "FAKE" and not follower.bus.is_connected
    assert follower.bus.calibration == follower.calibration
    assert list(follower.bus.motors) == [key.removesuffix(".pos") for key in JOINT_ORDER]
    assert follower.bus.protocol_version == 0


def test_tracer_real_clamp_emitter_blocks_sync_before_dispatch():
    r = runner()
    stop = r.StopLatch()
    bus, events = fake_bus(r, stop)
    bus.connect()
    follower = object.__new__(r.StopGuardedFollower)
    follower.bus, follower.stop, follower.cameras = bus, stop, {}
    follower.config = SimpleNamespace(max_relative_target=1.0, num_read_retries=0)
    with r.armed_stop(stop):
        with pytest.raises(r.SafetyStop):
            follower.send_action({"shoulder_pan.pos": 999.0})
    assert stop.clamp_warnings > 0
    assert not any(e[0] == "sync" for e in events)


def test_tracer_blocked_inference_never_dispatches_returned_actions():
    r = runner()
    stop = r.StopLatch()
    entered, finish = threading.Event(), threading.Event()

    class Policy:
        def get_action(self, observation, instruction):
            entered.set()
            assert finish.wait(2)
            return [dict.fromkeys(JOINT_ORDER, 0.0) for _ in range(16)]

    policy = r.CheckedPolicy(Policy(), stop)
    errors = []

    def invoke():
        try:
            policy.get_action({}, "banana")
        except r.SafetyStop as exc:
            errors.append(exc)

    thread = threading.Thread(target=invoke)
    thread.start()
    assert entered.wait(2)
    stop.trip("operator stop while inference blocked")
    finish.set()
    thread.join(2)
    assert errors and not thread.is_alive()


def setup_run(tmp_path):
    r = runner()
    live, rv = fixtures.approved(tmp_path)
    ev = gate.Evidence(tmp_path, test_only=True)
    clock = Clock()
    source = Runtime(clock, rv)
    source.instance = rv["attestation"]["instance"]
    events = []
    stop = r.StopLatch()

    class Controller:
        def __init__(self, **kwargs):
            assert (tmp_path / "preflights/run-0001.json").is_file()
            events.append("construct")

        def connect(self, calibrate):
            assert calibrate is False
            events.append("connect")

        def move_to_initial_pose(self):
            assert (tmp_path / "preflights/trial-01-0001.json").is_file()
            events.append("initial")
            stop.trip("test stop during initial reset")
            stop.check()

        def disconnect(self):
            events.append("disconnect")

    return r, ev, live, source, clock, stop, events, Controller


def test_tracer_release_and_trial_preflight_precede_construction_and_reset(tmp_path):
    r, ev, live, source, clock, stop, events, controller = setup_run(tmp_path)
    result = r.run(ev, preflight=live["path"], preflight_attempt=1, runtime_source=source,
                   clock=clock, stop=stop, controller_factory=controller,
                   policy_factory=lambda: SimpleNamespace(), observe=lambda _: {})
    assert result["status"] == "failed" and result["safety_stop"]
    assert events == ["construct", "connect", "initial", "disconnect"]
    assert result["trials"][0]["index"] == 1 and len(result["trials"]) == 1
    assert read_json(tmp_path / "live-run.json") == result
    assert result["evidence_kind"] == "test_only"


def test_tracer_missing_approval_never_constructs(tmp_path):
    r, ev, live, source, clock, stop, events, controller = setup_run(tmp_path)
    (tmp_path / "live-approval.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        r.run(ev, preflight=live["path"], preflight_attempt=1, runtime_source=source,
              clock=clock, stop=stop, controller_factory=controller,
              policy_factory=lambda: SimpleNamespace(), observe=lambda _: {})
    assert events == []
