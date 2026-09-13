"""Exercise inherited dispatch using fake ports; never open hardware."""
import importlib, inspect, logging, threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from policy_guard.contracts import JOINT_ORDER
def runner():
    from embodiment.so_arm10x import safety
    return safety

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
            return (-1 if (retry(events) if callable(retry) else retry) else 0), 0

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
    assert all(e[2] == {"normalize": False, "num_retry": 2} for e in events if e[0] == "read")
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


def test_tracer_configuration_failure_latches_before_torque_cleanup():
    r = runner()
    stop = r.StopLatch()
    from lerobot.motors.motors_bus import get_address
    address = get_address(r.StopGuardedBus.model_ctrl_table, "sts3215", "P_Coefficient")[0]
    bus, events = fake_bus(r, stop, retry=lambda journal: journal[-1][2] == address)
    bus.connect()
    with pytest.raises((ConnectionError, r.SafetyStop)):
        with bus.torque_disabled():
            bus.write("P_Coefficient", "shoulder_pan", 10, num_retry=2)
    failed_writes = [i for i, event in enumerate(events) if event[0] == "raw" and event[2] == address]
    assert len(failed_writes) == 3, "preserve the inherited retry budget before latching exhausted failure"
    assert not [event for event in events[failed_writes[-1] + 1:] if event[0] in ("raw", "sync")], (
        "failed configuration must latch before inherited torque_disabled finally enables torque"
    )
    assert stop.stopped


def test_tracer_transient_configuration_retry_can_recover_normally():
    r = runner()
    stop = r.StopLatch()
    from lerobot.motors.motors_bus import get_address
    address = get_address(r.StopGuardedBus.model_ctrl_table, "sts3215", "P_Coefficient")[0]
    def retry(journal):
        return journal[-1][2] == address and len([e for e in journal if e[0] == "raw" and e[2] == address]) == 1
    bus, events = fake_bus(r, stop, retry=retry)
    bus.connect()
    with bus.torque_disabled():
        bus.write("P_Coefficient", "shoulder_pan", 10, num_retry=2)
    assert not stop.stopped
    assert len([e for e in events if e[0] == "raw" and e[2] == address]) == 2
    assert events[-1][2:] == (55, 1, [1])


def test_dispatch_audit_refuses_changed_pinned_source_hash(monkeypatch):
    r = runner()
    r.audit_dispatch_paths()
    monkeypatch.setitem(r.AUDITED_SOURCES, "lerobot.motors.motors_bus", "0" * 64)
    with pytest.raises(ValueError, match="unaudited"):
        r.audit_dispatch_paths()
