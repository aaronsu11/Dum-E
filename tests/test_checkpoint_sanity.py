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

    def inputs(self):
        return {"controller": {"robot_port": "FAKE"}, "safety": {"stop_ready": True}}

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


def approved_runner(workspace, *, inputs=None):
    r = runner()
    identity, lock, pairs = fixtures.fixture_workspace(workspace)
    fixtures.complete_offline(workspace, identity, lock, pairs)
    ev = gate.Evidence(workspace, test_only=True)
    clock = Clock(19)
    source = Runtime(clock, fixtures.runtime(15))
    source.instance = source.value["attestation"]["instance"]
    if inputs is not None:
        source.inputs = lambda: inputs
    review = r.preflight(ev, stage="review", attempt=1, runtime_source=source, clock=clock)
    fixtures.cli().prepare_live(gate.Evidence(workspace, test_only=True), review["path"], clock=clock)
    answers = iter(["Fixture operator", "Test-only decision", "approve"])
    fixtures.cli().record_decision(workspace, "live", prompt=lambda _: next(answers), clock=clock, test_only=True)
    live = r.preflight(ev, stage="live", attempt=1, runtime_source=source, clock=clock)
    return live, source, clock


def setup_run(tmp_path, *, inputs=None):
    r = runner()
    live, source, clock = approved_runner(tmp_path, inputs=inputs)
    ev = gate.Evidence(tmp_path, test_only=True)
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


@pytest.mark.parametrize("reset_clamp", [False, True])
def test_tracer_actual_skill_reset_and_bounded_chunk(tmp_path, monkeypatch, reset_clamp):
    r, ev, live, source, clock, stop, events, _ = setup_run(tmp_path)
    from embodiment.so_arm10x import controller as cm, skills
    delays = []
    monkeypatch.setattr(cm, "time", SimpleNamespace(sleep=delays.append))
    monkeypatch.setattr(skills, "time", SimpleNamespace(sleep=delays.append))
    actions = []

    class Controller(r.StopGuardedController):
        def __init__(self, **kwargs):
            self.stop = stop
            events.append("construct")

        def connect(self, calibrate):
            assert calibrate is False
            events.append("connect")

        def set_target_state(self, action):
            stop.check()
            if isinstance(action, np.ndarray):
                events.append("reset-target")
                if reset_clamp:
                    logging.warning(cm.CLAMP_WARNING_TEXT)
                    stop.check()
                return dict(zip(JOINT_ORDER, action, strict=True))
            actions.append(action)
            return action

        def get_observation(self):
            return {**dict.fromkeys(JOINT_ORDER, 0.0), "front": np.zeros((2, 2, 3), np.uint8),
                    "wrist": np.zeros((2, 2, 3), np.uint8)}

        def disconnect(self):
            events.append("disconnect")

    class Policy:
        calls = 0

        def get_action(self, observation, instruction):
            assert instruction == "Grab a banana and put it on the plate"
            self.calls += 1
            if self.calls == 2:
                stop.trip("stop during next inference")
            return [dict.fromkeys(JOINT_ORDER, 0.0) for _ in range(16)]

    result = r.run(ev, preflight=live["path"], preflight_attempt=1, runtime_source=source,
                   controller_factory=Controller, policy_factory=Policy, observe=lambda _: {},
                   stop=stop, clock=clock)
    assert result["status"] == "failed" and len(result["trials"]) == 1
    assert result["clamp_warnings"] == int(reset_clamp)
    assert events[-1] == "disconnect"
    if reset_clamp:
        assert not actions and events.count("reset-target") == 1
    else:
        assert len(actions) == 16 and delays.count(0.05) == 16
        assert events.count("reset-target") == 3  # inherited ready -> initial -> ready
        assert result["trials"][0]["iterations"] == 1



def full_run(tmp_path, monkeypatch, *, labels=None, stop_trial=None, prepared=None, policy_factory=None):
    r, ev, live, source, clock, stop, events, _ = prepared or setup_run(tmp_path)
    from embodiment.so_arm10x import controller as cm, skills
    delays, targets = [], []
    monkeypatch.setattr(cm, "time", SimpleNamespace(sleep=delays.append))
    monkeypatch.setattr(skills, "time", SimpleNamespace(sleep=delays.append))

    class Controller(r.StopGuardedController):
        def __init__(self, **kwargs):
            self.stop = stop
            events.append("construct")

        def connect(self, calibrate):
            assert calibrate is False
            events.append("connect")

        def set_target_state(self, action):
            stop.check()
            if isinstance(action, np.ndarray):
                targets.append(tuple(action))
                return dict(zip(JOINT_ORDER, action, strict=True))
            targets.append(action)
            return action

        def get_observation(self):
            return {**dict.fromkeys(JOINT_ORDER, 0.0), "front": np.zeros((480, 640, 3), np.uint8),
                    "wrist": np.zeros((480, 640, 3), np.uint8)}

        def get_current_images(self):
            return {"front": self.get_observation()["front"], "wrist": self.get_observation()["wrist"]}

        def disconnect(self):
            events.append("disconnect")

    class Policy:
        calls = 0

        def get_action(self, observation, instruction):
            assert instruction == "Grab a banana and put it on the plate"
            assert set(observation) == set(JOINT_ORDER) | {"front", "wrist"}
            self.calls += 1
            return [dict.fromkeys(JOINT_ORDER, 0.0) for _ in range(16)]

    def observe(index):
        if stop_trial == index:
            stop.trip("operator safety stop")
        return (labels or [dict(coherent=True, wrong_target=False, erratic=False, grasp=False,
                               operator="Fixture observer") for _ in range(3)])[index - 1]

    result = r.run(ev, preflight=live["path"], preflight_attempt=1, runtime_source=source,
                   controller_factory=Controller, policy_factory=policy_factory or Policy, observe=observe,
                   stop=stop, clock=clock)
    return result, targets, delays, source, ev


def test_exact_three_trial_protocol_and_canonical_closeout(tmp_path, monkeypatch):
    result, targets, delays, _, ev = full_run(tmp_path, monkeypatch)
    assert result["status"] == "complete"
    assert len(result["trials"]) == 3 and result["directional_successes"] == 3
    assert all(t["iterations"] == 20 and t["actions_per_chunk"] == 16 and
               t["action_delay"] == 0.05 and not t["grasp"] for t in result["trials"])
    assert len([t for t in targets if isinstance(t, dict)]) == 3 * 20 * 16
    assert len([t for t in targets if isinstance(t, tuple)]) == 9
    assert delays.count(0.05) == 960
    run_ref = gate.Evidence(tmp_path, test_only=True).reference("live-run.json")
    final = read_json(tmp_path / "golden-replay.json")
    final.update(live_run=run_ref, started_at=fixtures.ts(1000), ended_at=fixtures.ts(1001))
    fixtures.save(tmp_path, "final-regression.json", final)
    assert gate.validate_closeout(gate.Evidence(tmp_path, test_only=True))["status"] == "complete"
    assert result["preflights"] == [result["live_preflight"], result["run_preflight"],
                                    *(t["preflight"] for t in result["trials"])]


@pytest.mark.parametrize("successes,wrong,erratic,grasp,expected", [
    (2, False, False, False, "complete"), (1, False, False, True, "failed"),
    (3, True, False, True, "failed"), (3, False, True, True, "failed"),
])
def test_directional_scoring_has_fixed_denominator(tmp_path, monkeypatch, successes, wrong, erratic, grasp, expected):
    labels = [dict(coherent=i < successes, wrong_target=wrong, erratic=erratic, grasp=grasp,
                   operator="Fixture observer") for i in range(3)]
    result, targets, _, _, _ = full_run(tmp_path, monkeypatch, labels=labels)
    assert result["status"] == expected and len(result["trials"]) == 3
    assert len([t for t in targets if isinstance(t, dict)]) == 960


def test_safety_stop_cannot_be_outvoted_or_drop_interrupted_trial(tmp_path, monkeypatch):
    result, _, _, _, _ = full_run(tmp_path, monkeypatch, stop_trial=3)
    assert result["status"] == "failed" and result["safety_stop"]
    assert len(result["trials"]) == 3 and result["trials"][2]["safety_stop"]
    assert result["directional_successes"] >= 2


@pytest.mark.parametrize("size", [0, 15, 17])
def test_chunk_budget_refuses_short_and_extended_actions(size):
    r = runner()
    stop = r.StopLatch()
    policy = r.CheckedPolicy(SimpleNamespace(get_action=lambda *_: [dict.fromkeys(JOINT_ORDER, 0.0)] * size), stop)
    with pytest.raises(r.SafetyStop, match="exactly 16"):
        policy.get_action({}, r.INSTRUCTION)
    assert stop.stopped and policy.iterations == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1"])
def test_nonfinite_or_non_numeric_policy_targets_stop(bad):
    r = runner()
    stop = r.StopLatch()
    policy = r.CheckedPolicy(SimpleNamespace(get_action=lambda *_: [dict.fromkeys(JOINT_ORDER, bad)] * 16), stop)
    with pytest.raises(r.SafetyStop):
        policy.get_action({}, r.INSTRUCTION)
    assert stop.stopped


def test_cli_exposes_only_fixed_budgets_and_explicit_preflight_attempts():
    r = runner()
    assert callable(getattr(r, "build_parser", None)), "Runnable preflight/run/check commands are required"
    parser = r.build_parser()
    args = parser.parse_args(["preflight", "--stage", "review", "--attempt", "0001", "--workspace", "example"])
    assert args.stage == "review" and args.attempt == 1
    args = parser.parse_args(["run", "--preflight", "preflights/live-0001.json", "--preflight-attempt", "0001"])
    assert args.preflight_attempt == 1
    for option in ("--trials", "--iterations", "--actions-per-chunk", "--action-delay", "--test-only", "--approve"):
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "--preflight", "preflights/live-0001.json", "--preflight-attempt", "0001", option, "4"])


def test_review_approval_live_restart_run_lifecycle(tmp_path, monkeypatch):
    r = runner()
    assert callable(getattr(r, "preflight", None)), "Real runner must collect immutable review/live attempts"
    identity, lock, pairs = fixtures.fixture_workspace(tmp_path)
    fixtures.complete_offline(tmp_path, identity, lock, pairs)
    ev = gate.Evidence(tmp_path, test_only=True)
    clock = Clock(19)
    source = Runtime(clock, fixtures.runtime(15))
    source.instance = source.value["attestation"]["instance"]
    ref = r.preflight(ev, stage="review", attempt=1, runtime_source=source, clock=clock)
    assert not (tmp_path / "live-approval.json").exists()
    fixtures.cli().prepare_live(gate.Evidence(tmp_path, test_only=True), ref["path"], clock=clock)
    answers = iter(["Fixture operator", "Test-only decision", "approve"])
    fixtures.cli().record_decision(tmp_path, "live", prompt=lambda _: next(answers), clock=clock, test_only=True)
    live = r.preflight(ev, stage="live", attempt=1, previous_preflight=ref["path"], reason="Approved transition",
                       runtime_source=source, clock=clock)
    old = {p: p.read_bytes() for p in (tmp_path / "preflights").iterdir()}
    restarted = fixtures.runtime(clock.second + 1, "second")
    source.value, source.instance = restarted, restarted["attestation"]["instance"]
    with pytest.raises(ValueError):
        r._release(ev, live, "live", source.current(), clock)
    renewed = r.preflight(ev, stage="live", attempt=2, previous_preflight=live["path"], reason="Same-profile restart",
                          runtime_source=source, clock=clock)
    r._release(ev, renewed, "live", source.current(), clock)
    with pytest.raises(ValueError, match="exists"):
        r.preflight(ev, stage="live", attempt=2, previous_preflight=live["path"], reason="Reuse refusal",
                    runtime_source=source, clock=clock)
    assert all(p.read_bytes() == value for p, value in old.items())
    assert read_json(tmp_path / renewed["path"])["attestation"] == source.current()["attestation"]
    result, _, _, _, _ = full_run(tmp_path, monkeypatch, prepared=(
        r, ev, renewed, source, clock, r.StopLatch(), [], None))
    assert result["status"] == "complete" and result["live_preflight"] == renewed
    assert all(p.read_bytes() == value for p, value in old.items())


def test_failed_preflight_is_immutable_evidence(tmp_path):
    r, ev, live, source, clock, _, _, _ = setup_run(tmp_path)
    source.collect = lambda: (_ for _ in ()).throw(TimeoutError("fake diagnostic timeout"))
    with pytest.raises(TimeoutError):
        r.collect_preflight(ev, stage="run", attempt=1, previous=live,
                            reason="Fresh run", runtime_source=source, clock=clock)
    target = tmp_path / "preflights/run-0001.json"
    assert target.exists(), "Failed freshness attempts must remain immutable and cannot release"
    assert read_json(target)["status"] == "failed"
    with pytest.raises(ValueError):
        gate.validate_preflight_record(gate.Evidence(tmp_path, test_only=True), "preflights/run-0001.json")


def test_operator_labels_cannot_override_budget_or_safety(tmp_path, monkeypatch):
    labels = [dict(coherent=True, wrong_target=False, erratic=False, grasp=False,
                   operator="Fixture observer", preflight={"path": "forged", "sha256": "0" * 64},
                   actions_per_chunk=99, safety_stop=False) for _ in range(3)]
    result, _, _, _, _ = full_run(tmp_path, monkeypatch, labels=labels)
    assert result["status"] == "failed", "Observation fields cannot replace protected trial evidence"
    assert result["trials"][0]["actions_per_chunk"] == 16


def test_production_cli_rejects_fixture_evidence_before_runtime(tmp_path):
    r = runner()
    assert callable(getattr(r, "main", None)), "Production CLI must fail closed on test_only evidence"
    fixtures.approved(tmp_path)
    assert r.main(["check", "--workspace", str(tmp_path)]) != 0



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



@pytest.mark.parametrize("kind", ["controller", "safety", "stale", "wrong-stage", "predecessor"])
def test_changed_release_inputs_refuse_before_construction(tmp_path, kind):
    r, ev, live, source, clock, stop, events, controller = setup_run(tmp_path)
    if kind == "controller":
        source.inputs = lambda: {"controller": {"robot_port": "OTHER"}, "safety": {"stop_ready": True}}
    elif kind == "safety":
        source.inputs = lambda: {"controller": {"robot_port": "FAKE"}, "safety": {"stop_ready": False}}
    elif kind == "stale":
        clock.second += 61
    elif kind == "wrong-stage":
        live = {"path": "preflights/review-0001.json"}
    else:
        fixtures.rewrite(tmp_path, "preflights/review-0001.json", lambda row: row.update(reason="mutated predecessor"))
    with pytest.raises((ValueError, FileNotFoundError)):
        r.run(ev, preflight=live["path"], preflight_attempt=1, runtime_source=source,
              clock=clock, stop=stop, controller_factory=controller,
              policy_factory=lambda: SimpleNamespace(), observe=lambda _: {})
    assert events == []
    assert not (tmp_path / "preflights/run-0001.json").exists()


def test_readonly_run_check_rejects_chronology_and_journal_tampering(tmp_path, monkeypatch):
    r = runner()
    result, _, _, _, ev = full_run(tmp_path, monkeypatch)
    assert r.check(ev)["status"] == "complete"
    fixtures.rewrite(tmp_path, "live-run.json", lambda row: row.update(constructed_at=fixtures.ts(1)))
    with pytest.raises(ValueError, match="chronology"):
        r.check(ev)
    (tmp_path / "live-run.json").write_bytes(fixtures.canonical(result))
    fixtures.rewrite(tmp_path, "live-run.json", lambda row: row["events"][0].update(operation="changed"))
    with pytest.raises(ValueError, match="journal"):
        r.check(ev)


def test_dispatch_audit_refuses_changed_pinned_source_hash(monkeypatch):
    r = runner()
    r.audit_dispatch_paths()
    monkeypatch.setitem(r.AUDITED_SOURCES, "lerobot.motors.motors_bus", "0" * 64)
    with pytest.raises(ValueError, match="unaudited"):
        r.audit_dispatch_paths()


def test_named_observations_have_no_default_or_motion_authority():
    r = runner()
    answers = iter(["", "Fixture observer", "", "maybe", "yes", "no", "no", "no", ""])
    result = r.observe_trial(1, prompt=lambda _: next(answers))
    assert result == {"operator": "Fixture observer", "coherent": True, "wrong_target": False,
                      "erratic": False, "grasp": False, "stop_reason": ""}


def attached_source(tmp_path, monkeypatch, *, clock=None, initial=None, restart=False):
    # The production attachment and inherited get_action mapping execute against
    # a fake owned Session class. Only the runtime validator test_only argument
    # is injected; every fixture byte stays labelled test_only in this tmpdir.
    r = runner()
    import torch
    from policy.lerobot import session as sm, backend as bm
    clock = clock or Clock(40)
    monkeypatch.setattr(r, "utc_now", clock)
    if not (tmp_path / "profiles.json").exists():
        fixtures.save(tmp_path, "profiles.json", {"serving_configuration": fixtures.semantics()})
    initial = initial or fixtures.runtime(30)
    path = tmp_path / "runtime/lerobot.json"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(fixtures.canonical(initial["attestation"]))
    events, hosts = [], []
    decoded = np.arange(96, dtype=np.float32).reshape(16, 6)

    class Session:
        def __init__(self, address, **kwargs):
            self.address, self.kwargs = address, kwargs
            events.append(("session", kwargs))

        def connect(self, specs):
            pytest.fail("the attachment must never send model-loading policy instructions")

        def probe_ready_or_raise(self):
            events.append(("ready",))

        def close(self):
            events.append(("close",))

        def infer(self, observation):
            events.append(("infer", observation))
            assert set(observation) == set(JOINT_ORDER) | {"front", "wrist", "task"}
            replaced = restart and sum(event[0] == "infer" for event in events) >= 2
            rv = fixtures.runtime(clock.second + 1, "replacement" if replaced else initial["attestation"]["instance"]["load_id"])
            if not replaced:
                rv["attestation"]["instance"] = initial["attestation"]["instance"]
                rv["host"].update(initial["attestation"]["instance"])
            rv["request"]["observation_sha256"] = gate.observation_fingerprint(observation)
            rv["request"]["output_sha256"] = gate.array_fingerprint(decoded)
            rv["attestation"]["request"] = rv["request"]
            clock.second += 3
            path.write_bytes(fixtures.canonical(rv["attestation"]))
            hosts.append(rv["host"])
            return [SimpleNamespace(get_action=lambda value=value: torch.from_numpy(value),
                                    get_timestamp=lambda: rv["request"]["timestamp"],
                                    get_timestep=lambda: rv["request"]["timestep"]) for value in decoded]

    monkeypatch.setattr(sm, "LeRobotPolicySession", Session)
    monkeypatch.setattr(bm, "LeRobotPolicySession", Session)
    def host(*args):
        hosts[-1]["checked_at"] = clock()
        return hosts[-1]
    monkeypatch.setattr(gate, "collect_runtime_host", host)
    actual_validator = gate.validate_runtime_attestation
    def validate(*args, **kwargs):
        assert args[0]["evidence_kind"] == "test_only"
        kwargs["test_only"] = True
        return actual_validator(*args, **kwargs)
    monkeypatch.setattr(gate, "validate_runtime_attestation", validate)
    source = r.RuntimeSource(SimpleNamespace(workspace=tmp_path, attestation=path, preflight=None,
                                            container="fake", endpoint="127.0.0.1:8080", checkpoint_mount="/checkpoint"))
    observation = {**dict.fromkeys(JOINT_ORDER, 0.0), "front": np.zeros((480, 640, 3), np.uint8),
                   "wrist": np.ones((480, 640, 3), np.uint8)}
    source._runtime = initial
    return source, observation, decoded, events


def test_actual_attached_policy_maps_observations_and_never_loads_model(tmp_path, monkeypatch):
    r = runner()
    source, observation, decoded, events = attached_source(tmp_path, monkeypatch)
    try:
        actions = source.policy.get_action(observation, r.INSTRUCTION)
        assert len(actions) == 16
        assert actions[-1] == dict(zip(JOINT_ORDER, decoded[-1], strict=True))
        assert [event[0] for event in events].count("ready") == 1
        assert source._policy._session.kwargs == {"max_attempts": 1, "handshake_max_attempts": 1}
        assert source._runtime["attestation"]["evidence_kind"] == "test_only"
    finally:
        source.close()



def test_controller_inputs_hash_and_consume_the_same_config_bytes(tmp_path, monkeypatch):
    r = runner()
    config = tmp_path / "robot.yaml"
    before = b"controller: {robot_type: so101_follower, robot_id: fake, robot_port: OLD}"
    after = before.replace(b"OLD", b"NEW")
    config.write_bytes(before)
    calibration = tmp_path / "calibration/robots/so_follower/fake.json"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("{}")
    fixtures.save(tmp_path, "session.json", {"session_id": "test-only"})
    fixtures.save(tmp_path, "calibration.json", {"robot_identity": {"robot_type": "so101_follower", "robot_id": "fake"}})
    monkeypatch.setenv("DUME_CONFIG", str(config))
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "calibration"))
    monkeypatch.setattr(gate, "validate_release_evidence", lambda _: {"calibration_sha256": r.sha256_file(calibration)})
    capture = r.capture_bytes
    def changed_at_capture(path):
        if path == config:
            config.write_bytes(after)
        return capture(path)
    monkeypatch.setattr(r, "capture_bytes", changed_at_capture)
    snapshot = r.controller_inputs(tmp_path)
    assert snapshot["config_sha256"] == r.hashlib.sha256(after).hexdigest()
    assert snapshot["controller"]["robot_port"] == "NEW", "settings must derive from the captured, hashed bytes"



def test_renewal_requires_explicit_reason_before_any_new_request(tmp_path):
    r, ev, live, source, clock, _, _, _ = setup_run(tmp_path)
    request = source.current()["request"]
    with pytest.raises(ValueError, match="reason"):
        r.preflight(ev, stage="live", attempt=2, previous_preflight=live["path"],
                    runtime_source=source, clock=clock)
    assert source.current()["request"] == request
    assert not (tmp_path / "preflights/live-0002.json").exists()


@pytest.mark.parametrize("restart", [False, True])
def test_active_trial_binds_actual_attached_session_instance(tmp_path, monkeypatch, restart):
    prepared = setup_run(tmp_path)
    r, ev, live, runtime, clock, stop, _, _ = prepared
    source, _, _, events = attached_source(tmp_path, monkeypatch, clock=clock,
                                          initial=runtime.value, restart=restart)
    original_collect = runtime.collect
    def collect():
        value = original_collect()
        source._runtime = value
        source.attestation_path.write_bytes(fixtures.canonical(value["attestation"]))
        return value
    runtime.collect = collect
    runtime.current = lambda: source._runtime
    result, targets, _, _, _ = full_run(tmp_path, monkeypatch, prepared=prepared,
                                       policy_factory=lambda: source.policy)
    actions = [target for target in targets if isinstance(target, dict)]
    if restart:
        assert result["status"] == "failed", "same-profile replacement cannot inherit the active trial release"
        assert stop.stopped and len(actions) == 16
        assert len(result["trials"]) == 1 and result["trials"][0]["iterations"] == 1
        assert sum(event[0] == "infer" for event in events) == 2
        assert "instance" in result["error"] or "runtime" in result["error"]
        assert not (tmp_path / "preflights/trial-02-0001.json").exists()
    else:
        assert result["status"] == "complete" and len(actions) == 960
        assert not stop.stopped


@pytest.mark.parametrize("substitution", ["loaded", "bus", "path", "none"])
def test_loaded_calibration_checked_before_connect_after_file_aba(tmp_path, substitution):
    r = runner()
    from lerobot.robots.so_follower import SO101FollowerConfig
    from lerobot.motors import MotorCalibration
    from dataclasses import asdict
    path = tmp_path / "robot-calibration/fake.json"
    path.parent.mkdir()
    mapping = {key.removesuffix(".pos"): dict(id=i + 1, drive_mode=0, homing_offset=0,
               range_min=10, range_max=4000) for i, key in enumerate(JOINT_ORDER)}
    approved_bytes = fixtures.canonical(mapping)
    path.write_bytes(approved_bytes)
    inputs = {"controller": {"robot_port": "FAKE"}, "calibration_path": str(path.resolve()),
              "calibration_sha256": r.hashlib.sha256(approved_bytes).hexdigest(),
              "calibration_mapping": mapping, "safety": {"stop_ready": True}}
    r, ev, live, source, clock, stop, events, _ = setup_run(tmp_path, inputs=inputs)
    loaded = []
    class Controller(r.StopGuardedController):
        def __init__(self, **kwargs):
            self.stop = stop
            changed = {name: dict(value) for name, value in mapping.items()}
            changed["shoulder_pan"]["range_min"] = 999
            if substitution == "loaded":
                path.write_bytes(fixtures.canonical(changed))
            try:
                self.robot = r.StopGuardedFollower(SO101FollowerConfig(
                    id="fake", port="FAKE", cameras={}, calibration_dir=path.parent), stop=stop)
            finally:
                path.write_bytes(approved_bytes)
            if substitution == "bus":
                self.robot.bus.calibration = {name: MotorCalibration(**value) for name, value in changed.items()}
            if substitution == "path":
                other = path.parent / "other.json"
                other.write_bytes(approved_bytes)
                self.robot.calibration_fpath = other
            loaded.append({name: asdict(value) for name, value in self.robot.bus.calibration.items()})
            events.append("construct")

        def connect(self, calibrate=False):
            events.append("connect/prearm")
            raise RuntimeError("fake boundary: never open hardware")

        def disconnect(self):
            events.append("disconnect")
    result = r.run(ev, preflight=live, preflight_attempt=1, runtime_source=source,
                   controller_factory=Controller, policy_factory=lambda: SimpleNamespace(),
                   observe=lambda _: {}, clock=clock, stop=stop)
    assert path.read_bytes() == approved_bytes
    if substitution in ("loaded", "bus"):
        assert loaded[0]["shoulder_pan"]["range_min"] == 999
    if substitution == "none":
        assert events == ["construct", "connect/prearm", "disconnect"]
    else:
        assert events == ["construct", "disconnect"], "loaded mapping/path must match approval before pre-arm"
        assert "calibration" in result["error"]
    assert result["status"] == "failed" and stop.stopped and not result["trials"]


@pytest.mark.parametrize("checker", ["runner", "canonical"])
@pytest.mark.parametrize("hide_stop_events", [False, True])
def test_authentic_final_trial_stop_cannot_be_hidden_by_summary_flags(tmp_path, monkeypatch, checker, hide_stop_events):
    r = runner()
    labels = [dict(coherent=True, wrong_target=False, erratic=False, grasp=False,
                   operator="Fixture observer") for _ in range(3)]
    labels[-1]["stop_reason"] = "Operator stopped after final actions"
    result, targets, _, _, ev = full_run(tmp_path, monkeypatch, labels=labels)
    assert result["status"] == "failed" and result["safety_stop"]
    assert len([target for target in targets if isinstance(target, dict)]) == 960
    assert any(event["kind"] == "stop" for event in result["events"])
    result.update(status="complete", safety_stop=False, clamp_warnings=0, stop_reason="")
    for trial in result["trials"]:
        trial.update(safety_stop=False, clamp_warnings=0, stop_reason="")
    if hide_stop_events:
        result["stop_events"] = []
    (tmp_path / "live-run.json").write_bytes(fixtures.canonical(result))
    final = read_json(tmp_path / "golden-replay.json")
    final.update(live_run=gate.Evidence(tmp_path, test_only=True).reference("live-run.json"),
                 started_at=fixtures.ts(1000), ended_at=fixtures.ts(1001))
    fixtures.save(tmp_path, "final-regression.json", final)
    with pytest.raises(ValueError, match="journal|stop|safety"):
        (r.check if checker == "runner" else gate.validate_closeout)(gate.Evidence(tmp_path, test_only=True))


@pytest.mark.parametrize("restart", [False, True])
def test_attached_session_replacement_cannot_dispatch_next_chunk(tmp_path, monkeypatch, restart):
    r = runner()
    source, observation, _, events = attached_source(tmp_path, monkeypatch, restart=restart)
    stop = r.StopLatch()
    policy = r.CheckedPolicy(source.policy, stop)
    first = policy.get_action(observation, r.INSTRUCTION)
    assert len(first) == 16 and policy.iterations == 1
    try:
        if restart:
            with pytest.raises(r.SafetyStop, match="instance|runtime"):
                policy.get_action(observation, r.INSTRUCTION)
            assert stop.stopped and policy.iterations == 1
        else:
            assert len(policy.get_action(observation, r.INSTRUCTION)) == 16
            assert not stop.stopped and policy.iterations == 2
        assert sum(event[0] == "infer" for event in events) == 2
    finally:
        source.close()



def test_constructor_settings_cannot_escape_approved_snapshot_on_input_aba(tmp_path):
    r, ev, live, source, clock, stop, _, _ = setup_run(tmp_path)
    original_inputs = source.inputs
    captures, constructed_ports = [], []
    def inputs():
        snapshot = original_inputs()
        captures.append(snapshot)
        if len(captures) == 2:
            return {**snapshot, "controller": {"robot_port": "UNAPPROVED-PORT"}}
        return snapshot
    source.inputs = inputs
    class Controller:
        def __init__(self, *, robot_port, stop):
            constructed_ports.append(robot_port)
        def connect(self, calibrate):
            raise RuntimeError("fake connect boundary; no hardware")
        def disconnect(self):
            pass
    try:
        r.run(ev, preflight=live, preflight_attempt=1, runtime_source=source,
              controller_factory=Controller, policy_factory=lambda: SimpleNamespace(),
              observe=lambda _: {}, clock=clock, stop=stop)
    except ValueError:
        pass  # Refusal before construction is also safe.
    assert "UNAPPROVED-PORT" not in constructed_ports, "constructor must consume the approved captured settings"


@pytest.mark.parametrize("index", [1, 2])
def test_single_trial_runner_stops_after_one_observation(tmp_path, monkeypatch, index):
    prepared = setup_run(tmp_path)
    validate = gate.validate_live_approval
    def one_trial(*args, **kwargs):
        return {**validate(*args, **kwargs), "approval_scope": gate.single_trial_scope(index)}
    monkeypatch.setattr(gate, "validate_live_approval", one_trial)
    result, targets, delays, _, _ = full_run(tmp_path, monkeypatch, prepared=prepared)
    assert result["authorized_trials"] == 1
    assert result["authorized_trial_indices"] == [index]
    assert result["trials"][0]["index"] == index
    assert result["status"] == "partial"
    assert len(result["trials"]) == 1 and result["trials"][0]["iterations"] == 20
    assert len([t for t in targets if isinstance(t, dict)]) == 20 * 16
    assert delays.count(0.05) == 320
