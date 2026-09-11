#!/usr/bin/env python3
# Phase 7 directional runner. Approval is external; this module never creates it.
from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import logging
import signal
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from embodiment.so_arm10x.controller import CLAMP_WARNING_TEXT, SO10xArmController
from embodiment.so_arm10x.skills import PickSkill
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.so_follower import SOFollower
from policy_guard import parity_gate as gate
from policy_guard.replay_contract import (
    JOINT_ORDER, PrerequisiteError, fingerprint_configuration as digest,
    write_evidence,
)

utc_now = gate.utc_now

INSTRUCTION = "Grab a banana and put it on the plate"
TRIALS, ITERATIONS, ACTIONS, ACTION_DELAY = 3, 20, 16, 0.05

# Read-only audit of every inherited target/enable dispatch, including private
# Feetech disable, pre-arm, torque_disabled finally, and SDK retry loops.
# Any source change requires a new audit before construction of live hardware.
AUDITED_SOURCES = {
    "lerobot.robots.so_follower.so_follower": "debe2661e7e44761a74f7cb6aaa83838a0771f91d86f80f42fa822ebb9cdd74b",
    "lerobot.motors.feetech.feetech": "0460413cd6a641cede015ac19ac53d2cfe9c343af1bd18ed74e1951229f64ff2",
    "lerobot.motors.motors_bus": "6ce6109e32590ac2830f6995d13bb478b1a984695cc9ca59c0763ff091666393",
    "scservo_sdk.protocol_packet_handler": "9932c85b9e2ac671a7e33f39e8e80b05d6592280d9f2bcebd85103845e36e655",
    "scservo_sdk.group_sync_write": "70d2a9e2157692932dd6e25515e53d600da9ca87ffab656c4660fb0de4d03476",
}


def audit_dispatch_paths():
    for name, expected in AUDITED_SOURCES.items():
        source = Path(inspect.getfile(importlib.import_module(name)))
        gate.require(hashlib.sha256(source.read_bytes()).hexdigest() == expected,
                     "unaudited hardware dispatch source: " + name)


class SafetyStop(RuntimeError):
    pass


class StopLatch:
    def __init__(self):
        self._stopped = threading.Event()
        self._lock = threading.RLock()
        self._flight = {}
        self._serial = 0
        self.events = []
        self.journal = []
        self.clamp_warnings = 0
        self.reason = ""

    @property
    def stopped(self):
        return self._stopped.is_set()

    def _record(self, kind, **fields):
        event = {"index": len(self.journal), "time": utc_now(), "kind": kind,
                 "previous": self.journal[-1]["sha256"] if self.journal else None, **fields}
        event["sha256"] = digest(event)
        self.journal.append(event)
        return event

    def trip(self, reason, *, clamp=False):
        # Set first: a thread can latch a stop while a dispatched call is blocked.
        self._stopped.set()
        with self._lock:
            self.clamp_warnings += int(clamp)
            if not self.reason:
                self.reason = str(reason)
            self.events.append(self._record("stop", reason=str(reason), clamp=clamp,
                                            in_flight=list(self._flight.values())))

    def check(self):
        if self.stopped:
            raise SafetyStop(self.reason or "operator safety stop")

    @contextmanager
    def dispatch(self, name):
        with self._lock:
            self.check()
            self._serial += 1
            key = self._serial
            self._flight[key] = {"operation": name, "dispatch": key}
            self._record("dispatch", operation=name, dispatch=key)
        try:
            # A call that has passed this dispatch boundary is in flight. Python
            # cannot retract its packet; the stop event retains that fact.
            self.check()
            yield
        finally:
            with self._lock:
                self._flight.pop(key, None)
                self._record("returned", operation=name, dispatch=key)


class _SDKProxy:
    def __init__(self, wrapped, stop):
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_stop", stop)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __setattr__(self, name, value):
        setattr(self._wrapped, name, value)


class StopGuardedPacketHandler(_SDKProxy):
    def writeTxRx(self, *args, **kwargs):
        with self._stop.dispatch("packet_handler.writeTxRx"):
            return self._wrapped.writeTxRx(*args, **kwargs)


class StopGuardedSyncWriter(_SDKProxy):
    def txPacket(self, *args, **kwargs):
        with self._stop.dispatch("sync_writer.txPacket"):
            return self._wrapped.txPacket(*args, **kwargs)


class StopGuardedBus(FeetechMotorsBus):
    def __init__(self, *args, stop, **kwargs):
        self.stop = stop
        super().__init__(*args, **kwargs)
        # Collaborators are replaced while the bus is unconnected. All existing
        # parameter setup/serialization/retry behavior remains inherited.
        self.packet_handler = StopGuardedPacketHandler(self.packet_handler, stop)
        self.sync_writer = StopGuardedSyncWriter(self.sync_writer, stop)

    def connect(self, *args, **kwargs):
        self.stop.check()
        return super().connect(*args, **kwargs)

    def write(self, *args, **kwargs):
        self.stop.check()
        return super().write(*args, **kwargs)

    def sync_write(self, *args, **kwargs):
        self.stop.check()
        return super().sync_write(*args, **kwargs)

    def enable_torque(self, *args, **kwargs):
        self.stop.check()
        return super().enable_torque(*args, **kwargs)

    @contextmanager
    def torque_disabled(self, *args, **kwargs):
        # Latch BODY failures before inherited finally reaches enable_torque.
        # SDK communication retries remain inherited and may recover normally.
        with super().torque_disabled(*args, **kwargs):
            try:
                yield
            except BaseException as exc:
                self.stop.trip("torque-disabled configuration failed: " + str(exc))
                raise

    def disconnect(self, disable_torque=True):
        return super().disconnect(False if self.stop.stopped else disable_torque)


class StopGuardedFollower(SOFollower):
    def __init__(self, config, *, stop):
        self.stop = stop
        stop.check()
        super().__init__(config)
        original = self.bus
        gate.require(not original.is_connected, "bus must be unconnected during composition")
        self.bus = StopGuardedBus(
            port=original.port, motors=original.motors, calibration=original.calibration,
            protocol_version=original.protocol_version, stop=stop,
        )

    def connect(self, calibrate=False):
        self.stop.check()
        gate.require(calibrate is False, "live parity never recalibrates hardware")
        return super().connect(calibrate=False)

    def disconnect(self):
        # The default follower disconnect writes torque registers and assumes all
        # cameras connected. Fault cleanup instead closes only open resources.
        errors = []
        for camera in self.cameras.values():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception as exc:
                errors.append(exc)
        try:
            if self.bus.is_connected:
                self.bus.disconnect(False)
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError("connection cleanup failed") from errors[0]


class StopGuardedController(SO10xArmController):
    def __init__(self, *, stop, **kwargs):
        audit_dispatch_paths()
        self.stop = stop
        stop.check()
        super().__init__(robot_factory=lambda config: StopGuardedFollower(config, stop=stop), **kwargs)

    def connect(self, calibrate=False):
        self.stop.check()
        gate.require(calibrate is False, "calibration is an offline prerequisite")
        return super().connect(calibrate=False)

    def set_target_state(self, target_state):
        self.stop.check()
        sent = super().set_target_state(target_state)
        self.stop.check()
        return sent


@contextmanager
def armed_stop(stop):
    class ClampHandler(logging.Handler):
        def emit(self, record):
            if CLAMP_WARNING_TEXT in record.getMessage():
                stop.trip(record.getMessage(), clamp=True)

    def sink(message):
        if CLAMP_WARNING_TEXT in str(message):
            stop.trip(str(message), clamp=True)

    handler = ClampHandler(level=logging.WARNING)
    root = logging.getLogger()
    root.addHandler(handler)
    sink_id = logger.add(sink, level="WARNING", catch=False)
    saved = {}
    try:
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                saved[sig] = signal.getsignal(sig)
                signal.signal(sig, lambda signum, frame: stop.trip("operator signal " + str(signum)))
        print("STOP ARMED: Ctrl-C / SIGTERM latches failure; no subsequent target or torque-enable dispatch.", flush=True)
        yield stop
    finally:
        for sig, previous in saved.items():
            signal.signal(sig, previous)
        root.removeHandler(handler)
        logger.remove(sink_id)


class CheckedPolicy:
    language_instruction = INSTRUCTION

    def __init__(self, policy, stop):
        self.policy, self.stop, self.iterations = policy, stop, 0

    def get_action(self, observation, instruction):
        with self.stop.dispatch("policy.get_action"):
            actions = self.policy.get_action(observation, instruction)
        self.stop.check()
        try:
            gate.require(isinstance(actions, list) and len(actions) == ACTIONS,
                         "policy chunk must contain exactly 16 actions")
            for action in actions:
                gate.require(isinstance(action, dict) and set(action) == set(JOINT_ORDER),
                             "policy action must contain exactly six canonical joints")
                gate.require(all(isinstance(value, (int, float, np.number)) and
                                 not isinstance(value, (bool, np.bool_)) and np.isfinite(value)
                                 for value in action.values()), "policy targets must be finite numbers")
        except (TypeError, ValueError) as exc:
            self.stop.trip(str(exc))
            raise SafetyStop(str(exc)) from exc
        self.iterations += 1
        return actions


def fresh_evidence(workspace):
    ev = gate.evidence(workspace)
    return gate.Evidence(ev.workspace, test_only=ev.test_only)


def collect_preflight(workspace, *, stage, attempt, runtime_source, previous=None, reason="", clock=utc_now):
    ev = fresh_evidence(workspace)
    started = clock()
    release = gate.validate_release_evidence(ev)
    if stage != "review":
        gate.validate_live_approval(ev)
    # Fail on a reused path before issuing any diagnostic request.
    gate.require(type(attempt) is int and 1 <= attempt <= 9999 and stage in gate.STAGES,
                 "explicit stage and positive attempt required")
    gate.require(not (ev.workspace / f"preflights/{stage}-{attempt:04d}.json").exists(),
                 "immutable preflight path already exists")
    if isinstance(previous, str):
        previous = ev.reference(previous)
    runtime = copy.deepcopy(runtime_source.collect())
    record = gate.PreflightRecord(
        session=release["session"], stage=stage, attempt=attempt, started_at=started,
        ended_at=clock(), status="complete",
        configuration_fingerprint=release["configuration_fingerprint"],
        calibration_sha256=release["calibration_sha256"],
        attestation=runtime["attestation"], attestation_sha256=digest(runtime["attestation"]),
        host=runtime["host"], request=runtime["request"], previous=previous, reason=reason,
        approval=ev.reference("live-approval.json") if stage != "review" else None,
        review_preflight=ev.json("release-review.json")["review_preflight"] if stage != "review" else None,
        evidence_kind="test_only" if ev.test_only else "real_model",
    )
    reference = gate.write_preflight_record(ev, record)
    return reference, runtime


def _release(workspace, reference, stage, runtime, clock):
    ev = fresh_evidence(workspace)
    calibration = gate.validate_release_evidence(ev)["calibration_sha256"]
    return gate.assert_live_release(ev, reference, expected_stage=stage, runtime=runtime,
                                    current_calibration_sha256=calibration, now=clock())


def run(workspace, *, preflight, preflight_attempt, runtime_source,
        controller_factory=StopGuardedController, policy_factory=None, observe=None,
        clock=utc_now, stop=None, settings=None):
    ev = fresh_evidence(workspace)
    stop = stop or StopLatch()
    gate.require(not (ev.workspace / "live-run.json").exists(), "a scored run cannot be repeated in this workspace")
    live = ev.reference(preflight) if isinstance(preflight, str) else preflight
    _release(ev, live, "live", runtime_source.current(), clock)
    construction, rv = collect_preflight(ev, stage="run", attempt=preflight_attempt,
                                         runtime_source=runtime_source, previous=live,
                                         reason="Approved construction preflight", clock=clock)
    _release(ev, construction, "run", rv, clock)
    result = {**ev.identity(), "schema_version": 1,
              "evidence_kind": "test_only" if ev.test_only else "real_model",
              "status": "failed", "instruction": INSTRUCTION, "approval": ev.reference("live-approval.json"),
              "live_preflight": live, "run_preflight": construction, "trials": []}
    controller = policy = None
    previous = construction
    with armed_stop(stop):
        try:
            stop.check()
            result["constructed_at"] = result["started_at"] = clock()
            controller = controller_factory(stop=stop, **(settings or {}))
            controller.connect(calibrate=False)
            stop.check()
            policy = policy_factory()
            for index in range(1, TRIALS + 1):
                stop.check()
                stage = f"trial-{index:02d}"
                reference, rv = collect_preflight(ev, stage=stage, attempt=preflight_attempt,
                                                  runtime_source=runtime_source, previous=previous,
                                                  reason="Fresh preflight before trial reset", clock=clock)
                _release(ev, reference, stage, rv, clock)
                checked = CheckedPolicy(policy, stop)
                trial = {"index": index, "preflight": reference, "started_at": clock(),
                         "iterations": 0, "actions_per_chunk": ACTIONS, "action_delay": ACTION_DELAY,
                         "coherent": False, "wrong_target": False, "erratic": False, "grasp": False,
                         "operator": "", "safety_stop": False, "clamp_warnings": 0}
                result["trials"].append(trial)
                try:
                    PickSkill(controller, checked).run(pose="initial", actions_to_execute=ITERATIONS,
                                                      action_horizon=ACTIONS, language_instruction=INSTRUCTION)
                    stop.check()
                    labels = observe(index)
                    for key in ("coherent", "wrong_target", "erratic", "grasp"):
                        gate.require(type(labels.get(key)) is bool, "named boolean judgment required: " + key)
                    gate.text(labels.get("operator"), "operator identity")
                    trial.update(labels)
                    if labels.get("stop_reason"):
                        stop.trip(labels["stop_reason"])
                    stop.check()
                finally:
                    trial.update(iterations=checked.iterations, ended_at=clock(), safety_stop=stop.stopped,
                                 clamp_warnings=stop.clamp_warnings, stop_reason=stop.reason)
                previous = reference
        except (Exception, KeyboardInterrupt) as exc:
            if not stop.stopped:
                stop.trip(str(exc) or type(exc).__name__)
            result["error"] = type(exc).__name__ + ": " + str(exc)
        finally:
            for resource in (controller, policy):
                if resource is not None:
                    try:
                        if resource is controller:
                            resource.disconnect()
                        elif hasattr(resource, "close"):
                            resource.close()
                    except Exception as exc:
                        stop.trip("cleanup failed: " + str(exc))
            result.update(ended_at=clock(), safety_stop=stop.stopped,
                          clamp_warnings=stop.clamp_warnings, stop_reason=stop.reason,
                          stop_events=stop.events, events=stop.journal)
    result["directional_successes"] = sum(
        t["iterations"] == ITERATIONS and t["coherent"] and not t["wrong_target"] and not t["erratic"]
        for t in result["trials"])
    if len(result["trials"]) == TRIALS and result["directional_successes"] >= 2 and not stop.stopped:
        result["status"] = "complete"
    write_evidence(ev.workspace, "live-run.json", result)
    return result
