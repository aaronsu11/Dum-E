#!/usr/bin/env python3
# Phase 7 directional runner. Approval is external; this module never creates it.
from __future__ import annotations

import argparse
import copy
import json
import os
import hashlib
import importlib
import importlib.metadata
import inspect
import logging
import signal
import sys
import threading
from contextlib import contextmanager
from dataclasses import asdict, replace
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
    now as utc_now, canonical, capture_bytes, contained, load_case, load_input_lock,
    read_json, sha256_file, write_evidence,
)

INSTRUCTION = "Grab a banana and put it on the plate"
TRIALS, ITERATIONS, ACTIONS, ACTION_DELAY = 3, 20, 16, 0.05

# Read-only audit of every inherited target/enable dispatch, including private
# Feetech disable, pre-arm, torque_disabled finally, and SDK retry loops.
# Any source change requires a new audit before construction of live hardware.
AUDITED_SOURCES = {
    "lerobot.robots.utils": "eb6a5039cd553a15e5a5c6888d2ce4989afc3e38103e89f9b913276110f30a42",
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


def runtime_identity(runtime):
    attestation = runtime["attestation"]
    return {"instance": copy.deepcopy(attestation["instance"]),
            "loaded_at": attestation["loaded_at"],
            "configuration_fingerprint": attestation["configuration_fingerprint"],
            "host": {k: v for k, v in runtime["host"].items() if k != "checked_at"}}


class CheckedPolicy:
    language_instruction = INSTRUCTION

    def __init__(self, policy, stop, *, runtime_source=None, runtime=None):
        self.policy, self.stop, self.iterations = policy, stop, 0
        self.runtime_source = runtime_source
        self.released_identity = runtime_identity(runtime) if runtime is not None else None

    def _check_runtime(self):
        if self.runtime_source is not None:
            gate.require(runtime_identity(self.runtime_source.current()) == self.released_identity,
                         "active trial runtime instance/load changed; explicit renewal required")

    def get_action(self, observation, instruction):
        try:
            self._check_runtime()
            with self.stop.dispatch("policy.get_action"):
                actions = self.policy.get_action(observation, instruction)
            self.stop.check()
            self._check_runtime()
            gate.require(isinstance(actions, list) and len(actions) == ACTIONS,
                         "policy chunk must contain exactly 16 actions")
            for action in actions:
                gate.require(isinstance(action, dict) and set(action) == set(JOINT_ORDER),
                             "policy action must contain exactly six canonical joints")
                gate.require(all(isinstance(value, (int, float, np.number)) and
                                 not isinstance(value, (bool, np.bool_)) and np.isfinite(value)
                                 for value in action.values()), "policy targets must be finite numbers")
        except Exception as exc:
            if not self.stop.stopped:
                self.stop.trip(str(exc) or type(exc).__name__)
            raise SafetyStop(str(exc)) from exc
        self.iterations += 1
        return actions


def fresh_evidence(workspace):
    ev = gate.evidence(workspace)
    return gate.Evidence(ev.workspace, test_only=ev.test_only)


def _runner_inputs(record):
    envelope = read_json(record["reason"].encode())
    gate.require(isinstance(envelope.get("runner_inputs"), dict), "reviewed runner inputs missing")
    return envelope["runner_inputs"]


def _check_inputs(ev, snapshot, stage):
    if stage != "review":
        review = ev.json(ev.json("release-review.json")["review_preflight"])
        gate.require(_runner_inputs(review) == snapshot,
                     "controller, source, inputs or safety readiness changed; renewed review required")


def collect_preflight(workspace, *, stage, attempt, runtime_source, previous=None, reason="", clock=utc_now):
    ev = fresh_evidence(workspace)
    gate.require(type(attempt) is int and 1 <= attempt <= 9999 and stage in gate.STAGES,
                 "explicit stage and positive attempt required")
    if attempt > 1:
        gate.require(previous is not None, "renewal requires an explicit previous preflight")
        gate.text(reason, "explicit preflight renewal reason")
    path = f"preflights/{stage}-{attempt:04d}.json"
    gate.require(not contained(ev.workspace, path).exists(), "immutable preflight path already exists")
    started = clock()
    release = gate.validate_release_evidence(ev)
    if stage != "review":
        gate.validate_live_approval(ev)
    if isinstance(previous, str):
        previous = ev.reference(previous)
    # Bind local controller, inputs and safety facts inside the existing reason
    # field. The exact review preflight hash is already in the approval chain;
    # this envelope adds no shared schema and no mutable sidecar dependency.
    envelope = {"reason": reason, "runner_inputs": {}}
    runtime = {"attestation": {}, "host": {}, "request": {}}
    record = gate.PreflightRecord(
        session=release["session"], stage=stage, attempt=attempt, started_at=started,
        ended_at=started, status="failed", configuration_fingerprint=release["configuration_fingerprint"],
        calibration_sha256=release["calibration_sha256"], attestation={}, attestation_sha256=digest({}),
        host={}, request={}, previous=previous, reason=canonical(envelope).decode(),
        approval=ev.reference("live-approval.json") if stage != "review" else None,
        review_preflight=ev.json("release-review.json")["review_preflight"] if stage != "review" else None,
        evidence_kind="test_only" if ev.test_only else "real_model",
    )
    # Validate predecessor/path chronology before any request. Failed attempts
    # cannot stand in for successful stage transitions.
    gate._preflight_record(ev, vars(record), path=path, success=False)
    try:
        envelope["runner_inputs"] = copy.deepcopy(runtime_source.inputs())
        _check_inputs(ev, envelope["runner_inputs"], stage)
        runtime = copy.deepcopy(runtime_source.collect())
        record = replace(record, ended_at=clock(), status="complete", attestation=runtime["attestation"],
                         attestation_sha256=digest(runtime["attestation"]), host=runtime["host"],
                         request=runtime["request"], reason=canonical(envelope).decode())
        reference = gate.write_preflight_record(ev, record)
    except Exception as exc:
        envelope["error"] = type(exc).__name__ + ": " + str(exc)
        failed = replace(record, ended_at=clock(), status="not_run" if isinstance(exc, (PrerequisiteError, FileNotFoundError))
                         else "failed", reason=canonical(envelope).decode())
        if not contained(ev.workspace, path).exists():
            gate.write_preflight_record(fresh_evidence(ev), failed)
        raise
    return reference, runtime


def preflight(workspace, *, stage, attempt, runtime_source, previous_preflight=None, reason="", clock=utc_now):
    if stage == "live" and attempt == 1 and previous_preflight is None:
        ev = fresh_evidence(workspace)
        gate.validate_live_approval(ev)
        previous_preflight = ev.json("release-review.json")["review_preflight"]
        reason = reason or "Approved live freshness after the exact reviewed preflight"
    return collect_preflight(workspace, stage=stage, attempt=attempt, runtime_source=runtime_source,
                             previous=previous_preflight, reason=reason, clock=clock)[0]


def _release(workspace, reference, stage, runtime, clock, runtime_source=None):
    ev = fresh_evidence(workspace)
    calibration = gate.validate_release_evidence(ev)["calibration_sha256"]
    if runtime_source is not None:
        # File hashing may be slow: do it before the final freshness clock,
        # then check the selected process/request again at the release boundary.
        current = runtime_source.inputs()
        record = gate.validate_preflight_record(ev, reference, expected_stage=stage)
        gate.require(_runner_inputs(record) == current, "persisted controller/safety inputs changed")
        _check_inputs(ev, current, stage)
        gate.require(runtime_source.current() == runtime, "runtime changed during release checks")
    return gate.assert_live_release(ev, reference, expected_stage=stage, runtime=runtime,
                                    current_calibration_sha256=calibration, now=clock())


def validate_loaded_calibration(controller, approved):
    expected = approved["calibration_mapping"]
    gate.require(isinstance(expected, dict) and bool(expected), "approved calibration mapping missing")
    robot = controller.robot
    path = Path(robot.calibration_fpath).resolve(strict=True)
    gate.require(str(path) == approved["calibration_path"], "loaded calibration path differs from approval")
    gate.require(sha256_file(path) == approved["calibration_sha256"], "loaded calibration file changed")
    for name, mapping in (("follower", robot.calibration), ("bus", robot.bus.calibration)):
        gate.require(isinstance(mapping, dict) and
                     {joint: asdict(value) for joint, value in mapping.items()} == expected,
                     name + " loaded calibration mapping differs from approval")
    gate.require(set(robot.bus.motors) == set(expected) and
                 all(motor.id == expected[name]["id"] for name, motor in robot.bus.motors.items()),
                 "bus motor/calibration identity differs from approval")


def run(workspace, *, preflight, preflight_attempt, runtime_source,
        controller_factory=StopGuardedController, policy_factory=None, observe=None,
        clock=utc_now, stop=None, settings=None, previous_preflight=None, reason=""):
    ev = fresh_evidence(workspace)
    if preflight_attempt > 1:
        gate.require(previous_preflight is not None, "run renewal requires explicit previous preflight")
        gate.text(reason, "explicit run renewal reason")
    stop = stop or StopLatch()
    gate.require(not (ev.workspace / "live-run.json").exists(), "a scored run cannot be repeated in this workspace")
    live = ev.reference(preflight) if isinstance(preflight, str) else preflight
    gate.require(ev.test_only or (controller_factory is StopGuardedController and policy_factory is None and observe is None),
                 "injected hardware/policy/observations require explicit test_only evidence")
    selected_runtime = runtime_source.current()
    _release(ev, live, "live", selected_runtime, clock, runtime_source)
    selected_identity = runtime_identity(selected_runtime)
    # Construct from the exact validated input snapshot.
    expected_settings = _runner_inputs(ev.json(live))["controller"]
    if settings is not None:
        gate.require(settings == expected_settings, "controller settings differ from reviewed inputs")
    settings = expected_settings
    policy_factory = policy_factory or (lambda: runtime_source.policy)
    observe = observe or observe_trial
    construction, rv = collect_preflight(ev, stage="run", attempt=preflight_attempt,
                                         runtime_source=runtime_source, previous=previous_preflight or live,
                                         reason=reason or "Approved construction preflight", clock=clock)
    gate.require(runtime_identity(rv) == selected_identity,
                 "runtime instance/load changed before construction; explicit renewal required")
    _release(ev, construction, "run", rv, clock, runtime_source)
    result = {**ev.identity(), "schema_version": 1,
              "evidence_kind": "test_only" if ev.test_only else "real_model",
              "status": "failed", "instruction": INSTRUCTION, "approval": ev.reference("live-approval.json"),
              "live_preflight": live, "run_preflight": construction, "trials": [],
              "preflights": [live, construction]}
    controller = policy = None
    previous = construction
    with armed_stop(stop):
        try:
            stop.check()
            result["constructed_at"] = result["started_at"] = clock()
            controller = controller_factory(stop=stop, **(settings or {}))
            approved_inputs = _runner_inputs(ev.json(construction))
            if not ev.test_only or "calibration_mapping" in approved_inputs:
                validate_loaded_calibration(controller, approved_inputs)
            controller.connect(calibrate=False)
            stop.check()
            policy = policy_factory()
            for index in range(1, TRIALS + 1):
                stop.check()
                stage = f"trial-{index:02d}"
                reference, rv = collect_preflight(ev, stage=stage, attempt=1,
                                                  runtime_source=runtime_source, previous=previous,
                                                  reason="Fresh preflight before trial reset", clock=clock)
                gate.require(runtime_identity(rv) == selected_identity,
                             "runtime instance/load changed before trial; explicit renewal required")
                _release(ev, reference, stage, rv, clock, runtime_source)
                result["preflights"].append(reference)
                checked = CheckedPolicy(policy, stop, runtime_source=runtime_source, runtime=rv)
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
                    gate.require(isinstance(labels, dict) and set(labels) <=
                                 {"coherent", "wrong_target", "erratic", "grasp", "operator", "stop_reason"},
                                 "observation cannot override protected trial evidence")
                    for key in ("coherent", "wrong_target", "erratic", "grasp"):
                        gate.require(type(labels.get(key)) is bool, "named boolean judgment required: " + key)
                    gate.text(labels.get("operator"), "operator identity")
                    gate.require(isinstance(labels.get("stop_reason", ""), str), "stop reason must be text")
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



def observe_trial(index, *, prompt=input):
    # Observations score direction, never authorize motion. No default answers.
    def named(question):
        while True:
            value = prompt(question).strip()
            if value:
                return value

    def boolean(question):
        while True:
            value = prompt(question + " [yes/no]: ").strip().lower()
            if value in ("yes", "no"):
                return value == "yes"

    print(f"Trial {index}/3: report any aggressive swing or table strike as a safety stop.")
    return {
        "operator": named("Observer name: "),
        "coherent": boolean("Coherent movement toward the banana"),
        "wrong_target": boolean("Moved toward the wrong target"),
        "erratic": boolean("Erratic motion"),
        "grasp": boolean("Completed grasp (optional for directional success)"),
        "stop_reason": prompt("Safety stop reason (blank only if none): ").strip(),
    }


def controller_inputs(workspace):
    from embodiment.so_arm10x.controller import (
        DUME_PID, resolve_calibration_file, resolve_max_relative_target, resolve_use_degrees,
    )
    from scripts.run_pick_baseline import DEFAULT_ROBOT_ID, DEFAULT_ROBOT_TYPE
    ev = fresh_evidence(workspace)
    release = gate.validate_release_evidence(ev)
    candidate = Path(os.getenv("DUME_CONFIG") or ROOT / "my-dum-e.yaml").expanduser()
    block, config_sha = {}, None
    if candidate.is_file():
        import yaml
        data = capture_bytes(candidate)
        loaded = yaml.safe_load(data) or {}
        gate.require(isinstance(loaded, dict), "controller configuration must be a mapping")
        block = loaded.get("controller", {})
        gate.require(isinstance(block, dict), "controller configuration block must be a mapping")
        config_sha = hashlib.sha256(data).hexdigest()

    def pick(key, fallback):
        return block[key] if block.get(key) is not None else fallback

    # Same Phase 5 precedence/defaults, resolved only from the captured bytes.
    settings = {
        "robot_type": pick("robot_type", DEFAULT_ROBOT_TYPE),
        "robot_id": pick("robot_id", DEFAULT_ROBOT_ID),
        "robot_port": pick("robot_port", os.getenv("SO_ARM_PORT")),
        "wrist_cam_idx": int(pick("wrist_cam_idx", 0)),
        "front_cam_idx": int(pick("front_cam_idx", 1)),
    }
    gate.require(settings["robot_port"] and settings["robot_type"] in ("so100_follower", "so101_follower"),
                 "configured follower identity and serial port required (no serial access in preflight)")
    settings["use_degrees"] = resolve_use_degrees(block.get("use_degrees"))
    settings["max_relative_target"] = resolve_max_relative_target(block.get("max_relative_target"))
    gate.require(settings["use_degrees"] is True and settings["max_relative_target"] == 160.0,
                 "Phase 5 units and clamp must remain unchanged")
    calibration_path = resolve_calibration_file("so_follower", settings["robot_id"]).resolve(strict=True)
    calibration_bytes = capture_bytes(calibration_path)
    gate.require(hashlib.sha256(calibration_bytes).hexdigest() == release["calibration_sha256"],
                 "resolved controller calibration changed")
    calibration_mapping = read_json(calibration_bytes)
    gate.require(isinstance(calibration_mapping, dict), "calibration must be a mapping")
    reference = ev.json("session.json").get("calibration_reference")
    calibration = (read_json(capture_bytes(Path(reference["path"]))) if reference else ev.json("calibration.json"))
    identity = calibration["robot_identity"]
    gate.require(all(identity[key] == settings[key] for key in ("robot_id", "robot_type")),
                 "controller identity differs from approved calibration")
    return {"controller": settings, "config_sha256": config_sha,
            "calibration_path": str(calibration_path), "calibration_sha256": release["calibration_sha256"],
            "calibration_mapping": calibration_mapping,
            "pid": DUME_PID}


class RuntimeSource:
    # Attach to an already loaded, opt-in attested server. No model load or
    # SendPolicyInstructions occurs here; source/profile drift refuses reuse.
    def __init__(self, args):
        self.args = args
        self.workspace = Path(args.workspace)
        self.attestation_path = Path(args.attestation or self.workspace / "runtime/lerobot.json")
        self._policy = None
        self._runtime = None
        self.selected = getattr(args, "preflight", None)

    def inputs(self):
        from policy_guard.replay_contract import CAMERA_ORDER
        audit_dispatch_paths()
        gate.require(threading.current_thread() is threading.main_thread(),
                     "main-thread signal stop handling must be available")
        gate.require(os.getenv("DUME_POLICY_BACKEND", "lerobot") == "lerobot", "live runner requires reviewed LeRobot backend")
        ev = fresh_evidence(self.workspace)
        lock = load_input_lock(self.args.corpus, self.args.checkpoint)
        gate.require(lock == ev.json("input-lock.json"), "current frozen/checkpoint inputs changed")
        snapshot = controller_inputs(ev)
        sources = (
            "scripts/run_checkpoint_sanity.py", "embodiment/so_arm10x/controller.py",
            "embodiment/so_arm10x/skills.py", "policy/lerobot/backend.py",
            "policy/lerobot/session.py", "policy/lerobot/features.py", "policy_guard/parity_gate.py",
        )
        if gate.is_milestone_release(ev):
            sources += ("policy_guard/milestone_acceptance.py", "policy_guard/milestone_golden.py",
                        "scripts/replay_milestone_golden.py")
        snapshot.update(
            input_fingerprint=lock["fingerprint"], source_files={name: sha256_file(ROOT / name) for name in sources},
            client_lock_sha256=sha256_file(ROOT / "uv.lock"),
            client_packages={dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                             if dist.metadata["Name"]},
            camera_order=list(CAMERA_ORDER), joint_order=list(JOINT_ORDER),
            endpoint=self.args.endpoint, checkpoint_mount=self.args.checkpoint_mount,
            safety={"signal_stop": True, "clamp_observer": True, "dispatch_audit": dict(AUDITED_SOURCES)},
            protocol={"trials": TRIALS, "iterations": ITERATIONS, "actions": ACTIONS,
                      "action_delay": ACTION_DELAY, "instruction": INSTRUCTION},
        )
        return snapshot

    @property
    def policy(self):
        if self._policy is None:
            from policy.lerobot.backend import LeRobotPolicyBackend
            from policy.lerobot.session import LeRobotPolicySession
            source = self

            class ObservedSession(LeRobotPolicySession):
                def infer(self, observation):
                    started = utc_now()
                    result = super().infer(observation)
                    ended = utc_now()
                    gate.require(len(result) == ACTIONS, "exactly 16 timed actions required")
                    decoded = np.stack([item.get_action().detach().cpu().float().numpy() for item in result])
                    gate.require(decoded.shape == (16, 6) and np.isfinite(decoded).all(), "finite full chunk required")
                    attestation = read_json(capture_bytes(source.attestation_path))
                    request = {
                        "observation_sha256": gate.observation_fingerprint(observation),
                        "timestamp": result[0].get_timestamp(), "timestep": result[0].get_timestep(),
                        "started_at": attestation["request"]["started_at"],
                        "completed_at": attestation["request"]["completed_at"],
                        "output_sha256": gate.array_fingerprint(decoded), "decoded_shape": [16, 6],
                    }
                    gate.require(gate.timestamp(started) <= gate.timestamp(request["started_at"]) <=
                                 gate.timestamp(request["completed_at"]) <= gate.timestamp(ended),
                                 "attestation does not describe this request")
                    host = gate.collect_runtime_host(attestation, source.args.container, source.args.endpoint,
                                                     source.args.checkpoint_mount)
                    gate.validate_runtime_attestation(
                        attestation, host=host, request=request,
                        expected_configuration=gate.Evidence(source.workspace).json("profiles.json")["serving_configuration"],
                        now=utc_now(),
                    )
                    observed = {"attestation": attestation, "host": host, "request": request}
                    if source._runtime is not None:
                        gate.require(runtime_identity(observed) == runtime_identity(source._runtime),
                                     "runtime instance/load changed during inference; explicit renewal required")
                    source._runtime = observed
                    return result

            class AttachedPolicy(LeRobotPolicyBackend):
                def _handshake(self):
                    # Ready clears the per-client queue and retains loaded weights.
                    attestation = read_json(capture_bytes(source.attestation_path))
                    expected = gate.Evidence(source.workspace).json("profiles.json")["serving_configuration"]
                    gate.require(attestation["status"] == "complete" and
                                 attestation["semantic_configuration"] == expected,
                                 "an already loaded and attested reviewed service is required")
                    self._session.probe_ready_or_raise()
                    self._handshaken = True

            gate.require(self.args.endpoint.startswith("127.0.0.1:"), "only loopback serving is permitted")
            host, port = self.args.endpoint.rsplit(":", 1)
            policy = AttachedPolicy(host=host, port=int(port), camera_keys=["front", "wrist"],
                                    robot_state_keys=list(JOINT_ORDER), language_instruction=INSTRUCTION)
            policy._session.close()
            policy._session = ObservedSession(self.args.endpoint, max_attempts=1, handshake_max_attempts=1)
            self._policy = policy
        return self._policy

    def collect(self):
        ev = gate.Evidence(self.workspace)
        lock = ev.json("input-lock.json")
        arrays, record = load_case(Path(self.args.corpus), lock, lock["schedule"][0])
        observation = {name: float(value) for name, value in zip(JOINT_ORDER, arrays["state"], strict=True)}
        observation.update(front=arrays["video_front"], wrist=arrays["video_wrist"])
        self.policy.get_action(observation, record["instruction"])
        return copy.deepcopy(self._runtime)

    def current(self):
        if self._runtime is None:
            gate.require(self.selected is not None, "explicit live preflight required")
            record = gate.validate_preflight_record(self.workspace, self.selected, expected_stage="live")
            self._runtime = {key: record[key] for key in ("attestation", "host", "request")}
        attestation = read_json(capture_bytes(self.attestation_path))
        gate.require(attestation == self._runtime["attestation"], "runtime changed; create an explicit new live preflight")
        host = gate.collect_runtime_host(attestation, self.args.container, self.args.endpoint, self.args.checkpoint_mount)
        gate.require({k: v for k, v in host.items() if k != "checked_at"} ==
                     {k: v for k, v in self._runtime["host"].items() if k != "checked_at"},
                     "current host/process identity changed")
        # Return the exact historical check only after independently comparing
        # current host facts. Its original age remains subject to the 60s gate.
        return copy.deepcopy(self._runtime)

    def close(self):
        if self._policy is not None:
            self._policy.close()


def check(workspace):
    ev = fresh_evidence(workspace)
    gate.validate_live_approval(ev)
    if not (ev.workspace / "live-run.json").exists():
        return {"status": "complete", "stage": "live-approval-check", "hardware_verified": False}
    result = ev.record("live-run.json")
    gate.validate_run_safety_journal(result)
    gate.require(result["approval"] == ev.reference("live-approval.json"), "run approval changed")
    gate.require(result["instruction"] == INSTRUCTION and result["safety_stop"] is False and
                 result["clamp_warnings"] == 0 and len(result["trials"]) == TRIALS,
                 "incomplete or stopped directional run")
    links = [result["live_preflight"], result["run_preflight"], *(t["preflight"] for t in result["trials"])]
    gate.require(result["preflights"] == links, "ordered preflight links changed")
    stages = ["live", "run", "trial-01", "trial-02", "trial-03"]
    previous = None
    for stage, reference in zip(stages, links, strict=True):
        record = gate.validate_preflight_record(ev, reference, expected_stage=stage)
        if previous is not None:
            gate.require(record["previous"] == previous, "preflight predecessor changed")
        _check_inputs(ev, _runner_inputs(record), stage)
        previous = reference
    construction = ev.json(result["run_preflight"])
    gate.require(gate.timestamp(construction["ended_at"]) < gate.timestamp(result["constructed_at"]) ==
                 gate.timestamp(result["started_at"]), "construction chronology changed")
    last_end = result["constructed_at"]
    for index, trial in enumerate(result["trials"], 1):
        record = ev.json(trial["preflight"])
        gate.require(gate.timestamp(last_end) < gate.timestamp(record["started_at"]) <=
                     gate.timestamp(record["ended_at"]) < gate.timestamp(trial["started_at"]) <=
                     gate.timestamp(trial["ended_at"]) <= gate.timestamp(result["ended_at"]),
                     "trial/reset chronology changed")
        last_end = trial["ended_at"]
        gate.text(trial["operator"], "operator identity")
        gate.require(all(type(trial[key]) is bool for key in ("coherent", "wrong_target", "erratic", "grasp")),
                     "named boolean observations required")
        gate.require(trial["index"] == index and trial["iterations"] == ITERATIONS and
                     trial["actions_per_chunk"] == ACTIONS and trial["action_delay"] == ACTION_DELAY and
                     trial["safety_stop"] is False and trial["clamp_warnings"] == 0,
                     "trial denominator/budget/safety changed")
    gate.require(sum(t["coherent"] and not t["wrong_target"] and not t["erratic"] for t in result["trials"]) >= 2,
                 "fewer than two directional successes")
    return {"status": "complete", "stage": "directional-run-check", "live_run": ev.reference("live-run.json")}


def build_parser():
    parser = argparse.ArgumentParser(description="Approval-gated three-trial checkpoint sanity runner")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "run", "check"):
        command = commands.add_parser(name)
        command.add_argument("--workspace", type=Path, default=ROOT / "corpus/phase7")
        if name == "check":
            continue
        command.add_argument("--corpus", type=Path, default=ROOT / "corpus/frozen_v1_0")
        command.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/GR00T-N1.7-3B-SO101")
        command.add_argument("--attestation", type=Path)
        command.add_argument("--container", default="lerobot-policy")
        command.add_argument("--endpoint", default="127.0.0.1:8080")
        command.add_argument("--checkpoint-mount", default="/checkpoints/model")
        command.add_argument("--previous-preflight")
        command.add_argument("--reason", default="")
        if name == "preflight":
            command.add_argument("--stage", choices=("review", "live"), required=True)
            command.add_argument("--attempt", type=int, required=True)
        else:
            command.add_argument("--preflight", required=True)
            command.add_argument("--preflight-attempt", type=int, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    source = None
    try:
        # Validate before creating any runtime client, including on fixture files.
        if args.command == "check":
            result = check(args.workspace)
        else:
            gate.validate_release_evidence(args.workspace)
            if args.command == "run" or args.stage == "live":
                gate.validate_live_approval(args.workspace)
            if args.command == "run" and not sys.stdin.isatty():
                raise PrerequisiteError("not run: an interactive named observer is required")
            source = RuntimeSource(args)
            if args.command == "preflight":
                reference = preflight(args.workspace, stage=args.stage, attempt=args.attempt,
                                      previous_preflight=args.previous_preflight, reason=args.reason,
                                      runtime_source=source)
                result = {"status": "complete", "preflight": reference, "hardware_verified": False}
            else:
                result = run(args.workspace, preflight=args.preflight, preflight_attempt=args.preflight_attempt,
                             runtime_source=source, previous_preflight=args.previous_preflight, reason=args.reason)
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "complete" else 1
    except (PrerequisiteError, FileNotFoundError) as exc:
        print(json.dumps({"status": "not_run", "error": "not run: " + str(exc)}))
        return 2
    except (Exception, KeyboardInterrupt) as exc:
        print(json.dumps({"status": "failed", "error": type(exc).__name__ + ": " + str(exc)}))
        return 1
    finally:
        if source is not None:
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
