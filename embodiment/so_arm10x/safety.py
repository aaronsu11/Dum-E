"""SO101 stop latching, audited dispatch and calibration identity checks."""
import hashlib, importlib, inspect, logging, signal, threading
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from loguru import logger
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.robots.so_follower import SOFollower
from embodiment.so_arm10x.controller import CLAMP_WARNING_TEXT, SO10xArmController
from policy.evidence import require, now as utc_now, fingerprint_configuration as digest, sha256_file

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
        require(hashlib.sha256(source.read_bytes()).hexdigest() == expected,
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
        require(not original.is_connected, "bus must be unconnected during composition")
        self.bus = StopGuardedBus(
            port=original.port, motors=original.motors, calibration=original.calibration,
            protocol_version=original.protocol_version, stop=stop,
        )

    def connect(self, calibrate=False):
        self.stop.check()
        require(calibrate is False, "live execution never recalibrates hardware")
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
        require(calibrate is False, "calibration is an offline prerequisite")
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


def validate_loaded_calibration(controller, approved):
    expected = approved["calibration_mapping"]
    require(isinstance(expected, dict) and bool(expected), "approved calibration mapping missing")
    robot = controller.robot
    path = Path(robot.calibration_fpath).resolve(strict=True)
    require(str(path) == approved["calibration_path"], "loaded calibration path differs from approval")
    require(sha256_file(path) == approved["calibration_sha256"], "loaded calibration file changed")
    for name, mapping in (("follower", robot.calibration), ("bus", robot.bus.calibration)):
        require(isinstance(mapping, dict) and
                     {joint: asdict(value) for joint, value in mapping.items()} == expected,
                     name + " loaded calibration mapping differs from approval")
    require(set(robot.bus.motors) == set(expected) and
                 all(motor.id == expected[name]["id"] for name, motor in robot.bus.motors.items()),
                 "bus motor/calibration identity differs from approval")
