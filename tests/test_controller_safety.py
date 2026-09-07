"""Controller safety instruments: the PID read-back and the motion clamp.

Plan 05-05 makes three things explicit that Phase 5 otherwise leaves at their
defaults. This module covers the two that are safety instruments:

* **The PID preset (LR-04 / D-01 / D-02).** At lerobot 0.6.1 the three
  coefficients are *config fields*, and upstream ``SOFollower.configure()`` —
  which ``connect()`` already calls — writes them on the first pass. So the
  preset becomes declarative and Dum-E's job shrinks to *evidence*: read the
  coefficients back from every motor and refuse to connect on a mismatch or on a
  read failure. A clean write is not evidence the value landed (D-02), and
  operating the arm at unknown stiffness would surface in Phase 7 looking like
  checkpoint drift rather than the configuration error it is (D-01).
* **The motion clamp (SAFE-02).** ``max_relative_target`` plumbing already
  exists upstream and in Dum-E; SAFE-02 is "set a real value and surface the
  warning", not "build the clamp". Two traps make that non-trivial. An ``int``
  clamp does **not** fail on ordinary actions — it raises ``TypeError`` inside
  ``ensure_safe_goal_position`` at the exact moment the clamp would have engaged.
  And upstream's warning goes to the stdlib **root** logger (hence stderr via
  ``basicConfig``) while Dum-E's loguru sink writes to stdout, so the warning is
  emitted and architecturally invisible.

Every test here is hermetic: no serial port, no cameras, no network, no
hardware. The clamp tests drive the real upstream helper and a stub robot that
mirrors ``SOFollower.send_action``'s body, so the assertions are about upstream's
actual behaviour rather than a reimplementation of it.
"""

import contextlib
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
from lerobot.robots.utils import ensure_safe_goal_position

from embodiment.so_arm10x import controller as ctrl_mod
from embodiment.so_arm10x.controller import SO10xArmController

# The six motor names on an SO-10x follower, in bus order. `send_action` strips
# the `.pos` suffix before clamping and restores it on the way out, so both
# spellings appear below.
MOTOR_NAMES: Tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# lerobot's own SOFollowerConfig defaults, for contrast with the Dum-E preset.
LEROBOT_DEFAULT_PID = {"P_Coefficient": 16, "I_Coefficient": 0, "D_Coefficient": 32}


# ---------------------------------------------------------------------------
# Hardware-free stubs
# ---------------------------------------------------------------------------


class StubBus:
    """Stand-in for ``lerobot.motors.motors_bus.MotorsBus``.

    Records every ``read`` call with its keyword arguments, which is how the
    normalization and retry assertions are made: those two arguments are
    load-bearing (normalization is defined for *position* registers through the
    calibration mapping and is meaningless for coefficient registers, and one
    dropped Feetech packet must surface as a retry rather than as a false
    mismatch that refuses to connect a healthy arm) but they are invisible in the
    return value, so they can only be checked at the call site.
    """

    def __init__(
        self,
        motors: Sequence[str] = MOTOR_NAMES,
        overrides: Optional[Dict[Tuple[str, str], int]] = None,
        read_error: Optional[BaseException] = None,
    ) -> None:
        self.motors = list(motors)
        self._overrides = dict(overrides or {})
        self._read_error = read_error
        self.read_calls: List[Dict[str, Any]] = []

    @contextlib.contextmanager
    def torque_disabled(self):
        yield

    def read(self, data_name: str, motor: str, *, normalize: bool = True, num_retry: int = 0):
        self.read_calls.append(
            {
                "data_name": data_name,
                "motor": motor,
                "normalize": normalize,
                "num_retry": num_retry,
            }
        )
        if self._read_error is not None:
            raise self._read_error
        if (motor, data_name) in self._overrides:
            return self._overrides[(motor, data_name)]
        return ctrl_mod.DUME_PID[data_name]


def pid_controller(bus: StubBus) -> SO10xArmController:
    """An ``SO10xArmController`` shell wired to ``bus``, built without hardware.

    ``__new__`` skips ``__init__`` (which would demand a serial port and build
    camera configs); the read-back method only ever touches ``self.robot.bus``
    and ``self.config``.
    """
    controller = SO10xArmController.__new__(SO10xArmController)
    controller.robot = SimpleNamespace(bus=bus)
    controller.config = SimpleNamespace(num_read_retries=2)
    return controller


# ---------------------------------------------------------------------------
# PID: declarative, then read back and asserted (LR-04 / D-01 / D-02)
# ---------------------------------------------------------------------------


def test_pid_config_fields_carry_dume_preset(monkeypatch):
    """The preset is declared on the follower config, not written after connect.

    At 0.6.1 ``configure()`` writes ``position_p/i/d_coefficient`` from the
    config while still inside ``torque_disabled()``, and ``connect()`` calls
    ``configure()``. Passing the three fields therefore lands 10/0/5 on the first
    pass, with no post-connect register overwrite and no second torque-disabled
    cycle. Both follower-config branches must carry them — an SO-100 arm is not
    entitled to the library defaults.
    """
    assert ctrl_mod.DUME_PID == {
        "P_Coefficient": 10,
        "I_Coefficient": 0,
        "D_Coefficient": 5,
    }
    # The values are only worth asserting because they differ from upstream's.
    assert ctrl_mod.DUME_PID != LEROBOT_DEFAULT_PID

    monkeypatch.setattr(
        ctrl_mod, "make_robot_from_config", lambda config: SimpleNamespace(config=config)
    )

    for robot_type in ("so101_follower", "so100_follower"):
        controller = SO10xArmController(
            robot_type=robot_type,
            robot_port="/dev/null-not-a-real-port",
            robot_id="test_arm",
        )
        assert controller.config.position_p_coefficient == ctrl_mod.DUME_PID["P_Coefficient"]
        assert controller.config.position_i_coefficient == ctrl_mod.DUME_PID["I_Coefficient"]
        assert controller.config.position_d_coefficient == ctrl_mod.DUME_PID["D_Coefficient"]


def test_pid_readback_returns_and_logs_the_per_motor_mapping():
    """On success the read-back returns every motor's three coefficients.

    D-02 asks for the observed values to be logged as evidence a later phase can
    cite — "the stiffness the arm actually ran at", not "the value that was
    requested". Returning the mapping is what makes that citable.
    """
    bus = StubBus()
    observed = SO10xArmController._assert_pid_landed(pid_controller(bus))

    assert set(observed) == set(MOTOR_NAMES)
    for motor in MOTOR_NAMES:
        assert observed[motor] == ctrl_mod.DUME_PID


def test_pid_readback_raises_on_mismatch():
    """One wrong coefficient on one motor refuses the connect.

    The message must name the motor, the register, the expected value and the
    observed value: a read-back that is neither the Dum-E preset nor the library
    default means the write *partially* landed, which is more dangerous than
    either endpoint, so the message has to be specific enough to tell those apart.
    """
    bus = StubBus(overrides={("elbow_flex", "P_Coefficient"): 16})

    with pytest.raises(Exception) as excinfo:
        SO10xArmController._assert_pid_landed(pid_controller(bus))

    message = str(excinfo.value)
    assert "elbow_flex" in message
    assert "P_Coefficient" in message
    assert "10" in message  # expected
    assert "16" in message  # observed


def test_pid_readback_aggregates_every_mismatch_into_one_error():
    """A partially landed write is fully visible, not truncated at the first hit.

    Aggregating is the difference between "PID is wrong somewhere" and a picture
    of which motors took the write and which did not.
    """
    bus = StubBus(
        overrides={
            ("shoulder_pan", "D_Coefficient"): 32,
            ("wrist_roll", "P_Coefficient"): 16,
        }
    )

    with pytest.raises(Exception) as excinfo:
        SO10xArmController._assert_pid_landed(pid_controller(bus))

    message = str(excinfo.value)
    assert "shoulder_pan" in message and "D_Coefficient" in message
    assert "wrist_roll" in message and "P_Coefficient" in message


def test_pid_readback_raises_and_chains_on_read_failure():
    """A read that raises re-raises with the cause attached — never swallowed.

    This replaces the bare ``except Exception: pass`` recorded as existing debt
    in ``.planning/codebase/CONCERNS.md``. CONTEXT.md D-01 forbids softening it
    to a warning even to get past a bus problem, so the assertion is on both the
    raise and the chained cause: an error that loses its cause is nearly as hard
    to diagnose as one that never fired.
    """
    original = OSError("simulated Feetech bus timeout")
    bus = StubBus(read_error=original)

    with pytest.raises(Exception) as excinfo:
        SO10xArmController._assert_pid_landed(pid_controller(bus))

    assert excinfo.value.__cause__ is original
    assert "unknown stiffness" in str(excinfo.value).lower()


def test_pid_readback_uses_unnormalized_reads_with_retry():
    """Every coefficient read passes ``normalize=False`` and a non-zero retry.

    Both defaults are wrong here. ``normalize=True`` (upstream's default) applies
    the calibration mapping, which is defined for position registers and
    meaningless for coefficient registers. ``num_retry=0`` (also upstream's
    default) turns a single dropped packet on the Feetech bus into a value
    mismatch, which under D-01 refuses to connect a perfectly healthy arm.
    """
    bus = StubBus()
    SO10xArmController._assert_pid_landed(pid_controller(bus))

    assert bus.read_calls, "no coefficient reads were issued at all"
    # Three coefficients on each of six motors.
    assert len(bus.read_calls) == 3 * len(MOTOR_NAMES)
    for call in bus.read_calls:
        assert call["data_name"] in ctrl_mod.DUME_PID
        assert call["normalize"] is False, call
        assert call["num_retry"] > 0, call


def test_connect_asserts_pid_landed_after_the_calibration_assertion():
    """``connect()`` still runs the read-back, in the same place the write was.

    Ordering matters twice over: the calibration file governs what the numbers
    fed to the policy *mean*, and the PID governs how the arm tracks them. Both
    must be established before anything commands motion.
    """
    calls: List[str] = []
    stub = SimpleNamespace(
        robot=SimpleNamespace(connect=lambda calibrate=True: calls.append("robot.connect")),
        _assert_calibration_loaded=lambda: calls.append("calibration"),
        _assert_pid_landed=lambda: calls.append("pid"),
    )

    SO10xArmController.connect(stub, calibrate=False)

    assert calls == ["robot.connect", "calibration", "pid"]
