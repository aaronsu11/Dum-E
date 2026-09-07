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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pytest
from loguru import logger
from lerobot.robots.utils import ensure_safe_goal_position

import utils
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


class StubRobot:
    """Stand-in for ``SOFollower`` whose ``send_action`` mirrors upstream's body.

    It calls the **real** ``ensure_safe_goal_position`` with the real config
    value, so every clamp assertion here is about upstream's actual arithmetic
    rather than a reimplementation of it — including the ``.pos`` suffix strip on
    the way in and restore on the way out, which is what makes the requested and
    returned dicts comparable key-for-key.
    """

    name = "so_follower"

    def __init__(
        self,
        present_pos: Optional[Dict[str, float]] = None,
        max_relative_target: Any = None,
    ) -> None:
        self.present_pos = dict(
            present_pos if present_pos is not None else {m: 0.0 for m in MOTOR_NAMES}
        )
        self.config = SimpleNamespace(
            max_relative_target=max_relative_target, num_read_retries=2
        )
        self.sent_actions: List[Dict[str, float]] = []

    def send_action(self, action: Dict[str, float]) -> Dict[str, float]:
        goal_pos = {
            key.removesuffix(".pos"): float(val)
            for key, val in action.items()
            if key.endswith(".pos")
        }
        if self.config.max_relative_target is not None:
            goal_present_pos = {
                key: (g_pos, self.present_pos[key]) for key, g_pos in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(
                goal_present_pos, self.config.max_relative_target
            )
        self.sent_actions.append(dict(goal_pos))
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}


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


def clamp_controller(robot: StubRobot) -> SO10xArmController:
    """An ``SO10xArmController`` shell whose ``set_target_state`` drives ``robot``."""
    controller = SO10xArmController.__new__(SO10xArmController)
    controller.robot = robot
    controller.config = robot.config
    controller._state_keys = [f"{motor}.pos" for motor in MOTOR_NAMES]
    return controller


@contextlib.contextmanager
def capture_loguru(level: str = "WARNING") -> Iterator[List[str]]:
    """Collect loguru MESSAGES through a real sink, not by inspecting source text.

    Capturing the sink is the whole point: SAFE-02's claim is that the clamp is
    *surfaced on Dum-E's own stream*, and only a sink proves that. Grepping the
    source would pass even if the message never reached loguru.

    ``record["message"]`` rather than the rendered line, deliberately: the
    rendered line carries a millisecond timestamp, so no two runs could ever be
    byte-identical and the determinism assertion below would be untestable. The
    claim under test is that the *message* is deterministic, not the clock.
    """
    messages: List[str] = []
    sink_id = logger.add(
        lambda message: messages.append(message.record["message"]), level=level
    )
    try:
        yield messages
    finally:
        logger.remove(sink_id)


@contextlib.contextmanager
def stdlib_bridge_installed() -> Iterator[None]:
    """Install the stdlib->loguru bridge and remove only what this added."""
    root = logging.getLogger()
    preexisting = [h for h in root.handlers if isinstance(h, utils.InterceptHandler)]
    utils.install_stdlib_to_loguru_bridge()
    try:
        yield
    finally:
        for handler in list(root.handlers):
            if isinstance(handler, utils.InterceptHandler) and handler not in preexisting:
                root.removeHandler(handler)


# ---------------------------------------------------------------------------
# The motion clamp: a real float, on by default, and surfaced (SAFE-02)
# ---------------------------------------------------------------------------


def test_default_clamp_is_a_float_not_an_int():
    """The default clamp is a strictly positive ``float``, above the policy's own motion.

    Three separate failure modes are excluded here. ``None`` disables the clamp
    entirely, which is what Dum-E shipped with. An ``int`` is the worse trap (see
    the next test). And a value *below* the checkpoint's own trained relative
    motion would fire on nominal operation, which Phase 7 requires to be
    warning-free and would read as a parity bug rather than as a mis-set clamp.
    """
    clamp = ctrl_mod.DEFAULT_MAX_RELATIVE_TARGET

    assert isinstance(clamp, float), type(clamp)
    assert not isinstance(clamp, bool)
    assert clamp > 0.0

    # The checkpoint's largest per-timestep relative-action bound, over all 16
    # timesteps and all five arm joints (statistics.json ->
    # new_embodiment.relative_action.single_arm.max[15][1], shoulder_lift).
    # The clamp must sit strictly above it.
    checkpoint_extreme = 137.47268676757812
    assert clamp > checkpoint_extreme, (clamp, checkpoint_extreme)


def test_int_clamp_would_raise_typeerror_in_upstream_clamp():
    """An ``int`` clamp raises inside the helper — proven, not described.

    ``ensure_safe_goal_position`` dispatches on ``isinstance(..., float)`` then
    ``isinstance(..., dict)`` and otherwise raises ``TypeError``, and
    ``isinstance(5, float)`` is ``False``. This is the worst available failure
    mode because it does NOT fail on ordinary actions: an int clamp works
    perfectly right up to the moment the clamp was supposed to engage.
    """
    clamp = ctrl_mod.DEFAULT_MAX_RELATIVE_TARGET
    goal_present = {"shoulder_pan": (clamp + 50.0, 0.0)}

    # The float the module actually configures clamps cleanly...
    clipped = ensure_safe_goal_position(goal_present, clamp)
    assert clipped["shoulder_pan"] == pytest.approx(clamp)

    # ...and the same value as an int does not.
    with pytest.raises(TypeError):
        ensure_safe_goal_position(goal_present, int(clamp))


def test_clamp_exactly_at_bound_is_not_clamped():
    """A delta exactly at the bound passes through; so does one 1e-5 above it.

    Upstream's divergence test is ``abs(safe_goal_pos - goal_pos) > 1e-4``,
    strictly greater. So the bound itself is untouched, and a request 1e-5 past
    it is clipped by 1e-5 — below the threshold, therefore NOT reported. Dum-E's
    detector uses the same threshold precisely so the two never disagree about
    whether the clamp fired.
    """
    clamp = ctrl_mod.DEFAULT_MAX_RELATIVE_TARGET
    robot = StubRobot(max_relative_target=clamp)
    controller = clamp_controller(robot)

    for offset in (0.0, 1e-5):
        with capture_loguru() as lines:
            sent = SO10xArmController.set_target_state(
                controller, {"shoulder_pan.pos": clamp + offset}
            )
        assert sent["shoulder_pan.pos"] == pytest.approx(clamp, abs=1e-4)
        assert not [line for line in lines if "clamped" in line], (offset, lines)


def test_clamp_one_step_above_bound_is_clamped_and_logged():
    """A delta past the bound comes back clipped AND is reported through loguru.

    The return value is the primary detector rather than the log because it is
    log-configuration independent and testable with no hardware — which is
    exactly what this test is.
    """
    clamp = ctrl_mod.DEFAULT_MAX_RELATIVE_TARGET
    requested = clamp + 25.0
    robot = StubRobot(max_relative_target=clamp)
    controller = clamp_controller(robot)

    with capture_loguru() as lines:
        sent = SO10xArmController.set_target_state(
            controller, {"shoulder_pan.pos": requested}
        )

    assert sent["shoulder_pan.pos"] == pytest.approx(clamp)
    clamp_lines = [line for line in lines if "clamped" in line]
    assert len(clamp_lines) == 1, lines
    line = clamp_lines[0]
    assert "shoulder_pan" in line
    assert f"{requested:.4f}" in line or str(requested) in line
    assert f"{clamp:.4f}" in line or str(clamp) in line


def test_clamp_warning_reaches_loguru_sink_with_upstream_wording():
    """Dum-E's own warning carries upstream's exact sentence.

    One grep must find both signals. Upstream emits
    ``Relative goal position magnitude had to be clamped to be safe.`` on the
    stdlib root logger; Dum-E re-emits through loguru. If the wordings diverged,
    an operator grepping for one would conclude the other never happened.
    """
    clamp = ctrl_mod.DEFAULT_MAX_RELATIVE_TARGET
    robot = StubRobot(max_relative_target=clamp)
    controller = clamp_controller(robot)

    with capture_loguru() as lines:
        SO10xArmController.set_target_state(
            controller, {"elbow_flex.pos": clamp + 10.0}
        )

    assert any(
        "Relative goal position magnitude had to be clamped to be safe." in line
        for line in lines
    ), lines


def test_no_clamp_warning_when_clamp_disabled():
    """With the clamp disabled, the code path does not run and nothing is reported.

    An absent clamp must be distinguishable from a clamp that never fired. Here
    the requested action is far past any plausible bound and still comes back
    untouched, with silence on the log — which is the signature of "no clamp",
    not of "no violation".
    """
    robot = StubRobot(max_relative_target=None)
    controller = clamp_controller(robot)
    requested = {f"{motor}.pos": 500.0 for motor in MOTOR_NAMES}

    with capture_loguru() as lines:
        sent = SO10xArmController.set_target_state(controller, dict(requested))

    assert sent == pytest.approx(requested)
    assert not [line for line in lines if "clamped" in line], lines


def test_clamp_log_lists_joints_in_requested_key_order():
    """Two runs over the same requested action produce identical clamp lines.

    Determinism here is what makes the log diffable across runs. The report is
    built by walking the *requested* action's keys, so its order is the caller's
    insertion order rather than set or dict-hash order.
    """
    clamp = ctrl_mod.DEFAULT_MAX_RELATIVE_TARGET
    robot = StubRobot(max_relative_target=clamp)
    controller = clamp_controller(robot)

    requested = {
        "wrist_flex.pos": clamp + 5.0,
        "shoulder_pan.pos": clamp + 7.0,
        "elbow_flex.pos": clamp + 9.0,
    }

    runs = []
    for _ in range(2):
        with capture_loguru() as lines:
            SO10xArmController.set_target_state(controller, dict(requested))
        runs.append([line for line in lines if "clamped" in line])

    assert runs[0] == runs[1], runs
    line = runs[0][0]
    positions = [line.index(key.removesuffix(".pos")) for key in requested]
    assert positions == sorted(positions), (positions, line)

    # And the ordering follows the request, not an alphabetical or sorted rule:
    # reversing the requested keys reverses the reported order.
    reversed_requested = dict(reversed(list(requested.items())))
    with capture_loguru() as lines:
        SO10xArmController.set_target_state(controller, reversed_requested)
    reversed_line = [line for line in lines if "clamped" in line][0]
    reversed_positions = [
        reversed_line.index(key.removesuffix(".pos")) for key in reversed_requested
    ]
    assert reversed_positions == sorted(reversed_positions)
    assert reversed_line != line


def test_clamp_validation_rejects_bad_values_before_they_reach_the_config():
    """A non-positive clamp or an incomplete per-motor mapping is refused early.

    Validating at the boundary rather than at the first clamped action matters
    because the upstream mapping branch only raises when it is *used* — so an
    incomplete mapping, like an int, would fail at the exact moment the clamp was
    needed. A complete mapping and a positive float both pass through.
    """
    resolve = ctrl_mod.resolve_max_relative_target

    assert resolve(12.5) == 12.5
    complete = {motor: 12.5 for motor in MOTOR_NAMES}
    assert resolve(complete) == complete

    for bad in (0.0, -1.0, float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            resolve(bad)

    with pytest.raises(ValueError):
        resolve({"shoulder_pan": 12.5})  # incomplete mapping
    with pytest.raises(ValueError):
        resolve({**complete, "not_a_motor": 12.5})  # extra key
    with pytest.raises(ValueError):
        resolve({**complete, "gripper": -1.0})  # non-positive entry


def test_clamp_resolves_from_environment_and_is_on_by_default(monkeypatch):
    """``None`` means "resolve", not "disable" — the clamp is on by default.

    The production call site passes neither the clamp nor the units parameter, so
    an unset default that meant "off" would leave SAFE-02 unsatisfied in exactly
    the configuration that ships. Disabling is therefore explicit and spelled out
    in the environment.
    """
    monkeypatch.delenv("DUME_MAX_RELATIVE_TARGET", raising=False)
    assert ctrl_mod.resolve_max_relative_target(None) == ctrl_mod.DEFAULT_MAX_RELATIVE_TARGET

    monkeypatch.setenv("DUME_MAX_RELATIVE_TARGET", "12.5")
    assert ctrl_mod.resolve_max_relative_target(None) == 12.5

    # Disabling is deliberate and explicit.
    monkeypatch.setenv("DUME_MAX_RELATIVE_TARGET", "none")
    assert ctrl_mod.resolve_max_relative_target(None) is None

    # An unparseable value raises rather than silently disabling the clamp.
    monkeypatch.setenv("DUME_MAX_RELATIVE_TARGET", "quite-far")
    with pytest.raises(ValueError):
        ctrl_mod.resolve_max_relative_target(None)


def test_stdlib_root_warning_is_bridged_to_loguru():
    """A plain stdlib root-logger warning reaches a loguru sink, installed once.

    This closes the gap Pitfall 9 names: ``ensure_safe_goal_position`` calls the
    module-level ``logging.warning``, i.e. the ROOT logger, which auto-installs a
    stderr ``StreamHandler`` — while Dum-E's loguru sink writes to stdout. Two
    disjoint streams, and the repo had no bridge at all, so upstream's own clamp
    warning was emitted and architecturally invisible. Setting the clamp without
    this bridge does not satisfy SAFE-02.
    """
    root = logging.getLogger()

    with stdlib_bridge_installed():
        utils.install_stdlib_to_loguru_bridge()  # second call must be a no-op
        bridges = [h for h in root.handlers if isinstance(h, utils.InterceptHandler)]
        assert len(bridges) == 1, bridges

        with capture_loguru() as lines:
            logging.warning("stdlib root bridge probe marker")
            logging.getLogger("some.upstream.module").warning("named logger probe marker")

    text = "\n".join(lines)
    assert "stdlib root bridge probe marker" in text, lines
    assert "named logger probe marker" in text, lines
    # Exactly one record each: a duplicated bridge would double every line.
    assert text.count("stdlib root bridge probe marker") == 1, lines


def test_bridged_record_is_attributed_to_the_originating_frame():
    """A bridged record names its real origin, not ``utils.py`` or ``logging``.

    Frame depth is not cosmetic here. The clamp warning is emitted from inside
    ``lerobot/robots/utils.py``, and an operator who greps the log to find where
    a clamp came from must be pointed there — not at Dum-E's logging setup, and
    not at ``logging/__init__.py:callHandlers``, both of which are where a naive
    depth calculation lands.
    """
    seen: List[Dict[str, Any]] = []
    sink_id = logger.add(lambda message: seen.append(message.record), level="WARNING")
    try:
        with stdlib_bridge_installed():
            logging.getLogger("probe").warning("attribution probe marker")
    finally:
        logger.remove(sink_id)

    records = [r for r in seen if "attribution probe marker" in r["message"]]
    assert records, seen
    origin = records[0]["file"].path
    assert origin.endswith("test_controller_safety.py"), origin
    assert not origin.endswith("utils.py"), origin
    assert "logging/__init__.py" not in origin, origin


def test_upstream_clamp_warning_itself_reaches_loguru_through_the_bridge():
    """The belt-and-braces half: upstream's OWN warning lands on Dum-E's stream.

    Distinct from the return-value detector — this asserts the bridge carries the
    warning ``ensure_safe_goal_position`` emits itself, so the two mechanisms are
    genuinely independent rather than one dressed up as two.
    """
    clamp = ctrl_mod.DEFAULT_MAX_RELATIVE_TARGET

    with stdlib_bridge_installed():
        with capture_loguru() as lines:
            ensure_safe_goal_position({"shoulder_pan": (clamp + 40.0, 0.0)}, clamp)

    assert any(
        "Relative goal position magnitude had to be clamped to be safe." in line
        for line in lines
    ), lines


def test_skill_loop_consumes_set_target_state_return_value(monkeypatch):
    """The pick loop uses the returned action instead of discarding it.

    ``set_target_state``'s return IS the clamp signal; a caller that throws it
    away reduces SAFE-02 to a log line nobody correlates with a task. The
    assertion is behavioural rather than source-level: with a controller that
    reports a clipped action, the loop must surface it.
    """
    from embodiment.so_arm10x import skills as skills_mod

    monkeypatch.setattr(skills_mod.time, "sleep", lambda *_: None)
    # Bypass tqdm rather than let it run: instantiating it starts tqdm's
    # background monitor thread, which persists for the rest of the session and
    # makes every later `os.fork()` in the multiprocessing tests emit a
    # multi-threaded-fork DeprecationWarning. The subject here is the clamp
    # signal, not the progress bar.
    monkeypatch.setattr(skills_mod, "tqdm", lambda iterable, **_: iterable)

    requested = {f"{motor}.pos": 10.0 for motor in MOTOR_NAMES}
    clipped = {**requested, "elbow_flex.pos": 1.0}

    controller = SimpleNamespace(
        get_observation=lambda: {"state": "irrelevant"},
        get_current_images=lambda: {"front": "img"},
        set_target_state=lambda action: dict(clipped),
    )
    policy = SimpleNamespace(
        language_instruction="pick it up",
        get_action=lambda obs, lang=None: [dict(requested)],
    )

    skill = skills_mod.PickSkill(controller, policy)
    with capture_loguru() as lines:
        skill.run(actions_to_execute=1, action_horizon=1)

    clamp_lines = [line for line in lines if "clamped" in line]
    assert clamp_lines, lines
    assert "elbow_flex" in "\n".join(clamp_lines)


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
