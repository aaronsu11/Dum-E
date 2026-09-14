'Controller safety instruments: the PID read-back and the motion clamp.'

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
    'Stand-in for ``lerobot.motors.motors_bus.MotorsBus``.'

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
    "Stand-in for ``SOFollower`` whose ``send_action`` mirrors upstream's body."

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
    'An ``SO10xArmController`` shell wired to ``bus``, built without hardware.'
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
    'Collect loguru MESSAGES through a real sink, not by inspecting source text.'
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
# The motion clamp: a real float, on by default, and surfaced
# ---------------------------------------------------------------------------


def test_default_clamp_is_a_float_not_an_int():
    "The default clamp is a strictly positive ``float``, above the policy's own motion."
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
    'An ``int`` clamp raises inside the helper — proven, not described.'
    clamp = ctrl_mod.DEFAULT_MAX_RELATIVE_TARGET
    goal_present = {"shoulder_pan": (clamp + 50.0, 0.0)}

    # The float the module actually configures clamps cleanly...
    clipped = ensure_safe_goal_position(goal_present, clamp)
    assert clipped["shoulder_pan"] == pytest.approx(clamp)

    # ...and the same value as an int does not.
    with pytest.raises(TypeError):
        ensure_safe_goal_position(goal_present, int(clamp))


def test_clamp_exactly_at_bound_is_not_clamped():
    'A delta exactly at the bound passes through; so does one 1e-5 above it.'
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
    'A delta past the bound comes back clipped AND is reported through loguru.'
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
    "Dum-E's own warning carries upstream's exact sentence."
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
    'With the clamp disabled, the code path does not run and nothing is reported.'
    robot = StubRobot(max_relative_target=None)
    controller = clamp_controller(robot)
    requested = {f"{motor}.pos": 500.0 for motor in MOTOR_NAMES}

    with capture_loguru() as lines:
        sent = SO10xArmController.set_target_state(controller, dict(requested))

    assert sent == pytest.approx(requested)
    assert not [line for line in lines if "clamped" in line], lines


def test_clamp_log_lists_joints_in_requested_key_order():
    'Two runs over the same requested action produce identical clamp lines.'
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
    'A non-positive clamp or an incomplete per-motor mapping is refused early.'
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
    '``None`` means "resolve", not "disable" — the clamp is on by default.'
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
    'A plain stdlib root-logger warning reaches a loguru sink, installed once.'
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
    'A bridged record names its real origin, not ``utils.py`` or ``logging``.'
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
    "The belt-and-braces half: upstream's OWN warning lands on Dum-E's stream."
    clamp = ctrl_mod.DEFAULT_MAX_RELATIVE_TARGET

    with stdlib_bridge_installed():
        with capture_loguru() as lines:
            ensure_safe_goal_position({"shoulder_pan": (clamp + 40.0, 0.0)}, clamp)

    assert any(
        "Relative goal position magnitude had to be clamped to be safe." in line
        for line in lines
    ), lines


def test_skill_loop_consumes_set_target_state_return_value(monkeypatch):
    'The pick loop uses the returned action instead of discarding it.'
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
# PID: declarative, then read back and asserted
# ---------------------------------------------------------------------------


def test_pid_config_fields_carry_dume_preset(monkeypatch):
    'The preset is declared on the follower config, not written after connect.'
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

    The observed values are logged as evidence a later phase can cite — "the stiffness the arm actually ran at", not "the value that was
    requested". Returning the mapping is what makes that citable.
    """
    bus = StubBus()
    observed = SO10xArmController._assert_pid_landed(pid_controller(bus))

    assert set(observed) == set(MOTOR_NAMES)
    for motor in MOTOR_NAMES:
        assert observed[motor] == ctrl_mod.DUME_PID


def test_pid_readback_raises_on_mismatch():
    'One wrong coefficient on one motor refuses the connect.'
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
    'A read that raises re-raises with the cause attached — never swallowed.'
    original = OSError("simulated Feetech bus timeout")
    bus = StubBus(read_error=original)

    with pytest.raises(Exception) as excinfo:
        SO10xArmController._assert_pid_landed(pid_controller(bus))

    assert excinfo.value.__cause__ is original
    assert "unknown stiffness" in str(excinfo.value).lower()


def test_pid_readback_uses_unnormalized_reads_with_retry():
    'Every coefficient read passes ``normalize=False`` and a non-zero retry.'
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
    '``connect()`` still runs the read-back, in the same place the write was.'
    calls: List[str] = []
    stub = SimpleNamespace(
        _prearm_goal_to_present=lambda: calls.append("prearm"),
        robot=SimpleNamespace(connect=lambda calibrate=True: calls.append("robot.connect")),
        _assert_calibration_loaded=lambda: calls.append("calibration"),
        _assert_pid_landed=lambda: calls.append("pid"),
    )

    SO10xArmController.connect(stub, calibrate=False)

    # The Goal_Position pre-arm leads: it must run while torque is still off,
    # before `robot.connect()` re-enables it (see the pre-arm section below).
    assert calls == ["prearm", "robot.connect", "calibration", "pid"]


# The Goal_Position pre-arm (the connect-time slam guard).
# Plan 05-06 found this by reading the live registers before commanding


class PrearmStubBus:
    "Stand-in for ``MotorsBus`` covering the pre-arm's register traffic."

    def __init__(
        self,
        present: Optional[Dict[str, int]] = None,
        goal: Optional[Dict[str, int]] = None,
        torque: Optional[Dict[str, int]] = None,
        writes_take: bool = True,
    ) -> None:
        self.motors = list(MOTOR_NAMES)
        self._present = dict(present or {m: 1000 for m in MOTOR_NAMES})
        self._goal = dict(goal or {m: 0 for m in MOTOR_NAMES})
        self._torque = dict(torque or {m: 0 for m in MOTOR_NAMES})
        self._writes_take = writes_take
        self.sync_read_calls: List[Dict[str, Any]] = []
        self.writes: List[Dict[str, Any]] = []
        self.connected = False
        self.disconnect_args: List[Any] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self, disable_torque: bool = True) -> None:
        self.connected = False
        self.disconnect_args.append(disable_torque)

    def sync_read(self, data_name: str, *, normalize: bool = True, num_retry: int = 0):
        self.sync_read_calls.append(
            {"data_name": data_name, "normalize": normalize, "num_retry": num_retry}
        )
        table = {
            "Torque_Enable": self._torque,
            "Present_Position": self._present,
            "Goal_Position": self._goal,
        }[data_name]
        return dict(table)

    def write(self, data_name: str, motor: str, value: int, *, normalize: bool = True, num_retry: int = 0):
        self.writes.append(
            {
                "data_name": data_name,
                "motor": motor,
                "value": value,
                "normalize": normalize,
                "num_retry": num_retry,
            }
        )
        if self._writes_take and data_name == "Goal_Position":
            self._goal[motor] = int(value)


def prearm_controller(bus: PrearmStubBus) -> SO10xArmController:
    """An ``SO10xArmController`` shell wired to ``bus``, built without hardware."""
    controller = SO10xArmController.__new__(SO10xArmController)
    controller.robot = SimpleNamespace(bus=bus)
    controller.config = SimpleNamespace(num_read_retries=2)
    return controller


def test_prearm_writes_present_position_into_goal_when_torque_is_off():
    'The core guard: Goal_Position becomes Present_Position before torque returns.'
    bus = PrearmStubBus(present={m: 1000 for m in MOTOR_NAMES}, goal={m: 0 for m in MOTOR_NAMES})
    controller = prearm_controller(bus)

    record = controller._prearm_goal_to_present()

    assert record["prearmed"] is True
    written = {w["motor"]: w["value"] for w in bus.writes if w["data_name"] == "Goal_Position"}
    assert written == {m: 1000 for m in MOTOR_NAMES}
    # The pending jump the guard just neutralised is reported, so a caller can log
    # how close it came rather than only that it succeeded.
    assert record["worst_pending_jump_ticks"] == 1000


def test_prearm_skips_when_torque_is_already_enabled():
    'With torque on, Goal_Position is LIVE -- overwriting it is the very command to avoid.'
    bus = PrearmStubBus(torque={m: 1 for m in MOTOR_NAMES})
    controller = prearm_controller(bus)

    record = controller._prearm_goal_to_present()

    assert record["prearmed"] is False
    assert "torque already enabled" in record["reason"]
    assert [w for w in bus.writes if w["data_name"] == "Goal_Position"] == []


def test_prearm_raises_when_the_write_did_not_take():
    """A write that silently fails must refuse the connect, not proceed hopefully.

    This is the PID read-back lesson applied to the pre-arm: a clean write call is not
    evidence the register changed, so the guard reads back and fails closed.
    """
    bus = PrearmStubBus(
        present={m: 1000 for m in MOTOR_NAMES}, goal={m: 0 for m in MOTOR_NAMES}, writes_take=False
    )
    controller = prearm_controller(bus)

    with pytest.raises(RuntimeError, match="pre-arm did not take"):
        controller._prearm_goal_to_present()


def test_prearm_reads_raw_ticks_with_retries():
    """Normalization OFF and a non-zero retry count on every read."""
    bus = PrearmStubBus()
    controller = prearm_controller(bus)

    controller._prearm_goal_to_present()

    assert bus.sync_read_calls, "the pre-arm must read the bus"
    for call in bus.sync_read_calls:
        assert call["normalize"] is False, f"{call['data_name']} must be read as raw ticks"
        assert call["num_retry"] >= 2, f"{call['data_name']} must tolerate a dropped packet"
    for write in bus.writes:
        assert write["normalize"] is False
        assert write["num_retry"] >= 2


def test_prearm_leaves_torque_state_exactly_as_found():
    '`disconnect(False)` -- the guard must not itself change the torque state.'
    bus = PrearmStubBus()
    controller = prearm_controller(bus)

    controller._prearm_goal_to_present()

    assert bus.disconnect_args == [False]
    assert bus.connected is False


def test_prearm_disconnects_even_when_the_readback_fails():
    """The bus must not be left open when the guard raises."""
    bus = PrearmStubBus(
        present={m: 1000 for m in MOTOR_NAMES}, goal={m: 0 for m in MOTOR_NAMES}, writes_take=False
    )
    controller = prearm_controller(bus)

    with pytest.raises(RuntimeError):
        controller._prearm_goal_to_present()

    assert bus.connected is False
    assert bus.disconnect_args == [False]


def test_connect_prearms_before_robot_connect():
    """Ordering is the whole guard: pre-arm must precede the torque-enabling connect.

    Running it afterwards would be strictly useless -- the slam happens inside
    `robot.connect()`, so a pre-arm that follows it arrives after the damage.
    """
    calls: List[str] = []
    stub = SimpleNamespace(
        _prearm_goal_to_present=lambda: calls.append("prearm"),
        robot=SimpleNamespace(connect=lambda calibrate=True: calls.append("robot.connect")),
        _assert_calibration_loaded=lambda: calls.append("calibration"),
        _assert_pid_landed=lambda: calls.append("pid"),
    )

    SO10xArmController.connect(stub, calibrate=False)

    assert calls == ["prearm", "robot.connect", "calibration", "pid"]
