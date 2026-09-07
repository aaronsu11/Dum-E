"""Calibration-file assertion at connect (D-05, plan 05-04 Task 2).

lerobot 0.6.1 consolidated ``so101_follower`` into ``so_follower``, and
``Robot.__init__`` derives its calibration directory from the robot class's own
``name``. The rename therefore moves the lookup from
``<root>/robots/so101_follower/`` to ``<root>/robots/so_follower/``. CONTEXT.md
D-05 locked *copying* the calibration file to the new path (pinning
``config.calibration_dir`` was offered and declined), and accepted the resulting
two-copies-can-drift tradeoff on the condition that ``connect()`` logs the loaded
path plus a content checksum so a wrong or missing file is visible.

Why the assertion is load-bearing rather than defensive: ``range_min`` /
``range_max`` in that file are raw encoder ticks that upstream normalization maps
onto the numbers fed to the policy, so the calibration file is part of the
checkpoint's input contract. And ``SOFollower.is_calibrated`` interrogates the
**motors**, not the file — if the servos still hold calibration in firmware,
``connect()`` skips calibration and reports success while the in-Python
calibration mapping stays empty, so the failure surfaces at the first bus read,
after connect already looked fine. These tests pin the assertion into that
window.

Every test here is hermetic: a temporary calibration root pointed at by the
documented environment variable, no serial port, no real robot, no network, no
dependence on this machine's own calibration file.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodiment.so_arm10x.controller import (
    SO10xArmController,
    assert_calibration_loaded,
    resolve_calibration_file,
    resolve_lerobot_calibration_root,
)

ROBOT_ID = "my_awesome_follower_arm"
SO_FOLLOWER_NAME = "so_follower"

# A minimal but realistically shaped calibration payload. The tick ranges matter
# to the checksum only, not to any assertion about their values.
CALIBRATION_PAYLOAD = {
    "shoulder_pan": {
        "id": 1,
        "drive_mode": 0,
        "homing_offset": 47,
        "range_min": 738,
        "range_max": 3372,
    },
    "gripper": {
        "id": 6,
        "drive_mode": 0,
        "homing_offset": -1015,
        "range_min": 2031,
        "range_max": 3134,
    },
}


def _write_calibration(directory: Path, robot_id: str = ROBOT_ID) -> Path:
    """Write a calibration JSON into ``directory`` and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{robot_id}.json"
    path.write_text(json.dumps(CALIBRATION_PAYLOAD, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def calibration_root(tmp_path, monkeypatch):
    """Point ``HF_LEROBOT_CALIBRATION`` at a temporary root and yield it."""
    root = tmp_path / "calibration"
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(root))
    monkeypatch.delenv("HF_LEROBOT_HOME", raising=False)
    return root


# --- Path derivation --------------------------------------------------------


def test_calibration_root_resolves_from_documented_environment_variable(
    tmp_path, monkeypatch
):
    """``HF_LEROBOT_CALIBRATION`` wins outright, exactly as upstream reads it."""
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "explicit"))
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "ignored"))
    assert resolve_lerobot_calibration_root() == tmp_path / "explicit"


def test_calibration_root_falls_back_to_lerobot_home_plus_calibration(
    tmp_path, monkeypatch
):
    """With no explicit override the root is ``HF_LEROBOT_HOME / "calibration"``.

    This mirrors ``lerobot.utils.constants``' ``default_calibration_path``. The
    point of resolving it here rather than importing the upstream constant is
    that the upstream constant is evaluated at import time, so it cannot be
    redirected by a test (or by anything that sets the variable late).
    """
    monkeypatch.delenv("HF_LEROBOT_CALIBRATION", raising=False)
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lerobot-home"))
    assert resolve_lerobot_calibration_root() == tmp_path / "lerobot-home" / "calibration"


def test_calibration_file_path_matches_the_upstream_derivation(calibration_root):
    """``<root>/robots/<robot class name>/<robot id>.json``, per ``Robot.__init__``.

    Upstream builds ``HF_LEROBOT_CALIBRATION / ROBOTS / self.name`` then appends
    ``f"{self.id}.json"``. The renamed class's ``name`` is ``so_follower``, which
    is precisely why the file has to move.
    """
    derived = resolve_calibration_file(SO_FOLLOWER_NAME, ROBOT_ID)
    assert derived == calibration_root / "robots" / SO_FOLLOWER_NAME / f"{ROBOT_ID}.json"

    # And the pre-rename path is a *different* directory — the migration is real.
    legacy = resolve_calibration_file("so101_follower", ROBOT_ID)
    assert legacy.parent != derived.parent
    assert legacy.name == derived.name


# --- The assertion itself ---------------------------------------------------


def test_calibration_assertion_returns_the_path_and_a_content_checksum(
    calibration_root,
):
    """With the file present, the resolved path and a content digest come back."""
    written = _write_calibration(calibration_root / "robots" / SO_FOLLOWER_NAME)
    path, checksum = assert_calibration_loaded(SO_FOLLOWER_NAME, ROBOT_ID)

    assert path == written
    assert checksum == hashlib.sha256(written.read_bytes()).hexdigest()

    # The digest is content-derived, so a wrong-file copy is visible — the one
    # failure mode the loud missing-file error does not cover.
    written.write_text(json.dumps({"shoulder_pan": {"id": 1}}), encoding="utf-8")
    _path, changed = assert_calibration_loaded(SO_FOLLOWER_NAME, ROBOT_ID)
    assert changed != checksum


def test_calibration_assertion_raises_naming_the_expected_path_when_absent(
    calibration_root,
):
    """An absent file raises, and the message names the exact path checked."""
    expected = calibration_root / "robots" / SO_FOLLOWER_NAME / f"{ROBOT_ID}.json"
    assert not expected.exists()

    with pytest.raises(FileNotFoundError) as excinfo:
        assert_calibration_loaded(SO_FOLLOWER_NAME, ROBOT_ID)
    assert str(expected) in str(excinfo.value)


def test_calibration_assertion_checks_the_file_not_the_containing_directory(
    calibration_root,
):
    """An existing but empty calibration directory is NOT evidence.

    ``Robot.__init__`` calls ``mkdir(parents=True, exist_ok=True)`` on the derived
    path, so after the rename a directory listing shows a freshly created,
    provisioned-looking, empty ``so_follower/`` directory. A guard that checked
    directory existence would pass on exactly the broken state this exists to
    catch.
    """
    empty_dir = calibration_root / "robots" / SO_FOLLOWER_NAME
    empty_dir.mkdir(parents=True, exist_ok=True)
    assert empty_dir.is_dir()

    with pytest.raises(FileNotFoundError):
        assert_calibration_loaded(SO_FOLLOWER_NAME, ROBOT_ID)

    # A file belonging to a DIFFERENT robot id in the same directory is not it
    # either — the lookup is keyed by id, not by "some calibration is present".
    _write_calibration(empty_dir, robot_id="someone_elses_arm")
    with pytest.raises(FileNotFoundError):
        assert_calibration_loaded(SO_FOLLOWER_NAME, ROBOT_ID)


def test_controller_calibration_assertion_prefers_the_robots_own_derived_path(
    calibration_root, tmp_path
):
    """The method uses ``robot.calibration_fpath`` when the robot exposes it.

    That attribute is the authoritative answer to "which file will lerobot
    load" — it already accounts for a ``config.calibration_dir`` override. Env
    re-derivation is the fallback for when no robot object is available.
    """
    authoritative_dir = tmp_path / "authoritative" / "robots" / SO_FOLLOWER_NAME
    written = _write_calibration(authoritative_dir)

    stub = SimpleNamespace(
        robot=SimpleNamespace(
            name=SO_FOLLOWER_NAME, id=ROBOT_ID, calibration_fpath=written
        ),
        _robot_id=ROBOT_ID,
    )
    path, checksum = SO10xArmController._assert_calibration_loaded(stub)
    assert path == written
    assert checksum == hashlib.sha256(written.read_bytes()).hexdigest()

    # And it still raises when that authoritative path holds no file.
    absent = tmp_path / "authoritative" / "robots" / SO_FOLLOWER_NAME / "gone.json"
    stub.robot.calibration_fpath = absent
    with pytest.raises(FileNotFoundError) as excinfo:
        SO10xArmController._assert_calibration_loaded(stub)
    assert str(absent) in str(excinfo.value)


def test_connect_asserts_calibration_before_reading_any_observation(
    calibration_root,
):
    """``connect()`` runs the calibration assertion in the pre-observation window.

    Ordering is the whole point: the robot's own ``is_calibrated`` reads the
    motors, so ``robot.connect()`` can return successfully with an empty
    in-Python calibration. The assertion must therefore run after the connect and
    before anything touches an observation, which is where the ``RuntimeError``
    would otherwise surface.
    """
    calls: list[str] = []

    def _fail_on_observation():
        calls.append("get_observation")
        raise AssertionError("connect() must not read an observation")

    stub = SimpleNamespace(
        robot=SimpleNamespace(
            connect=lambda calibrate=True: calls.append("robot.connect"),
            get_observation=_fail_on_observation,
        ),
        _assert_calibration_loaded=lambda: calls.append("assert_calibration"),
        # Plan 05-05 replaced `set_so10x_robot_preset()` (a torque-disabled write
        # loop wrapped in a bare except) with a read-back assertion at the same
        # point in `connect()`. The ordering this test pins is unchanged.
        _assert_pid_landed=lambda: calls.append("pid_readback"),
    )

    SO10xArmController.connect(stub, calibrate=False)

    assert "get_observation" not in calls
    assert calls.index("robot.connect") < calls.index("assert_calibration")
    assert calls.index("assert_calibration") < calls.index("pid_readback")
