"""Hermetic gate for the live pick-baseline runner (LR-06 / BACK-06).

The runner in ``scripts/run_pick_baseline.py`` produces the number Phase 7's
parity gate is measured against, so the parts of it that decide *what the record
says* are tested here rather than discovered during a ten-attempt live run. Three
properties get the most attention, because each one, if broken, would silently
corrupt the baseline instead of failing loudly:

1. **The record schema the standing baseline document is built from.** The
   document's verifier reads specific top-level and per-attempt keys; a drift
   there is only visible after the arm has already run.
2. **The success judgment cannot be machine-supplied.** No default answer, no
   bulk answer, no flag that supplies one, and a closed stdin raises instead of
   assuming. A fabricated success is worse than a missing baseline.
3. **A motion-free preflight cannot masquerade as a scored run.** Preflight
   evidence is written under a different directory prefix *and* a different
   filename, so the newest-``run.json`` selection the standing record uses can
   never pick up a run with zero scored attempts.

Every test is hermetic: no serial port, no camera, no policy server, no Docker.
The live half — the connect-time PID read-back and the scored attempts — is
exercised by the runner itself against real hardware, behind the operator gate.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make scripts/ importable so the tests and the harness share ONE source of truth
# (same idiom as tests/test_units_verdict.py and tests/test_container_contract.py).
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import run_pick_baseline as runner  # noqa: E402

CLAMP_TEXT = "Relative goal position magnitude had to be clamped to be safe."


# ---------------------------------------------------------------------------
# The clamp sentence is shared with the controller, not re-typed
# ---------------------------------------------------------------------------


def test_clamp_text_used_by_the_counter_is_the_controllers_own_sentence():
    """The counter must match the wording every emitter actually uses.

    All three emitters — upstream's bridged root-logger warning, the controller's
    per-step re-emission, and PickSkill's per-pick roll-up — embed this exact
    sentence, which is what makes a single sink sufficient. A private copy that
    drifted would silently count zero.
    """
    from embodiment.so_arm10x.controller import CLAMP_WARNING_TEXT

    assert CLAMP_TEXT == CLAMP_WARNING_TEXT


# ---------------------------------------------------------------------------
# count_clamp_warnings
# ---------------------------------------------------------------------------


def test_count_clamp_warnings_counts_every_emitters_wording():
    messages = [
        f"{CLAMP_TEXT} max_relative_target=160.0 clamped 1 joint(s): wrist_roll ...",
        f"{CLAMP_TEXT} 3 commanded joint target(s) were clamped during this pick ...",
        f"{CLAMP_TEXT} original goal_pos ...",
        "some unrelated warning",
    ]
    assert runner.count_clamp_warnings(messages, CLAMP_TEXT) == 3


def test_count_clamp_warnings_excludes_the_synthetic_self_test_probe():
    """The counter's own probe must never inflate the number the gate reads.

    The parity gate requires ZERO clamp warnings during nominal operation, so a
    self-test that counted itself would manufacture a finding.
    """
    messages = [f"{CLAMP_TEXT} {runner.CLAMP_SELFTEST_MARKER}"]
    assert runner.count_clamp_warnings(messages, CLAMP_TEXT) == 0


def test_count_clamp_warnings_on_an_empty_window_is_zero_not_missing():
    """Zero is a valid, reportable count — not an absence of measurement."""
    assert runner.count_clamp_warnings([], CLAMP_TEXT) == 0


# ---------------------------------------------------------------------------
# The record schema the standing baseline is built from
# ---------------------------------------------------------------------------


def _valid_attempt(index: int = 1, success: bool = True) -> dict:
    return runner.build_attempt_record(
        index=index,
        instruction="pick up the fruit",
        success=success,
        clamp_warnings=0,
        duration_s=12.3456,
        exception=None,
    )


def _valid_payload(attempts=None) -> dict:
    return {
        "instruction": "pick up the fruit",
        "backend": "groot-native",
        "lerobot_version": "0.6.1",
        "attempts": attempts if attempts is not None else [_valid_attempt()],
    }


def test_build_attempt_record_emits_every_key_the_record_needs():
    record = _valid_attempt()
    for key in runner.REQUIRED_ATTEMPT_KEYS:
        assert key in record, key
    assert record["clamp_warnings"] == 0 and isinstance(record["clamp_warnings"], int)
    assert record["success"] is True
    assert record["duration_s"] == pytest.approx(12.346, abs=1e-3)
    assert record["judgment_source"] == "operator-stdin"


def test_validate_record_accepts_a_well_formed_payload():
    assert runner.validate_record(_valid_payload()) == []


@pytest.mark.parametrize("missing", runner.REQUIRED_RUN_KEYS)
def test_validate_record_rejects_a_payload_missing_any_read_key(missing):
    payload = _valid_payload()
    payload.pop(missing)
    problems = runner.validate_record(payload)
    assert problems, f"dropping {missing!r} must be reported"
    assert any(missing in problem for problem in problems)


def test_validate_record_rejects_a_non_integer_clamp_count():
    """The record's total is summed arithmetically; a string would poison it."""
    payload = _valid_payload()
    payload["attempts"][0]["clamp_warnings"] = "0"
    assert any("clamp_warnings" in problem for problem in runner.validate_record(payload))


def test_validate_record_rejects_a_non_boolean_success():
    payload = _valid_payload()
    payload["attempts"][0]["success"] = "yes"
    assert any("success" in problem for problem in runner.validate_record(payload))


def test_validate_record_rejects_an_empty_instruction():
    """An unpinned instruction makes the whole comparison meaningless."""
    payload = _valid_payload()
    payload["instruction"] = ""
    assert any("instruction" in problem for problem in runner.validate_record(payload))


def test_validate_record_reports_a_missing_per_attempt_key():
    payload = _valid_payload()
    payload["attempts"][0].pop("clamp_warnings")
    assert any("clamp_warnings" in problem for problem in runner.validate_record(payload))


# ---------------------------------------------------------------------------
# record_run: where evidence lands, and what may not masquerade as a baseline
# ---------------------------------------------------------------------------


def test_record_run_writes_a_scored_run_where_the_record_verifier_looks(tmp_path):
    target = runner.record_run(_valid_payload(), dry_run=False, out_root=str(tmp_path))
    assert target.name == "run.json"
    assert target.parent.name.startswith(runner.SCORED_DIR_PREFIX)
    assert json.loads(target.read_text())["backend"] == "groot-native"
    # The selection the standing baseline document uses.
    assert [str(target)] == sorted(str(p) for p in tmp_path.glob("pick_baseline_*/run.json"))


def test_a_preflight_can_never_be_selected_as_a_scored_baseline(tmp_path):
    """The anti-fabrication property, asserted on the selection itself.

    A preflight has zero scored attempts. If its evidence could be picked up by
    the newest-``run.json`` selection, a motion-free wiring check would present
    itself as the live baseline — so it is separated by BOTH directory prefix and
    filename, and the glob must find nothing.
    """
    payload = _valid_payload(attempts=[])
    target = runner.record_run(payload, dry_run=True, out_root=str(tmp_path))
    assert target.name == runner.DRYRUN_FILENAME != runner.SCORED_FILENAME
    assert target.parent.name.startswith(runner.DRYRUN_DIR_PREFIX)
    assert sorted(tmp_path.glob("pick_baseline_*/run.json")) == []


def test_record_run_refuses_to_write_an_unreadable_record(tmp_path):
    payload = _valid_payload()
    payload.pop("backend")
    with pytest.raises(ValueError, match="could not read"):
        runner.record_run(payload, dry_run=False, out_root=str(tmp_path))
    assert list(tmp_path.glob("**/*.json")) == []


def test_a_scored_record_with_no_attempts_is_refused_outright(tmp_path):
    """A zero-attempt file may never be written under the scored filename.

    This is the preflight-abort path's failure: it set ``voided`` and fell through
    to a scored write with ``attempts == []``, producing a
    ``pick_baseline_<stamp>/run.json`` carrying ``score: null`` and
    ``attempts_performed: 0`` — which, being the newest, won the documented
    selection. Nothing about the schema caught it, because ``attempts`` was still a
    list.
    """
    payload = _valid_payload(attempts=[])
    assert runner.validate_record(payload) == []
    assert any("at least one attempt" in p for p in runner.validate_record(payload, scored=True))
    with pytest.raises(ValueError, match="at least one attempt"):
        runner.record_run(payload, dry_run=False, out_root=str(tmp_path))
    assert list(tmp_path.glob("**/*.json")) == []


def test_an_aborted_scored_series_lands_under_the_voided_name(tmp_path):
    """A void run is recorded, but never where a baseline is looked for.

    The evidence still has to be written — an aborted series is a finding worth
    keeping — so it goes to a third prefix AND filename. The selection glob must
    find nothing, whether the abort happened before the first attempt or partway
    through the series.
    """
    for attempts in ([], [_valid_attempt(1), _valid_attempt(2, success=False)]):
        target = runner.record_run(
            _valid_payload(attempts=attempts),
            dry_run=False,
            out_root=str(tmp_path),
            voided=True,
        )
        assert target.name == runner.VOIDED_FILENAME
        assert target.name not in (runner.SCORED_FILENAME, runner.DRYRUN_FILENAME)
        assert target.parent.name.startswith(runner.VOIDED_DIR_PREFIX)

    assert sorted(tmp_path.glob("pick_baseline_*/run.json")) == []


# ---------------------------------------------------------------------------
# The operator judgment: no default, no bulk answer, no invention
# ---------------------------------------------------------------------------


class _Prompts:
    """A scripted stand-in for ``input`` that records what it was asked."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.asked = []

    def __call__(self, text):
        self.asked.append(text)
        if not self.answers:
            raise EOFError("scripted prompts exhausted")
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def test_success_judgment_reads_yes_from_the_operator():
    prompt = _Prompts("y", "")
    success, note = runner.prompt_success_judgment(1, None, prompt=prompt)
    assert success is True and note is None


def test_success_judgment_reads_no_and_keeps_the_operators_note():
    prompt = _Prompts("n", "gripper closed early")
    success, note = runner.prompt_success_judgment(2, None, prompt=prompt)
    assert success is False and note == "gripper closed early"


def test_success_judgment_has_no_default_and_reprompts_on_an_empty_answer():
    """Pressing Enter must NOT be read as agreement."""
    prompt = _Prompts("", "  ", "n", "")
    success, _ = runner.prompt_success_judgment(3, None, prompt=prompt)
    assert success is False
    judgment_prompts = [text for text in prompt.asked if "pick up the object" in text]
    assert len(judgment_prompts) == 3


def test_success_judgment_raises_rather_than_assuming_when_stdin_is_closed():
    """A closed stdin is an abort, never an implied success."""
    with pytest.raises(runner.NonInteractiveError):
        runner.prompt_success_judgment(4, None, prompt=_Prompts(EOFError()))


def test_success_judgment_is_still_asked_when_the_attempt_raised():
    """An exception is recorded, but what the operator SAW is still the judgment."""
    prompt = _Prompts("n", "")
    success, _ = runner.prompt_success_judgment(5, "RuntimeError: boom", prompt=prompt)
    assert success is False


def test_place_object_prompt_accepts_enter_and_abort():
    assert runner.prompt_place_object(1, 10, prompt=_Prompts("")) == "go"
    assert runner.prompt_place_object(2, 10, prompt=_Prompts("abort")) == "abort"
    with pytest.raises(runner.NonInteractiveError):
        runner.prompt_place_object(3, 10, prompt=_Prompts(EOFError()))


def test_the_runner_exposes_no_flag_that_could_supply_a_success_judgment():
    """There must be no ``--yes``/``--assume-success``/``--auto`` escape hatch."""
    options = [
        option
        for action in runner.build_parser()._actions
        for option in action.option_strings
    ]
    forbidden = [
        option
        for option in options
        if any(token in option.lower() for token in ("yes", "assume", "auto", "success", "force"))
    ]
    assert forbidden == [], forbidden


def test_a_scored_series_refuses_to_run_without_an_interactive_terminal(monkeypatch, capsys):
    """The structural guard against a piped-in, fabricated baseline."""
    monkeypatch.setattr(runner.sys.stdin, "isatty", lambda: False, raising=False)
    code = runner.main(["--instruction", "pick up the fruit", "--attempts", "10"])
    assert code == 2
    assert "REFUSING TO RUN" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Which exceptions void the series
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exception",
    [
        "FileNotFoundError: LeRobot calibration file not found at ...",
        "ValueError: Dum-E PID preset did not land on the motor bus ...",
        "RuntimeError: policy server unreachable at localhost:5555 ...",
        "RuntimeError: Goal_Position pre-arm did not take ...",
    ],
)
def test_a_stack_changing_exception_voids_the_attempt_series(exception):
    assert runner.exception_voids_series(exception) is True


@pytest.mark.parametrize("exception", [None, "", "TimeoutError: the gripper missed"])
def test_an_ordinary_failed_attempt_does_not_void_the_series(exception):
    assert runner.exception_voids_series(exception) is False


# ---------------------------------------------------------------------------
# The selector is resolved through the allowlist, never overwritten
# ---------------------------------------------------------------------------


def test_backend_selection_sets_the_env_var_when_it_is_unset(monkeypatch):
    from policy.factory import POLICY_BACKEND_ENV_VAR

    monkeypatch.delenv(POLICY_BACKEND_ENV_VAR, raising=False)
    record = runner.resolve_backend_selection("groot-native")
    assert record == {
        "env_var": POLICY_BACKEND_ENV_VAR,
        "value": "groot-native",
        "was_already_set": False,
    }
    import os

    assert os.environ[POLICY_BACKEND_ENV_VAR] == "groot-native"


def test_backend_selection_refuses_to_overwrite_a_conflicting_env_var(monkeypatch):
    """Which network drives the arm is not something to change silently.

    A baseline attributed to the wrong backend is worse than no baseline, so a
    conflict raises instead of being resolved in favour of either value.
    """
    from policy.factory import POLICY_BACKEND_ENV_VAR

    monkeypatch.setenv(POLICY_BACKEND_ENV_VAR, "lerobot")
    with pytest.raises(ValueError, match="already set"):
        runner.resolve_backend_selection("groot-native")
    import os

    assert os.environ[POLICY_BACKEND_ENV_VAR] == "lerobot"


def test_backend_selection_rejects_a_name_outside_the_allowlist(monkeypatch):
    from policy.factory import POLICY_BACKEND_ENV_VAR

    monkeypatch.delenv(POLICY_BACKEND_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="allowlist"):
        runner.resolve_backend_selection("gr00t-nativ")


# ---------------------------------------------------------------------------
# The payload carries its caveat and its attributable stack
# ---------------------------------------------------------------------------


def _args(**overrides):
    from argparse import Namespace

    base = dict(
        instruction="pick up the fruit",
        backend="groot-native",
        attempts=10,
    )
    base.update(overrides)
    return Namespace(**base)


def test_run_payload_carries_the_keys_the_record_verifier_reads():
    payload = runner.build_run_payload(
        _args(),
        mode="scored",
        attempts=[_valid_attempt(1), _valid_attempt(2, success=False)],
        stack={"lerobot_version": "0.6.1"},
        checks={"selector": True},
        extras={},
    )
    assert runner.validate_record(payload) == []
    assert payload["backend"] == "groot-native"
    assert payload["lerobot_version"] == "0.6.1"
    assert payload["successes"] == 1
    assert payload["score"] == "1/2"
    assert payload["attempts_requested"] == 10
    assert payload["attempts_performed"] == 2


def test_run_payload_states_the_smoke_check_caveat_in_as_many_words():
    """D-10's load-bearing consequence must travel with the number itself."""
    payload = runner.build_run_payload(
        _args(),
        mode="scored",
        attempts=[_valid_attempt()],
        stack={"lerobot_version": "0.6.1"},
        checks={},
        extras={},
    )
    caveat = payload["caveat"].lower()
    assert "functional smoke check" in caveat
    assert "not a controlled numerical comparison" in caveat
    assert "corpus" in caveat
    assert "accepted" in payload["scene_variation"].lower()


def test_run_payload_totals_the_clamp_warnings_across_attempts():
    attempts = [
        runner.build_attempt_record(1, "i", True, 0, 1.0, None),
        runner.build_attempt_record(2, "i", True, 3, 1.0, None),
    ]
    payload = runner.build_run_payload(
        _args(), mode="scored", attempts=attempts, stack={}, checks={}, extras={}
    )
    assert payload["total_clamp_warnings"] == 3


# ---------------------------------------------------------------------------
# Read-only device inspection and configuration resolution
# ---------------------------------------------------------------------------


def test_device_inspection_reports_a_missing_serial_port(tmp_path):
    settings = {
        "robot_port": str(tmp_path / "ttyNOPE"),
        "wrist_cam_idx": 0,
        "front_cam_idx": 2,
    }
    report = runner.inspect_devices(settings)
    if report.get("skipped"):
        pytest.skip(report["skipped"])
    assert report["ok"] is False
    assert report["devices"]["serial"]["present"] is False


def test_device_inspection_is_skipped_off_linux(monkeypatch):
    monkeypatch.setattr(runner.sys, "platform", "darwin")
    report = runner.inspect_devices(
        {"robot_port": "/dev/nope", "wrist_cam_idx": 0, "front_cam_idx": 2}
    )
    assert report["ok"] is True and "skipped" in report


def test_controller_settings_come_from_the_live_config_not_the_template(tmp_path, monkeypatch):
    """The live config wins over the module fallback, and the template is unused.

    ``config.example.yaml`` still names ``front_cam_idx: 1``, which on the host
    this baseline was taken against is a V4L2 metadata node that cannot be
    opened — a template value must never become a device selection.
    """
    config = tmp_path / "live.yaml"
    config.write_text(
        "controller:\n"
        "  robot_type: so101_follower\n"
        "  robot_id: my_awesome_follower_arm\n"
        "  robot_port: /dev/ttyACM7\n"
        "  wrist_cam_idx: 0\n"
        "  front_cam_idx: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DUME_CONFIG", str(config))
    settings = runner.resolve_live_controller_settings(
        _args(
            robot_port=None,
            robot_id=None,
            robot_type=None,
            wrist_cam_idx=None,
            front_cam_idx=None,
        )
    )
    assert settings["robot_port"] == "/dev/ttyACM7"
    assert settings["front_cam_idx"] == 2


def test_explicit_flags_override_the_live_config(tmp_path, monkeypatch):
    config = tmp_path / "live.yaml"
    config.write_text("controller:\n  robot_port: /dev/ttyACM7\n", encoding="utf-8")
    monkeypatch.setenv("DUME_CONFIG", str(config))
    settings = runner.resolve_live_controller_settings(
        _args(
            robot_port="/dev/ttyACM9",
            robot_id=None,
            robot_type=None,
            wrist_cam_idx=None,
            front_cam_idx=1,
        )
    )
    assert settings["robot_port"] == "/dev/ttyACM9"
    assert settings["front_cam_idx"] == 1


# ---------------------------------------------------------------------------
# Provenance recording degrades rather than voiding a live run
# ---------------------------------------------------------------------------


def test_container_identity_reports_unavailable_rather_than_raising(monkeypatch):
    """An inaccessible Docker CLI is a provenance gap, not a reason to void ten
    live attempts."""

    def boom(*_args, **_kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(runner.subprocess, "run", boom)
    record = runner.container_image_identity("gr00t-server")
    assert record["available"] is False and "reason" in record


def test_stack_identity_reports_the_resolved_lerobot_version():
    stack = runner.stack_identity()
    assert stack["lerobot_version"] == "0.6.1"
    assert stack["python_version"]


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_the_pinned_instruction_is_required_not_defaulted():
    """There is no default instruction: the pinned string is the one controlled
    variable, so it must be stated explicitly on every run."""
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args([])


def test_cli_defaults_match_the_baseline_contract():
    args = runner.build_parser().parse_args(["--instruction", "pick up the fruit"])
    assert args.attempts == 10
    assert args.backend == runner.DEFAULT_BACKEND == "groot-native"
    assert args.action_horizon == 16
    assert args.dry_run is False and args.with_arm is False


def test_with_arm_is_rejected_outside_dry_run(capsys):
    code = runner.main(["--instruction", "x", "--with-arm"])
    assert code == 2
    assert "only meaningful with --dry-run" in capsys.readouterr().out
