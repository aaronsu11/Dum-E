#!/usr/bin/env python3
"""Operator-interactive live pick baseline for the ``groot-native`` backend (LR-06 / BACK-06).

Re-establishes the live comparison point the milestone is measured against: with
``groot-native`` selected **through the policy-backend switch** on the upgraded
LeRobot stack, score N pick attempts on the physical SO-ARM10x against a single
**pinned** instruction string.

Two properties make the resulting number worth anything, and both are enforced
here rather than assumed:

1. **The seam is exercised, not bypassed.** The backend is obtained from
   ``policy.factory.make_policy_backend()`` with ``DUME_POLICY_BACKEND`` resolved
   through the allowlist. This file never constructs the concrete GR00T client
   class itself, so a passing run proves the fallback is *functional*, not merely
   importable (BACK-06).
2. **The success judgment is a human's.** Whether the object was actually picked
   up is not observable from an exit code, so it is read from an interactive
   terminal, once per attempt, with no default and no way to answer it in bulk.
   There is deliberately **no** ``--yes``/``--assume-success`` flag: a
   machine-supplied judgment would be a fabricated baseline, and a fabricated
   baseline corrupts the exact comparison Phase 7's parity gate depends on.

**What this baseline is (CONTEXT.md D-10).** Only the instruction string is
pinned; scene variation between attempts is *accepted*. The score is therefore a
**functional smoke check, not a controlled numerical comparison** — the entire
numerical burden for the parity gate rests on the offline frozen-corpus evidence.
Do not cite a score produced by this script as evidence of numerical parity.

Modes
-----
``--dry-run``
    Motion-free preflight. Verifies the whole wiring path — the pinned
    instruction, device presence, the selector, server reachability, a full
    synthetic observation round trip through the backend, the clamp-warning
    counter (sink + stdlib bridge), controller construction, and the calibration
    file and its checksum — **without opening the serial bus and without
    commanding any motion**. Writes ``preflight.json``, never a ``run.json``.

``--dry-run --with-arm``
    As above, plus ``controller.connect()`` so the connect-time PID read-back and
    calibration assertion run on the live bus. Commands no motion (``connect()``
    pre-arms ``Goal_Position`` to ``Present_Position`` while torque is still off,
    so the torque-enable is a *hold*), and disconnects leaving torque exactly as
    it found it.

(no flag)
    The scored series: N attempts, operator-reset scene between attempts,
    operator success judgment after each. Writes ``run.json`` under the
    gitignored corpus directory. Exits non-zero unless every attempt succeeded.

Usage
-----
    # motion-free preflight (safe with nothing on the table)
    uv run python scripts/run_pick_baseline.py --dry-run --instruction "pick up the fruit"

    # preflight including the live bus assertions (no motion)
    uv run python scripts/run_pick_baseline.py --dry-run --with-arm --instruction "pick up the fruit"

    # the scored baseline (REQUIRES a human at the terminal and props on the table)
    uv run python scripts/run_pick_baseline.py --attempts 10 --instruction "pick up the fruit"

Evidence layout (all gitignored)
--------------------------------
    corpus/pick_baseline_<stamp>/run.json                completed scored runs only
    corpus/pick_baseline_dryrun_<stamp>/preflight.json   preflight runs only
    corpus/pick_baseline_voided_<stamp>/voided.json      aborted / voided scored runs

All three are deliberately distinct in BOTH directory prefix and filename: the
standing record's verifier selects the newest ``corpus/pick_baseline_*/run.json``,
and neither a preflight nor an aborted series must be able to present itself as a
scored baseline. ``run.json`` is additionally guaranteed to carry at least one
scored attempt — :func:`validate_record` refuses to write one that does not.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Gitignored evidence root. Raw per-attempt output goes here, never into a
#: tracked path — the committed artifact is the standing baseline document.
CORPUS_DIRNAME = "corpus"

#: Scored runs. The standing record's verifier globs exactly this shape.
SCORED_DIR_PREFIX = "pick_baseline_"
SCORED_FILENAME = "run.json"

#: Preflight runs. Different prefix AND different filename on purpose.
DRYRUN_DIR_PREFIX = "pick_baseline_dryrun_"
DRYRUN_FILENAME = "preflight.json"

#: Aborted or voided scored runs. A third prefix and filename rather than reuse of
#: the preflight pair, so "the operator aborted / the stack changed under the run"
#: stays distinguishable from "this was only ever a motion-free preflight". What
#: matters for the selection rule is the FILENAME: neither of these is `run.json`,
#: so the newest-`pick_baseline_*/run.json` glob cannot reach either.
VOIDED_DIR_PREFIX = "pick_baseline_voided_"
VOIDED_FILENAME = "voided.json"

#: The backend this baseline is for. Passed through the allowlist, never used to
#: build an import path.
DEFAULT_BACKEND = "groot-native"

#: PickSkill's checkpoint-matching defaults, restated so the recorded run says
#: which horizon produced the score.
#:
#: `actions_to_execute` is the obs -> policy -> action iteration budget, i.e. how
#: long the arm gets to finish the task. Raised from PickSkill's default of 10
#: after the validation attempt ran out of iterations mid-task: the arm had
#: grasped the banana, dropped it, and "stopped before it could retry". The
#: checkpoint is not fine-tuned for this table, so recovery attempts are expected
#: and the budget should not be what ends the attempt. The action HORIZON stays at
#: 16 — that one matches the trained checkpoint and must not be tuned.
DEFAULT_ACTIONS_TO_EXECUTE = 20
DEFAULT_ACTION_HORIZON = 16

#: Fallbacks for controller settings, used only when neither a flag nor the live
#: Dum-E YAML config names them. `config.example.yaml` is deliberately NOT
#: consulted: it is a template, and a template camera index must never silently
#: become a device selection (its `front_cam_idx: 1` is a V4L2 metadata node on
#: the host this was first run against).
DEFAULT_ROBOT_TYPE = "so101_follower"
DEFAULT_ROBOT_ID = "my_awesome_follower_arm"

#: Observation shape for the motion-free round trip. Matches what the cameras
#: deliver; the server resizes internally.
SYNTHETIC_FRAME_HEIGHT = 480
SYNTHETIC_FRAME_WIDTH = 640

#: Marker embedded in the clamp-counter self-test message. A message carrying it
#: is SYNTHETIC and is excluded from every reported clamp count, so a self-test
#: can never inflate the number the parity gate reads.
CLAMP_SELFTEST_MARKER = "[CLAMP COUNTER SELF-TEST — synthetic, no motion was commanded]"

#: Field carried ONLY by the controller's per-step clamp report
#: (`SO10xArmController._report_clamped_joints`). All three emitters embed
#: upstream's clamp sentence, so the sentence alone counts one clamped command up
#: to three times; this is what narrows the count to one emitter. Upstream's own
#: message carries `original goal_pos` instead, and PickSkill's roll-up carries
#: neither. Same discriminator `pose_sweep_units_probe.demo_clamp()` uses.
CONTROLLER_CLAMP_MARKER = "max_relative_target="

#: Substrings that identify an exception as "the stack changed under the run"
#: rather than "this attempt missed". Either voids the attempt series.
VOIDING_EXCEPTION_MARKERS = (
    "calibration",
    "PID preset",
    "stiffness",
    "unreachable",
    "Goal_Position pre-arm",
)

#: The top-level keys the standing record's verifier reads. Asserted before any
#: JSON is written, in both modes, so a schema drift surfaces in preflight rather
#: than after ten live attempts.
REQUIRED_RUN_KEYS: Tuple[str, ...] = (
    "attempts",
    "instruction",
    "backend",
    "lerobot_version",
)

#: The per-attempt keys it reads, plus the duration and exception fields the
#: record's per-attempt table is built from.
REQUIRED_ATTEMPT_KEYS: Tuple[str, ...] = (
    "index",
    "instruction",
    "success",
    "clamp_warnings",
    "duration_s",
    "exception",
)


# ---------------------------------------------------------------------------
# Terminal helpers (same shape as scripts/test_live_policy_server.py)
# ---------------------------------------------------------------------------


def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repo_on_path() -> None:
    """Put the repo root on ``sys.path`` so ``embodiment.*``/``policy.*`` resolve.

    Running this file as a script puts ``scripts/`` on ``sys.path[0]``, not the
    repo root, so the project imports fail without this.
    """
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _redact_home(path: Any) -> str:
    """``~``-relative form of a path, so no absolute home directory is quoted."""
    text = str(path)
    home = str(Path.home())
    return "~" + text[len(home):] if text.startswith(home) else text


class NonInteractiveError(RuntimeError):
    """Raised when a human judgment is required and no terminal can supply one."""


# ---------------------------------------------------------------------------
# Clamp-warning counting (SAFE-02 / parity-gate zero-warning requirement)
# ---------------------------------------------------------------------------


def _resolve_clamp_text(clamp_text: Optional[str]) -> str:
    """The clamp sentence, defaulting to the controller's single source of truth."""
    if clamp_text is not None:
        return clamp_text
    _repo_on_path()
    from embodiment.so_arm10x.controller import CLAMP_WARNING_TEXT

    return CLAMP_WARNING_TEXT


def is_clamp_warning(message: str, clamp_text: str) -> bool:
    """Whether ``message`` is a genuine (non-synthetic) clamp warning from ANY emitter."""
    return clamp_text in message and CLAMP_SELFTEST_MARKER not in message


def count_clamp_warnings(messages: List[str], clamp_text: Optional[str] = None) -> int:
    """Count CLAMPED COMMANDS — one per clamped command, not one per emitter.

    All three emitters embed upstream's clamp sentence by design, which is what
    makes a single sink sufficient to SEE every one of them. It also means one
    clamped joint on one step produces three counted messages if the sentence
    alone is the discriminator: upstream's bridged root-logger warning, the
    controller's per-step re-emission, and ``PickSkill``'s per-pick roll-up. The
    recorded ``clamp_warnings`` and ``total_clamp_warnings`` were therefore
    inflated roughly threefold, and the parity gate's "zero clamp warnings"
    criterion reads them as a count of clamped commands.

    So exactly ONE emitter is counted: the controller's per-step report, which
    fires once per ``set_target_state`` whose returned action diverged from the
    request. It is identified by the ``max_relative_target=`` field only that
    emitter carries — the same discriminator ``pose_sweep_units_probe.demo_clamp()``
    uses to tell the two live emitters apart. Choosing the controller's report
    rather than upstream's is deliberate: it is Dum-E's own primary detector,
    derived from ``send_action``'s return value rather than from a log, and it
    survives the stdlib bridge not being installed.

    ``count_clamp_warning_messages_all_emitters`` keeps the raw multi-emitter
    count for evidence, so nothing is lost — it is just no longer the number the
    gate reads.

    Args:
        messages: rendered log messages seen by the counting sink.
        clamp_text: the clamp sentence to match. Defaults to the controller's
            ``CLAMP_WARNING_TEXT`` so there is exactly one source of truth for
            the wording; passing it explicitly keeps this function importable
            without the LeRobot stack.

    Returns:
        The number of clamped commands reported by the controller.
    """
    resolved = _resolve_clamp_text(clamp_text)
    return sum(
        1
        for message in messages
        if is_clamp_warning(message, resolved) and CONTROLLER_CLAMP_MARKER in message
    )


def count_clamp_warning_messages_all_emitters(
    messages: List[str], clamp_text: Optional[str] = None
) -> int:
    """Every non-synthetic clamp message, from every emitter. Evidence, not a gate.

    Deliberately separate from :func:`count_clamp_warnings`: this number is a
    count of LOG LINES, and reading it as a count of clamped commands is the
    conflation that inflated the recorded totals.
    """
    resolved = _resolve_clamp_text(clamp_text)
    return sum(1 for message in messages if is_clamp_warning(message, resolved))


class ClampWarningCounter:
    """Counts clamp warnings per attempt off Dum-E's own loguru stream.

    Installs one loguru sink at WARNING and the stdlib-root bridge, which
    together see every clamp emitter: the controller's primary detector (derived
    from ``send_action``'s return value), ``PickSkill``'s roll-up, and upstream's
    own ``logging.warning`` from ``ensure_safe_goal_position``.
    """

    def __init__(self) -> None:
        self._sink_id: Optional[int] = None
        self._bridge_installed = False
        self.messages: List[str] = []
        self.clamp_text: Optional[str] = None

    def install(self) -> None:
        _repo_on_path()
        from loguru import logger

        from embodiment.so_arm10x.controller import CLAMP_WARNING_TEXT
        from utils import install_stdlib_to_loguru_bridge

        self.clamp_text = CLAMP_WARNING_TEXT
        # The bridge is what carries upstream's root-logger warning into loguru;
        # installing it is what lets ONE sink see every emitter. Idempotent.
        install_stdlib_to_loguru_bridge()
        self._bridge_installed = True
        self._sink_id = logger.add(
            lambda message: self.messages.append(message.record["message"]),
            level="WARNING",
        )

    def remove(self) -> None:
        if self._sink_id is None:
            return
        from loguru import logger

        logger.remove(self._sink_id)
        self._sink_id = None

    @property
    def installed(self) -> bool:
        return self._sink_id is not None and self._bridge_installed

    def begin_attempt(self) -> None:
        """Start a fresh per-attempt window."""
        self.messages.clear()

    def attempt_count(self) -> int:
        """Clamped COMMANDS in this window — the number the parity gate reads.

        One clamped command, not one log line: see :func:`count_clamp_warnings`.
        """
        return count_clamp_warnings(self.messages, self.clamp_text)

    def attempt_emitter_message_count(self) -> int:
        """Clamp log LINES in this window, across all emitters. Evidence only."""
        return count_clamp_warning_messages_all_emitters(self.messages, self.clamp_text)

    def attempt_messages(self) -> List[str]:
        """Every non-synthetic clamp message in this window, from every emitter.

        Kept whole on purpose: the raw stream is the evidence a later reader needs
        to see which emitters fired, even though only the controller's per-step
        report is counted.
        """
        clamp_text = self.clamp_text or ""
        return [
            message
            for message in self.messages
            if is_clamp_warning(message, clamp_text)
        ]

    def self_test(self) -> Dict[str, Any]:
        """Prove the sink AND the stdlib bridge actually carry a clamp warning.

        Emits one probe through the stdlib **root** logger — the stream upstream's
        clamp warning uses, and the one Dum-E's loguru sink does not natively see
        — carrying the real clamp sentence plus
        :data:`CLAMP_SELFTEST_MARKER`. The marker keeps the probe out of every
        reported count, so an installed-and-working counter is demonstrated
        without adding a warning the parity gate would have to explain.
        """
        if not self.installed or not self.clamp_text:
            return {"ok": False, "reason": "counter sink is not installed"}
        before_synthetic = len(self.messages)
        logging.getLogger("run_pick_baseline.clamp_selftest").warning(
            "%s %s", self.clamp_text, CLAMP_SELFTEST_MARKER
        )
        saw_probe = any(
            CLAMP_SELFTEST_MARKER in message
            for message in self.messages[before_synthetic:]
        )
        # Deliberately the ALL-EMITTERS count, not `count_clamp_warnings`. The
        # probe carries no `max_relative_target=` field, so the narrower count
        # would return 0 whether or not the synthetic marker were honoured — the
        # assertion would pass for the wrong reason. Against the broad count, a 0
        # here proves the marker exclusion is what keeps the probe out.
        counted = count_clamp_warning_messages_all_emitters(
            self.messages, self.clamp_text
        )
        self.messages.clear()
        return {
            "ok": bool(saw_probe) and counted == 0,
            "probe_reached_sink": bool(saw_probe),
            "synthetic_probe_counted_as_clamp": counted,
            "clamp_text": self.clamp_text,
            "marker": CLAMP_SELFTEST_MARKER,
            "note": (
                "The probe travels the stdlib root logger through the loguru "
                "bridge, so this proves both the bridge and the counting sink. "
                "It is excluded from every reported clamp count."
            ),
        }


# ---------------------------------------------------------------------------
# Environment / configuration resolution
# ---------------------------------------------------------------------------


def resolve_live_controller_settings(args: argparse.Namespace) -> Dict[str, Any]:
    """Serial port, robot identity and camera indices for the live session.

    Resolution order per field: the command-line flag, then the live Dum-E YAML
    config (``DUME_CONFIG`` or ``my-dum-e.yaml`` at the repo root), then the
    module fallback. ``config.example.yaml`` is never consulted (see
    :data:`DEFAULT_ROBOT_TYPE`).

    The camera indices matter even before an image is read: the controller always
    constructs two cameras and ``connect()`` connects both, so a wrong index
    fails the whole connect before a single tick is read.
    """
    config: Dict[str, Any] = {}
    explicit = os.environ.get("DUME_CONFIG")
    candidate = Path(explicit).expanduser() if explicit else REPO_ROOT / "my-dum-e.yaml"
    if candidate.is_file():
        import yaml  # lazy: the arm-free path needs no YAML parser

        loaded = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
        block = loaded.get("controller")
        config = block if isinstance(block, dict) else {}

    def pick(flag: Any, key: str, fallback: Any) -> Any:
        if flag is not None:
            return flag
        if key in config and config[key] is not None:
            return config[key]
        return fallback

    return {
        "config_file": str(candidate) if config else None,
        "config_file_redacted": _redact_home(candidate) if config else None,
        "robot_type": pick(args.robot_type, "robot_type", DEFAULT_ROBOT_TYPE),
        "robot_id": pick(args.robot_id, "robot_id", DEFAULT_ROBOT_ID),
        "robot_port": pick(args.robot_port, "robot_port", os.getenv("SO_ARM_PORT")),
        "wrist_cam_idx": int(pick(args.wrist_cam_idx, "wrist_cam_idx", 0)),
        "front_cam_idx": int(pick(args.front_cam_idx, "front_cam_idx", 1)),
    }


def inspect_devices(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Read-only presence check for the serial port and both camera nodes.

    Stats the device paths; never opens them. A missing device here is the
    hardware-attach precondition failing, and it must halt before the selector
    builds a backend or the bus is opened.
    """
    report: Dict[str, Any] = {"platform": sys.platform, "devices": {}, "ok": True}
    if not sys.platform.startswith("linux"):
        report["ok"] = True
        report["skipped"] = (
            f"device-node inspection is Linux-specific; platform is {sys.platform}"
        )
        return report

    entries = [("serial", settings["robot_port"])]
    for label, index in (
        ("wrist_camera", settings["wrist_cam_idx"]),
        ("front_camera", settings["front_cam_idx"]),
    ):
        entries.append((label, f"/dev/video{index}"))

    for label, path in entries:
        if not path:
            report["devices"][label] = {"path": None, "present": False}
            report["ok"] = False
            continue
        node = Path(path)
        present = node.exists()
        detail: Dict[str, Any] = {"path": str(node), "present": present}
        if present:
            mode = node.stat().st_mode
            detail["char_device"] = stat.S_ISCHR(mode)
            if not detail["char_device"]:
                report["ok"] = False
        else:
            report["ok"] = False
        report["devices"][label] = detail
    return report


def resolve_backend_selection(requested: str) -> Dict[str, Any]:
    """Make ``DUME_POLICY_BACKEND`` name ``requested``, or refuse.

    An environment variable already naming a DIFFERENT backend is a conflict, not
    something to overwrite: which neural network drives a physical arm is exactly
    the thing ``policy/factory.py`` refuses to decide silently, and a baseline
    attributed to the wrong backend is worse than no baseline.

    Returns a record of what the variable held and what it holds now.
    """
    _repo_on_path()
    from policy.factory import POLICY_BACKEND_ENV_VAR, POLICY_BACKENDS

    if requested not in POLICY_BACKENDS:
        raise ValueError(
            f"--backend {requested!r} is not in the allowlist {POLICY_BACKENDS}. "
            "The selector would refuse it; refusing here rather than building a "
            "run around a name that cannot resolve."
        )

    previous = os.environ.get(POLICY_BACKEND_ENV_VAR)
    if previous is not None and previous != requested:
        raise ValueError(
            f"{POLICY_BACKEND_ENV_VAR} is already set to {previous!r} but this run "
            f"needs {requested!r}. Refusing to overwrite it: the recorded baseline "
            "must be attributable to the backend that actually executed. Unset the "
            "variable or pass --backend to match it."
        )
    os.environ[POLICY_BACKEND_ENV_VAR] = requested
    return {
        "env_var": POLICY_BACKEND_ENV_VAR,
        "value": requested,
        "was_already_set": previous is not None,
    }


def container_image_identity(container_name: str) -> Dict[str, Any]:
    """Best-effort identity of the policy-server container.

    Recorded so the baseline is attributable to a specific served checkpoint.
    Best-effort on purpose: an inaccessible Docker CLI is a gap in the record's
    provenance, not a reason to void ten live attempts, so it is reported as
    unavailable with its reason rather than raised.
    """
    record: Dict[str, Any] = {"container": container_name, "available": False}
    try:
        proc = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.Image}}|{{.Config.Image}}|{{.State.Status}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record

    if proc.returncode != 0:
        record["reason"] = (proc.stderr or proc.stdout or "").strip() or (
            f"docker inspect exited {proc.returncode}"
        )
        return record

    parts = proc.stdout.strip().split("|")
    if len(parts) != 3:
        record["reason"] = f"unexpected docker output: {proc.stdout.strip()!r}"
        return record
    record.update(
        available=True,
        image_id=parts[0],
        image_ref=parts[1],
        state=parts[2],
    )
    return record


def stack_identity() -> Dict[str, Any]:
    """Library versions the baseline is attributable to."""
    from importlib.metadata import PackageNotFoundError, version

    def safe(package: str) -> Optional[str]:
        try:
            return version(package)
        except PackageNotFoundError:
            return None

    return {
        "lerobot_version": safe("lerobot"),
        "torch_version": safe("torch"),
        "python_version": sys.version.split()[0],
    }


# ---------------------------------------------------------------------------
# Motion-free policy round trip
# ---------------------------------------------------------------------------


def synthetic_observation(
    camera_keys: List[str],
    robot_state_keys: List[str],
    height: int = SYNTHETIC_FRAME_HEIGHT,
    width: int = SYNTHETIC_FRAME_WIDTH,
) -> Dict[str, Any]:
    """A raw observation dict shaped like the controller's, with no hardware.

    Values are arbitrary — this exercises the wire path, the pinned annotation
    key and the action decode, never policy quality.
    """
    import numpy as np

    obs: Dict[str, Any] = {
        key: np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        for key in camera_keys
    }
    for key in robot_state_keys:
        obs[key] = 0.0
    return obs


# ---------------------------------------------------------------------------
# The scored attempt
# ---------------------------------------------------------------------------


def prompt_place_object(index: int, total: int, prompt: Callable[[str], str] = input) -> str:
    """Ask the operator to reset the scene. Returns ``"go"`` or ``"abort"``.

    Scene variation between attempts is ACCEPTED by the locked decision (D-10),
    so this asks for a reset, not for a reproduction.
    """
    while True:
        try:
            answer = prompt(
                f"\n  Attempt {index}/{total}: place the object on the table, clear the "
                "arm's path, then press Enter to run it (or type 'abort'): "
            ).strip().lower()
        except EOFError as exc:
            raise NonInteractiveError(
                "stdin closed while waiting for the operator to reset the scene"
            ) from exc
        if answer in ("", "go", "y", "yes"):
            return "go"
        if answer in ("abort", "a", "q", "quit"):
            return "abort"
        print(_yellow("       Type Enter to run the attempt, or 'abort' to stop."))


def prompt_success_judgment(
    index: int,
    exception: Optional[str] = None,
    prompt: Callable[[str], str] = input,
) -> Tuple[bool, Optional[str]]:
    """Read the operator's success judgment for one attempt.

    There is no default and no bulk answer. An empty line re-prompts; a closed
    stdin raises. Whether the object was actually picked up is a human judgment,
    and a machine-supplied one would fabricate the baseline.

    Returns ``(success, operator_note)``.
    """
    if exception:
        print(
            _yellow(
                f"       attempt {index} raised: {exception}\n"
                "       Judge what you SAW, not what the exception implies."
            )
        )
    while True:
        try:
            answer = prompt(
                f"  Attempt {index}: did the arm actually pick up the object? [y/n]: "
            ).strip().lower()
        except EOFError as exc:
            raise NonInteractiveError(
                f"stdin closed while waiting for the attempt {index} success judgment"
            ) from exc
        if answer in ("y", "yes"):
            success = True
        elif answer in ("n", "no"):
            success = False
        else:
            print(_yellow("       Answer 'y' or 'n'. There is no default."))
            continue
        try:
            note = prompt(
                "  Optional note (anything that looked different from v1.0), "
                "Enter to skip: "
            ).strip()
        except EOFError:
            note = ""
        return success, note or None


def build_attempt_record(
    index: int,
    instruction: str,
    success: bool,
    clamp_warnings: int,
    duration_s: float,
    exception: Optional[str],
    **extra: Any,
) -> Dict[str, Any]:
    """Assemble one per-attempt record with every key the standing record needs."""
    record: Dict[str, Any] = {
        "index": index,
        "instruction": instruction,
        "success": bool(success),
        "clamp_warnings": int(clamp_warnings),
        "duration_s": round(float(duration_s), 3),
        "exception": exception,
        "judgment_source": "operator-stdin",
    }
    record.update(extra)
    return record


def run_attempt(
    controller: Any,
    backend: Any,
    args: argparse.Namespace,
    index: int,
    counter: ClampWarningCounter,
    prompt: Callable[[str], str] = input,
) -> Dict[str, Any]:
    """Drive ONE scored pick attempt and return its record.

    Uses the production ``PickSkill`` — the same bounded obs -> policy -> action
    loop the agent runs — with ``pose="initial"`` so the arm parks at its known
    pose before every attempt, and with the pinned instruction passed
    **explicitly** on every inference call (the backend raises rather than
    sending a null annotation, and relying on a stored instruction would leave
    which string conditioned the policy implicit).
    """
    _repo_on_path()
    from embodiment.so_arm10x.skills import PickSkill

    counter.begin_attempt()
    skill = PickSkill(controller, backend)
    started = _now()
    start = time.time()
    exception: Optional[str] = None
    try:
        skill.run(
            actions_to_execute=args.actions_to_execute,
            pose="initial",
            language_instruction=args.instruction,
            action_horizon=args.action_horizon,
        )
    except Exception as exc:  # noqa: BLE001 - recorded verbatim, never swallowed
        exception = f"{type(exc).__name__}: {exc}"
        print(_red(f"       attempt {index} raised {exception}"))
    duration = time.time() - start

    success, note = prompt_success_judgment(index, exception, prompt=prompt)
    return build_attempt_record(
        index=index,
        instruction=args.instruction,
        success=success,
        clamp_warnings=counter.attempt_count(),
        duration_s=duration,
        exception=exception,
        started_at=started,
        ended_at=_now(),
        clamp_warning_messages=counter.attempt_messages(),
        # The raw log-line count across all three emitters, recorded alongside
        # `clamp_warnings` (which counts clamped COMMANDS). Keeping both makes the
        # distinction explicit in the evidence rather than implicit in a helper.
        clamp_warning_messages_all_emitters=counter.attempt_emitter_message_count(),
        actions_to_execute=args.actions_to_execute,
        action_horizon=args.action_horizon,
        operator_note=note,
    )


def exception_voids_series(exception: Optional[str]) -> bool:
    """Whether an attempt's exception means the STACK changed under the run.

    A calibration, PID or transport failure invalidates the whole series — the
    arm was not in the state the baseline claims — whereas an ordinary miss is
    just a failed attempt.
    """
    if not exception:
        return False
    lowered = exception.lower()
    return any(marker.lower() in lowered for marker in VOIDING_EXCEPTION_MARKERS)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def validate_record(payload: Dict[str, Any], *, scored: bool = False) -> List[str]:
    """Return a list of schema problems in ``payload``; empty means valid.

    Checked before anything is written, in both modes, so a drift in the keys the
    standing record's verifier reads surfaces during preflight rather than after
    ten live attempts.

    Args:
        payload: the assembled run record.
        scored: whether this record is about to be written under the SCORED
            filename. A scored record must contain at least one scored attempt —
            a zero-attempt file at the path reserved for baselines wins the
            documented newest-``run.json`` selection rule while carrying no
            score at all, which is the mis-selection the distinct filenames
            exist to make impossible.
    """
    problems: List[str] = []
    for key in REQUIRED_RUN_KEYS:
        if key not in payload:
            problems.append(f"missing top-level key {key!r}")
    if not isinstance(payload.get("attempts"), list):
        problems.append("'attempts' must be a list")
    elif scored and not payload["attempts"]:
        problems.append(
            "a scored record must contain at least one attempt: a zero-attempt "
            "file under the scored filename would be selected as the standing "
            "baseline while recording no score"
        )
    if not payload.get("instruction"):
        problems.append("'instruction' must be a non-empty string")
    for position, attempt in enumerate(payload.get("attempts") or []):
        if not isinstance(attempt, dict):
            problems.append(f"attempts[{position}] is not an object")
            continue
        for key in REQUIRED_ATTEMPT_KEYS:
            if key not in attempt:
                problems.append(f"attempts[{position}] missing key {key!r}")
        if not isinstance(attempt.get("clamp_warnings"), int):
            problems.append(f"attempts[{position}].clamp_warnings must be an int")
        if not isinstance(attempt.get("success"), bool):
            problems.append(f"attempts[{position}].success must be a bool")
    return problems


def record_run(
    payload: Dict[str, Any],
    *,
    dry_run: bool,
    out_root: Optional[str] = None,
    voided: bool = False,
) -> Path:
    """Write the run payload under the gitignored corpus directory.

    Scored runs land at ``pick_baseline_<stamp>/run.json``; preflights land at
    ``pick_baseline_dryrun_<stamp>/preflight.json``; aborted or voided scored runs
    land at ``pick_baseline_voided_<stamp>/voided.json``. All three are distinct in
    both prefix and filename so nothing but a completed scored series can be
    mistaken for — or selected as — a baseline.

    Args:
        payload: the assembled run record.
        dry_run: this is a motion-free preflight.
        out_root: override the evidence root.
        voided: the scored series was aborted or invalidated. Routed away from the
            scored filename because a void run is not a baseline, and a
            zero-attempt one is the newest match for the selection glob.

    Raises:
        ValueError: the payload does not satisfy :func:`validate_record`. Writing
            an unreadable record would be worse than failing loudly.
    """
    scored = not dry_run and not voided
    problems = validate_record(payload, scored=scored)
    if problems:
        raise ValueError(
            "Refusing to write a run record that the standing baseline's verifier "
            f"could not read: {problems}"
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if dry_run:
        prefix, filename = DRYRUN_DIR_PREFIX, DRYRUN_FILENAME
    elif voided:
        prefix, filename = VOIDED_DIR_PREFIX, VOIDED_FILENAME
    else:
        prefix, filename = SCORED_DIR_PREFIX, SCORED_FILENAME
    root = Path(out_root).expanduser() if out_root else REPO_ROOT / CORPUS_DIRNAME
    directory = root / f"{prefix}{stamp}"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def build_run_payload(
    args: argparse.Namespace,
    *,
    mode: str,
    attempts: List[Dict[str, Any]],
    stack: Dict[str, Any],
    checks: Dict[str, bool],
    extras: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the full run record, including its non-negotiable caveat."""
    successes = sum(1 for attempt in attempts if attempt.get("success"))
    payload: Dict[str, Any] = {
        "kind": "pick_baseline",
        "mode": mode,
        "recorded_at": _now(),
        # --- the keys the standing record's verifier reads ---
        "instruction": args.instruction,
        "backend": args.backend,
        "lerobot_version": stack.get("lerobot_version"),
        "attempts": attempts,
        # --- score ---
        "attempts_requested": args.attempts,
        "attempts_performed": len(attempts),
        "successes": successes,
        "score": f"{successes}/{len(attempts)}" if attempts else None,
        "total_clamp_warnings": sum(
            int(attempt.get("clamp_warnings", 0)) for attempt in attempts
        ),
        # --- attributable stack ---
        "stack": stack,
        "checks": checks,
        "scene_variation": (
            "ACCEPTED (CONTEXT.md D-10): the instruction string is the only pinned "
            "variable; the operator resets the scene between attempts and variation "
            "is not treated as a defect."
        ),
        "caveat": (
            "This is a FUNCTIONAL SMOKE CHECK, not a controlled numerical "
            "comparison. Only the instruction string is pinned, so the entire "
            "numerical burden for the parity gate rests on the offline frozen-corpus "
            "evidence. Do not cite this score as evidence of numerical parity, and "
            "do not read a live discrepancy as policy drift when an uncontrolled "
            "scene could explain it."
        ),
        "judgment_policy": (
            "Every success value came from an interactive human judgment typed at "
            "the terminal. This runner has no flag that can supply one."
        ),
    }
    payload.update(extras)
    return payload


# ---------------------------------------------------------------------------
# Preflight: the motion-free wiring verification
# ---------------------------------------------------------------------------


def preflight(args: argparse.Namespace, counter: ClampWarningCounter) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """Verify the whole wiring path without commanding any motion.

    Order is deliberate and mirrors the production wiring: the instruction and
    the devices are checked read-only, then the SELECTOR runs and the server is
    proven reachable, and only then is the controller constructed — so a bad
    backend name or a dead server raises before the serial bus is touched at all.

    Returns ``(checks, context)`` where ``context`` carries the constructed
    backend and controller for the caller to use or dispose of.
    """
    checks: Dict[str, bool] = {}
    context: Dict[str, Any] = {"evidence": {}}
    evidence: Dict[str, Any] = context["evidence"]
    total = 8 if args.with_arm else 7
    step = 0

    def head(title: str) -> str:
        nonlocal step
        step += 1
        return f"\n[{step}/{total}] {title} ..."

    # 1 — the pinned instruction -------------------------------------------
    print(head("pinned instruction — recorded verbatim"))
    instruction_ok = bool(args.instruction and args.instruction.strip())
    if instruction_ok:
        print(_green(f"  PASS: instruction pinned: {args.instruction!r}"))
    else:
        print(_red("  FAIL: --instruction is empty; the policy must never be sent a null annotation"))
    checks["instruction_pinned"] = instruction_ok
    evidence["instruction"] = args.instruction

    # 2 — device presence, read-only ---------------------------------------
    print(head("hardware-attach precondition — device nodes, read-only"))
    settings = resolve_live_controller_settings(args)
    devices = inspect_devices(settings)
    evidence["controller_settings"] = settings
    evidence["devices"] = devices
    for label, detail in devices.get("devices", {}).items():
        print(f"       {label}: {detail}")
    if devices["ok"]:
        print(_green(f"  PASS: serial and both camera nodes present (config: {settings['config_file_redacted']})"))
    else:
        print(_red("  FAIL: a required device node is missing — the 05-06 hardware gate is no longer satisfied"))
    checks["devices_present"] = bool(devices["ok"])

    # 3 — the selector ------------------------------------------------------
    print(head(f"policy-backend selector — {args.backend!r} through the allowlist"))
    backend = None
    try:
        selection = resolve_backend_selection(args.backend)
        _repo_on_path()
        from policy.factory import make_policy_backend
        from shared import IPolicyBackend

        backend = make_policy_backend(
            host=args.host,
            port=args.port,
            language_instruction=args.instruction,
        )
        if not isinstance(backend, IPolicyBackend):
            raise TypeError(f"selector returned a {type(backend).__name__}, not an IPolicyBackend")
        evidence["backend_selection"] = selection
        evidence["backend_class"] = type(backend).__name__
        evidence["backend_module"] = type(backend).__module__
        evidence["backend_obtained_via"] = "policy.factory.make_policy_backend"
        print(
            _green(
                f"  PASS: {selection['env_var']}={selection['value']} -> "
                f"{type(backend).__module__}.{type(backend).__name__} "
                "(obtained from the selector, not constructed here)"
            )
        )
        checks["selector"] = True
    except Exception as exc:  # noqa: BLE001
        print(_red(f"  FAIL: selector raised {type(exc).__name__}: {exc}"))
        evidence["selector_error"] = f"{type(exc).__name__}: {exc}"
        checks["selector"] = False
    context["backend"] = backend

    # 4 — server reachability ----------------------------------------------
    print(head(f"policy server reachability — {args.host}:{args.port}"))
    if backend is None:
        print(_red("  FAIL: no backend to ping (the selector did not return one)"))
        checks["server_reachable"] = False
    else:
        try:
            reachable = bool(backend.ping())
            evidence["ping"] = reachable
            if reachable:
                print(_green("  PASS: server reachable (ping ok)"))
            else:
                print(_red("  FAIL: ping returned False — the policy server is not serving"))
            checks["server_reachable"] = reachable
        except Exception as exc:  # noqa: BLE001
            print(_red(f"  FAIL: ping raised {type(exc).__name__}: {exc}"))
            evidence["ping_error"] = f"{type(exc).__name__}: {exc}"
            checks["server_reachable"] = False

    # 5 — full round trip on a synthetic observation (no hardware) ----------
    print(head("policy round trip — synthetic observation, pinned instruction, no arm"))
    if backend is None or not checks.get("server_reachable"):
        print(_yellow("  FAIL: skipped — needs a reachable backend"))
        checks["policy_round_trip"] = False
    else:
        try:
            camera_keys = list(getattr(backend, "camera_keys", ["wrist", "front"]))
            state_keys = list(
                getattr(
                    backend,
                    "robot_state_keys",
                    [
                        "shoulder_pan.pos",
                        "shoulder_lift.pos",
                        "elbow_flex.pos",
                        "wrist_flex.pos",
                        "wrist_roll.pos",
                        "gripper.pos",
                    ],
                )
            )
            obs = synthetic_observation(camera_keys, state_keys)
            start = time.time()
            actions = backend.get_action(obs, args.instruction)
            latency = time.time() - start
            if not isinstance(actions, list) or not actions:
                raise TypeError(f"expected a non-empty list of action dicts, got {type(actions)}")
            missing = [key for key in state_keys if key not in actions[0]]
            if missing:
                raise KeyError(f"action dict missing keys: {missing}")
            evidence["round_trip"] = {
                "horizon": len(actions),
                "latency_s": round(latency, 3),
                "action_keys": sorted(actions[0].keys()),
                "instruction_passed_explicitly": True,
            }
            print(
                _green(
                    f"  PASS: {len(actions)} action steps in {latency:.2f}s "
                    f"(horizon={len(actions)}); instruction passed explicitly on the call"
                )
            )
            checks["policy_round_trip"] = True
        except Exception as exc:  # noqa: BLE001
            print(_red(f"  FAIL: round trip raised {type(exc).__name__}: {exc}"))
            evidence["round_trip_error"] = f"{type(exc).__name__}: {exc}"
            checks["policy_round_trip"] = False

    # 6 — the clamp-warning counter ----------------------------------------
    print(head("clamp-warning counter — loguru sink plus stdlib bridge"))
    self_test = counter.self_test()
    evidence["clamp_counter_self_test"] = self_test
    if self_test.get("ok"):
        print(
            _green(
                "  PASS: a clamp-worded warning emitted on the stdlib ROOT logger "
                "reached the counting sink, and the synthetic probe is excluded "
                "from every reported count"
            )
        )
    else:
        print(_red(f"  FAIL: clamp counter not wired: {self_test}"))
    checks["clamp_counter"] = bool(self_test.get("ok"))

    # 7 — controller construction and the calibration file ------------------
    print(head("controller construction and calibration file — no bus opened"))
    controller = None
    try:
        _repo_on_path()
        from embodiment.so_arm10x.controller import (
            SO10xArmController,
            assert_calibration_loaded,
        )

        if not settings["robot_port"]:
            raise ValueError(
                "No serial port: pass --robot-port, set SO_ARM_PORT, or name "
                "robot_port in the controller block of the live Dum-E config"
            )
        controller = SO10xArmController(
            robot_type=settings["robot_type"],
            robot_port=settings["robot_port"],
            robot_id=settings["robot_id"],
            wrist_cam_idx=settings["wrist_cam_idx"],
            front_cam_idx=settings["front_cam_idx"],
        )
        if controller.is_connected():
            raise RuntimeError(
                "the controller reports itself connected straight out of the "
                "constructor; the selector must be able to raise before the bus opens"
            )
        calibration_path, calibration_sha256 = assert_calibration_loaded(
            robot_name=getattr(controller.robot, "name", None),
            robot_id=settings["robot_id"],
            expected_path=getattr(controller.robot, "calibration_fpath", None),
        )
        evidence["controller"] = {
            "robot_class": type(controller.robot).__name__,
            "robot_name": getattr(controller.robot, "name", None),
            "use_degrees": bool(getattr(controller.config, "use_degrees", True)),
            "max_relative_target": getattr(controller.config, "max_relative_target", None),
            "connected_after_construction": controller.is_connected(),
        }
        evidence["calibration_path"] = str(calibration_path)
        evidence["calibration_path_redacted"] = _redact_home(calibration_path)
        evidence["calibration_sha256"] = calibration_sha256
        print(
            _green(
                f"  PASS: controller constructed unconnected; calibration at "
                f"{_redact_home(calibration_path)} sha256={calibration_sha256}"
            )
        )
        checks["controller_constructed"] = True
    except Exception as exc:  # noqa: BLE001
        print(_red(f"  FAIL: {type(exc).__name__}: {exc}"))
        evidence["controller_error"] = f"{type(exc).__name__}: {exc}"
        checks["controller_constructed"] = False
    context["controller"] = controller

    # 8 — optional: open the bus and run the connect-time assertions --------
    if args.with_arm:
        print(head("live bus — connect(), Goal_Position pre-arm, PID read-back (no motion)"))
        if controller is None:
            print(_red("  FAIL: no controller to connect"))
            checks["arm_connect"] = False
        else:
            try:
                connect_arm(controller, evidence)
                print(
                    _green(
                        "  PASS: connected; Goal_Position pre-armed, calibration "
                        "asserted, PID read back on every motor — no motion commanded"
                    )
                )
                checks["arm_connect"] = True
                context["connected"] = True
            except Exception as exc:  # noqa: BLE001
                print(_red(f"  FAIL: connect raised {type(exc).__name__}: {exc}"))
                evidence["connect_error"] = f"{type(exc).__name__}: {exc}"
                checks["arm_connect"] = False

    return checks, context


def connect_arm(controller: Any, evidence: Dict[str, Any]) -> None:
    """``controller.connect()`` and record what its connect-time guards reported.

    Goes through ``connect()`` deliberately: it writes ``Goal_Position <-
    Present_Position`` while torque is still OFF, which is what makes the
    subsequent torque-enable a *hold* rather than a slam toward raw tick 0. Never
    call ``robot.connect()`` directly and never re-implement the pre-arm here.
    """
    controller.connect()
    calibration_path, calibration_sha256 = controller._assert_calibration_loaded()
    pid_readback = controller._assert_pid_landed()
    evidence["calibration_path"] = str(calibration_path)
    evidence["calibration_path_redacted"] = _redact_home(calibration_path)
    evidence["calibration_sha256"] = calibration_sha256
    evidence["pid_readback"] = pid_readback


def disconnect_arm(controller: Any, *, keep_torque: bool = True) -> None:
    """Disconnect, by default leaving torque enabled so the arm holds its pose.

    Dropping torque at disconnect lets the arm sag under gravity — motion this
    script has no reason to cause — so the default preserves the hold.
    """
    if controller is None:
        return
    try:
        if keep_torque:
            controller.config.disable_torque_on_disconnect = False
        controller.disconnect()
    except Exception as exc:  # noqa: BLE001
        print(_yellow(f"       note: disconnect raised {type(exc).__name__}: {exc}"))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=10,
        help="Number of scored attempts (default 10).",
    )
    parser.add_argument(
        "--instruction",
        required=True,
        help="The PINNED instruction string, recorded verbatim and passed on every "
        "inference call. It is the only controlled variable of this baseline.",
    )
    parser.add_argument("--host", default="localhost", help="Policy server host.")
    parser.add_argument("--port", type=int, default=5555, help="Policy server port.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        help=f"Backend to select through the allowlist (default {DEFAULT_BACKEND}).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Override the evidence root (default: the gitignored corpus/ directory).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Motion-free preflight only: instruction, devices, selector, server "
        "reachability, a synthetic policy round trip, the clamp counter, the "
        "controller construction and the calibration checksum. Commands no motion "
        "and does not open the serial bus unless --with-arm is also given.",
    )
    parser.add_argument(
        "--with-arm",
        action="store_true",
        help="With --dry-run: additionally open the bus via controller.connect() to "
        "run the Goal_Position pre-arm, the calibration assertion and the PID "
        "read-back. Still commands no motion.",
    )
    parser.add_argument(
        "--container",
        default="gr00t-server",
        help="Name of the policy-server container, recorded for provenance.",
    )
    parser.add_argument(
        "--actions-to-execute",
        type=int,
        default=DEFAULT_ACTIONS_TO_EXECUTE,
        help=f"obs->policy->action iterations per attempt (default {DEFAULT_ACTIONS_TO_EXECUTE}).",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help=f"Action steps executed per chunk (default {DEFAULT_ACTION_HORIZON}, "
        "matching the trained checkpoint).",
    )
    # Controller settings default to None so the LIVE Dum-E YAML config supplies
    # them; see resolve_live_controller_settings.
    parser.add_argument("--robot-port", default=None, help="Arm serial port (else config, else SO_ARM_PORT).")
    parser.add_argument("--robot-id", default=None, help="Calibration id (else config).")
    parser.add_argument("--robot-type", default=None, help="Robot type (else config).")
    parser.add_argument("--wrist-cam-idx", type=int, default=None, help="Wrist camera index (else config).")
    parser.add_argument("--front-cam-idx", type=int, default=None, help="Front camera index (else config).")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.with_arm and not args.dry_run:
        print(_red("--with-arm is only meaningful with --dry-run (a scored run always connects)."))
        return 2
    if args.attempts < 1:
        print(_red("--attempts must be at least 1."))
        return 2

    # A scored series needs a human. Refuse rather than invent judgments: a
    # non-interactive scored run could only produce a fabricated baseline. Checked
    # FIRST, before any import or logging setup, so a refusal changes nothing.
    if not args.dry_run and not sys.stdin.isatty():
        print(
            _red(
                "  REFUSING TO RUN: a scored series needs an interactive terminal.\n"
                "  Each attempt's success is a human judgment typed here; there is no\n"
                "  flag that supplies one, because a machine-supplied judgment would\n"
                "  fabricate the baseline the parity gate is measured against."
            )
        )
        return 2

    _repo_on_path()
    from utils import setup_robot_logging

    setup_robot_logging()

    mode = "dry-run" if args.dry_run else "scored"
    print("=" * 72)
    print(f" groot-native live pick baseline (LR-06 / BACK-06) — mode: {mode}")
    print(f" instruction (pinned): {args.instruction!r}")
    if args.dry_run:
        print(" NO MOTION IS COMMANDED IN THIS MODE.")
    else:
        print(f" {args.attempts} scored attempts — each judged by the operator at this terminal.")
    print("=" * 72)

    counter = ClampWarningCounter()
    counter.install()
    stack = stack_identity()
    attempts: List[Dict[str, Any]] = []
    voided = False
    void_reason: Optional[str] = None
    controller = None
    connected = False

    try:
        checks, context = preflight(args, counter)
        controller = context.get("controller")
        backend = context.get("backend")
        connected = bool(context.get("connected"))
        evidence = context["evidence"]
        evidence["container"] = container_image_identity(args.container)

        if args.dry_run:
            payload = build_run_payload(
                args,
                mode=mode,
                attempts=[],
                stack=stack,
                checks=checks,
                extras={
                    "evidence": evidence,
                    "voided": False,
                    "void_reason": None,
                    "not_a_baseline": (
                        "PREFLIGHT ONLY. This file records a motion-free wiring "
                        "verification with zero scored attempts. It is not a "
                        "baseline and must never be cited as one."
                    ),
                    "deferred_to_operator": [
                        "the scored attempt series (needs props on the table and a "
                        "human to judge each attempt)",
                    ]
                    + (
                        []
                        if args.with_arm
                        else [
                            "the connect-time PID read-back and calibration assertion "
                            "on the live bus (re-run with --dry-run --with-arm)",
                        ]
                    ),
                },
            )
        else:
            failed = [name for name, ok in checks.items() if not ok]
            if failed:
                print(
                    _red(
                        f"\n  ABORTING before touching the arm: preflight failed on {failed}."
                    )
                )
                voided, void_reason = True, f"preflight failed: {failed}"
            else:
                print("\n" + "-" * 72)
                print(" Connecting the arm ...")
                connect_arm(controller, evidence)
                connected = True
                checks["arm_connect"] = True
                print(_green("  connected: pre-arm, calibration and PID read-back all passed"))

                print("\n" + "-" * 72)
                print(
                    f" Scored series: {args.attempts} attempts on {args.instruction!r}.\n"
                    " Scene variation between attempts is ACCEPTED (D-10) — reset the\n"
                    " scene, do not try to reproduce it. Answer each judgment honestly;\n"
                    " a sub-10 score is a finding to record, never a run to repeat."
                )
                with backend.session() as policy:
                    for index in range(1, args.attempts + 1):
                        if prompt_place_object(index, args.attempts) == "abort":
                            voided, void_reason = True, (
                                f"operator aborted before attempt {index}"
                            )
                            break
                        record = run_attempt(controller, policy, args, index, counter)
                        attempts.append(record)
                        mark = _green("SUCCESS") if record["success"] else _red("FAILURE")
                        print(
                            f"       attempt {index}: {mark}  "
                            f"clamp_warnings={record['clamp_warnings']}  "
                            f"{record['duration_s']:.1f}s"
                        )
                        if exception_voids_series(record["exception"]):
                            voided, void_reason = True, (
                                "the stack changed under the run: "
                                f"{record['exception']}"
                            )
                            print(
                                _red(
                                    "       This exception means the stack changed under "
                                    "the run — the attempt series is VOID."
                                )
                            )
                            break
                        if index < args.attempts:
                            policy.reset()
                # Via ready first: the final park runs from wherever the last
                # attempt ended, which is exactly the arbitrary-pose descent that
                # can sweep the arm into the table (see PickSkill.run).
                print("\n Parking the arm (via ready) at its initial pose ...")
                controller.move_to_ready_pose()
                controller.move_to_initial_pose()

            payload = build_run_payload(
                args,
                mode=mode,
                attempts=attempts,
                stack=stack,
                checks=checks,
                extras={
                    "evidence": evidence,
                    "voided": voided,
                    "void_reason": void_reason,
                },
            )

        # `voided` routes an aborted or invalidated scored series away from the
        # scored filename. The preflight-abort path in particular reaches here with
        # `attempts == []`, and writing that as `pick_baseline_<stamp>/run.json`
        # put a zero-attempt file at exactly the path the standing record's
        # newest-first selection reads.
        target = record_run(
            payload, dry_run=args.dry_run, out_root=args.out, voided=voided
        )
    except NonInteractiveError as exc:
        print(_red(f"\n  ABORTED: {exc}. No judgment was invented; nothing was recorded."))
        return 1
    finally:
        if connected:
            disconnect_arm(controller, keep_torque=True)
        counter.remove()

    print("\n" + "=" * 72)
    passed = sum(1 for ok in checks.values() if ok)
    for name, ok in checks.items():
        print(f"  {_green('PASS') if ok else _red('FAIL')}  {name}")
    print(f" {passed}/{len(checks)} checks passed")
    if args.dry_run:
        print(_yellow(" PREFLIGHT ONLY — no attempt was scored, this is NOT a baseline."))
        for item in payload["deferred_to_operator"]:
            print(_yellow(f"   deferred: {item}"))
    else:
        successes = payload["successes"]
        performed = payload["attempts_performed"]
        print(
            f" score: {successes}/{performed} successful "
            f"(requested {args.attempts}); "
            f"clamp warnings total: {payload['total_clamp_warnings']}"
        )
        if voided:
            print(_red(f" RUN VOID: {void_reason}"))
        print(_yellow(" Functional smoke check, NOT numerical parity evidence (D-10)."))
    print(f" evidence: {_redact_home(target)}")
    print("=" * 72)

    if passed != len(checks) or voided:
        return 1
    if args.dry_run:
        return 0
    return 0 if payload["successes"] == args.attempts and payload["attempts_performed"] == args.attempts else 1


if __name__ == "__main__":
    sys.exit(main())
