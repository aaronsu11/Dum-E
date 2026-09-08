"""The ``pose="initial"`` reset ordering, which is a physical-safety property.

``PickSkill.run(pose="initial")`` runs at the start of every scored baseline
attempt, so it executes from wherever the *previous* attempt left the arm — an
arbitrary policy pose, not a known one.

The initial pose is the low, extended one (``shoulder_lift`` -102,
``elbow_flex`` 96); the ready pose is retracted (-90, 75). Driving straight to
initial from an arbitrary pose was observed on hardware sweeping the arm toward
the table. Reaching ready first turns one unbounded move into two bounded ones.
It is also the reset-from-extreme-pose case plan 05-05 predicted would produce
the largest legitimate clamp delta (~190 on ``elbow_flex``, above the 160.0
clamp).

Two properties are pinned here, and the second is easy to lose while "fixing"
the first:

1. ready comes BEFORE initial, so the descent never starts from an arbitrary pose;
2. the sequence still ENDS at ready, because the pick begins from the ready pose
   and changing that would make the run non-comparable to the v1.0 baseline.

Hermetic: no serial port, no cameras, no policy server.
"""

from types import SimpleNamespace
from typing import List

from embodiment.so_arm10x.skills import PickSkill


class PoseRecordingController:
    """Records pose-move calls in order; every other member is inert."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    def move_to_initial_pose(self) -> None:
        self.calls.append("initial")

    def move_to_ready_pose(self) -> None:
        self.calls.append("ready")

    # `run` returns the latest images after the loop; with zero iterations the
    # pick loop body never executes, so these only need to exist.
    def get_current_images(self):
        return {}

    def get_observation(self):  # pragma: no cover - unreachable at 0 iterations
        raise AssertionError("the pick loop must not run in a pose-ordering test")


def _skill() -> PickSkill:
    controller = PoseRecordingController()
    # `actions_to_execute=0` isolates the pose reset: the obs -> policy -> action
    # loop never runs, so no policy client is needed and nothing commands motion.
    return PickSkill(controller, SimpleNamespace())


def test_initial_pose_reset_reaches_ready_before_initial():
    """The descent to the low initial pose must start from ready, not arbitrary."""
    skill = _skill()

    skill.run(actions_to_execute=0, pose="initial", language_instruction="x")

    calls = skill.controller.calls
    assert "ready" in calls and "initial" in calls
    assert calls.index("ready") < calls.index("initial"), (
        f"ready must precede initial so the descent is bounded; got {calls}"
    )


def test_initial_pose_reset_still_ends_at_ready():
    """The pick starts from ready -- keep the trailing move or the baseline shifts."""
    skill = _skill()

    skill.run(actions_to_execute=0, pose="initial", language_instruction="x")

    assert skill.controller.calls[-1] == "ready", (
        "the sequence must end at ready, because the pick begins there and the "
        f"v1.0 baseline was taken from there; got {skill.controller.calls}"
    )


def test_initial_pose_reset_full_expected_order():
    """Pin the whole sequence: ready -> initial -> ready."""
    skill = _skill()

    skill.run(actions_to_execute=0, pose="initial", language_instruction="x")

    assert skill.controller.calls == ["ready", "initial", "ready"]


def test_resume_pose_commands_no_reset_motion():
    """``pose="resume"`` must not move the arm at all."""
    skill = _skill()

    skill.run(actions_to_execute=0, pose="resume", language_instruction="x")

    assert skill.controller.calls == []
