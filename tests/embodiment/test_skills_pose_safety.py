'The initial-pose reset ordering, which is a physical-safety property.'

from types import SimpleNamespace
from typing import Any, List

import numpy as np

from embodiment.so_arm10x.controller import SO10xArmController
from embodiment.so_arm10x.skills import PickSkill, ResetPoseSkill

#: The two pose vectors, as `controller.py` defines them. Indexed by joint order
#: (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper).
READY = [0.0, -90, 75.0, 75.0, -90.0, 0.0]
INITIAL = [0.0, -102, 96.0, 76.0, -90.0, 0.0]


def _controller_recording_targets() -> Any:
    'A controller shell that records the pose VECTORS handed to the arm.'
    controller = SO10xArmController.__new__(SO10xArmController)
    controller.targets = []  # type: ignore[attr-defined]

    def _record(target):
        controller.targets.append(  # type: ignore[attr-defined]
            [round(float(v), 4) for v in np.asarray(target).tolist()]
        )
        return {}

    controller.set_target_state = _record  # type: ignore[assignment]
    return controller


def test_move_to_initial_pose_reaches_ready_first():
    """The single chokepoint: the descent to initial is preceded by ready."""
    controller = _controller_recording_targets()

    controller.move_to_initial_pose()

    assert controller.targets == [READY, INITIAL], (
        "move_to_initial_pose must command ready before initial so the descent "
        f"is never issued from an arbitrary pose; got {controller.targets}"
    )


def test_reset_pose_skill_inherits_the_waypoint():
    """`ResetPoseSkill` is the reset tool the model calls on failure.

    It was one of the three paths missed when the waypoint lived at call sites.
    """
    controller = _controller_recording_targets()
    controller.get_current_images = lambda: {}  # type: ignore[assignment]

    ResetPoseSkill(controller, SimpleNamespace()).run()

    assert controller.targets == [READY, INITIAL], (
        f"the reset tool must inherit the ready waypoint; got {controller.targets}"
    )


def test_pick_skill_initial_reset_ends_at_ready():
    """The pick begins from ready -- keep the trailing move or the baseline shifts."""
    controller = _controller_recording_targets()
    controller.get_current_images = lambda: {}  # type: ignore[assignment]

    # `actions_to_execute=0` isolates the pose reset: the obs -> policy -> action
    # loop never runs, so no policy client is needed and nothing else commands motion.
    PickSkill(controller, SimpleNamespace()).run(
        actions_to_execute=0, pose="initial", language_instruction="x"
    )

    assert controller.targets == [READY, INITIAL, READY], (
        "the pick reset must be ready -> initial -> ready: the waypoint protects "
        "the descent, and the final ready is where the pick begins; got "
        f"{controller.targets}"
    )
    assert controller.targets[-1] == READY


def test_resume_pose_commands_no_reset_motion():
    """``pose="resume"`` must not move the arm at all."""
    controller = _controller_recording_targets()
    controller.get_current_images = lambda: {}  # type: ignore[assignment]

    PickSkill(controller, SimpleNamespace()).run(
        actions_to_execute=0, pose="resume", language_instruction="x"
    )

    assert controller.targets == []


def test_ready_pose_does_not_route_through_initial():
    """Guard against a symmetric "fix" that makes the two helpers mutually recursive.

    ``move_to_ready_pose`` must stay a single primitive move; routing it via
    initial would both invert the safety property and recurse forever.
    """
    controller = _controller_recording_targets()

    controller.move_to_ready_pose()

    assert controller.targets == [READY]
