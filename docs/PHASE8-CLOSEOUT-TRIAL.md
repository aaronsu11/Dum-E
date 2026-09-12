# Phase 8 remaining integrated verification

Status: prepared; not physically executed or approved yet.

One supervised physical trial is proposed. Aaron is beside the arm, able to
stop it, with banana and apple visible and reachable in a clear workspace. Keep
the wrist camera cable secured throughout motion. No intentional server failure
is part of this run.

The existing voice/MCP path triggers the guarded GPU/controller runner, starting
with the banana instruction. Initial/ready reset is followed by at most 160
chunks of 16 actions, paced at 50 ms: nominally 128 seconds. A separate 140-second
active-time limit latches a stop. The previous 320-action approval does not
release this larger trial. Do not continue unsafe motion for a duration target.

When inference becomes active, the child announces readiness through MCP. Aaron
says to change the running task to the apple. The relay accepts only an
unambiguous banana/apple instruction, calls the existing AsyncPickSkill.set_task,
and acknowledges application only after the scheduler has changed epoch. The
original single inference worker, stale-action checks and raw hardware stop
latch remain active. Cancellation sends SIGTERM to the guarded child. No
automatic restart, reset after a fault, or second trial is enabled.

## Voice measurements

Before arming, ask three short questions, one at a time, and let each answer
finish: “Can you hear me?”, “What color is a ripe banana?”, “Are you still there?”
Record turn IDs, user-stop and bot-start timings from the existing voice logs.
Repeat the same questions during the active GPU/physical task, spaced through
the run. Distinguish conversational replies from tool/progress announcements.

Report all three baseline and all three active-task latencies, their medians,
and any missing/failed/interrupted turns. This is a small descriptive check;
three samples do not establish a statistical tail guarantee. Review deviations
against the measured baseline band instead of inventing a post-hoc pass threshold.

## Evidence and outcome

Retain current approval/source/service/device bindings, real joint/command trace,
model logs, voice dispatch/control events and application acknowledgments. Verify
that the policy remains in the same service instance with no reload and that
no old-epoch action is dispatched after retarget. Aaron must observe the physical
change toward the apple and report safe behavior. A software acknowledgment
alone does not prove physical retargeting.

If at least two minutes of integrated operation and the measurements are not
obtained, preserve the incomplete result; do not mark the full-duration criterion
passed. Startup optimization stays deferred. Phase completion also requires an
honest verification report/review of the accumulated evidence, not simply one
successful task result.
