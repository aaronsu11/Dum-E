# Multi-task voice session startup

The second physical voice trial succeeded: 320 actions, no safety stop or clamp,
and Aaron confirmed success. Evidence is in
`corpus/phase8-voice-physical-trial2-20260912/trial-result.json`.

Dispatch to the recorded motion-stage entry took 36.729 seconds:

- 15.429 seconds before the runner start record: process/import startup, initial
  snapshot checks and camera probes.
- 18.660 seconds for policy setup/load, warmup and repeated snapshot validation.
- 2.639 seconds for controller setup, calibration and PID readback.

The motion-stage timestamp precedes the skill's live warmup and reset; it is not
an independently measured first movement. Server chunk generation during this
live voice run averaged 163.284 ms over 40 chunks, distinct from startup delay.

The one-shot physical verification harness launches a new process/client for
each trial. The normal agent worker instead retains its policy object between
successful tasks; the backend preserves its handshake across reset. A new
hardware-free regression test executes two successive picks with different
instructions and verifies one handshake, the same session, fresh action queues,
and no policy close. This is a reuse check, not a measured warm hardware start.

## Next implementation: persistent session worker

Prepare and validate checkpoint/source identity and warm the GPU once when the
session starts. Advertise readiness only after preparation finishes. Reuse that
same worker and policy client for successive tasks; perform bounded health,
fresh-camera and stop-latch checks before each task. Revalidate expensive inputs
when their identity changes. Keep new task queues/instruction epochs separate.

Initially retain controller connection/calibration checks between tasks, then
measure whether their roughly 2.6-second cost warrants a separately reviewed
session-long hardware connection. Preserve signal/clamp guards and disconnect
on failure. A fault must invalidate session readiness and require recovery;
never silently clear the stop latch to accept another task.

This optimization is recorded but not implemented in the one-shot harness.
Validate with two sequential GPU-backed tasks without physical motion first,
then one explicitly prepared multi-task physical session. Measure cold readiness
and warm dispatch-to-first-command separately. Do not repeat a large parity sweep.
