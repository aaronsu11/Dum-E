# Async server-loss hardware test

Status: prepared and simulated; no physical server-loss test has run.
Aaron requested hardware verification before the audible-failure check.

## Protocol for one trial

Aaron is beside the arm, ready to stop it, with the banana and other obstacles
removed from the reachable workspace. The arm uses the existing calibration,
PID, units, and clamp; no changes to those settings are proposed.

1. Start the existing read-only GPU container `dume-async-physical-20260912`.
   Snapshot the current sources, checkpoint, devices, calibration and service;
   record fresh approval of `async_server_loss_physical_trial` in a new workspace.
   The successful pick approval cannot release this different protocol.
2. Probe both cameras and warm the loaded CUDA policy before constructing the
   controller. Verify the guarded model load and lightweight observer.
3. Move through the normal initial/ready reset. Begin the same banana instruction
   in the cleared workspace, with the ordinary 50 ms action cadence.
4. After the fourth policy action, a separate thread verifies the service identity
   and kills only that container. The trial has a hard budget of 32 policy actions;
   failure to stop on server loss is a failed test.
5. Measure last-command and fault-detection times relative to the kill request.
   The current conservative software bound is 700.098 ms. Any command after the
   hardware stop latch, a clamp, or a missed bound fails the test.
6. Check that a reset request is refused without hardware dispatch. Collect 40
   read-only joint samples at approximately 50 ms intervals after the fault.
   Preserve torque state and issue no park/reset/torque-enable commands. Disconnect
   the serial/camera resources after sampling. No second trial is automatic.
7. Aaron confirms whether the arm held safely, without collapse, unexpected motion,
   or unsafe contact. Report measured joint drift, including settling, separately
   from the software stop time; do not label callback delivery a physical hold pass.

Runner: `scripts/run_async_server_loss_trial.py`. It requires a terminal stop
channel and fresh, named approval bound to the exact snapshot. Ctrl-C/SIGTERM and
existing raw-write guards remain active. The fault is expected, but unexpected
faults still fail the attempt. If a stop occurs, do not manually resume this process.

The runner records the failure callback only. Task-manager publication, audible
notification, spoken retargeting and longer-run voice responsiveness require
separate integrated tests; they are not passed by this hardware test.

## Local validation

`tests/test_async_server_loss_trial.py` exercises the complete runner with a
simulated controller and injected server death. It verifies the single-trial
approval distinction, bounded detection, refused reset, 40 read-only hold samples,
and zero target dispatches after the stop latch. No physical device or model is
used by these tests.

## Successful pick trace review

Input: `corpus/phase8-physical-trial2-20260912/traces/` (320 actions and state samples).
Reproducible analysis: `scripts/analyze_async_trace.py`.
Output: `corpus/phase8-physical-trial2-20260912/analysis/`.

Thirty-nine boundaries are the first action after each newly accepted chunk.
The transition window includes that interval and the next, allowing one sample
of mechanical lag. Velocities are finite differences at actual state-read times.

| Joint | Transition-window absolute velocity p95 | Elsewhere p95 | Unit |
|---|---:|---:|---|
| Shoulder pan | 16.36 | 16.77 | degrees/s |
| Shoulder lift | 35.14 | 40.69 | degrees/s |
| Elbow flex | 27.67 | 31.85 | degrees/s |
| Wrist flex | 24.62 | 25.72 | degrees/s |
| Wrist roll | 10.56 | 12.20 | degrees/s |
| Gripper | 38.40 | 35.77 | percentage points/s |

There is no p95 elevation in this transition window for the five arm joints.
Looking only at the boundary interval gives higher p95 values for shoulder lift
and elbow; requested target changes are also larger at boundaries. Preserve both
views instead of selecting only the favorable window. This one operator-confirmed
smooth trial is descriptive evidence, not a formal no-spike guarantee: there is
no prespecified spike threshold, encoder quantization affects finite differences,
and joints have no independent read timestamps. No aggregation change is justified
solely by these results.
