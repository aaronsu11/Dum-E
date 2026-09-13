# Phase10: physical integration trials

Aaron clarified the objective: test how easily SO101, its input modalities and
the inference/execution pipeline can be configured for different model
backends. Task accuracy and fine-tuning are not the goal. A missed grasp is not
an integration failure.

## Pass criteria

- Configuration selects the intended checkpoint, transport, camera roles and
  named state/action mapping.
- Both required cameras provide actual scene content with the expected RGB
  layout; no silently missing or substituted image.
- The calibration file actually loaded by the controller is recorded and
  validated. Joint units and gripper normalization remain explicit.
- Live observations reach the model and complete, finite action chunks return.
- Bounded commands reach the intended joints, with observed state and timing
  recorded; stop/failure prevents further target dispatch and retains hold.
- Sync/async support is reported as implemented, unsupported or pending.
  An unsupported configuration must not silently fall back to another mode.
- Raw model predictions are separate from safety-limited physical commands.
  Limiting a prediction is visible in the trace, not scored as model accuracy.

## First trial: G0.5 locally, synchronous

One 32-action chunk from the current pose; no automatic reset to a training
pose. Target cadence is 20 Hz. Each command is limited to 0.25° per arm joint
(0.25 gripper points), with total travel limited to 5°/5 points from the starting
pose and to the actual calibrated range. The projection considers both measured
pose and prior command so tracking lag cannot accumulate into a jump.

This is deliberately limited motion for plumbing verification. It does not
demonstrate the model's unmodified trajectories or picking skill. Inference
finishes before execution: this first check makes no async claim.

The existing guarded controller supplies pre-arm-to-present, PID checks, torque
hold on disconnect, and a latched stop at packet dispatch. A secondary per-command
controller clamp is 1°. A clamp warning, invalid camera, divergent state, missing
server or operator stop ends the attempt. The operator remains beside the arm.

```bash
MPLCONFIGDIR=/tmp/dume-matplotlib .venv/bin/python scripts/run_integration_trial.py \
  prepare --workspace corpus/a-new-integration-trial
```

Prepare constructs a disconnected controller and records its resolved calibration,
configuration, protocol and source hashes. It does not open the robot or cameras.
Running requires a separate current operator-present authorization bound to that
snapshot, a TTY stop channel and working camera feeds. No authorization is generated
by the runner itself.

## Backend coverage

| Backend | Sync | Async | Next physical use |
|---|---|---|---|
| GR00T LeRobot | Implemented; prior physical validation | Implemented; prior physical validation | Short regression under the new integration criteria |
| G0.5 native | Implemented | Explicitly unsupported by current adapter | First bounded physical plumbing trial |
| MolmoAct2 | Evaluation server exists; controller adapter pending | Not integrated | Add explicit mapped controller transport before motion |
| Pi0.5 base | Live-input inference can be tested without motion | Not integrated | Define SO101 output mapping before motor execution |

Pi0.5's mapping requirement is distinct from fine-tuning or accuracy. An
anonymous 32-dimensional base output is not automatically a six-joint command.

## Current preflight findings

- Host devices exist: arm `/dev/ttyACM0`, configured front `/dev/video2`, wrist
  `/dev/video0`. The initial black wrist feed was corrected before trial1.
  A transient camera-open failure then cleared in a fresh capture process.
- The current controller loads `so_follower/my_awesome_follower_arm.json`,
  not the older `so101_follower` file audited in Phase9. See the correction in
  `MODEL-MODALITY-COMPATIBILITY.md`. No calibration was changed.
- Trial1 has completed as a partial integration result, described below.
  G0.5 server loading itself does not connect hardware.

Keep trials sequential and bounded. Observe the first trial before choosing
the next backend/mode; do not run a success-rate matrix or fine-tune a model.

## Trial1 result — partial integration

Aaron confirmed the camera correction and presence and authorized one trial.
The prepared configuration, source hashes and actual loaded calibration matched.
The controller pre-armed its goals to the current pose and verified PID settings.

| Measurement | Result |
|---|---:|
| Bounded commands dispatched | 32 |
| Live-observation chunk RPC | 1262.205 ms |
| Server inference | 1187.614 ms |
| Median / maximum command interval | 50.154 / 50.284 ms |
| Controller clamp warnings | 0 |
| Recorded joint excursion, all six channels | 0 |

Aaron's observation: **“No visible movement; held safely.”**

Live camera/state input, inference and bounded command dispatch were exercised.
Actual joint movement was **not demonstrated**. Task accuracy was not scored.
The trace contains 32 varying targets within 0.25°/point of the unchanged measured
pose. The feedback-relative limit prevents a command from building beyond that
small window while feedback remains stationary. Deadband, quantization or small
model actions are possible explanations; a hardware cause has not been proved.

Before another trial, review separate command slew-rate and tracking-error
allowances while retaining calibrated limits, the 5°/point total envelope and
latched stop. Do not silently widen the limit or reuse trial1 authorization.
No second trial was run.

Evidence: `corpus/phase10-integration-20260913/trial-1-candidate/`, including
`result.json`, `live-observation.npz`, `raw-actions.json`, `summary.json`,
`no-motion-diagnostic.json`, `operator-observation.json` and `assessment.json`.
The raw runner's `physical_motion` field records that targets were sent, not
measured movement; the encoder trace and final assessment establish the latter.
