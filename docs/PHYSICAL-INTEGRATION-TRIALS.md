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

### Three-chunk follow-up (2026-09-13 UTC)

At Aaron's request, increased the synchronous G0.5 run to three fresh-observation
chunks, 32 commands each (96 total). Origin and previous command persist across
chunk boundaries; all existing motion limits remain unchanged. Per-chunk inputs,
raw actions and timing are recorded. New results use `motor_targets_sent` to avoid
conflating dispatch with observed motion. Four focused guard tests pass, including
96-command total excursion and stationary-feedback cases.

Evidence: `corpus/phase10-integration-20260913/trial-2-three-chunks/`.
All 96 commands dispatched without errors or clamp warnings; chunk RPC times were
1276.10, 1258.60 and 1254.88 ms. All six measured joint positions remained unchanged,
including final readback. Operator observation is pending. More chunks did not
demonstrate motion; investigate the feedback-relative 0.25-degree/point envelope
before another physical run. No movement-limit increase was made.

### Three-chunk repeat (2026-09-13 UTC)

Aaron requested another identical run. Trial 3 dispatched 96 commands with unchanged
limits and fresh authorization. Evidence: `corpus/phase10-integration-20260913/trial-3-three-chunks/`.
Chunk RPC times (ms): 1263.98, 1259.80, 1251.09.
Maximum measured displacement per joint: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
No errors or clamp warnings. Operator observation remains pending.

### Shoulder-pan diagnostic (2026-09-13 UTC)

Aaron confirmed trial 3 held safely with no visible movement and requested visible
motion. Trial 4 used an explicitly synthetic +3-degree shoulder-pan target for 40
commands. The diagnostic alone allows 0.75-degree measured-pose lead, retaining
0.25-degree command slew and the 5-degree global envelope; normal model trials
remain unchanged. Five guard tests pass. Other joints retained starting targets.

Evidence: `corpus/phase10-integration-20260913/trial-4-pan-diagnostic/`.
The run completed without errors/clamp warnings, but all in-trial and final encoder
readbacks were unchanged. Its final pan command was only +0.75 degrees because
feedback did not follow. This does not establish visible or model-driven movement;
operator observation remains pending.

Subsequent read-only motor inspection found torque enabled, position mode, zero
status faults and goal register 2040 versus measured pan 2031 ticks. Configured
CW/CCW dead zones were 1 tick. Commands reached the motor register; no motor
settings were changed during inspection. Small motor response remains unresolved;
do not attribute it solely to the configured dead zone. Later readback is stored
separately from immediate trial observations. No second diagnostic was run.

### Fivefold diagnostic tracking allowance (2026-09-13 UTC)

Aaron requested a 5x larger limiter. Diagnostic tracking allowance increased from
0.75 to 3.75 degrees, with a 4-degree controller backstop. Command slew remains
0.25 degrees, the synthetic pan target remains +3 degrees, and the global
envelope remains 5 degrees. Normal model trial limits are unchanged. Six focused
guard tests pass. One 40-command diagnostic completed without errors or clamp
warnings. Evidence: `corpus/phase10-integration-20260913/trial-5-pan-relaxed/`.
Maximum measured displacement per joint: [2.637362637362637, 0.0, 0.0, 0.0, 0.0879120879120876, 0.0].
Final displacement: [2.637362637362637, 0.0, 0.0, 0.0, 0.0, 0.0].
Operator observation remains pending; this is synthetic controller motion, not
model-generated motion. No further physical trial was run.

### Doubled pan diagnostic target (2026-09-13 UTC)

Aaron reported trial 5 motion was not visible and requested 2x more. Trial 6
doubled the synthetic shoulder-pan target from +3 to +6 degrees, with a 6-degree
total excursion cap. Tracking allowance remains 3.75 degrees, command slew
0.25 degrees and controller backstop 4 degrees. Normal model trials are unchanged.
Seven focused guard tests pass. One 40-command diagnostic completed without
errors or clamp warnings. Evidence: `corpus/phase10-integration-20260913/trial-6-pan-six-degrees/`.
Maximum measured displacement per joint: [5.714285714285714, 0.0, 0.0, 0.0, 0.2637362637362628, 0.0].
Final displacement: [5.626373626373627, 0.0, 0.0, 0.0, 0.0879120879120876, 0.0].
Operator observation remains pending. This is synthetic controller motion; no
model-motion result is claimed. No further physical run was started.

### G0.5 policy after visible diagnostic (2026-09-13 UTC)

Aaron confirmed trial 6 was visible and smooth (yes to the safe-hold question),
then requested policy inference. Trial 7 used three genuine G0.5 chunks (96 commands),
fresh live cameras/state for each chunk, no target amplification, 3.75-degree
tracking allowance, 0.25-degree command slew, 5-degree total cap and 4-degree
controller backstop. Seven focused guard tests pass.
Evidence: `corpus/phase10-integration-20260913/trial-7-g05-policy-relaxed/`.
Chunk RPC times (ms): 1296.90, 1286.95, 1278.51.
Maximum measured displacement per joint: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
Maximum raw target error per joint: [0.4042061144834036, 0.2673249297089626, 0.3693067948896811, 0.5105067661830418, 0.4309288276420915, 0.4670298065154128].
All 96 commands dispatched without errors/clamp warnings. Operator observation
remains pending; do not conflate successful dispatch with visible movement.

### Ten-chunk G0.5 policy trial (2026-09-13 UTC)

Aaron requested 10 policy chunks after reporting previous motion too small to
notice. Trial 8 used 10 genuine chunks / 320 commands with fresh observations and
unchanged 3.75-degree tracking allowance, 0.25-degree slew, 5-degree global cap
and 4-degree controller backstop. Seven focused guard tests pass.
Evidence: `corpus/phase10-integration-20260913/trial-8-g05-ten-chunks/`.
Mean chunk RPC time: 1289.2005650999636 ms.
Maximum measured displacement per joint: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0].
Maximum raw target error per joint: [0.4345809692864889, 0.2451844267792751, 0.3838636796553061, 0.5242396763392918, 0.46172869336474776, 0.47040676726980735].
All 320 commands dispatched without errors/clamp warnings. Operator observation
remains pending. No additional physical run was started.

### G0.5 handoff and next policy

Aaron accepted recording the small-target result and moving to the next policy.
Across 10 chunks, raw targets stayed near the observed pose: pan +0.128..+0.435°,
lift -0.007..+0.245°, elbow -0.384..+0.005°, wrist flex +0.338..+0.524°, wrist
roll -0.462..-0.191°, and gripper -0.098..+0.470 points. These are absolute targets,
not accumulating increments. 310/320 commands passed unchanged; only each
chunk's first command was slew-limited. No in-trial encoder movement occurred.
This establishes live-input inference and bounded dispatch, not visible policy
motion or task accuracy. Separate synthetic diagnostics established smooth
visible controller motion. Do not combine those into a policy-motion pass.

Starting horizon reset to three chunks. Next: MolmoAct2 SO100/101, 30 actions
per chunk (90 commands), served by the existing EC2 L4 because its prior local
RTX3060 forward pass OOMed. The experimental synchronous controller bridge
uses the pinned SO101 frame conversion and requires both live front/wrist feeds.
Full production factory/async integration is not claimed.

### Molmo experimental trial connection

`policy/molmo_backend.py` supplies a synchronous `IPolicyBackend` bridge for the
attended runner. It verifies the complete pinned profile in health and inference
responses, checks the 30x6 action shape/finiteness and seed, preserves front/wrist
RGB roles, maps centered arm degrees into the legacy SO101 frame, and maps the
absolute output back. It rejects async mode. It does not mark the evaluation
server as physically certified and is not yet exposed through the production
backend factory. Calibration, projection and stop/hold remain runner-owned.

Prepare without opening hardware:

```bash
MPLCONFIGDIR=/tmp/dume-matplotlib .venv/bin/python scripts/run_integration_trial.py \
  prepare --profile molmoact2-so101 --workspace corpus/a-new-molmo-trial
```

The local bridge uses the SSM loopback port 18081. Each action retains the current
server-health check, so cloud round trips may reduce achieved command cadence;
measure recorded command intervals rather than claiming a sustained 20 Hz.
The shared origin and previous-command state persist across all three chunks.
45 scoped software regressions pass (hardware/GPU-free).

### First Molmo physical trial (2026-09-13 UTC)

Trial 9 completed all 3 chunks / 90 commands using actual front/wrist images and
state, with the explicit mapped EC2 bridge. No software errors or controller
clamp warnings occurred; deliberate projection is recorded separately.
Evidence: `corpus/phase10-integration-20260913/trial-9-molmo-three-chunks/`.
Chunk RPC times (ms): 3728.20, 3765.67, 3749.17.
GPU generation times (ms): 697.22, 698.54, 692.78.
Maximum measured displacement per joint: [1.406593406593407, 0.0, 0.439560439560438, 2.285714285714292, 4.747252747252745, 0.06863417982155084].
Within-chunk command interval median/max (ms): 50.135141998907784 / 55.680241999652935.
Operator observation is pending. No task accuracy or async support is claimed.
The Molmo server `dume-molmo-phase10` remains loaded on the existing EC2 L4,
with SSM local port 18081 open. Local `dume-g05` is stopped. EC2 root storage
remains 150 GB, about 13 GB free after image staging; user workloads preserved.

### Molmo three-chunk repeat (2026-09-13 UTC)

Aaron requested an identical repeat. Trial 10 completed 90 commands with fresh
observations and unchanged limits relative to its starting pose, without software
errors or controller clamp warnings.
Evidence: `corpus/phase10-integration-20260913/trial-10-molmo-three-chunks/`.
Chunk RPC times (ms): 3828.80, 3763.72, 3726.26.
GPU generation times (ms): 706.29, 706.16, 685.16.
Maximum measured displacement per joint: [1.4945054945054945, 0.7032967032967008, 0.0879120879120876, 1.9340659340659414, 5.098901098901095, 0.0].
Within-chunk command interval median/max (ms): 50.13908899854869 / 59.66345600245404.
Operator observation remains pending for this repeat; no follow-on run started.

### Molmo repeat operator observation

Aaron confirmed: “Yes there was small but visible motion.” This confirms visible
model-driven movement for trial 10, alongside live front/wrist/state inference
and bounded mapped dispatch. Smoothness and safe hold were not separately stated
in the reply; do not record those as explicitly confirmed. Task accuracy is unscored
and async remains unsupported by this experimental bridge. No new run started.

### Pi0.5 base live-input check (2026-09-13 UTC)

Aaron requested the next policy. The local RTX3060 Pi0.5 server produced three
finite 50x32 chunks from fresh front/wrist images and raw joint-state readings.
Actual bus calibration matched the current loaded calibration before each capture.
Motor register writes were disabled in the checker; no output was dispatched.
The six live joints are padded to 32, with front/base_0 and wrist/left_wrist_0
images and an explicit zero right-wrist placeholder. No SO101 normalization or
semantic action mapping is verified: this is a live-input pipeline check, not a
physical policy or task-accuracy pass.
Evidence: `corpus/phase10-integration-20260913/pi05-live-three-chunks/`.
RPC times (ms): 1033.84, 591.93, 578.19.
GPU generation times (ms): 992.77, 568.07, 561.97.
Peak PyTorch allocation: 9091.13916015625 MiB.
The Pi0.5 container `dume-pi05-phase10` remains loaded locally. The now-idle
Molmo EC2 container is stopped and its SSM forwarding session closed; EC2 itself
remains running under Aaron’s control. GR00T regression remains the next
physical-capable model check.

### Mapped Pi0.5 SO101 profile ready for physical trial

At Aaron's request, researched and applied the author-recommended Project-IRA
SO101 checkpoint 008000, pinned at revision 4b48932cc74a61f685841a4fff467ef31caa9ce1.
This is a separate `pi05-so101` profile; the base-model output remains unmapped.
Reference recording code and its exact locked LeRobot 0.5.1 wheel establish five
joints in centered degrees plus a 0–100 gripper. Saved six-joint MEAN_STD processor
files handle normalization; no degree-to-percent conversion or Molmo/G0.5 axis
transform is applied. Front maps to desk_view and wrist to wrist_left, with no
third-camera placeholder. Configured action names and 50x6 outputs are verified.

See `docs/PI05-SO101-CONFIGURATION.md` for primary sources, exact artifacts,
image digest and selection instructions. The image is published to ECR and running
on the existing L4; the 16.57 GB weight file resides on EFS because local storage
was insufficient. No disk expansion, recalibration or fine-tuning was performed.
All runtime artifact hashes are verified before strict model loading.

54 scoped software tests pass. Three fresh live-input chunks returned finite
50x6 named targets, with no motor dispatch. GPU generation times were
1312.91 / 326.97 / 329.89 ms; end-to-end RPC times were
4497.51 / 3508.35 / 3494.69 ms. Evidence:
`corpus/phase10-integration-20260913/pi05-so101-live-mapped-check/`.
The prepared physical candidate is `trial-11-pi05-so101-candidate`: three chunks
/150 commands, 0.25-degree slew, 3.75-degree tracking allowance and a shared
5-degree total cap. It has no motor-run authorization record and has not run.
Async is explicitly rejected by this bridge. Task accuracy remains unscored.

### Pi0.5 SO101 first physical trial (2026-09-13 UTC)

Aaron explicitly confirmed presence/workspace readiness and authorized the
prepared first physical run. Trial 11 completed three chunks / 150 mapped
commands with actual front/wrist images and state; no software errors or
controller clamp warnings occurred. Deliberate target projection is recorded.
Evidence: `corpus/phase10-integration-20260913/trial-11-pi05-so101-candidate/`.
Chunk RPC times (ms): 3350.61, 3389.46, 3387.36.
GPU generation times (ms): 326.65, 324.50, 329.71.
Maximum measured displacement per joint: [4.571428571428571, 4.219780219780219, 2.197802197802204, 3.1648351648351536, 1.5824175824175768, 4.735758407687028].
Within-chunk command interval median/max (ms): 50.071626999852015 / 100.36749099890585.
Operator movement/smoothness/safe-hold observation is pending. No further
physical run started. No task-accuracy or async claim is made.

### Pi0.5 physical-trial operator observation

Aaron confirmed: “Yes it was visible.” Visible policy-driven movement is confirmed
for trial 11, alongside live camera/state inference and mapped bounded dispatch.
Smoothness and safe hold were not separately stated, so those are not recorded
as explicitly confirmed. Task accuracy is unscored. No additional run started.

### GR00T calibrated integration trial (2026-09-13 UTC)

Aaron authorized the remaining GR00T check. Trial 12 completed 3 chunks /
48 commands on the local RTX3060 using the validated BF16 LeRobot loader and
pinned HTTP evaluation transport, with observer off. The experimental bridge
explicitly converts controller degrees to/from checkpoint RANGE_M100_100 using
the actual current calibration; gripper remains 0–100. This is a calibrated
integration check, not a like-for-like replay of the historically mismatched
production degree convention. Production native/LeRobot backends were unchanged.

Evidence: `corpus/phase10-integration-20260913/trial-12-groot-three-chunks/`.
Chunk RPC times (ms): 211.04, 214.82, 253.00.
GPU generation times (ms): 170.68, 160.51, 159.72.
Maximum measured displacement per joint: [4.747252747252746, 4.747252747252745, 0.2637362637362628, 4.043956043956044, 4.571428571428569, 4.529855868222374].
Within-chunk command interval median/max (ms): 50.07145200215746 / 50.20982099813409.
All 48 commands dispatched without software errors or controller clamp warnings.
Deliberate projection is recorded separately. Operator observation remains
pending; no accuracy or HTTP-async claim is made. No further physical run started.

### GR00T three-chunk repeat (2026-09-13 UTC)

Aaron requested an identical repeat. Trial13 completed 3 chunks /48 commands
with fresh observations and unchanged limits relative to its starting pose.
No software errors or controller clamp warnings occurred.
Evidence: `corpus/phase10-integration-20260913/trial-13-groot-three-chunks/`.
Chunk RPC times (ms): 247.86, 211.52, 241.71.
GPU generation times (ms): 169.24, 156.11, 156.50.
Maximum measured displacement per joint: [4.571428571428571, 4.483516483516482, 0.3516483516483646, 4.219780219780219, 4.659340659340657, 4.804392587508582].
Within-chunk command interval median/max (ms): 50.14373200174305 / 50.24112099999911.
Operator observation remains pending. No further physical run started.

### Campaign summary and GR00T observation

Aaron confirmed GR00T trial13 movement was visible. Smoothness/safe hold were
not separately stated. The consolidated report is `docs/PHASE10-INFERENCE-TRIAL-SUMMARY.md`;
per-run CSV and full JSON are in `corpus/phase10-integration-20260913/summary/`.
It covers all 13 numbered runs (10 policy runs plus 3 synthetic diagnostics),
1,186 total commands, exact configurations, weight-file sizes, measured GPU
allocation, latency/cadence, operator observations and separately labeled earlier
GPU/network/async evidence. No physical run was started while preparing it.

### Physical Pi0.5 RTC trial14 — operator observation pending

Aaron confirmed current presence and authorized one trial. The prepared run
completed all 100 targets from three overlapping chunks without errors or
controller clamp warnings. Client RPCs: 442.88 / 426.69 / 429.64 ms; server
generation: 423.65 / 414.18 / 414.17 ms. Replacements installed after nine ticks
each, at actions 34 and 59. Median/max action intervals: 50.154 / 50.230 ms.
Maximum measured arm excursion was 4.835 degrees; gripper excursion 4.736 points.
No reset or limit expansion. Cleanup retained torque hold; actual smoothness
and safe hold await Aaron's observation. No further run authorized.
Evidence: `corpus/phase10-integration-20260913/trial-14-pi05-rtc/`.

Aaron subsequently answered "Yes. The movement was visible and smooth." to
the combined motion/safe-hold question. Trial14 now has operator-confirmed
visible smooth motion and safe hold. This passes the bounded RTC integration
trial; it is not a task-accuracy or comprehensive fault qualification. The
original runner result is retained, with the observation stored separately in
`trial-14-pi05-rtc/operator-observation.json`. No further motion was run.

## Phase10 closed — 2026-09-13

Completed under Aaron’s bounded-integration/Pi0.5-only async scope. All14runs
and1286targets are summarized.69focused regression tests pass; the deadline
installation guard was tightened after the accepted physical trial and tested
in software. No further hardware run. UAT, three plan summaries, review and
verification complete. Original GX-05 and wider operating conditions remain
explicit deferrals. See `docs/PHASE10-CLOSEOUT.md`. Historical pending notes
above are superseded by this closeout. Milestone audit is not performed here.
