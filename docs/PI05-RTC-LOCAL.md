# Local Pi0.5 RTC check

**Current closeout:** the model check, guarded HTTP integration and physical
trial14 passed under the bounded phase scope. Aaron confirmed visible smooth
motion and safe hold. The sections below retain the sequence of checks;
earlier “pending” notes are superseded by the physical result and confirmation.
See [PHASE10-CLOSEOUT.md](PHASE10-CLOSEOUT.md) for final review and retained gaps.

Date: 2026-09-13 UTC. Scope: local RTX 3060, recorded inputs, zero motor commands.

**Result: passed.** The pinned SO101 checkpoint fits locally with RTC, and a
background-generated chunk replaced the old chunk during a dummy 20 Hz replay.
This is not a physical arm trial.

## Measured results

Three matched recorded-input comparisons, one seed each; warmups excluded:

| Mode | Mean generation | Mean including preprocessing/postprocessing | Peak allocated VRAM |
|---|---:|---:|---:|
| Ordinary Pi0.5 | 414.22 ms | 417.09 ms | 8.83 GiB |
| RTC Pi0.5 | 422.43 ms | 425.42 ms | 8.86 GiB |

RTC added 8.20 ms (about 2%) to mean generation in this small check.
Normalized prefix RMSE decreased by 96.7%, 97.2% and 97.6% in the three pairs.
This measures agreement with queued targets, not task success or physical motion.

The dummy consumer requested at tick 25 and installed the replacement at tick 34,
discarding the nine steps elapsed during inference. All 50 target ticks completed.
Maximum observed tick interval was 50.77 ms; replacement total inference was
425.83 ms. Peak allocated VRAM during the threaded replay was 8.87 GiB. All
guided calls fit both the estimated 11-tick (550 ms) delay and the 25-tick
(1.25 second) available overlap. These are a few observations, not a tail-latency
guarantee.

Model loading took 61.55 seconds (excluding the preceding manifest check);
first ordinary inference took 764.88 ms including preprocessing/postprocessing.
First guided inference took 432.70 ms. These startup values are excluded from
the warm comparison and reinforce the need to keep the model resident for a
multi-task session.

Evidence: `corpus/phase10-integration-20260913/pi05-rtc-local/run-1/result.json`,
paired arrays in `run-1/pair-{1,2,3}.npz`, and `run-1.log`.
Image: `dume-model-swap:pi05-so101-phase10`, previously recorded image digest
`sha256:6a1ff7e22a1c4202a68f7ccfd85d6559dcd7d51ea08433461956d7f9d68d34fb`.
The test container had network disabled and no arm/camera device mappings.
The previous local GR00T server was stopped to free the GPU; the ephemeral
RTC test container exited after completion. No inference server is left on the
local GPU by this check.

Unused Docker build cache was reclaimed to stage the checkpoint locally.
No model image or existing checkpoint was removed. The downloaded weights
remain under `checkpoints/pi05-so101-project-ira/`.

## Checkpoint and runtime

Use the same Project-IRA SO101 checkpoint `008000` as physical trial 11:
revision `4b48932cc74a61f685841a4fff467ef31caa9ce1`, with all six artifacts
checked against `policy_lab/pi05-so101-manifest.json` before loading.
The installed LeRobot 0.6.1 Pi0.5 implementation exposes `supports_rtc()` and
accepts `prev_chunk_left_over`, `inference_delay` and `execution_horizon`
through `predict_action_chunk`. Source hashes are saved with the test evidence.

## Configuration

- Same saved camera mapping and processors as the validated SO101 bridge:
  front → desk_view; wrist → wrist_left; six absolute joint targets in degrees
  and gripper 0–100. Inputs are the three recorded observations from trial 11.
- CUDA only; BF16 with existing FP32 components; ten inference steps;
  compilation off; one CPU intra-op thread.
- Fifty-action output chunks. RTC receives the remaining 25 normalized
  actions after a synthetic 25-action advance. Degrees must not be used as
  the guidance prefix.
- Linear prefix schedule, maximum guidance weight 10, execution horizon 25.
  Guidance delay is estimated from a guided warmup plus two 50 ms ticks,
  capped at 24; deadline-fit booleans are recorded separately.
- `torch.no_grad()` around inference, allowing the RTC processor's internal
  `torch.enable_grad()` block. The existing server's `torch.inference_mode()`
  wrapper is unsuitable for guided RTC.

## Small test

`scripts/check_pi05_rtc.py` strictly loads the checkpoint, performs one ordinary
and one guided warmup, then compares ordinary and RTC outputs on three recorded
observations with identical seeds. It saves generation and preprocessing-inclusive
latency, peak allocated VRAM, normalized prefix RMSE, and both normalized and
decoded arrays.

A final 50-tick dummy consumer runs at nominal 20 Hz. It requests replacement
inference in a worker while consuming the old chunk, discards elapsed prefix
steps when the replacement arrives, and records the actual handoff and cadence.
There are no motor, camera or controller connections in this script.

This initial check is an RTC model and dummy-consumer check, not a physical
scheduler qualification. The subsequent guarded integration is described below.

## Guarded scheduler integration

The attended runner now accepts `--profile pi05-so101 --scheduler rtc`.
The global controller configuration and production `DUME_ASYNC_INFERENCE`
selection are unchanged. This explicit trial path connects to local port 8081.

The loop projects an entire candidate chunk into the existing 0.25-degree/point
slew, 3.75-degree/point tracking and 5-degree/point excursion bounds before
queueing it. Each tick rechecks the current pose. If a queued target would need
to change, or the controller returns a different target, the trial fails and
holds instead of silently invalidating the RTC prefix. Existing stop-latch,
camera, calibration/configuration and controller clamp checks remain active.

With 25 actions left, one background worker sends their **bounded queued values**
to `/infer/rtc`. The server strictly verifies the pinned checkpoint and inverts
its saved action postprocessor to obtain normalized guidance. A numerical
round-trip check verifies that those normalized values decode to the supplied
queue. Identity metadata binds the instruction epoch, request number, delay and
exact prefix digest. Late, mismatched and exhausted-prefix replies are refused;
instruction changes require a fresh trial. There is no automatic retargeting in
this scoped trial implementation.

Health checks use a separate connection with a 100 ms timeout and do not wait for
the inference lock. The attended test uses a 750 ms inference deadline and
15-step guidance delay, inside the 25-step/1.25-second remaining buffer. It
discards the steps already played when a replacement arrives. Three generated
50-action chunks therefore produce **100 played targets** with overlap, about
five seconds at 20 Hz. This is not 150 sequential targets.

### Verification

35 focused tests passed, including bounded queue preservation, invalid complete
chunks, changed observed pose, changed controller output, expired replies,
old epochs, server errors, health-check independence, operator stops, instruction
changes and absence of late dispatch after failure.

The exact scheduler was also run through the local HTTP server with recorded
camera frames and a dummy controller:

| Final dry run | Result |
|---|---|
| Played dummy targets / motor commands | 100 / 0 |
| Generated chunks | 3 |
| Full client RPC | 357.49 / 360.38 / 358.33 ms |
| Server generation | 338.26 / 341.04 / 338.08 ms |
| Request-to-install time | 379.39 / 402.21 / 404.39 ms |
| Replacement handoffs | after 8 ticks each |
| Median / maximum action interval | 50.20 / 50.26 ms |

Evidence: `corpus/phase10-integration-20260913/pi05-rtc-http-dummy-2/`.
The earlier integration dry run is retained in `pi05-rtc-http-dummy-1/`; it
passed with two nine-tick handoffs before final instrumentation separated RPC
time from installation time. The first run's `chunk_rpc_ms` includes installation
and polling, so it must not be interpreted as pure client RPC.

The evidence check matched each supplied prefix to the dummy commands actually
consumed before its handoff. Bounds passed with a `1e-5` float32 numerical
tolerance; the largest initial-step rounding excess was about `0.000004` degree.
These dummy checks do not establish real-arm smoothness or fault response.

### Physical trial prepared, not executed

`corpus/phase10-integration-20260913/trial-14-pi05-rtc/snapshot.json` binds the
current sources, controller configuration and actual calibration, without opening
hardware. No approval or physical result has been generated. The runner requires
a fresh operator-present authorization and its normal camera/prearm checks.
There is no reset pose and no expansion of motion limits.

The local `dume-pi05-rtc-server` is warm and resident on the RTX 3060, using the
existing image with current repository code mounted read-only. This supersedes
the idle-GPU state after the initial model-only check above. It has no arm or
camera devices. No EC2 configuration was changed.

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
