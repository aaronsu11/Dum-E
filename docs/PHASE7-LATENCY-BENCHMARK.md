# Phase 7: successful trial 1 and latency investigation

2026-09-11. Aaron confirmed the trial was successful. The one-trial runner
recorded coherent banana targeting, completed grasp, no erratic/wrong-target
motion, no safety stop, and zero clamp warnings; 20 chunks executed in 30 seconds.
The controller disconnected afterward. `status: partial` and CLI exit 1 describe
the unfinished three-trial milestone, not a failed physical trial. No trial 2
was authorized or run. Physical evidence:
`corpus/phase7-trial1-retry2-20260911/live-run.json`.

## Bounded benchmark

One frozen observation (`record_0005.npz`), seed 20265907, RTX 3060 12 GiB,
BF16 model parameters, four flow steps, 16×6 decoded output. Each configuration
used one warmup and three measured calls. Containers ran sequentially, without
network or robot devices; model inference remained on CUDA. Load time, transport,
physical action pacing, and live attestation are excluded from local pipeline
times. CUDA synchronization bounded timed regions. Native uses the pinned
operational Gr00tPolicy path; LeRobot uses the existing full-chunk server path.

| Measurement | Warm times (ms) | Median (ms) |
|---|---|---|
| Native pipeline | 124.37, 123.90, 123.26 | 123.90 |
| LeRobot pipeline, initial pass, no runtime observer | 204.57, 226.81, 226.91 | 226.81 |
| LeRobot pipeline with operation/model observer | 226.58, 298.50, 383.13 | 298.50 |
| LeRobot default 20 CPU threads, paired thread test | 307.36, 231.85, 308.70 | 307.36 |
| LeRobot 1 CPU thread, same loaded GPU model | 132.84, 133.45, 132.35 | 132.84 |

The CPU-thread comparison retained exactly equal full decoded outputs against
the default-thread reference for this observation. No cross-backend equality
claim is made. Native remains FlashAttention2 / PyTorch2.7.1+CUDA12.8;
LeRobot remains SDPA / PyTorch2.11.0+CUDA13.0. Changing CPU threads did not change
model device, parameter precision, attention selection, or checkpoint weights.

## Where the difference comes from

A synchronized stage pass measured native model execution at118.53–119.60ms
and LeRobot at120.99–121.04ms. The substantial gap is outside the model call.
LeRobot's image conversion/preprocessing varied sharply, while decoding stayed
around0.17–0.19ms. In the paired CPU-thread experiment:

- Default20threads: image conversion56.93–63.37ms, preprocessor44.91–127.19ms,
  model120.96–121.20ms.
- One thread: image conversion1.35–1.49ms, preprocessor8.28–8.99ms,
  model120.54–120.83ms.

These measurements support CPU thread overhead in image preparation as a major
cause on this machine. They do not support blaming GPU capacity or attention
implementation for the observed large delay. Native's remaining ~9ms advantage
in the one-thread comparison is small relative to the removed overhead.

Runtime observation is an additional cost. The initial observer pass measured
higher and more variable pipeline times, but only three samples and fixed mode
order do not isolate a stable observer tax. The successful physical trial
separately averaged242.30ms generation and52.96ms validation; its1.50s control
iteration also includes16×50ms action playback and client checks. Those are
different timing boundaries and must not be compared as if all were model time.

## Limits and next implementation step

The investigation used32predictions total across the successive diagnostic
passes (including warmups), all on one observation. It is a latency diagnostic,
not a task-quality evaluation, exhaustive numerical gate, or statistical p95.
Timing varies with host scheduling. Stage instrumentation adds synchronization;
the within-process default-versus-one-thread comparison applies it equally.

The next implementation step is a documented serving CPU-thread limit (one is a
measured candidate), followed by a small check of the complete guarded RPC path.
The one-thread result has not been deployed to the physical serving service.
Do not remove live safety/identity checks on the basis of this benchmark. No
production inference source, attention setting, or acceptance tolerance was
changed during this investigation. The completed-trial server was stopped to
free GPU memory; benchmark containers exited and no robot controller is active.

Raw samples, pinned image commands, stdout logs, and reproducible worker/launcher
scripts are in `corpus/phase7-latency-benchmark-20260911/`: `native.json`,
`lerobot.json`, `native-stages.json`, `lerobot-stages.json`,
`lerobot-breakdown.json`, and `lerobot-threads.json`.
