# Async inference checkpoint

The Phase 8 implementation is opt-in. Software and GPU simulation checks pass;
physical stop/hold, joint continuity, and integrated voice verification remain pending.

## Runtime behavior

One inference worker requests a new 16-action chunk when eight actions remain.
The controller sends actions every 50 ms. Overlapping predictions use a weighted
average (0.3 old, 0.7 new); expired contributions are discarded at dispatch.
Retargeting clears queued actions and rejects results from the previous instruction.
The backend serializes each observation/action exchange, and the server serializes
preprocessing through decoding so relative actions retain the correct state anchor.

A missed request deadline, stale action, inference error, or controller clamp latches
a fault, closes the policy connection, and reports failure to the task and voice
pipelines. Further movement and automatic reset are refused. This stops issuing
commands; actual mechanical stopping/holding must still be measured on the arm.
The controller owns the serial bus. Robot and inference calls run off the event loop.

## Measured evidence

`corpus/phase8-latency-20260912/latency.json` records one warmup and 100 warm
GPU RPCs using one frozen observation and the lightweight observer:

- Median request latency: 143.818 ms.
- Empirical p99 (higher order statistic): 150.098 ms.
- Request deadline: p99 + 100 ms = 250.098 ms.
- Eight-action overlap: 400 ms.

This is a small operating-point measurement, not a statistical tail guarantee.
The settings loader checks the artifact, sample count, and bound source hashes.
Re-measure after changes to those sources or the serving configuration.

`corpus/phase8-async-serving-retry-20260912/async-serving-check.json` records
real GPU/gRPC inference with a simulated controller. Retargeting kept the loaded
model instance. Deliberately killing the server produced a fault in 299.872 ms;
the last simulated command was 250.150 ms after the kill request. Reset was refused
after the fault. The configured conservative software bound is 700.098 ms, assuming
timely controller I/O and scheduling. No arm was connected to this test.

The initial simulation exposed expiry of an old blending contribution. Its failed
artifact is retained in the original latency workspace; the implementation now
discards expired contributions before dispatch. The final targeted suite passed
128 tests, including concurrency, staleness, retargeting, failure publication,
observer identity, MCP control, and existing voice/backend interfaces.

## Configuration

After preparing the supervised physical check, set these controller keys:

```yaml
controller:
  policy_backend: lerobot
  async_inference: true
  async_latency_path: corpus/phase8-latency-20260912/latency.json
  async_trace_directory: corpus/phase8-physical-traces
```

Existing exported environment variables take precedence. The corresponding names
are `DUME_POLICY_BACKEND`, `DUME_ASYNC_INFERENCE`, `DUME_ASYNC_LATENCY_PATH`,
and `DUME_ASYNC_TRACE_DIRECTORY`. Async requires CUDA and a loopback server.
Per-action state readback is optional and intended for physical verification;
it adds controller I/O. Async remains disabled in the example configuration.

The transport uses pickle and is trusted-local only. Bind Docker publications to
127.0.0.1; do not expose this server to untrusted clients. Revisit authentication
and safe serialization before remote access or multiple independent clients.

## Shared observer images

`scripts/build_observed_policy_images.sh` builds from the existing pinned local
images without changing model dependencies. Both launchers default to lightweight
observation; off and exhaustive modes remain available. See
[Shared chunk observer](SHARED-CHUNK-OBSERVER.md).

Built images:

- `dume-native-lightweight:20260912`: `sha256:78cab5833cb5d41ec08f689e5de2a0143244416fda2d28630ffe334be27bbd59`
- `dume-lerobot-lightweight:20260912`: `sha256:b7074a5c29c12d90696d51c26ee3d9a3c699aeedde8f65176edb503f4fa126aa`

## Remaining operator checks

Run one test at a time with Aaron beside the arm and able to stop it. Begin with
one bounded async pick using command/state traces; check motion and chunk-boundary
velocity before introducing failure. Then prepare a separate server-loss trial
in a collision-free pose and measure last-command time, physical hold, task failure,
and audible notification. Finally verify spoken retargeting and measure voice
round trips during the required longer integrated run against a normal baseline.
Each trial needs a concrete runner/protocol and current operator readiness before
motion. Do not infer a physical pass from the simulated controller or prior sync trials.
