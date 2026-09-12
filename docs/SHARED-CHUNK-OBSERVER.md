# Shared native/LeRobot chunk observer

The optional serving adapters use the same `policy_guard.chunk_observer.ChunkObserver`
with two explicit modes: `off` and `lightweight`. Existing entrypoints and Phase 7
attestation are unchanged. These adapters are opt-in and have not been deployed
to a physical trial.

| Matched GPU pipeline | Observer off | Lightweight |
|---|---:|---:|
| Native | 123.48 ms | 123.49 ms |
| LeRobot | 131.92 ms | 131.83 ms |

Those are warm medians from one observation and seed, one warmup plus three
timed calls per mode, interleaved inside each loaded backend. Both use one CPU
intra-op thread and the same synchronized timing boundaries. Model inference
runs on RTX 3060 GPU with the unchanged backend-specific BF16 configuration.
The lightweight overhead is not distinguishable from timing variation in this
small check. Decoded outputs stayed exactly identical within each backend.
No cross-backend bitwise equality or p95 claim is made.

The final check used 16 predictions. The earlier 16-prediction exploratory pass
had incomplete metadata, and a subsequent native attempt failed an overly narrow
keyword-input metadata assertion after two predictions. These attempts are
preserved in their own workspaces. The final observer handles keyword inputs and
native BatchFeature containers; real metadata and four flow steps are asserted
for every lightweight prediction. Total diagnostic predictions: 34, all GPU-only.

Final recipe, logs, exact pinned container commands, source hashes and samples:
`corpus/latency-shared-observer-final-20260912/`. Earlier attempts:
`corpus/latency-shared-observer-20260912/` and
`corpus/latency-shared-observer-fixed-20260912/`. Copy recipes to a new output
workspace before repeating them; preserve the existing results.

## Coverage

Lightweight mode records generation time, backbone tensor shapes/dtypes/devices,
and action-encoder invocation count (effective flow steps). Three module hooks
are installed for the request and always removed, including on exceptions.
Metadata capture neither copies tensors nor moves them off GPU, and does not
change RNG. Timing starts after hook installation and finishes before hook
removal; both adapters use the same boundaries. Timing excludes RPC, model
loading, attestation completion and physical action playback.

Unlike the exhaustive observer, lightweight mode does NOT inspect every operation,
observe attention kernels, prove intermediate precision/autocast/TF32 state,
capture raw model outputs, or count/inspect sampling noise. Records explicitly
say `exhaustive_attestation: false` and identify coverage. Off mode retains timing
but installs no hooks and makes no observation claims. This is telemetry, not a
substitute for actuator checks or a source-bound parity release.

## Serving adapters

Run these inside the corresponding pinned GPU image with this repository's
`scripts`, `policy_guard` and (for LeRobot) `docker/lerobot-policy` mounted under
a common repository root. Checkpoint/cache mounts are as in the existing serving
recipes. Both need the appropriate backend dependencies in their container.

Native example, assuming the repository mount is `/app`:

```sh
python /app/scripts/serve_observed_native.py \
  --model-path /checkpoints/model --host 0.0.0.0 --port 5555 \
  --observer-mode lightweight --cpu-threads 1
```

Use `--observer-mode off` for the matched observer-off configuration. The native
wrapper preserves the upstream ZMQ endpoint contract, including options/reset,
and uses the pinned native policy with strict validation and GPU execution.

LeRobot example in its own pinned image:

```sh
DUME_CHUNK_OBSERVER=lightweight python3 /app/scripts/serve_observed_lerobot.py \
  --checkpoint-path /checkpoints/model --host 0.0.0.0 --port 8080 --cpu-threads 1
```

Use `DUME_CHUNK_OBSERVER=off` for timing-only. This delegates to the existing
entrypoint's preflight and SAFE-01 checks; it changes only the prediction wrapper.
Publish either container port on host loopback. Both services log the selected
mode and generation time for each completed or failed prediction. Neither adapter
constructs a robot controller.

The LeRobot adapter refuses `DUME_PARITY_ATTESTATION_PATH` and refuses an attached
exhaustive attestor. It cannot silently replace the evidence expected by the
existing Phase 7 physical runner. A new physical observer-off trial needs a
separately reviewed trial release under the requested reduced-observation scope;
old approvals/attestations must not be reused as proof for this configuration.
No physical trial has occurred in this work; current operator presence is pending.

## Validation

Eleven focused tests passed: tensor identity/RNG preservation, real four-step
hook behavior, error cleanup, keyword inputs/custom containers, both LeRobot
modes and native endpoint request/response routing, and rejection of an
attestation downgrade. GPU validation exercised both real inference pipelines
with the shared observer. The new network serving adapters have behavioral
fixture coverage; a real socket-level serving check remains separate from the
local GPU pipeline measurements. No production source or historical evidence
was changed, so the completed Phase 7 verification remains valid.
