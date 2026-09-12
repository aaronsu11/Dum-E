# Pi0.5 and MolmoAct 2 evaluation

Phase 8.1 adds an isolated, GPU-only LeRobot evaluation server. Switch models with
`--profile pi05-base` or `--profile molmoact2-so101`. The original GR00T server,
controller, MCP/voice path and Phase 8 watchdog settings are unchanged.

This is an evaluation interface, not yet a production `IPolicyBackend` adapter.
It uses LeRobot policies and processors behind bounded JSON HTTP on loopback8081,
not the existing GR00T-specific gRPC wrapper. No robot/camera device is mounted;
all responses advertise `physical_ready: false`. Remote use requires SSH tunnel.

| Profile | Checkpoint revision | Full chunk | Intended check |
| --- | --- | --- | --- |
| pi05-base | lerobot/pi05_base @ b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba | 50 × 32 | Base-model inference smoke; no SO101 action semantics |
| molmoact2-so101 | allenai/MolmoAct2-SO100_101 @ 152569fe57914d97be91055800035f54e250d009 | 30 × 6 | Official SO100/101 variant; calibration compatibility still unverified |

Both use BF16 (policy-specific FP32 components remain), CUDA only, no compilation,
and no forced16-step truncation. MolmoAct2 uses continuous actions,10 solver steps,
CUDA graphs disabled for initial memory measurement, and checkpoint normalization
with `norm_tag=so100_so101_molmoact2`, including gripper normalization. LeRobot's
vendored implementation loads safetensors without executing downloaded model code.
Pi05 weights load strictly: this wrapper deliberately does not use upstream's
catch-and-return-initialized-model fallback on weight-loading errors.

Pi05's base profile maps front/wrist to base/left-wrist images, supplies an explicit
zero right-wrist placeholder, and pads raw6D state to32D. These are smoke-test
inputs only: no invented SO101 normalization, decoded physical-action guarantee,
or task-success claim. Molmo's state/action metadata names match the six joints,
but its calibration frame and training camera placement differ from ours. Neither
profile can be connected to the robot through this server.

## Local run

Run from repository root on Linux. Cache the pinned public checkpoints with
Hugging Face snapshot_download. The pinned PaliGemma tokenizer requires previously
accepted provider access; build uses only tokenizer artifacts, never credentials.
`MODEL_SWAP_TOKENIZER_SOURCE` may point at the pinned snapshot directory.

```bash
scripts/build_model_swap_image.sh
MODEL_SWAP_EVIDENCE="$PWD/corpus/model-swap-molmo-startup" \
  scripts/run_model_swap_server.sh molmoact2-so101
# In another terminal:
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/check_model_swap.py \
  --profile molmoact2-so101 --output corpus/model-swap-molmo-check
```

Stop the container before selecting the other profile. The launcher refuses an
already-occupied compute GPU. No automatic CPU fallback or OOM retry. Server
allows one inference request at a time, rejects concurrent requests, and latches
CUDA OOM until restart. Health remains separate from the inference request.

Validation is two warmups plus12observations (first frozen observation from each
of12recorded episodes), one recorded seed each. It records exact input-lock hash,
profile/revision, full action arrays, finiteness, per-dimension ranges, stage timing,
round-trip latency and allocated/reserved GPU memory. This is neither a GR00T
parity comparison nor a statistically valid latency-tail estimate. A failed run
retains partial evidence and is never overwritten.

## EC2 fallback

Aaron provisions the workstation and provides SSH access. Do not start or allocate
EC2 instances as part of this script. AWS's instance API reports32GiB host RAM and
22,888MiB GPU memory for both `g5.2xlarge` (A10G) and `g6.2xlarge` (L4). Either is a
candidate for BF16 testing; actual fit and latency must still be measured. Use a
NVIDIA driver compatible with the image's Torch2.11.0+cu130 runtime and NVIDIA
Container Toolkit. Prefer at least100GB free disk for image/cache; model snapshots
are approximately22GB Molmo and14.5GB Pi05 on disk. Run one model at a time.

The image contains runtime dependencies, adapters and pinned tokenizer artifacts;
public checkpoint weights download into a persistent Hugging Face cache. No AWS/HF
credentials or frozen robot observations are baked into the image. The Dockerfile
extends the pinned previously verified dependency image; runtime model libraries
are LeRobot0.6.1 / Transformers5.5.4 / Torch2.11.0+cu130.

Once published, pull the exact ECR digest recorded in the handoff evidence:

```bash
aws ecr get-login-password --region us-west-2 | \
  docker login --username AWS --password-stdin 177118830501.dkr.ecr.us-west-2.amazonaws.com
docker pull IMAGE_URI_AT_SHA256
# Use the run_model_swap_server.sh launcher with IMAGE_URI_AT_SHA256 as argument2.
# Or run the same docker command: --gpus all --network host, cache/evidence mounts,
# --profile molmoact2-so101 --evidence-dir /evidence. Bind only127.0.0.1:8081.
```

Allow SSH from the operator's address; do not open8081 publicly. From the local
workstation, forward `ssh -N -L 8081:127.0.0.1:8081 USER@HOST` and run the same
12-observation client against localhost. That sends frozen camera/state samples
to your authenticated workstation; it never sends actuator commands.

Before a physical trial: establish SO101 calibration/action mapping and useful
camera/task compatibility, add a reviewed controller adapter, measure a suitable
execution horizon/latency budget, then prepare one bounded supervised trial. The
new model's raw output must not inherit GR00T's physical approval or tolerances.

## Local results — 2026-09-12

Pi0.5 base: 12/12 finite50×32 outputs; median RPC446.507ms, median model generation 427.117ms. Peak PyTorch allocation9091.139MiB, reserved9420MiB. RTX3060 local inference feasible with the explicitly synthetic base-model input mapping.

MolmoAct2 SO101: weights load at10507.536MiB, but first warmup fails in embedding concatenation: attempted756MiB allocation with599.12MiB free. BF16, no CUDA graphs, no CPU fallback. Zero successful observations; larger-GPU evaluation is pending. This establishes failure for this profile/runtime, not impossibility under every possible optimization. ECR publication and user-provisioned EC2 are the authorized next step.

Retained artifacts: `corpus/phase8.1-model-swap-20260912/`, including failed-run logs, checkpoint revisions, complete Pi outputs, calibration caveats, image IDs and source hashes.
