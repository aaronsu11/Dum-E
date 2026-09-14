# Reproduce policy serving

The client uses Python 3.12 and locked LeRobot 0.6.1 (`uv sync --locked`). Model dependencies live in separate GPU containers. Run one policy per GPU; the launchers reject an occupied compute GPU. Servers never receive serial/camera devices. Use loopback or an SSH/SSM tunnel; the GR00T pickle transports must not be exposed to untrusted networks.

## Supported routes

| Model | Server / transport | Selection | Output | Execution support |
|---|---|---|---|---|
| GR00T N1.7 SO101 | Isaac-GR00T / ZMQ :5555 | `groot-native` (default) | 16×6 | Existing synchronous agent |
| Same GR00T | LeRobot / gRPC :8080 | `lerobot` | 16×6 | Synchronous agent; opt-in GR00T async |
| Same GR00T | LeRobot / HTTP :8081 | `groot-so101` trial profile | 16×6 | Calibrated bounded synchronous runner |
| G0.5 SO101 | Galaxea native / WebSocket :8765 | `galaxea`; trial `g05-so101` | 32×6 | Synchronous adapter and bounded runner |
| Pi0.5 SO101 | LeRobot / HTTP :8081 | `pi05-so101` | 50×6 | Synchronous adapter; bounded RTC runner |
| MolmoAct 2 SO100/101 | LeRobot / HTTP :8081 | `molmoact2-so101` trial profile | 30×6 | Bounded synchronous runner; not in agent factory |
| Pi0.5 base | LeRobot / HTTP :8081 | `pi05-base` benchmark only | 50×32 | No physical action mapping |

Factory selection is `DUME_POLICY_BACKEND`; `controller.policy_backend` supplies the launcher default. Explicit environment values take precedence. The ordinary SO101 pick skill still executes a default 16-action prefix and uses its existing reset/ready pose sequence. Factory selection alone does not qualify another model's starting pose or make its full native horizon execute. Use the bounded runner for the additional models until task-level behavior is qualified.

## Inference configuration

The measured LeRobot GR00T route uses BF16 storage/compute, SDPA and the checkpoint's four flow steps; its native GR00T comparison used the native attention stack. Pi0.5 preserves the required FP32 components alongside BF16, with compilation disabled and ten inference steps in the pinned measured checkpoints. MolmoAct 2 uses BF16, continuous actions, ten steps, `norm_tag=so100_so101_molmoact2` and CUDA graphs disabled. G0.5 uses SDPA, compilation disabled and its native ten-step flow stage; a 300-token generation budget bounds autoregressive work.

The model runtime and profile metadata are authoritative when changing a checkpoint. Record effective parameter dtypes, generation settings and observer mode with new results; do not infer them from weight-file size. Pi0.5 RTC's additional guidance settings are documented separately in the async guide.

## Checkpoints and pins

`policy/checkpoints.py` is authoritative for HTTP profiles. `policy/backends/lerobot/models/pi05_so101_manifest.json` and `policy/backends/galaxea/checkpoint_manifest.json` additionally verify exact runtime artifacts.

| Model | Repository | Revision / identity |
|---|---|---|
| GR00T | `aaronsu11/GR00T-N1.7-3B-SO101-FruitPicking` | HF revision `26b179c37f35168359ccdf2e51fed6c2690186cf`; content inventory SHA256 `9bd09a2a40c04637b5d6e010790ac2e3b6cbd75f81c1d59443856e4e37f27c09` (not an HF revision) |
| Pi0.5 base | `lerobot/pi05_base` | `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` |
| Pi0.5 SO101 | `Project-IRA/TPSoSe2026_Pi05_LeRobot_SO101_Finetuning_V7_Full_V2` | `4b48932cc74a61f685841a4fff467ef31caa9ce1` |
| MolmoAct 2 | `allenai/MolmoAct2-SO100_101` | `152569fe57914d97be91055800035f54e250d009` |
| G0.5 | `OpenGalaxea/G05` | `e312be81e90c56a55bcb26b57429bd39a335b449` |

The GR00T revision above was recovered from the local HF download metadata for every weight shard and sidecar. Download that revision, then retain the inventory check; a revision alone is not an action-space compatibility check.

For Pi0.5 SO101 download only `outputs_V8/train/pi05_6gpu_fsdp_V2/checkpoints/008000/pretrained_model/*`; point `MODEL_SWAP_PI05_SO101_CHECKPOINT` at that directory. For G0.5 download only paths in its manifest, including the action tokenizer and base processor. Use `huggingface_hub.snapshot_download(repo, revision=..., allow_patterns=...)` and retain the returned snapshot; do not download every training checkpoint. Check model licenses/access requirements before redistribution.

GR00T's HTTP runtime hashes the complete checkpoint inventory before loading. Its gRPC guard checks structure/processor compatibility, not complete weight provenance. Verify the inventory separately when reproducing the exact benchmark checkpoint. The base and Molmo profiles download their pinned snapshots automatically into `MODEL_SWAP_CACHE`.

## Build

```bash
uv sync --locked
# HF_TOKEN must have access to Cosmos and PaliGemma. It is passed as a BuildKit secret.
bash scripts/build_lerobot_policy_image.sh
bash scripts/build_model_swap_image.sh dume-model-swap:local
# Galaxea has a separate Python/CUDA stack, pinned in its Dockerfile.
docker build -f docker/galaxea/Dockerfile -t dume-g05:local .
```

The model-swap wrapper builds the GR00T base from source by default. To reuse a published dependency base, set `MODEL_SWAP_BASE_IMAGE=REGISTRY/IMAGE@sha256:DIGEST`; mutable tag overrides are rejected. Registry authentication is the operator's responsibility. No dated local image ID or host tokenizer cache is required. Rebuilding can require substantial disk space for layers, backbone and checkpoints. Record the resulting image digest and `pip freeze` alongside measurements: top-level pins and source recipes do not promise bit-identical transitive resolution across future builds.

## Start

GR00T gRPC, checkpoint mounted read-only:

```bash
docker run --rm --name dume-lerobot --gpus all \
  -p 127.0.0.1:8080:8080 --shm-size 1g \
  -v "$PWD/checkpoints/GR00T-N1.7-3B-SO101:/checkpoints/model:ro" \
  -e DUME_CHUNK_OBSERVER=lightweight lerobot-policy
```

HTTP routes use the same image, one profile per process:

```bash
MODEL_SWAP_PI05_SO101_CHECKPOINT=/absolute/path/pretrained_model \
  bash scripts/run_model_swap_server.sh pi05-so101 dume-model-swap:local
# Alternatives: pi05-base, molmoact2-so101, groot-so101.
# GR00T requires MODEL_SWAP_GROOT_CHECKPOINT=/absolute/path/checkpoint.
```

G0.5:

```bash
G05_CHECKPOINT_ROOT=/absolute/path/G05-snapshot \
  bash scripts/run_galaxea_server.sh dume-g05:local
```

HTTP health is `GET /health`; it reports the profile, runtime timing/memory and faults. A ready server does not authorize motion. Checkpoint/startup failure must not fall back to another model or CPU inference. For native GR00T use the existing `build_gr00t_image.sh` workflow in the README. Its optional `serve_observed_native.py` launcher requires this repository's `policy` package and scripts mounted on its Python path, plus the pinned Cosmos cache.

Continue with [validation](POLICY-VALIDATION.md), [SO101 mappings](SO101-POLICY-CONTRACTS.md), [async execution](ASYNC-INFERENCE.md), or [EC2](EC2-INFERENCE.md).

Deployment selection and package ownership are documented in [ARCHITECTURE.md](ARCHITECTURE.md).
