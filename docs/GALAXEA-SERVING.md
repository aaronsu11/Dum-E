# G0.5 native serving

The G0.5 SO101 server runs in an isolated container. Select the client backend
with `DUME_POLICY_BACKEND=galaxea`; it implements the existing synchronous
`IPolicyBackend` contract. It owns no cameras or serial devices.

## Pinned runtime

- GalaxeaVLA: `89f2322b4ad016e192437adc1a2c253b05bab246`
- OpenGalaxea/G05: `e312be81e90c56a55bcb26b57429bd39a335b449`, `g05-so101`
- Python 3.10.16, CUDA runtime 12.8.1, torch 2.7.1, transformers 4.57.1
- Flash Linear Attention 0.5.0; SDPA attention; torch.compile disabled
- BF16 model weights with upstream FP32 exceptions preserved, including the
  action tokenizer. CPU is used for loading/staging, never model inference.
- Both native checkpoint deserialization sites use `weights_only=True`.

The server verifies all 12 checkpoint/processor files against
`docker/galaxea-policy/checkpoint-manifest.json` before loading. The image
contains neither model weights nor Hugging Face credentials. Obtain gated model
access through the normal Hugging Face flow, then download the pinned revision.
Do not replace checkpoint paths with an unpinned base model.

## Build and run

```bash
docker build -f docker/galaxea-policy/Dockerfile -t dume-g05:phase9 .
scripts/run_galaxea_server.sh dume-g05:phase9
```

The launcher defaults to the pinned Hugging Face snapshot in the local cache.
Set `G05_CHECKPOINT_ROOT` for a materialized checkpoint directory and
`G05_EVIDENCE` for startup evidence. Cached snapshots' relative blob links are
mounted explicitly. The server binds only `127.0.0.1:8765` on Linux host networking.

The launcher refuses to start if `nvidia-smi` reports any compute process. Stop
the previous model's process/container before switching; emptying a CUDA cache
does not unload a model.

```bash
docker stop --timeout 2 dume-g05
```

## Client and remote access

```bash
export DUME_POLICY_BACKEND=galaxea
export DUME_GALAXEA_POLICY_PORT=8765
export DUME_ASYNC_INFERENCE=0
```

`controller.galaxea_policy_port` is forwarded by `dum_e.py` if supplied. An
exported environment value takes precedence. The existing legacy
`controller.policy_port` does not select this WebSocket endpoint.

For EC2, keep the server loopback-only and forward its port using SSM:

```bash
aws ssm start-session --region us-west-2 \
  --target i-06ef273a3d9fea631 \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8765"],"localPortNumber":["18765"]}'
```

Use `DUME_GALAXEA_POLICY_PORT=18765` through that tunnel. Do not expose a public
robot-inference port. This backend deliberately rejects LeRobot's
`DUME_ASYNC_INFERENCE=1`: its native protocol is a different transport, and
Phase8's async queue/watchdog integration is not implicitly inherited.

## Hardware-free verification

```bash
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/check_galaxea.py \
  --port 8765 --output corpus/a-new-g05-check
```

This executes two warmups and 12 observations, one per recorded episode and
one seed each, through the real client backend. It saves all action arrays,
server-stage timings, token counts, dtype/memory identity and client RPC time.
Output directories must be new so failed attempts remain inspectable.

The backend verifies the server's checkpoint identity, resets the native cache,
sends one full observation, then drains 31 cached actions. Returned chunks must
contain 32 finite six-joint actions. The server admits only one client at a time.
Client errors close the connection; there is no automatic motion retry.

The autoregressive budget is 300 generated tokens across stages. A budget
overrun or CUDA OOM faults the model session; restart is required. The
`cot_text` field is diagnostic only. This SO101 checkpoint emitted action-code
tokens in that field during the local check, not usable natural-language
subtask narration.

## Scope and next test

See `MODEL-MODALITY-COMPATIBILITY.md` for exact camera/frame/calibration mappings
and the limitations of frozen-corpus inputs. All four models' physical tests
remain a separate, supervised session. This integration does not silently
change GR00T's validated units, certify Pi0.5 base as a robot policy, or enable
LeRobot async mode for G0.5.
