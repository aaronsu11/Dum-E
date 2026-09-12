# GPU model-swap evaluation on EC2

This is an evaluation-only JSON/HTTP server. It imports no robot controller and
receives recorded observations. A successful run establishes inference, finite
output, and timing; it does not establish task success or authorize physical motion.
The production LeRobot controller uses a different gRPC transport. These HTTP
benchmarks must not be described as a production-controller integration pass.

## Workstation and access

Aaron supplied `i-06ef273a3d9fea631` in `us-west-2`, named
`IsaacLabDcvStack/DCV/Instance`: g6.2xlarge, NVIDIA L4 with 23,034 MiB VRAM.
The missing module for kernel `6.8.0-1063-aws` was repaired with driver
`580.178.04`; host and Docker GPU checks passed without reboot.

The root volume remains 150 GB. Aaron authorized removal of the 80.4 GB
`gr00t-finetune:latest` image. Docker retained its stopped training container and
files, so actual reclamation was about 27 GiB. About 13 GiB remained after the
inference image pull. Isaac Lab, W&B, and user volumes were preserved.

Checkpoints and evidence live under `/mnt/efs/dume-phase8.1-20260912` on the
existing EFS mount. The validated GR00T checkpoint already existed on EFS;
every weight and sidecar hash was checked against the local frozen input lock.
No new workstation, expanded volume, or public inference ingress was created.

Use Session Manager forwarding from the local workstation:

```bash
aws ssm start-session --region us-west-2 --target i-06ef273a3d9fea631 \
  --document-name AWS-StartPortForwardingSession \
  --parameters '{"portNumber":["8081"],"localPortNumber":["18081"]}'
```

## Immutable image and profiles

The all-model image is:

```text
177118830501.dkr.ecr.us-west-2.amazonaws.com/dume/model-swap:phase8.1-dc905a7
sha256:bd14d39f7e5a391091e806cbf5f47d92e1cd411d2becda49ed65cd697e8e2558
```

Pi05 and Molmo evaluation initially used the earlier image `phase8.1-73a4c32`,
digest `sha256:0a89526b9265868c5a27ac6216750c08a79f1f1380cfdfb9e3ebae914adcfe5d`.
The later image adds the GR00T profile and current validated loader/guard sources;
it retains the same dependency base and Pi05/Molmo inference configuration.
Publication records live in `corpus/phase8.1-model-swap-20260912`.

| Profile | Output per chunk | Configuration and limits |
|---|---|---|
| `groot-so101` | 16 × 6 | Validated Dum-E checkpoint; BF16 materialization; existing serving geometry, normalization and decode guard; observer off. |
| `pi05-base` | 50 × 32 | Pinned LeRobot base; upstream mixed BF16/FP32; compilation off; six raw joints padded to32, two mapped cameras and a zero third camera. No SO101 action mapping. |
| `molmoact2-so101` | 30 × 6 | Pinned official SO100/101 checkpoint; BF16; continuous actions,10 steps, checkpoint normalization including gripper; CUDA graphs off. Dum-E calibration/camera compatibility remains unverified. |

Run one model per GPU. `scripts/run_model_swap_server.sh` rejects an occupied GPU
and mounts no arm or camera devices. Set `MODEL_SWAP_CACHE` and
`MODEL_SWAP_EVIDENCE` to suitable directories. GR00T additionally accepts
`MODEL_SWAP_GROOT_CHECKPOINT`; its checksum must match the validated checkpoint.
GR00T uses the image's pinned Cosmos cache offline. Pi05's pinned tokenizer is
included without HF credentials. Checkpoints remain outside the image.

## Measurement and integration scope

For each profile, run two warmups and exactly12 observations, one per recorded
episode, with one seed each. Run once with a client on EC2 and once from this
workstation through the SSM tunnel. The server runs on EC2 in both cases.

```bash
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/check_model_swap.py \
  --profile pi05-base --endpoint http://127.0.0.1:18081 \
  --output corpus/phase8.1-model-swap-20260912/NEW-UNUSED-OUTPUT-DIRECTORY
```

Preserve output directories instead of overwriting or silently retrying failures.
The report separates preprocessing, GPU generation, postprocessing, and client RPC
elapsed time. RPC includes JSON serialization and uncompressed RGB transfer
through SSM; the residual is not pure geographic network latency. Report medians
and maxima for this small sample, not production tail-latency guarantees.
Native chunk horizons and action meanings differ across models; do not treat the
table as a task-quality or action-parity comparison.

Protocol checks cover malformed requests, unknown routes, and server health after
rejection without additional model inference. The unit suite covers concurrent
request rejection and health responsiveness. Physical arm, calibrated model
adapters, and production-controller transport remain separate gates.

Raw evidence and the final benchmark report are kept in
`corpus/phase8.1-model-swap-20260912`. Stop evaluation containers and temporary
evidence-serving processes after retrieving results. Aaron controls EC2 stop or
termination and associated costs.
