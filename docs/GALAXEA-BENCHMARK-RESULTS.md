# Phase 9 — G0.5 GPU and network results

G0.5 SO101 passed the bounded inference check on the local RTX3060 and the EC2
L4, including the real workstation client over SSM. No physical arm movement
was commanded. Physical testing remains a separate session.

## Protocol and results

Each location ran two warmups plus 12 recorded observations, one per episode
and one seed each. Each result is a finite `(32,6)` action chunk returned through
`GalaxeaPolicyBackend`, including frame conversion and native cache draining.

| Server / client location | Passed | Median full-chunk RPC | Maximum RPC | Median server inference |
|---|---:|---:|---:|---:|
| RTX3060 / local workstation | 12/12 | 933.705 ms | 941.858 ms | 871.528 ms |
| EC2 L4 / EC2-local client | 12/12 | 2171.138 ms | 2183.821 ms | 1982.480 ms |
| EC2 L4 / workstation through SSM | 12/12 | 4096.741 ms | 5897.118 ms | 1990.469 ms |

All 12 action arrays were **bit-for-bit identical between the EC2-local and
workstation clients**. This verifies the transport against the same server;
it is not a cross-GPU or cross-model parity claim.

Peak PyTorch allocation was **6276.863 MiB (6.13 GiB)** on both GPUs.
G0.5 fits the RTX3060 without CPU inference or offloading. The first cold
inference was 21.807 seconds locally and 36.184 seconds on L4, excluded from
the measured samples. Model loading itself took 24.167 and 74.113 seconds,
respectively, excluding earlier imports and checkpoint hashing.

The model uses BF16 with 380 FP32 and 565 BF16 parameter tensors, preserving
upstream exceptions and the separate action tokenizer. SDPA is used;
`torch.compile=False`. Native Triton kernels still incur first-use compilation.

## What accounts for latency

| Median server stage | RTX3060 | L4, EC2-local |
|---|---:|---:|
| Prefill | 98.008 ms | 86.967 ms |
| Autoregressive action-token generation | 637.791 ms | 1485.779 ms |
| Flow matching | 126.014 ms | 390.712 ms |

The L4 run is slower in action generation despite faster prefill. These
measurements locate the difference but do not establish whether host launch
overhead or individual GPU kernels dominate it; no causal profiler experiment
was added to this milestone.

The native protocol sends one full observation and retrieves one action per
request, requiring 32 action exchanges plus reset per chunk. The client RPC
includes packing, compression, transport, cache draining and decoding. The
workstation median is about **1.93 seconds above EC2-local**, with variable
network tails. This is not a measurement of bare network RTT.

For context, the earlier Phase8.1 EC2-local/workstation HTTP chunk medians were
GR00T 172/3139 ms, Pi0.5 408/3357 ms and MolmoAct2 737/3651 ms. Those transports
return a whole chunk in one reply, use different horizons and model
configurations, and were not rerun here. See `MODEL-SWAP-EC2-RESULTS.md`.

## CoT and failure behavior

The checkpoint generated 39 tokens per measured request. Its `cot_text` field
contained action-code tokens, not usable natural-language `Subtask` narration.
The verdict is **no usable subtask CoT observed**. An early local report's
`cot_observed=true` meant only that the raw field was nonempty; the explicit
interpretation is preserved in `local-summary.json`.

The wrapper enforces a 300-token total budget across generation stages, refuses
incomplete output at that boundary, and faults on overrun or CUDA OOM. It checks
the actual SO101 codec presence pattern and rejects malformed observations.
Real local and network protocol checks both verified a 400 error, connection
closure and successful fresh health connection. Language/reset behavior is
covered without commanding motion.

With GR00T loaded at roughly 6 GiB, the G0.5 launcher exited with code 3 and an
actionable occupied-GPU message before loading weights. Both models were stopped
after their checks. Nineteen scoped tests passed, including WebSocket
integration, malformed arrays, exact action groups, frame conversions and the
token budget.

## Published image and evidence

- Implementation commit: `43e8661`
- ECR: `177118830501.dkr.ecr.us-west-2.amazonaws.com/dume/galaxea-policy:phase9-smoke-v1`
- Image digest: `sha256:cbe14b86ede7deefedb3770a578ea9f8cce525edebe0d16232a41ab8ace7b417`
- GalaxeaVLA: `89f2322b4ad016e192437adc1a2c253b05bab246`
- OpenGalaxea/G05: `e312be81e90c56a55bcb26b57429bd39a335b449`
- EC2: `i-06ef273a3d9fea631`, us-west-2, L4, driver 580.178.04
- EC2 workspace/checkpoints/evidence: `/mnt/efs/dume-phase9-20260912`
- Local evidence: `corpus/phase9-g05-20260912`

Key local evidence paths:

- `local-check-03/result.json`, raw action arrays and `local-summary.json`
- `ec2-evidence/ec2-local/result.json`, raw action arrays and server logs
- `workstation-to-ec2-03/result.json`, raw action arrays
- `transport-comparison.json`
- `local-protocol-check.json`, `network-protocol-check.json`
- `groot-blocks-g05.log`, `scoped-tests.log`
- `ecr-image.json`, `ec2-stop-result.json`, `ec2-evidence-stop-result.json`
- `modality-audit.json`

Failed startup and tunnel attempts remain alongside successful evidence. The
startup fixes were HF cache blob mounts, moving the separately held tokenizer
to CUDA, and recognizing the checkpoint's three deliberately absent codec
groups. A stale SSM tunnel stalled before reaching the server; a fresh tunnel
on local port 18766 completed the network check. No server port was exposed
publicly.

Checkpoint upload resumed after intermittent DNS failures, preserving completed
multipart data. Only our three stopped Phase8.1 inference containers/images
were removed from EC2 to make room. The stopped training container, IsaacLab
and W&B were preserved. The root disk remains 150 GB, with about 16 GB free.

At closeout the G0.5 EC2 container and temporary evidence service are stopped,
the inference/evidence tunnels are closed, and the L4 reports 4 MiB in use.
The EC2 instance remains running under Aaron's control.

## Physical-test handoff

See `MODEL-MODALITY-COMPATIBILITY.md` and `GALAXEA-SERVING.md`.
The current front/wrist and six-joint software mappings are explicit, but live
camera identity/orientation and empirical calibration checks are pending.
The devices were absent during this audit.

Pi0.5 base still has no verified SO101 action mapping, and Molmo's Phase8.1
evaluation transport is not yet a production controller adapter. Those
prerequisites must be completed before calling all four models physically ready.
G0.5 also does not inherit LeRobot's async queue/watchdog automatically.
Keep cold-start optimization, native chunk batching and deeper L4 profiling in
the backlog rather than expanding this bounded evaluation.
