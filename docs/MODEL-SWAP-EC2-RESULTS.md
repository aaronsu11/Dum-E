# EC2 model evaluation — 2026-09-12

All three models passed two warmups plus12 frozen observations on the EC2 L4,
and again from the local workstation through an SSM tunnel. Each observation used
one seed. All matched local/network action arrays were bit-identical. No arm moved.

| Model | Chunk shape | EC2 local RPC median / max | Workstation RPC median / max | EC2 generation median | Peak PyTorch VRAM |
|---|---|---:|---:|---:|---:|
| GR00T N1.7 SO101 | 16 × 6 | 172.4 / 174.7 ms | 3139.4 / 3289.7 ms | 146.9 ms | 6093.5 MiB |
| Pi0.5 base | 50 × 32 | 407.6 / 530.0 ms | 3356.9 / 3439.6 ms | 389.2 ms | 9091.1 MiB |
| MolmoAct2 SO100/101 | 30 × 6 | 737.1 / 741.9 ms | 3650.6 / 3731.3 ms | 709.2 ms | 11309.7 MiB |

The server ran on EC2 for both client locations. This measures the evaluation HTTP
client/server integration, not production LeRobot gRPC controller integration.
Requests contain approximately2.46MB of JSON/base64 raw camera pixels. RPC includes
serialization, transfer through SSM, parsing and inference. The roughly3-second
residual must not be interpreted as pure geographic network latency. Smaller error
requests completed successfully; malformed bodies returned400/413 and unknown routes
returned404, with health still ready for all profiles. Concurrency rejection and
responsive health were also covered by the scoped21-test local suite.

Pi0.5 is an unmapped32-dimensional base-model smoke test; Molmo uses the official
SO100/101 normalization but Dum-E joint/camera compatibility remains unverified.
Different horizons/action spaces prevent treating these results as task-quality
or grasping comparisons. The GR00T checkpoint matches every validated local hash.

Pi05 and Molmo used ECR image73a4c32. GR00T used dc905a7, which adds the guarded
GR00T profile and refreshes the loader sources while retaining the dependency base.
See [deployment details](MODEL-SWAP-EC2.md) for full immutable digests and settings.

Local RTX3060 evidence: Pi05 passed12/12 at446.5ms median RPC; Molmo loaded BF16
weights but OOMed on its first forward pass. The L4 completed Molmo without CPU
fallback. Reported load times exclude snapshot download/checkpoint hashing; warmup
requests are excluded from the table. This small sample gives no production tail
latency guarantee.

The physical arm test remains paused. The raw-frame SSM path adds substantial delay;
production transport, model-specific action mapping and controller timing must be
reviewed before physical execution. No controller deadlines or safety limits changed.

Raw per-observation outputs, timings, startup logs, checkpoint/image identities,
protocol checks and summary JSON are under `corpus/phase8.1-model-swap-20260912/`.
