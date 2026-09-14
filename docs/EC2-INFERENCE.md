# EC2 inference and network comparison

Use the same model image, checkpoint pins and benchmark recordings as the local setup. The GPU server receives observations only; robot hardware stays on the client. The validated workstation was an L4 instance in us-west-2. Instance IDs, mounted volumes and credentials are deployment inputs, not code defaults.

## Deploy

1. Verify the NVIDIA driver/container runtime and available disk space. Keep model weights outside the image on a suitable data volume. Do not assume the root disk can hold all checkpoints and build layers.
2. Build using [POLICY-SERVING.md](POLICY-SERVING.md), then publish an immutable tag:

   ```bash
   AWS_REGION=us-west-2 bash scripts/publish_model_swap_ecr.sh \
     dume-model-swap:local YOUR_IMMUTABLE_TAG
   ```

   The helper prints the registry image digest and size. Record the digest; authenticate Docker on EC2 with its assigned AWS permissions and pull `REGISTRY/IMAGE@sha256:DIGEST`. G0.5 uses its separate image; publish it through the same helper with `MODEL_SWAP_ECR_REPO=dume/g05`.
3. Download only the pinned model runtime artifacts. Use read-only checkpoint mounts and writable cache/evidence directories. Start the same launch script with the pulled digest as its image argument. Do not mount serial or video devices on the server.
4. Keep inference bound to loopback. Forward it to the local client:

   ```bash
   aws ssm start-session --region us-west-2 --target YOUR_INSTANCE_ID \
     --document-name AWS-StartPortForwardingSession \
     --parameters '{"portNumber":["8081"],"localPortNumber":["18081"]}'
   # Or SSH: ssh -N -L 18081:127.0.0.1:8081 YOUR_WORKSTATION
   ```

   G0.5 uses server port 8765; LeRobot gRPC uses 8080. Public inference ingress is unnecessary.

## Compare

Run `scripts/benchmark_policy.py` with the same 12 observation files, instruction, seed, warmup count and checkpoint, once on EC2 against its loopback server and once locally through the tunnel. Preserve both output directories. Compare the complete action arrays as well as timing. RPC includes serialization, raw image transfer, health checks where applicable and server computation; subtracting GPU generation does not isolate geographic network latency.

Historical results, two warmups plus 12 observations, one seed each:

| Model | EC2-local RPC median / max | Client over SSM median / max | EC2 generation median | Peak allocation |
|---|---:|---:|---:|---:|
| GR00T HTTP | 172.4 / 174.7 ms | 3139.4 / 3289.7 ms | 146.9 ms | 6093.5 MiB |
| Pi0.5 base HTTP | 407.6 / 530.0 ms | 3356.9 / 3439.6 ms | 389.2 ms | 9091.1 MiB |
| MolmoAct 2 HTTP | 737.1 / 741.9 ms | 3650.6 / 3731.3 ms | 709.2 ms | 11309.7 MiB |
| G0.5 WebSocket | 2171.1 / 2183.8 ms | 4096.7 / 5897.1 ms | 1982.5 ms | 6276.9 MiB |

For each model, all 12 EC2-local versus tunneled outputs were bit-identical. This compares clients against the same server, not models or GPUs. G0.5 local RTX3060 RPC was 933.7 ms median / 941.9 ms maximum; cold inference and loading were excluded. Its L4 action-token stage was slower; the cause remains unqualified.

HTTP requests contained about 2.46 MB of base64/raw RGB JSON, explaining why these measurements should not be presented as optimized remote-control latency. Remote synchronous bounded trials later ran, but this network path is not qualified for the local Pi0.5 RTC deadline. Keep the server resident for warm measurements and record startup separately.

Historical image digests, full source and experiment narratives are recoverable from `archive/pr13-before-consolidation`; current builds have different source and must receive new digests and measurements. No cloud resource resizing, driver repair or image deletion is part of the consolidation.
