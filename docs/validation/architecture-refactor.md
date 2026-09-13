# Architecture refactor validation — 2026-09-13

The refactor separates embodiment IO/mappings/safety from policy runtimes and moves Python server code out of Docker directories. The common factory serves agent, benchmark and trial clients. Deployment YAMLs distinguish runtime, model, checkpoint, transport and execution; unsupported combinations fail before client construction.

## Completed software checks

- Full suite: **476 passed, 3 skipped** (38.53 seconds). Skips are opt-in GPU/speech tests. Includes simulated HTTP/MCP/shared-memory task dispatch, failure, retarget and cancellation; calibration and raw hardware-dispatch protections; full action chunks; and RTC deadline/prefix behavior.
- Focused checks after import cleanup: **44 passed**. Architecture tests reject runtime/scheduler imports of embodiments and Python implementations in Docker folders, exercise injected mappings with different joint names, and reject unimplemented embodiments, unqualified Pi0.5 base outputs and unsupported execution modes.
- Compared 29 definitions/classes to commit `2aae99a` using ASTs, normalizing imports and moved source-path literals: controller, low-level safety, GR00T async scheduling, native client, SO101 frame conversion and the model inference method retain their executable behavior.
- Imported the relocated server modules in the cached LeRobot, native GR00T and Galaxea dependency images. The GR00T metadata serving guard passed; G05 found its manifest beside its server. No hardware controller was imported.
- Built small source-only layers on those three dependency images and ran their actual module entry points with `--help`, networking disabled and no GPU, checkpoint or robot devices attached. Tested HTTP serving plus the GR00T gRPC launcher, native ZMQ launcher and Galaxea launcher.
- Python compilation, shell syntax, CLI help and relative documentation links checked.

These checks used the existing dependency images (`dume-model-swap:pi05-so101-phase10`, `gr00t:latest`, `dume-g05:phase9`). They do not establish a fresh build of all downloaded dependencies or new GPU latency/accuracy results. Checkpoint revisions, loading precision and model inference settings were retained. Historical measurements remain in [policy-trials.csv](policy-trials.csv) and the existing serving/validation guides.

To reproduce the software suite:

```bash
uv sync --locked
uv run pytest -q
```

To check a relocated server inside its compatible dependency image without loading weights:

```bash
docker run --rm --network none --read-only --tmpfs /tmp:rw,size=512m \
  -w /tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/app \
  -v "$PWD/policy:/app/policy:ro" --entrypoint python3 \
  dume-model-swap:pi05-so101-phase10 -m policy.backends.lerobot.http_server --help
```

Use the normal build scripts and [serving guide](../POLICY-SERVING.md) for deployment images. The source-layer validation tags are not new dependency baselines.

## Final gate: attended physical validation

**Pending with Aaron. No physical trial was run during this refactor.**

Use a fresh workspace and current calibration/camera checks for each attempt. Prepare and approve the source-bound snapshot with the present operator, then run one bounded trial at a time. Start with the existing three-chunk protocol, confirm safe hold after each run, and cover the four mapped policies. Pi0.5 RTC exercises the relocated prefix scheduler; the other policies can use their existing synchronous bounded checks. Task accuracy is not the acceptance criterion.

Old physical approvals and GR00T async latency admission files refer to previous source paths/hashes and must be regenerated before reuse. Do not copy historical approvals onto the refactored source. GR00T production async testing additionally needs current measured admission settings; a simple chunk benchmark is not that qualification.

The final result remains pending until Aaron confirms physical behavior and safe stopping. See [architecture](../ARCHITECTURE.md) for package ownership and [validation](../POLICY-VALIDATION.md) for the exact commands.
