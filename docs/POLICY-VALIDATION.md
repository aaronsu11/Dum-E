# Validate and reproduce measurements

Start with `uv sync --locked`, then `uv run pytest -q`. Default tests use fake hardware, local transport servers and tracked metadata fixtures; no downloaded weights or GPU are required. Live GPU and speech tests require their explicit opt-in variables. Default tests may bind loopback ports and shared memory.

| Area | Main tests |
|---|---|
| Transport, backend selection, dependency boundaries | `test_container_contract.py`, `test_policy_backend.py`, `test_lerobot_backend.py` |
| GR00T checkpoint/normalization/geometry | `test_groot_guard.py`, `test_par05_image_geometry.py`, `test_lerobot_upstream_surface.py` |
| Mapped policies, strict requests and RTC | `test_model_swap.py`, `test_so101_contract.py`, `test_pi05_rtc.py`, `test_galaxea.py` |
| Motor dispatch, calibration and bounded targets | `test_hardware_safety.py`, `test_controller_safety.py`, `test_calibration_assertion.py`, `test_integration_trial.py` |
| Async queue and application lifecycle | `test_async_chunks.py`, `test_async_pick.py`, `test_mcp_interfaces.py` |

`tests/fixtures/groot-so101` contains only metadata, not model weights. It tests guard/processor behavior, not authenticity of a downloaded checkpoint. Archive-only approval/corpus/golden tests were removed with the machinery they tested; runtime negative checks and low-level dispatch tests remain.

## Capture a portable SO101 observation

Stop other owners of the serial/camera devices first. The following reads actual calibration, state and RGB cameras without follower connect, torque changes or motor register writes:

```bash
uv run python scripts/run_policy_trial.py capture \
  --config my-dum-e.yaml --workspace outputs/capture-01
```

The NPZ contains exactly `state` `(6,)`, `front` and `wrist` `(480,640,3)`; state is centered degrees and gripper 0–100. Use a new output directory for each capture. The same format can be produced from a dataset after an explicit, verified unit/camera conversion. Old normalized GR00T corpus arrays cannot simply be renamed into this format.

## Small inference benchmark: no hardware

```bash
uv run python scripts/benchmark_policy.py --profile pi05-so101 \
  --observations outputs/capture-01/observation.npz \
  --instruction 'Pick up the banana and put it on the plate' \
  --port 8081 --samples 12 --warmups 2 --output outputs/pi05-benchmark-01
```

Supported profiles are listed in `--help`; GR00T requires `--calibration /absolute/path/calibration.json`. Supply 12 distinct recordings to reproduce the one-per-episode sampling design; one supplied file is cycled and represents only that observation. The report records hashes, profile, seed, warmups, client RPC, server metadata and complete finite output arrays. The base Pi0.5 profile deliberately uses padded input and does not return robot-ready actions. Report warm medians/maxima, not statistical tail guarantees or accuracy.

A live gRPC smoke can also run with `DUME_RUN_LIVE_LEROBOT_TESTS=1 uv run pytest tests/test_lerobot_serving_live.py -q`. It sends two synthetic observations to the provisioned loopback server and neither starts nor replaces containers. Wrong-horizon/failure guard cases are tested with isolated fake pipelines by default.

## One bounded physical trial

Prepare, approve and run with the **same** flags. Approval is interactive, names the present operator, binds the prepared snapshot and expires after 30 minutes. Preparation does not connect hardware. This example is three synchronous Pi0.5 chunks:

```bash
trial_args=(--profile pi05-so101 --config my-dum-e.yaml \
  --warmup-observation outputs/capture-01/observation.npz \
  --workspace outputs/pi05-trial-01 --port 8081 --chunks 3)
uv run python scripts/run_policy_trial.py prepare "${trial_args[@]}"
uv run python scripts/run_policy_trial.py approve "${trial_args[@]}" --operator YOUR_NAME
uv run python scripts/run_policy_trial.py run "${trial_args[@]}"
```

Other profiles: `g05-so101`, `molmoact2-so101`, `groot-so101`. Only mapped Pi0.5 accepts `--scheduler rtc`; that bounded mode plays 100 targets for the default three-chunk run. The runner records raw versus bounded/sent commands, observations, checkpoint metadata, config/source/calibration identities, timings and stop events. Its terminal result requests an operator observation; software completion is not visible motion, grasp success or safe-hold confirmation. Do not reuse an attempted workspace.

`--motion-probe` is an explicit synthetic pan diagnostic, not model output. It has a separate 6-degree excursion and must not count as policy success. Default policy trials retain a 5-degree envelope. Further poses, action budgets or embodiments require separately reviewed bounds.

## Async and agent checks

See [ASYNC-INFERENCE.md](ASYNC-INFERENCE.md) for the distinct GR00T queue and Pi0.5 RTC routes. Two small specialist tools remain because they measure different boundaries: `check_pi05_rtc.py` compares guided model inference, while `check_pi05_rtc_scheduler.py` drives the real HTTP scheduler with a dummy controller. Both accept explicit recorded-trial and output directories and never open hardware. Run the first inside the model image, with the repository mounted on `PYTHONPATH` and the checkpoint read-only; run the second on the client against the ready RTC server. Recorded trial directories contain `chunk-{1,2,3}/live-observation.npz` and `snapshot.json`.

Agent/MCP failure, retarget and cancel are covered by `test_mcp_interfaces.py` with isolated shared memory and simulated hardware. Optional audible checks attach to an existing voice stack using `scripts/check_agent_integration.py failure|control --workspace NEW_DIRECTORY`; the simulated robot is named `Voice Test Robot`. These modes use the real voice/MCP services but scripted robot-side tool selection. They must not be reported as physical trials.

## Recorded results (before consolidation)

Physical campaign: September 12 PDT / September 13 UTC, 2026. **14 numbered runs: 11 policy runs and three synthetic diagnostics; 1,286 targets.** Picking accuracy/fine-tuning was outside the accepted integration scope. Per-run data: [policy-trials.csv](validation/policy-trials.csv); passes and deferrals: [coverage.json](validation/coverage.json).

| Representative run | Server / client | Generation mean | Client RPC mean | Peak PyTorch allocation | Weight files | Observed result |
|---|---|---:|---:|---:|---:|---|
| G0.5 / 8 | RTX3060 / local | 1.218 s | 1.289 s | 6.13 GiB | 11.95 GB | Small targets; no visible policy motion |
| MolmoAct 2 / 10 | EC2 L4 / SSM client | 0.699 s | 3.773 s | 11.04 GiB | 21.77 GB | Small visible motion |
| Pi0.5 SO101 / 11 | EC2 L4 / SSM client | 0.327 s | 3.376 s | 8.83 GiB | 16.57 GB | Visible motion |
| GR00T / 13 | RTX3060 / local | 0.161 s | 0.234 s | 5.95 GiB | 12.58 GB | Visible motion |
| Pi0.5 RTC / 14 | RTX3060 / local | See async comparison | 443/427/430 ms per chunk | ~8.87 GiB supporting replay | Same SO101 weights | Visible smooth motion; safe hold confirmed |

G0.5's generation field and other models' generation fields have different internal timing boundaries. GB is decimal checkpoint bytes, not parameter count or image size; GiB is peak allocated tensor memory, not total GPU use. Pi0.5 base had three input-only chunks (50×32), RPC 1.034/0.592/0.578 s, 8.88 GiB allocation, 14.47 GB weights, and zero motor commands.

## Archive and open work

The full pre-consolidation source and narratives are preserved at `archive/pr13-before-consolidation`, commit `0c08e405a33e7567152f2aa808625cb6536e7343`. Example: `git show archive/pr13-before-consolidation:docs/PHASE10-INFERENCE-TRIAL-SUMMARY.md`. Raw recordings/checkpoints remain local, ignored assets; do not claim a fresh clone contains them.

Still open: coherent G0.5 task motion; dataset-specific starting-pose/all-joint qualification; sustained runs and physical RTC fault qualification; explicit second-serial-owner rejection; startup reuse optimization and unexplained latency outliers. The post-trial RTC expiration fix has software regression coverage but was not part of physical trial14's executed source. No old physical result certifies the newly refactored code without another attended trial.
