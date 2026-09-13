# Multi-policy SO101 trial summary

**Closeout:** Phase 10 passed under the operator-amended integration scope. Pi0.5 physical RTC trial14 passed with visible smooth motion and safe hold. See [PI05-RTC-LOCAL.md](PI05-RTC-LOCAL.md) and [PHASE10-CLOSEOUT.md](PHASE10-CLOSEOUT.md).

Physical campaign: **2026-09-12 PDT / 2026-09-13 UTC**. This report covers every numbered run in the current Phase 10 campaign, its supporting input-only checks, and separately labeled earlier GPU/network and GR00T async evidence. It is not a rerun of the earlier benchmarks.

**Outcome:** G0.5 exercised live inputs, inference and bounded dispatch but requested only small targets and showed no measured policy motion. MolmoAct2, the SO101 Pi0.5 checkpoint and calibrated GR00T produced operator-confirmed visible policy motion. The raw Pi0.5 base model was tested with motor output disabled. Picking accuracy was not an acceptance criterion.

**Campaign count:** 14 numbered motor-command runs: 11 policy runs and 3 synthetic pan diagnostics; 1,286 commands in total, including 1,166 policy-derived commands. These counts exclude startup warmups and input-only checks. No additional motion was run to create this report.

## Latest representative result for each mapped policy

Latency below is the **mean across chunks in the indicated run**. G0.5 uses its recorded server `infer_ms`; other profiles use `generation_ms`. Those internal timing boundaries are not identical. RPC is the client’s whole `get_action` call, including transport, decoding and any health request inside it; it excludes robot playback and preflight.

| Policy / run | Server and client | Chunk | Server inference mean | Client RPC mean | Peak allocated VRAM | Weight files | Outcome |
|---|---|---:|---:|---:|---:|---:|---|
| G0.5 SO101 / 8 | RTX 3060, local | 32 × 6 | 1.218 s | 1.289 s | 6.13 GiB | 11.95 GB | No measured/visible policy motion; small raw targets |
| MolmoAct2 SO100/101 / 10 | L4 on EC2; local client over SSM | 30 × 6 | 0.699 s | 3.773 s | 11.04 GiB | 21.77 GB | Small visible policy motion confirmed |
| Pi0.5 SO101 (Project-IRA 008000) / 11 | L4 on EC2; local client over SSM | 50 × 6 | 0.327 s | 3.376 s | 8.83 GiB | 16.57 GB | Visible policy motion confirmed |
| GR00T N1.7 SO101 / 13 | RTX 3060, local | 16 × 6 | 0.161 s | 0.234 s | 5.95 GiB | 12.58 GB | Visible policy motion confirmed |

**Pi0.5 base, input-only:** three live-input chunks of 50×32, RPC **1.034 / 0.592 / 0.578 s**, peak allocation **8.88 GiB**, weight file **14.47 GB**. Zero motor commands; its 32-dimensional outputs were never treated as six SO101 joints.

Weight size is decimal GB of checkpoint tensor files, not container size or a parameter count. G0.5 includes its 11.44 GB main checkpoint plus 0.507 GB action tokenizer. Other rows count the model safetensors file(s), excluding small processor files. Peak VRAM is recorded PyTorch allocation, not total board usage or allocator-reserved memory. Stored checkpoint precision differs from inference precision, so disk size and VRAM are not interchangeable.

## Exact model and inference configuration

| Profile | Checkpoint and pin | Backend / transport in this campaign | Parameter precision and generation settings | Seed |
|---|---|---|---|---|
| `g05-so101` | `OpenGalaxea/G05`, `g05-so101`; revision `e312be81e90c56a55bcb26b57429bd39a335b449` | Native Galaxea implementation `89f2322b4ad016e192437adc1a2c253b05bab246`; WebSocket; one observation then native action-cache draining | BF16 with FP32 exceptions; SDPA; compile off; 10 flow steps; 39 generated action-code tokens observed; 300-token budget | Server-reported default `0` |
| `molmoact2-so101` | `allenai/MolmoAct2-SO100_101` @ `152569fe57914d97be91055800035f54e250d009` | LeRobot MolmoAct2 runtime; evaluation JSON/HTTP server plus explicit SO101 controller bridge | BF16; continuous actions; 10 inference steps; CUDA graphs off; `norm_tag=so100_so101_molmoact2`; gripper normalization enabled | `20265907` |
| `pi05-base` | `lerobot/pi05_base` @ `b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba` | LeRobot PI05Policy, evaluation HTTP; input-only | BF16 + FP32 components; compile off; 10 steps; raw six-state padding to 32, without verified SO101 normalization | `20265907` |
| `pi05-so101` | `Project-IRA/TPSoSe2026_Pi05_LeRobot_SO101_Finetuning_V7_Full_V2` @ `4b48932cc74a61f685841a4fff467ef31caa9ce1`; checkpoint `008000` | LeRobot PI05Policy; HTTP plus mapped bridge; shared factory selector `pi05-so101` | BF16 + FP32 components; compile off; 10 steps; saved STATE/ACTION MEAN_STD processors; six named absolute targets | `20265907` |
| `groot-so101` | Local validated `GR00T-N1.7-3B-SO101`; checkpoint fingerprint `9bd09a2a40c04637b5d6e010790ac2e3b6cbd75f81c1d59443856e4e37f27c09` | Validated LeRobot GR00T loader/guard; evaluation HTTP plus calibrated trial bridge; not the native GR00T server | BF16 parameters; `model_params_fp32=False`; four flow steps; observer off; 16 decoded actions from the guarded full raw horizon | `20265907` |

The Pi0.5 SO101 checkpoint is an existing community fine-tune, not new training performed here. Its nested path is `outputs_V8/train/pi05_6gpu_fsdp_V2/checkpoints/008000/pretrained_model`. It has six explicitly named action features and saved normalization. Its deployment does not make the original base model physically mapped. Full sources/hashes are in [PI05-SO101-CONFIGURATION.md](PI05-SO101-CONFIGURATION.md).

### Cameras, state and action units

All live trials read the actual front `/dev/video2` and wrist `/dev/video0` streams as RGB 640×480. The arm is `/dev/ttyACM0`. The loaded calibration is `so_follower/my_awesome_follower_arm.json`, SHA-256 `ef68ae670b75d88f57260866653a484f224b1c31fdd8ff1d8e3cf2a1270b6f5b`. No calibration was rewritten.

| Profile | Camera mapping | State/action mapping |
|---|---|---|
| G0.5 | front → exterior; wrist → wrist_right; unused wrist_left placeholder required by native codec | Arm degrees → legacy SO101 reference frame: multiply by `[1,-1,1,1,1,1]`, add `[0,90,90,0,0,0]`; inverse on output. Gripper 0–100. |
| MolmoAct2 | front and wrist, in that declared image order | Same explicit legacy SO101 reference frame; checkpoint normalization/denormalization on server; gripper 0–100. |
| Pi0.5 base | front → base_0_rgb; wrist → left_wrist_0_rgb; right_wrist_0_rgb zero placeholder | Raw six live values padded to 32 for evaluation. No verified physical output mapping. |
| Pi0.5 SO101 | front → desk_view; wrist → wrist_left; no placeholder | Identity degrees on the five arm joints, gripper 0–100; saved six-joint MEAN_STD pre/postprocessors. No axis flip or degree-to-percent conversion. |
| GR00T trial bridge | front/wrist through validated LeRobot preprocessing | Degrees ↔ RANGE_M100_100 using the current calibration span; gripper identity 0–100. |

**GR00T comparison boundary:** the new trial bridge corrects the degree/normalized-value mismatch documented in [UNITS-VERDICT.md](UNITS-VERDICT.md). Historical production native/LeRobot backends were left unchanged. Runs 12–13 therefore verify the explicit calibrated mapping; they are not numerical parity runs against the old deployment.

### Sync versus async

| Path | Actually exercised here | Async status |
|---|---|---|
| G0.5 native bridge | Sync chunks | Explicitly unsupported by this adapter |
| Molmo HTTP bridge | Sync chunks | Explicitly unsupported |
| Pi0.5 base | Sync input-only requests | No arm scheduler exercised |
| Pi0.5 SO101 HTTP bridge | Sync chunks | Explicitly rejected, even though the checkpoint authors have a separate async deployment |
| GR00T HTTP trial bridge | Sync chunks | Explicitly rejected; prior production LeRobot async support is separate |
| GR00T production LeRobot, earlier Phase 8 | Previously tested async pick/retarget/server-loss/voice paths | Implemented and previously validated within amended Phase 8 scope; not rerun in this campaign |

For synchronous runs 1–13, inference finishes before playing that chunk, so the arm can pause between chunks; their “20 Hz” is within-chunk cadence. Trial14 overlaps inference and execution; its reported intervals include both chunk handoffs.

## Every numbered run

All runs used fresh per-run snapshots and no automatic reset pose. The instruction for policy runs was “Grab a banana and put it on the plate.” Synthetic diagnostics commanded only a shoulder-pan offset and are not policy-motion evidence.

“Lead/cap” means maximum target lead over measured pose / total target excursion from that run’s starting pose. Units are degrees for arm joints and 0–100 points for the gripper. All runs retained 0.25°/point command slew per step. The total cap is per joint, not Cartesian travel or a sum across joints; the origin is retained across chunks but renewed for a separately authorized run.

| Run | Policy / diagnostic | Chunks × horizon = commands | Lead / cap | Mean RPC (range), ms | Mean server inference, ms | Max measured arm / gripper | Median / max within-chunk interval, ms | Observation |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | g05-so101 | 1 × 32 = 32 | 0.25 / 5 | 1262.20 (1262.20–1262.20) | 1187.61 | 0.00° / 0.00 points | 50.15 / 50.28 | No visible motion; held safely |
| 2 | g05-so101 | 3 × 32 = 96 | 0.25 / 5 | 1263.19 (1254.88–1276.10) | 1196.37 | 0.00° / 0.00 points | 50.13 / 50.22 | No measured motion; no separate operator reply |
| 3 | g05-so101 | 3 × 32 = 96 | 0.25 / 5 | 1258.29 (1251.09–1263.98) | 1184.54 | 0.00° / 0.00 points | 50.13 / 50.23 | No visible motion; held safely |
| 4 | Synthetic pan +3° | 1 × 40 = 40 | 0.75 / 5 | N/A — synthetic | — | 0.00° / 0.00 points | 50.14 / 50.23 | No measured motion; no separate operator reply |
| 5 | Synthetic pan +3° | 1 × 40 = 40 | 3.75 / 5 | N/A — synthetic | — | 2.64° / 0.00 points | 50.06 / 50.06 | 2.64° measured pan; operator did not notice |
| 6 | Synthetic pan +6° | 1 × 40 = 40 | 3.75 / 6 | N/A — synthetic | — | 5.71° / 0.00 points | 50.06 / 50.21 | Visible and smooth; hold confirmed |
| 7 | g05-so101 | 3 × 32 = 96 | 3.75 / 5 | 1287.45 (1278.51–1296.90) | 1215.33 | 0.00° / 0.00 points | 50.07 / 50.25 | Too small to notice; no measured motion |
| 8 | g05-so101 | 10 × 32 = 320 | 3.75 / 5 | 1289.20 (1274.36–1301.48) | 1218.40 | 0.00° / 0.00 points | 50.14 / 50.30 | No measured motion; small-target result accepted |
| 9 | molmoact2-so101 | 3 × 30 = 90 | 3.75 / 5 | 3747.68 (3728.20–3765.67) | 696.18 | 4.75° / 0.07 points | 50.14 / 55.68 | Measured movement; no individual operator reply |
| 10 | molmoact2-so101 | 3 × 30 = 90 | 3.75 / 5 | 3772.93 (3726.26–3828.80) | 699.20 | 5.10° / 0.00 points | 50.14 / 59.66 | Small visible motion confirmed |
| 11 | pi05-so101 | 3 × 50 = 150 | 3.75 / 5 | 3375.81 (3350.61–3389.46) | 326.95 | 4.57° / 4.74 points | 50.07 / 100.37 | Visible motion confirmed |
| 12 | groot-so101 | 3 × 16 = 48 | 3.75 / 5 | 226.29 (211.04–253.00) | 163.63 | 4.75° / 4.53 points | 50.07 / 50.21 | Measured movement; operator requested repeat |
| 13 | groot-so101 | 3 × 16 = 48 | 3.75 / 5 | 233.70 (211.52–247.86) | 160.62 | 4.66° / 4.80 points | 50.14 / 50.24 | Visible motion confirmed |
| 14 | pi05-so101 RTC, local | 3 × 50 overlapping → 100 | 3.75 / 5 | 433.07 (426.69–442.88) | 417.33 | 4.84° / 4.74 points | 50.15 / 50.23 | Visible, smooth; safe hold confirmed |

All 14 numbered runs completed their command budget without software faults or controller clamp warnings. Intentional projection was active and separately logged: a lack of controller clamp warnings does **not** mean raw model targets were sent unchanged. Mechanical smoothness and safe hold were not separately confirmed in every later operator reply; only the observations actually supplied are recorded.

Run 10 measured 5.10° wrist-roll excursion, slightly beyond the 5° target cap but below the configured stop threshold of cap + 0.5°. Run 11 had one approximately 100 ms command interval; its median remained near 50 ms. These deviations are retained rather than hidden behind a nominal 20 Hz label.

### Why G0.5 did not visibly move

In run 8, raw offsets from measured pose were pan +0.128…+0.435°, lift −0.007…+0.245°, elbow −0.384…+0.005°, wrist flex +0.338…+0.524°, wrist roll −0.462…−0.191°, and gripper −0.098…+0.470 points. These were absolute targets, not accumulating deltas. Only 10 of 320 commands—the first in each chunk—were slew-limited; the other 310 passed unchanged. Increasing from three to ten chunks did not create larger movement. Synthetic runs 5–6 established motor response but are not counted as G0.5 policy success.

## Supporting input-only checks in this campaign

| Check | Deployment | Outputs | RPC times | Motor output |
|---|---|---|---|---|
| Pi0.5 base live inputs | Local RTX 3060 | 3 × (50×32) | 1.034 s / 0.592 s / 0.578 s | Disabled; no SO101 mapping |
| Pi0.5 SO101 mapped live inputs | EC2 L4 through SSM | 3 × (50×6) | 4.498 s / 3.508 s / 3.495 s | Disabled; physical run 11 followed after approval |
| Molmo mapped recorded-input preflight | EC2 L4 through SSM | 1 × (30×6) | 5.521 s | Disabled |
| GR00T calibrated recorded-input preflight | Local RTX 3060 | 1 × (16×6) | 1.141 s | Disabled |

The runner also warmed the model before opening the arm in each numbered run. Warmups, camera retries and startup checks are excluded from physical chunk latency means. The synthetic pan diagnostics did not use model output for their motor targets, even though the shared runner performed a model warmup.

## Earlier frozen-observation GPU/network checks — separate measurements

Each successful location used two warmups plus 12 observations, one per recorded episode and one seed per observation. These are **medians**, not the means above. No arm was commanded. EC2-local and workstation outputs matched bit-for-bit for their matched requests. Input content, hardware and transport differ from the later live trials.

| Model | Chunk | RTX 3060 local RPC | L4 server, EC2-local client RPC | L4 server, workstation/SSM RPC | L4 server generation/inference |
|---|---|---:|---:|---:|---:|
| GR00T SO101 | 16×6 | Not rebenchmarked in that comparison | 172.4 ms | 3139.4 ms | 146.9 ms |
| Pi0.5 base | 50×32 | 446.5 ms, 12/12 | 407.6 ms | 3356.9 ms | 389.2 ms |
| MolmoAct2 SO101 | 30×6 | Weights loaded; first forward OOM; zero successful observations | 737.1 ms | 3650.6 ms | 709.2 ms |
| G0.5 SO101 | 32×6 | 933.7 ms, 12/12 | 2171.1 ms | 4096.7 ms | 1982.5 ms (EC2-local series) |

The roughly three-second HTTP client/server residual includes raw-image JSON/base64 serialization, transfer through SSM, parsing and client work; it is not pure geographic RTT. G0.5 uses a different native request/cache-draining protocol, so these are not transport-normalized model rankings.

G0.5 cold first inference was 21.807 s on RTX 3060 and 36.184 s on L4 in the earlier benchmark, excluded from warm latency. Current campaign startup records show approximately 27.55 s G0.5 model load, 238.40 s Molmo load, 64.40 s Pi0.5 base load and 9.60 s GR00T load; these recorded loader boundaries exclude some imports/downloads/hash work and are not end-to-end task-start delays. The SO101 Pi0.5 model load is retained in its live-response health metadata; checkpoint hash verification is outside that loader timer.

Sources: [MODEL-SWAP-EC2-RESULTS.md](MODEL-SWAP-EC2-RESULTS.md), [GALAXEA-BENCHMARK-RESULTS.md](GALAXEA-BENCHMARK-RESULTS.md).

## Earlier GR00T native/LeRobot and async context

| Earlier check | Configuration | Result |
|---|---|---|
| Matched native vs LeRobot observer toggle | RTX 3060; BF16; four flow steps; one CPU thread; one warmup + three timed predictions per mode; native FlashAttention2 versus LeRobot SDPA | Pipeline medians: native 124.40 ms; LeRobot observer off 132.89 ms; exhaustive observer on 204.33 ms. Excludes RPC and motor playback. |
| Observer-off physical trial | Existing GR00T LeRobot deployment; 20×16 actions; 20 Hz; no fixed inference seed | Mean generation 134.03 ms; mean guarded client call 178.91 ms. Operator: coherent motion, no grasp, no unsafe behavior. |
| Phase 8 async operating-point measurement | GR00T LeRobot/gRPC; lightweight observer; one warmup + 100 warm requests | Median 143.818 ms; empirical p99 150.098 ms; derived deadline 250.098 ms. |
| Earlier async scheduler/integration | 16-action chunks; request at eight remaining; 20 Hz; blend 0.3 old/0.7 new; discard stale contributions and old instruction epochs | Physical pick, retarget, server-loss hold and voice paths accepted within amended Phase 8 scope. Sustained-run/outlier work was deferred. Not rerun for the new HTTP bridges. |

Sources: [INFERENCE-LATENCY-20260912.md](INFERENCE-LATENCY-20260912.md), [OBSERVER-OFF-PHYSICAL-TRIAL.md](OBSERVER-OFF-PHYSICAL-TRIAL.md), [ASYNC-INFERENCE.md](ASYNC-INFERENCE.md). These older configuration differences are why their timings should not be substituted for the current GR00T run-13 mean.

## Reproducibility and remaining coverage

- G0.5 image: `sha256:cbe14b86ede7deefedb3770a578ea9f8cce525edebe0d16232a41ab8ace7b417`.
- Molmo and Pi0.5 base image: `sha256:bd14d39f7e5a391091e806cbf5f47d92e1cd411d2becda49ed65cd697e8e2558`.
- SO101 Pi0.5 and current GR00T image: `sha256:6a1ff7e22a1c4202a68f7ccfd85d6559dcd7d51ea08433461956d7f9d68d34fb` (ECR `dume/model-swap:phase10-pi05-so101-v1`).
- Each numbered workspace contains the exact protocol/source/calibration snapshot, authorization, input arrays, raw/bounded/sent targets, per-chunk metadata, final encoder state and any operator observation. See the CSV for exact workspace paths.
- Current source checks: 54 scoped tests passed for the SO101 Pi0.5 integration; the later GR00T bridge plus existing mapping/guard tests passed 27 focused tests. These are different suites, not an additive test count.
- Phase closeout is complete under the amended scope. Visible motion is an integration result, not a grasp-success evaluation or a blanket certification of all poses/faults; retained gaps are listed in PHASE10-CLOSEOUT.md.

## Machine-readable evidence

[Per-run CSV](verification/phase10/physical-runs.csv) · [Full summary JSON](../corpus/phase10-integration-20260913/summary/summary.json)

The CSV includes command counts, limits, raw timing statistics, maximum arm/gripper excursions, cadence, projection counts, software stop/clamp fields and operator observations. Historical raw run records were not rewritten to add a success claim.
