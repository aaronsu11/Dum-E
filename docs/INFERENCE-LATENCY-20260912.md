# Inference latency investigation — 2026-09-12

The remaining chunk-generation gap is mainly the runtime observer, not the
GR00T model execution. A fresh paired GPU check reproduced the earlier results
with the serving CPU thread count fixed to one.

| Local timing boundary | Native | LeRobot, observer off | LeRobot, observer on |
|---|---:|---:|---:|
| Median complete chunk pipeline | 124.40 ms | 132.89 ms | 204.33 ms |
| Warm pipeline range | 123.50–124.71 ms | 132.72–132.91 ms | 204.31–206.22 ms |
| Median model call, including any active observer | 118.99 ms | 120.86 ms | 190.59 ms |

The LeRobot observer adds 71.44 ms to the complete pipeline, accounting for
89.4% of its observed 79.93 ms gap to native. Without observation, the remaining
pipeline difference is 8.49 ms, or 6.8%. Model calls differ by 1.87 ms.
These are differences of small-sample medians, not confidence intervals.

## What is doing the work

`docker/lerobot-policy/server.py` wraps chunk generation with
`ServingObservation` when parity attestation is enabled. It installs both
`TorchFunctionMode` and `TorchDispatchMode`, plus backbone/action-encoder hooks.
Callbacks inspect autocast/TF32 state, recursively collect tensor dtypes and
identify attention operations. The measured chunk recorded 11,059 floating
operations, 196 SDPA calls, four flow steps and one noise draw. Those Python
callbacks occur inside the reported generation interval. The observed model
call therefore includes monitoring overhead; 190.59 ms is not pure GPU time.

The paired toggle includes the deployed raw-output capture wrapper as well as
the observation modes/hooks. It isolates their combined cost, not the individual
share of each callback or clone. Most of the measured difference falls inside
the model-call boundary (69.73 ms), consistent with per-operation observation.

At one CPU thread, LeRobot's unobserved stage medians were 1.39 ms for image
conversion, 8.41 ms for preprocessing, 120.86 ms for the model, and 0.18 ms for
decoding. The earlier 20-thread CPU oversubscription has already been fixed at
the serving entrypoint. The earlier seconds-long per-request weight hashing has
also already been replaced by load-time hashing and per-request metadata checks.

## Timing boundaries matter

This benchmark excludes RPC, attestation completion, client host checks, model
loading and robot action playback. Its 204 ms observed pipeline closely matches
the successful physical trials' approximately 208–209 ms generation time.
The prior guarded serving benchmark separately measured approximately 45 ms
server validation and 603 ms complete guarded client calls.

Code inspection explains additional client work: `CheckedPolicy.get_action`
checks host/process identity before and after inference, and
`RuntimeSource.ObservedSession.infer` checks it again after receiving the
response. Each `collect_runtime_host` invokes Docker inspect and Docker exec,
starting a Python interpreter for process identity. That is three host probes
(six Docker CLI invocations) per guarded chunk. Their individual costs were not
remeasured here; the rest of the old 603 ms total cannot be assigned solely to
these probes. Server validation also checks source/file metadata, effective
configuration and publishes request attestation. These are separate from the
chunk-generation difference the user asked about.

## Experiment and reproducibility

Exactly 12 predictions on RTX 3060 12 GiB: four native, four LeRobot bare and
four LeRobot observed. Each mode used one warmup and three timed predictions;
LeRobot modes were interleaved in one loaded process. Both backends used GPU
BF16 parameters, one CPU intra-op thread, four flow steps, fixed observation
`record_0005.npz` and seed 20265907. Native uses its pinned FlashAttention2
runtime; LeRobot uses its pinned SDPA runtime. These remain distinct deployed
configurations, not an attention-only comparison.

CUDA synchronization bounded pipeline and stage measurements. This adds some
instrumentation overhead equally to the LeRobot modes. Full 16×6 decoded outputs
were finite and exactly equal across all repetitions within each backend,
including observer on/off. No cross-backend equality claim is made. One input
and three timed samples per mode are enough for this bounded diagnosis, not a
p95, throughput or task-quality evaluation.

Raw results, exact pinned Docker argument arrays, worker source and logs:
`corpus/latency-observer-20260912/`. The worker and launcher are `worker.py` and
`run.py`; per-backend JSON includes relevant source hashes. The launcher is an
archived recipe: copy it to a fresh output directory before rerunning to preserve
these measurements. Containers had no network or robot-device mounts, and ran
sequentially. Both exited successfully. No production source was edited, no
service was started, and no robot motion occurred.

## Optimization direction

The measured target for ordinary LeRobot chunk generation is approximately
133 ms under these conditions. Reaching it requires separating exhaustive
per-operation parity diagnostics from the normal serving hot path. A candidate
implementation would retain load-time identity/profile checks, lightweight
request shape/finite checks, sampling/flow-step observations and the existing
motion safety guards, while giving exhaustive operation tracing an explicit
diagnostic mode. The lost per-operation drift detection must be documented and
the replacement checks reviewed; the present attestation schema requires those
observations and cannot simply accept their absence.

A separate optimization can reduce Docker subprocess overhead while preserving
instance-change detection. Neither change was implemented by this investigation.
The Phase 7 source-bound closeout and its verified evidence remain unchanged.
