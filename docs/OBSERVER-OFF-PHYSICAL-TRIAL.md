# Observer-off physical latency trial — 2026-09-12 UTC

Aaron explicitly requested one physical observer-off trial and confirmed he was
beside the arm. The runner completed the same initial/ready reset and 20 chunks
of 16 actions at 0.05 seconds per action. Aaron reported: “Coherent motion, no
grasp, no unsafe behavior.” This is one directional success, not a grasp success.

| Measurement | Result |
|---|---:|
| Mean server chunk generation | 134.03 ms |
| Median server chunk generation | 133.75 ms |
| Generation range, 20 physical chunks | 132.86–137.85 ms |
| Mean client call including identity checks | 178.91 ms |
| Action loop, excluding reset/load/preflight | approximately 20 seconds |
| Recorded clamp warnings / safety stops | 0 / 0 |

One frozen-observation warmup request occurred before controller construction and
is excluded from these timing statistics. The real service completed its normal
six startup checks and SAFE-01 model-load guard. GPU inference used the same
pinned LeRobot image, checkpoint, four flow steps, BF16 model and one CPU thread.
The observer mode was explicitly off, with no fixed inference seed.

Prior observed physical trials averaged 208–209 ms generation; this trial's
134 ms mean is about 36% lower. It also agrees with the approximately 132 ms
observer-off local GPU benchmark. Different live camera frames and ambient noise
mean this physical comparison is not a paired numerical-equality experiment.
The local matched checks separately showed unchanged outputs when toggling the
shared observer.

The shorter total loop must not be attributed entirely to operation observation.
This trial also omits exhaustive attestation completion and its old three
Docker-exec process probes. It checks current read-only container identity,
exact command/environment/source mounts, owned source bytes, checkpoint metadata,
calibration and device identities before/after each request. These are explicitly
reduced diagnostics, not the former exhaustive runtime-precision proof.

The separate `scripts/run_observer_latency_trial.py` enforces one trial, validates
fresh named approval against a current controller/source/service snapshot, hashes
current checkpoint/frozen inputs before release, checks both cameras, warms the
actual socket service before controller construction, and reuses the original
`StopGuardedController`, bus/follower composition, `armed_stop`, `CheckedPolicy`
and `PickSkill`. Calibration and PID readback passed. Signal stops, reset-inclusive
clamps and raw dispatch/cleanup guards remain active. The local operator observed
the trial; no machine-generated score substituted for that observation.

The original safety journal validates with 826 events: 413 dispatches and 413
returns, zero stops/clamps. Approval < preflight < construction < motion < end
chronology passed. The run's `awaiting_operator_observation` record is preserved;
Aaron's later observation and final result are separate immutable records.

Evidence: `corpus/observer-off-physical-20260912/` contains the exact server command,
new approval, preflight, original run/journal, all 21 server timing records
(one warmup plus 20 physical), operator observation and final `trial-result.json`.
Final result SHA256:
`df02d3aba1477159fcc2f4e23d995e1dc31a787168c394d53efe0b0358d95cc9`.

Six new approval/scope tests and two existing raw-stop/cleanup tests passed before
motion. The final checkpoint-metadata addition passed the six approval tests;
actual socket warmup and the physical trial then passed. This one-trial script is
bound to the recorded local service and baseline settings, not a general launcher.

The physical runner exited and disconnected. The test serving container was
stopped after recording Aaron's observation. No second trial was run. Phase 7's
historical source/evidence and completed verification remain unchanged.
