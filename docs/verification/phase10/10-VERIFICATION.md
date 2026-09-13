---
phase: 10-physical-embodiment-integration
status: passed
scope: operator-amended bounded integration; Pi0.5-only async
original_GX_05: deferred
review: inline; no independent reviewer claimed
---
# Phase 10 verification

Phase complete under Aaron's amended scope, 2026-09-13. This is not completion of original coherent G0.5 task behavior or the entire milestone.

## Accepted results

- 14 physical command runs: 11 policy runs and three synthetic diagnostics; 1,286 targets total. Synthetic movement is not model-performance evidence.
- Four explicit model profiles/checkpoints exercised with actual front/wrist/state inputs. Pi0.5 base remains input-only; the separately pinned community SO101 checkpoint supplies six named outputs and saved normalization.
- G0.5 live inference/dispatch passed with small targets and no measured visible policy motion; Aaron accepted that limitation and moved on. MolmoAct2, mapped Pi0.5 and calibrated GR00T produced operator-confirmed visible motion.
- Only Pi0.5 async is required. Local RTC trial14 completed100 targets, two nine-tick handoffs, zero errors/clamp warnings; Aaron confirmed visible smooth motion and safe hold. Peak arm excursion4.835 degrees, gripper4.736 points; maximum interval50.230ms.
- All69 focused regression tests pass. Deadline-installation gap fixed; see 10-CODE-REVIEW.md. No new physical test at closeout.

## Scope reconciliation

The older roadmap required coherent spoken-target G0.5 behavior from its dataset-derived ready pose, an all-joint in-air qualification, explicit second-owner rejection, and full-task memory/voice behavior. Aaron changed the objective to embodiment/modalities/inference plumbing, excluded accuracy/fine-tuning, accepted G0.5's small-target outcome, and limited new async trials to Pi0.5. Those original criteria are archived verbatim in10-ORIGINAL-CRITERIA.md. They are not retroactively labeled passed. GX-05 stays unchecked; wider gaps have explicit deferred results in10-UAT.json.

## Evidence

- docs/PHASE10-INFERENCE-TRIAL-SUMMARY.md and corpus/.../summary/physical-runs.csv: all14runs, exact pins/configs, raw versus limited commands, memory and latency.
- docs/PI05-SO101-CONFIGURATION.md: source-backed checkpoint semantics and calibration distinction.
- docs/PI05-RTC-LOCAL.md: local model comparison, dummy HTTP scheduler checks and physicaltrial14.
- Per-trial immutable snapshots, approvals, result.json and operator-observation.json. Historical results remain unchanged; observations and closeout decisions are separate records.
- 10-UAT.md/JSON, 10-CODE-REVIEW.md and 10-01/02/03-SUMMARY.md.

The working tree contains the implementation; no commit/push is implied. Local Pi0.5 server was left resident after the attended trial; no hardware runner was started at closeout. EC2 settings were not changed. Milestone audit and existing deferrals remain separate work.
