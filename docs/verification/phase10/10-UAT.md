---
phase: 10-physical-embodiment-integration
status: passed
scope: operator-amended bounded integration
---
# Phase 10 UAT

Machine-readable results: `10-UAT.json`. Eight scoped checks passed; three wider qualifications remain deferred. No task-accuracy claim.

| Check | Result | Evidence |
|---|---|---|
| Actual front/wrist streams, six named joints and loaded calibration | passed | trial snapshots, live-observation.npz, docs/PI05-SO101-CONFIGURATION.md |
| Live inputs, native inference and bounded dispatch; no visible policy motion honestly retained | passed | trials1-3,7-8; synthetic4-6 separate |
| Mapped MolmoAct2 bounded synchronous motion | passed | trial10 operator: small but visible motion |
| Base input-only check, anonymous32 outputs never sent to motors | passed | pi05-live-three-chunks |
| Pinned named/normalized community SO101 checkpoint, visible bounded motion | passed | trial11 operator observation |
| Calibrated LeRobot HTTP bridge with visible bounded motion | passed | trials12-13; trial13 operator observation |
| Local asynchronous physical execution, visible/smooth with safe hold | passed | trial14:100 targets;442.88/426.69/429.64ms RPC;50.23ms max interval |
| Deadline/identity/stop/failure handling and bounded target guards | passed | 69 focused tests, closeout/regression.xml; CPU fault checks, not a physical fault campaign |

## Retained deferrals

- [ ] D10-01: Coherent G0.5 target-directed task motion; model-ready-pose and broad per-joint direction/magnitude qualification. **Result: deferred.** Aaron accepted small-target integration evidence and excluded model task accuracy; no full joint-by-joint physical qualification was performed.
- [ ] D10-02: Sustained/endurance, physical RTC fault injection and unattended/voice deployment qualification. **Result: deferred.** Scope is a small bounded RTC trial; software faults and historical GR00T tests do not establish these Pi0.5 operating conditions.
- [ ] D10-03: Explicit second serial-owner rejection test across deployment paths. **Result: deferred.** Campaign used one attended runner; no adversarial second-owner trial or blanket exclusive-ownership guarantee is claimed.

Operator evidence is preserved verbatim in per-trial observation files. Trial14 answered yes to the combined motion/safe-hold question, explicitly describing visible smooth motion.
