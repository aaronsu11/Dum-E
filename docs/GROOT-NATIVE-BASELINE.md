# `groot-native` Live Pick Baseline

**Status:** Recorded. Standing baseline — not a changelog entry.

This document records the live pick score for the `groot-native` policy backend on the upgraded
`lerobot` 0.6.1 stack, taken after the dependency bump and before any policy migration. It is the
comparison point the checkpoint parity gate is measured against, and it is the resolution
referenced by requirements LR-06 and BACK-06.

The harness that produced every number below is `scripts/run_pick_baseline.py`. The sibling
resolution document for the verified stack this baseline is attributable to is
`docs/UNITS-VERDICT.md`; its live-confirmation section holds the units, PID, calibration and
clamp measurements taken on the same arm.

---

## 1. The result

> **Score: 9 of 10 successful attempts. Number of runs performed: 1.**
> **Total clamp warnings across the run: 0.**

Date: 2026-09-07. One scored run was performed and this is that run — the count of runs performed
is recorded here precisely so that a single reported score can never be a best-of-N. No attempt
series was discarded, and no result was re-sampled to obtain a better number.

### 1.1 The pinned instruction, verbatim

```
Grab a banana and put it on the plate
```

This string is byte-identical to the `task` field of the frozen corpus's source dataset, so the
live smoke check and the offline numerical instrument were conditioned on the same language. It
was pinned across all ten attempts (the recorded run carries exactly one distinct instruction
value across the series).

### 1.2 The success rule this score was judged against

> **Success = the arm grasps the object and lifts it clear of the table.** Retaining it, and
> placing it on the plate, are **not** required.

This rule was set by the operator before the series and applied identically to all ten attempts.
The stated rationale: the checkpoint is not fine-tuned for this table and this scene, so some
performance degradation is expected; behaviour that moves in the right direction with the right
shape counts as a pass, and the same rule will be applied to every other model evaluated against
this baseline so that the comparison stays fair.

**This rule is more permissive than the instruction the policy was conditioned on.** The
instruction asks for a pick *and* a place; the score measures the pick alone. Any later reader
comparing against this number must apply the same rule, and must not read it as evidence that a
completed pick-and-place succeeded 9 times out of 10. That gap is precisely where a forgiving
tabletop task hides defects (§4).

---

## 2. Per-attempt record

| # | Result | Clamp warnings | Duration | Exception | Operator note |
|---|--------|----------------|----------|-----------|---------------|
| 1 | SUCCESS | 0 | 23.2 s | — | — |
| 2 | SUCCESS | 0 | 23.2 s | — | — |
| 3 | SUCCESS | 0 | 23.1 s | — | — |
| 4 | SUCCESS | 0 | 23.4 s | — | — |
| 5 | SUCCESS | 0 | 23.0 s | — | — |
| 6 | SUCCESS | 0 | 23.4 s | — | — |
| 7 | **FAILURE** | 0 | 23.1 s | — | Ran out of time |
| 8 | SUCCESS | 0 | 23.4 s | — | — |
| 9 | SUCCESS | 0 | 23.3 s | — | — |
| 10 | SUCCESS | 0 | 23.4 s | — | — |

No attempt raised. The series was not voided.

Every attempt consumed its full budget of 20 obs-to-policy-to-action iterations at an action
horizon of 16, which is why the durations are near-identical: the loop always exhausts its
iteration budget, and success depends on whether the task completed inside it. Attempt 7's
failure was therefore a **timeout, not a misbehaviour** — the arm was still working the task when
the budget ended.

The iteration budget was raised from 10 to 20 before this run, after a validation attempt
grasped the object, dropped it, and stopped before it could retry. The action horizon was left at
16 because that value matches the trained checkpoint and tuning it would change what the baseline
measures rather than how long the arm gets.

---

## 3. The clamp-warning total, and what it means

**The total across all ten attempts is 0.** The per-step motion clamp never engaged.

This is the required outcome, and it is a real measurement rather than an absence of
instrumentation: the counter was self-tested at the start of the run by emitting a clamp-worded
warning on the stdlib **root** logger and confirming it reached the counting sink, which
exercises both the sink and the standard-library-to-loguru bridge in one check. The synthetic
probe is excluded from every reported count.

**A non-zero total here would have had two consequences, and neither applies:**

1. **The parity gate's zero-warning requirement would be at risk.** The gate expects a clean run
   with no clamp engagement; warnings during the baseline would mean the baseline itself was
   taken on motion the clamp was truncating, so any later comparison would be against
   already-modified trajectories.
2. **The clamp value would need re-deriving.** `max_relative_target` is 160.0, derived
   arithmetically from the checkpoint's cumulative per-timestep relative-action extreme
   (137.47269 x 1.15), not measured on this arm. Warnings on nominal motion would mean the
   derivation was too tight and the value had to be re-derived from observed deltas.

Because the total is 0, the clamp is confirmed as non-interfering on nominal motion while still
demonstrably functional — the clamp was separately shown engaging on a deliberately oversized
delta, clipping a 1.5x request to exactly 160.0000, as recorded in the sibling document.

A note on why no warning fired even on the pose resets: the reset path was changed before this
run to reach the retracted ready pose *before* descending to the low initial pose, so the
largest previously-predicted legitimate clamp delta (a reset to initial from an extreme policy
pose) no longer occurs as a single unbounded move.

---

## 4. The caveat — load-bearing, not boilerplate

> **This is a functional smoke check. It is not a controlled numerical comparison.**

The governing decision pins **only the instruction string**. Scene variation between attempts is
explicitly accepted and was not controlled: the object was repositioned by the operator between
attempts and no attempt reproduces another's scene. Consequently **the entire numerical burden
for the parity gate rests on the offline corpus evidence, not on this score.**

Two prohibitions follow, and they bind any later phase:

1. **Do not cite this score as evidence of numerical parity.** It is a count of operator
   judgments on an uncontrolled scene, under the permissive success rule in §1.2.
2. **Do not read a live discrepancy against this score as proof of policy drift.** An
   uncontrolled scene can explain a difference of one or two attempts on its own, and this run
   already contains a timeout that a slightly different object placement would plausibly have
   avoided.

**A forgiving tabletop task can hide real defects.** Each of the following can still sometimes
get the fruit off the table, and so can still pass the rule in §1.2:

- a joint bias,
- a gripper-aperture error,
- a late-chunk compression,
- a swapped camera,
- a cropped or shifted field of view.

This is exactly why the caveat is load-bearing. A 9-of-10 here constrains very little on its own;
it establishes that the integrated path runs end to end on real hardware and produces
directionally correct behaviour.

**The instrument that carries the numerical burden is the frozen observation-to-action corpus,
and its gate is `scripts/verify_frozen_corpus.py`.** That verifier fails loudly on an absent,
short, or malformed corpus, and replays every record with pickle loading disabled. Any parity claim
must be made against that instrument.

**Known limit of that gate — corrected after a security audit.** An earlier version of this section
claimed the verifier "checks the recorded producer so a mock-produced corpus can never be cited as
v1.0 evidence". **That overstated what the code does, and the claim is withdrawn.** The producer
check tests a *self-declared* `--server-label` (default `gr00t:latest`) against a list of
mock-ish substrings; it is never cross-checked against the container that actually served the
requests. Array validation covers dtype and shape only — there is no degeneracy, variance or
non-zero check — so an all-zeros corpus at the correct shape would pass all eight checks.

The corpus shipped here **is** genuine: it was captured against the running `gr00t:latest`
container, and its action samples are non-degenerate (mean ≈ 5.2–8.7, std ≈ 33–68 across records,
no all-zero record). The artifact is sound; the mechanism that was supposed to guarantee it is not.
Closing the gap means recording the container image digest at capture time and failing the verifier
on an unverified producer, and/or adding a non-degeneracy check. Tracked as threat T-05-09.

---

## 5. The attributable stack

Everything below was read during the same run that produced the score, so the number and the
stack it came from cannot drift apart.

| Property | Value |
|----------|-------|
| Selected backend | `groot-native`, obtained through the selector allowlist |
| Resolved backend class | `embodiment.so_arm10x.controller.Gr00tRobotInferenceClient` |
| `lerobot` version | `0.6.1` |
| `torch` version | `2.7.1` (CUDA 12.6) |
| Policy server container image | `sha256:e263056fffe7a60a7f48b6309a8b8f2fb3ea9f8f2afa9c94a0105ed5b7d2eeaf` (ref `gr00t`, state `running`) |
| PID read-back, all six motors | P 10 / I 0 / D 5 — exact match to the declared preset |
| Calibration file | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json` |
| Calibration checksum (sha256) | `ef68ae670b75d88f57260866653a484f224b1c31fdd8ff1d8e3cf2a1270b6f5b` |
| Units mode in effect | `use_degrees=True` on the bus; see the sibling document's live-confirmation section |
| Iteration budget / action horizon | 20 / 16 |

The backend was obtained from the selector rather than constructed directly, so this run also
demonstrates that the `groot-native` path is selectable *and* functional end to end, not merely
importable.

### 5.1 Calibration provenance — this baseline and the sibling document cite different files

**The arm was deliberately recalibrated by the operator between the validation attempt and this
scored run.** The consequence must be stated rather than glossed:

| Artifact | Calibration checksum |
|----------|----------------------|
| The sibling document's live-confirmation measurements | `5bd471fbbb4e1be0c6ede80472d365b527bc0808befdd67f123148ed49e3dc50` |
| **This baseline** | `ef68ae670b75d88f57260866653a484f224b1c31fdd8ff1d8e3cf2a1270b6f5b` |

So the units, envelope and scale measurements in the sibling document were taken against the
*earlier* calibration, and this score was taken against the *later* one. Concretely,
`elbow_flex.range_max` moved from 3090 to 3100 across the recalibration.

That matters for one specific citation: the sibling document's strongest live datum is the
`elbow_flex` at-limit fingerprint, which reads exactly 100.0000 percent at raw tick 3090 because
3090 was that joint's calibrated maximum. **Against the current file that tick is no longer the
limit.** The reading was correct when taken and remains valid as a historical measurement, but
re-running the probe now will produce different raw ticks, and anyone re-deriving that fingerprint
must use the calibration in force at the time.

The units *verdict* is not weakened by this. Its argument does not depend on that one tick: the
plus-or-minus-100.0 clip fingerprint, the reachability falsification, and the training-dataset
cross-check are all structural and survive a recalibration. But a reader comparing the two
documents number-for-number needs to know they were measured against different calibration files.

---

## 6. Limits

1. **One run, nine of ten, under a permissive rule.** The score is not a distribution and no
   confidence interval is claimed. Ten attempts is more than a smoke check strictly needs, and a
   smaller count would carry the same weight.
2. **The single failure is a timeout, not a diagnosis.** "Ran out of time" says the budget ended
   before the task did. It does not distinguish a slow-but-correct policy from one that had
   stalled, and no per-step trace was captured to tell them apart.
3. **Scene uncontrolled by design.** See §4.
4. **Success judged by a human, by design.** Whether the object left the table is a human
   judgment; there is deliberately no flag that supplies one, because a machine-supplied judgment
   would fabricate the number the parity gate is measured against.
5. **Attributable to a recalibration that post-dates the stack verification.** See §5.1.
