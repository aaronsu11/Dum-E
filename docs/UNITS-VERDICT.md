# Normalization Units Verdict

**Status:** Resolved offline (§1-§8), then **CONFIRMED on hardware (§9)**. Standing engineering
resolution — not a changelog entry.

This document records which joint-value convention the GR00T checkpoint
`GR00T-N1.7-3B-SO101` was trained in, the evidence for it, the requirement mechanism it had
to replace, and the limits of what has actually been established. It is the resolution
referenced by requirements PAR-04 and PAR-06.

The harness that computes every number below is `scripts/pose_sweep_units_probe.py`; the
always-running gate over the same numbers is `tests/test_units_verdict.py`. Figures in this
document are cited by harness function name so the document and the code cannot drift apart
silently.

---

## 1. The verdict

> **The checkpoint's `state` and `action` spaces are `MotorNormMode.RANGE_M100_100` —
> equivalently, `use_degrees=False`.**

Dum-E currently runs `use_degrees=True`, so it feeds the policy joint values scaled by a
per-joint factor. That is a real latent mismatch in v1.0, not a coincidence; §4 explains why
the system nonetheless works, and §7 explains why this phase records the verdict without
flipping the setting.

---

## 2. Three independent arguments

### 2.1 The plus-or-minus 100.0 clip fingerprint — strongest

`clip_fingerprint_count()` finds **7** distinct (joint, bound) pairs whose magnitude is
exactly 100.0, within a tolerance of 1e-9, across the checkpoint's `state` and `action`
`single_arm` min/max arrays:

| # | Array | Joint | Bound | Value |
|---|-------|-------|-------|-------|
| 1 | `state.single_arm.min` | `shoulder_lift` | min | `-100.0` |
| 2 | `state.single_arm.max` | `elbow_flex` | max | `100.0` |
| 3 | `state.single_arm.max` | `wrist_flex` | max | `100.0` |
| 4 | `action.single_arm.min` | `shoulder_lift` | min | `-100.0` |
| 5 | `action.single_arm.min` | `wrist_roll` | min | `-100.0` |
| 6 | `action.single_arm.max` | `elbow_flex` | max | `100.0` |
| 7 | `action.single_arm.max` | `wrist_flex` | max | `100.0` |

`RANGE_M100_100` produces exactly plus-or-minus 100.0 **by construction**: it clamps the raw
encoder tick to the calibrated range and maps those endpoints onto a 200-wide span offset by
-100, so the endpoints land on exactly -100 and +100. The `MotorNormMode.DEGREES` branch has
**no clamp and no plus-or-minus 100 boundary of any kind** — it is
`(tick - midpoint) * 360 / 4095`, unbounded and calibration-span-dependent. Seven statistics
landing on exactly plus-or-minus 100.0 is the signature of a clipping function, not an
artefact of a degree scale.

**This is the primary argument because it rests on no assumption** — in particular, not on
which calibration file was in use at training time.

### 2.2 The `elbow_flex` falsification

Computing each joint's reachable degree range from the real calibration file
(`degrees_reachable_range()`, using `midpoint = (range_min + range_max) / 2` and a servo
resolution of 4095):

| Joint | min tick | max tick | reachable degrees | checkpoint `q99` | reachable? |
|-------|---------:|---------:|------------------:|-----------------:|------------|
| `shoulder_pan` | 792 | 3443 | ±116.53 | 44.049 | yes |
| `shoulder_lift` | 851 | 3211 | ±103.74 | 54.045 | yes |
| **`elbow_flex`** | **898** | **3090** | **±96.35** | **100.000** | **NO** |
| `wrist_flex` | 926 | 3219 | ±100.79 | 98.611 | yes |
| `wrist_roll` | 148 | 3965 | ±167.78 | -14.248 | yes |

`elbow_flex` cannot physically reach 100.0 degrees on this arm — its full mechanical span is
±**96.35** degrees. Yet the checkpoint records `max = 100.0` and `q99 = 100.0`. Under
`RANGE_M100_100` that is simply the joint driven to its calibrated upper limit. Under
`DEGREES` it is impossible. It is also the *only* joint that falsifies, which is what makes
this a pointed argument rather than a wholesale mismatch.

### 2.3 The `wrist_roll` cross-check against the training dataset

The training dataset's episode-0 `observation.state` statistics put `wrist_roll` in a narrow
band — `[-58.952, -50.564]`, mean `-56.201`, standard deviation `2.885` — a joint the
operator held essentially fixed.

Dum-E hardcodes `wrist_roll = -90.0` in every pose. Converting that same physical pose into
the percent convention: `-90.0 / 1.67780 = ` **`-53.64`**.

- **`-53.64` lands inside the dataset band**, 2.6 units from its mean.
- **`-90.0` is more than 10 standard deviations outside it** (11.7 sigma, by
  `wrist_roll_cross_check()`).

The two hypotheses give different answers here. That is what makes it evidence.

---

## 3. Why PAR-04's own mechanism was replaced

PAR-04 specifies: *"asserting the observed ready-pose state falls inside the checkpoint's
`state` q01/q99 envelope."* Run exactly as written, on all five arm joints:

| Pose | Joint | As degrees | Inside envelope? | As percent | Inside envelope? |
|------|-------|-----------:|:----------------:|-----------:|:----------------:|
| ready | `shoulder_pan` | 0.00 | yes | 0.00 | yes |
| ready | `shoulder_lift` | -90.00 | yes | -86.76 | yes |
| ready | `elbow_flex` | 75.00 | yes | 77.84 | yes |
| ready | `wrist_flex` | 75.00 | yes | 74.41 | yes |
| ready | `wrist_roll` | -90.00 | yes | -53.64 | yes |
| **initial** | **`shoulder_lift`** | **-102.00** | **no** | **-98.33** | **yes** |

**The ready pose passes under BOTH conventions on all five joints.** PAR-04's assertion
therefore returns PASS regardless of the truth, and encoding it as the resolution mechanism
would let this phase close with a wrong verdict recorded as evidence. That is a silent-pass
hazard in the requirement itself.

The four replacement discriminators, in order of strength:

1. **The plus-or-minus 100.0 clip fingerprint** (§2.1) — arm-free, assumption-free,
   deterministic, re-runnable.
2. **The `elbow_flex` degrees falsification** (§2.2) — arm-free.
3. **The `wrist_roll` dataset cross-check** (§2.3) — arm-free; the joint with the largest
   lever arm and therefore the most sensitive offline discriminator available.
4. **The envelope check against the `initial` pose** — retained *only* against `initial`,
   whose `shoulder_lift` of -102 sits just outside the `q01` of `-99.743` read as degrees and
   just inside it at `-98.33` read as percent. One step either side of that boundary behaves
   differently, unlike the ready pose.

`tests/test_units_verdict.py` asserts the ready-pose result **as a documented negative** —
proof that the mechanism does not discriminate — never as the verdict's evidence.

---

## 4. The real convention seam (roadmap criterion 2a, answered by a corrected question)

### 4.1 Part (a): the upgrade delta is identity

The `_normalize` body in `lerobot/motors/motors_bus.py` is **byte-identical** between lerobot
0.3.3 and 0.6.1, and `MotorNormMode` is identical in both. Given the same calibration file and
the same `use_degrees`, the same physical pose therefore yields the identical joint vector.

**The 0.3.3-to-0.6.1 upgrade introduces no per-joint sign, offset or scale change at the robot
layer. The delta is `identity`.** `test_upgrade_normalization_delta_is_identity_across_swept_ticks`
re-checks this mechanically: for every raw tick across each joint's calibrated span — and past
it, where the clamped and unclamped branches diverge from each other — the two versions'
transcribed formulas return bit-identical values in all three modes. If a pose sweep ever
shows anything other than identity across the bump, the cause is the calibration file, not the
version change.

### 4.2 Part (b): the delta that actually matters is a pure per-joint scale

Both formulas are affine in the raw tick and — because **every motor has drive mode 0** — share
the same centre `midpoint = (range_min + range_max) / 2`:

```
RANGE_M100_100 :  (tick - min) / (max - min) * 200 - 100   ==   (tick - midpoint) * 200 / (max - min)
DEGREES        :  (tick - midpoint) * 360 / 4095
```

Dividing gives `degrees = percent * (max - min) * 360 / (4095 * 200)`. **There is no sign flip
and no additive offset** — the seam is a single multiplicative factor per joint
(`deg_per_pct_table()`):

| Joint | span (ticks) | `deg_per_pct` | `pct_per_deg` | error if the conventions are swapped |
|-------|-------------:|--------------:|--------------:|-------------------------------------:|
| `shoulder_pan` | 2651 | 1.16527 | 0.85817 | **+16.53 %** |
| `shoulder_lift` | 2360 | 1.03736 | 0.96398 | +3.74 % |
| `elbow_flex` | 2192 | 0.96352 | 1.03786 | -3.65 % |
| `wrist_flex` | 2293 | 1.00791 | 0.99215 | +0.79 % (smallest divergence) |
| **`wrist_roll`** | **3817** | **1.67780** | **0.59602** | **+67.78 % (largest divergence)** |

The **gripper is excluded**: upstream hardcodes it as `MotorNormMode.RANGE_0_100` rather than
gating it on `use_degrees`, so it is `RANGE_0_100` under both settings and its value needs no
conversion in either direction. That also means the gripper arithmetic in
`release_at_remote_pose` is already convention-independent.

This table explains why v1.0 works anyway: on `wrist_flex` (+0.79 %), `elbow_flex` (-3.65 %)
and `shoulder_lift` (+3.74 %) the two conventions are near-interchangeable, and those are the
joints that do the picking. `wrist_roll` is off by 68 % but is held at a constant value, so its
relative-action deltas are near zero and the error never accumulates. `shoulder_pan` at
+16.53 % would show as lateral overshoot.

### 4.3 The corrected question, stated as such

Roadmap criterion 2a asks for *"the v2.1 to v3.0 convention delta measured per joint"*.
**Criterion 2a is satisfied here by a corrected question:** the delta it names is provably
identity (§4.1), and the delta that actually matters is the degrees-to-percent scale above
(§4.2). This is recorded as a *correction of a wrong premise*, not as a criterion quietly
redefined — the original question has a real answer, and that answer is "no change".

**The scale magnitudes were derived by arithmetic here, and are MEASURED in §9.5.** They are
exact given a verified formula and a verified calibration file, and the raw-tick round-trip probe
(`units_verdict()` / `active_mode_from_raw_tick()` / `live_pose_sweep()`) has since reproduced
every one of them on the live bus to five decimal places at three independent poses. The
gripper's convention-independence is confirmed live too. **No longer pending hardware.**

The scale transform is deliberately **not implemented** anywhere in the runtime in this phase.
It is documented here precisely so a later phase does not inherit it unexamined.

---

## 5. PAR-06's designated derivation source has no content

PAR-06's primary method was to derive the transform from upstream's dataset conversion code.
`lerobot/scripts/convert_dataset_v21_to_v30.py` exists at the pinned version (588 lines), and
its documented job is, verbatim: generate per-episode stats, check consistency between the new
and old stats, remove the deprecated aggregate stats file, update the codebase version in
`info.json`, and push the result. A case-insensitive search of that whole file for
`sign|offset|degree|radian|deg2rad|* -1|np.pi|90` returns **no matches**.

**The v2.1-to-v3.0 conversion is a metadata and statistics restructuring. It performs no value
transform of any kind — not for dataset files, and not for live robot readings.** The
designated derivation therefore has nothing to derive from, and this resolution uses the
**documented empirical fallback** instead: the three arguments in §2, executed entirely
offline. PAR-06 itself authorizes that escalation when its primary method turns out to have no
content, so this is the documented fallback rather than a substitution.

---

## 6. The Galaxea cross-check: inapplicable, and why

The roadmap asked for Galaxea's signs/offsets fixup as a **cross-check, not a value to copy**.
Its result:

```python
_SIGNS   = [ 1, -1,  1,  1,  1,  1]
_OFFSETS = [ 0,  90, 90,  0,  0,  0]   # degrees
```

**Both sides of that transform are degrees.** Its own docstrings say so twice
(*"lerobot v3.0 degrees to training v2.1 degrees"*), the offsets are annotated `(degrees)`, and
its home-pose comments cite model-frame magnitudes of `124.3` and `121.5` — values far outside
plus-or-minus 100, which `RANGE_M100_100` cannot produce.

So the fixup bridges two *degree* frames with a different zero point on two joints and one
inverted axis. It is a **kinematic frame-convention** difference between Galaxea's own rig and
what LeRobot reports — not a units conversion, and not evidence of a lerobot v2.1-to-v3.0 value
transform, despite the label.

**Outcome: INAPPLICABLE.** Its algebraic form (sign plus additive offset, in degrees) is
structurally incompatible with Dum-E's actual seam (a pure multiplicative scale between degrees
and percent, with no sign flip and no offset). **Neither its values nor its shape are copied.**
Reaching for a `[1,-1,1,1,1,1]` / `[0,90,90,0,0,0]` fixup in this codebase would be applying a
transform of the wrong algebraic form.

---

## 7. The deferred flip, and what it would cost

The correct value is `use_degrees=False`. **This phase does not flip it.** It records the
verdict, and it lands the setting as an explicit configuration key at its **current effective
value** so that a later flip needs no code edit.

Flipping in this phase would cost:

- **Four hardcoded pose vectors get reinterpreted as percentages.** They are literals in the
  controller carrying a comment stating that the targets mirror legacy degree behaviour — the
  unit convention is encoded implicitly, so the flip silently changes what they mean.
- **The initial pose's `shoulder_lift` of `-102` would be clipped to `-100`** by the percent
  mode's bounded-value clamp.
- **`wrist_roll = -90` would become a physically different pose** — `-90` percent rather than
  the `-53.64` percent that `-90` degrees actually is, moving the wrist roughly 60 degrees from
  where the policy was trained.
- **The live re-baseline would be invalidated** and would have to re-run after the flip.

If the flip is taken, every pose must convert in the same commit. The conversion is
`percent = degrees / deg_per_pct`:

| Pose | As degrees (today) | Equivalent `RANGE_M100_100` |
|------|--------------------|------------------------------|
| `initial` | `[0.0, -102.0, 96.0, 76.0, -90.0]` | `[0.00, -98.33, 99.64, 75.40, -53.64]` |
| `ready` | `[0.0, -90.0, 75.0, 75.0, -90.0]` | `[0.00, -86.76, 77.84, 74.41, -53.64]` |
| `remote` | `[0.0, 0.0, 0.0, 50.0, -90.0]` | `[0.00, 0.00, 0.00, 49.61, -53.64]` |

**This table is a starting hypothesis to validate with the raw-tick probe, not a set of final
values.** It depends on the calibration file currently on disk; recalibration changes it. The
gripper column is omitted because it needs no conversion (§4.2).

**The flip is a later-phase decision**, to be taken against offline parity evidence rather than
alongside this record.

---

## 8. Limits and assumptions

1. **The training-time calibration is assumed to be the file now on disk.** If the dataset was
   recorded and the checkpoint fine-tuned against a different calibration with a wider
   `elbow_flex` span, `100.0` degrees could become reachable and §2.2 weakens. **The clip
   fingerprint (§2.1) does not depend on this**, so the verdict survives with two of three
   arguments intact.

   `check_pinned_constants_against_local_artifacts()` compares the pinned constants against
   the local artifacts and reports any difference as a **failure**, so a recalibration or a
   checkpoint swap surfaces instead of silently invalidating the scale table. **It is currently
   FAILING, correctly, and that is the honest state of this limit rather than a defect.** The
   arm was recalibrated after §9's measurements were taken (§9.1), so every joint's tick range
   now differs from the pinned table, and the check says so on every run:

   ```
   [5/5] pinned-constants drift against local data artifacts ...
          DRIFT calibration elbow_flex.range_max: pinned 3090 != recorded 3100
          ... (all six joints)
     FAIL: a resolved artifact drifted from the pinned constants — the units verdict
           must be RE-DERIVED before proceeding
   ```

   **What that failure does and does not mean.** It means §4.2's / §9.5's per-joint scale table
   and §7's conversion table are keyed to a calibration no longer on disk, and must be
   re-derived before either is used to command anything. It does **not** weaken the verdict
   itself: §2.1's clip fingerprint is a property of the checkpoint statistics alone, and those
   are checked in the same run and still match. The failure also now blocks the probe's whole
   arm half — the pre-motion reachability guard is computed from the live `bus.calibration`, but
   the scale table it would be measured against is not, so nothing derived from the pinned
   constants may be measured or commanded until they are re-derived.

   **Historical note on why the guard was worth fixing.** Until this was corrected, the check
   built the calibration path from Dum-E's `robot_type` (`so101_follower`) while `lerobot`
   0.6.1 derives that directory from the robot CLASS's name (`so_follower`). It therefore read
   the stale pre-0.6.x copy, which had not changed, and reported "no drift" through the entire
   recalibration. The guard was present, green and inert. It now resolves the path through
   `controller.resolve_calibration_file()` — the same derivation the bus uses.
2. ~~**The per-joint scale factors are exact arithmetic on verified inputs, not measured on
   hardware.**~~ **DISCHARGED — see §9.5.** The raw-tick round-trip probe measured all five
   factors on the live bus, at three independent poses, and reproduced the derived table of §4.2
   exactly to five decimal places. Research assumption A2 is closed. The conversion table in §7
   remains a *starting hypothesis for the flip* for a different reason — it depends on the
   calibration file currently on disk, and recalibration changes it — but its scale column is no
   longer unverified arithmetic.
3. **The `wrist_roll` cross-check assumes the dataset's operator held the wrist near where
   Dum-E holds it.** If the recording used a materially different wrist angle, the
   `-53.64`-inside-the-band coincidence is weaker evidence. §2.1 and §2.2 are independent of
   this assumption.
4. **Only episode 0 was inspected in per-episode detail.** The corpus-wide checkpoint
   statistics corroborate its `wrist_roll` concentration, but the per-episode spread across the
   remaining episodes has not been measured. The dataset statistics are also remote, so the
   drift check cannot verify them offline — they are the one pinned constant in the harness that
   is not locally re-checkable.

---

## 9. Live confirmation

**Status: the offline verdict is CONFIRMED on hardware.** Everything below was measured on the
physical arm behind the hardware-attach gate, on a bus running `lerobot` 0.6.1. Where a number
was previously derived by arithmetic and is now measured, both appear side by side with their
provenance; nothing derived has been silently relabelled as measured.

### 9.0 The corrected mechanism, stated as such

Roadmap criterion 2 asks that parking the arm at fixed poses report *"the same joint vector to
under 0.5 degrees per joint before and after the upgrade"*. **A literal pre-upgrade live reading
is unobtainable:** no serial device was present at any point before the version bump, and the
bump had already landed by the time hardware was attached. Rather than drop the comparison or
present a one-sided measurement as if it were two, it was taken in **one live run**:

1. Read raw encoder ticks with `normalize=False`. That keyword is present and keyword-only in
   **both** 0.3.3 and 0.6.1, which is exactly why one probe is valid on both sides of the bump.
2. Compute what the **pre-upgrade** stack would have reported from those same ticks. This is
   *exact*, not approximate: the `_normalize` body is byte-identical between the two versions
   (§4.1), and `test_upgrade_normalization_delta_is_identity_across_swept_ticks` re-proves that
   mechanically across every raw tick in each joint's calibrated span.
3. Compare against what the upgraded stack actually reports for the same registers.

**This is at least as strong as two separate live runs, and strictly less noisy**, because both
sides come from the *same* physical pose and the *same* tick read — a two-run comparison would
carry servo read noise and re-parking error on top of any real delta. It also yields something a
before/after diff cannot: it identifies *which* normalization mode is active, by proof rather
than inference. This is a comparison taken a better way, **not** a criterion quietly redefined.

### 9.1 Provenance of the run

| Item | Value |
|------|-------|
| Harness | `scripts/pose_sweep_units_probe.py --pose-sequence initial,ready,remote` and `--demo-clamp` |
| Client stack | `lerobot` 0.6.1, `torch` 2.7.1 + CUDA 12.6 |
| Calibration file | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json` |
| Calibration `checksum` (sha256) at time of run | `5bd471fbbb4e1be0c6ede80472d365b527bc0808befdd67f123148ed49e3dc50` — **superseded, see below** |
| Effective controller config | `use_degrees=True`, `max_relative_target=160.0` |
| Poses swept | `initial`, `ready`, `remote` (3), plus an as-found reading taken before any motion |
| Raw numeric output | the gitignored `corpus/pose_sweep_<timestamp>/results.json`; this section is the committed record |

**The arm was recalibrated by the operator on 2026-09-07, after this run.** The path above is
correct — it is the file `lerobot` 0.6.1 loads — but **re-hashing it today will NOT reproduce
`5bd471fb…`**. The same path now hashes to
`ef68ae670b75d88f57260866653a484f224b1c31fdd8ff1d8e3cf2a1270b6f5b`, and all six joints' tick
ranges moved; concretely, `elbow_flex.range_max` went from **3090 to 3100**. `5bd471fb…` is now
the checksum of the *sibling* pre-0.6.x `so101_follower` copy, which was not recalibrated —
so a reader who re-hashes and finds a different value is looking at a recalibration, not at
tampering.

Every measurement in this section is **correct as taken** and remains valid as a historical
record, but it is **not reproducible against the current calibration file**. §5.1 of
`docs/GROOT-NATIVE-BASELINE.md` records the same split from the other side (that baseline was
taken against the *later* calibration) and states which citations it affects — most notably
§9.7's `elbow_flex` at-limit fingerprint, whose raw tick 3090 is no longer that joint's
calibrated maximum. Anyone re-deriving a number from this section must use the calibration in
force at the time, not the file on disk now.

### 9.2 What was read before anything moved

Every tick read, the PID read-back and the calibration assertion were taken **before** the first
command that moved a joint, so a bus or calibration fault would have surfaced while the arm was
stationary. The as-found reading (no motion commanded at all):

| Joint | raw tick | reported (`use_degrees=True`) | recomputed percent |
|-------|---------:|------------------------------:|-------------------:|
| `shoulder_pan` | 2028 | -7.8681 | -6.7522 |
| `shoulder_lift` | 875 | -101.6264 | -97.9661 |
| `elbow_flex` | 3090 | 96.3516 | **100.0000** |
| `wrist_flex` | 2930 | 75.3846 | 74.7928 |
| `wrist_roll` | 1006 | -92.3516 | -55.0432 |
| `gripper` | 2056 | 0.7634 | 0.7634 |

**A safety precondition was discovered by taking those reads first, and it is the reason the
read-only-first ordering is not ceremony.** With torque disabled, every motor reported
`Goal_Position = 0`. Upstream's `configure()` runs inside `bus.torque_disabled()`, whose exit
calls `enable_torque()`, and `enable_torque()` writes `Torque_Enable` and `Lock` only — it does
**not** synchronise `Goal_Position` to the present position. Connecting without intervention
would therefore have commanded all six joints to raw tick 0 the instant torque returned, a jump
of up to 3090 ticks on `elbow_flex`. The controller's `connect()` now writes
`Goal_Position <- Present_Position` while torque is still off
(`SO10xArmController._prearm_goal_to_present()`), which cannot itself move the arm and turns the
torque-enable into a hold. **The dangerous path was never exercised, so this is a hazard
prevented, not a hazard demonstrated.** The pre-arm was originally discovered and implemented in
this harness; it now lives on the controller only, so every entry point that connects the arm
inherits it and there is one copy of the tolerance and the skip condition.

### 9.3 `active_mode`: two different questions, answered separately

The probe recomputes all three candidate normalizations from the calibration file and reports
which one matched the value the bus reported, within 1e-3. **Exactly one candidate matched every
joint at every pose** — no ambiguity, at no point, so no tolerance was widened.

| Question | Answer | How it was established |
|----------|--------|------------------------|
| Which mode is **the bus** producing? | `DEGREES` on all five arm joints, `RANGE_0_100` on the gripper | Recompute-and-match against the bus's reported value |
| Which mode does the bus produce at `use_degrees=False`? | `RANGE_M100_100` on all five arm joints, `RANGE_0_100` on the gripper | The bus's own norm modes flipped in place and the same physical pose re-read through the same upstream `_normalize` |
| Which mode was **the checkpoint** trained in? | **`RANGE_M100_100`** — the offline verdict, **CONFIRMED** | §9.6 and §9.7 below, using only rows that discriminate |

**These are not the same question, and the first is not evidence about the third.** The bus
produces whatever `use_degrees` selects, and Dum-E deliberately runs `use_degrees=True` (§7), so
`DEGREES` is a fact about the *configuration*. The checkpoint question is settled by which
reading of a physical pose is consistent with the checkpoint's own recorded statistics. Reading
the first answer as a refutation of the verdict would be the category error this section exists
to prevent. The gripper's `RANGE_0_100` under `use_degrees=True` is also now confirmed live,
which is what makes §4.2's claim that the gripper needs no conversion a measurement.

### 9.4 `before_after`: the per-joint upgrade comparison

Per joint at each pose: the raw tick, what the upgraded stack reported, what the **pre-upgrade**
0.3.3 formula computes from that same tick, and the difference. Criterion 2 budgets 0.5.

| Pose | Joint | raw tick | post-upgrade reported | pre-upgrade computed | difference |
|------|-------|---------:|----------------------:|---------------------:|-----------:|
| `initial` | `shoulder_pan` | 2114 | -0.3077 | -0.3077 | **0.0** |
| `initial` | `shoulder_lift` | 875 | -101.6264 | -101.6264 | **0.0** |
| `initial` | `elbow_flex` | 3090 | 96.3516 | 96.3516 | **0.0** |
| `initial` | `wrist_flex` | 2931 | 75.4725 | 75.4725 | **0.0** |
| `initial` | `wrist_roll` | 1027 | -90.5055 | -90.5055 | **0.0** |
| `initial` | `gripper` | 2049 | 0.2776 | 0.2776 | **0.0** |
| `ready` | `shoulder_pan` | 2115 | -0.2198 | -0.2198 | **0.0** |
| `ready` | `shoulder_lift` | 1005 | -90.1978 | -90.1978 | **0.0** |
| `ready` | `elbow_flex` | 2889 | 78.6813 | 78.6813 | **0.0** |
| `ready` | `wrist_flex` | 2933 | 75.6484 | 75.6484 | **0.0** |
| `ready` | `wrist_roll` | 1026 | -90.5934 | -90.5934 | **0.0** |
| `ready` | `gripper` | 2049 | 0.2776 | 0.2776 | **0.0** |
| `remote` | `shoulder_pan` | 2115 | -0.2198 | -0.2198 | **0.0** |
| `remote` | `shoulder_lift` | 2038 | 0.6154 | 0.6154 | **0.0** |
| `remote` | `elbow_flex` | 2043 | 4.3077 | 4.3077 | **0.0** |
| `remote` | `wrist_flex` | 2655 | 51.2088 | 51.2088 | **0.0** |
| `remote` | `wrist_roll` | 1026 | -90.5934 | -90.5934 | **0.0** |
| `remote` | `gripper` | 2904 | 59.6114 | 59.6114 | **0.0** |

**Every difference is exactly zero — bit-for-bit, not merely under 0.5.** That is the outcome
§4.1 predicted from byte-identical formula bodies, and it is the strongest available form of the
criterion: the upgrade delta at the robot layer is *identity*, measured rather than argued. A
nonzero difference here would have implicated the calibration file, not the version change.

Note that `reported` tracks the *commanded* pose only to within the servos' own PID lag — e.g.
`elbow_flex` settles at 78.68 against a commanded 75.0 at the `ready` pose, sagging under gravity
at `P=10`. That lag is irrelevant to this comparison, which is precisely its virtue: both sides
are derived from the same tick, so wherever the arm actually came to rest, the two formulas are
compared at the same physical position.

### 9.5 The per-joint scale: derived, and now measured

Measured by reading the same register twice at one physical pose — once as degrees off the bus,
once as percent off the bus with its norm modes flipped in place — and taking the ratio. **Both
sides come from hardware**, and the measurement was repeated independently at all three poses.

| Joint | derived (§4.2, arithmetic) | measured (live, all 3 poses) | agreement |
|-------|---------------------------:|-----------------------------:|-----------|
| `shoulder_pan` | 1.16527 | **1.16527** | exact to 5dp |
| `shoulder_lift` | 1.03736 | **1.03736** | exact to 5dp |
| `elbow_flex` | 0.96352 | **0.96352** | exact to 5dp |
| `wrist_flex` | 1.00791 | **1.00791** | exact to 5dp |
| `wrist_roll` | 1.67780 | **1.67780** | exact to 5dp |

**The derived table is now a measurement.** Research assumption A2 — that the per-joint scale
factors were arithmetic rather than hardware-confirmed — is **discharged**. The two tables agree,
so there is no disagreement to record with competing provenances; had they disagreed, both would
have stayed on the record and A2 would have remained open.

### 9.6 The `initial`-pose envelope check, run live

Run against the `initial` pose, never the ready pose (§3). All five arm joints, both conventions,
against the checkpoint's `state` q01/q99 envelope:

| Joint | q01 | q99 | live as degrees | inside? | live as percent | inside? | discriminates? |
|-------|----:|----:|----------------:|:-------:|----------------:|:-------:|:--------------:|
| `shoulder_pan` | -63.7670 | 44.0485 | -0.3077 | yes | -0.2641 | yes | no |
| **`shoulder_lift`** | **-99.7433** | **54.0454** | **-101.6264** | **NO** | **-97.9661** | **yes** | **YES** |
| `elbow_flex` | -53.3910 | 100.0000 | 96.3516 | yes | 100.0000 | yes | no |
| `wrist_flex` | 12.6699 | 98.6112 | 75.4725 | yes | 74.8801 | yes | no |
| `wrist_roll` | -99.6330 | -14.2483 | -90.5055 | yes | -53.9429 | yes | no |

**One joint discriminates, and it decides for percent.** `shoulder_lift`'s live reading falls
*outside* the envelope read as degrees (-101.6264 against a q01 of -99.7433) and *inside* it read
as percent (-97.9661). The four non-discriminating rows are recorded for completeness and are
**not** cited as evidence — under the prohibition against evidence that reads identically under
both conventions, only the `shoulder_lift` row is admissible. This is the offline table of §3
reproduced from live encoder ticks instead of from hardcoded pose literals.

### 9.7 The `elbow_flex` at-limit fingerprint — the strongest live datum

The arm was found with `elbow_flex` at raw tick **3090**, which is *exactly* its calibrated
`range_max`, i.e. the joint sitting at its recorded mechanical limit. At that physical position:

- read as **degrees** it reports **96.3516** — the full extent of its physical span;
- read as **percent** it reports **exactly 100.0000**, by construction of the clamp;
- the checkpoint records `state.single_arm.max[elbow_flex] = 100.0` **and** `q99 = 100.0`.

**The checkpoint recorded precisely the number the percent convention produces at this exact
physical position, and a number the degrees convention cannot produce at any position.** This
converts §2.2 from a falsification computed off the calibration file into a direct hardware
observation, and it is independent of the envelope check in §9.6.

### 9.8 `pid_readback` on the live bus

Read back from **every** motor with `normalize=False` and a retry floor, at connect, on the real
Feetech bus:

| Motor | P | I | D |
|-------|--:|--:|--:|
| `shoulder_pan` | 10 | 0 | 5 |
| `shoulder_lift` | 10 | 0 | 5 |
| `elbow_flex` | 10 | 0 | 5 |
| `wrist_flex` | 10 | 0 | 5 |
| `wrist_roll` | 10 | 0 | 5 |
| `gripper` | 10 | 0 | 5 |

**All six motors report the Dum-E preset 10/0/5.** Every measurement in this section was
therefore taken at a known, recorded stiffness. This is the value a later phase should cite as
the stiffness the arm actually ran at — not the value that was requested.

### 9.9 The `clamp` demonstrated on real motion

A single per-step delta of **1.5x the clamp** was commanded on one joint. `wrist_roll` was chosen
because it is the only joint with more than twice the clamp in calibrated travel (+/-167.78
degrees against +/-116.53 for the next widest), which means **even a clamp that failed to engage
would have commanded a physically reachable pose** — the demonstration cannot damage the arm by
succeeding *or* by failing.

| Item | Value |
|------|-------|
| Start pose | `initial` (the `move_to_initial_pose()` target; operator-designated safe testing pose) |
| Joint | `wrist_roll` |
| Present | -89.9780 |
| Requested | +150.0220 (a delta of 240.0 = 1.5 x the clamp) |
| Returned by `send_action` | **+70.0220** |
| Returned delta | **exactly 160.0000** — the configured clamp |
| Settled | +68.2637 (1.76 of PID lag) |

All three assertions hold, two of them from data rather than from a log:

1. **The returned action differs from the requested action** on that joint, by 80.0.
2. **The magnitude of the returned delta equals the clamp** within upstream's own 1e-4
   divergence threshold.
3. **The warning reached Dum-E's own loguru stream**, naming the joint, the requested value and
   the clipped value, and carrying upstream's exact sentence:
   `Relative goal position magnitude had to be clamped to be safe. max_relative_target=160.0 clamped 1 joint(s): wrist_roll.pos requested=150.0220 clipped=70.0220`

**Upstream's own root-logger warning ALSO arrived**, through the stdlib-to-loguru bridge, carrying
its `original goal_pos` / `safe goal_pos` payload. Both emitters are confirmed live, so the
documented failure mode — a check grepping stderr while Dum-E logs to stdout — is closed on
hardware and not merely in a unit test.

**A nominal reset does not trip the clamp.** The production `move_to_initial_pose()` was called
from the `remote` pose as the single unsmoothed command it is: the worst commanded delta was
**102.62** on `shoulder_lift`, comfortably below 160.0, and **no clamp warning was emitted**. The
worst case reachable from Dum-E's own fixed poses is 102 (`remote` -> `initial`). The ~190
`elbow_flex` delta that was flagged as the likeliest legitimate trigger requires an *extreme
policy* pose, not one of Dum-E's fixed poses, so it remains unobserved and is a live-run question
for the parity phase. No clamp fired at any point during the interpolated pose sweep either.

### 9.10 What this run did NOT establish

- **It did not flip `use_degrees`, and did not need to.** §7 still stands unchanged: the correct
  value is `False`, the flip requires converting all four pose vectors in the same change, and it
  is a later-phase decision. The two-configuration read used here changed the bus's norm modes
  in place for the duration of a single read and restored them, verified, before returning.
- **It did not validate the magnitude of the clamp for nominal operation.** It shows the clamp
  *engages* and *is visible*. Whether 160.0 is right for nominal operation is settled by the
  requirement of zero clamp warnings during a nominal run. If nominal operation trips it,
  re-derive the value — do not remove the clamp and do not suppress the warning.
- **It did not confirm which calibration was in use at training time** (see §8.1). Nothing
  available on this hardware can.

---

## References

- **Harness:** `scripts/pose_sweep_units_probe.py` — `clip_fingerprint_count()`,
  `degrees_reachable_range()`, `deg_per_pct_table()`, `wrist_roll_cross_check()`,
  `envelope_contains()`, `normalize_v033()` / `normalize_v061()`,
  `check_pinned_constants_against_local_artifacts()`, and the arm-required
  `units_verdict()` / `active_mode_from_raw_tick()` / `live_pose_sweep()` /
  `compare_pre_and_post_upgrade()` / `measured_deg_per_pct()` / `envelope_row()` /
  `demo_clamp()`, plus the safety guard `assert_pose_reachable()` / `tick_for_degrees()`, which
  validates against the live `bus.calibration` rather than the pinned table.
- **Safety guard on the controller:** `SO10xArmController._prearm_goal_to_present()`, run by
  `connect()` before `robot.connect()` re-enables torque.
- **Gate:** `tests/test_units_verdict.py` — hermetic tests, none of which may skip.
- **Upstream:** `lerobot.motors.motors_bus.MotorsBus._normalize` and
  `lerobot.motors.motors_bus.MotorNormMode` (identical in 0.3.3 and 0.6.1);
  `lerobot.robots.so_follower` (the gripper's hardcoded `RANGE_0_100`);
  `lerobot.scripts.convert_dataset_v21_to_v30` (no value transform).
- **Artifacts:** the checkpoint's `statistics.json` and the LeRobot follower calibration JSON.
  Both are untracked local data, resolved by the harness from `HF_LEROBOT_CALIBRATION` /
  `HF_LEROBOT_HOME` / `DUME_CHECKPOINT_STATISTICS` or the `--statistics` / `--calibration`
  flags — never from a hardcoded absolute path.


## 10. Current calibration re-derived offline (2026-09-11)

This section supersedes the **current-calibration arithmetic** in §§2, 4 and 7,
not the historical measurements or calibration identities in §9. No arm connection,
pose sweep, recalibration, or clamp demonstration was performed for this derivation.
Phase 5’s 9/10 baseline and its ready→initial reset sequence remain unchanged.

The pre-change offline probe exited 1 with 12 range-field drift reports (all six joints).
The original failure is retained in the Phase 7 execution-attempts directory. The snapshot
was independently re-derived from the controller’s `resolve_calibration_file` selection;
the drift verifier does not refresh its pins automatically.

Resolved calibration: `/home/aaron/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json`.

Calibration SHA-256: `ef68ae670b75d88f57260866653a484f224b1c31fdd8ff1d8e3cf2a1270b6f5b`.

Checkpoint statistics SHA-256: `92f33c314351f6a0facc85b76fd8c5d257410c668c2901e9a97fb6fba170ce58`.

The installed LeRobot 0.6.1 `MotorsBus._normalize`/`_unnormalize` formulas use
`max_res = 4096 - 1`: degrees = `(tick - midpoint) * 360 / 4095`, and the
arm scale = `(range_max - range_min) * 360 / (4095 * 200)`. All six drive modes
are zero. The gripper remains `RANGE_0_100` under both configurations; its
degree range in the machine record is hypothetical arithmetic, not its active mode.

| Joint | Historical ticks | Current ticks | Degrees per percent | Reachable arm degrees |
|---|---|---|---|---|
| `shoulder_pan` | 792–3443 | 841–3433 | 1.13934 | ±113.93407 |
| `shoulder_lift` | 851–3211 | 860–3208 | 1.03209 | ±103.20879 |
| `elbow_flex` | 898–3090 | 903–3100 | 0.96571 | ±96.57143 |
| `wrist_flex` | 926–3219 | 920–3204 | 1.00396 | ±100.39560 |
| `wrist_roll` | 148–3965 | 0–4095 | 1.80000 | ±180.00000 |
| `gripper` | 2045–3486 | 2044–3501 | not applicable | not active |

All four unchanged pose vectors (`initial`, `ready`, `remote`, `release_lift`)
pass `assert_pose_reachable` against this current file. The `initial` prediction
for shoulder-lift is −102° → −98.83%, still outside/inside the checkpoint envelope
respectively. Elbow-flex’s current upper endpoint is **tick 3100 → 96.57143° / 100%**.
This is an **at-limit prediction**, not a new observation; §9.7’s measured
tick 3090 → 96.3516° belongs to the historical calibration.

The checkpoint retains seven exact ±100 clip bounds. Its recorded elbow maximum
100 remains above the current reachable degree maximum. However, wrist-roll’s
−90° now maps to **−50%**, outside the historical episode-0 band
[−58.952, −50.564]. That cross-check no longer supports the current calibration.
The remote episode-0 statistics remain unverified offline, and training-time
calibration is still unknown. The clip and initial-envelope arguments persist;
the old wrist-roll argument must not be silently reused.

`use_degrees=True`, PID 10/0/5, the 160.0 clamp, controller targets and reset
sequence remain unchanged. The current wrist-roll ±180° range still contains
the old clamp-demo target, but no new clamp behavior was measured. This arithmetic
does not approve motion or validate operational parity.

The evidence was generated once with:

```bash
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/pose_sweep_units_probe.py --skip-hardware --write-derivation corpus/phase7/calibration.json
```

Completed evidence is immutable: repeating this command at the same destination
fails. `--write-derivation` rejects motion modes and explicit calibration overrides.
Missing calibration/statistics and injected single-tick drift fail the prerequisite.
Any recalibration requires fresh derivation and release review; future approval
must bind the exact calibration digest above.
