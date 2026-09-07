# Normalization Units Verdict

**Status:** Resolved (offline, no hardware). Standing engineering resolution — not a changelog entry.

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

**The scale magnitudes are a hypothesis derived by arithmetic, not a hardware measurement.**
They are exact given a verified formula and a verified calibration file, but nothing on the arm
has confirmed them. `units_verdict()` / `active_mode_from_raw_tick()` — the raw-tick round-trip
probe, shipped unrun in this harness behind `--skip-hardware` — is what converts them into a
measurement, and it runs in the plan that follows the hardware-attach gate. **Pending hardware.**

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
offline. That escalation is explicitly authorized by the phase's own decision record (D-11).

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
   arguments intact. `check_pinned_constants_against_local_artifacts()` reports drift between
   the pinned constants and the local artifacts as a failure, so a recalibration or a
   checkpoint swap surfaces instead of silently invalidating the scale table.
2. **The per-joint scale factors are exact arithmetic on verified inputs, not measured on
   hardware.** Everything in §4.2 and the conversion table in §7 is derived. **Pending
   hardware:** the raw-tick round-trip probe is what measures it.
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

## References

- **Harness:** `scripts/pose_sweep_units_probe.py` — `clip_fingerprint_count()`,
  `degrees_reachable_range()`, `deg_per_pct_table()`, `wrist_roll_cross_check()`,
  `envelope_contains()`, `normalize_v033()` / `normalize_v061()`,
  `check_pinned_constants_against_local_artifacts()`, and the arm-required
  `units_verdict()` / `active_mode_from_raw_tick()`.
- **Gate:** `tests/test_units_verdict.py` — eight hermetic tests, none of which may skip.
- **Upstream:** `lerobot.motors.motors_bus.MotorsBus._normalize` and
  `lerobot.motors.motors_bus.MotorNormMode` (identical in 0.3.3 and 0.6.1);
  `lerobot.robots.so_follower` (the gripper's hardcoded `RANGE_0_100`);
  `lerobot.scripts.convert_dataset_v21_to_v30` (no value transform).
- **Artifacts:** the checkpoint's `statistics.json` and the LeRobot follower calibration JSON.
  Both are untracked local data, resolved by the harness from `HF_LEROBOT_CALIBRATION` /
  `HF_LEROBOT_HOME` / `DUME_CHECKPOINT_STATISTICS` or the `--statistics` / `--calibration`
  flags — never from a hardcoded absolute path.
