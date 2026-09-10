# LeRobot Serving Verdicts

**Status:** PAR-05 is **resolved on both sides, and the cross-backend comparison now AGREES** —
Isaac-GR00T `(256, 256, 3)` and the LeRobot **serving** path `(256, 256, 3)`, byte-identical
(sha256 `c30150ec…`). ~~its cross-backend comparison has since been run and MISMATCHED — LeRobot
`(256, 340, 3)` vs Isaac-GR00T `(256, 256, 3)`, so PAR-05 is flagged for review and criterion 5's
"identical shape" clause is not satisfied~~ — superseded by the gap-closure fix recorded in the last
section: the serving path now **forces the letterbox pad on**, which is the single stage the two
backends disagreed about. Criterion 5's *"identical shape"* clause **is** satisfied for the serving
path. LRG-06 **enforced** by a committed test (§2). The ROADMAP's Phase 6 criteria carry six recorded
corrections (§3) and three named open gaps (§4), of which gap 1 (A8) is closed as a measurement **and
its mismatch is now resolved**. Standing engineering resolution — not a changelog entry.

**The one assumption this rests on, stated up front because it was accepted knowingly.** Matching
Isaac rather than honouring the checkpoint's own flag rests on an **inference**: "the weights were
trained on Isaac's padded square" follows from "Isaac trained this checkpoint". **The training recipe
itself was never read, and no local artifact records it.** The operator chose to act on the inference
rather than spend a step confirming it. If Phase 7's parity work disappoints, this is the first thing
to re-examine. It is not settled fact and must not be written up as one.

This document records what Phase 6 settled about serving the fine-tuned
`checkpoints/GR00T-N1.7-3B-SO101` checkpoint through LeRobot: the checkpoint's real preprocessed
image geometry, the mechanism that enforces the policy container's loopback-only reachability, the
places where the ROADMAP's stated criteria describe a mechanism the pinned release does not have,
and the gaps that stayed open. It is the resolution referenced by requirements **PAR-05** and
**LRG-06**.

Figures are cited by the harness that computes them so the document and the code cannot drift apart
silently. The harness for §1 is `scripts/dump_preprocessed_image.py` and for the cross-backend
comparison `scripts/dump_gr00t_native_preprocessed_image.py`; the always-running gates are
`tests/test_par05_image_geometry.py` (§1 and the comparison's LeRobot-side anchor) and
`tests/test_loopback_publish_spec.py` (§2).

---

## PAR-05 image-geometry verdict

> **THE SERVING VERDICT: a 480x640x3 uint8 frame becomes exactly `(256, 256, 3)` uint8** on the path
> Dum-E's `lerobot-policy` container actually serves, because that path **forces the letterbox pad
> on**.
>
> **HISTORY, kept because the fix only makes sense against it:** ~~a 480x640x3 uint8 frame becomes
> exactly `(256, 340, 3)` uint8 under this checkpoint's own image recipe~~ — still true of the
> **unpatched** upstream transform run as this checkpoint configures it
> (`letter_box_transform: false`), and no longer what the server feeds the VLM.

Both are resolutions stated as exact integers, not approximations and not notes. The transform is
a deterministic cv2 `INTER_AREA` resize plus a floored center crop, so there is no tolerance to
allow: `tests/test_par05_image_geometry.py` asserts tuple equality on integers, and the
within-LeRobot comparisons use `numpy.array_equal`, never `numpy.allclose`.

**The effective pipelines** (`processor_groot.py:1423-1468`):

```
SERVING   :  letterbox-pad-to-square (FORCED)  ->  resize-shortest-edge-to-256  ->  center-crop-95%  ->  resize-shortest-edge-to-256
UNPATCHED :                                        resize-shortest-edge-to-256  ->  center-crop-95%  ->  resize-shortest-edge-to-256
```

**Where the forced pad is injected, and how it is verified.** One definition,
`policy_guard.groot_guard.serving_preprocessor_overrides()`, merged into the same
`make_pre_post_processors` call in `docker/lerobot-policy/server.py` that already carries the device
and rename-map overrides — upstream's own public step-override seam
(`processor_groot.py:401-455`), not a monkeypatch and not a hand-rolled resize. `SAFE-01/5` then
reads the **effective** value back off the built `GrootN17VLMEncodeStep` and refuses the handshake
if the override did not land, so a silently-dropped pad cannot become a silently-wrong geometry.
Measured in the shipped image (`docker exec`, no weights, no GPU): `(256, 256, 3)` uint8, sha256
`c30150ec…`, `served_letter_box_transform=True` against the checkpoint's declared `False`.

### The recipe, and where it comes from

Every value below is read from `checkpoints/GR00T-N1.7-3B-SO101/processor_config.json` ->
`processor_kwargs`. `scripts/dump_preprocessed_image.py` pins them as constants and re-reads the file
as its check 1; `test_recipe_constants_match_the_checkpoint_processor_config` re-reads it on every
suite run. A checkpoint swap that changes one of them goes RED rather than silently producing a
different shape.

| Key | Value | Role in the verdict |
|---|---|---|
| `letter_box_transform` | `false` | What the checkpoint **declares**. The unpatched transform honours it; **the serving path overrides it to `true`** (see above). Asserted as a drift catcher: every number in this document was measured against a checkpoint declaring `false` |
| `crop_fraction` | `0.95` | The center crop; **set**, so it takes precedence over `image_crop_size` |
| `image_crop_size` | `[230, 230]` | **Provably inert** — see below |
| `image_target_size` | `[256, 256]` | Supplies `target_h` as the fallback resize edge only; **not** the output shape |
| `shortest_image_edge` | `256` | The edge both resize passes target — this is why the output HEIGHT is 256 |
| `use_albumentations` | `true` | Selects the cv2/numpy transform; the torch path would give `(3, 256, 256)` |

### Observed shapes

Measured by `scripts/dump_preprocessed_image.py --outdir outputs/par05 --seed 0` (**9/9** checks PASS,
exit 0), transcribed from `outputs/par05/manifest.json`. Seed **0**; source frame **480x640x3
uint8**, built with `numpy.random.RandomState(0)` so the bytes are reproducible.

| Case | Input shape | Output shape | dtype |
|---|---|---|---|
| **`serving_path`** (pad FORCED, geometry read off the BUILT pipeline) | `[480, 640, 3]` | **`[256, 256, 3]`** | `uint8` |
| `checkpoint_recipe` (unpatched, pad as declared) | `[480, 640, 3]` | `[256, 340, 3]` | `uint8` |
| `placeholder_corrupted` (C-3, unpatched) | `[224, 224, 3]` | `[256, 256, 3]` | `uint8` |
| `square_256` | `[256, 256, 3]` | `[256, 256, 3]` | `uint8` |
| `letterbox` (`letter_box_transform=True`) | `[480, 640, 3]` | `[256, 256, 3]` | `uint8` |

The `serving_path` row is not a re-run of the `letterbox` row with a different name, even though the
two agree byte-for-byte. It is measured off the **real built pipeline**: the probe calls
`make_pre_post_processors` with the server's own override fragment, reads the five geometry settings
off the constructed `GrootN17VLMEncodeStep`, and hands them to the transform that step itself calls.
Check 3 fails if the override did not land, so the row cannot become a restatement of an intention.

Three further measured facts from the same manifest:

- **`crop_size_inert: true`** — the `[230, 230]` output and a `[999, 999]` output are byte-identical
  under `numpy.array_equal`. `image_crop_size` is consulted **only** when `crop_fraction is None`
  (`processor_groot.py:1453-1454`), so `[230, 230]` is dead configuration on this checkpoint.
  **Recorded so a future reader does not spend time tuning a value that does nothing.** Proven by
  measurement, not by citing upstream's source.
- **`replay_identical: true`** and **`serving_replay_identical: true`** — two invocations on the same
  frame are byte-identical, on both geometries. The serving path takes the deterministic CENTER crop;
  the train-time random crop is gated on `self.training and torch.is_grad_enabled()` and both are
  false there.
- **`serving_corruption_shape_invisible: true`** — **the honest cost of the forced pad, recorded as a
  measurement rather than omitted.** Because the pad squares every input, the C-3 corruption and a
  correct frame now emerge with the **same** shape `(256, 256, 3)` on the serving path and differ only
  in pixels. The shape-only discriminability described in the *Fidelity decision* below was a property
  of the **unpatched** path and is no longer available on the served one. What carries C-3 now is
  **prevention**, not detection: `fixup_policy_features` sets `input_features` before
  `from_pretrained`, so the `(3, 224, 224)` placeholder branch — guarded by `config is None`
  (`modeling_groot.py:247-261`) — never runs at all. Do not inherit "a shape check catches C-3" as a
  live reassurance.

### Stale framing

The ROADMAP asks to settle *"341x256-crop vs 256x256-letterbox empirically"*. **That framing is a
misread, and `REQUIREMENTS.md:50` already records it as one** — the ROADMAP section never caught up.
Its two halves are real but neither described the answer, and the served answer is now the second of
them **for a reason the framing did not contain**:

- **`256x341` is an INTERMEDIATE**, not an output: it is what the first resize-shortest-edge of a
  480x640 frame produces, before the 95% center crop and the second resize.
- **`256x256` is the LETTERBOX branch.** The checkpoint's own config declines it
  (`letter_box_transform: false`); the serving path **takes it anyway**, by override, because Isaac's
  code took it when it trained these weights. It is also reachable *without* letterboxing, from any
  square input — which is why "256x256" on its own still identifies nothing, and why the served
  geometry has to be evidenced by the pad flag rather than by the shape alone.

Both facts are documented by **named passing tests** rather than omitted, so they cannot be
"simplified" away later:

- `test_letterbox_branch_is_256x256x3_and_this_checkpoint_declares_it_off`
- `test_square_256_input_is_256x256x3`

Each docstring states that it documents the stale framing rather than evidencing the verdict. The
letterbox test additionally asserts that forcing the pad reproduces the SERVED output byte for byte, so
the two facts — "this branch is reachable" and "the serving path takes it" — are pinned in one place and
cannot drift apart.

### Fidelity decision

| Comparison leg | Fidelity | Why |
|---|---|---|
| Cross-container (LeRobot vs pinned Isaac-GR00T `23ace64f`) | **shape only** | Two different OpenCV builds and two different Python versions (the GR00T container is Py3.10); bit-exactness is not a reasonable contract across them |
| Within-LeRobot (two dumps, same interpreter) | **exact array equality** (`numpy.array_equal`) | Same OpenCV build in the same process, so an interpolation or floored-crop change must be a hard failure, not an approximate pass |

~~**Shape-only is defensible here for a specific, measured reason, not for convenience.** A shape
check would normally be weak. It is strong on this leg because the corruption it must catch **changes
the shape**: a frame pre-resized to 224x224 by `from_pretrained`'s placeholder feature emerges as
`(256, 256, 3)`, **not** `(256, 340, 3)`.~~

**SUPERSEDED for the serving path by the forced pad, and the replacement is weaker — stated plainly
rather than glossed.** The struck rationale was a property of the **unpatched** geometry and still
holds there (`test_placeholder_pre_resize_corruption_is_256x256x3_and_differs_only_unpatched` asserts
it). On the **served** geometry it does not: the pad squares every input, so the C-3 corruption (the
placeholder is `(3, 224, 224)` at `modeling_groot.py:255-261`, and `prepare_raw_observation` resizes
every camera to it at `helpers.py:165-168`) emerges as `(256, 256, 3)` — **the same shape as a correct
frame**, differing only in pixels. That same test asserts both halves of the new situation too (same
shape, different bytes), so the loss of discriminability is itself under test rather than merely
described.

What carries C-3 now is **prevention**: `fixup_policy_features` sets `input_features` before
`from_pretrained`, so the placeholder branch — guarded by `config is None` — never executes. The
cross-container leg's fidelity stays **shape only** as a *contract*; the byte-level agreement recorded
in the last section is an observation on top of it, not a promotion of it.

---

## LRG-06 loopback enforcement

> **The policy container's loopback guarantee lives in the host-side publish spec
> `-p 127.0.0.1:8080:8080`, and `tests/test_loopback_publish_spec.py` asserts it.**

### The mechanism

- **Host side:** the documented `docker run` command publishes with the mandatory `127.0.0.1:`
  host-IP prefix. Without the prefix, Docker publishes on **every** interface.
- **Container side:** the server binds `0.0.0.0` **deliberately**. This is not an oversight to
  tighten: a container-internal loopback bind is unreachable from the host, so
  `PolicyServerConfig`'s `host="localhost"` default (`configs.py:56`) is simply the wrong value for a
  published container, and `docker/lerobot-policy/entrypoint.py` overrides it on purpose with the
  reason written at the flag.
- **Rejected alternative:** `--network host`. It is simpler and would make the upstream default
  literally correct, but it hands the container the whole host network stack with no publish spec
  left to constrain it. D-12 rejected it explicitly.

### Why it is a test and not a default

This publish spec is the **sole** mitigation for an accepted risk. LeRobot's async gRPC transport
`pickle.loads` peer bytes in **both** directions by upstream design (ASY-06, PROJECT.md Constraints),
and the service carries no authentication, so anyone who can reach the port can execute code in the
server process. There is no auth to add on top. D-12's requirement is therefore that the guarantee
"cannot rest on a default" — the real guarantee lives in a run command that a human copies, so the
guard has to read that command.

**Widening the spec to make something work is forbidden.** If the gRPC path appears to fail through a
host-IP-scoped publish, the diagnosis is the diagnosis; widening it silently converts an accepted,
mitigated risk into an unmitigated one.

### What asserts it

| Test | What it proves |
|---|---|
| `test_documented_lerobot_policy_run_block_publishes_with_the_host_ip_prefix` | The documented command declares exactly one publish flag and its value equals exactly `127.0.0.1:8080:8080` (full-string equality, so a port typo is caught too) |
| `test_documented_run_block_binds_all_interfaces_inside_the_container_deliberately` | The command does not tighten the container's INTERNAL bind, and the entrypoint's `--host` default is still the all-interfaces address *with its explanation present* |
| `test_extractor_fails_loudly_when_the_anchor_is_absent` | **Non-vacuity (synthetic).** `extract_run_block` RAISES on a missing anchor and on an anchor with no fence after it — it never returns `""` |
| `test_guard_rejects_a_synthetic_command_without_the_host_ip_prefix` | **Non-vacuity (synthetic).** A bare `8080:8080` publish value FAILS the guard |
| `test_guard_rejects_a_synthetic_command_that_shares_the_host_network_namespace` | **Non-vacuity (synthetic).** `--network host`, `--network=host` and `--net host` all FAIL the guard; a command with no publish flag at all also fails rather than passing by vacuity |
| `test_incumbent_gr00t_block_is_out_of_scope_and_the_scoping_is_deliberate` | The scoping decision below, as a test |

The guard is located by the stable anchor comment
`# lerobot-policy container (LRG-06: loopback publish only)`, which sits on its own line immediately
**above** the fenced block, outside the fence. The anchoring is load-bearing in both directions: a
second fence (the `--preflight-only` command) follows further down, so an unanchored "next fence"
search would eventually assert against the wrong block. Both helpers (`extract_run_block`,
`publish_flags`) **raise** rather than returning an empty value, because a guard that silently passes
on zero blocks is worse than no guard — it reads as coverage (T-06-27).

### Deliberate scoping: the incumbent Isaac-GR00T block is left unchanged

The incumbent Isaac-GR00T runbook publishes `5555` with **no** host-IP prefix at all, so it would
FAIL this guard — and a test that globbed every `docker run` fence in `README.md` would go red on it.
It is left byte-unchanged in this phase on purpose: narrowing the GR00T-native container's
reachability would change the fallback path Phase 7's live parity gate runs on, which is a different
failure axis and deserves its own change with its own verification.

`test_incumbent_gr00t_block_is_out_of_scope_and_the_scoping_is_deliberate` records that as a test
rather than prose. It asserts three things about the *scoping* and nothing about what the incumbent
line ought to be: that the incumbent block still exists (located by its own `gr00t-server` token),
that an unanchored glob really would collide with it (the guard raises on it), and that the anchored
extractor does not reach it.

---

## Corrections to the ROADMAP's Phase 6 criteria

Six places where the ROADMAP states a mechanism the pinned `lerobot==0.6.1` release does not have.
Each was **recorded** rather than silently applied, so a reader of the criteria is not left with a
mechanism that cannot work.

| # | Criterion | Stated mechanism | Actual mechanism | Plan |
|---|---|---|---|---|
| 1 | Checkpoint loads unconverted | Pass `--policy.base_model_path` as an entrypoint flag | It is **not a flag on this entrypoint** — it is a client-supplied handshake value. Assert the server-side *resolved* `policy.config.base_model_path` instead | 06-01, 06-06 |
| 2 | 16-step horizon | Set `use_relative_actions` / `relative_exclude_joints` | Those are the **wrong knobs** (`use_relative_actions` is `False` while the checkpoint's `use_relative_action` is `True`, and the exclude list is training-path only). The load-bearing horizon input is `embodiment_tag == "new_embodiment"` — the only one of the checkpoint's nine tags carrying 16 `delta_indices` | 06-02, 06-06 |
| 3 | Emitted chunk length is 16 | Read the chunk length | Chunk length 16 **proves nothing**: three independent truncations force it, and `T=16`, `T=40` and `T=50` all decode to `(1, 16, 6)`. Assert the CONFIG-level values | 06-02 |
| 4 | The server "answers requests" | Assert the RPC succeeds | RPC success **hides a silent failure**: `GetActions` can return `Empty()` from a method declared to return `Actions`, so a blanket `except` becomes a successful RPC carrying zero bytes. Assert the **decoded chunk**, never RPC success | 06-01 |
| 5 | Normalization has not fallen back to identity | Read `GrootConfig.normalization_mapping` | A **guaranteed false alarm**: that mapping is IDENTITY by design here. Three discriminating signals replace it — the decode-step class, non-empty checkpoint stats, and `use_percentiles` | 06-02 |
| 6 | 5 seeded repeats produce identical chunks | Send a seed over the wire | There is **no seed to send**: `RemotePolicyConfig` has no seed field, and decoding uses flow matching over `num_inference_timesteps: 4` drawn from the ambient torch RNG. Reframed as an eval-mode assertion plus an in-container **in-process** seeded probe, reported as *"deterministic under an in-process seed"* — **never** as *"the server honours a seed"* (Phase 5 recorded `seed_verdict: not-honored` for the sibling GR00T-native server) | 06-03 |

Correction 6's keyless half is settled here: the image transform **replays identically** under the
serving configuration (`test_transform_replays_identically_under_the_serving_configuration`, exact
array equality). That test's docstring states explicitly that it is not a claim about the
flow-matching sampler and must not be cited as evidence that the server honours a seed. The
in-container half is plan 06-03's and is appended below when measured.

---

## Open gaps carried forward

Named explicitly so none of them disappears into a summary.

### 1. ~~The cross-container PAR-05 dump was never attempted~~ ~~— RUN, and it MISMATCHED~~ — RUN, MISMATCHED, and now RESOLVED (assumption A8)

**This gap is CLOSED. The dump was run, it mismatched, and the mismatch has been fixed.** The
GR00T-native dump was run in the pinned Isaac-GR00T image on the same `RandomState(0)` 480x640x3 frame
under the same six recipe values. It first showed a divergence; the serving path was then changed to
force the letterbox pad on, and a re-run shows agreement:

| Backend | Measured output | Fidelity |
|---|---|---|
| LeRobot **serving** path (§1, pad forced) | **`(256, 256, 3)`** uint8, sha256 `c30150ec…` | executed, exact |
| Isaac-GR00T `image_augmentations` (below) | **`(256, 256, 3)`** uint8, sha256 `c30150ec…` | executed, exact |
| ~~LeRobot `processor_groot` as configured~~ (unpatched, kept as history) | ~~**`(256, 340, 3)`** uint8~~ | executed, exact |

Criterion 5's *"identical shape"* clause **is satisfied for the serving path** — and satisfied
byte-for-byte, not merely shape-for-shape. Full evidence, provenance, root cause and the fix: **the
"PAR-05 cross-backend comparison" section at the end of this document**.

**What is no longer true about this section:** it was a one-sided record (T-06-29), then a two-sided
record of a divergence, and is now a two-sided record of agreement. Each framing is struck through
rather than deleted, so the sequence (never attempted → attempted → mismatched → resolved) stays
legible. **What replaces the old open item is not another open item but an ASSUMPTION**: the fix
matches Isaac on the inference that Isaac's geometry is the training-time geometry, which was never
confirmed against a training recipe. That assumption is recorded at the top of this document and in
the last section; it is the thing to re-examine first if Phase 7's parity numbers disappoint.

No live `gr00t-server` was used or started at any point: that container remains `Exited` by standing
decision, and both dumps ran in throwaway `docker run --rm --network none` containers, with no GPU, no
weights, no published port and no token.

### 2. Backbone-revision provenance is unestablished by design (assumption A6)

The pinned `nvidia/Cosmos-Reason2-2B` snapshot SHA is the **current** revision, deliberately chosen
by the user to stop silent drift. It is **not** confirmed to be the training-time revision, and no
local artifact records what the training-time revision was. Every message the build wrapper and the
container preflight emit therefore claims **drift protection only** — they prove the image carries
the revision the build pinned, and nothing more. `_build_n1_7_processor` takes no `revision`
argument, so cache contents plus `HF_HUB_OFFLINE=1` are the only enforcement available (D-08).

### 3. Seven edge-probe rows stayed `unclassified`

Their risk is semantic or deployment-shaped rather than shape-of-data, so the probe's boundary /
precision categories do not reach them. Surfaced rather than papered over, each with the executed
mechanism that carries it instead:

| Requirement | Why the probe's categories do not reach it | Mechanism carrying it |
|---|---|---|
| BACK-05 | A silent joint **permutation** passes every shape and key-set check | Ordering authority derived from `build_lerobot_features`, an ordered-list comparison, a construction-time `assert_state_ordering`, `zip(..., strict=True)`, and a **non-uniform** action vector proving dim *i* lands on joint *i* (06-04) |
| LRG-01 | "Answers requests" is a liveness claim, not a data-shape claim | Every assertion is on the **decoded chunk**: 16 `TimedAction`s of shape `(6,)` over real gRPC, plus the zero-length-`data` guard raising a named error instead of `EOFError` (06-01, 06-04) |
| LRG-02 | "Loads the right checkpoint" is a provenance claim | A six-check refuse-to-**start** preflight that returns before `add_insecure_port`, plus the resolved `base_model_path` asserted from the real load path (06-01, 06-06); the SAFE-01 post-load arming is 06-03's |
| LRG-03 | Relative-vs-absolute is a semantic claim about decoding | Two robot states through one observation: the arm targets track the anchor while the gripper does not, measured against a same-state noise floor (06-01) |
| LRG-04 | Horizon 16 is a config fact, and the emitted length is forced by three truncations | `infer_groot_n1_7_action_horizon(path, "new_embodiment") == 16` asserted at CONFIG level from two independent places, with the tag passed **explicitly** because inference returns `None` here (06-02, 06-06) |
| LRG-06 | A **deployment-shape** risk (what the run command publishes), not a shape-of-data risk | §2 above: the publish-spec test with three synthetic non-vacuity proofs (06-05) |
| SAFE-01 | Five serving-contract assertions are semantic, not dimensional | One pure validator over a frozen **17**-field snapshot, with `SAFE-01/N`-prefixed errors proven to fire on **thirteen** single-field mutations of the REAL checkpoint's snapshot (06-02, +1 for the forced-pad gap closure); arming it on the real load path is 06-03's |

### 4. Recorded, non-blocking: `outputs/` is not gitignored

`scripts/dump_preprocessed_image.py` writes to `outputs/par05/`, and those artifacts are
**deliberately not committed**. `outputs/` is still absent from `.gitignore` (Phase 5 deferred item
#1, a different axis, deliberately not folded into this phase). The numbers that matter are
transcribed into this document, which **is** committed, and
`tests/test_par05_image_geometry.py` re-derives them from the checkpoint on every suite run — so the
committed verdict has a live guard behind it rather than resting on a stale `.npz` (T-06-30).

---

## References

- `scripts/dump_preprocessed_image.py` — the PAR-05 LeRobot-side harness (keyless, GPU-free,
  weight-free); measures BOTH the unpatched transform and the real built serving pipeline
- `scripts/dump_gr00t_native_preprocessed_image.py` — the GR00T-native-side harness; runs inside
  `gr00t:latest` with no GPU, no weights, no network and no token, and verifies the image's revision by
  content digest before measuring anything
- `policy_guard/groot_guard.py` — `serving_preprocessor_overrides()`, the ONE definition of the forced
  letterbox pad, and the `SAFE-01/5` assertion that verifies it landed on the served pipeline
- `tests/test_par05_image_geometry.py` — the always-running geometry gate (15 tests, never skips)
- `scripts/build_gr00t_image.sh` — the `PIN=` the cross-backend measurement is recorded against
- `tests/test_loopback_publish_spec.py` — the LRG-06 publish-spec guard (6 tests, never skips)
- `README.md` — the documented `lerobot-policy` run command, under the LRG-06 anchor
- `docker/lerobot-policy/entrypoint.py` — the six-check refuse-to-start preflight and the
  deliberate all-interfaces internal bind
- `policy_guard/groot_guard.py` — the SAFE-01 serving-contract assertions
- `docs/UNITS-VERDICT.md` — the sibling recorded verdict (PAR-04 / PAR-06), whose format this
  document follows
- `checkpoints/GR00T-N1.7-3B-SO101/processor_config.json` — the source of every recipe value in §1

<!-- plan 06-03 appends its measured in-container sections below this line -->

---

## SAFE-01 wiring evidence (measured in-container)

> **SAFE-01's five assertions run from two call sites against one shared pure function, and the
> guard is proven WIRED rather than merely correct in isolation:** a deliberately wrong handshake
> (`actions_per_chunk=40`) is **refused on the real load path** in **5.11 s**, with the guard's own
> `SAFE-01/2` message reaching the client un-retried and the container log at ERROR level; and a
> healthy handshake emits one `SAFE-01 guard: PASS` line naming the five values the guard actually
> saw.

Measured against the running `lerobot-policy` container (bf16, `cuda_allocated_MiB=6015.0`) by
`tests/test_lerobot_serving_live.py`. The implementation is `policy_guard/groot_guard.py`, unmodified
by this plan and imported by both sites.

### Two call sites, deliberately unequal in strength

| Site | File | What it can see | What it validates |
|---|---|---|---|
| **Preflight** (config-only, check 4 of 6) | `docker/lerobot-policy/entrypoint.py` | The bind-mounted checkpoint directory only. No policy, no processors, no GPU | SAFE-01/1, /2, /4 and the geometry half of /5 — against `EXPECTED_HORIZON`, because no client exists yet |
| **Post-load** | `docker/lerobot-policy/server.py` | The loaded `GrootPolicy.config` plus the rebuilt preprocessor/postprocessor — the objects that will actually run inference | All five, against the **client-supplied** `self.actions_per_chunk` |

Both are needed and neither is redundant:

- The **preflight** is what makes SAFE-01's *"refuses to **START** with a named, specific error"*
  literally true. `main()` returns before `grpc.server(...)` and before `add_insecure_port`, so a
  refusal never reaches a listening socket — and it happens before a single one of the checkpoint's
  12.6 GB of shards is read.
- The **post-load** site is the only place that sees the served object at all, because upstream
  constructs the policy *inside* the `SendPolicyInstructions` handler. It is also the only site that
  can validate `decode_step_type` and the two processor `training` flags: on a config-only path no
  processor object exists, so `snapshot_from_checkpoint_dir` sets those three to their passing values
  **by construction** and says so in a comment. **The preflight is therefore deliberately weaker than
  the post-load site**, and is documented that way rather than presented as equivalent coverage.

The preflight check's own PASS line, verbatim, from the serving path:

```
[4/6] SAFE-01 serving contract for '/checkpoints/model' (config-only) ...
  PASS: SAFE-01/1..5 hold config-only: embodiment_tag='new_embodiment', checkpoint_horizon=16,
  use_relative_action=True, use_percentiles=True, stats_non_empty=True, crop_fraction=0.95,
  shortest_image_edge=256, letter_box_transform=False (decode_step_type and the two training flags
  are NOT validated here — post-load site only)
```

That check has been **red**, not merely written: against a directory that passes checks 1-3 (all
three sidecars present, `is_raw_groot_n1_7_checkpoint` True, horizon 16) but whose
`processor_config.json` sets `use_percentiles: false`, the preflight exits **1** with
`FAIL: SAFE-01/4 use_percentiles is False; ... Refusing to unnormalize with the wrong statistic.`
and `3/4 checks passed`. An entry that has never been red is an entry that has never been tested.

### The healthy pass, verbatim

```
INFO 2026-09-10 06:44:54 y/server.py:685 SAFE-01 guard: PASS | base_model_path=/checkpoints/model | embodiment_tag=new_embodiment | actions_per_chunk=16 | checkpoint_horizon=16 | decode_step=GrootN17ActionDecodeStep | checkpoint_letter_box_transform=False | served_letter_box_transform=True
```

The last two fields were added by the forced-pad gap closure and are the only pair on this line that
**disagree by design**: the checkpoint declares the letterbox off and the serving path forces it on.
Both are printed because either alone would be uninformative — the forced `True` alone could be a
constant the guard prints without having read the pipeline, and the declared `False` alone says nothing
about what the server does.

One line, so a single `docker logs | grep` shows both **that** the guard ran and **what** it saw. A
guard that passes silently is indistinguishable from a guard that never ran — that is D-05's stated
failure mode and threat T-06-14. `test_live_guard_pass_is_logged_on_the_real_load_path` asserts on
this line through a `docker logs --since <microsecond RFC3339 stamp taken immediately before the
connect>` window, so a stale line from an earlier handshake cannot satisfy it and the assertion
cannot go vacuous after its first successful run.

**This line also discharges what plan 06-02 could not.** `snapshot_from_loaded` locates the pack step
by `state_dropout_prob`, the encode step by `letter_box_transform` and the decode step by
`env_action_dim`; 06-02's coverage D7 is flagged `human_judgment: true` because those markers had only
ever been exercised against minimal stand-ins. Here they run against the real constructed
`GrootPolicy` and processor objects. An absent marker raises `ValueError` naming what it looked for,
so `decode_step=GrootN17ActionDecodeStep` is **positive evidence that all three located correctly**,
and both `training` flags read False on the real pipeline.

### The refusal, verbatim

Client side, with every handshake field correct except `actions_per_chunk=40`:

```
LeRobot policy server at 127.0.0.1:8080 rejected the call with gRPC status
StatusCode.FAILED_PRECONDITION: "SAFE-01/2 configured actions_per_chunk=40 disagrees with the
checkpoint's delta_indices (16). The horizon is configurable (D-11), so it can be configured wrong;
refusing to obey a configured horizon the checkpoint cannot decode.". Not retried — a non-transient
status is the server's own diagnosis, and three retries would only bury it under a generic
'unreachable'.
```

Server side, same refusal, at ERROR level:

```
ERROR 2026-09-10 05:17:08 y/server.py:602 SAFE-01 guard: REFUSED | SAFE-01/2 configured actions_per_chunk=40 disagrees with the checkpoint's delta_indices (16). The horizon is configurable (D-11), so it can be configured wrong; refusing to obey a configured horizon the checkpoint cannot decode.
```

| Measurement | Value |
|---|---|
| Elapsed, client `connect()` to `RuntimeError` | **5.11 s** (budget 60 s) |
| gRPC status chosen | `FAILED_PRECONDITION` |
| Retries burned | **0** — the status is absent from `RETRYABLE_CODES` |
| Recovery after refusal | a correct handshake immediately afterwards returns **16** actions of shape `(6,)` |

`FAILED_PRECONDITION` is chosen deliberately on both counts. Semantically the request was well-formed
and the *server's* state is what is unacceptable. Operationally,
`policy/lerobot/session.py`'s `RETRYABLE_CODES` holds only `UNAVAILABLE` and `DEADLINE_EXCEEDED`, so
this status raises on the first attempt carrying the server's own message. A retryable or
transport-shaped status would instead have burned three handshake retries — each starting **another**
concurrent multi-GB weight load — and then reported a generic unreachable error, sending the operator
to debug the network instead of the checkpoint.

**What the refusal does NOT do, stated so it is not overread:** it does not happen before the weights
are read. The post-load site exists precisely because the policy is constructed inside the request
handler, so a rejected handshake still costs one full load (~5 s here, page cache warm). The site
that refuses before any shard is read is the preflight, and it is a different instrument answering a
different question.

**40 is the well-lit wrong path, twice over**, which is why it is the injected value rather than an
arbitrary number: `GrootConfig`'s own `chunk_size`/`n_action_steps` defaults are 40, and this
checkpoint's own `config.json` advertises `action_horizon: 40`. D-11 made the horizon configurable;
this is the assertion that keeps a configurable horizon from being an *obeyable wrong* horizon.

### Named non-discriminating mechanism: `GrootConfig.normalization_mapping`

Recorded in the `tests/test_units_verdict.py` tradition — a mechanism proven not to discriminate is
written down as such rather than quietly omitted.

`GrootConfig.normalization_mapping` is **IDENTITY for `VISUAL`, `STATE` and `ACTION` by design** on
every healthy launch, and upstream states it is not consulted at all
(`configuration_groot.py:258-269`: GR00T normalizes state/action internally in its processor steps
and the backbone's image processor handles images, so the policy does not use LeRobot's
`NormalizerProcessorStep`). Reading it as the "normalization has fallen back to identity" signal —
the ROADMAP's stated mechanism, correction 5 in §3 — is a **guaranteed false alarm**. The guard
therefore **never reads it**, proven at AST level rather than by grep because the module docstring
quotes upstream's reason at length.

The three signals that carry SAFE-01/4 and SAFE-01/3 instead, all measured on the real checkpoint:

| Signal | Observed | Why it discriminates |
|---|---|---|
| `decode_step_type` | `GrootN17ActionDecodeStep` | The legacy `GrootActionUnpackUnnormalizeStep` is installed **only** when the checkpoint's stats are unusable, and it collapses `(B,T,D)` chunks to a single timestep. Its presence in a live pipeline *is* the identity-normalization tell |
| `stats_non_empty` | `True` | An empty stats table makes the decoder return normalized `[-1, 1]` actions while every log line looks healthy |
| `use_percentiles` | `True` | This checkpoint normalizes with q01/q99; min/max would rescale every action |

---

## Determinism verdict (measured in-container)

> **Five repeats of one byte-identical observation return byte-identical decoded chunks —
> deterministic under an in-process seed set by Dum-E's own `PolicyServer` subclass.** Compared with
> exact equality (`torch.equal` per timestep), no tolerance of any kind.

### The measurement

| | Value |
|---|---|
| Seed | `1234`, via `DUME_POLICY_SEED` |
| Where the seed is applied | in-process, inside `DumEGrootPolicyServer._predict_action_chunk`, immediately before `_get_action_chunk` |
| Repeats | 5, one reused observation object (byte-identical by construction, not by coincidence) |
| Chunks returned | 5, each of length 16 |
| Comparison method | `torch.equal` per timestep — **exact**, no `atol`, no `rtol`, no `approx` |
| **Result** | **5/5 byte-identical**; 0 mismatching timesteps out of 4 x 16 |
| Unseeded control (throwaway probe, same observation, same container recreated without the variable) | **0/4 identical**, max abs difference **14.61** |

The unseeded control is what makes the seeded result discriminating rather than trivially true: with
no seed set, five repeats of the *same* observation diverge by up to 14.61 in joint space, consistent
with plan 06-01's measured same-state noise floor (mean `|delta|` per dimension
`[1.1153, 1.5060, 2.3234, 1.4766, 1.0816, 5.3045]`, with one individual gripper pair reaching
-10.06). The seed is what collapses that to exact equality.

The mechanism is opt-in and proven inert when unused:
`test_live_no_seed_variable_means_no_seed_is_set` asserts `DUME_POLICY_SEED` is genuinely **absent**
from the container's environment (`docker exec printenv` exits non-zero) and that inference still
returns a well-formed 16 x `(6,)` chunk. With the variable unset the server calls neither
`torch.manual_seed` nor `torch.cuda.manual_seed_all` and does not touch the RNG at all, so the
production path Phase 7 will measure is byte-identical to plan 06-01's. A **malformed** value raises,
naming the variable and the offending value, rather than being silently ignored — a determinism
instrument that silently does nothing would let a nondeterministic result be recorded as a seeded one
(threat T-06-16).

That test deliberately does **not** assert that unseeded repeats differ. The control number above was
measured as a throwaway probe and recorded here precisely so it does not become a committed
assertion: a test that depends on nondeterminism manifesting would be flaky in the one direction a
safety suite must never be flaky — green when the mechanism is broken.

### Why the criterion had to be reframed

The ROADMAP asks that "5 seeded repeats of one observation produce identical chunks". Two independent
facts make that unobtainable as written, both re-verified:

1. **`RemotePolicyConfig` has no seed field at all** (`async_inference/helpers.py:266-273` — the six
   fields are `policy_type`, `pretrained_name_or_path`, `lerobot_features`, `actions_per_chunk`,
   `device`, `rename_map`). There is no seed to send over this wire.
2. Decoding uses **flow matching over `num_inference_timesteps: 4`**, whose initial noise is drawn
   from the ambient torch RNG **inside the server process**.

This is correction 6 in §3, and this section is the in-container half it defers to.

The **keyless eval-mode half** is independently discharged and is what makes the seeded probe
meaningful rather than accidental. `training` is a constructor kwarg set from `dataset_meta`
(`processor_groot.py:1225, 1266` — `training=dataset_meta is not None`), and `policy_server` passes no
`dataset_meta`, so it is `False` on the serving path; both stochastic steps are *additionally* gated
on `torch.is_grad_enabled()` (`processor_groot.py:2099` and `:1885`) while `predict_action_chunk` is
decorated `@torch.no_grad()` (`modeling_groot.py:473`). Isaac's train-time random crop and this
checkpoint's `state_dropout_prob: 0.2` are therefore **doubly** disabled. `SAFE-01/5` asserts both
`training` flags are False, and the post-load site is the only one that can — verified against the
real constructed processors this plan.

### What this verdict does NOT say

1. **It does not say the server honours a seed.** It does not, and cannot: `RemotePolicyConfig`
   carries no seed field, so no seed can reach the server from a client. The seed here is set
   in-process by Dum-E's own subclass, in code this repo owns.
2. **It does not say the wire carries a seed.** Nothing about a seed travels over
   `transport.AsyncInference` in either direction. `DUME_POLICY_SEED` is read from the container's
   environment at inference time.
3. **It does not overturn or extend Phase 5's `seed_verdict: not-honored`** for the sibling
   GR00T-native N1.7 server (05-02: same-seed max `|diff|` 5.51 vs different-seed 4.63, with `seed`,
   `random_seed` and `rng_seed` all tried). Those are *different claims about different things* — one
   about whether a server honours a client-supplied seed, one about whether a process is repeatable
   once its own RNG is fixed. This project's classifier draws that distinction deliberately and this
   result must not be cited as collapsing it.
4. **It does not license Phase 7 to assume a replayable seed** without setting one in-process itself.
   The parity harness needs a seeded, in-process instrument; over this wire it has none, and plan
   06-01 measured why — on the gripper dimension the per-call re-sampling noise (5.30) is several
   times larger than the signal being asserted (1.28).
5. **It does not extend to the image path**, which was already settled separately and by a different
   mechanism: §1's `replay_identical` is exact array equality over the *transform*, with no sampler
   involved.

---

### ~~Section 4.1 of `## Open gaps carried forward` is unchanged by this plan~~ — SUPERSEDED TWICE

~~The cross-container PAR-05 dump from the pinned Isaac-GR00T `23ace64f` image (assumption A8) was
**still not attempted** and remains a one-sided record, not a two-sided comparison. Plan 06-03 did not
absorb it and does not mark it resolved: `gr00t-server` is `Exited` by standing decision for the
remainder of Phase 6, so any GR00T-native-side comparison must go through the mock or dumped tensors.
It stays open in `.planning/WINDOWS.md` (entry 17).~~

Accurate **as of plan 06-03** and kept as history. Superseded first by the A8-closing measurement (the
dump ran in a throwaway `docker run --rm --network none` container without starting `gr00t-server`, so
the stated blocker turned out not to be one), and then by the forced-pad gap closure, which resolved
the divergence that measurement exposed. §4.1 above carries the current state; `.planning/WINDOWS.md`
entry 17 is now `fixed`.

---

## PAR-05 cross-backend comparison — ~~**MISMATCH**~~ **RESOLVED: the serving path now MATCHES** (measured in-container)

> **The two backends preprocess to the same geometry, byte for byte.** On byte-identical input and
> identical recipe values, Isaac-GR00T produces **`(256, 256, 3)`** and Dum-E's LeRobot **serving**
> path produces **`(256, 256, 3)`** — both hashing to `c30150ec…`. Criterion 5's *"identical shape"*
> clause **is satisfied** for the serving path.
>
> ~~**The two backends do NOT preprocess to the same geometry.** On byte-identical input and identical
> recipe values, LeRobot produces **`(256, 340, 3)`** and Isaac-GR00T produces **`(256, 256, 3)`**.
> Criterion 5's *"identical shape"* clause is **not satisfied**, and PAR-05's `Complete` status rests
> on the LeRobot half only.~~ — the state this section recorded before the gap-closure fix below. Still
> true of the **unpatched** upstream transform; no longer true of what the server serves.

Closes assumption **A8** as a measurement, and closes the mismatch that measurement exposed. The
preceding 06-03 note remains accurate *as of plan 06-03* and is superseded; §4.1 above is updated to
match. Harnesses: `scripts/dump_gr00t_native_preprocessed_image.py` (**9/9 checks PASS, exit 0**, re-run
after the fix) and `scripts/dump_preprocessed_image.py` (**9/9 checks PASS, exit 0**). Keyless gate: the
last five tests in `tests/test_par05_image_geometry.py`. Live gate:
`test_live_guard_pass_is_logged_on_the_real_load_path`, which measures the geometry **inside the shipped
image**.

### The comparison

| Backend | Harness | Effective pipeline | Measured output |
|---|---|---|---|
| **LeRobot serving path** `lerobot==0.6.1` | built pipeline → `_transform_n1_7_image_for_vlm_albumentations` | **letterbox-pad-to-square (FORCED)** → resize-shortest-256 → center-crop-95% → resize-shortest-256 | **`(256, 256, 3)`** uint8, `c30150ec…` |
| Isaac-GR00T `23ace64f` | `build_image_transformations_albumentations` (eval) | **letterbox-pad-to-square** → resize-shortest-256 → center-crop-95% → resize-shortest-256 | **`(256, 256, 3)`** uint8, `c30150ec…` |
| ~~LeRobot as configured~~ (unpatched, history) | `_transform_n1_7_image_for_vlm_albumentations` | resize-shortest-256 → center-crop-95% → resize-shortest-256 | ~~**`(256, 340, 3)`** uint8, `e8e4939b…`~~ |

Both measured on the SAME frame: seed **0**, `numpy.random.RandomState(0)`, `480x640x3` uint8,
sha256 `70eecaa5a18341fcf3e9d22091d94d92e9d4c5aef7e839ae94ff3d11fd6612ee`. **That the bytes are
identical is itself a check, not an assumption** (check 3): the two interpreters run numpy 2.2.6 on
Python 3.12 and numpy 1.26.4 on Python 3.10, and the probe fails without the expected digest — a
comparison of two independently generated frames would not be a comparison at all.

The six recipe values are read from the same
`checkpoints/GR00T-N1.7-3B-SO101/processor_config.json` on both sides (check 2, bind-mounted into the
container), so neither side can be running a different recipe.

### The fix, and where it is injected

**One stage changed, at the seam plan 06-01 established.** `docker/lerobot-policy/server.py` merges a
third override into the same `make_pre_post_processors` call that already carries the device and
rename-map overrides:

```python
preprocessor_overrides.update(serving_preprocessor_overrides())
#  -> {"groot_n1_7_vlm_encode_v1": {"letter_box_transform": True}}
```

Four properties of that choice are load-bearing and were selected over the alternatives on purpose:

| Property | Why it matters |
|---|---|
| It is **upstream's own public override seam** (`processor_groot.py:401-455`) | Not a monkeypatch, not a patched private function, not a vendored copy — the sanctioned mechanism this milestone is built around |
| The value has **one definition**, `policy_guard.groot_guard.serving_preprocessor_overrides()` | The guard that asserts the served value and the server that injects it resolve to the same constant, so they cannot drift into two numbers. `policy_guard` is `COPY`-ed into the image, so this holds across the process boundary |
| **Nothing is reimplemented** | Upstream documents its `cv2.INTER_AREA` resize and floored center crop as needing to stay bit-exact (`processor_groot.py:1394-1401`); a hand-rolled pad, resize or crop would have *manufactured* the mismatch this fix removes |
| It **fails loudly** | An override key matching no step raises `KeyError` listing the available keys; an unknown field raises `TypeError` listing the available fields. A pinned-lerobot rename breaks the handshake instead of silently dropping the pad |

**And the injection is verified rather than trusted.** `SAFE-01/5` now reads the **effective**
`letter_box_transform` off the built `GrootN17VLMEncodeStep` — never re-derived from the checkpoint,
which declares the opposite — and refuses the handshake with a named error if it is not `True`. That
assertion is proven RED, not merely green:
`tests/test_groot_guard.py::test_violation_5a2_served_letterbox_off_raises` and
`test_snapshot_from_loaded_reads_the_served_letterbox_off_the_encode_step` drive the
override-did-not-land case against a snapshot built from the real checkpoint. **Changing a guard's
expected value is exactly the edit that can silently defang it**, so the inverted expectation was
required to fail on a wrong geometry before it was allowed to pass on the right one.

**bf16 is undisturbed**, re-measured on the rebuilt image after the change:
`param_dtypes={'bfloat16': 3144016000}`, `cuda_allocated_MiB=6015.0` — identical to plan 06-01's
recorded figures. The three precision knobs (`load_bf16=True`, the `transformers` `dtype`, and
`GrootConfig.model_params_fp32=False`) are untouched.

### Root cause of the original divergence: one pipeline stage, gated on one side and unconditional on the other

- **LeRobot gates the pad on the flag.** `if letter_box_transform:` wraps the `cv2.copyMakeBorder`
  call (`processor_groot.py:1423-1433`). This checkpoint sets `letter_box_transform: false`, so no pad
  runs and the 4:3 aspect ratio survives to the output.
- **Isaac-GR00T pads unconditionally.** `LetterBoxPad()` is element 1 of **both** the train and eval
  albumentations pipelines (`gr00t/model/gr00t_n1d7/image_augmentations.py:420-487`), and
  `Gr00tN1d7Processor` lists `letter_box_transform` under
  `# Backward-compat params (stored but not actively used)`
  (`processing_gr00t_n1d7.py:171-172, 198`) — it is stored on the instance and never read. The frame
  is padded 480x640 → 640x640 first, so the output is square.

**The cause was isolated to exactly that stage, by measurement — this is not an inference.** Isaac's
eval output and LeRobot's output *with the pad forced on* are **byte-identical**:
sha256 `c30150ec8d9d7ccb648aade0588aed2d18a356ab7f984a510dc57bb6c485927f` from both, across
Python 3.10/3.12, numpy 1.26.4/2.2.6 and OpenCV 4.11.0/4.13.0. Every other stage — both `INTER_AREA`
resizes and the floored 95% center crop — therefore agrees bit-for-bit between the backends. **That is
what made a one-stage fix sufficient**, and it is why forcing the pad was a config change rather than a
pipeline rewrite. The UNPATCHED transform's output remains a different artifact, sha256
`e8e4939bac10afa7cced1edb739656e14deb1acbbd509b80dfcfd7810fd2ce5a`, and that difference is asserted so
"the fix changed something" is measured rather than claimed.

**This does not upgrade the fidelity decision in §1.** The cross-container leg stays **shape only**: a
future OpenCV build may legitimately move a pixel, and the bit-exactness above is recorded as an
observation, not promoted to a contract.

### Which side was made to match, and the inference that decision rests on

**Operator decision: match the old path. The LeRobot serving path always pads to square.**

The reasoning: the checkpoint's weights were TRAINED by Isaac's code, so the geometry those weights
actually learned on is Isaac's padded square. LeRobot's flag-honouring behaviour — which reads as more
correct in isolation, since it obeys the checkpoint's own `processor_config.json` — is therefore the
deviation from training-time behaviour, and before this fix it fed the policy a geometry it had never
seen.

**The inference boundary, recorded explicitly because it was accepted knowingly rather than proven:**
*"training used Isaac's geometry"* is inferred from *"Isaac trained the checkpoint"*. **The actual
training recipe was NOT read**, and no local artifact records it — the checkpoint's `config.json`
carries `model_name` only. The operator chose to act on the inference rather than spend a step
confirming it. **If Phase 7's parity work disappoints, this assumption is the first thing to
re-examine.** It is not presented as settled fact here and must not be quietly upgraded to one.

Recorded in code as well as prose, so it travels with the mechanism rather than only with the document:
the module docstrings of `policy_guard/groot_guard.py`, `docker/lerobot-policy/server.py`,
`scripts/dump_preprocessed_image.py`, `scripts/dump_gr00t_native_preprocessed_image.py` and
`tests/test_par05_image_geometry.py` each state it, and both probe manifests carry it as an
`inference_boundary` field.

### Alternatives closed off, so the original mismatch could not be explained away

| Alternative explanation | Ruled out by | Result |
|---|---|---|
| "The eval/train branch was picked wrongly" | check 6 — the **train** pipeline was measured too | `(256, 256, 3)`: the random-vs-center crop changes WHERE, not HOW BIG |
| "The layouts differ, not the geometry" | check 5 — the real serving call site `apply_with_replay` | `(3, 256, 256)` `torch.uint8`: CHW layout, **same H/W** |
| "The GR00T side is nondeterministic" | check 7 — two eval invocations | byte-identical (`numpy.array_equal`) |
| "The recipes differ" | check 2 — six values re-read from the checkpoint | `letter_box_transform=False, crop_fraction=0.95, image_crop_size=[230, 230], image_target_size=[256, 256], shortest_image_edge=256, use_albumentations=True` |
| "The frames differ" | check 3 — input sha256 | identical across both interpreters |

### Provenance: which image, and is it the pin?

**The `23ace64f` pin is PROVEN, but by content — not by anything the image says about itself.**
`gr00t:latest` carries **no `.git`, no `gr00t.__version__`, no OCI revision label and no recorded
build arg** (all four recorded as data in the manifest's `version_markers`), so its revision cannot be
read off the image. What was done instead:

| Evidence | Value |
|---|---|
| Image used | `gr00t:latest`, ID `sha256:e263056fffe7a60a7f48b6309a8b8f2fb3ea9f8f2afa9c94a0105ed5b7d2eeaf`, created 2026-06-15 |
| `gr00t` package digest, in-image and in clone | **`b18c8578c077cddf02705b80da815e5d838752cab9f391c19e4edca7a6e74a40`** over **64** `.py` files — identical |
| Clone state | `/home/aaron/Projects/Isaac-GR00T` at HEAD `23ace64f17aa5015259b8609d371eb61a357c776`, `git diff --quiet 23ace64f -- gr00t` clean |
| `pyproject.toml` / `uv.lock` | byte-identical between image and clone (`9d67b119…` / `ad705311…`) |
| Pin source | `scripts/build_gr00t_image.sh`'s `PIN=`, asserted against the recorded constant by `test_the_gr00t_native_measurement_records_which_image_it_came_from` |

**Stated precisely, so Phase 7 does not over-read it:** the preprocessing source that produced this
measurement is byte-identical to revision `23ace64f`, and so is the dependency lock. What is *not*
established is that the image was **built** by the pin-enforcing wrapper — no build arg or label
records that — and the 64-file content match is what stands in for it. That is a stronger claim than
"unverified" and a weaker one than a signed provenance record; it is recorded as neither.

### How it was run (no GPU, no weights, no token, no server)

`gr00t-server` was **not started** — it remains `Exited` by standing decision. Both the original run and
the post-fix re-run used a throwaway container, and the transform builder was called directly rather
than through `Gr00tN1d7Processor`, whose `__init__` calls `build_processor` and would pull the Cosmos
backbone:

```bash
uv run python scripts/dump_gr00t_native_preprocessed_image.py --emit-expectations \
    --package-dir ../Isaac-GR00T/gr00t          # -> the two expectation digests

docker run --rm --network none \
  -v "$PWD/scripts:/probe/scripts:ro" \
  -v "$PWD/checkpoints/GR00T-N1.7-3B-SO101/processor_config.json:/probe/processor_config.json:ro" \
  --entrypoint python gr00t:latest /probe/scripts/dump_gr00t_native_preprocessed_image.py \
    --recipe-json /probe/processor_config.json \
    --expect-frame-sha256 70eecaa5a18341fcf3e9d22091d94d92e9d4c5aef7e839ae94ff3d11fd6612ee \
    --expect-package-digest b18c8578c077cddf02705b80da815e5d838752cab9f391c19e4edca7a6e74a40
```

`--network none` (the albumentations version-check warning it produces is expected), no `--gpus`, no
published port, no `HF_TOKEN`, `--rm`. `outputs/` was not written to: `--outdir` defaults to writing
nothing, because under `docker run` it would leave root-owned files in the host tree.

### What runs in CI, and what honestly cannot

The GR00T-native **measurement** needs the 42.8 GB image and cannot run in CI. Rather than add a test
that silently skips, the keyless gate pins the *LeRobot-side anchor* of the comparison — which is
executable anywhere the suite runs, because the serving pipeline is built for real and its geometry read
off the constructed step:

| Test | What it pins |
|---|---|
| `test_gr00t_native_geometry_is_256x256x3_and_the_serving_path_matches_it` | The recorded state in BOTH directions: `GEOMETRY_MATCHES_LEROBOT_SERVING is True` **and** `GEOMETRY_MATCHES_LEROBOT_UNPATCHED is False` |
| `test_serving_path_on_480x640_is_256x256x3` | **Executed on the built pipeline**: the served geometry, measured off the constructed encode step |
| `test_serving_pipeline_carries_the_forced_letterbox_pad` | **Executed**: the override landed, and only the pad moved — the other four settings are the checkpoint's |
| `test_serving_letterbox_constant_matches_the_guards_definition` | The probe's pinned copy equals the guard's definition, and the override key matches a real step of the built pipeline |
| `test_lerobot_reproduces_the_gr00t_native_shape_only_with_the_pad_forced_on` | **Executed**: served → `(256, 256, 3)`; as declared → `(256, 340, 3)`, so the agreement is DUE to the override rather than coincidental |
| `test_lerobot_serving_path_reproduces_the_gr00t_native_bytes` | **Executed, byte-exact**: the cross-container digest reproduced locally, plus the shared input digest, plus the unpatched digest as a distinct artifact |
| `test_lerobot_gates_the_letterbox_pad_on_the_flag_this_checkpoint_sets_false` | **AST level**: every `copyMakeBorder` call still sits inside the `if letter_box_transform:` branch — the fix feeds that branch `True`, it does not remove it, and a future upstream hoist would make the override a silent no-op |
| `test_the_gr00t_native_measurement_records_which_image_it_came_from` | Image ID, package digest, file count, and the pin read back off the build wrapper |

**And in the live (opt-in) suite:**
`test_live_guard_pass_is_logged_on_the_real_load_path` asserts the PASS line reports
`served_letter_box_transform=True` against `checkpoint_letter_box_transform=False`, then measures the
geometry **inside the shipped image** via `docker exec` — `(256, 256, 3)`, sha256 `c30150ec…`, on the
image's own numpy 2.2.6 — so the cross-backend agreement is a property of the container that serves, not
only of the host venv.

**What no test in this repo asserts:** Isaac's *unconditional* pad. Importing `gr00t` would skip
everywhere, so that half is carried by the in-container probe and the source citation above. Recorded
as a limitation rather than covered by a test that can never run.

### Consequences

1. **Phase 7's numerical parity gate is now meaningful on this axis.** The two backends see the same
   pixels — byte-identically, on the seed-0 frame — so a chunk comparison measures the policy rather
   than the preprocessing gap. This is precisely why PAR-05 sat in Phase 6 rather than Phase 7. It
   remains true that a parity comparison must re-verify the served geometry before trusting its
   numbers; the live test above is the instrument for that.
2. **PAR-05's verification is satisfied as its text asks.** Its text demands verification *"by
   comparing dumped tensor shapes between backends"*. The comparison has now been performed and
   **passes**, byte-for-byte. `REQUIREMENTS.md` is deliberately NOT edited by this gap closure — what
   `Complete` should mean is a requirements decision, and this document is the evidence it would rest
   on.
3. ~~**Which side is "right" is NOT settled here, and the LeRobot side is not authoritative by
   default.**~~ **Decided by the operator: match Isaac.** The full reasoning and the inference it rests
   on are in *"Which side was made to match"* above. In short: the checkpoint's own
   `processor_config.json` sets `letter_box_transform: false`, which LeRobot honours and Isaac ignores —
   but the checkpoint was *trained* by Isaac's code, so the geometry it saw in training is Isaac's padded
   `(256, 256, 3)`, and LeRobot's flag-honouring path was the deviation from training-time behaviour.
   **Establishing that would still need the training recipe, which was never read.** The decision was
   made on the inference, knowingly.
4. **A related axis, checked only by grep and labelled as such:** the GR00T-native serving path
   (`gr00t/policy/gr00t_policy.py`, `server_client.py`, `eval/run_gr00t_server.py`) shows no camera
   pre-resize, unlike LeRobot's C-3 placeholder-feature resize in §1. Not executed, not a verdict. Note
   that C-3 is now shape-invisible on the served geometry (see §1's
   `serving_corruption_shape_invisible`), so this axis matters slightly more than it did, not less.
5. **Out of scope and deliberately untouched:** `policy/lerobot/features.py:67`'s comment still cites
   the `(256, 340, 3)` verdict when explaining why the client sends 480x640 frames. `policy/` was
   outside this gap closure's declared scope. The frame size it justifies is unchanged and correct; only
   the parenthetical verdict it cites is stale.
