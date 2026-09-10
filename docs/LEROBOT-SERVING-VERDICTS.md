# LeRobot Serving Verdicts

**Status:** PAR-05 **resolved** by executed evidence (§1). LRG-06 **enforced** by a committed test
(§2). The ROADMAP's Phase 6 criteria carry six recorded corrections (§3) and three named open gaps
(§4). Standing engineering resolution — not a changelog entry.

This document records what Phase 6 settled about serving the fine-tuned
`checkpoints/GR00T-N1.7-3B-SO101` checkpoint through LeRobot: the checkpoint's real preprocessed
image geometry, the mechanism that enforces the policy container's loopback-only reachability, the
places where the ROADMAP's stated criteria describe a mechanism the pinned release does not have,
and the gaps that stayed open. It is the resolution referenced by requirements **PAR-05** and
**LRG-06**.

Figures are cited by the harness that computes them so the document and the code cannot drift apart
silently. The harness for §1 is `scripts/dump_preprocessed_image.py`; the always-running gates are
`tests/test_par05_image_geometry.py` (§1) and `tests/test_loopback_publish_spec.py` (§2).

---

## PAR-05 image-geometry verdict

> **A 480x640x3 uint8 frame becomes exactly `(256, 340, 3)` uint8** under this checkpoint's own
> image recipe.

This is a resolution stated as exact integers, not an approximation and not a note. The transform is
a deterministic cv2 `INTER_AREA` resize plus a floored center crop, so there is no tolerance to
allow: `tests/test_par05_image_geometry.py` asserts tuple equality on integers, and the
within-LeRobot comparisons use `numpy.array_equal`, never `numpy.allclose`.

**The effective pipeline** (`processor_groot.py:1435-1468`):

```
resize-shortest-edge-to-256  ->  center-crop-95%  ->  resize-shortest-edge-to-256
```

### The recipe, and where it comes from

Every value below is read from `checkpoints/GR00T-N1.7-3B-SO101/processor_config.json` ->
`processor_kwargs`. `scripts/dump_preprocessed_image.py` pins them as constants and re-reads the file
as its check 1; `test_recipe_constants_match_the_checkpoint_processor_config` re-reads it on every
suite run. A checkpoint swap that changes one of them goes RED rather than silently producing a
different shape.

| Key | Value | Role in the verdict |
|---|---|---|
| `letter_box_transform` | `false` | This checkpoint does **not** take the letterbox branch |
| `crop_fraction` | `0.95` | The center crop; **set**, so it takes precedence over `image_crop_size` |
| `image_crop_size` | `[230, 230]` | **Provably inert** — see below |
| `image_target_size` | `[256, 256]` | Supplies `target_h` as the fallback resize edge only; **not** the output shape |
| `shortest_image_edge` | `256` | The edge both resize passes target — this is why the output HEIGHT is 256 |
| `use_albumentations` | `true` | Selects the cv2/numpy transform; the torch path would give `(3, 256, 256)` |

### Observed shapes

Measured by `scripts/dump_preprocessed_image.py --outdir outputs/par05 --seed 0` (6/6 checks PASS,
exit 0), transcribed from `outputs/par05/manifest.json`. Seed **0**; source frame **480x640x3
uint8**, built with `numpy.random.RandomState(0)` so the bytes are reproducible.

| Case | Input shape | Output shape | dtype |
|---|---|---|---|
| `checkpoint_recipe` | `[480, 640, 3]` | **`[256, 340, 3]`** | `uint8` |
| `placeholder_corrupted` (C-3) | `[224, 224, 3]` | `[256, 256, 3]` | `uint8` |
| `square_256` | `[256, 256, 3]` | `[256, 256, 3]` | `uint8` |
| `letterbox` (`letter_box_transform=True`) | `[480, 640, 3]` | `[256, 256, 3]` | `uint8` |

Two further measured facts from the same manifest:

- **`crop_size_inert: true`** — the `[230, 230]` output and a `[999, 999]` output are byte-identical
  under `numpy.array_equal`. `image_crop_size` is consulted **only** when `crop_fraction is None`
  (`processor_groot.py:1453-1454`), so `[230, 230]` is dead configuration on this checkpoint.
  **Recorded so a future reader does not spend time tuning a value that does nothing.** Proven by
  measurement, not by citing upstream's source.
- **`replay_identical: true`** — two invocations on the same frame are byte-identical. The serving
  path takes the deterministic CENTER crop; the train-time random crop is gated on
  `self.training and torch.is_grad_enabled()` and both are false there.

### Stale framing

The ROADMAP asks to settle *"341x256-crop vs 256x256-letterbox empirically"*. **That framing is a
misread, and `REQUIREMENTS.md:50` already records it as one** — the ROADMAP section never caught up.
The real answer is neither option, and both halves of the framing are real but non-decisive:

- **`256x341` is an INTERMEDIATE**, not an output: it is what the first resize-shortest-edge of a
  480x640 frame produces, before the 95% center crop and the second resize.
- **`256x256` is the LETTERBOX branch**, which this checkpoint does not take
  (`letter_box_transform: false`). It is also reachable *without* letterboxing, from any square
  input — which is why "256x256" on its own identifies nothing.

Both negatives are documented by **named passing tests** rather than omitted, so they cannot be
"simplified" away later:

- `test_letterbox_branch_is_256x256x3_and_this_checkpoint_does_not_take_it`
- `test_square_256_input_is_256x256x3`

Each docstring states that it documents the stale framing and is **never the verdict's evidence**.

### Fidelity decision

| Comparison leg | Fidelity | Why |
|---|---|---|
| Cross-container (LeRobot vs pinned Isaac-GR00T `23ace64f`) | **shape only** | Two different OpenCV builds and two different Python versions (the GR00T container is Py3.10); bit-exactness is not a reasonable contract across them |
| Within-LeRobot (two dumps, same interpreter) | **exact array equality** (`numpy.array_equal`) | Same OpenCV build in the same process, so an interpolation or floored-crop change must be a hard failure, not an approximate pass |

**Shape-only is defensible here for a specific, measured reason, not for convenience.** A shape check
would normally be weak. It is strong on this leg because the corruption it must catch **changes the
shape**: a frame pre-resized to 224x224 by `from_pretrained`'s placeholder feature (C-3 — the
placeholder is `(3, 224, 224)` at `modeling_groot.py:255-261`, and
`prepare_raw_observation` resizes every camera to it at `helpers.py:165-168`) emerges as
`(256, 256, 3)`, **not** `(256, 340, 3)`.
`test_placeholder_pre_resize_corruption_is_256x256x3_and_differs` asserts both the corrupted shape
and that it differs from the correct one, so the discriminating property is itself under test.

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

### 1. The cross-container PAR-05 dump was never attempted (assumption A8)

Dumping the **pinned Isaac-GR00T `23ace64f` image's** own transform output on the same synthetic
frame and comparing shapes was never run. It is carried as an explicit `backstop`, not quietly
omitted.

**What this means, stated precisely:** the LeRobot side is nonetheless **fully settled at
`(256, 340, 3)` by executed evidence**. What does not exist is the *comparison*. This section is a
**one-sided record, not a two-sided comparison**, and the LeRobot-side dump must not be presented as
one (T-06-29). If the pinned image is later available, the procedure is: run the same
`RandomState(0)` 480x640 frame through its own transform, dump `.npy`, and compare shapes against
`outputs/par05/manifest.json`'s `checkpoint_recipe` entry.

The comparison also cannot be run against a live `gr00t-server`: that container is `Exited` by
standing decision for the remainder of Phase 6, so any GR00T-native-side comparison must go through
the mock or dumped tensors.

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
| LRG-02 | "Loads the right checkpoint" is a provenance claim | A five-check refuse-to-**start** preflight that returns before `add_insecure_port`, plus the resolved `base_model_path` asserted from the real load path (06-01, 06-06); the SAFE-01 post-load arming is 06-03's |
| LRG-03 | Relative-vs-absolute is a semantic claim about decoding | Two robot states through one observation: the arm targets track the anchor while the gripper does not, measured against a same-state noise floor (06-01) |
| LRG-04 | Horizon 16 is a config fact, and the emitted length is forced by three truncations | `infer_groot_n1_7_action_horizon(path, "new_embodiment") == 16` asserted at CONFIG level from two independent places, with the tag passed **explicitly** because inference returns `None` here (06-02, 06-06) |
| LRG-06 | A **deployment-shape** risk (what the run command publishes), not a shape-of-data risk | §2 above: the publish-spec test with three synthetic non-vacuity proofs (06-05) |
| SAFE-01 | Five serving-contract assertions are semantic, not dimensional | One pure validator over a frozen 16-field snapshot, with `SAFE-01/N`-prefixed errors proven to fire on twelve single-field mutations of the REAL checkpoint's snapshot (06-02); arming it on the real load path is 06-03's |

### 4. Recorded, non-blocking: `outputs/` is not gitignored

`scripts/dump_preprocessed_image.py` writes to `outputs/par05/`, and those artifacts are
**deliberately not committed**. `outputs/` is still absent from `.gitignore` (Phase 5 deferred item
#1, a different axis, deliberately not folded into this phase). The numbers that matter are
transcribed into this document, which **is** committed, and
`tests/test_par05_image_geometry.py` re-derives them from the checkpoint on every suite run — so the
committed verdict has a live guard behind it rather than resting on a stale `.npz` (T-06-30).

---

## References

- `scripts/dump_preprocessed_image.py` — the PAR-05 harness (keyless, GPU-free, weight-free)
- `tests/test_par05_image_geometry.py` — the always-running geometry gate (7 tests, never skips)
- `tests/test_loopback_publish_spec.py` — the LRG-06 publish-spec guard (6 tests, never skips)
- `README.md` — the documented `lerobot-policy` run command, under the LRG-06 anchor
- `docker/lerobot-policy/entrypoint.py` — the five-check refuse-to-start preflight and the
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
INFO 2026-09-10 05:16:39 y/server.py:623 SAFE-01 guard: PASS | base_model_path=/checkpoints/model | embodiment_tag=new_embodiment | actions_per_chunk=16 | checkpoint_horizon=16 | decode_step=GrootN17ActionDecodeStep
```

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

### Section 4.1 of `## Open gaps carried forward` is unchanged by this plan

The cross-container PAR-05 dump from the pinned Isaac-GR00T `23ace64f` image (assumption A8) was
**still not attempted** and remains a one-sided record, not a two-sided comparison. Plan 06-03 did not
absorb it and does not mark it resolved: `gr00t-server` is `Exited` by standing decision for the
remainder of Phase 6, so any GR00T-native-side comparison must go through the mock or dumped tensors.
It stays open in `.planning/WINDOWS.md` (entry 17).
