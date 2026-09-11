# Checkpoint parity: local numerical instrument

## Physical review ready — 2026-09-11T22:54:59.431070+00:00

Active evidence workspace: `corpus/phase7-small-gpu-20260911`. It is an explicitly documented immutable evidence view of the original session, with separate retry artifacts. `retry-evidence-view.json` preserves the failed launcher attempt and predecessor identities. The original strict report remains unchanged.

- Relaxed physical-unit acceptance passed, SHA `7ec01b1dcf9605914fa81035df137aa2d26ff41a4a5d19720dde1e215346d3fd`.
- Native reference candidate SHA `36431859e9d9b23923254b610842928b3815be08f816c78868dca6ec40b780a7`. Fresh 12-case GPU replay passed with raw/noise/decoded maximum error exactly zero, SHA `2c28a4876778e74b8a297e79f6c0a630c6ae9a5534231c7cd2f055f762448185`; runtime about 28 seconds. Includes 12 observer-control calls, no CPU model execution.
- Pinned LeRobot serving image `sha256:6758186bd24cd0745b7442dafbb6680cbc2a986fe387eb2de5aaf0b75dce9c98` is running as `dume-milestone-serving-01`, loopback `127.0.0.1:8080`. No hardware devices are mounted. Fresh arm-free serving attestation passed, SHA `2ec5c8ba2d5283f3267982bb0f5ce60b9d8a37eaa8966296bde9b4a63d6a44e3`. Runtime snapshots use `runtime-live/lerobot.json` in this workspace.
- Actual review preflight `preflights/review-0001.json` passed, SHA `f6a14f8e90710c675e53cc60f16c29fbc56127fa4fdb06e0f65ecae586b49e01`. Exact review bundle `release-review.json` SHA `04e12d94d1e31cdc3e284c77a3b9b00cef16d7a4a04c8184646be89d58f91244` is ready.
- User reconnected the devices; `/dev/ttyACM0`, `/dev/video0`, `/dev/video2` are present. No controller has been constructed or connected; no robot motion occurred.
- Next gate: explicit combined approval of this native reference and the fixed three-trial physical test, with operator present, banana in reach, clear workspace and accessible stop. The prior request to prepare/proceed is preserved, but no exact-bundle approval or presence confirmation is fabricated. After that decision, record it canonically, collect a fresh `live` preflight, then run the guarded physical protocol in a PTY so stop and per-trial observations remain interactive.

Review and upcoming guarded commands (do not execute motion before that decision):

```sh
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/approve_parity_evidence.py live --workspace corpus/phase7-small-gpu-20260911
DUME_POLICY_BACKEND=lerobot UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/run_checkpoint_sanity.py preflight --workspace corpus/phase7-small-gpu-20260911 --stage live --attempt 1 --container dume-milestone-serving-01 --endpoint 127.0.0.1:8080 --attestation corpus/phase7-small-gpu-20260911/runtime-live/lerobot.json
DUME_POLICY_BACKEND=lerobot UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/run_checkpoint_sanity.py run --workspace corpus/phase7-small-gpu-20260911 --preflight preflights/live-0001.json --preflight-attempt 1 --container dume-milestone-serving-01 --endpoint 127.0.0.1:8080 --attestation corpus/phase7-small-gpu-20260911/runtime-live/lerobot.json
```

Software validation: 11 scoped comparison/acceptance tests, 12 scoped native tests, 25 integration tests, 31 guard regressions passed. Read-only review found no blocking issue. The native launcher originally failed before model import on a Python3.10-incompatible eager host-helper import; that attempt is preserved, and the lazy-import fix passed the real GPU replay. Source commits: `25bd555`, `6c3da4b`, `71e35cf` (following reducer `96f95b1`). No fresh package install or upstream/model modification.

## Current acceptance and physical-test preparation (2026-09-11)

The operator reviewed the approximately one-degree discrepancy and explicitly asked to relax the milestone criteria and proceed to physical testing. The resulting `milestone-criteria-authorization.json` records this **retrospective** instruction. The original strict report remains failed and unchanged; the separate `milestone-acceptance.json` passes calibrated physical-unit limits.

| Metric | Five arm joints | Gripper |
|---|---|---|
| Maximum absolute difference | 2 degrees | 1 normalized point |
| Mean absolute difference | 0.5 degrees | 0.25 normalized points |
| Absolute mean signed bias | 0.5 degrees | 0.25 normalized points |

Measured arm maxima are 0.605–1.017 degrees, mean absolute differences 0.122–0.194 degrees, and maximum absolute signed bias 0.079 degrees. Gripper maximum is 0.1773 points. Raw/input representation differences and chunk-index slope remain diagnostics. These limits are acceptance criteria, not a collision-safety envelope or a measured Cartesian-error bound.

The small native reference and one fresh 12-case GPU regression must validate before review readiness. A single explicit final review will bind the exact scoped native reference and the physical test together. No prior golden approval is fabricated. The physical protocol is three banana-directed trials of 20 × 16 actions each; at least two must move coherently toward the target, grasp optional. Existing calibration, runtime freshness, reset-inclusive zero-clamp checks and latched stops remain required. Operator presence, a clear workspace and accessible stop controls must be confirmed before construction/connection/reset. No hardware has been accessed during this preparation.

The new release branch uses this explicitly declared milestone evidence. The archived exhaustive gate remains separate; it has not passed and will not be rerun on CPU. Implementation and validation details follow below as history.


## Active milestone scope: 12 observations, one seed (2026-09-11)

The operator cancelled the exhaustive 7,200-inference matrix and prohibited CPU model inference, then explicitly selected **12 observations, one seed**. Select the middle stored observation from each of the 12 recorded episodes and its first locked seed. Both configurations use deployed GPU BF16. This section supersedes the exhaustive workflow documented below, which remains historical context.

The check reuses the preserved native/LeRobot operational GPU captures. It launches **zero new inference runs**; original source evidence and approved numerical bounds remain unchanged. The original full experiment was cancelled, not passed. Its stock producer/consumer result remains a separately preserved pass.

- Frozen selection: `corpus/phase7-stock-tests-20260911/milestone-scope.json`, SHA-256 `fae6835e8df81b453e2bbc19fe9cdbf64982af25d3569982ea4ce2d782172ba2`.
- Scoped result: `corpus/phase7-stock-tests-20260911/milestone-report.json`, SHA-256 `975b291c46dcbb30e8f7cf097e32dcab3e805f88a32fd314900c937fca5c2af5`.
- Result: **failed**, 12/12 selected pairs evaluated. Largest decoded difference: **0.985710 percentage points**, against the unchanged **0.1-point** maximum bound. Mean absolute error by joint ranges from **0.057775 to 0.182683 points**, against **0.05**. Signed bias and chunk-index slope bounds also fail. Full per-joint/per-index and per-trace matrices are linked from the report.
- Raw maximum absolute difference: **0.0370953**; the approved raw atol/rtol check fails. All twelve input comparisons report value and schema differences. These are separately reported; differing backend input schemas alone do not prove a behavioral bug.
- Native sampler noise is BF16; LeRobot sampler noise is FP32. Captured noise differs even with the selected equal seed. Native attention is FlashAttention2; LeRobot attention is SDPA. The report uses the original independent-noise rule and does not claim that it isolates the cause of output differences.
- Reduction runtime: **13.55 seconds**. Both original operational GPU captures had already completed 600 cases, in approximately 207/215 seconds; no CPU inference is needed to use this evidence.
- No common-input isolation rerun, exhaustive precision bridges, reviewed native golden, release readiness, or hardware authorization is claimed. Do not loosen thresholds or choose different samples in response to these results.

The reducer is `scripts/check_milestone_parity.py`; its seven focused tests cover sample membership, partial coverage, local bias cancellation, slope, exact tokens, raw shape/finite values, differing noise, and input-schema reporting. Existing sealed numerical/worker source files were not changed. A read-only review found no remaining blocking issue. The first arithmetic attempt refused differing input keys; it is retained as `milestone-reduction-attempt-01.json`, and the final reducer reports those differences instead of discarding output metrics.

Initial execution (writes the report once and refuses overwrite):

```sh
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/check_milestone_parity.py --workspace corpus/phase7-stock-tests-20260911
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run pytest tests/test_milestone_parity.py -q
```

Phase 07 remains incomplete because the selected operational check exceeds the agreed bounds. Investigate sampling/preprocessing differences within the small GPU-only scope; any intentional numerical-acceptance change must be explicit and prospective, preserving this failed report. Golden and live steps remain pending.

## Historical exhaustive instrument

Phase 07 Plan 04 Task 1 implements the instrument. Its tests use fabricated
arrays and tiny fabricated Torch modules only. No real cross-backend comparison,
stock producer/consumer experiment, repeatability measurement, tolerance agreement,
or physical approval is established by those tests.

Task 2 waits for parent review and confirmation that Plans 05/06 and all relevant
sources are stable. The parent owns global planning state. Task 3 is an explicit
`blocking-human` tolerance decision; there is no default agreement.

## Experiment order

1. Create an explicit successor workspace with immutable `session.json`,
   predecessor path/hash, and reason. Derive current calibration afresh.
2. Run fresh feasibility for this instrument, then the fixed same-backend
   repeatability schedule, with one accelerator process at a time.
3. Inspect and bind an external unmodified LeRobot v0.6.1 harness checkout.
   Measure a fresh arm-free LeRobot request and archive its host, request, and
   attestation as `arm-free-attestation.json`.
4. Prepare a numerical proposal from same-backend observations and stock
   semantics. Present actual measured values and the proposed bounds.
5. Obtain explicit operator agreement binding that exact proposal.
6. Execute both stock stages, then full frozen-corpus comparisons and every
   operational bridge. Recheck the archived numerical evidence.
7. Continue the separately owned golden and release-review workflow. A numerical
   pass is not approval to move hardware.

The original `corpus/phase7-final-20260911` feasibility is historical. Its profiles
predate subsequent source changes and cannot be reused as current experiment
profiles. Preserve it, all earlier attempts, the frozen corpus, and the existing
`seed_verdict: not-honored`.

## Profiles and controls

Diagnostics require actual fp32 parameters, buffers, floating inputs, backbone,
sampler and compute; SDPA; eval mode; disabled conflicting autocast/TF32; four
observed flow steps; full `(1,40,132)` raw/noise; and `(16,6)` decoded actions.
Casting returned values does not satisfy precision.

The last actual feasibility measured both diagnostics on CPU, native operational
bf16 on CUDA with `flash_attention_2`, and LeRobot operational bf16 on CUDA with
SDPA. These are historical observations, not defaults to silently impose on a
new operational experiment. The new experiment binds each independently measured
profile and preserves its actual attention/device/dtype configuration.

The LeRobot operational profile records the complete shared model/policy/processor
configuration and an explicit ambient/fixed deployed seed policy. Replay seeding
at the sampling boundary is a separate intervention. The serving attestation
must independently match those semantics; copying an attestation into purported
replay measurements is not evidence.

Repeatability uses records 0000/0060/0080/0090/0100/0119. For each of four profiles:
five first-seed warm repeats per record, one changed-seed control, and two new
processes repeating the first seed per record. There are 48 cases and 13 process
groups per profile. Decoded outputs, full noise, actual process identities,
raw/preprocessing variation, signed bias and slopes are archived. No different
backend outputs are paired during this reduction.

## Commands and immutable artifacts

Use the existing pinned environment, with no implicit synchronization/download:

```sh
export UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never
```

All commands run from the project root. Substitute one explicit new workspace
consistently; `corpus/phase7-instrument-20260911` below is an example name, not an
assertion that its prerequisites or measurements exist.

After the parent permits Task 2 and the successor session/calibration exist:

```sh
uv run python scripts/replay_checkpoint_parity.py feasibility \
  --workspace corpus/phase7-instrument-20260911 \
  --diagnostic-device cpu --stock-device cpu
uv run python scripts/replay_checkpoint_parity.py repeatability \
  --workspace corpus/phase7-instrument-20260911 --diagnostic-device cpu
uv run python scripts/replay_checkpoint_parity.py check --stage repeatability \
  --workspace corpus/phase7-instrument-20260911
uv run python scripts/replay_upstream_parity.py pin \
  --workspace corpus/phase7-instrument-20260911 \
  --source /tmp/lerobot-v0.6.1-source
```

`pin` reads an existing external checkout at immutable commit
`7e241bd630a3719a56157a497ce5d08f244784f1`; it verifies both source files against
the git object and binds their SHA-256 hashes. It neither installs packages nor
runs a model. The checkout must remain outside this project's source tree.

Archive the actual arm-free request through Plan 03's attestation APIs before:

```sh
uv run python scripts/replay_checkpoint_parity.py prepare-tolerances \
  --workspace corpus/phase7-instrument-20260911
```

The proposal is not approved. Stop and present its measured basis and exact
digest. Only an actual affirmative operator response may be recorded through
the canonical `approve_parity_evidence.py tolerances` command.

After that blocking decision:

```sh
uv run python scripts/replay_upstream_parity.py run \
  --workspace corpus/phase7-instrument-20260911
uv run python scripts/replay_upstream_parity.py check \
  --workspace corpus/phase7-instrument-20260911
uv run python scripts/replay_checkpoint_parity.py full \
  --workspace corpus/phase7-instrument-20260911
uv run python scripts/replay_checkpoint_parity.py check --stage offline \
  --workspace corpus/phase7-instrument-20260911
```

Numerical source hashes are sealed in the repeatability/proposal. Relevant source
changes require fresh repeatability and explicit agreement in a preserved
successor session. Completed stage names are never overwritten. Missing
prerequisites return exit 2 / `not_run`; failed evidence returns 1; success returns
0 only for the requested completed stage.

## Numerical definitions

The complete ordered frozen manifest defines 120 records × five seeds = 600
cases per profile/tier. Each record's instruction is retained. The instrument
rejects missing, duplicate or permuted cases, wrong camera/joint order, stale
inputs, nonfinite values, unequal full shapes, and incomplete metadata before
arithmetic. Tokens and masks compare exactly.

The diagnostic delta is LeRobot minus native. All named comparisons use right
minus left, with the explicit profile pair in the report. Reductions use float64;
the original observed inference dtype remains metadata. The signed matrix retains
case identity, all sixteen chunk indices and all six joints.

For each joint the report retains mean/max absolute error, aggregate signed bias,
per-index signed means, each trace's signed bias, and each trace's OLS slope
against indices 0..15. Both trace and aggregate bias/slopes are gated. Opposite
trace offsets or ramps cannot cancel into acceptance. Arm units are checkpoint
percent; gripper units are gripper percent; slopes are those units/action index.

Preprocessing floating atol/rtol, raw atol/rtol, decoded mean/max, signed-bias,
slope and native-golden thresholds are distinct. The proposal algorithm uses
four times the largest corresponding measured same-backend variation, with
explicit prospective floors. Raw atol/rtol start at the stock `1e-3` defaults.
Decoded floors are 0.1 maximum, 0.05 mean, 0.02 bias percent and 0.002
percent/action-index slope. These are reviewable proposals, never preapproved
values or bounds fitted to cross-backend residuals.

Three additional full-corpus comparisons cover native diagnostic → native
operational, LeRobot diagnostic → LeRobot operational, and native operational →
LeRobot operational. Their independently sampled noise is explicitly labelled.
Equal integer seeds do not prove cross-dtype noise equality. The proposed rule
retains every trace's error/bias/slope plus aggregate bounds; it can fail, and
failure cannot trigger automatic loosening.

## Full and stock execution

`full` first collects four independent complete paths, then executes both sides
of each of four common-input raw comparisons: twelve serial 600-case worker
runs. Common input archives retain all numeric model keys and original dtypes.
Each side's decoded witness comes from its own complete preprocessing/decoding
path. Common-input raw/noise witnesses are separately captured. The reducer and
archived recheck are the same implementation used by fabricated tests.

Each comparison archives separate independent/common bundle references for both
sides. Every bundle binds an immutable launch, schedule, worker manifest and its
durable case captures. Validation checks the measured profile, record digest and
instruction, process identity, serial chronology, current instrument hashes and
all captured inputs. Each numerical launch has a unique execution ID and schedule
binding; every durable case records that execution and its capture start/end.
Case times must fit the worker lifetime in order, so redating a worker around
older captures fails. It reconstructs aggregates from those cases before checking
the signed matrix and thresholds. Removing worker files while retaining aggregate
arrays cannot pass.

Repeatability uses the same worker validation. Its decoded/noise aggregates,
signed statistics and raw/preprocessing maxima are recomputed from captured
within-profile repetitions before a proposal can use them.

The dependency-layer functions `parity_gate.validate_offline_evidence` and
`validate_upstream_evidence` are shared by the CLI and canonical release gate.
Release requires stock JUnit and runtime coverage, full collated inputs, signed
matrices and complete worker provenance. Test-only fixtures satisfy these schemas
but remain incapable of authorizing production release.

The stock wrapper executes the unchanged producer and consumer in their
respective existing containers. It uses seed 42, `new_embodiment`, the SO101
checkpoint, and agreement-bound raw bounds. A private fresh producer directory
and hash lock delimit stock's pickle-enabled inputs; new evidence is numeric-only
NPZ and JSON.

A Python profiling context observes the top-level stock `get_action` boundary;
Torch function/dispatch modes and temporary forward hooks observe actual noise,
compute and steps. It compares an observer-disabled same-backend control while
restoring RNG state. It never replaces an upstream function or modifies upstream
source. Unsupported observation, wrong controls or infeasible two-model execution
cannot pass.

Stock producer exit zero alone is insufficient. The expected artifact and
positive tag summary are required. Consumer JUnit and collected/setup/call/teardown
identities must show exactly the expected test, with no skip, xfail, xpass or
missing execution. Actual pre-crop outputs must both be `(2,40,132)`. Full input
and sampler-noise witnesses must match. A shared-prefix match cannot pass.

Ground truth and historical stochastic mean/spread are archived separately.
They never override controlled parity. Accepted forced-letterbox geometry,
historical training geometry, backbone provenance, and current calibration
caveats remain visible rather than being represented as newly proved facts.

## Hermetic verification

```sh
uv run pytest tests/test_checkpoint_parity.py tests/test_parity_gate.py \
  tests/test_upstream_parity.py -q
```

Tests explicitly use `Evidence(test_only=True)` and injected worker/subprocess
collaborators. CLI paths cannot enable that fixture mode. A synthetic 600-case
report proves orchestration membership and refusal behavior; it is not actual
600-case inference evidence.

Canonical closeout also calls the pure
`parity_gate.validate_run_safety_journal(run)` immediately after reading the live
run. It validates the journal hash chain and reconciles stop entries, clamp
counts, stop reasons and safety flags. A preserved stop event cannot be hidden
by changing report or trial summaries. The runner uses the same validator, and
the gate imports no controller or motor modules.

Operational replay preserves the deployed runtime's TF32 flags; only diagnostic
replay and stock diagnostics force TF32 off. Read the flags from the measured
profile and independent serving attestation. An image default such as cuDNN TF32
enabled must not be silently changed to make those records agree. A source or
profile correction requires a new successor session, fresh calibration derivation,
fresh feasibility profiles and the complete repeatability schedule.

Replay profiles, stock observations and serving semantics bind
`tf32_matmul` and `tf32_cudnn` independently. Both must be actual booleans; the
retained `tf32` summary must equal their OR and cannot establish equivalence.
Changing matmul TF32 while cuDNN TF32 stays enabled changes the operational
configuration and invalidates the attestation. Diagnostics require both flags
false. Older evidence lacking the pair is historical context, not a current
profile or release witness.


## Measured proposal and subsequent agreement

Workspace: `corpus/phase7-tf32-pair-20260911`, a fresh successor linked through
`session.json` to the retained earlier attempts. All measured instrument files
match reviewed source at `2dc8cc6`. Calibration was freshly derived through the
canonical resolver; no robot was constructed or connected.

Proposal file: `corpus/phase7-tf32-pair-20260911/tolerance-proposal.json`.
The exact file SHA-256 used by the decision writer is:

```text
579e1d8361f71b9a514ad1f97cdc4364c6f21e56fb41627f2db3c4de435c5266
```

The fixed schedule completed **192 real cases in 52 serial processes** from
2026-09-11 16:10:40 to 16:32:39 UTC. Each profile completed 48 cases in 13
processes: six fixed records, five same-seed warm repeats and one changed-seed
control per record, then two independent same-seed cold processes per record.
All 24 changed-seed controls passed; every same-seed raw, preprocessing and
decoded deviation was zero, including per-joint and per-trace bias/slope metrics.
The independent `check --stage repeatability` reconstructed and validated the
worker/case witnesses, process identities, chronology, statistics and source hashes.

| Measured profile | Device / parameters | Attention | TF32 matmul / cuDNN |
| --- | --- | --- | --- |
| Native diagnostic | CPU / FP32 | SDPA | false / false |
| LeRobot diagnostic | CPU / FP32 | SDPA | false / false |
| Native operational | CUDA:0 / BF16 | FlashAttention 2 | false / true |
| LeRobot operational | CUDA:0 / BF16 | SDPA | false / true |

Both diagnostics observed only FP32 compute; both operational profiles observed
BF16/FP32 compute. All four observed four flow steps. CPU diagnostics are the
measured fallback: checkpoint FP32 parameter storage is 12,576,064,000 bytes,
greater than the GPU's measured 12,487,294,976-byte capacity.

Before the long schedule, a real frozen-record gRPC request independently matched
the LeRobot operational configuration, ordered processors, both TF32 flags and
ambient deployed seed policy. It returned the complete `(16,6)` chunk, with one
`(1,40,132)` sampler draw, four flow steps and 196 observed SDPA calls. This
attestation is retained historical measurement evidence; later live release
still requires its own fresh request and host observation.

The unchanged pinned stock producer also completed on CPU, retaining its two-model
lifetime and dumping exactly `new_embodiment` at seed 42. Observation verified
full `(2,40,132)` raw/noise tensors, actual FP32/SDPA/four-step computation,
both TF32 flags off, and observer inertness. Its consumer has not run.
Producer feasibility uses its own directory and does not replace the future
agreement-bound stock producer/consumer execution.

All four comparison scopes—diagnostic, native precision bridge, LeRobot precision
bridge and operational—propose the following bounds for every joint:

| Boundary / metric | Proposed bound |
| --- | --- |
| Floating preprocessing | atol `1e-6`, rtol `1e-6` |
| Tokens, masks and categorical data | exact equality |
| Full raw output | atol `1e-3`, rtol `1e-3` |
| Decoded maximum absolute error | `0.1` |
| Decoded mean absolute error | `0.05` |
| Absolute decoded bias, individual trace and aggregate | `0.02` |
| Absolute OLS slope over indices 0–15, individual trace and aggregate | `0.002` per index |
| Native operational golden replay | atol `1e-5`, rtol `0` |

Decoded units are checkpoint percent for each of the five arm joints and gripper
percent for the sixth. The rule is four times the largest relevant same-backend
measurement with the stated floors. Since all observed repeatability deviations
were zero, the floors determine this proposal. Raw floors follow the pinned
stock defaults; decoded floors are an explicit prospective engineering budget,
requiring human agreement. The factor of four is not a confidence interval.
For the current arm calibration, the `0.1` percent maximum-error floor corresponds
to approximately 0.0966–0.18 degrees, depending on the joint; this arithmetic
interpretation grants no motion permission.

Diagnostic noise must match exactly in the eventual comparison. Operational
bridges retain independent noise and all individual/aggregate error bounds;
equal integer seeds do not establish cross-dtype noise equality. These strict
bounds may fail under independent noise, and no adaptive loosening is authorized.
The six-record repeatability schedule does not establish actual 600-case
cross-backend coverage or parity. Historical `seed_verdict: not-honored`,
unknown training calibration/backbone provenance, and forced-letterbox geometry
caveats remain unchanged.

The Plan 04 blocking-human numerical decision was resolved by Aaron: "Approve,
but we may adjust them later upon more experiments". The canonical agreement
was recorded at `2026-09-11T17:33:05.244020+00:00`, SHA-256
`2e6459457fc06072490b159aeaebcfe1e60a84060ce97b9986540f74a8fa20c8`.
Its tolerance-stage check passed. Future adjustments require a new documented
proposal and explicit approval before a NEW comparison. This agreement grants
neither golden nor physical approval.

## Plan 07 actual stock attempt: execution prerequisite failure

Workspace: `corpus/phase7-tf32-pair-20260911`. Starting HEAD was `bb08455`;
all ten sealed instrument files remained unchanged. The external harness was
pristine at `7e241bd630a3719a56157a497ce5d08f244784f1`. Both stock stage processes
were invoked serially by the unchanged strict wrapper, with a fresh private
`stock/producer` directory. Every inference container had network disabled and
read-only source/input mounts; no serial or camera device was mounted.

Commands actually executed:

```sh
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/approve_parity_evidence.py check --stage tolerances --workspace corpus/phase7-tf32-pair-20260911
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run pytest tests/test_upstream_parity.py tests/test_checkpoint_parity.py -q
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/replay_upstream_parity.py run --workspace corpus/phase7-tf32-pair-20260911
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/replay_upstream_parity.py check --workspace corpus/phase7-tf32-pair-20260911
```

| Stage/check | Actual result |
| --- | --- |
| Canonical agreement | Exit 0, complete |
| Hermetic wrapper/comparator tests | Exit 0; 62 passed, zero skips, 78.99 seconds |
| Fresh stock producer | Exit 0; exactly one new_embodiment artifact at seed 42 |
| Stock consumer process | Exit 1: No module named pytest |
| Strict wrapper | Exit 1; upstream-result.json is failed |
| Archived stock check | Exit 1: required evidence is failed or incomplete |
| Full corpus and operational bridges | Not run: required stock gate failed |
| Golden generation/approval/promotion/replay | Not run: dependent gate blocked |
| Hardware | No access or approval |

The wrapper ran from `17:39:20.022797` to `17:40:14.230129` UTC on September 11,
2026, after agreement. Producer process time was 38.390 seconds
(`17:39:25.158266`–`17:40:03.547950`); consumer process time was 10.547 seconds
(`17:40:03.681442`–`17:40:14.228694`). These include setup, not just inference.
No new peak-memory telemetry was collected; launcher limits are not measurements.

The producer saved 6,343,453 bytes and independently observed full `(2,40,132)`
raw output/noise. Parameters, buffers, floating inputs, backbone and compute were
FP32 on CPU, with SDPA (244 observed calls), four flow steps, one sampler draw,
eval mode, no autocast, both TF32 flags false, and exact observer-disabled
same-backend control equality. Bootstrap FlashAttention warnings remain in its
log; the fair model inference controls were separately observed and validated.

The consumer stopped at the wrapper import before stock test collection or
consumer model loading. No consumer JUnit, coverage, raw/noise observation,
paired-noise measurement or cross-backend residual exists. The wrapper correctly
blocks release with its preserved failed status; consumer inference itself was
**not run**. This is an execution-prerequisite failure, not evidence of a numerical
parity failure or of tolerances being too strict.

| Artifact relative to this workspace | SHA-256 |
| --- | --- |
| upstream-result.json | 24f1d992acf64ca7fa4cc5ccb63f1cbfec80439978495169d29320588c28c8a7 |
| stock/producer/original_n1_7_new_embodiment.npz | d6e308ee56e76d86f4e28aef16d6dda0ea7b5a0f2002e43d4cac56c516313515 |
| stock/producer-observation.json | bf4021e8a6809ee135a89615d9ee17f98d7d7e4bbc537cb6cbdd16a424da2349 |
| stock/consumer.log | 979a601c2ce05fca18710f587f3536aae7b7ffbf16c4280882fea01b8c395c03 |

The result records exact Docker argv, allowlisted environment, image IDs,
chronology and log hashes. `plan07-execution/` retains tests, command results,
read-only environment/cache diagnosis and post-failure checks. Containers exited
and no GPU compute process remained.

Read-only inspection confirmed no alternate pytest/interpreter path in the exact
consumer image `sha256:5da8133fa5669c2132325dc8aa68063dd70c3186729a28efd0ab876fe5ac725d`.
The missing packages are pytest, pluggy and iniconfig. Existing packaging 25.0
and Pygments 2.21.0 satisfy pytest requirements; all 113 existing distributions
pass pip check. Native/host test environments have different identities and were
not substituted. Exact host-lock wheels for pluggy 1.6.0 and iniconfig 2.3.0
exist in the pip cache; pytest 9.1.0 has an expanded uv cache but its original
lock-hashed wheel was not found. No package was installed or downloaded.

The concrete review proposal is `.planning/phases/07-checkpoint-parity-gate/07-07-REMEDIATION.md`:
a derived image adding only pytest 9.1.0, pluggy 1.6.0 and iniconfig 2.3.0, with
exact wheel hashes, cache paths, proposed Dockerfile/build commands and inventory
checks. It requires explicit approval for the additions and the missing wheel
acquisition. No source/image/profile remedy has been applied.

A changed image/package inventory changes the approved subject. After a reviewed
remedy, preserve this failed workspace and create a linked successor with fresh
canonical calibration, feasibility/profiles, repeatability, real arm-free
attestation, a new documented proposal and explicit numerical approval before
new stock comparisons. Never rerun into these immutable stage paths, bypass
stock into the full corpus, or reinterpret old bounds/results. Golden approval
remains a separate later blocking-human gate.


## Approved test-image remedy and fresh evidence session

Aaron’s subsequent “Go ahead” authorized the prepared packaging remedy and fresh
measurements. The parent built and verified
`lerobot-policy:phase7-stock-tests-20260911`, immutable image:

```text
sha256:6758186bd24cd0745b7442dafbb6680cbc2a986fe387eb2de5aaf0b75dce9c98
```

The derived image retains all 113 existing distribution versions and adds only
pytest 9.1.0, pluggy 1.6.0, and iniconfig 2.3.0. The parent inventory comparison
and pip check passed; `lerobot-policy:latest` still identifies the old image.
Build evidence is
`.planning/phases/07-checkpoint-parity-gate/execution-attempts/07-07-test-image-build.json`.
No further download, installation, image build, or source correction was needed
for this refresh. Packaging approval does not constitute numerical approval.

The new exclusive workspace is `corpus/phase7-stock-tests-20260911`. Its canonical
calibration/session writer links directly to the preserved failed
`upstream-result.json` in `corpus/phase7-tf32-pair-20260911`. The old proposal,
agreement, and failed stock result remain byte-unchanged. All ten original sealed
instrument files still match their original hashes. Global planning state remains
parent-owned.

Fresh feasibility passed for both CPU FP32/SDPA diagnostics, both CUDA:0 BF16
operational profiles, and native two-model CPU stock capacity. Diagnostics
observed TF32 matmul/cuDNN false/false; operational profiles observed false/true.
Native operational attention remains FlashAttention 2 and LeRobot operational
attention remains SDPA. All four inference profiles observed four flow steps.
Every LeRobot profile explicitly binds the new image.

The reused arm-free driver independently exercised that same image through the
actual loopback gRPC server. Its frozen-record request ran from
`2026-09-11T18:11:45.598962+00:00` to `2026-09-11T18:11:52.528756+00:00`,
returned a finite `(16,6)` chunk, and matched the complete operational
configuration, ordered processors, ambient deployed seed policy, and both TF32
flags. Observation captured one `(1,40,132)` FP32 CUDA sampler draw, four flow
steps, and 196 SDPA calls. The container was stopped after attestation.

The unchanged producer-only feasibility run completed from
`2026-09-11T18:12:04.991835+00:00` to `2026-09-11T18:12:43.090962+00:00`.
It retained its two-model lifetime, used CPU FP32/SDPA, seed 42 and exactly
`new_embodiment`, and passed the strict full-shape/control observer checks.
This remains producer feasibility; any later approved stock comparison must
run both stages afresh.

Before the long schedule, a GPU-free, network-disabled container using the new
image successfully imported the test dependencies and `GR00TN17`, then ran
pytest with `--collect-only` against the unchanged pinned consumer. It collected
exactly `test_groot_get_action_parity[new_embodiment]`: **one test collected,
zero tests executed, zero model loads**, exit 0, in 6.795 seconds including
container/import startup. The fresh producer artifact was mounted read-only
for filename discovery. No fixture setup or consumer inference occurred.
Exact argv and output are in `execution/consumer-collection.json` and its log.

The full repeatability command selects both immutable image IDs explicitly:

```sh
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/replay_checkpoint_parity.py repeatability \
  --workspace corpus/phase7-stock-tests-20260911 \
  --diagnostic-device cpu --stock-device cpu \
  --native-image sha256:e263056fffe7a60a7f48b6309a8b8f2fb3ea9f8f2afa9c94a0105ed5b7d2eeaf \
  --lerobot-image sha256:6758186bd24cd0745b7442dafbb6680cbc2a986fe387eb2de5aaf0b75dce9c98
```

The first native warm group began while the existing hermetic readiness suite
was running. Following the parent sequencing instruction, only its coordinator
was held; the model worker finished all 36 cases uninterrupted at
`2026-09-11T18:18:36.012527+00:00`. No container remained active. Its exited
Docker client appeared as a zombie solely because the held coordinator had not
reaped it. `execution/readiness-launch-hold.json` and
`execution/warm-completed-during-hold.json` preserve the timing and exact manifest
hash. At `2026-09-11T18:28:57.744097+00:00`, a later explicit parent instruction
authorized resuming the same job while readiness continued. The conditional
watcher was cancelled and the original coordinator received SIGCONT. It reaped
the actual exit 0 and began the next prescribed cold group normally. No extra
broad tests, duplicate model workers, rewritten status, or discarded cases were
introduced. `execution/readiness-launch-resumed.json` records that readiness had
107 reported passes and was still running; it does not claim an early suite pass.
Collection wall time includes this explicit readiness wait; it is not an
isolated performance benchmark.

The subsequent parent instruction cancelled the redundant, unchanged-instrument
hermetic rerun. Only pytest PID 666578 received SIGINT; it exited 2 at
`2026-09-11T18:30:28.766041+00:00` after 1080.446 seconds. Its preserved log
reports 120 partial passes, all excluded from new acceptance pass totals. This
was an explicitly cancelled duplicate check, not a completed suite or a model
failure. Existing validated instrument regressions and the new image inventory,
pip check, import and exact collection evidence remain the readiness basis.
No additional broad retests were launched.

### Fresh measurement result and renewed numerical checkpoint

The complete repeatability command exited 0: **192 cases in 52 distinct serial
processes**, exactly 48 cases and 13 processes per profile. All same-seed raw,
preprocessing, decoded max/mean, trace and aggregate bias/slope, and per-index
statistics were zero. All 24 changed-seed controls changed both actual noise
and decoded output. Same-seed noise was exact within each profile; all **48
matched diagnostic noise pairs** were also exactly equal. Different-backend
model outputs were not compared.

Collection ran `2026-09-11T18:14:59.670373+00:00` through
`2026-09-11T18:48:26.849648+00:00`: 2007.179 seconds. The outer command took
2010.312 seconds (33m30s), including validation. The coordinator hold lasted
748.834 seconds (12m29s), of which 621.732 seconds (10m22s) occurred after the
warm model worker had finished. Summed actual worker lifetimes were 1051.527
seconds (17m32s), excluding container launch/closure and host validation. These
separate timings prevent the hold from being reported as model computation.

The required independent `check --stage repeatability` exited 0. Proposal
preparation and the additional capture-derived audit also exited 0. They
reconstructed worker/process/case chronology and statistics, verified source,
input and profile identities, revalidated canonical calibration and real serving
attestation, and confirmed the old proposal/agreement/failure hashes unchanged.
The audit is `execution/measurement-audit.json`; timing and cleanup are in
`execution/postflight.json`. Parent assigned Einstein a separate readiness review;
its verdict remains parent-managed and is not claimed by this executor.

New proposal: `corpus/phase7-stock-tests-20260911/tolerance-proposal.json`.
Exact SHA-256:

```text
a8794bcaff088ac1e9638872337ecb336ce6f3156ce5f6f470cfe63c3e18d8cf
```

The comparison definitions and golden bounds exactly equal the prior proposal:

| Boundary / metric | Fresh proposed bound |
| --- | --- |
| Floating preprocessing | atol `1e-6`, rtol `1e-6` |
| Tokens, masks, categorical data | exact equality |
| Full raw output | atol `1e-3`, rtol `1e-3` |
| Decoded maximum / mean absolute error | `0.1` / `0.05` |
| Absolute trace and aggregate bias | `0.02` |
| Absolute trace and aggregate OLS slope | `0.002` per action index |
| Native operational golden replay | atol `1e-5`, rtol `0` |

All four scopes use these bounds: diagnostic, native precision bridge, LeRobot
precision bridge, and operational. Decoded units are checkpoint percent for
five arm joints and gripper percent for the sixth. Zero observed variation
leaves the same prospectively defined floors. Four times observed variation is
an engineering margin, not a confidence interval. Independent operational noise
may fail these strict trace bounds; no adaptive loosening is authorized.
Training calibration, backbone provenance, forced-letterbox geometry and the
historical seed-not-honored caveats remain. Six-record repeatability is not
full-600 parity or a physical safety guarantee.

The canonical tolerance check returned the expected exit 2 / `not_run` because
this new session has **no tolerance agreement**. No stock consumer inference,
cross-backend output residual, full-600 comparison, golden, or robot activity
was performed. No container or GPU compute process remained after measurement.
The old approval does not authorize this new image or proposal.

The fresh blocking-human handoff and summary are:

- `.planning/phases/07-checkpoint-parity-gate/execution-attempts/07-07-stock-tests-20260911/CHECKPOINT.md`
- `.planning/phases/07-checkpoint-parity-gate/execution-attempts/07-07-stock-tests-20260911/SUMMARY.md`

An actual new affirmative decision must name this exact proposal before any
comparison. After that decision, the canonical writer records it and the unchanged
stock wrapper runs BOTH stages in this fresh session. Collection-only and
producer-feasibility artifacts do not replace the agreement-bound stock run.
Any failure blocks the full-600 and golden stages; hardware approval remains
separate. No numerical or physical approval was inferred from packaging approval.


## Plan 07 renewed approval and passing stock gate — 2026-09-11

The new response **"Approve"** explicitly approved proposal
`a8794bcaff088ac1e9638872337ecb336ce6f3156ce5f6f470cfe63c3e18d8cf`
in `corpus/phase7-stock-tests-20260911`. The canonical `record_decision`
checkpoint-transcription seam recorded it with `test_only=False`, operator
`Aaron (session user; explicit chat approval)`, the verbatim response and exact
proposal/image context. Its own current UTC timestamp is
`2026-09-11T19:04:08.666885+00:00`; no historical chat time was invented.
Agreement SHA-256:
`1b79b1b28d8ee1d9aff349635a8aed677af6617f674eb2a2316ff5e6c214ceea`.

Before recording, current corpus/checkpoint bytes matched the input lock;
all ten instrument hashes, the pristine external harness, current calibration,
new-image profiles and independent historical arm-free attestation validated.
The exact image remains
`sha256:6758186bd24cd0745b7442dafbb6680cbc2a986fe387eb2de5aaf0b75dce9c98`.
The existing latest tag and all prior failed-session artifacts are unchanged.
The canonical tolerance check then exited 0.

The unchanged stock wrapper ran both stages into a fresh private producer
directory. The producer exited 0 and emitted exactly `new_embodiment`, seed 42.
The consumer exited 0: **one collected and executed case passed**, zero
skips/xfails/xpasses or missing cases. Both independently observed full raw and
noise shapes were `(2,40,132)`; collated inputs and actual noise matched.
Both sides observed CPU FP32 parameters, buffers, inputs, backbone and compute,
SDPA (244 calls), four flow steps, eval mode, no autocast, both TF32 flags off,
and exact observer-disabled control equality. Maximum raw difference was
`4.768372e-7`, mean absolute difference `2.895536e-8`, within the unchanged
raw `atol=rtol=1e-3` bounds. No minimum-shape crop hid a mismatch.

The stock command ran `19:04:49.056987`–`19:06:11.040235` UTC, **81.984 seconds**
including launches and validation. Its immutable `upstream-result.json`
SHA-256 is
`6a3a72977a9d9b30415ae0c95096258303019badb99be0dba944cec465ce2faa`.
The separate strict archived stock check also exited 0 (1.805 seconds).

```sh
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/approve_parity_evidence.py check --stage tolerances --workspace corpus/phase7-stock-tests-20260911
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/replay_upstream_parity.py run --workspace corpus/phase7-stock-tests-20260911
UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never uv run python scripts/replay_upstream_parity.py check --workspace corpus/phase7-stock-tests-20260911
```

Exact argv, timestamps, exits, logs and pre-approval validation are archived
under `plan07-execution/` in this workspace; JUnit, coverage, runtime identities
and numeric captures are under `stock/` and `tensors/`. No new broad hermetic
suite was run: the unchanged instrument retains its prior validated regression
evidence, and the explicitly cancelled 120 partial passes remain excluded.

Task 1 is complete. Task 2 must still execute all full 600-case tiers and
operational bridges before generating the native candidate. Golden approval
and hardware release remain separate gates. Future threshold changes require
a fresh prospective proposal and explicit approval before a new experiment.
