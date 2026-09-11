# Checkpoint parity: local numerical instrument

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


## Measured proposal awaiting Task 3 agreement

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

**Task 3 is a blocking-human decision.** No tolerance agreement, cross-backend
residual, stock consumer, full-600 comparison or hardware run has occurred.
The next action is explicit named human review of the exact proposal digest.
The executor may transcribe an actual affirmative response and its provenance;
absence of a response, parent readiness or a successful measurement is not assent.
