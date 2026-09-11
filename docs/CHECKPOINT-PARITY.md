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
