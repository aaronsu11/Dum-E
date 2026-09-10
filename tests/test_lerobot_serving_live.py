"""Live LeRobot serving tests — the phase's ONLY non-keyless test module.

This module is the single sanctioned GPU-gated instrument for Phase 6. It requires
the ``lerobot-policy`` container ALREADY SERVING on ``127.0.0.1:8080`` with the
fine-tuned checkpoint bind-mounted read-only at ``/checkpoints/model``, and it
loads real weights on the far side of the wire. Everything else in this phase was
proven keyless and GPU-free and must stay that way.

It lives in its OWN module deliberately: the module-level ``pytestmark`` below
would otherwise leak a skip onto keyless guards sharing the file, and a skipped
guard is a silent pass on the facts this phase turns on.

**It must never touch the arm.** Nothing here imports ``SO10xArmController``,
constructs a robot, or opens a serial port. Every observation is synthetic.

==================== WHY EVERY ASSERTION IS ON THE DECODED CHUNK ====================
A test that asserts only "the RPC succeeded" is WORTHLESS against this server.
``GetActions`` wraps its body in a blanket ``except Exception`` and returns
``services_pb2.Empty()`` from a method declared to return ``Actions``
(``policy_server.py:259-266``), which protobuf serializes to ``b''``. A totally
broken server therefore answers every call successfully. So does a server that
decodes relative actions as flat-absolute. Every assertion below is on the
CONTENT of the decoded chunk.

Likewise the emitted chunk LENGTH is corroboration only, never horizon evidence:
three independent truncations force 16 regardless of whether the config is right
(``modeling_groot.py:308-324``, ``policy_server.py:328``, and the decode step's
own ``valid_horizon`` truncation). The config-level horizon assertion belongs to
plan 06-06.

==================== TWO TESTS RECREATE THE CONTAINER. DELIBERATELY. ====================
``test_live_five_seeded_repeats_return_identical_chunks`` needs
``DUME_POLICY_SEED`` PRESENT in the server's environment and
``test_live_no_seed_variable_means_no_seed_is_set`` needs it ABSENT, so no single
container configuration lets both pass. Each therefore recreates
``lerobot-policy-server`` into the configuration it needs, using the run command
extracted from `README.md`'s LRG-06 anchor (so there is one source of truth for
the publish spec and it is re-asserted loopback-only before being executed).

The alternative — asserting a precondition and failing when the operator started
the container the other way — would make the module unrunnable end to end, and
skipping would be a silent pass on the determinism claim. Recreating is also what
plan 06-03's own verify commands do; owning it here just means a plain
``pytest tests/test_lerobot_serving_live.py`` proves everything in one run.

The seeded test runs FIRST, so the module leaves the container in the unseeded
PRODUCTION configuration. Every recreate costs a fresh handshake (~6 s once the
shards are in page cache) because upstream reloads the weights on every handshake
(D-01) — no state is lost that was worth keeping.

Run it:
    bash scripts/build_lerobot_policy_image.sh
    docker run -d --gpus all -p 127.0.0.1:8080:8080 \
        -v "$(pwd)/checkpoints/GR00T-N1.7-3B-SO101:/checkpoints/model:ro" \
        --name lerobot-policy-server lerobot-policy
    DUME_RUN_LIVE_LEROBOT_TESTS=1 uv run pytest tests/test_lerobot_serving_live.py -q
"""

import datetime
import os
import socket
import subprocess
import time

import numpy as np
import pytest
import torch

from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction

from policy.lerobot.features import (
    CAMERA_KEYS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    ROBOT_STATE_KEYS,
    build_lerobot_features,
)
from policy.lerobot.session import LeRobotPolicySession

RUN_LIVE = os.getenv("DUME_RUN_LIVE_LEROBOT_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_LIVE,
    reason=(
        "live LeRobot serving tests are opt-in: build the image, start the container, "
        "then set DUME_RUN_LIVE_LEROBOT_TESTS=1"
    ),
)

SERVER_ADDRESS = os.getenv("DUME_LEROBOT_POLICY_ADDRESS", "127.0.0.1:8080")
CONTAINER_NAME = os.getenv("DUME_LEROBOT_CONTAINER_NAME", "lerobot-policy-server")
CHECKPOINT_MOUNT = "/checkpoints/model"
EXPECTED_HORIZON = 16
EXPECTED_ACTION_DIM = 6
EXPECTED_TAG = "new_embodiment"

#: The horizon deliberately injected into one handshake to prove the SAFE-01 guard
#: is WIRED. 40 is the well-lit wrong path and it is lit twice — ``GrootConfig``'s
#: own default ``chunk_size``/``n_action_steps`` are 40, and this checkpoint's own
#: ``config.json`` advertises ``action_horizon: 40`` — so it is the value a copied
#: config or a reasonable guess would actually supply.
WRONG_ACTIONS_PER_CHUNK = 40

#: The env var ``docker/lerobot-policy/server.py`` reads to decide whether to seed
#: the ambient torch RNG in-process before each inference call. Its value is
#: recorded in ``docs/LEROBOT-SERVING-VERDICTS.md`` alongside the result, because a
#: determinism verdict that does not name the seed it used is not reproducible.
SEED_ENV_VAR = "DUME_POLICY_SEED"

#: The seed used for the determinism probe. Arbitrary; what matters is that it is
#: FIXED and recorded.
DETERMINISM_SEED = 1234

#: How many repeats of one byte-identical observation the determinism probe sends.
DETERMINISM_REPEATS = 5

#: How long to wait for a freshly recreated container to answer ``Ready``.
CONTAINER_READY_BUDGET_S = 240.0

#: How long a refusal may take, end to end. Generous relative to the observed
#: handshake (~6 s once the shards are in page cache) but far below what THREE
#: retried handshakes would cost, which is the thing being ruled out: a retried
#: ``FAILED_PRECONDITION`` would burn the whole budget and then be reported as a
#: generic "unreachable".
REFUSAL_BUDGET_S = 60.0

#: A language instruction is REQUIRED, not decorative. The LeRobot-side language
#: key is "task" (``processor_groot.py:1547``); when it is absent
#: ``prepare_n1_7_language_batch`` silently substitutes "Perform the task."
#: (``lerobot/policies/groot/utils.py:239-241``), so an omitted instruction is a
#: silent degradation rather than an error.
TASK_INSTRUCTION = "I want one banana on the plate"


def _fixed_frames() -> dict:
    """Two DETERMINISTIC camera frames, so only the joint state varies between calls."""
    rng = np.random.RandomState(0)
    return {cam: rng.randint(0, 255, (FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8) for cam in CAMERA_KEYS}


def _synthetic_observation(frames: dict | None = None, joint_value: float = 0.0) -> dict:
    """One raw observation: six joints at ``joint_value``, fixed frames, a task string."""
    obs: dict = dict(frames if frames is not None else _fixed_frames())
    for key in ROBOT_STATE_KEYS:
        obs[key] = float(joint_value)
    obs["task"] = TASK_INSTRUCTION
    return obs


def _specs(actions_per_chunk: int = EXPECTED_HORIZON) -> RemotePolicyConfig:
    """The handshake payload. ``actions_per_chunk`` is the ONLY parameter.

    Kept parameterized in exactly one field because plan 06-03's wiring proof
    turns on injecting a WRONG horizon while every other field stays correct — if
    the negative case built its own spec, it could differ in a second field and
    the refusal would no longer be attributable to the horizon.

    The default is 16, NOT the ``GrootConfig`` default 40 and NOT the checkpoint's
    own ``config.json: action_horizon: 40`` — both are traps. This value is
    client-supplied and truncates the chunk server-side
    (``policy_server.py:328``).
    """
    return RemotePolicyConfig(
        policy_type="groot",
        pretrained_name_or_path=CHECKPOINT_MOUNT,
        lerobot_features=build_lerobot_features(),
        actions_per_chunk=actions_per_chunk,
        device="cuda",
        # Deliberately empty: rename_map feeds a processor step that runs AFTER
        # the key lookup it would have to fix. See policy/lerobot/features.py.
        rename_map={},
    )


def _utc_now_rfc3339() -> str:
    """Now, as an RFC3339 UTC stamp with microseconds, for ``docker logs --since``.

    Microseconds rather than whole seconds on purpose: a whole-second window can
    admit a line emitted earlier in the SAME second by a previous handshake, which
    is exactly the staleness the ``--since`` window exists to exclude. UTC because
    ``docker logs --since`` interprets a trailing ``Z`` as UTC and the container's
    own log timestamps are UTC, while this host is not.
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _container_logs_since(since: str) -> str:
    """``docker logs --since <since> <container>``, stdout and stderr combined."""
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", "logs", "--since", since, CONTAINER_NAME],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"`docker logs --since {since} {CONTAINER_NAME}` exited "
        f"{completed.returncode}: {completed.stderr!r}"
    )
    return completed.stdout + completed.stderr


@pytest.fixture(scope="module")
def session():
    """One handshake for the whole module.

    Module-scoped on purpose: upstream calls ``GrootPolicy.from_pretrained(...)``
    on EVERY handshake rather than once per container lifetime
    (``policy_server.py:151``), so each ``connect()`` is one full ~12.6 GB weight
    load costing minutes of wall clock on an RTX 3060.
    """
    specs = _specs()
    sess = LeRobotPolicySession(SERVER_ADDRESS)
    sess.connect(specs)
    try:
        yield sess
    finally:
        sess.close()


#: The anchor shift applied between the two robot states.
ANCHOR_SHIFT = 10.0

#: How many (0.0, 10.0) pairs to average the anchor shift over.
#:
#: More than one is REQUIRED, and the reason is measured rather than defensive.
#: The policy re-samples flow-matching noise on every call — this checkpoint's
#: ``num_inference_timesteps`` is 4 and there is no seed field anywhere on the
#: wire — so a single pair's delta is (anchor shift) + (noise difference), and the
#: noise difference was measured at up to 2.41 on the arm and -4.60 on the
#: gripper. Averaging over independent pairs leaves the systematic anchor shift
#: intact while shrinking the zero-mean noise term by ~1/sqrt(N). Phase 5 already
#: established there is no seed to reach for here: ``seed_verdict: not-honored``
#: (05-02-SUMMARY.md).
#:
#: 5 rather than 3 because the measured GRIPPER noise floor (mean |same-state
#: delta| = 3.79) is LARGER than the gripper's measured mean shift (2.81): the
#: only thing separating signal from noise on that dimension is the averaging, and
#: each extra pair is ~0.3s of wall clock against a container that is already
#: loaded.
REPEATS = 5


def _step0(chunk) -> np.ndarray:
    """Timestep 0 of a decoded chunk, as float64."""
    return chunk[0].get_action().numpy().astype(np.float64)


@pytest.fixture(scope="module")
def anchor_shift_measurement(session):
    """Measure the anchor-shift response AND the re-sampling noise floor.

    Every observation goes through ``infer``, which sets ``must_go=True``; without
    it the repeats are silently dropped by ``observations_similar`` (atol=1 in
    joint space) and the test would compare chunks against themselves.

    Returns:
        ``(shifted_deltas, control_deltas, first_chunk)`` where ``shifted_deltas``
        holds ``REPEATS`` deltas between a 0.0-state and a 10.0-state chunk, and
        ``control_deltas`` holds the SAME-state deltas. The control is the whole
        point: it measures the pure re-sampling noise with NO anchor shift in it,
        so the tolerance below is derived from this server's own measured
        behaviour instead of from a guess.
    """
    frames = _fixed_frames()

    shifted_deltas = []
    first_chunk = None
    for _ in range(REPEATS):
        chunk_a = session.infer(_synthetic_observation(frames, joint_value=0.0))
        chunk_b = session.infer(_synthetic_observation(frames, joint_value=ANCHOR_SHIFT))
        shifted_deltas.append(_step0(chunk_b) - _step0(chunk_a))
        if first_chunk is None:
            first_chunk = chunk_a

    control_deltas = []
    for _ in range(REPEATS):
        chunk_c = session.infer(_synthetic_observation(frames, joint_value=0.0))
        chunk_d = session.infer(_synthetic_observation(frames, joint_value=0.0))
        control_deltas.append(_step0(chunk_d) - _step0(chunk_c))

    return np.array(shifted_deltas), np.array(control_deltas), first_chunk


def test_live_get_actions_returns_sixteen_six_dim_actions(session):
    """One real GetActions returns a decoded 16-step, 6-dim chunk (LRG-01)."""
    actions = session.infer(_synthetic_observation())

    assert isinstance(actions, list), f"expected list[TimedAction], got {type(actions).__name__}"
    assert len(actions) == EXPECTED_HORIZON, (
        f"expected exactly {EXPECTED_HORIZON} TimedActions, got {len(actions)}"
    )
    for i, timed in enumerate(actions):
        assert isinstance(timed, TimedAction), (
            f"element {i} is {type(timed).__name__}, expected TimedAction"
        )
        action = timed.get_action()
        assert tuple(action.shape) == (EXPECTED_ACTION_DIM,), (
            f"action {i} has shape {tuple(action.shape)}, expected "
            f"({EXPECTED_ACTION_DIM},). A 132-wide action means output_features "
            f"was left to validate_features' max_action_dim default."
        )


# ==================== TOLERANCES: WIDENED, WITH THE MEASURED REASON ====================
# The plan specified `abs(delta - 10.0) < 0.1` for the arm and `< 1e-4` for the
# gripper. Those came from RESEARCH pushing ONE normalized action through two
# anchors, which isolates the decode arithmetic and is exact to floating point.
#
# Over the wire it cannot be exact, and this is the first run that could measure
# why: the server re-samples flow-matching noise on every call (4 inference
# timesteps, no seed field anywhere in RemotePolicyConfig, and Phase 5 already
# recorded `seed_verdict: not-honored` for this checkpoint family). So a delta is
# (anchor shift) + (noise difference), and the noise term is not small. FIRST
# MEASURED SINGLE PAIR, recorded verbatim:
#
#   anchor 0.0  step0: [ 0.0673,  0.2218, -2.3467, -0.6753, 0.0290, 14.3047]
#   anchor 10.0 step0: [10.8852, 10.1858,  6.6125, 11.7352, 9.6478,  9.7083]
#   delta            : [10.8179,  9.9640,  8.9591, 12.4105, 9.6188, -4.5964]
#
# The tolerances below are therefore derived from a SAME-STATE control measured in
# the same run (see the fixture) rather than chosen to make the test pass, and
# BOTH halves of the contrast are kept: an arm that does not track the anchor and
# a gripper that does are still both failures. What was deleted is only the
# pretence of floating-point exactness across a stochastic sampler.
#
# Measured same-state noise floor, mean |delta| per dimension, REPEATS=5:
#   [1.1153, 1.5060, 2.3234, 1.4766, 1.0816, 5.3045]
# alongside the mean shifted delta from the same run:
#   [9.9112, 11.1817, 9.7024, 9.8227, 9.0474, 1.2792]
#
# Note the last entry of each. The GRIPPER's re-sampling noise (5.30, with one
# individual same-state pair reaching -10.06) is several times LARGER than the
# gripper shift this test asserts is ~0 (1.28). Recorded rather than smoothed
# over, because it bounds what this test can prove: it establishes that the
# gripper does not track the ANCHOR — which IS the LRG-03 claim, and the arm/
# gripper contrast is unambiguous — but it cannot resolve a small systematic
# gripper offset out of that noise. Phase 7's parity work needs a seeded,
# in-process instrument for that, not this wire.
ARM_DELTA_TOL = 3.0

#: 5.0, not the 3.0 first tried: 3.0 sat BELOW the measured 3.79 gripper noise
#: floor, so it was a latent flake that happened to pass on its first run. 5.0 is
#: the midpoint between "unmoved" and the +10.0 an anchor-tracking gripper would
#: show, which is exactly where the closer-to-0-than-to-10 discriminator below
#: already draws its line — so the two agree instead of one silently overriding
#: the other, and a real relative-decoding regression still fails both.
GRIPPER_DELTA_TOL = 5.0

#: Multiple of the measured same-state noise floor that the mean shifted delta
#: must stay inside. Belt-and-braces on top of the absolute bands above: if this
#: server ever gets dramatically noisier, the bands catch it even when the
#: noise-relative check would have scaled with it.
NOISE_FLOOR_MULTIPLE = 3.0


def test_live_two_robot_states_yield_different_absolute_arm_targets(anchor_shift_measurement):
    """Relative-arm / absolute-gripper decoding, proven by an anchor shift (LRG-03).

    Shifting the robot state by +10.0 must shift the ABSOLUTE ARM targets by the
    same +10.0 (the arm group is decoded RELATIVE to the cached anchor state)
    while leaving the GRIPPER target unmoved (that group is decoded ABSOLUTE).
    A flat-absolute misread would move NEITHER; a fully-relative misread would
    move BOTH. Both halves are asserted, never either — the contrast between them
    IS the proof, and reading this checkpoint's output as flat-absolute inflates
    commanded motion 1.83x-3.01x.

    The discriminating assertion is deliberately "closer to X than to Y" rather
    than a raw band: it is the actual claim, and unlike a band it cannot be
    defeated by a noisier sampler. The bands are kept alongside it so a gross
    regression still fails on magnitude.

    Also asserts the per-timestep statistics are applied PER STEP rather than
    collapsed: the arm's ``relative_action.single_arm`` q01/q99 are ``(16, 5)``,
    one row per timestep, so joint 0 must trace a varied path across the chunk.
    """
    shifted, control, first_chunk = anchor_shift_measurement

    mean_shift = shifted.mean(axis=0)
    arm_shift, gripper_shift = mean_shift[:5], float(mean_shift[5])
    # The noise floor: how much a delta moves with NO anchor shift applied at all.
    noise_floor = np.abs(control).mean(axis=0)
    arm_noise, gripper_noise = noise_floor[:5], float(noise_floor[5])

    diagnostic = (
        f"\n  shifted deltas ({REPEATS} pairs, state 0.0 -> {ANCHOR_SHIFT}):"
        + "".join(f"\n    {np.round(row, 4).tolist()}" for row in shifted)
        + f"\n  mean shifted delta : {np.round(mean_shift, 4).tolist()}"
        + f"\n  same-state control ({REPEATS} pairs, pure re-sampling noise):"
        + "".join(f"\n    {np.round(row, 4).tolist()}" for row in control)
        + f"\n  mean |noise|       : {np.round(noise_floor, 4).tolist()}"
        + f"\n  arm mean shift (want ~+{ANCHOR_SHIFT}) : {np.round(arm_shift, 4).tolist()}"
        + f"\n  gripper mean shift (want ~0.0)     : {round(gripper_shift, 4)}"
    )

    # Printed unconditionally, not only on failure: these numbers are the LRG-03
    # evidence and the input to Phase 7's parity comparison, so they belong in the
    # run record rather than only in a traceback. `pytest -s` surfaces them.
    print(diagnostic)

    # ---- half 1: the ARM tracks the anchor (relative decoding) ----
    assert np.all(np.abs(arm_shift - ANCHOR_SHIFT) < np.abs(arm_shift - 0.0)), (
        "at least one arm joint's target is closer to UNMOVED than to the anchor shift, so "
        "the arm group is NOT being decoded as relative-to-state — a flat-absolute misread "
        f"inflates commanded motion 1.83x-3.01x.{diagnostic}"
    )
    assert np.all(np.abs(arm_shift - ANCHOR_SHIFT) < ARM_DELTA_TOL), (
        f"an arm joint's mean shift is further than {ARM_DELTA_TOL} from the +{ANCHOR_SHIFT} "
        f"anchor shift. It still tracks the anchor, so this is a magnitude regression rather "
        f"than a decoding-mode error — check the per-timestep statistics.{diagnostic}"
    )
    assert np.all(np.abs(arm_shift - ANCHOR_SHIFT) < NOISE_FLOOR_MULTIPLE * np.maximum(arm_noise, 1e-6)), (
        f"an arm joint's deviation from the anchor shift exceeds {NOISE_FLOOR_MULTIPLE}x this "
        f"server's own measured same-state noise floor, so it is not attributable to "
        f"flow-matching re-sampling.{diagnostic}"
    )

    # ---- half 2: the GRIPPER does NOT track the anchor (absolute decoding) ----
    assert abs(gripper_shift - 0.0) < abs(gripper_shift - ANCHOR_SHIFT), (
        "the gripper target is closer to anchor-shifted than to unmoved, so the gripper "
        "group is being decoded as RELATIVE when the checkpoint's action_configs declare it "
        f"ABSOLUTE.{diagnostic}"
    )
    assert abs(gripper_shift) < GRIPPER_DELTA_TOL, (
        f"the gripper's mean shift is further than {GRIPPER_DELTA_TOL} from 0.0. It is still "
        f"nearer unmoved than anchor-shifted, so this is a magnitude regression rather than a "
        f"decoding-mode error.{diagnostic}"
    )
    assert abs(gripper_shift) < NOISE_FLOOR_MULTIPLE * max(gripper_noise, 1e-6), (
        f"the gripper's mean shift exceeds {NOISE_FLOOR_MULTIPLE}x its own measured same-state "
        f"noise floor, so it is not attributable to re-sampling.{diagnostic}"
    )

    # ---- the per-timestep statistics are not collapsed ----
    joint0 = [round(float(timed.get_action()[0]), 4) for timed in first_chunk]
    distinct = len(set(joint0))
    assert distinct >= 8, (
        f"joint 0 takes only {distinct} distinct values across the {len(joint0)} timesteps "
        f"of one chunk, expected at least 8. Per-timestep stats look collapsed.\n"
        f"  joint0 across chunk: {joint0}"
    )


def test_live_resolved_base_model_path_is_the_mounted_checkpoint():
    """LRG-02/LRG-04 asserted against the REAL load path, from the container log.

    ``lerobot.async_inference.policy_server`` accepts no ``--policy.*`` flag at
    all, so there is no CLI flag to verify. What matters is the value
    ``GrootPolicy.from_pretrained`` actually resolved onto
    ``policy.config.base_model_path`` (``modeling_groot.py:249, 263-264``) and the
    ``embodiment_tag`` on the object that serves inference — both emitted on the
    effective-values INFO line by ``DumEGrootPolicyServer.SendPolicyInstructions``.
    """
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", "logs", CONTAINER_NAME],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"`docker logs {CONTAINER_NAME}` exited {completed.returncode}: {completed.stderr!r}"
    )
    log = completed.stdout + completed.stderr

    assert "DumE effective serving values" in log, (
        "the effective-values INFO line is absent from the container log, so the "
        "DumEGrootPolicyServer override did not run — a vanilla PolicyServer cannot "
        "serve this checkpoint."
    )
    assert CHECKPOINT_MOUNT in log, (
        f"the container log does not mention {CHECKPOINT_MOUNT!r}. The resolved "
        "base_model_path is not the bind-mounted checkpoint, which means "
        "configuration_groot.py:382-383 may have fallen back to the hub base weights."
    )
    assert EXPECTED_TAG in log, (
        f"the container log does not mention the embodiment tag {EXPECTED_TAG!r}. This "
        "checkpoint carries 9 tags and only 'new_embodiment' has 16 delta_indices; the "
        "other 8 have 40."
    )


def test_live_unreachable_server_raises_instead_of_hanging():
    """An unreachable server raises a descriptive RuntimeError, bounded (LRG-01, D-10).

    Matches ``groot-native``'s contract: the message names the address and
    contains the literal substring ``unreachable``, and it arrives inside the
    bounded budget instead of parking forever. ``wait_for_ready`` is never passed
    on any stub call — its default (False) is what makes this true.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    closed_port = probe.getsockname()[1]
    probe.close()

    address = f"127.0.0.1:{closed_port}"
    sess = LeRobotPolicySession(address)
    started = time.time()
    try:
        with pytest.raises(RuntimeError) as excinfo:
            sess.infer(_synthetic_observation())
    finally:
        sess.close()
    elapsed = time.time() - started

    message = str(excinfo.value)
    assert "unreachable" in message.lower(), (
        f"RuntimeError message lacks the operator-facing 'unreachable' substring: {message!r}"
    )
    assert address in message, f"RuntimeError message does not name the address {address}: {message!r}"
    assert elapsed < 60.0, (
        f"took {elapsed:.1f}s to raise against a closed port; the bounded budget is "
        f"~45s and anything longer suggests wait-for-ready behaviour."
    )


# ==================== SAFE-01 IS WIRED: THE FAIL-FIRST PROOF ON THE REAL PATH ====================
# ``tests/test_container_contract.py:236-249`` states the non-vacuity standard this
# section answers to: "an entry that has never been red is an entry that has never
# been tested". ``tests/test_groot_guard.py``'s twelve keyless violation tests prove
# the guard FUNCTION is correct; they cannot prove it RUNS on the object that serves
# inference, which is D-05's stated failure mode verbatim — "keyless-only coverage
# lets a fixture that misrepresents the real config shape pass while the live guard
# never fires". The two tests below are the other half: one proves the guard ran and
# what it saw, the other proves a deliberately mis-configured handshake is REFUSED
# on the real load path.
#
# 40 is the injected wrong value, and it is chosen because it is the WELL-LIT wrong
# path rather than an arbitrary number: ``GrootConfig``'s own default
# ``chunk_size``/``n_action_steps`` are 40, and this checkpoint's own ``config.json``
# advertises ``action_horizon: 40``. A horizon that plausible is exactly the one a
# future operator or a copied config would supply, and D-11 made the horizon
# configurable — so this is the assertion that keeps a configurable horizon from
# being an OBEYABLE wrong horizon.
#
# Both tests drive the ALREADY-RUNNING container over the wire, and neither loads
# weights in THIS process — proven at AST level rather than by substring, because
# this module legitimately discusses upstream's per-handshake ``from_pretrained``
# in prose: an AST walk over the module reports ZERO calls to it. Plan 06-03's
# acceptance criterion phrases that as a whole-file grep; the grep counts prose,
# so it is satisfied at its intent level the same discriminating way 06-06 handled
# the identical collision.


def test_live_guard_pass_is_logged_on_the_real_load_path():
    """The SAFE-01 guard RAN on the object that serves inference (D-05 wiring proof).

    This is the half keyless coverage cannot reach. A guard that is correct in
    isolation and never invoked is indistinguishable, from the outside, from one
    that is invoked and passes — which is why the pass is an observable log line
    (threat T-06-14, repudiation) rather than silence.

    It also discharges what plan 06-02 could not: ``snapshot_from_loaded`` locates
    the pack step by ``state_dropout_prob``, the encode step by
    ``letter_box_transform`` and the decode step by ``env_action_dim``, and 06-02's
    coverage D7 is flagged ``human_judgment: true`` precisely because those markers
    were only ever exercised against minimal stand-ins. Here they run against the
    REAL constructed ``GrootPolicy``/processor objects: an absent marker raises
    ``ValueError`` naming what it looked for, so a PASS line naming
    ``GrootN17ActionDecodeStep`` is positive evidence that all three located.

    The ``--since`` window is taken immediately BEFORE the connect. Without it the
    assertion would be satisfiable by a line from any earlier handshake in this
    container's history — i.e. it would go vacuous after its first successful run,
    which is the failure mode a wiring proof least tolerates.
    """
    since = _utc_now_rfc3339()
    sess = LeRobotPolicySession(SERVER_ADDRESS)
    try:
        sess.connect(_specs())
    finally:
        sess.close()

    log = _container_logs_since(since)

    assert "SAFE-01 guard: PASS" in log, (
        "the literal 'SAFE-01 guard: PASS' is absent from the container log for the "
        f"window opened at {since}, so the guard did NOT run on this handshake. A "
        "guard that passes silently is indistinguishable from a guard that never "
        f"ran — that is D-05's stated failure mode.\n  log window:\n{log}"
    )
    for expected in (CHECKPOINT_MOUNT, EXPECTED_TAG, str(EXPECTED_HORIZON)):
        assert expected in log, (
            f"the guard's PASS line does not report {expected!r}, so the values the "
            f"guard actually saw are not observable to an operator (LRG-02)."
            f"\n  log window:\n{log}"
        )
    assert "GrootN17ActionDecodeStep" in log, (
        "the guard's PASS line does not name GrootN17ActionDecodeStep as the decode "
        "step. Either the legacy GrootActionUnpackUnnormalizeStep was installed (which "
        "SAFE-01/3 should have refused) or snapshot_from_loaded's env_action_dim marker "
        f"no longer locates the decode step.\n  log window:\n{log}"
    )


def test_live_wrong_actions_per_chunk_is_refused_with_safe01_2():
    """A deliberately wrong handshake is REFUSED on the real load path (SAFE-01, LRG-04).

    The fail-first half. Every field of the handshake is correct except
    ``actions_per_chunk``, which is 40 — so a refusal is attributable to the
    horizon and nothing else.

    Three things are asserted, because any one alone would be weak:

    1. ``connect`` RAISES, and the raised message carries the guard's own
       ``SAFE-01/2`` identifier together with BOTH numbers (40 asked for, 16 the
       checkpoint decodes). A refusal that does not name the observed value is not
       actionable.
    2. It arrives PROMPTLY. ``FAILED_PRECONDITION`` is excluded from
       ``policy/lerobot/session.py``'s ``RETRYABLE_CODES``, so the client must raise
       on the first attempt rather than burning three handshake retries — each of
       which would start ANOTHER concurrent multi-GB weight load — and then burying
       the server's own diagnosis under a generic "unreachable".
    3. The refusal does not WEDGE the server (threat T-06-18): a correct handshake
       afterwards still yields 16 decoded actions.

    Note what is deliberately NOT asserted: that the refusal happened before the
    weights were read. It did not, and cannot — the post-load site exists precisely
    because the policy is constructed inside this request handler, so a rejected
    handshake still costs one full load. The site that refuses BEFORE any shard is
    read is the entrypoint preflight, which is a different instrument.
    """
    since = _utc_now_rfc3339()
    sess = LeRobotPolicySession(SERVER_ADDRESS)
    started = time.time()
    try:
        with pytest.raises(RuntimeError) as excinfo:
            sess.connect(_specs(actions_per_chunk=WRONG_ACTIONS_PER_CHUNK))
    finally:
        sess.close()
    elapsed = time.time() - started

    message = str(excinfo.value)
    assert "SAFE-01/2" in message, (
        "the client-side error does not carry the guard's own 'SAFE-01/2' identifier, so "
        "the operator cannot tell WHICH assertion refused: "
        f"{message!r}"
    )
    for number in (str(WRONG_ACTIONS_PER_CHUNK), str(EXPECTED_HORIZON)):
        assert number in message, (
            f"the client-side error does not name {number}; a horizon refusal must report "
            f"both the value asked for and the value the checkpoint decodes: {message!r}"
        )
    # The refusal must have arrived through the session's NON-RETRYABLE branch, not
    # through its retry-exhaustion branch. Asserted by each branch's own
    # distinguishing text rather than by the word "unreachable": the non-retryable
    # message legitimately QUOTES that word while explaining why it is not used
    # ("three retries would only bury it under a generic 'unreachable'"), so a
    # substring check on it is red for the wrong reason against any correct
    # implementation. Same self-referential-grep class 06-02 hit with
    # `normalization_mapping` and 06-05 hit with `allclose`; solved the same way, by
    # asserting the discriminating thing instead of weakening either message.
    assert "rejected the call with gRPC status" in message, (
        "the refusal did not come through the session's non-retryable branch, so the "
        f"server's own diagnosis was not surfaced verbatim: {message!r}"
    )
    assert "FAILED_PRECONDITION" in message, (
        "the refusal does not name FAILED_PRECONDITION. An 'unavailable' or 'unknown' "
        "status would be retryable or indistinguishable from a transport fault, and "
        f"RETRYABLE_CODES would then bury this message: {message!r}"
    )
    assert "attempts (last gRPC status" not in message, (
        "the message carries the retry-EXHAUSTION wording, which means a non-retryable "
        "FAILED_PRECONDITION was retried until the budget ran out and the server's own "
        f"diagnosis was buried under a generic unreachable report: {message!r}"
    )
    assert elapsed < REFUSAL_BUDGET_S, (
        f"the refusal took {elapsed:.1f}s, over the {REFUSAL_BUDGET_S}s budget. "
        f"FAILED_PRECONDITION must not be retried — three handshake retries would each "
        f"start another concurrent multi-GB weight load."
    )

    # The same refusal must be diagnosable from the SERVER side too, at ERROR level:
    # an operator reading `docker logs` should not have to reconstruct it from a
    # client traceback they may not have.
    log = _container_logs_since(since)
    assert "SAFE-01 guard: REFUSED" in log, (
        f"the container log for the window opened at {since} carries no 'SAFE-01 guard: "
        f"REFUSED' line, so the refusal is only visible client-side.\n  log window:\n{log}"
    )
    assert "SAFE-01/2" in log, (
        f"the container log does not carry the SAFE-01/2 text.\n  log window:\n{log}"
    )
    assert any("ERROR" in line and "SAFE-01" in line for line in log.splitlines()), (
        f"the refusal was not logged at ERROR level, so it is easy to miss in a healthy-"
        f"looking log.\n  log window:\n{log}"
    )

    # T-06-18: the refusal must not wedge the server for subsequent clients.
    recovered = LeRobotPolicySession(SERVER_ADDRESS)
    try:
        recovered.connect(_specs())
        actions = recovered.infer(_synthetic_observation())
    finally:
        recovered.close()
    assert len(actions) == EXPECTED_HORIZON, (
        f"after a refused handshake a CORRECT handshake yielded {len(actions)} actions, "
        f"expected {EXPECTED_HORIZON}. The refusal path wedged the server."
    )
    for i, timed in enumerate(actions):
        assert tuple(timed.get_action().shape) == (EXPECTED_ACTION_DIM,), (
            f"post-refusal action {i} has shape {tuple(timed.get_action().shape)}, "
            f"expected ({EXPECTED_ACTION_DIM},)"
        )


# ==================== CRITERION 5: DETERMINISM, UNDER AN IN-PROCESS SEED ====================
# The ROADMAP asks that "5 seeded repeats of one observation produce identical
# chunks". Two independent facts make that unobtainable AS WRITTEN, and both are
# re-verified rather than assumed:
#
# 1. ``RemotePolicyConfig`` has NO seed field at all
#    (``async_inference/helpers.py:266-273`` — the six fields are ``policy_type``,
#    ``pretrained_name_or_path``, ``lerobot_features``, ``actions_per_chunk``,
#    ``device``, ``rename_map``). There is no seed to send over this wire.
# 2. The checkpoint decodes with flow matching over ``num_inference_timesteps: 4``,
#    whose initial noise is drawn from the ambient torch RNG INSIDE the server
#    process.
#
# And Phase 5 already established that the sibling GR00T-native server does not
# honour a seed (``seed_verdict: not-honored``, 05-02-SUMMARY.md: same-seed
# max|diff| 5.51 vs different-seed 4.63, with ``seed``/``random_seed``/``rng_seed``
# all tried).
#
# So the mechanism is an env-gated ``torch.manual_seed`` set IN-PROCESS by Dum-E's
# own ``PolicyServer`` subclass, and the claim these two tests support is
# "**deterministic under an in-process seed set by Dum-E's own PolicyServer
# subclass**" — NEVER "the server honours a seed", and never evidence that a seed
# travelled over the wire. This project built a classifier that distinguishes those
# two claims; collapsing them would fabricate a capability.


def _documented_run_command() -> str:
    """The `README.md` LRG-06 run command, re-asserted loopback-only before use.

    Extracted rather than restated so there is ONE source of truth for the publish
    spec: a second hardcoded copy here could widen it to ``-p 8080:8080`` without
    plan 06-05's guard ever noticing, which is threat T-06-26 exactly. Running
    ``assert_loopback_only`` on the extracted text before executing it means this
    module cannot start a container that is reachable off-host even if `README.md`
    regresses.
    """
    from tests.test_loopback_publish_spec import (
        README,
        assert_loopback_only,
        extract_run_block,
    )

    command = extract_run_block(README.read_text())
    assert_loopback_only(command)
    return command


def _recreate_container(extra_env: dict[str, str] | None = None) -> None:
    """Recreate ``lerobot-policy-server`` from the documented command, plus ``-e`` flags.

    Waits for ``Ready`` and asserts the fresh container's preflight printed no
    ``FAIL:`` and reached its listening socket, so a test never runs against a
    container that refused to start (which would otherwise surface as a confusing
    transport error rather than as the refusal it is).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    command = _documented_run_command()
    if extra_env:
        flags = " ".join(f"-e {name}={value}" for name, value in extra_env.items())
        # One replacement, at the head of the command, so the publish spec, the
        # mount and the image name are untouched.
        command = command.replace("docker run -d", f"docker run -d {flags}", 1)

    subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    started = subprocess.run(  # noqa: S603 - the documented command, guarded above
        ["bash", "-c", command],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert started.returncode == 0, (
        f"the documented run command exited {started.returncode}: {started.stderr!r}"
    )

    probe = LeRobotPolicySession(SERVER_ADDRESS)
    deadline = time.time() + CONTAINER_READY_BUDGET_S
    try:
        while time.time() < deadline:
            if probe.ready():
                break
            time.sleep(2.0)
        else:
            log = _container_logs_since("1h")
            pytest.fail(
                f"{CONTAINER_NAME} did not answer Ready within "
                f"{CONTAINER_READY_BUDGET_S}s of being recreated.\n  log:\n{log}"
            )
    finally:
        probe.close()

    log = _container_logs_since("1h")
    assert "FAIL:" not in log, (
        f"the recreated container's preflight printed a FAIL line, so it refused to "
        f"start:\n{log}"
    )
    assert "DumEGrootPolicyServer started on" in log, (
        f"the recreated container never reached its listening socket:\n{log}"
    )


@pytest.fixture
def seeded_container():
    """Recreate the container WITH ``DUME_POLICY_SEED`` set. Returns the seed."""
    _recreate_container({SEED_ENV_VAR: str(DETERMINISM_SEED)})
    return DETERMINISM_SEED


@pytest.fixture
def unseeded_container():
    """Recreate the container with ``DUME_POLICY_SEED`` ABSENT from its environment."""
    _recreate_container(None)


def test_live_five_seeded_repeats_return_identical_chunks(seeded_container):
    """Five repeats of ONE observation return byte-identical chunks (criterion 5).

    The claim, in the exact form it is reported in
    ``docs/LEROBOT-SERVING-VERDICTS.md``: **deterministic under an in-process seed
    set by Dum-E's own ``PolicyServer`` subclass.**

    This is NOT the claim that the server honours a seed, and it must never be
    written up as one. ``RemotePolicyConfig`` has no seed field
    (``async_inference/helpers.py:266-273``), so nothing about a seed travels over
    this wire; the seed is applied in-process, immediately before
    ``_get_action_chunk``, by ``docker/lerobot-policy/server.py``. Phase 5 recorded
    ``seed_verdict: not-honored`` for the sibling GR00T-native server, and this
    result neither overturns nor extends that verdict — they are different claims.

    Comparison is EXACT: ``torch.equal`` per timestep, no tolerance of any kind. A
    tolerance here would convert "identical" into "close", which is precisely the
    hedge that lets a real nondeterminism regression pass a later parity gate.

    If fewer than five chunks come back, that is a TRANSPORT finding rather than a
    determinism finding: the server's ``observations_similar`` filter (atol=1 in
    joint space, ``helpers.py:281-283``) drops a replayed identical observation, and
    ``must_go=True`` on every observation is what defeats it. Check the container
    log for ``has been filtered out`` before concluding anything about seeds.
    """
    printenv = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", "exec", CONTAINER_NAME, "printenv", SEED_ENV_VAR],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert printenv.returncode == 0 and printenv.stdout.strip() == str(seeded_container), (
        f"the positive control failed: {SEED_ENV_VAR} is not {seeded_container} inside "
        f"{CONTAINER_NAME} (exit {printenv.returncode}, value {printenv.stdout.strip()!r}). "
        f"Without it the server sets NO seed, and an identical-chunks result would be "
        f"reported as seeded when it was not."
    )

    sess = LeRobotPolicySession(SERVER_ADDRESS)
    try:
        sess.connect(_specs())
        # ONE observation object, reused — not rebuilt per call — so the five
        # requests are byte-identical by construction rather than by coincidence.
        observation = _synthetic_observation()
        chunks = [sess.infer(observation) for _ in range(DETERMINISM_REPEATS)]
    finally:
        sess.close()

    assert len(chunks) == DETERMINISM_REPEATS, (
        f"expected {DETERMINISM_REPEATS} decoded chunks, got {len(chunks)}"
    )
    for i, chunk in enumerate(chunks):
        assert len(chunk) == EXPECTED_HORIZON, (
            f"repeat {i} returned {len(chunk)} actions, expected {EXPECTED_HORIZON}. "
            f"Fewer means the server's similarity filter dropped a repeat — a transport "
            f"finding, not a determinism finding. Check `docker logs` for "
            f"'has been filtered out'."
        )

    reference = chunks[0]
    mismatches = []
    worst = 0.0
    for i, chunk in enumerate(chunks[1:], start=1):
        for step, (want, got) in enumerate(zip(reference, chunk, strict=True)):
            want_action, got_action = want.get_action(), got.get_action()
            if not torch.equal(want_action, got_action):
                diff = float((want_action.double() - got_action.double()).abs().max())
                worst = max(worst, diff)
                mismatches.append((i, step, diff))

    assert not mismatches, (
        f"{len(mismatches)} of {DETERMINISM_REPEATS - 1} x {EXPECTED_HORIZON} timesteps "
        f"differ from repeat 0 under a fixed in-process seed of {seeded_container}. "
        f"Observed maximum absolute difference: {worst}. "
        f"First five mismatches (repeat, timestep, max|diff|): {mismatches[:5]}. "
        f"Do NOT widen a tolerance and do NOT weaken the claim to 'approximately "
        f"identical' — record the number, name a non-deterministic CUDA kernel in the "
        f"backbone or the flow-matching sampler as the leading candidates, and carry it "
        f"forward as an open question for Phase 7's parity harness."
    )


def test_live_no_seed_variable_means_no_seed_is_set(unseeded_container):
    """The determinism instrument is genuinely OPT-IN (the non-vacuity half).

    Proves two things about the production path: the seed variable is genuinely
    absent from the container's environment, and inference still returns a
    well-formed chunk with it absent — so the instrument added for criterion 5 does
    not perturb the path Phase 7 will measure.

    It deliberately does NOT assert that unseeded repeats DIFFER. That would make
    the test depend on nondeterminism actually manifesting on this GPU, which this
    phase never established, and it would be flaky in the one direction a safety
    suite must never be flaky: green when the mechanism is broken.
    """
    printenv = subprocess.run(  # noqa: S603 - fixed argv, no shell
        ["docker", "exec", CONTAINER_NAME, "printenv", SEED_ENV_VAR],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert printenv.returncode != 0, (
        f"{SEED_ENV_VAR} is PRESENT in {CONTAINER_NAME}'s environment "
        f"(value {printenv.stdout.strip()!r}), so the determinism instrument is not "
        f"genuinely opt-in and the production path is not the one being measured here."
    )

    sess = LeRobotPolicySession(SERVER_ADDRESS)
    try:
        sess.connect(_specs())
        actions = sess.infer(_synthetic_observation())
    finally:
        sess.close()

    assert len(actions) == EXPECTED_HORIZON, (
        f"with no seed set, inference returned {len(actions)} actions, expected "
        f"{EXPECTED_HORIZON}"
    )
    for i, timed in enumerate(actions):
        assert tuple(timed.get_action().shape) == (EXPECTED_ACTION_DIM,), (
            f"unseeded action {i} has shape {tuple(timed.get_action().shape)}, expected "
            f"({EXPECTED_ACTION_DIM},)"
        )
