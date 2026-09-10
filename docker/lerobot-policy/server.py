"""Dum-E's ``PolicyServer`` subclass — the four fixes vanilla 0.6.1 needs.

This module runs INSIDE the ``lerobot-policy`` container. It exists because the
vanilla ``lerobot.async_inference.policy_server.PolicyServer`` **cannot serve the
fine-tuned ``GR00T-N1.7-3B-SO101`` checkpoint at all, and fails silently**. Four
independent blockers stack up, each confirmed by execution against the installed
0.6.1 wheel:

1. ``policy_server.py:368-379`` postprocesses the action chunk ONE TIMESTEP AT A
   TIME — it loops over the chunk index and hands the postprocessor a single
   ``action_tensor[:, i, :]`` slice per iteration — and
   ``processor_groot.py:2345-2350``'s ``GrootN17ActionDecodeStep`` raises
   ``NotImplementedError`` for exactly that call shape when native relative
   actions are in play — which this checkpoint's ``processor_config.json`` enables
   (``use_relative_action: true``). Fixed by ``_predict_action_chunk`` below.
2. ``modeling_groot.py:255-261`` — the raw-GR00T load path injects a SINGLE
   placeholder visual feature ``observation.images.camera`` at ``(3, 224, 224)``
   because the checkpoint's own ``input_features`` is empty. Two real cameras
   then raise ``KeyError: 'observation.images.front'`` in
   ``prepare_raw_observation``. Fixed by ``fixup_policy_features`` below.
3. Same placeholder, silent variant: with a single camera actually NAMED
   ``camera``, ``resize_robot_observation_image`` pre-resizes the 480x640 frame to
   the placeholder's 224x224 BEFORE the checkpoint's own geometry runs, destroying
   the frame's aspect ratio while every log line looks healthy. Same fix — and note
   that this corruption is now shape-INVISIBLE on the serving path, because the
   forced letterbox pad below squares every input: a 224x224 frame and a 480x640
   frame both emerge as ``(256, 256, 3)``, differing only in pixels. It is
   PREVENTED (``fixup_policy_features`` runs before ``from_pretrained``, so the
   placeholder branch never executes) rather than detected by a shape check.

4. **Blocker 4 — fp32 weight storage, which does not fit the GPU.** This one was
   found by executing the load, not by reading, and it is why this module takes
   the plan's STATED FALLBACK instead of its ``super()``-calling override. It is
   recorded at length in the next block because it is the number Phase 7's
   parity gate is measured against.

Blocker 1 is what makes all four lethal rather than merely wrong: ``GetActions``
wraps its whole body in a blanket ``except Exception`` and returns
``services_pb2.Empty()`` from a method declared to return ``Actions``
(``policy_server.py:259-266``). protobuf serializes that mismatch to ``b''``, so
the client sees a SUCCESSFUL RPC carrying zero bytes. A test that asserts only
"the RPC succeeded" would pass against a completely broken server.

==================== SANCTIONED MECHANISM: subclass + super() ONLY ====================
The ONLY sanctioned mechanism here is composition: subclass ``PolicyServer`` and
call ``super()``. Three alternatives are forbidden, in the terms the phase
context sets out:

* NEVER patch installed files inside ``.venv`` or the installed package tree.
* NEVER fork or vendor a copy of upstream source into this repo.
* NEVER monkeypatch upstream internals.

``lerobot`` stays a pinned pip dependency. The reason is failure MODE, not
purity: a subclass that calls ``super()`` fails LOUDLY on a version bump — a
renamed method stops being overridden and blocker 1's ``NotImplementedError``
reappears immediately and visibly — whereas a monkeypatch fails silently, which
is the exact failure class this milestone is structured to avoid.

Every mutation this module performs is an attribute assignment on
``self.policy.config`` / ``self.preprocessor`` / ``self.postprocessor``: instance
state of objects this subclass owns. It never rebinds an attribute on an imported
upstream class.

==================== SERVING PRECISION: bf16, BY HUMAN DECISION ====================
**LeRobot's default for this checkpoint is fp32 weight STORAGE, and it does not
fit an RTX 3060.** Measured, not inferred:

* the checkpoint's three shards hold **3.144B parameters, every tensor ``F32``**
  (12.58 GB on disk);
* the first live attempt, on LeRobot's defaults, measured **11.47 GiB allocated /
  11750 MiB resident** and OOMed on a 12288 MiB card;
* the v1.0 Isaac-GR00T baseline this milestone must not regress against measured
  **6322 MiB**, because ``gr00t_policy.py:99-102`` casts the model to bf16.

Two independent knobs make LeRobot materialize fp32, and BOTH are handled below:

1. ``GrootConfig.model_params_fp32`` defaults to **True**
   (``configuration_groot.py:347``), and ``modeling_groot.py:116`` acts on it by
   casting every floating-point parameter UP to fp32 after the load. Injected as
   ``False``.
2. ``transformers`` 5.5.4 defaults its ``dtype`` argument to ``"auto"``
   (``modeling_utils.py:270-271``), and ``"auto"`` resolves the ``dtype`` entry in
   the model's ``config.json`` — which this checkpoint sets to **``'float32'``**.
   So the whole model, backbone and action head alike, materializes fp32 before
   ``model_params_fp32`` is ever consulted. Injected as ``torch.bfloat16`` at the
   ``GR00TN17.from_pretrained`` call, i.e. BEFORE weights materialize.

``load_bf16=True`` (``groot_n1_7.py:288-289`` → ``extra_kwargs["torch_dtype"]``)
is injected alongside it. It is the knob the checkpoint's own ``config.json``
names — and declares ``False`` while also declaring ``model_dtype: 'bfloat16'`` —
but on its own it is NOT sufficient here: it reaches only the nested
Cosmos-Reason2-2B backbone load, not the action head, and the outer ``"auto"``
resolution above governs the shard load that follows. It is injected because it
keeps the backbone from transiting fp32 in host RAM, and because it is the
declared knob; ``dtype`` is what actually makes the whole model bf16.

Two knobs that look relevant and are NOT:

* ``GrootConfig.use_bf16`` (default True) is **compute autocast only**, consumed
  at ``modeling_groot.py:458,498``. It is not storage precision and was already
  on during the fp32 measurement.
* ``model_dtype: 'bfloat16'`` in the checkpoint's ``config.json`` is **dead** for
  GR00T under LeRobot: it appears exactly once in ``lerobot/policies/groot/``, at
  ``groot_n1_7.py:77``, inside ``GR00T_N1_7_DEFAULTS``, and is never read.

**This was escalated to the human and approved as a deliberate decision, not
applied as an incidental fix**, because it is a precision change on the serving
path and Phase 7's parity gate is measured against it. What it means for Phase 7:
weight storage now rounds fp32 → bf16 while compute was ALREADY bf16 autocast, so
the change narrows storage only — and it makes LeRobot's storage precision MATCH
the v1.0 Isaac-GR00T baseline, so Phase 7 compares bf16 against bf16 rather than
bf16 against fp32. Do not "restore upstream defaults" here without re-opening
that decision: fp32 does not fit this GPU at all.

==================== SERVING IMAGE GEOMETRY: THE PAD IS FORCED ON ====================
**The letterbox pad is forced ON for the serving path**, against the flag the
checkpoint's own ``processor_config.json`` sets. Measured, not reasoned:

* LeRobot honours the flag — ``if letter_box_transform:`` wraps its
  ``cv2.copyMakeBorder`` (``processor_groot.py:1423-1433``) — and this checkpoint
  sets it ``false``, so a 480x640 frame reached the VLM as ``(256, 340, 3)``.
* Isaac-GR00T, which TRAINED these weights, pads unconditionally:
  ``LetterBoxPad()`` is element 1 of both its albumentations pipelines
  (``image_augmentations.py:420-487``) and it files ``letter_box_transform`` under
  ``# Backward-compat params (stored but not actively used)``
  (``processing_gr00t_n1d7.py:171-172, 198``). Its output is ``(256, 256, 3)``.
* Forcing the pad ON in LeRobot makes its output **byte-identical** to Isaac's
  (sha256 ``c30150ec…`` from both, across Python 3.10/3.12, numpy 1.26.4/2.2.6 and
  OpenCV 4.11.0/4.13.0). Every other stage already agreed bit-for-bit, so the pad
  gating was the WHOLE divergence.

The injection is upstream's own public step-override seam, in the SAME
``make_pre_post_processors`` call that already carries the device and rename-map
overrides — see ``policy_guard.groot_guard.serving_preprocessor_overrides``, which
owns the one definition of the value, and ``SAFE-01/5``, which reads the EFFECTIVE
value back off the built step and refuses the handshake if the override did not
land. No monkeypatch, no hand-rolled resize or crop: upstream documents its
``cv2.INTER_AREA`` resize and floored center crop as needing to stay bit-exact
(``processor_groot.py:1394-1401``), so a copy would manufacture the mismatch this
removes.

**THE INFERENCE THIS RESTS ON, recorded because it was accepted knowingly.** "The
weights were trained on Isaac's padded square" is INFERRED from "Isaac trained this
checkpoint". The training recipe itself was **not read** — no local artifact records
it. The operator chose to act on the inference rather than confirm it. If Phase 7's
parity work disappoints, this is the FIRST assumption to re-examine. It is not a
settled fact and must not be written up as one.

==================== STATED FALLBACK, TAKEN ====================
The plan's primary shape was ``super().SendPolicyInstructions()`` followed by a
post-load feature fixup. That shape cannot inject precision, because by the time
``super()`` returns the 12.58 GB of fp32 weights have already been materialized
and moved to the GPU — the OOM happens INSIDE the ``super()`` call. So this
module takes the plan's explicitly stated fallback: ``SendPolicyInstructions`` is
REIMPLEMENTED here, duplicating roughly 25 lines of
``policy_server.py:116-171``, so the ``GrootConfig`` can be constructed with the
correct features and precision BEFORE ``from_pretrained`` runs. A second, smaller
duplication (``GrootPolicy._create_groot_model``, ~20 lines) is what carries the
``dtype``/``load_bf16`` kwargs to ``GR00TN17.from_pretrained``, which upstream's
version does not forward.

Both duplications are still composition under D-03 — no patching, no vendoring,
no monkeypatching — but an unpinned duplication of upstream lines is exactly the
drift this project refuses to carry, so ``tests/test_lerobot_upstream_surface.py``
pins every upstream symbol and behaviour they depend on. If an upstream bump
moves those lines, the suite breaks before the robot does.

==================== UPSTREAM INCONSISTENCY, RECORDED ====================
Upstream's own decode-step docstring believes the opposite of blocker 1
(``processor_groot.py:2311-2313``): *"Engines that decode the whole chunk right
after prediction (RTC, async policy server) therefore use the prediction-time
state"*. That claim is contradicted by ``policy_server.py:368-379`` in the SAME
release. Treat it as an upstream inconsistency in 0.6.1, not as guidance.
"""

import collections
import os
import pickle  # nosec B403 - the lerobot wire protocol is pickle in both directions
import time

import grpc
import torch

from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import (
    Observation,
    RemotePolicyConfig,
    TimedAction,
    raw_observation_to_observation,
)
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies import make_pre_post_processors
from lerobot.policies.groot.configuration_groot import GrootConfig

# GR00TN17 is the transformers model class GrootPolicy loads; _tie_unused_qwen_lm_head
# is PRIVATE and imported deliberately. _create_groot_model below duplicates
# upstream's body in order to add two kwargs, and upstream's body calls this
# helper to restore the TF4 weight tie between the unused LM head and the input
# embeddings. Dropping it would leave lm_head untied — ~310M parameters
# (151k vocab x 2048) of dead weight, ~0.6 GiB even at bf16 — so reproducing the
# call is the CONSERVATIVE choice and re-deriving it would be a behaviour change.
# tests/test_lerobot_upstream_surface.py pins the symbol so a rename breaks the
# suite rather than silently inflating VRAM.
from lerobot.policies.groot.groot_n1_7 import GR00TN17, _tie_unused_qwen_lm_head
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.transport import services_pb2
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# ==================== SAFE-01: ONE implementation, TWO call sites ====================
# ``policy_guard`` is Dum-E first-party source, COPIED into the image by
# ``docker/lerobot-policy/Dockerfile`` (``COPY policy_guard/ /app/policy_guard/``,
# with ``ENV PYTHONPATH=/app``) so this module and the Dum-E venv's keyless tests
# import the SAME file. That single COPY is what makes "one implementation, two
# call sites" true across the process boundary; a second copy of the assertions
# inside the image would be exactly the "two code paths to keep in sync" cost
# D-04 option 3 was warned about, and it would be able to drift.
#
# EXPECTED_HORIZON / EXPECTED_TAG are IMPORTED, not restated. They used to be
# module-level literals here (plan 06-01) and ``entrypoint.py`` read them from
# this module; now the guard, this server and the preflight all resolve to ONE
# definition, so they cannot drift to three different numbers. The provenance of
# each — why 40 is the well-lit wrong path, twice, and why the tag is a
# load-bearing input rather than a label — is documented at the constants
# themselves in ``policy_guard/groot_guard.py``.
from policy_guard.groot_guard import (
    EXPECTED_HORIZON,
    EXPECTED_TAG,
    SERVING_LETTER_BOX_TRANSFORM,
    assert_groot_serving_contract,
    serving_preprocessor_overrides,
    snapshot_from_loaded,
)

__all__ = [
    "EXPECTED_HORIZON",
    "EXPECTED_TAG",
    "SEED_ENV_VAR",
    "SERVING_DTYPE",
    "SERVING_LETTER_BOX_TRANSFORM",
    "DumEGrootPolicy",
    "DumEGrootPolicyServer",
    "fixup_policy_features",
    "parameter_dtype_histogram",
]

#: Optional, DIAGNOSTIC-ONLY env var: when it holds an integer, this server seeds
#: the ambient torch RNG in-process immediately before each inference call. When
#: it is unset or empty NOTHING is seeded and the RNG is not touched at all, so
#: the production path is byte-identical to plan 06-01's.
#:
#: **How any result obtained with it must be reported.** ``RemotePolicyConfig``
#: has NO seed field (``async_inference/helpers.py:266-273`` — the six fields are
#: ``policy_type``, ``pretrained_name_or_path``, ``lerobot_features``,
#: ``actions_per_chunk``, ``device``, ``rename_map``), so no seed can travel over
#: this wire. This one is set IN-PROCESS by Dum-E's own ``PolicyServer``
#: subclass. Any determinism result obtained with it is therefore
#: "**deterministic under an in-process seed set by Dum-E's own subclass**" and
#: NEVER "the server honours a seed" — Phase 5 recorded
#: ``seed_verdict: not-honored`` for the sibling GR00T-native server
#: (05-02-SUMMARY.md, same-seed max|diff| 5.51 vs different-seed 4.63 with
#: ``seed``/``random_seed``/``rng_seed`` all tried) and this project's classifier
#: draws that distinction deliberately. Collapsing the two claims would fabricate
#: a capability.
SEED_ENV_VAR: str = "DUME_POLICY_SEED"

#: Weight STORAGE dtype, forced at materialization. See the SERVING PRECISION
#: block in the module docstring: this is a human-approved decision, it matches
#: the v1.0 Isaac-GR00T baseline's bf16, and LeRobot's fp32 default measured
#: 11.47 GiB against a 12288 MiB card. Not a performance tweak — a fit
#: requirement and a Phase 7 parity precondition.
SERVING_DTYPE: torch.dtype = torch.bfloat16


def _refuse(context, message: str):
    """Abort the RPC with ``FAILED_PRECONDITION`` and the guard's own message.

    ``FAILED_PRECONDITION`` is chosen deliberately, and the choice is load-bearing
    on BOTH sides of the wire:

    * **Semantics.** The request was well-formed and the client did nothing
      malformed; the SERVER's state — the checkpoint it resolved, the horizon it
      was told to serve with — is what is unacceptable. That is precisely
      ``FAILED_PRECONDITION``'s meaning.
    * **Retry behaviour.** ``policy/lerobot/session.py``'s ``RETRYABLE_CODES``
      holds only ``UNAVAILABLE`` and ``DEADLINE_EXCEEDED``, so this status is NOT
      retried: the client raises immediately, carrying THIS message. Aborting with
      an "unavailable" or "unknown" status instead would either be retried three
      times — burning the whole budget and then burying the server's own diagnosis
      under a generic "unreachable" — or be indistinguishable from a transport
      fault. The operator would debug the network instead of the checkpoint.

    ``context.abort`` raises, so control never returns to the caller; the
    ``return`` below exists only so a reader is not left wondering.
    """
    return context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)


def _maybe_seed_rng(logger) -> int | None:
    """Seed the ambient torch RNG from :data:`SEED_ENV_VAR`, or do nothing at all.

    Returns the seed applied, or ``None`` when the variable is unset or empty — in
    which case NEITHER ``torch.manual_seed`` nor ``torch.cuda.manual_seed_all`` is
    called and the RNG is not touched in any way. The production path must be
    byte-identical to plan 06-01's; a diagnostic instrument that perturbs the path
    it measures is worthless.

    A malformed value RAISES rather than being ignored, following the
    ``policy/factory.py:57-62`` idiom: a determinism instrument that silently does
    nothing would let a NON-deterministic result be recorded as a seeded one,
    which is threat T-06-16 exactly. The raise is logged at ERROR first because
    this runs on the ``GetActions`` path, whose blanket ``except Exception ->
    Empty()`` (``policy_server.py:214-266``) would otherwise turn it into a
    successful RPC carrying zero bytes with no explanation in the log.
    """
    raw = os.environ.get(SEED_ENV_VAR)
    if raw is None or not raw.strip():
        return None
    try:
        seed = int(raw)
    except ValueError as exc:
        message = (
            f"{SEED_ENV_VAR}={raw!r} is not an integer, so no seed could be applied. "
            f"Refusing to run the inference path with a determinism instrument that "
            f"silently does nothing — a nondeterministic result would then be "
            f"indistinguishable from a seeded one. Unset {SEED_ENV_VAR} to disable "
            f"seeding, or set it to an integer."
        )
        logger.error(message)
        raise ValueError(message) from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def fixup_policy_features(
    config,
    camera_keys,
    height: int,
    width: int,
    state_dim: int,
    action_dim: int,
) -> None:
    """Set the real camera/state/action geometry on a ``GrootConfig``, in place.

    Closes blockers 2 and 3 from the module docstring. Under the stated fallback
    this runs BEFORE ``from_pretrained``, so it PREVENTS the ``(3, 224, 224)``
    ``observation.images.camera`` placeholder rather than repairing it: that
    injection is guarded by ``config is None`` (``modeling_groot.py:247-261``) and
    a config arriving with non-empty ``input_features`` never reaches it.

    Setting ``output_features`` EXPLICITLY is not cosmetic: when it is absent,
    ``validate_features`` inserts ``PolicyFeature(ACTION, shape=(max_action_dim,))``
    with ``max_action_dim = 132`` (``configuration_groot.py:465-470``, ``:256``).
    ``env_action_dim`` is then read straight off that shape
    (``processor_groot.py:1210``) and the decode step's truncation
    ``if self.env_action_dim and decoded.shape[-1] > self.env_action_dim``
    (``:2429-2430``) never fires — so the server would return a 132-wide padded
    action instead of the 6 joints the arm has.
    """
    config.input_features = {
        f"{OBS_IMAGES}.{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
        for cam in camera_keys
    }
    config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}


def parameter_dtype_histogram(module: torch.nn.Module) -> dict[str, int]:
    """Count parameters per dtype. The MEASUREMENT behind the precision claim.

    Emitted on the effective-values INFO line so ``docker logs`` shows what
    precision the served weights are actually in, rather than what the config
    asked for. Specifically this is what confirms — rather than assumes — that
    ``groot_n1_7.py:310``'s ``if load_bf16 and trainable_params_fp32:`` cast back
    to fp32 is inert here: if any parameters return as ``float32``, they show up
    in this histogram with their count.
    """
    counts: collections.Counter[str] = collections.Counter()
    for parameter in module.parameters():
        counts[str(parameter.dtype).replace("torch.", "")] += parameter.numel()
    return dict(counts)


class DumEGrootPolicy(GrootPolicy):
    """``GrootPolicy`` whose weights MATERIALIZE in bf16 instead of fp32.

    Duplicates ``GrootPolicy._create_groot_model``
    (``modeling_groot.py:88-118``) for one reason: it must add ``dtype`` and
    ``load_bf16`` to the ``GR00TN17.from_pretrained`` call, and upstream's version
    builds that kwargs dict from a FIXED set of ``self.config`` fields with no
    passthrough. ``GrootConfig`` carries no ``dtype`` and no ``load_bf16`` field
    (checked: neither appears in ``configuration_groot.py``), so no amount of
    config injection at the ``GrootPolicy`` level can reach the call —
    ``GrootPolicy.from_pretrained``'s own kwargs passthrough is guarded by
    ``if hasattr(config, key)`` (``modeling_groot.py:266-268``) and would drop
    both SILENTLY.

    A post-load cast is NOT an equivalent alternative: it is precisely what
    ``model_params_fp32`` already does in the opposite direction, and it would
    still require materializing 12.58 GB of fp32 weights first.

    Everything else here is upstream's body, unchanged, so a version bump that
    adds a field to that dict will produce a visibly stale copy —
    ``tests/test_lerobot_upstream_surface.py`` pins the fields this copy carries.
    """

    def _create_groot_model(self):
        model_kwargs = {
            "pretrained_model_name_or_path": self.config.base_model_path,
            "tune_llm": self.config.tune_llm,
            "tune_visual": self.config.tune_visual,
            "tune_projector": self.config.tune_projector,
            "tune_diffusion_model": self.config.tune_diffusion_model,
            # Forwarded as a GR00TN17Config override; read back by set_trainable_parameters.
            "tune_top_llm_layers": self.config.tune_top_llm_layers,
            "use_flash_attention": self.config.use_flash_attention,
            # ---- THE ONLY TWO ADDITIONS TO UPSTREAM'S DICT ----
            # `dtype` is popped by transformers' from_pretrained
            # (modeling_utils.py:236) and governs BOTH module instantiation and
            # the shard load, so it is what actually makes the action head bf16.
            # Without it, dtype defaults to "auto", which resolves this
            # checkpoint's config.json `dtype: 'float32'` and materializes 3.144B
            # fp32 parameters — 11.47 GiB, which does not fit the card.
            "dtype": SERVING_DTYPE,
            # `load_bf16` is not recognized by from_pretrained, so it lands on
            # GR00TN17Config and reaches groot_n1_7.py:288-289, where it sets
            # torch_dtype on the nested Cosmos-Reason2-2B backbone load. That
            # keeps the backbone from transiting fp32 in host RAM. It is the knob
            # the checkpoint's own config.json names (declaring False, while also
            # declaring model_dtype 'bfloat16' — which LeRobot never reads).
            "load_bf16": True,
        }
        # Surface the inference-time knobs onto the model config only when the user set them; None
        # leaves the value baked into the checkpoint untouched.
        if self.config.num_inference_timesteps is not None:
            model_kwargs["num_inference_timesteps"] = self.config.num_inference_timesteps
        if self.config.rtc_ramp_rate is not None:
            model_kwargs["rtc_ramp_rate"] = self.config.rtc_ramp_rate

        model = GR00TN17.from_pretrained(
            **model_kwargs,
            tune_vlln=self.config.tune_vlln,
            transformers_loading_kwargs={"trust_remote_code": True},
        )
        backbone = getattr(model, "backbone", None)
        qwen_model = getattr(backbone, "model", None)
        if qwen_model is not None:
            _tie_unused_qwen_lm_head(qwen_model)
        # Upstream casts every floating-point parameter UP to fp32 here when
        # config.model_params_fp32 is True — which is its DEFAULT. The
        # SendPolicyInstructions below injects False, so this stays inert; the
        # branch is reproduced rather than deleted so the copy stays diffable
        # against upstream.
        if self.config.model_params_fp32:
            self._cast_model_parameters_to_fp32(model)
        return model


class DumEGrootPolicyServer(PolicyServer):
    """Composition-only ``PolicyServer`` subclass. Every override calls ``super()``
    or a public/protected upstream helper — never a patch, never a fork."""

    #: The cameras this server serves, and their frame geometry. These must agree
    #: with ``policy/lerobot/features.py``'s ``CAMERA_KEYS``/``FRAME_*`` on the
    #: client side: the client's ``lerobot_features`` decides which
    #: ``observation.images.<cam>`` keys arrive, and ``config.input_features``
    #: decides which ones are looked up. A disagreement is a ``KeyError``.
    CAMERA_KEYS: tuple[str, ...] = ("wrist", "front")
    FRAME_HEIGHT: int = 480
    FRAME_WIDTH: int = 640
    STATE_DIM: int = 6
    ACTION_DIM: int = 6

    def SendPolicyInstructions(self, request, context):  # noqa: N802 - upstream gRPC name
        """Build the config, THEN load — the plan's stated fallback.

        Reimplements ``policy_server.py:116-171`` rather than calling ``super()``.
        The reason is precision, not features: ``super()`` materializes the weights
        itself, so by the time it returns, 3.144B fp32 parameters (11.47 GiB) have
        already been created and moved to a 12288 MiB card. Every value that has to
        be right BEFORE the load — features, embodiment tag, storage dtype — is
        therefore injected into a ``GrootConfig`` here and handed to
        ``from_pretrained`` via its ``config=`` argument. See the module
        docstring's SERVING PRECISION and STATED FALLBACK blocks.

        A pleasant side effect: with ``input_features`` correct up front,
        ``from_pretrained``'s ``(3, 224, 224)`` placeholder branch
        (``modeling_groot.py:249-261``) never runs at all — it is guarded by
        ``config is None`` — so blockers 2 and 3 are prevented rather than
        repaired.
        """
        # Upstream's guard, reproduced: SendPolicyInstructions returns early when
        # the server is not running, and only Ready() clears shutdown_event
        # (policy_server.py:108-121). This is what makes the client's
        # Ready-before-SendPolicyInstructions ordering mandatory.
        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec B301 - upstream wire format

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        requested_path = str(policy_specs.pretrained_name_or_path)

        # ---- THE CONFIG INJECTION: everything that must be true before the load ----
        config = GrootConfig(
            base_model_path=requested_path,
            # Explicit, not inherited from the default, per D-11: this is the ONE
            # tag of the checkpoint's nine with 16 delta_indices, and tag
            # inference returns None for this checkpoint.
            embodiment_tag=EXPECTED_TAG,
            # False, against LeRobot's True default. See SERVING PRECISION: True
            # casts every float parameter UP to fp32 after the load
            # (modeling_groot.py:116), which would undo the bf16 materialization
            # DumEGrootPolicy just performed.
            model_params_fp32=False,
        )
        fixup_policy_features(
            config,
            camera_keys=self.CAMERA_KEYS,
            height=self.FRAME_HEIGHT,
            width=self.FRAME_WIDTH,
            state_dim=self.STATE_DIM,
            action_dim=self.ACTION_DIM,
        )

        start = time.perf_counter()
        # DumEGrootPolicy, not get_policy_class(self.policy_type)'s GrootPolicy:
        # the subclass is what carries dtype/load_bf16 into the model load.
        self.policy = DumEGrootPolicy.from_pretrained(requested_path, config=config)
        self.policy.to(self.device)

        # Read the resolved path back off the config rather than trusting the
        # request. GrootPolicy.from_pretrained SETS
        # config.base_model_path = str(pretrained_name_or_path) itself
        # (modeling_groot.py:249, 263-264), so this value is what the loaded
        # object will actually read the checkpoint sidecars from — which makes it
        # simultaneously the SAFE-01/1 evidence. The pickled request only says
        # what the client ASKED for.
        pretrained_path = self.policy.config.base_model_path

        # Same public call upstream makes at policy_server.py:152-163, including
        # the rename_observations_processor override, so the pipelines are built
        # against the corrected features and env_action_dim is 6 rather than 132.
        #
        # ---- THE IMAGE-GEOMETRY INJECTION ----
        # A THIRD override rides the same call: the letterbox pad, forced ON. See the
        # SERVING IMAGE GEOMETRY block in the module docstring for the measurement,
        # and `serving_preprocessor_overrides` for why the value lives in
        # policy_guard rather than here. It is merged rather than restated so the
        # value the SAFE-01 guard asserts and the value injected here cannot become
        # two numbers. An override key that no longer matches a step raises KeyError
        # listing the available keys, and an unknown field raises TypeError listing
        # the available fields (processor_groot.py:428-450) — so a pinned-lerobot
        # rename fails at the handshake, loudly, instead of silently dropping the pad
        # and serving a geometry the weights never saw.
        device_override = {"device": self.device}
        preprocessor_overrides = {
            "device_processor": device_override,
            "rename_observations_processor": {"rename_map": policy_specs.rename_map},
        }
        preprocessor_overrides.update(serving_preprocessor_overrides())
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=pretrained_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()
        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        # One INFO line carrying every EFFECTIVE value, so an operator can read
        # the resolved configuration out of `docker logs` without a debugger. The
        # emitted chunk length is deliberately NOT among them: three independent
        # truncations force 16 regardless of whether the config is right, so it
        # corroborates and never evidences.
        #
        # param_dtypes and cuda_allocated_MiB ARE among them: they are the
        # measurement behind the bf16 decision, and a silent regression to fp32
        # would otherwise only show up as an OOM on a busier GPU.
        decode_step_type = type(self.postprocessor.steps[0]).__name__
        cuda_allocated_mib = (
            round(torch.cuda.memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else None
        )
        self.logger.info(
            "DumE effective serving values | base_model_path=%s | embodiment_tag=%s | "
            "actions_per_chunk=%s | decode_step=%s | input_features=%s | output_features=%s | "
            "model_params_fp32=%s | param_dtypes=%s | cuda_allocated_MiB=%s",
            pretrained_path,
            self.policy.config.embodiment_tag,
            self.actions_per_chunk,
            decode_step_type,
            sorted(self.policy.config.input_features),
            sorted(self.policy.config.output_features),
            self.policy.config.model_params_fp32,
            parameter_dtype_histogram(self.policy),
            cuda_allocated_mib,
        )

        # ==================== SAFE-01 GUARD: THE POST-LOAD CALL SITE ====================
        # This is the ONLY place in the system that sees the object which will
        # actually run inference: upstream loads the policy INSIDE this request
        # handler, so there is no earlier moment at which ``self.policy``,
        # ``self.preprocessor`` and ``self.postprocessor`` exist. D-04 option 3's
        # post-load site.
        #
        # ``self.actions_per_chunk`` is the CLIENT-supplied handshake value, not a
        # re-read of the checkpoint. That is what makes this the D-11 config-drift
        # catcher: the horizon is configurable, so it can be configured WRONG, and
        # a client that asks for 40 must be refused rather than obeyed. Both 40s
        # are traps and both are well lit — ``GrootConfig``'s own default
        # ``chunk_size``/``n_action_steps`` are 40 and this checkpoint's own
        # ``config.json`` advertises ``action_horizon: 40``.
        #
        # The snapshot build is INSIDE the try on purpose: ``snapshot_from_loaded``
        # locates the pack step by ``state_dropout_prob``, the encode step by
        # ``letter_box_transform`` and the decode step by ``env_action_dim``, and
        # raises ``ValueError`` naming what it looked for if a pinned-lerobot
        # pipeline reshape moved any of them. That is a refusal to serve for the
        # same reason a failed assertion is, and it must reach the operator as one
        # rather than as an un-named exception escaping a gRPC handler.
        try:
            snapshot = snapshot_from_loaded(
                self.policy.config,
                self.preprocessor,
                self.postprocessor,
                self.actions_per_chunk,
            )
            assert_groot_serving_contract(snapshot)
        except ValueError as exc:
            self.logger.error("SAFE-01 guard: REFUSED | %s", exc)
            # Drop the un-validated policy BEFORE refusing. ``context.abort``
            # terminates THIS RPC, but a client that ignores the abort and calls
            # SendObservations/GetActions anyway must not be served from a policy
            # the guard rejected — and ``GetActions``'s blanket
            # ``except Exception -> Empty()`` would turn the resulting AttributeError
            # into a SUCCESSFUL RPC carrying zero bytes, which
            # policy/lerobot/session.py's zero-length guard names explicitly.
            # There is deliberately NO path here that logs a violation and returns
            # the reply: no warn-and-serve, and the ValueError is re-raised as an
            # abort rather than caught and continued (T-06-13).
            self.policy = None
            _refuse(context, str(exc))

        # ONE INFO line, so a single `docker logs | grep` shows BOTH that the guard
        # ran and what it saw. A guard that passes SILENTLY is indistinguishable
        # from a guard that never ran, which is D-05's stated failure mode and
        # threat T-06-14 — the repudiation risk is the whole reason this line
        # exists. Every value is read off the snapshot the assertions just ran
        # against, never re-derived, so the line cannot report something the guard
        # did not actually check.
        #
        # served_letter_box_transform is on this line rather than only in the guard
        # because it is the ONE value here that the checkpoint's own configuration
        # CONTRADICTS: the checkpoint declares False and the serving path forces True.
        # An operator reading `docker logs` must be able to see which of the two is
        # actually in effect without attaching a debugger, and a future reader must be
        # able to tell the forced value apart from the declared one.
        self.logger.info(
            "SAFE-01 guard: PASS | base_model_path=%s | embodiment_tag=%s | "
            "actions_per_chunk=%s | checkpoint_horizon=%s | decode_step=%s | "
            "checkpoint_letter_box_transform=%s | served_letter_box_transform=%s",
            snapshot.base_model_path,
            snapshot.embodiment_tag,
            snapshot.configured_actions_per_chunk,
            snapshot.checkpoint_horizon,
            snapshot.decode_step_type,
            snapshot.letter_box_transform,
            snapshot.served_letter_box_transform,
        )
        return services_pb2.Empty()

    def _predict_action_chunk(self, observation_t) -> list[TimedAction]:
        """Upstream's pipeline with steps 4-5 replaced by ONE full-chunk decode.

        Steps 1-3 are upstream's, called through the same helpers upstream calls.
        Step 4 is the fix for blocker 1: ``self.postprocessor`` is invoked ONCE on
        the whole ``(B, T, D)`` tensor instead of ``T`` times on ``(B, D)`` slices.
        """
        # 1. Prepare observation (upstream policy_server.py:342-346).
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )

        # 2. Apply preprocessor. This is also what caches the anchor state inside
        #    GrootN17PackInputsStep that the decode step reads by direct object
        #    reference (processor_groot.py:2394) — which is why the relative->
        #    absolute decode must happen server-side and cannot be moved onto the
        #    wire.
        observation = self.preprocessor(observation)
        self.last_processed_obs = observation_t

        # ==================== SEEDED-DETERMINISM HOOK (opt-in, inert when unset) ====================
        # Env-gated and DIAGNOSTIC. When DUME_POLICY_SEED is unset or empty this
        # touches the RNG in no way, so the production path is byte-identical to
        # plan 06-01's; see _maybe_seed_rng and SEED_ENV_VAR.
        #
        # It sits HERE, immediately before _get_action_chunk, because that is where
        # the flow-matching sampler draws its initial noise from the ambient torch
        # RNG (this checkpoint decodes over num_inference_timesteps: 4). Seeding
        # earlier — at startup, or at handshake — would seed once and then let five
        # successive calls advance the same generator, which measures RNG
        # continuation rather than repeatability.
        #
        # REPORTING CONSTRAINT, restated at the call site because this is where a
        # future reader will be tempted: RemotePolicyConfig has no seed field
        # (helpers.py:266-273), so this seed is set IN-PROCESS by Dum-E's own
        # PolicyServer subclass. Any result obtained with it is "deterministic under
        # an in-process seed", NEVER "the server honours a seed" — Phase 5 recorded
        # seed_verdict: not-honored for the sibling GR00T-native server and this
        # project's classifier draws that distinction deliberately.
        _maybe_seed_rng(self.logger)

        # 3. Inference. Truncates to the client-supplied actions_per_chunk.
        action_tensor = self._get_action_chunk(observation)

        # 4. Postprocess the WHOLE chunk in ONE call. Upstream loops per timestep
        #    here (policy_server.py:368-379); GrootN17ActionDecodeStep refuses
        #    that shape on native relative actions (processor_groot.py:2345-2350)
        #    because the per-timestep (16, 5) statistics and the cached anchor
        #    state only make sense against a full chunk.
        action_tensor = self.postprocessor(action_tensor)
        action_tensor = action_tensor.squeeze(0).detach().cpu()

        # 5. Convert to TimedAction list (upstream _time_action_chunk).
        return self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
