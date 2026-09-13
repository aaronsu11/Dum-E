"Dum-E's ``PolicyServer`` subclass — the four fixes vanilla 0.6.1 needs."

import collections
import os
import time
import pickle  # nosec B403 - the lerobot wire protocol is pickle in both directions
from typing import NoReturn

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
from lerobot.policies.groot.groot_n1_7 import GR00TN17, _tie_unused_qwen_lm_head
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.transport import services_pb2
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

# Loading and serving share the same model contract.
from policy.backends.lerobot.models.groot import (
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
SEED_ENV_VAR: str = "DUME_POLICY_SEED"

#: Weight STORAGE dtype, forced at materialization. See the SERVING PRECISION
#: block in the module docstring: this is a human-approved decision, it matches
#: the v1.0 Isaac-GR00T baseline's bf16, and LeRobot's fp32 default measured
#: 11.47 GiB against a 12288 MiB card. Not a performance tweak — a fit
#: requirement and a Phase 7 parity precondition.
SERVING_DTYPE: torch.dtype = torch.bfloat16


def _refuse(context, message: str) -> NoReturn:
    "Abort the RPC with ``FAILED_PRECONDITION`` and the guard's own message."
    context.abort(grpc.StatusCode.FAILED_PRECONDITION, message)
    # Only reachable via a context whose abort RETURNS. Refusing loudly is the
    # only acceptable outcome: this function's callers have already dropped the
    # policy, so continuing would report a pass for a refusal.
    raise RuntimeError(
        "context.abort() returned instead of raising, so this refusal did not "
        "terminate the RPC. Refusing by raising instead: the caller's next "
        "statement logs 'SAFE-01 guard: PASS' and returns an OK handshake reply, "
        f"and the guard REFUSED. Original refusal: {message}"
    )


def _maybe_seed_rng(logger) -> int | None:
    'Seed the ambient torch RNG from :data:`SEED_ENV_VAR`, or do nothing at all.'
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
    'Set the real camera/state/action geometry on a ``GrootConfig``, in place.'
    config.input_features = {
        f"{OBS_IMAGES}.{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(3, height, width))
        for cam in camera_keys
    }
    config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    config.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,))}


def parameter_dtype_histogram(module: torch.nn.Module) -> dict[str, int]:
    'Count parameters per dtype. The MEASUREMENT behind the precision claim.'
    counts: collections.Counter[str] = collections.Counter()
    for parameter in module.parameters():
        counts[str(parameter.dtype).replace("torch.", "")] += parameter.numel()
    return dict(counts)


class DumEGrootPolicy(GrootPolicy):
    '``GrootPolicy`` whose weights MATERIALIZE in bf16 instead of fp32.'

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
            # `dtype` is popped by transformers' from_pretrained
            # (modeling_utils.py:236) and governs BOTH module instantiation and
            "dtype": SERVING_DTYPE,
            # `load_bf16` is not recognized by from_pretrained, so it lands on
            # GR00TN17Config and reaches groot_n1_7.py:288-289, where it sets
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
    """Inject the pinned GR00T config before load and decode complete relative chunks."""

    #: The cameras this server serves, and their frame geometry.
    #: **PINNED COPIES, and ``policy/backends/lerobot/features.py`` is the DEFINITION** —
    CAMERA_KEYS: tuple[str, ...] = ("wrist", "front")
    FRAME_HEIGHT: int = 480
    FRAME_WIDTH: int = 640
    STATE_DIM: int = 6
    ACTION_DIM: int = 6


    def SendPolicyInstructions(self, request, context):  # noqa: N802 - upstream gRPC name
        "Build the config, THEN load — the plan's stated fallback."
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
        # Defence in depth for the same root cause as the client-side reset() fix:
        # `from_pretrained` allocates and moves a NEW ~6 GB policy while the
        self.policy = None
        self.preprocessor = None
        self.postprocessor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # DumEGrootPolicy, not get_policy_class(self.policy_type)'s GrootPolicy:
        # the subclass is what carries dtype/load_bf16 into the model load.
        self.policy = DumEGrootPolicy.from_pretrained(requested_path, config=config)
        self.policy.to(self.device)

        # Read the resolved path back off the config rather than trusting the
        # request. GrootPolicy.from_pretrained SETS
        pretrained_path = self.policy.config.base_model_path

        # Same public call upstream makes at policy_server.py:152-163, including
        # the rename_observations_processor override, so the pipelines are built
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

        # This is the ONLY place in the system that sees the object which will
        # actually run inference: upstream loads the policy INSIDE this request
        try:
            snapshot = snapshot_from_loaded(
                self.policy.config,
                self.preprocessor,
                self.postprocessor,
                self.actions_per_chunk,
            )
            assert_groot_serving_contract(snapshot)
        except Exception as exc:  # noqa: BLE001 - a guard that cannot run must refuse, not serve
            self.logger.error("SAFE-01 guard: REFUSED | %s: %s", type(exc).__name__, exc)
            # Drop the un-validated policy BEFORE refusing, UNCONDITIONALLY.
            # ``context.abort`` terminates THIS RPC, but a client that ignores the
            self.policy = None
            self.preprocessor = None
            self.postprocessor = None
            # RETURNED, not called as a statement: see the note in ``_refuse``. The
            # PASS log below must be unreachable from here even if ``abort`` ever
            # stops raising.
            return _refuse(context, f"{type(exc).__name__}: {exc}")

        # ONE INFO line, so a single `docker logs | grep` shows BOTH that the guard
        # ran and what it saw. A guard that passes SILENTLY is indistinguishable
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


    def _predict_action_chunk(self, observation_t):
        return self._predict_action_chunk_impl(observation_t)

    def _predict_action_chunk_impl(self, observation_t) -> list[TimedAction]:
        "Upstream's pipeline with steps 4-5 replaced by ONE full-chunk decode."
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

        # Env-gated and DIAGNOSTIC. When DUME_POLICY_SEED is unset or empty this
        # touches the RNG in no way, so the production path is byte-identical to
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
