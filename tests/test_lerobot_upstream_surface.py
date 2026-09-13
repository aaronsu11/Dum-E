'Keyless pins on the upstream LeRobot surface Dum-E DUPLICATES.'

import ast
import inspect
import textwrap
from pathlib import Path

import grpc
import pytest

from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedAction, TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.policies import make_pre_post_processors
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1_7 import GR00T_N1_7_DEFAULTS, GR00TN17, _tie_unused_qwen_lm_head
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.transport import services_pb2, services_pb2_grpc

#: Dum-E's copy. Read as TEXT, never imported: it is container code that pulls in
#: the ``groot`` extra's transformers stack, which the client venv deliberately
#: does not carry (``tests/test_container_contract.py``'s dependency-isolation
#: guard is what keeps it out).
SERVER_PY = Path(__file__).resolve().parent.parent / "docker" / "lerobot-policy" / "server.py"

#: The real fine-tuned checkpoint. Read CONFIG-ONLY (three small JSON sidecars);
#: no weights, no GPU, no network. Absent -> ``pytest.fail``, never a skip.
REAL_CHECKPOINT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "groot-so101"


def _parse_server_py() -> ast.Module:
    return ast.parse(SERVER_PY.read_text(encoding="utf-8"))


def _find_function(tree: ast.Module, class_name: str | None, func_name: str) -> ast.FunctionDef:
    """Locate a function definition, optionally scoped to a class."""
    scopes: list[ast.AST] = [tree]
    if class_name is not None:
        scopes = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        ]
        assert scopes, f"{SERVER_PY.name} defines no class {class_name!r}"
    for scope in scopes:
        for node in ast.walk(scope):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
    raise AssertionError(f"{SERVER_PY.name} defines no {class_name}.{func_name}")


def _self_assignments(func: ast.FunctionDef) -> set[str]:
    """Every ``self.<name>`` assigned in ``func``, tuple targets included."""
    found: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign):
            continue
        targets: list[ast.expr] = []
        for target in node.targets:
            targets.extend(target.elts if isinstance(target, ast.Tuple) else [target])
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
            ):
                found.add(target.attr)
    return found


def _self_config_reads(func: ast.FunctionDef) -> set[str]:
    """Every ``self.config.<name>`` read in ``func``."""
    found: set[str] = set()
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "config"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "self"
        ):
            found.add(node.attr)
    return found


def _upstream_ast(func) -> ast.FunctionDef:
    """Parse an installed upstream function back into an AST node."""
    parsed = ast.parse(textwrap.dedent(inspect.getsource(func)))
    node = parsed.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


# ---------------------------------------------------------------------------
# Duplication 1: PolicyServer.SendPolicyInstructions
# ---------------------------------------------------------------------------


def test_upstream_send_policy_instructions_still_exists_to_override():
    """The method Dum-E overrides must still exist, or the override is dead code."""
    assert callable(getattr(PolicyServer, "SendPolicyInstructions", None)), (
        "lerobot.async_inference.policy_server.PolicyServer.SendPolicyInstructions is gone. "
        "DumEGrootPolicyServer's override would silently stop overriding anything and the "
        "vanilla handshake would run — which cannot serve this checkpoint at all."
    )


def test_dume_copy_assigns_exactly_upstream_send_policy_instructions_state():
    'The copy must set the SAME server state upstream sets — no more, no fewer.'
    upstream = _self_assignments(_upstream_ast(PolicyServer.SendPolicyInstructions))
    ours = _self_assignments(
        _find_function(_parse_server_py(), "DumEGrootPolicyServer", "SendPolicyInstructions")
    )

    missing = upstream - ours
    extra = ours - upstream
    assert not missing, (
        f"docker/lerobot-policy/server.py's SendPolicyInstructions copy does not assign "
        f"{sorted(missing)}, which upstream's version does. The copy is STALE against the "
        f"installed lerobot: re-diff it against policy_server.py:116-171. Missing per-handshake "
        f"state is a silent wrong-inference bug, not a crash."
    )
    assert not extra, (
        f"the copy assigns {sorted(extra)}, which upstream does not. Either upstream dropped "
        f"state the copy still sets, or the copy grew a responsibility that belongs elsewhere."
    )


def test_remote_policy_config_fields_the_copy_reads_still_exist():
    """The handshake fields the copy unpacks are the client's whole contract."""
    fields = set(RemotePolicyConfig.__dataclass_fields__)
    required = {
        "policy_type",
        "pretrained_name_or_path",
        "lerobot_features",
        "actions_per_chunk",
        "device",
        "rename_map",
    }
    missing = required - fields
    assert not missing, (
        f"RemotePolicyConfig no longer carries {sorted(missing)}. Both Dum-E's "
        f"SendPolicyInstructions copy and policy/lerobot/session.py's connect() build this "
        f"payload, so a renamed field breaks the handshake in BOTH directions."
    )


def test_groot_is_still_a_supported_policy_type():
    """The copy reproduces upstream's ``policy_type not in SUPPORTED_POLICIES`` guard."""
    assert "groot" in SUPPORTED_POLICIES, (
        f"'groot' is no longer in SUPPORTED_POLICIES ({SUPPORTED_POLICIES}), so the copy's "
        f"validation guard would reject the only policy type this phase serves."
    )


# ---------------------------------------------------------------------------
# Duplication 2: GrootPolicy._create_groot_model
# ---------------------------------------------------------------------------


def test_upstream_create_groot_model_still_exists_to_override():
    assert callable(getattr(GrootPolicy, "_create_groot_model", None)), (
        "GrootPolicy._create_groot_model is gone. DumEGrootPolicy's override — the ONLY place "
        "the bf16 storage dtype is injected before the weights materialize — would stop being "
        "called, and the load would silently revert to 3.144B fp32 parameters (11.47 GiB, "
        "which does not fit a 12288 MiB card)."
    )


def test_dume_copy_reads_exactly_upstream_create_groot_model_config_fields():
    'The copy must forward the SAME config fields upstream forwards.'
    upstream = _self_config_reads(_upstream_ast(GrootPolicy._create_groot_model))
    ours = _self_config_reads(
        _find_function(_parse_server_py(), "DumEGrootPolicy", "_create_groot_model")
    )

    missing = upstream - ours
    extra = ours - upstream
    assert not missing, (
        f"docker/lerobot-policy/server.py's _create_groot_model copy does not read "
        f"{sorted(missing)}, which upstream's version forwards to GR00TN17.from_pretrained. "
        f"Re-diff the copy against modeling_groot.py:88-118."
    )
    assert not extra, (
        f"the copy reads {sorted(extra)}, which upstream's version does not. If upstream "
        f"dropped one of these, the copy is keeping a dead knob alive."
    )


def test_tie_unused_qwen_lm_head_still_exists():
    """The copy reproduces upstream's call to this private helper.

    Dropping the tie leaves ``lm_head`` untied from the input embeddings — roughly
    310M parameters (151k vocab x 2048) of dead weight, ~0.6 GiB even at bf16.
    """
    assert callable(_tie_unused_qwen_lm_head), (
        "lerobot.policies.groot.groot_n1_7._tie_unused_qwen_lm_head is gone or not callable. "
        "It is a PRIVATE symbol imported deliberately by docker/lerobot-policy/server.py "
        "because the _create_groot_model copy must reproduce upstream's call to it."
    )


def test_groot_n17_from_pretrained_forwards_unknown_kwargs_downstream():
    '``dtype`` and ``load_bf16`` reach transformers/GR00TN17Config only via ``**kwargs``.'
    source = inspect.getsource(GR00TN17.from_pretrained)
    assert "super().from_pretrained(" in source, (
        "GR00TN17.from_pretrained no longer delegates to super().from_pretrained; the dtype "
        "injection path has changed shape."
    )
    assert "**kwargs" in source.split("super().from_pretrained(", 1)[1], (
        "GR00TN17.from_pretrained no longer forwards **kwargs to super().from_pretrained, so "
        "Dum-E's injected dtype=torch.bfloat16 and load_bf16=True would be SILENTLY DROPPED "
        "and the model would materialize in fp32 again."
    )


# ---------------------------------------------------------------------------
# Why the duplication exists at all: the precision knobs' actual locations
# ---------------------------------------------------------------------------


def test_load_bf16_is_not_a_lerobot_groot_config_field():
    'The reason config injection alone cannot force bf16.'
    fields = set(GrootConfig.__dataclass_fields__)
    assert "load_bf16" not in fields, (
        "GrootConfig now HAS a load_bf16 field. DumEGrootPolicy._create_groot_model exists "
        "only because it did not; re-check whether that ~20-line duplication can be replaced "
        "by config injection."
    )
    assert "dtype" not in fields, (
        "GrootConfig now HAS a dtype field. Same conclusion as load_bf16 above: re-check "
        "whether the _create_groot_model duplication is still necessary."
    )


def test_model_params_fp32_defaults_true_and_is_what_casts_weights_up():
    """LeRobot's own default is fp32 storage; the injection turns it off."""
    assert GrootConfig.__dataclass_fields__["model_params_fp32"].default is True, (
        "GrootConfig.model_params_fp32 no longer defaults to True. Dum-E injects False "
        "against that default; if upstream changed it, re-read the SERVING PRECISION block "
        "in docker/lerobot-policy/server.py before assuming the injection is still needed."
    )
    source = inspect.getsource(GrootPolicy._create_groot_model)
    assert "if self.config.model_params_fp32:" in source, (
        "the model_params_fp32 branch is gone from _create_groot_model. That branch casting "
        "every float parameter UP to fp32 is one of the two reasons the bf16 injection exists."
    )


def test_use_bf16_is_autocast_only_not_storage():
    "Recorded so nobody 'fixes' the precision by flipping this instead."
    assert GrootConfig.__dataclass_fields__["use_bf16"].default is True
    source = inspect.getsource(GrootPolicy)
    autocast_lines = [line for line in source.splitlines() if "use_bf16" in line]
    assert autocast_lines, "use_bf16 is no longer read in modeling_groot's GrootPolicy"
    assert all("autocast" in line for line in autocast_lines), (
        f"use_bf16 is now read somewhere other than a torch.autocast call: {autocast_lines}. "
        f"It may no longer be compute-only; re-derive the storage-precision reasoning."
    )


def test_model_dtype_is_dead_for_groot_under_lerobot():
    "The checkpoint declares ``model_dtype: 'bfloat16'`` and LeRobot never reads it."
    groot_package = Path(inspect.getfile(GrootConfig)).parent
    occurrences = {
        path.name: path.read_text(encoding="utf-8").count("model_dtype")
        for path in sorted(groot_package.glob("*.py"))
        if "model_dtype" in path.read_text(encoding="utf-8")
    }
    assert occurrences == {"groot_n1_7.py": 1}, (
        f"'model_dtype' now appears in lerobot/policies/groot/ as {occurrences}, not as a "
        f"single unread entry in GR00T_N1_7_DEFAULTS. Re-check whether LeRobot now honours "
        f"the checkpoint's declared bfloat16 intent on its own."
    )
    assert GR00T_N1_7_DEFAULTS["model_dtype"] == "bfloat16"
    assert GR00T_N1_7_DEFAULTS["load_bf16"] is False, (
        "GR00T_N1_7_DEFAULTS['load_bf16'] is no longer False, so the checkpoint's own "
        "config.json value may no longer be what decides backbone precision."
    )


def test_build_n1_7_processor_takes_no_revision_and_is_what_the_build_asserts():
    "The pinned-cache mechanism, and the Dockerfile's offline build assertion."
    from lerobot.policies.groot.configuration_groot import GROOT_N1_7_BACKBONE_MODEL
    from lerobot.policies.groot.processor_groot import _build_n1_7_processor

    parameters = inspect.signature(_build_n1_7_processor).parameters
    assert "revision" not in parameters, (
        "_build_n1_7_processor now accepts a `revision` argument. The pinned-cache + "
        "HF_HUB_OFFLINE=1 mechanism exists only because it did not; a revision argument "
        "would be a stronger, more direct pin. Re-read D-08 before leaving this as is."
    )
    assert parameters["model_name"].default == GROOT_N1_7_BACKBONE_MODEL == "nvidia/Cosmos-Reason2-2B", (
        f"_build_n1_7_processor's default model_name is {parameters['model_name'].default!r}; "
        f"the Dockerfile pre-caches nvidia/Cosmos-Reason2-2B and writes refs/main for THAT "
        f"repo, so a different default would make the offline cache miss at first inference."
    )

    source = inspect.getsource(_build_n1_7_processor)
    assert source.count("from_pretrained(model_name") == 3, (
        "the three .from_pretrained(model_name, ...) calls in _build_n1_7_processor "
        "(AutoTokenizer, Qwen2VLImageProcessor, Qwen3VLVideoProcessor — "
        "processor_groot.py:1371-1373) have changed shape. Each resolves revision 'main' "
        "from the hub cache, which is what the Dockerfile's hand-written refs/main serves."
    )


def test_placeholder_visual_feature_injection_is_guarded_by_config_is_none():
    'The copy PREVENTS the 224x224 placeholder rather than repairing it.'
    source = inspect.getsource(GrootPolicy.from_pretrained)
    assert "if config is None:" in source, (
        "GrootPolicy.from_pretrained no longer guards its default-config construction with "
        "`if config is None:`; re-check whether passing config= still prevents the "
        "(3, 224, 224) observation.images.camera placeholder injection."
    )
    head, _, tail = source.partition("if config is None:")
    assert "224" in tail and "224" not in head, (
        "the (3, 224, 224) placeholder is no longer injected exclusively inside the "
        "`config is None` branch, so passing a config may no longer prevent it."
    )


# ---------------------------------------------------------------------------
# Plan 06-06: the whole binding surface, not just the two duplications
# ---------------------------------------------------------------------------

#: Every ``PolicyServer`` member ``docker/lerobot-policy/server.py`` overrides or
#: calls through. Overridden: ``SendPolicyInstructions``, ``_predict_action_chunk``.
#: Inherited and served as-is: ``SendObservations``, ``GetActions``, ``Ready``.
#: Called from inside the override: ``_get_action_chunk``, ``_time_action_chunk``,
#: ``policy_image_features``.
OVERRIDDEN_POLICY_SERVER_ATTRS = (
    "_predict_action_chunk",
    "SendPolicyInstructions",
    "SendObservations",
    "GetActions",
    "Ready",
    "_get_action_chunk",
    "_time_action_chunk",
    "policy_image_features",
)

#: The four methods of the ``AsyncInference`` service. This wire is the whole
#: contract between ``policy/lerobot/session.py``, ``DumEGrootPolicyServer`` and
#: ``scripts/mock_grpc_policy_server.py``'s servicer.
ASYNC_INFERENCE_METHODS = frozenset(
    {"Ready", "SendPolicyInstructions", "SendObservations", "GetActions"}
)


def test_policy_server_still_exposes_the_overridden_methods():
    'A rename un-hooks an override with NO import error and no crash.'
    missing = [name for name in OVERRIDDEN_POLICY_SERVER_ATTRS if not hasattr(PolicyServer, name)]
    assert not missing, (
        f"lerobot.async_inference.policy_server.PolicyServer no longer has {missing}. "
        f"docker/lerobot-policy/server.py overrides or calls every one of "
        f"{list(OVERRIDDEN_POLICY_SERVER_ATTRS)}; a rename silently un-hooks the override "
        f"instead of raising, and the relative-action NotImplementedError reappears at the "
        f"robot as a zero-byte action reply."
    )


def test_remote_policy_config_field_names_are_unchanged():
    "The handshake payload's field names, asserted as an EXACT set."
    assert set(RemotePolicyConfig.__dataclass_fields__) == {
        "policy_type",
        "pretrained_name_or_path",
        "lerobot_features",
        "actions_per_chunk",
        "device",
        "rename_map",
    }, (
        f"RemotePolicyConfig's fields are now "
        f"{sorted(RemotePolicyConfig.__dataclass_fields__)}. policy/lerobot/session.py builds "
        f"this payload and docker/lerobot-policy/server.py unpacks it; re-diff both against "
        f"helpers.py:266-273 before changing this pin."
    )


def test_timed_observation_carries_must_go():
    '``must_go=True`` on every observation is what makes a single-shot inference work.'
    assert "must_go" in TimedObservation.__dataclass_fields__, (
        f"TimedObservation no longer has a must_go field (fields: "
        f"{sorted(TimedObservation.__dataclass_fields__)}). policy/lerobot/session.py passes "
        f"must_go=True on EVERY observation because a single-shot infer() would otherwise "
        f"sit in the server's queue below the chunk threshold and never be processed."
    )
    assert TimedObservation.__dataclass_fields__["must_go"].default is False, (
        "TimedObservation.must_go no longer DEFAULTS to False. session.py passes True "
        "explicitly, so a changed default is not itself a break — but it means upstream's "
        "queue-admission semantics moved, and the reason the explicit True exists must be "
        "re-derived rather than assumed."
    )
    assert callable(getattr(TimedAction, "get_action", None)), (
        "TimedAction.get_action is gone. policy/lerobot/backend.py reads every decoded "
        "timestep through it before the (16,6) -> named-joint reindex."
    )


def test_timed_action_field_order_and_transfer_begin_are_unchanged():
    "The two surfaces plan 06-04's gRPC mock reads, pinned at its request."
    assert list(TimedAction.__dataclass_fields__) == ["timestamp", "timestep", "action"], (
        f"TimedAction's field order is now {list(TimedAction.__dataclass_fields__)}. "
        f"scripts/mock_grpc_policy_server.py constructs it by keyword against the "
        f"(timestamp, timestep, action) order it inherits from TimedData; re-check that "
        f"mock's construction site before changing this pin."
    )
    assert hasattr(services_pb2, "TRANSFER_BEGIN"), (
        "services_pb2.TRANSFER_BEGIN is gone. The mock's chunk reassembly reads it to know "
        "when a new client-streamed observation starts; without it the mock would "
        "concatenate two observations into one payload."
    )


def test_async_inference_service_has_the_four_expected_methods():
    'The four-method wire, asserted on BOTH the descriptor and the generated stub.'
    service = services_pb2.DESCRIPTOR.services_by_name.get("AsyncInference")
    assert service is not None, (
        f"services_pb2 no longer declares an AsyncInference service (services: "
        f"{sorted(services_pb2.DESCRIPTOR.services_by_name)})."
    )
    assert set(service.methods_by_name) == set(ASYNC_INFERENCE_METHODS), (
        f"the AsyncInference service's methods are now {sorted(service.methods_by_name)}, "
        f"not {sorted(ASYNC_INFERENCE_METHODS)}. Every one of them is bound by "
        f"policy/lerobot/session.py, docker/lerobot-policy/server.py and "
        f"scripts/mock_grpc_policy_server.py."
    )

    # A lazy channel: grpc does not connect until an RPC is issued, so this
    # constructs the stub without any socket activity.
    channel = grpc.insecure_channel("127.0.0.1:1")
    try:
        stub = services_pb2_grpc.AsyncInferenceStub(channel)
        missing = [name for name in sorted(ASYNC_INFERENCE_METHODS) if not callable(getattr(stub, name, None))]
    finally:
        channel.close()
    assert not missing, (
        f"the generated AsyncInferenceStub does not expose {missing} as callables, so "
        f"policy/lerobot/session.py's calls would raise AttributeError at connect time."
    )

    assert callable(services_pb2_grpc.add_AsyncInferenceServicer_to_server), (
        "services_pb2_grpc.add_AsyncInferenceServicer_to_server is gone or not callable. "
        "docker/lerobot-policy/entrypoint.py and scripts/mock_grpc_policy_server.py both "
        "register their servicer through it; without it nothing can serve this wire."
    )


def test_groot_horizon_and_tag_helpers_still_resolve_sixteen():
    'The three CONFIG-level facts LRG-04 actually rests on.'
    from lerobot.policies.groot.configuration_groot import (
        infer_groot_n1_7_action_horizon,
        infer_groot_n1_7_embodiment_tag,
    )

    from policy_guard.groot_guard import EXPECTED_HORIZON, EXPECTED_TAG

    if not (REAL_CHECKPOINT / "config.json").is_file():
        pytest.fail(
            f"checkpoint not found at {REAL_CHECKPOINT} — the three facts LRG-04 rests on "
            f"cannot be asserted against the real config, and a skip here would be a silent "
            f"pass on the one requirement this pin exists to defend."
        )

    checkpoint = str(REAL_CHECKPOINT)
    assert infer_groot_n1_7_action_horizon(checkpoint, EXPECTED_TAG) == EXPECTED_HORIZON, (
        f"infer_groot_n1_7_action_horizon({checkpoint!r}, {EXPECTED_TAG!r}) is no longer "
        f"{EXPECTED_HORIZON}. Both nearby 40s are traps: GrootConfig's own action_horizon "
        f"default is 40 and this checkpoint's config.json also says action_horizon: 40, "
        f"while the real value is the 16 delta_indices on the {EXPECTED_TAG!r} tag."
    )
    assert infer_groot_n1_7_embodiment_tag(checkpoint) is None, (
        f"infer_groot_n1_7_embodiment_tag({checkpoint!r}) now returns a tag instead of None. "
        f"That is a BEHAVIOUR CHANGE, not a fix to absorb silently: the container preflight "
        f"and server.py both pass the tag explicitly BECAUSE inference returned None here. "
        f"Re-read D-11 before relying on inference."
    )
    assert GrootConfig(base_model_path=checkpoint).embodiment_tag == EXPECTED_TAG, (
        f"GrootConfig.embodiment_tag no longer resolves to {EXPECTED_TAG!r} for this "
        f"checkpoint. This is the default that makes the load work at all — the other eight "
        f"tags in the checkpoint carry 40 delta_indices."
    )


def test_checkpoint_processor_assets_loader_is_the_pinned_private_surface():
    "``policy_guard/groot_guard.py`` CALLS this private symbol; pinned at 06-02's request."
    from lerobot.policies.groot.processor_groot import _load_n1_7_checkpoint_processor_assets

    assert callable(_load_n1_7_checkpoint_processor_assets), (
        "lerobot.policies.groot.processor_groot._load_n1_7_checkpoint_processor_assets is "
        "gone or not callable. policy_guard/groot_guard.py derives BOTH assets_present and "
        "stats_non_empty from it, so SAFE-01/1 and SAFE-01/4 lose their evidence."
    )
    parameters = inspect.signature(_load_n1_7_checkpoint_processor_assets).parameters
    assert list(parameters) == ["config"], (
        f"_load_n1_7_checkpoint_processor_assets' signature is now {list(parameters)}, not "
        f"['config']. policy_guard/groot_guard.py calls it with a single GrootConfig."
    )


def test_albumentations_transform_helper_is_importable():
    "Plan 06-05's PAR-05 geometry dump hook."
    from lerobot.policies.groot.processor_groot import (
        _transform_n1_7_image_for_vlm_albumentations,
    )

    assert callable(_transform_n1_7_image_for_vlm_albumentations), (
        "lerobot.policies.groot.processor_groot._transform_n1_7_image_for_vlm_albumentations "
        "is gone or not callable. It is the CPU/cv2 transform this checkpoint's "
        "use_albumentations=True forces, and plan 06-05's PAR-05 dump reads it directly."
    )


def test_make_pre_post_processors_is_importable():
    'The public builder BOTH the server and the container preflight call.'
    assert callable(make_pre_post_processors), (
        "lerobot.policies.make_pre_post_processors is gone or not callable. It is the one "
        "public call that rebuilds the pre/post pipelines against corrected features; "
        "without it the server cannot serve 6-wide actions and the preflight's check 5 "
        "cannot force the lazy processor build."
    )
