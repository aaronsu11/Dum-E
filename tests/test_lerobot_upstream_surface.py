"""Keyless pins on the upstream LeRobot surface Dum-E DUPLICATES.

``docker/lerobot-policy/server.py`` takes plan 06-01 Task 2's STATED FALLBACK: it
reimplements ``PolicyServer.SendPolicyInstructions`` (~25 lines of
``lerobot/async_inference/policy_server.py:116-171``) and
``GrootPolicy._create_groot_model`` (~20 lines of
``lerobot/policies/groot/modeling_groot.py:88-118``) instead of calling
``super()``, because both values that MUST be correct before the weights
materialize — the input/output features and the storage dtype — are consumed
inside the call ``super()`` would have made. See that module's STATED FALLBACK and
SERVING PRECISION docstring blocks for the full reasoning.

Duplicated upstream lines are exactly the drift this project refuses to carry, so
this module pins them. **The pins are structural, not textual**: they compare the
SET of attributes upstream's version assigns, and the SET of config fields
upstream's version reads, against the sets Dum-E's copies assign and read. A
version bump that adds one assignment or one config field therefore turns this
suite RED with a message naming the missing name, rather than leaving a silently
stale copy that drops a setting on the floor.

Every test here is **keyless, GPU-free and never skips**, in the style of
``tests/test_units_verdict.py``. Nothing is loaded: the pins read the installed
wheel's source with ``inspect``/``ast`` and Dum-E's copies as text. The one
GPU-gated instrument in this phase is ``tests/test_lerobot_serving_live.py`` and
this module must never become a second one.

Plan 06-06 EXTENDED this module (it was scheduled to create it; the fallback made
it first). Do not narrow what is pinned here when adding to it. Plan 06-06's
additions widen the surface from "the two duplications" to **everything plan
06-01's overrides and its client session bind to**, plus the private symbol plan
06-02's SAFE-01 guard depends on:

* the eight ``PolicyServer`` attributes ``DumEGrootPolicyServer`` overrides or
  calls through — a rename un-hooks ``_predict_action_chunk`` with no import
  error, and blocker 1's relative-action ``NotImplementedError`` reappears at
  the robot;
* the exact ``RemotePolicyConfig`` field NAMES, ``TimedObservation.must_go`` and
  ``TimedAction``'s accessor + field order — the wire payloads
  ``policy/lerobot/session.py`` builds;
* the four gRPC method names, asserted on both the service descriptor and the
  generated stub;
* the three CONFIG-level facts LRG-04 actually rests on. Never the emitted chunk
  length: three independent truncations force 16 regardless of configuration
  (T=16, T=40 and T=50 all decode to ``(1, 16, 6)``), so a chunk-length
  assertion would pass against a misconfigured server;
* ``_load_n1_7_checkpoint_processor_assets``, which ``policy_guard/groot_guard.py``
  CALLS rather than reimplements — a deliberately pinned private surface;
* the two helpers later plans import (``make_pre_post_processors``, and the
  albumentations transform plan 06-05's PAR-05 dump hook reads).

**Checkpoint-dependent assertions call ``pytest.fail`` naming the absent
directory — never a skip**, for the reason in the paragraph above: a skipped
upstream-surface test is a silent pass on the assumption that the pinned symbols
still exist. This module declares no skip call, no conditional-skip marker and
no module-level marker of any kind — asserted by grep in plan 06-06's
verification, which is why none of those three names is spelled out here.
"""

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
REAL_CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "GR00T-N1.7-3B-SO101"


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
    """The copy must set the SAME server state upstream sets — no more, no fewer.

    This is the drift pin that matters. Upstream's ``SendPolicyInstructions``
    installs the server's whole per-handshake state; a bump that adds one more
    assignment would leave Dum-E's copy silently missing it, and the symptom would
    be a wrong inference rather than an error.
    """
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
    """The copy must forward the SAME config fields upstream forwards.

    A bump that adds a field to upstream's ``model_kwargs`` would otherwise leave
    Dum-E's copy quietly dropping it — the model would load with a stale knob and
    nothing would say so.
    """
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
    """``dtype`` and ``load_bf16`` reach transformers/GR00TN17Config only via ``**kwargs``.

    ``GR00TN17.from_pretrained`` pops the knobs it handles itself and forwards the
    REST to ``super().from_pretrained``. That forwarding is the whole delivery
    mechanism for Dum-E's two injected kwargs: ``dtype`` is popped by transformers
    as the model dtype, and ``load_bf16`` falls through onto ``GR00TN17Config``.
    """
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
    """The reason config injection alone cannot force bf16.

    ``load_bf16`` lives on ``GR00TN17Config`` (built from the CHECKPOINT's
    ``config.json``, which declares ``False``), not on LeRobot's ``GrootConfig``.
    ``GrootPolicy.from_pretrained``'s kwargs passthrough is guarded by
    ``if hasattr(config, key)`` (``modeling_groot.py:266-268``), so passing
    ``load_bf16=True`` there would be dropped WITHOUT a warning.

    If this test goes red because upstream ADDED the field, that is good news: the
    ``_create_groot_model`` duplication can then be deleted in favour of plain
    config injection. Delete it deliberately — do not just widen this pin.
    """
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
    """Recorded so nobody 'fixes' the precision by flipping this instead.

    ``use_bf16`` already defaulted True during the fp32 measurement of 11.47 GiB.
    It gates ``torch.autocast`` at ``modeling_groot.py:458,498`` — compute, not
    storage — so it can never make the weights fit.
    """
    assert GrootConfig.__dataclass_fields__["use_bf16"].default is True
    source = inspect.getsource(GrootPolicy)
    autocast_lines = [line for line in source.splitlines() if "use_bf16" in line]
    assert autocast_lines, "use_bf16 is no longer read in modeling_groot's GrootPolicy"
    assert all("autocast" in line for line in autocast_lines), (
        f"use_bf16 is now read somewhere other than a torch.autocast call: {autocast_lines}. "
        f"It may no longer be compute-only; re-derive the storage-precision reasoning."
    )


def test_model_dtype_is_dead_for_groot_under_lerobot():
    """The checkpoint declares ``model_dtype: 'bfloat16'`` and LeRobot never reads it.

    Pinned because it is load-bearing for the honesty of the bf16 write-up: the
    checkpoint's own declared intent is bf16, LeRobot ignores that declaration, and
    the fp32 it uses instead comes from the separate ``dtype: 'float32'`` entry. If
    upstream ever starts READING ``model_dtype``, the injection may become
    unnecessary — and this pin is what says so.
    """
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
    """The pinned-cache mechanism, and the Dockerfile's offline build assertion.

    ``_build_n1_7_processor`` is PRIVATE and is named in two Dum-E artifacts: the
    Dockerfile asserts an offline build against it, and the pin comment in
    ``scripts/build_lerobot_policy_image.sh`` cites it. Two properties matter:

    1. It accepts **no** ``revision`` argument, which is why the image's HF cache
       contents plus ``HF_HUB_OFFLINE=1`` are the ONLY available revision
       enforcement (D-08) — there is no call-site knob to pass a SHA to.
    2. It resolves the backbone by NAME, so it asks the hub cache for revision
       ``main``. That is what makes the hand-written ``refs/main`` in the
       Dockerfile load-bearing rather than decorative.
    """
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
    """The copy PREVENTS the 224x224 placeholder rather than repairing it.

    ``from_pretrained`` injects a single ``observation.images.camera`` at
    ``(3, 224, 224)`` only when it had to build the config itself. Dum-E passes a
    config with correct ``input_features``, so that branch never runs — which is
    what closes blockers 2 and 3 up front. If the guard changes, the prevention
    claim in server.py's docstring stops being true.
    """
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
    """A rename un-hooks an override with NO import error and no crash.

    This is the pin that stops the worst failure mode in the phase. If upstream
    renames ``_predict_action_chunk``, ``DumEGrootPolicyServer``'s override stops
    being called, upstream's per-timestep postprocess loop
    (``policy_server.py:368-379``) runs instead, and
    ``GrootN17ActionDecodeStep`` raises ``NotImplementedError`` for that call
    shape on this native-relative-action checkpoint — inside ``GetActions``,
    whose blanket ``except Exception`` returns ``Empty()`` from a method declared
    to return ``Actions``. The client sees a SUCCESSFUL RPC carrying zero bytes.
    Nothing else in the suite would go red.
    """
    missing = [name for name in OVERRIDDEN_POLICY_SERVER_ATTRS if not hasattr(PolicyServer, name)]
    assert not missing, (
        f"lerobot.async_inference.policy_server.PolicyServer no longer has {missing}. "
        f"docker/lerobot-policy/server.py overrides or calls every one of "
        f"{list(OVERRIDDEN_POLICY_SERVER_ATTRS)}; a rename silently un-hooks the override "
        f"instead of raising, and the relative-action NotImplementedError reappears at the "
        f"robot as a zero-byte action reply."
    )


def test_remote_policy_config_field_names_are_unchanged():
    """The handshake payload's field names, asserted as an EXACT set.

    Stronger than the subset check above on purpose: ``session.py`` CONSTRUCTS
    this dataclass by keyword and ``SendPolicyInstructions`` UNPACKS it, so an
    ADDED required field breaks the client just as surely as a removed one — and
    a subset check cannot see an addition.
    """
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
    """``must_go=True`` on every observation is what makes a single-shot inference work.

    Two independent upstream paths gate on it: the server's queue only admits an
    observation when it is ``must_go`` or the queue is below threshold, and
    ``_enqueue_observation`` short-circuits on it. If the field is renamed, the
    ``TimedObservation(...)`` construction in ``session.py`` raises — but if it
    silently becomes a NON-default keyword or changes default, a single-shot
    ``infer()`` hangs instead. ``TimedAction.get_action`` is the accessor the
    backend's reindex reads every decoded step through.
    """
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
    """The two surfaces plan 06-04's gRPC mock reads, pinned at its request.

    ``TimedAction`` inherits ``TimedData``, so its field order is
    ``(timestamp, timestep, action)``. ``scripts/mock_grpc_policy_server.py``
    constructs it by KEYWORD for exactly this reason — positional construction
    against a reordered base would produce a mock that passes while the real
    server's replies fail — and it reassembles chunked observations by looking
    for ``TRANSFER_BEGIN``. An upstream reshape of either would make the mock
    silently disagree with the server it exists to stand in for.
    """
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
    """The four-method wire, asserted on BOTH the descriptor and the generated stub.

    The descriptor is the declaration; the stub is what ``session.py`` actually
    calls. Asserting only the descriptor would miss a codegen change, and
    asserting only the stub would miss a service renamed out from under it. A
    FIFTH method appearing is also a failure here, deliberately: plan 06-04
    rejected a sentinel ``"kill"`` RPC on its mock precisely because a fifth
    method would stop the mock from being this contract.
    """
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
    """The three CONFIG-level facts LRG-04 actually rests on.

    Deliberately NOT an assertion about the emitted chunk length: three
    independent truncations force 16 regardless of whether the configuration is
    right (``T=16``, ``T=40`` and ``T=50`` all decode to ``(1, 16, 6)``), so a
    chunk-length check would pass against a server loaded at the wrong horizon.
    What is load-bearing is the ``embodiment_tag``:

    1. horizon inference with the tag passed EXPLICITLY returns 16 — this
       checkpoint carries nine tags and only ``new_embodiment``'s
       ``delta_indices`` are ``[0..15]``;
    2. tag INFERENCE returns ``None`` for this checkpoint, which is why check 3
       of the container preflight passes the tag rather than relying on
       inference;
    3. ``GrootConfig.embodiment_tag`` DEFAULTS to ``new_embodiment``
       (``configuration_groot.py:288``) — the default that saves the load, and
       which ``server.py`` nonetheless sets explicitly per D-11.

    ``EXPECTED_HORIZON``/``EXPECTED_TAG`` are IMPORTED from
    ``policy_guard.groot_guard`` rather than restated as literals here, so the
    SAFE-01 guard, the preflight and this pin cannot drift to three different
    numbers.
    """
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
    """``policy_guard/groot_guard.py`` CALLS this private symbol; pinned at 06-02's request.

    ``assets_present`` is ``_load_n1_7_checkpoint_processor_assets(config) is not
    None`` and ``stats_non_empty`` is ``bool(assets.raw_stats)`` — the SAME asset
    table the decoder will use, rather than a second independently-parsed copy of
    ``statistics.json``. Depending on a private symbol is the deliberate choice
    there (a rename breaks loudly at import instead of producing a second source
    of truth); this pin is what makes "breaks loudly" true for the SUITE and not
    only for the container.
    """
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
    """Plan 06-05's PAR-05 geometry dump hook.

    ``use_albumentations=True`` in this checkpoint's ``processor_config.json``
    forces the cv2/CPU transform path (``processor_groot.py:2076-2078``), so this
    private helper — not the torchvision path — is what actually produces the
    preprocessed image geometry PAR-05 measures.
    """
    from lerobot.policies.groot.processor_groot import (
        _transform_n1_7_image_for_vlm_albumentations,
    )

    assert callable(_transform_n1_7_image_for_vlm_albumentations), (
        "lerobot.policies.groot.processor_groot._transform_n1_7_image_for_vlm_albumentations "
        "is gone or not callable. It is the CPU/cv2 transform this checkpoint's "
        "use_albumentations=True forces, and plan 06-05's PAR-05 dump reads it directly."
    )


def test_make_pre_post_processors_is_importable():
    """The public builder BOTH the server and the container preflight call.

    ``docker/lerobot-policy/server.py`` rebuilds the pipelines through it after
    the feature fixup (which is what makes ``env_action_dim`` 6 instead of 132),
    and ``entrypoint.py``'s check 5 calls it CONFIG-ONLY to force the lazy
    backbone processor build at startup.
    """
    assert callable(make_pre_post_processors), (
        "lerobot.policies.make_pre_post_processors is gone or not callable. It is the one "
        "public call that rebuilds the pre/post pipelines against corrected features; "
        "without it the server cannot serve 6-wide actions and the preflight's check 5 "
        "cannot force the lazy processor build."
    )
