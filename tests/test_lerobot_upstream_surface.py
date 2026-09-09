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

Plan 06-06 EXTENDS this module (it was scheduled to create it; the fallback made
it first). Do not narrow what is pinned here when adding to it.
"""

import ast
import inspect
import textwrap
from pathlib import Path

from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.groot.groot_n1_7 import GR00T_N1_7_DEFAULTS, GR00TN17, _tie_unused_qwen_lm_head
from lerobot.policies.groot.modeling_groot import GrootPolicy

#: Dum-E's copy. Read as TEXT, never imported: it is container code that pulls in
#: the ``groot`` extra's transformers stack, which the client venv deliberately
#: does not carry (``tests/test_container_contract.py``'s dependency-isolation
#: guard is what keeps it out).
SERVER_PY = Path(__file__).resolve().parent.parent / "docker" / "lerobot-policy" / "server.py"


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
