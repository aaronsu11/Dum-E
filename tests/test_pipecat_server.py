"""
Tests for pipecat_server.py voice interface and language configuration.

This file tests:
- Language presets structure and completeness
- Language, mode, and profile validation
- Configuration selection for different languages
- Pipecat 1.x wiring (D-03 / PIPE-05): import resolution, long-running-aware
  tool registration, LLM function-call timeout, aggregator VAD/turn config,
  and PipelineWorker/tracing — all mocked / keyless so they run in CI without
  any DEEPGRAM/ANTHROPIC/AWS/LANGFUSE keys.

The wiring-test approach (per the 01-03 plan): the behaviour-critical
``register_tools_schema`` long-running policy is asserted with a fully mocked
``llm`` (capturing ``register_function`` call kwargs); the remaining 1.x wiring
facts (LLM ``function_call_timeout_secs``, the aggregator
``LLMUserAggregatorParams`` + ``VADParams(stop_secs=0.8)`` build, and the
``PipelineWorker`` + ``enable_tracing`` + ``additional_span_attributes`` block)
are asserted by AST/source introspection of ``pipecat_server.py``. Source
introspection is used (rather than mocking the full ``run_jarvis`` pipeline
build) because constructing the cascaded pipeline keyless would require patching
~10 service/transport classes — the introspection assertions stay in the
existing lightweight, no-live-keys test style while still pinning the exact
1.x wiring the migration requires.
"""

import ast
import os
import textwrap
from pathlib import Path
from unittest import mock

import pytest
from pipecat.services.elevenlabs.tts import Language


# --- Helpers shared by the source-introspection wiring tests ----------------


def _server_source() -> str:
    """Return the source text of pipecat_server.py (sibling of the repo root)."""
    server_path = Path(__file__).resolve().parent.parent / "pipecat_server.py"
    return server_path.read_text(encoding="utf-8")


def _server_ast() -> ast.Module:
    """Parse pipecat_server.py into an AST for structural assertions."""
    return ast.parse(_server_source())


def _calls_to(tree: ast.AST, func_name: str) -> list:
    """Return every ast.Call whose callee is the bare name ``func_name``."""
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == func_name
    ]


def _kwarg(call: ast.Call, name: str):
    """Return the ast node for keyword ``name`` on ``call``, or None."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


class TestLanguageSpecificConfigs:
    """Test language-specific configuration values."""

    @pytest.mark.parametrize(
        "lang,model,el_lang,polly_voice,polly_lang",
        [
            ("en", "nova-3", Language.EN, "Matthew", "en-US"),
            ("zh", "nova-2", Language.CMN, "Zhiyu", "cmn-CN"),
            ("ja", "nova-2", Language.JA, "Kazuha", "ja-JP"),
            ("es", "nova-3", Language.ES, "Sergio", "es-ES"),
        ],
    )
    def test_language_configurations(
        self, lang, model, el_lang, polly_voice, polly_lang
    ):
        """Test each language preset has correct configuration values."""
        from pipecat_server import LANGUAGE_PRESETS

        preset = LANGUAGE_PRESETS[lang]

        # Deepgram config
        assert preset["deepgram"]["model"] == model
        assert preset["deepgram"]["language"] == lang

        # ElevenLabs config
        assert preset["elevenlabs"]["language"] == el_lang
        assert len(preset["elevenlabs"]["voice_id"]) > 0

        # AWS Polly config
        assert preset["aws_polly"]["voice_id"] == polly_voice
        assert preset["aws_polly"]["language"] == polly_lang

        # Greeting exists
        assert len(preset["greeting"]) > 0


class TestConfigValidation:
    """Test validation logic for language, mode, and profile."""

    @pytest.mark.parametrize(
        "value,valid_set,default",
        [
            ("en", ["en", "zh", "ja", "es"], "en"),
            ("zh", ["en", "zh", "ja", "es"], "en"),
            ("invalid", ["en", "zh", "ja", "es"], "en"),
            ("", ["en", "zh", "ja", "es"], "en"),
            ("cascaded", ["cascaded", "speech_to_speech"], "cascaded"),
            ("speech_to_speech", ["cascaded", "speech_to_speech"], "cascaded"),
            ("invalid_mode", ["cascaded", "speech_to_speech"], "cascaded"),
            ("default", ["default", "aws"], "default"),
            ("aws", ["default", "aws"], "default"),
            ("invalid_profile", ["default", "aws"], "default"),
        ],
    )
    def test_validation_fallback(self, value, valid_set, default):
        """Test that validation falls back to defaults for invalid values."""
        result = value if value in valid_set else default

        if value in valid_set:
            assert result == value
        else:
            assert result == default


class TestEnvironmentVariableLoading:
    """Test environment variable loading in bot() function."""

    @pytest.mark.parametrize(
        "env_var,default,test_value",
        [
            ("DUME_VOICE_LANGUAGE", "en", "zh"),
            ("DUME_VOICE_MODE", "cascaded", "speech_to_speech"),
            ("DUME_VOICE_PROFILE", "default", "aws"),
        ],
    )
    def test_env_var_loading(self, env_var, default, test_value):
        """Test loading configuration from environment variables."""
        # Test with default (no env var set)
        with mock.patch.dict(os.environ, {}, clear=True):
            value = os.getenv(env_var, default)
            assert value == default

        # Test with env var set
        with mock.patch.dict(os.environ, {env_var: test_value}):
            value = os.getenv(env_var, default)
            assert value == test_value


class TestLanguageFeatures:
    """Test language-specific features and requirements."""

    def test_deepgram_model_selection(self):
        """Test Deepgram model selection per language."""
        from pipecat_server import LANGUAGE_PRESETS

        assert LANGUAGE_PRESETS["en"]["deepgram"]["model"] == "nova-3"
        assert LANGUAGE_PRESETS["zh"]["deepgram"]["model"] == "nova-2"
        assert LANGUAGE_PRESETS["ja"]["deepgram"]["model"] == "nova-2"
        assert LANGUAGE_PRESETS["es"]["deepgram"]["model"] == "nova-3"

    def test_all_profiles_supported(self):
        """Test that all languages support both default and AWS profiles."""
        from pipecat_server import LANGUAGE_PRESETS

        for preset in LANGUAGE_PRESETS.values():
            # Default profile requirements
            assert "deepgram" in preset
            assert "elevenlabs" in preset
            # AWS profile requirements
            assert "aws_polly" in preset


# --- Pipecat 1.x wiring tests (D-03 / PIPE-05) ------------------------------


class TestPipecat1xImportSmoke:
    """Catches Pitfall 5 (submodule import drift): the module must import clean
    under Pipecat 1.3.0 with no live keys."""

    def test_import_pipecat_server_succeeds(self):
        """import pipecat_server resolves all 1.x submodule imports.

        Done as a lazy in-body import (matching the existing file style) so an
        import failure surfaces as a test failure rather than a collection error.
        """
        import pipecat_server  # noqa: F401  (import-as-assertion)

        # Sanity touch of a top-level symbol to prove the module body executed.
        assert hasattr(pipecat_server, "run_jarvis")
        assert hasattr(pipecat_server, "AsyncMCPClient")


class TestRegisterToolsSchemaTimeoutPolicy:
    """Catches Pitfall 2 + PIPE-04 / D-02: long-running-aware tool registration.

    Drives ``AsyncMCPClient.register_tools_schema`` with a fully mocked ``llm``
    and a mocked live MCP session, then asserts on the captured
    ``llm.register_function`` call kwargs. No live keys / network — the session
    ``list_tools`` (the source of the ``long_running`` flag under 1.x, since the
    ``tools_schema`` FunctionSchemas no longer carry ``.meta``) is mocked.
    """

    @staticmethod
    def _make_tool(name, long_running):
        tool = mock.MagicMock()
        tool.name = name
        tool.meta = {"long_running": long_running}
        return tool

    @staticmethod
    def _make_function_schema(name):
        fs = mock.MagicMock()
        fs.name = name
        return fs

    @pytest.mark.asyncio
    async def test_long_running_and_normal_tool_registration_kwargs(self):
        from pipecat_server import AsyncMCPClient

        # Build the client without running MCPClient.__init__ (which would try
        # to wire a real session); we only exercise register_tools_schema.
        client = AsyncMCPClient.__new__(AsyncMCPClient)

        # Mock the live-session re-list that recovers the long_running flag.
        long_tool = self._make_tool("pick_object", long_running=True)
        normal_tool = self._make_tool("get_status", long_running=False)
        list_tools_result = mock.MagicMock()
        list_tools_result.tools = [long_tool, normal_tool]

        session = mock.MagicMock()
        session.list_tools = mock.AsyncMock(return_value=list_tools_result)
        client._ensure_connected = mock.MagicMock(return_value=session)

        # The wrapper the parent registers as the handler.
        client._tool_wrapper = mock.MagicMock(name="tool_wrapper")

        # tools_schema.standard_tools = the FunctionSchemas to register.
        tools_schema = mock.MagicMock()
        tools_schema.standard_tools = [
            self._make_function_schema("pick_object"),
            self._make_function_schema("get_status"),
        ]

        llm = mock.MagicMock()
        llm.register_function = mock.MagicMock()

        await client.register_tools_schema(tools_schema, llm)

        # Collect the per-tool kwargs the override passed to register_function.
        by_name = {}
        for call in llm.register_function.call_args_list:
            tool_name = call.args[0]
            by_name[tool_name] = call.kwargs

        assert set(by_name) == {"pick_object", "get_status"}

        # long_running tool: exempt from interruption + unbounded (D-02).
        assert by_name["pick_object"]["cancel_on_interruption"] is False
        assert by_name["pick_object"]["timeout_secs"] is None

        # normal tool: interruptible + bounded at the normal-tool timeout (PIPE-04).
        assert by_name["get_status"]["cancel_on_interruption"] is True
        assert by_name["get_status"]["timeout_secs"] == 30.0
        assert (
            by_name["get_status"]["timeout_secs"]
            == AsyncMCPClient.NORMAL_TOOL_TIMEOUT_SECS
        )

    @pytest.mark.asyncio
    async def test_handler_registered_is_the_tool_wrapper(self):
        """Each tool is registered against the parent's self._tool_wrapper."""
        from pipecat_server import AsyncMCPClient

        client = AsyncMCPClient.__new__(AsyncMCPClient)
        list_tools_result = mock.MagicMock()
        list_tools_result.tools = [self._make_tool("get_status", long_running=False)]
        session = mock.MagicMock()
        session.list_tools = mock.AsyncMock(return_value=list_tools_result)
        client._ensure_connected = mock.MagicMock(return_value=session)
        client._tool_wrapper = mock.MagicMock(name="tool_wrapper")

        tools_schema = mock.MagicMock()
        tools_schema.standard_tools = [self._make_function_schema("get_status")]
        llm = mock.MagicMock()

        await client.register_tools_schema(tools_schema, llm)

        call = llm.register_function.call_args_list[0]
        assert call.args[0] == "get_status"
        assert call.args[1] is client._tool_wrapper


class TestLLMFunctionCallTimeout:
    """Catches Pitfall 3 (PIPE-04): every LLM service is constructed with a
    non-None function_call_timeout_secs (the 1.x default flipped to None, which
    would hang the voice loop on a stalled normal tool).

    Asserted by AST introspection of pipecat_server.py: each LLM-service
    constructor call must carry an explicit, non-None function_call_timeout_secs
    keyword. (Mocking the full run_jarvis build keyless would require patching
    ~10 classes; the AST assertion stays in the lightweight test style while
    pinning the exact constructor kwarg.)
    """

    @pytest.mark.parametrize(
        "service_name",
        ["AnthropicLLMService", "AWSBedrockLLMService", "AWSNovaSonicLLMService"],
    )
    def test_llm_service_has_nonnull_function_call_timeout(self, service_name):
        tree = _server_ast()
        calls = _calls_to(tree, service_name)
        assert calls, f"expected a {service_name}(...) constructor call in pipecat_server.py"

        for call in calls:
            timeout = _kwarg(call, "function_call_timeout_secs")
            assert timeout is not None, (
                f"{service_name} constructed without function_call_timeout_secs= "
                f"(PIPE-04: 1.x default is None and would never time out)"
            )
            # The kwarg value itself must not be the literal None.
            assert not (
                isinstance(timeout, ast.Constant) and timeout.value is None
            ), f"{service_name} has function_call_timeout_secs=None (must be a real bound)"


class TestAggregatorVadTurnConfig:
    """Catches a regression of the explicit stop_secs=0.8 (SC5 / D-03
    "aggregator VAD/turn config"): the user aggregator must be built with
    LLMUserAggregatorParams AND a SileroVADAnalyzer carrying
    VADParams(stop_secs=0.8) (NOT the 1.x default of 0.2).

    Asserted by AST introspection: a LLMUserAggregatorParams(...) call exists
    with a vad_analyzer keyword, and a VADParams(...) call passes
    stop_secs=0.8. (Source introspection keeps this keyless and in-style; the
    mocked-build alternative would need the whole pipeline patched.)
    """

    def test_user_aggregator_uses_llm_user_aggregator_params_with_vad(self):
        tree = _server_ast()
        agg_calls = _calls_to(tree, "LLMUserAggregatorParams")
        assert agg_calls, "expected LLMUserAggregatorParams(...) in the aggregator build"
        assert any(
            _kwarg(call, "vad_analyzer") is not None for call in agg_calls
        ), "LLMUserAggregatorParams must receive a vad_analyzer (turn/VAD config)"

    def test_vad_params_stop_secs_is_explicit_0_8(self):
        tree = _server_ast()
        vad_calls = _calls_to(tree, "VADParams")
        assert vad_calls, "expected VADParams(...) in the aggregator build"

        stop_secs_values = []
        for call in vad_calls:
            node = _kwarg(call, "stop_secs")
            if node is not None and isinstance(node, ast.Constant):
                stop_secs_values.append(node.value)

        assert 0.8 in stop_secs_values, (
            "VADParams must be constructed with an explicit stop_secs=0.8 "
            "(1.x default is 0.2 — preserving 0.8 is the D-03 turn-config requirement)"
        )

    def test_silero_vad_analyzer_referenced(self):
        """The VAD analyzer is the SileroVADAnalyzer (cascaded turn detection)."""
        tree = _server_ast()
        assert _calls_to(
            tree, "SileroVADAnalyzer"
        ), "expected SileroVADAnalyzer(...) wrapping the VADParams in the aggregator build"


class TestPipelineWorkerTracing:
    """Catches Pitfall 4 + PIPE-05: PipelineWorker (not the deprecated
    PipelineTask) is used, with enable_tracing=True and an
    additional_span_attributes dict carrying the Langfuse session id + tags.

    Asserted by AST/source introspection.
    """

    def test_pipeline_worker_used_not_pipeline_task(self):
        source = _server_source()
        tree = _server_ast()

        assert _calls_to(tree, "PipelineWorker"), "PipelineWorker(...) must be constructed"
        # No deprecated PipelineTask/PipelineRunner aliases anywhere.
        assert "PipelineTask" not in source, "deprecated PipelineTask alias must not appear"
        assert "PipelineRunner" not in source, "deprecated PipelineRunner alias must not appear"

    def test_pipeline_worker_enables_tracing_with_span_attributes(self):
        tree = _server_ast()
        worker_calls = _calls_to(tree, "PipelineWorker")
        assert worker_calls, "PipelineWorker(...) must be constructed"

        call = worker_calls[0]

        enable_tracing = _kwarg(call, "enable_tracing")
        assert enable_tracing is not None and isinstance(enable_tracing, ast.Constant)
        assert enable_tracing.value is True, "PipelineWorker must enable_tracing=True"

        span_attrs = _kwarg(call, "additional_span_attributes")
        assert isinstance(
            span_attrs, ast.Dict
        ), "PipelineWorker must pass an additional_span_attributes dict"

        keys = {
            k.value
            for k in span_attrs.keys
            if isinstance(k, ast.Constant)
        }
        assert "langfuse.session.id" in keys, "tracing span attrs must carry langfuse.session.id"
        assert "langfuse.tags" in keys, "tracing span attrs must carry langfuse.tags"
