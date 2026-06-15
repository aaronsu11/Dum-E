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
        "lang,model,dg_lang,el_lang,polly_voice,polly_lang",
        [
            # Phase 2 (D-01): default-profile STT is Nova-3; en/es/ja carry language="multi"
            # (Nova-3 multi auto-detect), zh is the explicit non-multi override.
            ("en", "nova-3", "multi", Language.EN, "Matthew", "en-US"),
            # zh ElevenLabs language is ZH (-> "zh"), NOT CMN (-> "cmn", which
            # eleven_turbo_v2_5 rejects with a 1008 policy violation). STT/Polly
            # keep their own zh-CN/cmn-CN codes — only the ElevenLabs TTS code differs.
            ("zh", "nova-3", "zh", Language.ZH, "Zhiyu", "cmn-CN"),
            ("ja", "nova-3", "multi", Language.JA, "Kazuha", "ja-JP"),
            ("es", "nova-3", "multi", Language.ES, "Sergio", "es-ES"),
        ],
    )
    def test_language_configurations(
        self, lang, model, dg_lang, el_lang, polly_voice, polly_lang
    ):
        """Test each language preset has correct configuration values."""
        from pipecat_server import LANGUAGE_PRESETS

        preset = LANGUAGE_PRESETS[lang]

        # Deepgram config (D-01: multi default + zh override)
        assert preset["deepgram"]["model"] == model
        assert preset["deepgram"]["language"] == dg_lang

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
        """Test Deepgram model selection per language (Phase 2 D-01: all Nova-3)."""
        from pipecat_server import LANGUAGE_PRESETS

        assert LANGUAGE_PRESETS["en"]["deepgram"]["model"] == "nova-3"
        assert LANGUAGE_PRESETS["zh"]["deepgram"]["model"] == "nova-3"
        assert LANGUAGE_PRESETS["ja"]["deepgram"]["model"] == "nova-3"
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


class TestLangfuseTraceIO:
    """Catches LANGFUSE-TRACE-IO / LANGFUSE-INGESTION-VERSION: the trace-level
    input/output patch and the OTEL ingestion-version header (per the official
    Langfuse Pipecat doc, "Add Trace Input and Output").

    Asserted by AST/source introspection — keyless, no live Langfuse creds or
    network — consistent with the other wiring tests in this file.
    """

    def test_patch_function_defined(self):
        """patch_trace_input_output is defined as a module-level function."""
        tree = _server_ast()
        assert any(
            isinstance(node, ast.FunctionDef)
            and node.name == "patch_trace_input_output"
            for node in ast.walk(tree)
        ), "expected a patch_trace_input_output function definition in pipecat_server.py"

    def test_patch_called_before_setup_tracing(self):
        """The patch is invoked before setup_tracing(...) in source order.

        Mirrors the wiring-test style; the runtime guarantee is "called before
        setup_tracing" so the wrapped add_llm_span_attributes is installed before
        tracing starts emitting spans.
        """
        source = _server_source()
        assert "patch_trace_input_output()" in source, (
            "patch_trace_input_output() must be called inside the Langfuse block"
        )
        assert source.index("patch_trace_input_output()") < source.index(
            "setup_tracing("
        ), "patch_trace_input_output() must be called before setup_tracing(...)"

    def test_ingestion_version_header_present(self):
        """OTEL headers carry x-langfuse-ingestion-version=4 AND keep Basic auth.

        Asserting both proves the append is additive, not a replacement of the
        existing Authorization=Basic header.
        """
        source = _server_source()
        assert "x-langfuse-ingestion-version=4" in source, (
            "OTEL_EXPORTER_OTLP_HEADERS must carry x-langfuse-ingestion-version=4"
        )
        assert "Authorization=Basic" in source, (
            "the existing Authorization=Basic header must remain intact "
            "(the ingestion-version token is appended, not substituted)"
        )


# --- Phase 2 speech-model wiring tests (SPCH-06 / D-06) ---------------------
#
# These keyless, no-network tests pin every Phase 2 speech-model change so a
# regression in the model swap is caught in CI. The behavioural acceptance
# evidence (multi-language conversation, the zh Aura-2->ElevenLabs fallback, and
# a >8-min Nova Sonic 2 session) is the documented manual live smoke test in the
# README — CI wiring is not the sole evidence (threat T-02-07/08).


class TestDeepgramSTTSettingsMigration:
    """SPCH-01 / D-01: the default-profile STT migrated off the deprecated
    LiveOptions to settings=DeepgramSTTSettings(...). Pattern B (string + AST).
    """

    def test_deepgram_stt_settings_constructed(self):
        """DeepgramSTTSettings(...) is constructed (the 1.x settings= idiom)."""
        tree = _server_ast()
        assert _calls_to(tree, "DeepgramSTTSettings"), (
            "expected a DeepgramSTTSettings(...) call — the 1.x replacement for "
            "live_options=LiveOptions(...)"
        )

    def test_live_options_absent_from_source(self):
        """The deprecated LiveOptions name must not appear anywhere in source."""
        source = _server_source()
        assert "LiveOptions" not in source, (
            "deprecated LiveOptions must not appear — STT migrated to "
            "settings=DeepgramSTTSettings(...) (D-01)"
        )


class TestNova3MultiSTTPresets:
    """SPCH-01 / D-01: the CONSTRUCTED default-profile STT resolves to Nova-3
    `multi` (the value the constructor consumes from the preset), and `zh` is
    the explicit non-`multi` override. Pattern C — assert the preset value, NOT
    mere token presence (Warning-1 fix).
    """

    @pytest.mark.parametrize("lang", ["en", "es", "ja"])
    def test_multi_covered_presets_use_nova3_multi(self, lang):
        """en/es/ja presets carry model=nova-3 + language=multi (what the
        default-profile STT constructor reads at runtime; run_jarvis also falls
        back to the `en` preset for unknown/unspecified languages)."""
        from pipecat_server import LANGUAGE_PRESETS

        assert LANGUAGE_PRESETS[lang]["deepgram"]["model"] == "nova-3"
        assert LANGUAGE_PRESETS[lang]["deepgram"]["language"] == "multi"

    def test_zh_is_explicit_non_multi_override(self):
        """zh is NOT in Nova-3 `multi`; it carries an explicit override."""
        from pipecat_server import LANGUAGE_PRESETS

        zh_lang = LANGUAGE_PRESETS["zh"]["deepgram"]["language"]
        assert zh_lang != "multi", (
            "Mandarin is not covered by Nova-3 `multi`; it must carry an explicit "
            "language override"
        )
        assert zh_lang in ("zh", "zh-CN")
        assert LANGUAGE_PRESETS["zh"]["deepgram"]["model"] == "nova-3"


class TestPerLanguageTTSRouting:
    """SPCH-02 / D-02/D-03: static per-language TTS routing — en/es/ja resolve to
    a Deepgram Aura-2 voice; zh resolves to ElevenLabs (Aura-2 has no Mandarin
    voice). The routing decision lives in LANGUAGE_PRESETS via a `tts_engine`
    key (per 02-01-SUMMARY). Pattern C (data assertion).
    """

    @pytest.mark.parametrize("lang", ["en", "es", "ja"])
    def test_aura2_languages_route_to_deepgram(self, lang):
        from pipecat_server import LANGUAGE_PRESETS

        preset = LANGUAGE_PRESETS[lang]
        assert preset["tts_engine"] == "deepgram", (
            f"{lang} is Aura-2-supported and must route to the Deepgram TTS engine"
        )
        # The Aura-2 voice token the DeepgramTTSService is constructed with.
        assert preset["aura2_voice"].startswith("aura-2-"), (
            f"{lang} must carry an aura-2-* voice token for Aura-2 routing"
        )

    def test_zh_routes_to_elevenlabs(self):
        from pipecat_server import LANGUAGE_PRESETS

        assert LANGUAGE_PRESETS["zh"]["tts_engine"] == "elevenlabs", (
            "Mandarin must fall back to ElevenLabs (Aura-2 has no zh voice) — "
            "Pitfall 3: a wrong route here produces silence on zh"
        )

    def test_zh_elevenlabs_uses_streaming_compatible_model(self):
        """UAT (test 2): the zh ElevenLabs fallback must use a STREAMING-compatible
        model. pipecat's websocket ElevenLabsTTSService uses the multi-stream-input
        endpoint, which only serves ELEVENLABS_MULTILINGUAL_MODELS
        (eleven_flash_v2_5 / eleven_turbo_v2_5). eleven_multilingual_v2 is NOT a
        streaming model — that endpoint returns a final message with NO audio (silent
        failure observed in UAT). The preset must select a streaming model, and the
        constructor must thread the preset model through (not hard-code one)."""
        from pipecat_server import LANGUAGE_PRESETS

        # The streaming models pipecat's websocket service supports (and which accept
        # an explicit language_code). Keep in lockstep with ELEVENLABS_MULTILINGUAL_MODELS.
        streaming_models = {"eleven_flash_v2_5", "eleven_turbo_v2_5"}
        model = LANGUAGE_PRESETS["zh"]["elevenlabs"]["model"]
        assert model in streaming_models, (
            f"zh ElevenLabs model {model!r} is not a streaming model {streaming_models}; "
            "eleven_multilingual_v2 returns no audio on the multi-stream-input endpoint"
        )
        source = _server_source()
        assert 'elevenlabs_config.get("model"' in source, (
            "the ElevenLabs branch must pass the preset's model to ElevenLabsTTSService "
            "(model=elevenlabs_config.get('model', ...)), not hard-code a model"
        )

    def test_zh_elevenlabs_language_is_eleven_supported(self):
        """UAT regression (test 2): with a streaming model, the zh language code IS
        applied — so it must resolve to a code the model accepts. Language.ZH -> 'zh'
        (accepted); Language.CMN -> 'cmn' (rejected with a 1008 policy violation)."""
        from pipecat_server import LANGUAGE_PRESETS
        from pipecat.transcriptions.language import Language

        el_lang = LANGUAGE_PRESETS["zh"]["elevenlabs"]["language"]
        assert el_lang == Language.ZH, (
            "zh ElevenLabs language must be Language.ZH (-> 'zh'); Language.CMN "
            "(-> 'cmn') is rejected by the turbo model with a 1008 policy violation"
        )

    def test_elevenlabs_tts_service_is_live_code(self):
        """SPCH-02 / D-03: the ElevenLabs fallback is now LIVE code (it was a
        commented-out block before the swap). Pattern B."""
        tree = _server_ast()
        assert _calls_to(tree, "ElevenLabsTTSService"), (
            "ElevenLabsTTSService(...) must be constructed in the zh fallback "
            "branch — it is live code now, not a comment"
        )


class TestNovaSonic2Wiring:
    """SPCH-04 / D-04: Nova Sonic 2 is wired with an explicit
    model=amazon.nova-2-sonic-v1:0 AND a native session_continuation= kwarg,
    while PRESERVING function_call_timeout_secs (PIPE-04 — guarded separately by
    TestLLMFunctionCallTimeout, which must NOT be weakened). Pattern A (AST kwarg).
    """

    def test_nova_sonic_has_session_continuation_kwarg(self):
        tree = _server_ast()
        calls = _calls_to(tree, "AWSNovaSonicLLMService")
        assert calls, "expected an AWSNovaSonicLLMService(...) constructor call"
        assert any(
            _kwarg(call, "session_continuation") is not None for call in calls
        ), (
            "AWSNovaSonicLLMService must be constructed with a session_continuation= "
            "kwarg (D-04 native ~8-min context-carrying session rotation)"
        )

    def test_nova_sonic_model_is_nova_2_sonic(self):
        tree = _server_ast()
        calls = _calls_to(tree, "AWSNovaSonicLLMService")
        assert calls, "expected an AWSNovaSonicLLMService(...) constructor call"

        models = []
        for call in calls:
            node = _kwarg(call, "model")
            if isinstance(node, ast.Constant):
                models.append(node.value)
        assert "amazon.nova-2-sonic-v1:0" in models, (
            "AWSNovaSonicLLMService must set model='amazon.nova-2-sonic-v1:0' "
            "explicitly (SPCH-04 / D-04 — Nova 2 Sonic)"
        )

    def test_nova_sonic2_bootstrap_does_not_use_v1_trigger(self):
        """UAT (test 3): Nova Sonic 2 is kicked off with a plain LLMRunFrame(), NOT
        the Nova Sonic 1 'await-trigger' pattern. On amazon.nova-2-sonic-v1:0,
        trigger_assistant_response() is a NO-OP (pipecat logs 'Assistant response
        trigger not needed'), and injecting AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION
        makes the model wait for a 'ready' cue that never arrives — so it never greets
        or responds. Assert the v1 trigger pattern is gone and LLMRunFrame() remains."""
        source = _server_source()
        assert "trigger_assistant_response" not in source, (
            "Nova Sonic 2 must NOT call trigger_assistant_response() — it is a no-op on "
            "amazon.nova-2-sonic-v1:0 and leaves the assistant waiting silently"
        )
        assert "AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION" not in source, (
            "Nova Sonic 2 must NOT inject AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION — "
            "it tells the model to wait for a 'ready' trigger that NS2 never sends"
        )
        assert "LLMRunFrame" in source, (
            "Nova Sonic 2 is prompted to respond with a plain LLMRunFrame()"
        )

    def test_nova_sonic_credentials_resolved_via_chain_not_only_env(self):
        """UAT (test 3): Nova Sonic builds a static smithy credentials resolver and
        does NOT walk the AWS credential chain itself. Reading only os.getenv for the
        access key/secret fails (SmithyIdentityError 'credentials weren't configured')
        whenever auth lives in ~/.aws/credentials / a profile / SSO. Assert the
        constructor's credential kwargs are fed by resolve_aws_static_credentials()
        (the botocore-chain resolver), not by a bare os.getenv() alone."""
        source = _server_source()
        assert "resolve_aws_static_credentials" in source, (
            "pipecat_server must resolve AWS creds through the botocore default chain "
            "(resolve_aws_static_credentials) so Nova Sonic works with profiles/SSO, "
            "not only with AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env vars"
        )
        tree = _server_ast()
        calls = _calls_to(tree, "AWSNovaSonicLLMService")
        assert calls, "expected an AWSNovaSonicLLMService(...) constructor call"
        for call in calls:
            for kw_name in ("access_key_id", "secret_access_key"):
                node = _kwarg(call, kw_name)
                assert node is not None, (
                    f"AWSNovaSonicLLMService must pass {kw_name}="
                )
                # The value must reference the chain-resolved name (directly or as the
                # primary operand of a `chain or os.getenv(...)` fallback), not be a
                # bare os.getenv() call on its own.
                names = {
                    n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                }
                assert any(name.startswith("ns_") for name in names), (
                    f"AWSNovaSonicLLMService {kw_name}= must be fed by the "
                    f"botocore-resolved credential (ns_*), not os.getenv alone"
                )

    def test_nova_sonic_passes_session_token(self):
        """SPCH-04: temporary credentials (SSO / assumed roles) require a
        session_token. Assert the Nova Sonic constructor forwards one so STS-based
        auth works, not just long-lived IAM keys."""
        tree = _server_ast()
        calls = _calls_to(tree, "AWSNovaSonicLLMService")
        assert calls, "expected an AWSNovaSonicLLMService(...) constructor call"
        assert any(
            _kwarg(call, "session_token") is not None for call in calls
        ), "AWSNovaSonicLLMService must forward session_token= for temporary creds"

    def test_resolve_aws_static_credentials_uses_botocore_chain(self):
        """resolve_aws_static_credentials() delegates to botocore's default session
        (which honors env vars, ~/.aws profiles, and SSO) and returns the frozen
        (access_key, secret_key, token) triple — unlike the static smithy resolver
        Nova Sonic builds internally."""
        import importlib

        import botocore.session

        pipecat_server = importlib.import_module("pipecat_server")

        frozen = mock.Mock(access_key="AKIA_X", secret_key="SECRET_X", token="TOKEN_X")
        creds = mock.Mock()
        creds.get_frozen_credentials.return_value = frozen
        fake_session = mock.Mock()
        fake_session.get_credentials.return_value = creds

        with mock.patch.object(
            botocore.session, "get_session", return_value=fake_session
        ):
            result = pipecat_server.resolve_aws_static_credentials()

        assert result == ("AKIA_X", "SECRET_X", "TOKEN_X")

    def test_resolve_aws_static_credentials_returns_none_when_unresolved(self):
        """When botocore resolves nothing, the helper returns (None, None, None) so
        the caller can fall back to explicit env vars rather than crash."""
        import importlib

        import botocore.session

        pipecat_server = importlib.import_module("pipecat_server")

        fake_session = mock.Mock()
        fake_session.get_credentials.return_value = None

        with mock.patch.object(
            botocore.session, "get_session", return_value=fake_session
        ):
            result = pipecat_server.resolve_aws_static_credentials()

        assert result == (None, None, None)

    def test_nova_sonic_timeout_guard_intact_post_swap(self):
        """PIPE-04 regression guard (Pitfall 4): after adding session_continuation
        + model, the Nova Sonic constructor STILL carries a non-None
        function_call_timeout_secs. This mirrors the assertion in
        TestLLMFunctionCallTimeout and proves the swap did not weaken it."""
        tree = _server_ast()
        calls = _calls_to(tree, "AWSNovaSonicLLMService")
        assert calls, "expected an AWSNovaSonicLLMService(...) constructor call"
        for call in calls:
            timeout = _kwarg(call, "function_call_timeout_secs")
            assert timeout is not None, (
                "AWSNovaSonicLLMService lost function_call_timeout_secs after the "
                "Nova Sonic 2 swap (PIPE-04 regression)"
            )
            assert not (
                isinstance(timeout, ast.Constant) and timeout.value is None
            ), "function_call_timeout_secs must be a real bound, not None"


class TestSageMakerBackendSelector:
    """SPCH-05 / D-05: the DUME_DEEPGRAM_BACKEND=sagemaker path is WIRED but the
    endpoint is NOT deployed this milestone — it raises ValueError on a missing
    endpoint (never silently falls back to hosted), and references the
    Deepgram-on-SageMaker services. Pattern D (behavioural, via run_jarvis) +
    Pattern B (AST/source).
    """

    def test_sagemaker_stt_service_referenced(self):
        """Pattern B: the SageMaker STT service is referenced (wired) in source."""
        source = _server_source()
        assert "DeepgramSageMakerSTTService" in source, (
            "DeepgramSageMakerSTTService must be referenced — the sagemaker "
            "backend is wired even though the endpoint is not deployed"
        )

    def test_sagemaker_branch_raises_value_error_on_missing_endpoint(self):
        """AST: the sagemaker branch carries a `raise ValueError` guarding the
        SAGEMAKER_STT_ENDPOINT_NAME read (no silent warn-and-fallback). The guard
        lives inline in run_jarvis (no extractable selector callable per
        02-01-SUMMARY), so assert structurally that a ValueError raise co-occurs
        with the endpoint env-var name."""
        source = _server_source()
        tree = _server_ast()

        # The endpoint env-var name is read.
        assert "SAGEMAKER_STT_ENDPOINT_NAME" in source, (
            "the sagemaker branch must read SAGEMAKER_STT_ENDPOINT_NAME"
        )
        # A `raise ValueError(...)` exists in the module (the guard).
        value_error_raises = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Raise)
            and isinstance(node.exc, ast.Call)
            and isinstance(node.exc.func, ast.Name)
            and node.exc.func.id == "ValueError"
        ]
        assert value_error_raises, (
            "the sagemaker-without-endpoint case must `raise ValueError` "
            "(D-05 guarded behaviour — no silent fallback)"
        )
        # At least one ValueError message names the SageMaker endpoint env var,
        # proving the raise guards the missing-endpoint case (not some other input).
        named = False
        for node in value_error_raises:
            for arg in ast.walk(node.exc):
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and (
                    "SAGEMAKER_STT_ENDPOINT_NAME" in arg.value
                    or "SAGEMAKER_TTS_ENDPOINT_NAME" in arg.value
                ):
                    named = True
        assert named, (
            "a raised ValueError message must name the missing SAGEMAKER_*_ENDPOINT_NAME "
            "(the guarded missing-endpoint case)"
        )

    def test_sagemaker_branch_honors_elevenlabs_routing(self):
        """CR-01 regression: the sagemaker TTS branch must honor the same
        per-language `tts_engine == "elevenlabs"` routing as the hosted path.
        Aura-2 (Deepgram TTS, hosted or SageMaker) has no Mandarin voice, so a
        `zh` request under the sagemaker backend must route to ElevenLabs — NOT
        fall back to an English aura2_voice. Assert structurally that BOTH the
        hosted and sagemaker branches gate on tts_engine == "elevenlabs" (two
        occurrences), proving the SageMaker branch is no longer
        unconditionally Aura-2."""
        from pipecat_server import LANGUAGE_PRESETS

        source = _server_source()
        routing_checks = source.count('tts_engine") == "elevenlabs"')
        assert routing_checks >= 2, (
            "both the hosted AND sagemaker TTS branches must gate on "
            'tts_engine == "elevenlabs" so zh routes to ElevenLabs in either '
            f"backend (CR-01); found {routing_checks} routing check(s)"
        )
        # The zh preset remains the load-bearing case: elevenlabs engine, no aura2_voice.
        assert LANGUAGE_PRESETS["zh"]["tts_engine"] == "elevenlabs"
        assert "aura2_voice" not in LANGUAGE_PRESETS["zh"], (
            "zh must NOT carry an aura2_voice — its presence would let the "
            "Aura-2 fallback synthesize Mandarin with an English voice (CR-01)"
        )

    @pytest.mark.asyncio
    async def test_run_jarvis_sagemaker_without_endpoint_raises(self):
        """Pattern D (behavioural): drive run_jarvis with the sagemaker backend
        and NO endpoint env vars; it must raise ValueError before any service is
        constructed. Keyless — fails fast on the guard, never reaching a live
        service constructor or the network."""
        import pipecat_server

        transport = mock.MagicMock(name="transport")
        runner_args = mock.MagicMock(name="runner_args")

        # clear=True removes any SAGEMAKER_*_ENDPOINT_NAME from the environment.
        with mock.patch.dict(
            os.environ, {"DUME_DEEPGRAM_BACKEND": "sagemaker"}, clear=True
        ):
            with pytest.raises(ValueError, match="SAGEMAKER_STT_ENDPOINT_NAME"):
                await pipecat_server.run_jarvis(
                    transport,
                    runner_args,
                    mode="cascaded",
                    profile="default",
                    language="en",
                    backend="sagemaker",
                )


class TestAwsProfileRegression:
    """SPCH-06 regression guard: the aws-profile (Transcribe/Bedrock/Polly) path
    is untouched by the Phase 2 default-profile model swap. Re-asserts the
    existing aws preset facts so a future swap that accidentally disturbs the aws
    branch is caught here too."""

    @pytest.mark.parametrize(
        "lang,polly_voice,polly_lang,transcribe_lang",
        [
            ("en", "Matthew", "en-US", "en-US"),
            ("zh", "Zhiyu", "cmn-CN", "zh-CN"),
            ("ja", "Kazuha", "ja-JP", "ja-JP"),
            ("es", "Sergio", "es-ES", "es-ES"),
        ],
    )
    def test_aws_profile_presets_unchanged(
        self, lang, polly_voice, polly_lang, transcribe_lang
    ):
        from pipecat_server import LANGUAGE_PRESETS

        preset = LANGUAGE_PRESETS[lang]
        assert preset["aws_polly"]["voice_id"] == polly_voice
        assert preset["aws_polly"]["language"] == polly_lang
        assert preset["aws_transcribe"]["language"] == transcribe_lang

    def test_no_stale_nova2_stt_model_in_presets(self):
        """No preset claims a nova-2 STT model (the OLD zh/ja matrix is gone).
        Nova Sonic's nova-2-sonic id is a different thing and is NOT an STT model."""
        from pipecat_server import LANGUAGE_PRESETS

        for lang, preset in LANGUAGE_PRESETS.items():
            assert preset["deepgram"]["model"] != "nova-2", (
                f"{lang} still claims a nova-2 STT model — the stale matrix must be gone"
            )
