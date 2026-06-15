import base64
import json
import os
import time
from dotenv import load_dotenv
from typing import Callable, Literal, List


print("🚀 Starting Pipecat bot...")

from loguru import logger
from mcp import ClientSession, ListToolsResult
from mcp.client.session_group import StreamableHttpParameters
from mcp.shared.session import ProgressFnT
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.pipeline.task import PipelineParams
from pipecat.workers.runner import WorkerRunner
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.aws.llm import AWSBedrockLLMService
from pipecat.services.aws.stt import AWSTranscribeSTTService
from pipecat.services.aws.tts import AWSPollyTTSService
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.services.aws.nova_sonic.session_continuation import SessionContinuationParams
from pipecat.services.deepgram.stt import DeepgramSTTService, DeepgramSTTSettings
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, Language
from pipecat.services.llm_service import LLMService
from pipecat.services.mcp_service import MCPClient
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.utils.tracing.setup import setup_tracing

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

# Setup clean logging for voice assistant (but preserve loguru for pipecat)
# setup_robot_logging(log_level="INFO", include_timestamps=True)

# Language presets for STT and TTS services in cascaded mode
LANGUAGE_PRESETS = {
    "en": {
        "deepgram": {
            "model": "nova-3",
            "language": "multi",  # D-01: Nova-3 multi auto-detect (covers en/es/fr/de/hi/ru/pt/ja/it/nl)
        },
        # D-02 static TTS routing: en is Aura-2-supported -> Deepgram TTS.
        "tts_engine": "deepgram",
        "aura2_voice": "aura-2-thalia-en",
        "aws_transcribe": {
            "language": "en-US",
        },
        "elevenlabs": {
            "voice_id": "iP95p4xoKVk53GoZ742B",  # Default English voice
            "language": Language.EN,
        },
        "aws_polly": {
            "voice_id": "Matthew",
            "language": "en-US",
            "engine": "generative",
        },
        "greeting": "At your service, sir.",
    },
    "zh": {
        "deepgram": {
            "model": "nova-3",
            "language": "zh",  # D-01 override: Mandarin is NOT in Nova-3 `multi`; use explicit zh
        },
        # D-02/D-03 static TTS routing: Aura-2 does NOT support Mandarin -> ElevenLabs fallback.
        "tts_engine": "elevenlabs",
        "aws_transcribe": {
            "language": "zh-CN",
        },
        "elevenlabs": {
            # Sarah — a PREMADE voice (free-tier usable). eleven_turbo_v2_5 is
            # multilingual, so with language_code "zh" it speaks Mandarin. Library/
            # professional voices (e.g. the dedicated Mandarin "Stacy"
            # hkfHEbBvdQFNX4uWHqRF) require a PAID ElevenLabs plan — on free tier the
            # API returns 402 payment_required and the websocket path fails silently
            # (no audio). Use a premade voice unless the deployment has a paid plan.
            "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah (premade, multilingual via turbo)
            # eleven_turbo_v2_5 is required for the streaming websocket path pipecat
            # uses (the multi-stream-input endpoint only serves the streaming models
            # eleven_flash_v2_5 / eleven_turbo_v2_5). eleven_multilingual_v2 is NOT a
            # streaming model — the endpoint returns a final message with no audio.
            # turbo_v2_5 is also in ELEVENLABS_MULTILINGUAL_MODELS, so the zh language
            # code below IS applied (Language.ZH -> "zh", which the model accepts).
            "model": "eleven_turbo_v2_5",
            "language": Language.ZH,
        },
        "aws_polly": {
            "voice_id": "Zhiyu",
            "language": "cmn-CN",
            "engine": "neural",
        },
        "greeting": "你好呀，主人",  # "Hello, master" in Mandarin
    },
    "ja": {
        "deepgram": {
            "model": "nova-3",
            "language": "multi",  # D-01: moved off nova-2 — Nova-3 multi now covers Japanese
        },
        # D-02 static TTS routing: Japanese is Aura-2-supported -> Deepgram TTS.
        "tts_engine": "deepgram",
        "aura2_voice": "aura-2-fujin-ja",  # [ASSUMED A1] confirm exact ja Aura-2 token at smoke test
        "aws_transcribe": {
            "language": "ja-JP",
        },
        "elevenlabs": {
            "voice_id": "8kgj5469z1URcH4MB2G4",  # Japanese voice
            "language": Language.JA,
        },
        "aws_polly": {
            "voice_id": "Kazuha",
            "language": "ja-JP",
            "engine": "neural",
        },
        "greeting": "準備完了。",  # "Preparations complete" in Japanese
    },
    "es": {
        "deepgram": {
            "model": "nova-3",
            "language": "multi",  # D-01: Nova-3 multi covers Spanish
        },
        # D-02 static TTS routing: Spanish is Aura-2-supported -> Deepgram TTS.
        "tts_engine": "deepgram",
        "aura2_voice": "aura-2-celeste-es",  # [ASSUMED A1] confirm exact es Aura-2 token at smoke test
        "aws_transcribe": {
            "language": "es-ES",
        },
        "elevenlabs": {
            "voice_id": "htFfPSZGJwjBv1CL0aMD",  # Spanish voice
            "language": Language.ES,
        },
        "aws_polly": {
            "voice_id": "Sergio",
            "language": "es-ES",
            "engine": "generative",
        },
        "greeting": "Listo para ayudar",  # "Ready to help" in Spanish
    },
}

# We store functions so objects (e.g. SileroVADAnalyzer) don't get
# instantiated. The function will be called when the desired transport gets
# selected.
transport_params = {
    "twilio": lambda: FastAPIWebsocketParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
}


def _as_tool_result_object(response: str):
    """Coerce a tool's concatenated text response into a JSON-object result.

    SPCH-04: function-call results flow through two serialization layers — the
    universal aggregator does ``json.dumps(frame.result)`` and stores the string,
    then Nova Sonic forwards it verbatim. Nova Sonic's Bedrock side REJECTS a
    tool result whose top-level JSON is not an object:
    "Unsupported JSON type in Tool Result. Please provide the Tool Result as a JSON
    object." Returning a bare string therefore double-encodes to a JSON *string
    literal* and fails. Returning a dict makes the aggregator emit a JSON *object*,
    which Nova Sonic accepts and the cascaded Bedrock/Anthropic adapters handle
    identically (Bedrock already json.loads object-looking content; Anthropic
    re-dumps the dict to the same text).

    - Empty response -> a benign object (never the old "could not call" sentinel,
      which some tools legitimately return empty for).
    - Response that is already a JSON object -> that object (pass-through).
    - Any other text (incl. JSON arrays/scalars) -> wrapped as {"result": <text>}.
    """
    if not response:
        return {"result": "Sorry, could not call the mcp tool"}
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, ValueError):
        return {"result": response}
    if isinstance(parsed, dict):
        return parsed
    # JSON arrays / scalars are valid JSON but not objects — wrap them so the
    # top-level tool result is always an object.
    return {"result": parsed}


class AsyncMCPClient(MCPClient):
    """Override pipecat's MCPClient to work better with asynchronous/long-running tasks."""

    progress_callback: ProgressFnT | None = None

    def set_progress_callback(self, progress_callback: ProgressFnT):
        self.progress_callback = progress_callback

    async def _call_tool(
        self,
        session: ClientSession,
        function_name: str,
        arguments: dict,
        result_callback,
    ):
        """Override the _call_tool method to use the progress callback."""
        logger.debug(f"Calling mcp tool '{function_name}'")
        try:
            results = await session.call_tool(
                function_name,
                arguments=arguments,
                progress_callback=self.progress_callback,
            )
        except Exception as e:
            error_msg = f"Error calling mcp tool {function_name}: {str(e)}"
            logger.error(error_msg)
            await result_callback({"error": error_msg})
            return

        response = ""
        if results:
            if hasattr(results, "content") and results.content:
                for i, content in enumerate(results.content):
                    if hasattr(content, "text") and content.text:
                        logger.debug(f"Tool response chunk {i}: {content.text}")
                        response += content.text
                    else:
                        # logger.debug(f"Non-text result content: '{content}'")
                        pass
                logger.info(f"Tool '{function_name}' completed successfully")
                logger.debug(f"Final response: {response}")
            else:
                logger.error(f"Error getting content from {function_name} results.")

        await result_callback(_as_tool_result_object(response))

    # Normal (non-long-running) tool timeout. The Pipecat 1.x default for
    # function_call_timeout_secs flipped to None, so without an explicit bound a
    # stalled normal tool would hang the voice loop forever (PIPE-04 / D-02).
    NORMAL_TOOL_TIMEOUT_SECS: float = 30.0

    async def register_tools_schema(self, tools_schema, llm):
        """Register MCP tools, deriving cancel_on_interruption + timeout_secs from long_running metadata.

        Replaces the dead per-transport list-tools override (which never ran under 0.0.104 or 1.x).
        The 1.x parent `register_tools_schema(self, tools_schema, llm)` receives a
        ToolsSchema of FunctionSchema objects that no longer carry the FastMCP `.meta`
        flag, so we re-list tools from the live session to recover the `long_running`
        flag, then register each tool with the parent's `self._tool_wrapper`.

        - long_running tools: cancel_on_interruption=False, timeout_secs=None (exempt —
          legitimate multi-minute robot tasks must not be killed; D-02).
        - normal tools: cancel_on_interruption=True, timeout_secs=NORMAL_TOOL_TIMEOUT_SECS
          (true hangs are bounded; PIPE-04).
        """
        # Recover the long_running flag per tool name from the live MCP session.
        long_running_by_name: dict[str, bool] = {}
        try:
            session = self._ensure_connected()
            available_tools = await session.list_tools()
            for tool in available_tools.tools:
                meta = getattr(tool, "meta", None) or {}
                long_running_by_name[tool.name] = bool(meta.get("long_running", False))
        except Exception as e:
            logger.error(f"Failed to read long_running metadata from MCP session: {str(e)}")

        for function_schema in tools_schema.standard_tools:
            tool_name = function_schema.name
            is_long_running = long_running_by_name.get(tool_name, False)
            # If the tool is long running, we don't want to interrupt it on new voice input.
            cancel_on_interruption = not is_long_running
            timeout_secs = None if is_long_running else self.NORMAL_TOOL_TIMEOUT_SECS
            logger.debug(
                f"Registering function handler for '{tool_name}' with "
                f"cancel_on_interruption={cancel_on_interruption}, timeout_secs={timeout_secs}"
            )
            try:
                llm.register_function(
                    tool_name,
                    self._tool_wrapper,
                    cancel_on_interruption=cancel_on_interruption,
                    timeout_secs=timeout_secs,
                )
            except Exception as e:
                logger.error(f"Failed to register tool '{tool_name}': {str(e)}")
                logger.exception("Full exception details:")
                continue


def resolve_aws_static_credentials():
    """Resolve AWS credentials via the botocore default provider chain.

    SPCH-04: AWSNovaSonicLLMService builds a smithy StaticCredentialsResolver from
    the access_key_id/secret_access_key/session_token it is handed and does NOT walk
    the AWS credential chain itself. Reading only os.getenv("AWS_ACCESS_KEY_ID")/
    ("AWS_SECRET_ACCESS_KEY") therefore fails (SmithyIdentityError "credentials weren't
    configured") whenever auth lives in ~/.aws/credentials, a named AWS_PROFILE, or SSO
    rather than in those two env vars — even though the boto3-based cascaded services
    (Transcribe/Polly/Bedrock) resolve fine. Resolve once here through botocore (which
    honors env vars, shared credentials files, profiles, and SSO) and hand the frozen
    static creds to Nova Sonic.

    Returns:
        Tuple of (access_key_id, secret_access_key, session_token). session_token is
        None for long-lived IAM-user keys. Returns (None, None, None) if the chain
        resolves nothing, so the caller can fall back to explicit env vars.
    """
    try:
        import botocore.session

        creds = botocore.session.get_session().get_credentials()
        if creds is None:
            return None, None, None
        frozen = creds.get_frozen_credentials()
        return frozen.access_key, frozen.secret_key, frozen.token
    except Exception as e:
        logger.warning(f"botocore credential resolution failed: {e}")
        return None, None, None


def _sanitize_schema_for_nova_sonic(node):
    """Recursively coerce a JSON-Schema fragment into the restricted shape that
    Nova Sonic's Bedrock tool validator accepts.

    SPCH-04: Nova Sonic (amazon.nova-*-sonic) rejects tool inputSchemas that use
    constructs its bidirectional-streaming tool validator does not support —
    surfacing as "Invalid input request, please fix your input and try again."
    the instant the tool-bearing prompt-start event is sent — even though the
    boto3/Converse path (Claude) tolerates them. The MCP tool schemas emitted by
    FastMCP/Pydantic use exactly those constructs for optional params:
      - ``anyOf: [<T>, {"type": "null"}]`` (Optional[...] fields)
      - ``default`` keys
      - ``additionalProperties`` / ``title``
    Collapse ``anyOf``/``oneOf`` to the first non-null branch and strip the
    unsupported keys. The result remains a valid schema for every other model.
    """
    if isinstance(node, list):
        return [_sanitize_schema_for_nova_sonic(n) for n in node]
    if not isinstance(node, dict):
        return node

    node = dict(node)

    # Collapse anyOf/oneOf (commonly Optional[...] == [T, null]) to the first
    # non-null branch, merging any sibling keys (e.g. description) into it.
    for combinator in ("anyOf", "oneOf"):
        if combinator in node:
            branches = [
                b
                for b in node.pop(combinator)
                if not (isinstance(b, dict) and b.get("type") == "null")
            ]
            chosen = branches[0] if branches else {"type": "string"}
            merged = {**node, **chosen}
            return _sanitize_schema_for_nova_sonic(merged)

    # Strip keys the validator does not accept / does not need.
    for key in ("default", "additionalProperties", "title"):
        node.pop(key, None)

    # Recurse into nested schema positions.
    if isinstance(node.get("properties"), dict):
        node["properties"] = {
            k: _sanitize_schema_for_nova_sonic(v) for k, v in node["properties"].items()
        }
    if "items" in node:
        node["items"] = _sanitize_schema_for_nova_sonic(node["items"])

    return node


def sanitize_tools_for_nova_sonic(tools):
    """Return a ToolsSchema whose FunctionSchema properties are coerced to the
    restricted JSON-Schema shape Nova Sonic accepts (see
    _sanitize_schema_for_nova_sonic).

    Tool NAMES and required lists are preserved so handler dispatch is unaffected.
    Returns ``tools`` unchanged if it is falsy. Only the Nova Sonic speech-to-speech
    path needs this — the cascaded Claude/Converse path tolerates the raw MCP schemas.
    """
    if not tools:
        return tools
    sanitized = []
    for fn in tools.standard_tools:
        clean_props = {
            name: _sanitize_schema_for_nova_sonic(spec)
            for name, spec in fn.properties.items()
        }
        sanitized.append(
            FunctionSchema(
                name=fn.name,
                description=fn.description,
                properties=clean_props,
                required=fn.required,
            )
        )
    return ToolsSchema(standard_tools=sanitized)


def patch_trace_input_output():
    """Promote the first LLM input and latest LLM output to the Langfuse TRACE level.

    Per the official Langfuse Pipecat integration doc ("Add Trace Input and Output",
    https://langfuse.com/integrations/frameworks/pipecat). Without this patch, live
    traces show ``input:null, output:null`` at the TRACE level — the conversation
    transcript and LLM responses are only captured on child ``llm`` GENERATION spans.

    This wraps ``pipecat.utils.tracing.service_decorators.add_llm_span_attributes``
    so that:
    - On the FIRST call where the ``messages`` kwarg is truthy, it sets the span
      attribute ``langfuse.trace.input`` to that ``messages`` value.
    - ``span.set_attribute`` is wrapped so any ``"output"`` key also mirrors to
      ``langfuse.trace.output`` (last write wins).

    Note: installed Pipecat 1.3.0 passes ``messages`` as a JSON-serialized STRING
    kwarg (not a list). The patch is value-agnostic and sets whatever the kwarg
    holds, so no coercion or re-parsing is needed.

    Idempotent / value-agnostic: must be called before ``setup_tracing(...)``.
    """
    from pipecat.utils.tracing import service_decorators

    original = service_decorators.add_llm_span_attributes
    first_call = [True]

    def patched(span, *args, **kwargs):
        original(span, *args, **kwargs)

        # Mirror any "output" attribute write to the trace-level output.
        original_set_attribute = span.set_attribute

        def set_attribute(key, value):
            original_set_attribute(key, value)
            if key == "output":
                original_set_attribute("langfuse.trace.output", value)

        span.set_attribute = set_attribute

        # Promote the first truthy messages payload to the trace-level input.
        if first_call[0]:
            messages = kwargs.get("messages")
            if messages:
                span.set_attribute("langfuse.trace.input", messages)
                first_call[0] = False

    service_decorators.add_llm_span_attributes = patched


system_prompt = """You are a helpful robot assistant speaking out loud to users.

CRITICAL RULES FOR VOICE OUTPUT:
- Use the language of the user to respond
- Speak naturally as if in conversation - no special characters ever
- No markdown, asterisks, brackets, quotes, dashes, or number points
- No lists or bullet points - speak in flowing sentences
- Keep responses to 1-3 short sentences maximum

RESPONSE STYLE:
- Be direct and conversational
- Skip explanations of what you're doing - just give results
- Don't ask follow-up questions unless necessary
- Use natural speech patterns"""


async def run_jarvis(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    mode: Literal["cascaded", "speech_to_speech"] = "cascaded",
    profile: Literal["default", "aws"] = "default",
    voice_updates: bool = True,  # only supported in cascaded mode
    language: str = "en",
    backend: Literal["hosted", "sagemaker"] = "hosted",
):
    """
    Run JARVIS voice agent with the given transport, runner arguments, mode, and profile.

    Args:
        transport: The transport to use for the bot
        runner_args: The runner arguments to use for the bot
        mode: The mode to use for the bot. Voice status update is only supported in cascaded mode.
        profile: The profile to use for the bot
        language: Language code for voice interface (en, zh, ja, es)
        backend: Deepgram backend for the default profile (D-05). "hosted" (default) uses the
            hosted Deepgram STT/TTS; "sagemaker" uses the Deepgram-on-SageMaker services
            (wired but NOT deployed this milestone — raises ValueError if no endpoint is set).
    """

    logger.info(f"Starting bot")

    # Get language configuration from presets
    language_code = language
    if language_code not in LANGUAGE_PRESETS:
        logger.warning(
            f"Unsupported language code '{language_code}', falling back to 'en'"
        )
        language_code = "en"

    language_preset = LANGUAGE_PRESETS[language_code]
    logger.info(f"Using language: {language_code}")

    # Speech to speech
    if mode == "speech_to_speech":
        if profile == "aws":
            # SPCH-04: Nova Sonic uses a static smithy credentials resolver, so resolve
            # the full AWS chain (env vars, ~/.aws profiles, SSO) up front and pass the
            # frozen creds in. Fall back to explicit env vars if the chain resolves nothing.
            ns_access_key, ns_secret_key, ns_session_token = resolve_aws_static_credentials()
            llm = AWSNovaSonicLLMService(
                secret_access_key=ns_secret_key or os.getenv("AWS_SECRET_ACCESS_KEY"),
                access_key_id=ns_access_key or os.getenv("AWS_ACCESS_KEY_ID"),
                session_token=ns_session_token or os.getenv("AWS_SESSION_TOKEN"),
                region=os.getenv("AWS_REGION"),  # SPCH-04: must be us-east-1 / us-west-2 / ap-northeast-1 for Nova-2 Sonic
                model="amazon.nova-2-sonic-v1:0",  # SPCH-04 / D-04: Nova 2 Sonic (the default; explicit for testability)
                voice_id="matthew",  # Voices: matthew, tiffany, amy
                function_call_timeout_secs=30.0,  # PIPE-04: bound normal tool calls (long_running exempt per-tool)
                # D-04: native ~8-min context-carrying session rotation (no hand-rolled reconnect).
                session_continuation=SessionContinuationParams(),
            )
        else:
            raise NotImplementedError(
                "Currently only AWS Nova Sonic is supported for speech-to-speech mode"
            )
    else:  # Cascaded
        if profile == "aws":
            transcribe_config = language_preset["aws_transcribe"]
            stt = AWSTranscribeSTTService(
                api_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                # aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
                region=os.getenv("AWS_REGION"),
                language=transcribe_config["language"],
            )

            llm = AWSBedrockLLMService(
                api_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                # aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
                aws_region=os.getenv("AWS_REGION"),
                model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
                params=AWSBedrockLLMService.InputParams(
                    temperature=0.1,
                    max_tokens=500,
                ),
                function_call_timeout_secs=30.0,  # PIPE-04: bound normal tool calls (long_running exempt per-tool)
            )

            polly_config = language_preset["aws_polly"]
            tts = AWSPollyTTSService(
                api_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                # aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
                region=os.getenv("AWS_REGION"),
                voice_id=polly_config["voice_id"],
                params=AWSPollyTTSService.InputParams(
                    engine=polly_config["engine"],
                    language=polly_config["language"],
                    rate="1.3",
                ),
            )
        else:
            deepgram_config = language_preset["deepgram"]

            if backend == "sagemaker":
                # D-05 / SPCH-05: Deepgram-on-SageMaker is WIRED but NOT deployed this milestone.
                # Require explicit endpoint env vars and raise (never silently fall back to hosted).
                stt_endpoint = os.getenv("SAGEMAKER_STT_ENDPOINT_NAME")
                if not stt_endpoint:
                    raise ValueError(
                        "DUME_DEEPGRAM_BACKEND=sagemaker requires SAGEMAKER_STT_ENDPOINT_NAME; "
                        "the SageMaker endpoint is wired but NOT deployed this milestone"
                    )
                tts_endpoint = os.getenv("SAGEMAKER_TTS_ENDPOINT_NAME")
                if not tts_endpoint:
                    raise ValueError(
                        "DUME_DEEPGRAM_BACKEND=sagemaker requires SAGEMAKER_TTS_ENDPOINT_NAME; "
                        "the SageMaker endpoint is wired but NOT deployed this milestone"
                    )
                # Pitfall 1: lazy-import inside the branch so the default hosted path never
                # triggers the aws_sdk_sagemaker_runtime_http2 import.
                from pipecat.services.deepgram.sagemaker.stt import DeepgramSageMakerSTTService
                from pipecat.services.deepgram.sagemaker.tts import DeepgramSageMakerTTSService

                stt = DeepgramSageMakerSTTService(
                    endpoint_name=stt_endpoint,
                    region=os.getenv("AWS_REGION"),
                    settings=DeepgramSageMakerSTTService.Settings(
                        model=deepgram_config["model"],
                        language=deepgram_config["language"],
                    ),
                )
                # CR-01: honor the same per-language tts_engine routing as the hosted path.
                # Aura-2 (Deepgram TTS, hosted or SageMaker) has no Mandarin voice, so zh must
                # still route to ElevenLabs — otherwise the aura2_voice fallback would synthesize
                # Mandarin with an English voice. ElevenLabs is a hosted service independent of
                # the STT backend, so it is valid inside the SageMaker branch too.
                if language_preset.get("tts_engine") == "elevenlabs":
                    elevenlabs_config = language_preset["elevenlabs"]
                    tts = ElevenLabsTTSService(
                        api_key=os.getenv("ELEVENLABS_API_KEY"),
                        voice_id=elevenlabs_config["voice_id"],
                        model=elevenlabs_config.get("model", "eleven_turbo_v2_5"),
                        sample_rate=24000,
                        params=ElevenLabsTTSService.InputParams(
                            language=elevenlabs_config["language"]
                        ),
                    )
                else:
                    tts = DeepgramSageMakerTTSService(
                        endpoint_name=tts_endpoint,
                        region=os.getenv("AWS_REGION"),
                        settings=DeepgramSageMakerTTSService.Settings(
                            voice=language_preset.get("aura2_voice", "aura-2-thalia-en"),
                        ),
                    )
            else:
                # backend == "hosted" (default): hosted Deepgram STT (Nova-3 multi, D-01).
                stt = DeepgramSTTService(
                    api_key=os.getenv("DEEPGRAM_API_KEY"),
                    settings=DeepgramSTTSettings(
                        model=deepgram_config["model"],       # "nova-3"
                        language=deepgram_config["language"],  # "multi" default; "zh" override
                    ),
                )

                # D-02/D-03 static per-language TTS routing: Aura-2 for supported languages,
                # ElevenLabs fallback for the rest (Mandarin — Aura-2 has no zh voice).
                if language_preset.get("tts_engine") == "elevenlabs":
                    elevenlabs_config = language_preset["elevenlabs"]
                    tts = ElevenLabsTTSService(
                        api_key=os.getenv("ELEVENLABS_API_KEY"),
                        voice_id=elevenlabs_config["voice_id"],
                        model=elevenlabs_config.get("model", "eleven_turbo_v2_5"),
                        sample_rate=24000,
                        params=ElevenLabsTTSService.InputParams(
                            language=elevenlabs_config["language"]
                        ),
                    )
                else:
                    tts = DeepgramTTSService(
                        api_key=os.getenv("DEEPGRAM_API_KEY"),
                        voice=language_preset.get("aura2_voice", "aura-2-thalia-en"),
                    )

            llm = AnthropicLLMService(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model="claude-haiku-4-5",
                params=AnthropicLLMService.InputParams(temperature=0.1, max_tokens=500),
                function_call_timeout_secs=30.0,  # PIPE-04: bound normal tool calls (long_running exempt per-tool)
            )

    # Use enhanced MCPClient to avoid interrupting long-running tools and enable voice updates on progress.
    # Under Pipecat 1.x the MCP session must be started (async with / start()) before register_tools,
    # otherwise _ensure_connected() raises RuntimeError. Everything that depends on the live session
    # (tool registration, pipeline build, worker run) happens inside this session block.
    async with AsyncMCPClient(
        server_params=StreamableHttpParameters(url="http://localhost:8000/mcp")
    ) as mcp:
        # This registers the tools with the LLM using register_tools_schema which sets the
        # cancel_on_interruption flag and per-tool timeout based on the long_running metadata
        # defined in the FastMCP server.
        tools = await mcp.register_tools(llm)

        # @llm.event_handler("on_function_calls_started")
        # async def on_function_calls_started(service, function_calls):
        #     await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

        if profile == "aws":
            # Nova Sonic 2 (amazon.nova-2-sonic-v1:0) bootstrap. The OLD Nova Sonic 1
            # pattern appended an "await-trigger" instruction ("start speaking when you
            # hear 'ready'") and sent a synthetic "ready" audio cue. On NS2 that audio
            # cue is a NO-OP (pipecat logs "Assistant response trigger not needed"), so
            # the model would sit waiting for a 'ready' cue that never arrives and never
            # greet. NS2 is kicked off with a plain LLMRunFrame() instead (see
            # on_client_connected), so we must NOT inject the await-trigger instruction
            # here — keep only the greeting instruction.
            if mode == "speech_to_speech":
                # SPCH-04: Nova Sonic's tool validator rejects the raw MCP schemas
                # (anyOf/null, default, additionalProperties) with "Invalid input
                # request" at session setup. Coerce to its restricted shape — this
                # path only; the cascaded Claude path below tolerates them as-is.
                nova_sonic_tools = sanitize_tools_for_nova_sonic(tools)
                context = LLMContext(
                    messages=[
                        {
                            "role": "system",
                            "content": [
                                {
                                    "text": "\n".join(
                                        [
                                            system_prompt,
                                            "Greet the user by saying 'Hello there!'",
                                        ]
                                    )
                                }
                            ],
                        }
                    ],
                    tools=nova_sonic_tools,
                )
            else:
                context = LLMContext(
                    messages=[
                        {
                            "role": "system",
                            "content": [{"text": system_prompt}],
                        }
                    ],
                    tools=tools,
                )
        else:
            context = LLMContext(
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                ],
                tools=tools,
            )

        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8))
            ),
        )

        if mode == "cascaded":
            pipeline = Pipeline(
                [
                    transport.input(),  # Transport user input
                    stt,  # Speech-to-text
                    user_aggregator,  # User responses
                    llm,  # LLM
                    tts,  # Text-to-speech
                    transport.output(),  # Transport bot output
                    assistant_aggregator,  # Assistant spoken responses
                ]
            )
        elif mode == "speech_to_speech":
            pipeline = Pipeline(
                [
                    transport.input(),
                    user_aggregator,
                    llm,
                    transport.output(),
                    assistant_aggregator,
                ]
            )

        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            enable_tracing=True,  # Enables both turn and conversation tracing
            additional_span_attributes={
                "langfuse.session.id": time.strftime("%Y-%m-%d"),
                "langfuse.tags": ["pipecat-server"],
            },
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected")
            # Kick off the conversation.
            if mode == "speech_to_speech" and profile == "aws":
                # Nova Sonic 2: a plain LLMRunFrame() prompts the assistant to greet
                # (the context ends with the system greeting instruction). The old
                # Nova Sonic 1 "ready"-audio response cue is a NO-OP on NS2
                # (amazon.nova-2-sonic-v1:0) and is intentionally NOT called here —
                # calling it just logged a warning and did nothing.
                await worker.queue_frames([LLMRunFrame()])
            else:
                # Use language-specific greeting
                greeting = language_preset["greeting"]
                await worker.queue_frames([TTSSpeakFrame(greeting)])

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            logger.info(f"Client disconnected")
            await worker.cancel()

        async def progress_handler(
            progress: float, total: float | None, message: str | None
        ) -> None:
            """Handle voice updates on MCP progress notifications."""
            if voice_updates and mode == "cascaded" and message:
                await worker.queue_frame(TTSSpeakFrame(message))

        # Set the progress callback with reference to the worker for centralized queueing
        mcp.set_progress_callback(progress_handler)

        runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)

        # Run the worker inside the live MCP session so the session stays open for the
        # whole conversation (tool calls reuse the persistent session).
        await runner.run(worker)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport = await create_transport(runner_args, transport_params)

    # Get configuration from environment variables (for pipecat runner compatibility)
    language = os.getenv("DUME_VOICE_LANGUAGE", "en")
    mode = os.getenv("DUME_VOICE_MODE", "cascaded")
    profile = os.getenv("DUME_VOICE_PROFILE", "default")
    backend = os.getenv("DUME_DEEPGRAM_BACKEND", "hosted")

    # Validate mode
    if mode not in ["cascaded", "speech_to_speech"]:
        logger.warning(f"Invalid mode '{mode}', falling back to 'cascaded'")
        mode = "cascaded"

    # Validate profile
    if profile not in ["default", "aws"]:
        logger.warning(f"Invalid profile '{profile}', falling back to 'default'")
        profile = "default"

    # Validate Deepgram backend selector (D-05). An UNKNOWN value warns + falls back to
    # hosted; the explicit "sagemaker" value (with no endpoint) is guarded downstream by a
    # ValueError in run_jarvis — never a silent fallback.
    if backend not in ["hosted", "sagemaker"]:
        logger.warning(f"Invalid backend '{backend}', falling back to 'hosted'")
        backend = "hosted"

    # Configure Langfuse for OpenTelemetry tracing
    # See https://langfuse.com/integrations/frameworks/pipecat for more information
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        # Build Langfuse Auth header.
        LANGFUSE_AUTH = base64.b64encode(
            f"{os.environ.get('LANGFUSE_PUBLIC_KEY')}:{os.environ.get('LANGFUSE_SECRET_KEY')}".encode()
        ).decode()

        # Configure OpenTelemetry endpoint & headers
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = (
            os.environ.get("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
            + "/api/public/otel"
        )
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = (
            f"Authorization=Basic {LANGFUSE_AUTH},x-langfuse-ingestion-version=4"
        )

        # Configured automatically from .env
        exporter = OTLPSpanExporter()

        # Promote first LLM input + latest LLM output to the TRACE level, per the
        # official Langfuse Pipecat doc. Must run before setup_tracing(...).
        patch_trace_input_output()

        setup_tracing(
            service_name="pipecat-demo",
            exporter=exporter,
        )

    await run_jarvis(
        transport, runner_args, mode=mode, profile=profile, language=language, backend=backend
    )


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
