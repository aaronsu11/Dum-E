import base64
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
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
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
            "language": "en",
        },
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
            "model": "nova-2",
            "language": "zh",
        },
        "aws_transcribe": {
            "language": "zh-CN",
        },
        "elevenlabs": {
            "voice_id": "hkfHEbBvdQFNX4uWHqRF",  # Mandarin voice
            "language": Language.CMN,
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
            "model": "nova-2",
            "language": "ja",
        },
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
            "language": "es",
        },
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
            await result_callback(error_msg)
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

        final_response = (
            response if len(response) else "Sorry, could not call the mcp tool"
        )
        await result_callback(final_response)

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
):
    """
    Run JARVIS voice agent with the given transport, runner arguments, mode, and profile.

    Args:
        transport: The transport to use for the bot
        runner_args: The runner arguments to use for the bot
        mode: The mode to use for the bot. Voice status update is only supported in cascaded mode.
        profile: The profile to use for the bot
        language: Language code for voice interface (en, zh, ja, es)
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
            llm = AWSNovaSonicLLMService(
                secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                region=os.getenv("AWS_REGION"),
                voice_id="matthew",  # Voices: matthew, tiffany, amy
                function_call_timeout_secs=30.0,  # PIPE-04: bound normal tool calls (long_running exempt per-tool)
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
            stt = DeepgramSTTService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                live_options=LiveOptions(
                    model=deepgram_config["model"],
                    language=deepgram_config["language"],
                ),
            )

            llm = AnthropicLLMService(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model="claude-haiku-4-5",
                params=AnthropicLLMService.InputParams(temperature=0.1, max_tokens=500),
                function_call_timeout_secs=30.0,  # PIPE-04: bound normal tool calls (long_running exempt per-tool)
            )

            elevenlabs_config = language_preset["elevenlabs"]
            # tts = ElevenLabsTTSService(
            #     api_key=os.getenv("ELEVENLABS_API_KEY"),
            #     voice_id=elevenlabs_config["voice_id"],
            #     sample_rate=24000,
            #     params=ElevenLabsTTSService.InputParams(
            #         language=elevenlabs_config["language"]
            #     ),
            # )
            tts = DeepgramTTSService(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                voice="aura-2-hermes-en"
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
            # For Nova Sonic speech-to-speech, append the special trigger instruction so the
            # assistant will start speaking when it hears the synthetic "ready" trigger.
            if mode == "speech_to_speech":
                context = LLMContext(
                    messages=[
                        {
                            "role": "system",
                            "content": [
                                {
                                    "text": "\n".join(
                                        [
                                            system_prompt,
                                            AWSNovaSonicLLMService.AWAIT_TRIGGER_ASSISTANT_RESPONSE_INSTRUCTION,
                                            "Greet the user by saying 'Hello there!'",
                                        ]
                                    )
                                }
                            ],
                        }
                    ],
                    tools=tools,
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
                # For Nova Sonic (speech-to-speech), trigger the assistant to speak using the synthetic "ready" audio,
                # so the model responds without having to wait for real user audio.
                await worker.queue_frames([LLMRunFrame()])
                await llm.trigger_assistant_response()
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

    # Validate mode
    if mode not in ["cascaded", "speech_to_speech"]:
        logger.warning(f"Invalid mode '{mode}', falling back to 'cascaded'")
        mode = "cascaded"

    # Validate profile
    if profile not in ["default", "aws"]:
        logger.warning(f"Invalid profile '{profile}', falling back to 'default'")
        profile = "default"

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
        transport, runner_args, mode=mode, profile=profile, language=language
    )


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
