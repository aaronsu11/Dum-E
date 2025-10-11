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
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.anthropic.llm import AnthropicLLMService, AnthropicLLMContext
from pipecat.services.aws.llm import AWSBedrockLLMService, AWSBedrockLLMContext
from pipecat.services.aws.stt import AWSTranscribeSTTService
from pipecat.services.aws.tts import AWSPollyTTSService
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from pipecat.services.aws.nova_sonic.context import AWSNovaSonicLLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, Language
from pipecat.services.llm_service import FunctionCallResultCallback, LLMService
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
        vad_analyzer=SileroVADAnalyzer(),
    ),
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
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
        result_callback: FunctionCallResultCallback,
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

    async def _list_tools(
        self, session: ClientSession, mcp_tool_wrapper: Callable, llm: LLMService
    ):
        """Override the _list_tools method to use long_running flag from the tool metadata for setting cancel_on_interruption."""
        available_tools: ListToolsResult = await session.list_tools()
        tool_schemas: List[FunctionSchema] = []

        try:
            logger.debug(f"Found {len(available_tools)} available tools")
        except:
            pass

        for tool in available_tools.tools:
            tool_name = tool.name
            # If the tool is long running, we don't want to interrupt it on new voice input
            cancel_on_interruption = (
                False if tool.meta.get("long_running", False) else True
            )
            logger.debug(f"Processing tool: {tool_name}")
            logger.debug(f"Tool description: {tool.description}")
            logger.debug(f"Tool metadata: {tool.meta}")

            try:
                # Convert the schema
                function_schema = self._convert_mcp_schema_to_pipecat(
                    tool_name,
                    {"description": tool.description, "input_schema": tool.inputSchema},
                )

                # Register the wrapped function
                logger.debug(
                    f"Registering function handler for '{tool_name}' with cancel_on_interruption: {cancel_on_interruption}"
                )
                llm.register_function(
                    tool_name,
                    mcp_tool_wrapper,
                    cancel_on_interruption=cancel_on_interruption,
                )

                # Add to list of schemas
                tool_schemas.append(function_schema)
                logger.debug(f"Successfully registered tool '{tool_name}'")

            except Exception as e:
                logger.error(f"Failed to register tool '{tool_name}': {str(e)}")
                logger.exception("Full exception details:")
                continue

        logger.debug(f"Completed registration of {len(tool_schemas)} tools")
        tools_schema = ToolsSchema(standard_tools=tool_schemas)

        return tools_schema


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
                model="us.anthropic.claude-3-5-haiku-20241022-v1:0",
                params=AWSBedrockLLMService.InputParams(
                    temperature=0.1,
                    max_tokens=500,
                ),
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
                model="claude-3-5-haiku-20241022",
                params=AnthropicLLMService.InputParams(temperature=0.1, max_tokens=500),
            )

            elevenlabs_config = language_preset["elevenlabs"]
            tts = ElevenLabsTTSService(
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                voice_id=elevenlabs_config["voice_id"],
                sample_rate=24000,
                params=ElevenLabsTTSService.InputParams(
                    language=elevenlabs_config["language"]
                ),
            )

    # Use enhanced MCPClient to avoid interrupting long-running tools and enable voice updates on progress
    mcp = AsyncMCPClient(
        server_params=StreamableHttpParameters(url="http://localhost:8000/mcp")
    )

    # This registers the tools with the LLM using _list_tools which sets the cancel_on_interruption flag
    # based on the tool metadata defined in FastMCP server
    tools = await mcp.register_tools(llm)

    # @llm.event_handler("on_function_calls_started")
    # async def on_function_calls_started(service, function_calls):
    #     await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

    if profile == "aws":
        # For Nova Sonic speech-to-speech, append the special trigger instruction so the
        # assistant will start speaking when it hears the synthetic "ready" trigger.
        if mode == "speech_to_speech":
            context = AWSNovaSonicLLMContext(
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
            context = AWSBedrockLLMContext(
                messages=[
                    {
                        "role": "system",
                        "content": [{"text": system_prompt}],
                    }
                ],
                tools=tools,
            )
    else:
        context = AnthropicLLMContext(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
            ],
            tools=tools,
        )

    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    if mode == "cascaded":
        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                rtvi,  # RTVI processor
                stt,  # Speech-to-text
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # Text-to-speech
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )
    elif mode == "speech_to_speech":
        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                context_aggregator.user(),
                llm,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

    task = PipelineTask(
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
        observers=[RTVIObserver(rtvi)],
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        if mode == "speech_to_speech" and profile == "aws":
            # For Nova Sonic (speech-to-speech), trigger the assistant to speak using the synthetic "ready" audio,
            # so the model responds without having to wait for real user audio.
            await task.queue_frames([LLMRunFrame()])
            await llm.trigger_assistant_response()
        else:
            # Use language-specific greeting
            greeting = language_preset["greeting"]
            await task.queue_frames([TTSSpeakFrame(greeting)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    async def progress_handler(
        progress: float, total: float | None, message: str | None
    ) -> None:
        """Handle voice updates on MCP progress notifications."""
        if voice_updates and mode == "cascaded" and message:
            await task.queue_frame(TTSSpeakFrame(message))

    # Set the progress callback with reference to the task for centralized queueing
    mcp.set_progress_callback(progress_handler)

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


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
            f"Authorization=Basic {LANGFUSE_AUTH}"
        )

        # Configured automatically from .env
        exporter = OTLPSpanExporter()

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
