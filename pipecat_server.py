import os
from typing import Callable, Literal, List


print("🚀 Starting Pipecat bot...")

from dotenv import load_dotenv
from loguru import logger
from mcp import ClientSession, ListToolsResult
from mcp.client.session_group import StreamableHttpParameters
from mcp.shared.session import ProgressFnT
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.anthropic.llm import AnthropicLLMService, AnthropicLLMContext
from pipecat.services.aws.llm import AWSBedrockLLMService, AWSBedrockLLMContext
from pipecat.services.aws.stt import AWSTranscribeSTTService
from pipecat.services.aws.tts import AWSPollyTTSService
from pipecat.services.aws_nova_sonic.aws import AWSNovaSonicLLMService
from pipecat.services.aws_nova_sonic.context import AWSNovaSonicLLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, Language
from pipecat.services.llm_service import FunctionCallResultCallback, LLMService
from pipecat.services.mcp_service import MCPClient
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

# Setup clean logging for voice assistant (but preserve loguru for pipecat)
# setup_robot_logging(log_level="INFO", include_timestamps=True)

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
- Speak naturally as if in conversation - no special characters ever
- No markdown, asterisks, brackets, quotes, dashes, or number points
- No lists or bullet points - speak in flowing sentences
- Keep responses to 1-3 short sentences maximum
- If multiple pieces of information, use words like "first, second, also, and"

RESPONSE STYLE:
- Be direct and conversational
- Skip explanations of what you're doing - just give results
- Don't ask follow-up questions unless necessary
- Use natural speech patterns"""


async def run_dum_e(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    mode: Literal["cascaded", "speech_to_speech"] = "cascaded",
    profile: Literal["default", "aws"] = "default",
    voice_updates: bool = True,  # only supported in cascaded mode
):
    """
    Run Dum-E with the given transport, runner arguments, mode, and profile.

    Args:
        transport: The transport to use for the bot
        runner_args: The runner arguments to use for the bot
        mode: The mode to use for the bot. Voice status update is only supported in cascaded mode.
        profile: The profile to use for the bot
    """

    logger.info(f"Starting bot")

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
            stt = AWSTranscribeSTTService(
                api_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                # aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
                region=os.getenv("AWS_REGION"),
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

            tts = AWSPollyTTSService(
                api_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                # aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
                region=os.getenv("AWS_REGION"),
                voice_id="Matthew",
                params=AWSPollyTTSService.InputParams(
                    engine="generative", language="en-AU", rate="1.3"
                ),
            )
        else:
            stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

            llm = AnthropicLLMService(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model="claude-3-5-haiku-20241022",
                params=AnthropicLLMService.InputParams(temperature=0.1, max_tokens=500),
            )

            tts = ElevenLabsTTSService(
                api_key=os.getenv("ELEVENLABS_API_KEY"),
                voice_id="iP95p4xoKVk53GoZ742B",
                sample_rate=24000,
                params=ElevenLabsTTSService.InputParams(language=Language.EN),
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
            await task.queue_frames([TTSSpeakFrame(f"At your service, sir.")])

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

    await run_dum_e(transport, runner_args, mode="cascaded", profile="default")


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
