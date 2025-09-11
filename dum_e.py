import asyncio
import os
from typing import Literal

print("🚀 Starting Pipecat bot...")

import httpx
from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.aws.llm import AWSBedrockLLMService, AWSBedrockLLMContext
from pipecat.services.aws.stt import AWSTranscribeSTTService
from pipecat.services.aws.tts import AWSPollyTTSService
from pipecat.services.aws_nova_sonic.aws import AWSNovaSonicLLMService
from pipecat.services.aws_nova_sonic.context import AWSNovaSonicLLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, Language
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.anthropic.llm import AnthropicLLMService, AnthropicLLMContext
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport

from embodiment.so_arm10x.agent import create_robot_agent

logger.info("✅ All components loaded successfully!")

load_dotenv(override=True)

# Setup clean logging for voice assistant (but preserve loguru for pipecat)
# setup_robot_logging(log_level="INFO", include_timestamps=True)

# Global set to keep track of background tasks to prevent garbage collection
_background_tasks = set()

# Global robot agent for resource management
_robot_agent = None


async def fetch_weather_from_api(
    params: FunctionCallParams,
    location: str,
    unit: Literal["celsius", "fahrenheit"] = "celsius",
):
    """
    Fetch current weather using Open-Meteo and Nominatim for geocoding.

    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The temperature unit to use. Infer this from the user's location.
    """

    async def _speak_status_update(delay: float = 2):
        await asyncio.sleep(delay)
        # This will be ignored in speech-to-speech mode
        await params.llm.queue_frame(TTSSpeakFrame("Let me check on that."))

    status_update_task = asyncio.create_task(_speak_status_update(1))

    try:
        # 1. Geocode location to lat/lon
        async with httpx.AsyncClient() as client:
            geo_resp = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location, "format": "json", "limit": 1},
                headers={"User-Agent": "robot-assistant/1.0"},
            )
            geo_resp.raise_for_status()
            geo_data = geo_resp.json()
            if not geo_data:
                await params.result_callback({"error": "location not found"})
                status_update_task.cancel()
                return

            lat = geo_data[0]["lat"]
            lon = geo_data[0]["lon"]

            # 2. Query Open-Meteo for current weather
            weather_url = "https://api.open-meteo.com/v1/forecast"
            weather_params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
                "temperature_unit": ("celsius" if unit == "celsius" else "fahrenheit"),
            }
            weather_resp = await client.get(weather_url, params=weather_params)
            weather_resp.raise_for_status()
            weather_data = weather_resp.json()
            current = weather_data.get("current_weather", {})

            if not current:
                await params.result_callback({"error": "weather not found"})
                status_update_task.cancel()
                return

            # 3. Return weather info
            conditions = f"{current.get('weathercode', 'unknown')}"
            temperature = current.get("temperature", "unknown")
            await params.result_callback(
                {"conditions": conditions, "temperature": temperature}
            )

    except Exception as e:
        await params.result_callback({"error": str(e)})
    finally:
        status_update_task.cancel()


# TODO: replace this with a MCP server
def get_robot_agent():
    """Get or create a shared robot agent instance for resource management."""
    global _robot_agent
    if _robot_agent is None:
        # Create callback handler that uses loguru (same as pipecat)
        from logging_config import create_clean_callback_handler

        callback_handler = create_clean_callback_handler(show_thinking=False)

        _robot_agent = create_robot_agent(
            robot_port=os.getenv("SO_ARM_PORT"),
            callback_handler=callback_handler,
        )
    return _robot_agent


async def run_robot_agent(params: FunctionCallParams, instruction: str):
    """
    Execute robot agent tasks in the background based on the given instruction with real-time audio feedback.

    Args:
        instruction: The task to be executed by the robot
    """

    async def _run_robot_agent_background():
        try:
            # Use shared robot agent for better resource management
            so10x_agent = get_robot_agent()
            logger.info(f"🎯 Processing robot instruction: {instruction}")

            # Camera warmup and robot connection is handled internally by the agent
            # Use async iterator for streaming events
            async for event in so10x_agent.astream(instruction):
                # Handle different event types for better TTS experience
                if (
                    "message" in event
                    and isinstance(event["message"], dict)
                    and event["message"].get("role") == "assistant"
                ):
                    for content in event["message"].get("content", []):
                        if (
                            isinstance(content, dict)
                            and "text" in content
                            and not _is_verbose_content(content["text"])
                        ):
                            await params.llm.queue_frame(TTSSpeakFrame(content["text"]))
                            logger.info(f"🔊 Speaking: {content['text']}")
                elif event.get("type") == "warmup_progress":
                    # Optionally announce warmup progress
                    if event.get("progress", 0) == 1.0:  # Only announce when complete
                        await params.llm.queue_frame(
                            TTSSpeakFrame("Robot ready, executing task...")
                        )
                        logger.info("🤖 Robot cameras warmed up and ready")
                elif event.get("type") == "completion":
                    await params.llm.queue_frame(
                        TTSSpeakFrame("Task completed successfully.")
                    )
                    logger.info("✅ Robot task completed successfully")
                elif event.get("type") == "error":
                    error_msg = event.get("error", "Unknown error")
                    await params.llm.queue_frame(
                        TTSSpeakFrame(f"Task failed: {error_msg}")
                    )
                    logger.error(f"❌ Robot task failed: {error_msg}")

            # Robot cleanup is handled internally by the agent

        except Exception as e:
            await params.llm.queue_frame(TTSSpeakFrame(f"An error occurred: {str(e)}"))
            logger.error(f"💥 Error in robot agent: {e}")
        finally:
            logger.info("🏁 Robot agent background task completed")

    def _is_verbose_content(text: str) -> bool:
        """Check if content should be filtered from TTS (signatures, binary data, etc.)."""
        if not isinstance(text, str):
            return True

        # Filter out very long strings that look like signatures/hashes
        if len(text) > 500 and " " not in text:
            return True

        # Filter out base64-like content
        if len(text) > 100 and any(char in text for char in ["=", "+", "/"]):
            non_printable = sum(1 for c in text if ord(c) < 32 or ord(c) > 126)
            if non_printable / len(text) > 0.1:
                return True

        return False

    async def run_with_timeout():
        try:
            await asyncio.wait_for(_run_robot_agent_background(), timeout=300)
        except asyncio.TimeoutError:
            await params.llm.queue_frame(
                TTSSpeakFrame(
                    "The robot task timed out and was cancelled. Would you like to try again?"
                )
            )
        except Exception as e:
            await params.llm.queue_frame(
                TTSSpeakFrame(f"An error occurred during the robot task: {str(e)}")
            )

    logger.info("Robot agent initiated - running in background")
    task = asyncio.create_task(run_with_timeout())
    _background_tasks.add(task)
    task.add_done_callback(lambda t: _background_tasks.discard(t))
    await params.result_callback(
        {"status": "command sent", "response": "task will run in background"}
    )


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


system_prompt = """You are a helpful robot assistant. \
Your goal is to demonstrate your capabilities in a succinct way. 

Don't include special characters and be concise in your answers. 

Respond to what the user said in a professional and helpful way."""


async def run_dum_e(
    transport: BaseTransport,
    runner_args: RunnerArguments,
    mode: Literal["cascaded", "speech_to_speech"] = "cascaded",
    profile: Literal["default", "aws"] = "default",
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
                voice_id="cgSgspJ2msm6clMCkdW9",
                sample_rate=24000,
                params=ElevenLabsTTSService.InputParams(language=Language.EN),
            )

    # by default, function calls can be interrupted
    llm.register_direct_function(fetch_weather_from_api)
    llm.register_direct_function(run_robot_agent, cancel_on_interruption=False)

    # @llm.event_handler("on_function_calls_started")
    # async def on_function_calls_started(service, function_calls):
    #     await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

    tools = ToolsSchema(standard_tools=[fetch_weather_from_api, run_robot_agent])

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
                # system=system_prompt, # There is a system message conversion issue as of 0.0.81
            )
    else:
        context = AnthropicLLMContext(tools=tools, system=system_prompt)

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
            await task.queue_frames([context_aggregator.user().get_context_frame()])
            await llm.trigger_assistant_response()
        else:
            await task.queue_frames([TTSSpeakFrame(f"Hello there!")])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    transport = await create_transport(runner_args, transport_params)

    await run_dum_e(transport, runner_args, mode="speech_to_speech", profile="aws")


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
