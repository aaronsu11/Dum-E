import argparse
import asyncio
import os

import httpx
from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, Language
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams
from pipecat.transports.services.daily import DailyParams

from embodiment.so_arm10x.agent import create_robot_agent_with_config

load_dotenv(override=True)

# Setup clean logging for voice assistant (but preserve loguru for pipecat)
# setup_robot_logging(log_level="INFO", include_timestamps=True)

# Global set to keep track of background tasks to prevent garbage collection
_background_tasks = set()

# Global robot agent for resource management
_robot_agent = None


def get_robot_agent():
    """Get or create a shared robot agent instance for resource management."""
    global _robot_agent
    if _robot_agent is None:
        # Create callback handler that uses loguru (same as pipecat)
        from logging_config import create_clean_callback_handler

        callback_handler = create_clean_callback_handler(show_thinking=False)

        _robot_agent = create_robot_agent_with_config(
            enable_camera=True,
            arm_cam_idx=2,
            top_cam_idx=0,
            callback_handler=callback_handler,
        )
    return _robot_agent


async def fetch_weather_from_api(params: FunctionCallParams):
    """
    Fetch current weather using Open-Meteo and Nominatim for geocoding.
    """
    location = params.arguments.get("location")
    temp_unit = params.arguments.get("format", "celsius")

    async def _speak_status_update(delay: float = 2):
        await asyncio.sleep(delay)
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
                "temperature_unit": (
                    "celsius" if temp_unit == "celsius" else "fahrenheit"
                ),
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


async def run_robot_agent(params: FunctionCallParams):
    """
    Execute robot agent tasks in the background with real-time audio feedback.
    Uses async iterators for streaming updates with clean logging.
    """
    import logging

    logger = logging.getLogger("voice_assistant.robot")

    await params.llm.queue_frame(TTSSpeakFrame("OK!"))

    async def _run_robot_agent_background():
        try:
            # Setup clean logging for robot components only (without conflicting with pipecat)
            import logging

            # Only setup robot-specific loggers, don't touch loguru
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("botocore").setLevel(logging.WARNING)
            logging.getLogger("boto3").setLevel(logging.WARNING)
            logging.getLogger("anthropic").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)

            # Use shared robot agent for better resource management
            so100_agent = get_robot_agent()
            logger.info(
                f"🎯 Processing robot instruction: {params.arguments['instruction']}"
            )

            # Camera warmup and robot connection is handled internally by the agent
            # Use async iterator for streaming events
            async for event in so100_agent.astream(params.arguments["instruction"]):
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
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
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


async def run_dum_e(
    transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool
):
    logger.info(f"Starting bot")

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="cgSgspJ2msm6clMCkdW9",
        sample_rate=24000,
        params=ElevenLabsTTSService.InputParams(language=Language.EN),
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # You can also register a function_name of None to get all functions
    # sent to the same callback with an additional function_name parameter.
    llm.register_function("get_current_weather", fetch_weather_from_api)
    llm.register_function("run_robot_agent", run_robot_agent)

    # @llm.event_handler("on_function_calls_started")
    # async def on_function_calls_started(service, function_calls):
    #     await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

    weather_function = FunctionSchema(
        name="get_current_weather",
        description="Get the current weather",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "format": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "The temperature unit to use. Infer this from the user's location.",
            },
        },
        required=["location", "format"],
    )

    robot_function = FunctionSchema(
        name="run_robot_agent",
        description="Run the robot agent to execute physical actions based on the given instruction. Confirm the action with the user before executing it.",
        properties={
            "instruction": {
                "type": "string",
                "description": "The task to be executed by the robot",
            },
        },
        required=["instruction"],
    )
    tools = ToolsSchema(standard_tools=[weather_function, robot_function])

    messages = [
        {
            "role": "system",
            "content": "You are a helpful robot assistant. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters and be concise in your answers. Respond to what the user said in a professional and helpful way.",
        },
    ]

    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            report_only_initial_ttfb=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"Client connected")
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_dum_e, transport_params=transport_params)
