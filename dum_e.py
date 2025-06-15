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
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService, Language
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.anthropic.llm import AnthropicLLMService, AnthropicLLMContext
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams

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
            wrist_cam_idx=2,
            front_cam_idx=0,
            callback_handler=callback_handler,
        )
    return _robot_agent


async def order_food(params: FunctionCallParams):
    """
    Order food from a restaurant.
    """
    # Use these to call the actual APIs
    restaurant = params.arguments.get("restaurant")
    food = params.arguments.get("food")

    async def _speak_status_update(delay: float = 2):
        await asyncio.sleep(delay)
        await params.llm.queue_frame(TTSSpeakFrame("Working on that."))

    status_update_task = asyncio.create_task(_speak_status_update(1))

    try:
        await asyncio.sleep(5)
        # Sample response
        await params.result_callback(
            {
                "status": "order delivered",
                "response": "Items are available at the user's table",
            }
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

    async def _run_robot_agent_background():
        try:
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


system_prompt = """You are a helpful robot assistant that assists the user with their requests.

You have access to public APIs as well as a physical robot to perform your tasks.

Your output will be converted to audio so don't include special characters and be concise in your answers. \
Respond to what the user said in a professional and helpful way."""


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

    llm = AnthropicLLMService(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-5-haiku-20241022",
        params=AnthropicLLMService.InputParams(temperature=0.7, max_tokens=500),
    )

    # You can also register a function_name of None to get all functions
    # sent to the same callback with an additional function_name parameter.
    llm.register_function("order_food", order_food)
    llm.register_function("run_robot_agent", run_robot_agent)

    # @llm.event_handler("on_function_calls_started")
    # async def on_function_calls_started(service, function_calls):
    #     await tts.queue_frame(TTSSpeakFrame("Let me check on that."))

    food_function = FunctionSchema(
        name="order_food",
        description="Order food from a restaurant",
        properties={
            "restaurant": {
                "type": "string",
                "description": "The name of the restaurant",
            },
            "food": {
                "type": "string",
                "description": "The food to order",
            },
        },
        required=["restaurant", "food"],
    )

    robot_function = FunctionSchema(
        name="run_robot_agent",
        description="Run the robot agent to physically assist the user and execute actions based on the given instruction.",
        properties={
            "instruction": {
                "type": "string",
                "description": "The task to be executed by the robot",
            },
        },
        required=["instruction"],
    )
    tools = ToolsSchema(standard_tools=[food_function, robot_function])

    context = AnthropicLLMContext(messages=[], tools=tools, system=system_prompt)
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
        # await task.queue_frames([context_aggregator.user().get_context_frame()])
        await task.queue_frames(
            [TTSSpeakFrame(f"Hi, Dummy is here! How can I help you?")]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(run_dum_e, transport_params=transport_params)
