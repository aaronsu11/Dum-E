"""
Standardized run-time and logging configuration for Dum-E.

This module provides:
- Clean, structured logging without verbose output using loguru
- Custom callback handler that filters out binary data and signatures
- Configurable log levels and formatters
- Integration with voice assistant for clean terminal output
- Config file loading utilities
"""

import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

# Use loguru normally - let pipecat handle the setup when in voice assistant context
from loguru import logger


class CleanFormatter(logging.Formatter):
    """Custom formatter that provides clean, structured log output."""

    def __init__(self):
        super().__init__()
        self.format_string = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"

    def format(self, record):
        # Use ISO format for timestamps
        record.asctime = datetime.fromtimestamp(record.created).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Truncate long messages
        if hasattr(record, "msg") and isinstance(record.msg, str):
            if len(record.msg) > 200:
                record.msg = record.msg[:197] + "..."

        return super().format(record)


class RobotCallbackHandler:
    """
    Clean callback handler for robot agent that filters out verbose content.

    This handler follows the same pattern as PrintingCallbackHandler but provides:
    - Filters out binary data (images, signatures)
    - Structured, readable output with emojis
    - Integrates cleanly with voice assistant
    - Focuses on task progress and completion
    """

    def __init__(self, log_level: str = "INFO", show_thinking: bool = False):
        """
        Initialize the clean callback handler.

        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            show_thinking: Whether to show reasoning content (default: False)
        """
        self.log_level = log_level.upper()
        self.show_thinking = show_thinking
        self.tool_count = 0
        self.previous_tool_use = None

    def __call__(self, **kwargs: Any) -> None:
        """
        Main callback method that processes strands agent events.

        This follows the same signature as PrintingCallbackHandler but with clean filtering.

        Args:
            **kwargs: Callback event data including:
            - reasoningText (Optional[str]): Reasoning text (filtered unless show_thinking=True)
            - data (str): Text content to display
            - complete (bool): Whether this is the final chunk of a response
            - current_tool_use (dict): Information about the current tool being used
        """
        reasoning_text = kwargs.get("reasoningText", "")
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        current_tool_use = kwargs.get("current_tool_use", {})

        # Handle reasoning text (filter unless explicitly requested)
        if reasoning_text:
            if self.show_thinking:
                if self.log_level in ["DEBUG"]:
                    self._log_debug(
                        f"🧠 Reasoning: {reasoning_text[:100]}{'...' if len(reasoning_text) > 100 else ''}"
                    )
            else:
                if self.log_level in ["DEBUG"]:
                    self._log_debug("🧠 Model reasoning completed")

        # Handle tool usage
        if current_tool_use and current_tool_use.get("name"):
            tool_name = current_tool_use.get("name", "Unknown tool")
            if self.previous_tool_use != current_tool_use:
                self.previous_tool_use = current_tool_use
                self.tool_count += 1
                self._log_info(f"🔧 Tool #{self.tool_count}: {tool_name}")

        # Handle data output with filtering
        if data:
            clean_data = self._filter_verbose_content(data)
            if clean_data:
                if complete:
                    self._log_info(f"🤖 Agent: {clean_data}")
                else:
                    # For streaming, we can accumulate or just log partial updates
                    if self.log_level in ["DEBUG"]:
                        self._log_debug(
                            f"🤖 Streaming: {clean_data[:50]}{'...' if len(clean_data) > 50 else ''}"
                        )

    def _log_info(self, message: str):
        """Log info message using loguru."""
        logger.info(message)

    def _log_debug(self, message: str):
        """Log debug message using loguru."""
        logger.debug(message)

    def _filter_verbose_content(self, content: str) -> Optional[str]:
        """Filter out verbose content like signatures and binary data."""
        if not isinstance(content, str):
            return None

        # Filter out signatures and binary-looking content
        if self._is_signature_or_binary(content):
            return None

        # Filter out very long single-line content (likely technical data)
        if len(content) > 1000 and "\n" not in content:
            return content[:100] + "... [content truncated]"

        return content

    def _is_signature_or_binary(self, text: str) -> bool:
        """Check if text appears to be a signature or binary data."""
        if not isinstance(text, str):
            return False

        # Check for signature patterns (base64-like content)
        if len(text) > 100 and any(char in text for char in ["=", "+", "/"]):
            # Looks like base64 or similar encoding
            non_printable = sum(1 for c in text if ord(c) < 32 or ord(c) > 126)
            if non_printable / len(text) > 0.1:  # More than 10% non-printable
                return True

        # Check for very long strings that might be signatures/hashes
        if len(text) > 500 and " " not in text:
            return True

        return False


class InterceptHandler(logging.Handler):
    """Forward stdlib ``logging`` records into loguru.

    Why this exists (SAFE-02, research Pitfall 9): LeRobot's motion clamp warns
    via ``lerobot.robots.utils.ensure_safe_goal_position`` -> the *module-level*
    ``logging.warning``, i.e. the stdlib **root** logger. With no handler on root,
    that call auto-invokes ``basicConfig()`` and installs a ``StreamHandler`` on
    **stderr** — while Dum-E's loguru sink writes to **stdout**. Two disjoint
    streams, and before this handler the repo had no bridge at all, so a
    safety-relevant warning was emitted and architecturally invisible: wrong
    stream, unformatted, outside loguru, and never reaching the message broker or
    the voice narration.

    Installing this on root also has a second effect worth naming: once root has
    a handler, ``logging.warning`` stops auto-installing the stderr one, so the
    record travels to loguru instead of being duplicated onto another stream.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Map the stdlib level onto loguru's, falling back to the numeric level
        # for custom levels loguru does not know by name.
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Walk out of the logging machinery so loguru reports the ORIGINATING
        # frame rather than this handler or a `logging/__init__.py` internal.
        # Without this the clamp warning would be attributed to Dum-E's logging
        # setup instead of to lerobot, which is actively misleading when someone
        # greps the log to find where a clamp came from.
        #
        # Starts at THIS frame (`emit`, which lives in utils.py) and always steps
        # at least once — hence the `depth == 0` clause — then skips every frame
        # belonging to the stdlib logging module.
        frame, depth = inspect.currentframe(), 0
        while frame is not None and (
            depth == 0 or frame.f_code.co_filename == logging.__file__
        ):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def install_stdlib_to_loguru_bridge() -> bool:
    """Install :class:`InterceptHandler` on the stdlib ROOT logger, idempotently.

    Returns:
        ``True`` when a handler was added, ``False`` when one was already
        present.

    Idempotence is load-bearing: a duplicated handler duplicates every bridged
    record, and a clamp warning appearing twice reads as the clamp having fired
    twice.
    """
    root = logging.getLogger()
    if any(isinstance(handler, InterceptHandler) for handler in root.handlers):
        return False

    root.addHandler(InterceptHandler())
    # Root defaults to WARNING, which is already low enough for the clamp
    # warning. Only raise the floor when root was left at NOTSET or higher than
    # WARNING, which would drop the very record this bridge exists to carry.
    if root.level == logging.NOTSET or root.level > logging.WARNING:
        root.setLevel(logging.WARNING)
    return True


_logging_configured = False


def setup_robot_logging(
    log_level: str = "INFO", include_timestamps: bool = True
) -> None:
    """
    Setup standardized logging for the robot system using loguru.

    If loguru already has handlers (e.g., from pipecat), skip custom setup.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        include_timestamps: Whether to include timestamps in console output
    """
    global _logging_configured

    # Only configure once to avoid duplicate handlers
    if _logging_configured:
        return

    # Check if loguru already has handlers (e.g., from pipecat)
    if len(logger._core.handlers) > 0:
        # Loguru is already configured (probably by pipecat), just suppress external libraries
        pass
    else:
        # Remove default loguru handler and add custom one
        logger.remove()

        # Configure loguru format
        if include_timestamps:
            format_string = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>robot</cyan> | <level>{message}</level>"
        else:
            format_string = "<level>{level: <8}</level> | <cyan>robot</cyan> | <level>{message}</level>"

        logger.add(
            sys.stdout, format=format_string, level=log_level.upper(), colorize=True
        )

    # Bridge stdlib logging into loguru BEFORE the per-library level tuning
    # below. The named loggers below are noise suppression; this is the opposite
    # concern — the ROOT logger, which nothing here touches and which is where
    # LeRobot's motion-clamp warning lands (SAFE-02).
    install_stdlib_to_loguru_bridge()

    # Suppress verbose logs from external libraries (always do this)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Suppress robot component verbose logs
    logging.getLogger("so_arm10x.implementations").setLevel(logging.WARNING)

    # Suppress very verbose logs
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Set all third-party loggers to WARNING to reduce noise
    for name in ["requests", "urllib", "aiohttp", "websockets"]:
        logging.getLogger(name).setLevel(logging.WARNING)

    _logging_configured = True


def setup_logging(log_level: str = "INFO", include_timestamps: bool = True) -> None:
    """Alias for setup_robot_logging for backward compatibility."""
    setup_robot_logging(log_level, include_timestamps)


def create_clean_callback_handler(show_thinking: bool = False) -> RobotCallbackHandler:
    """
    Create a clean callback handler for robot agents.

    Args:
        show_thinking: Whether to show model reasoning (default: False for clean output)

    Returns:
        Configured callback handler
    """
    return RobotCallbackHandler(show_thinking=show_thinking)


def load_config_file(config_path: Optional[str]) -> Dict[str, Any]:
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "PyYAML is required to load YAML configs. Install with: pip install pyyaml"
            ) from e
        return yaml.safe_load(path.read_text()) or {}

    raise ValueError("Unsupported config format. Use .json or .yaml/.yml")
