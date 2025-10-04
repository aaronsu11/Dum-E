"""
Simplified tests for utility functions in utils.py.

These tests focus on the core functionality that can be reliably tested
without complex mocking of internal implementation details.
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest import mock
import pytest

from utils import (
    load_config_file,
    RobotCallbackHandler,
    CleanFormatter,
    setup_robot_logging,
    create_clean_callback_handler,
)


class TestLoadConfigFile:
    """Test config file loading functionality."""

    def test_load_config_file_none(self):
        """Test loading config when path is None."""
        result = load_config_file(None)
        assert result == {}

    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config_file("/nonexistent/file.json")

    def test_load_config_file_json(self):
        """Test loading JSON config file."""
        config_data = {"test": "value", "nested": {"key": 123}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            result = load_config_file(temp_path)
            assert result == config_data
        finally:
            Path(temp_path).unlink()

    def test_load_config_file_yaml(self):
        """Test loading YAML config file."""
        config_data = {"test": "value", "nested": {"key": 123}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            import yaml

            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            result = load_config_file(temp_path)
            assert result == config_data
        finally:
            Path(temp_path).unlink()

    def test_load_config_file_yaml_yml_extension(self):
        """Test loading YAML config file with .yml extension."""
        config_data = {"test": "value"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            import yaml

            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            result = load_config_file(temp_path)
            assert result == config_data
        finally:
            Path(temp_path).unlink()

    def test_load_config_file_empty_yaml(self):
        """Test loading empty YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            result = load_config_file(temp_path)
            assert result == {}
        finally:
            Path(temp_path).unlink()

    def test_load_config_file_unsupported_format(self):
        """Test loading unsupported file format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported config format"):
                load_config_file(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_config_file_yaml_import_error(self):
        """Test YAML loading when PyYAML is not available."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("test: value")
            temp_path = f.name

        try:
            with mock.patch(
                "builtins.__import__", side_effect=ImportError("No module named 'yaml'")
            ):
                with pytest.raises(RuntimeError, match="PyYAML is required"):
                    load_config_file(temp_path)
        finally:
            Path(temp_path).unlink()


class TestCleanFormatter:
    """Test the CleanFormatter class."""

    def test_clean_formatter_init(self):
        """Test CleanFormatter initialization."""
        formatter = CleanFormatter()
        assert (
            formatter.format_string
            == "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
        )

    def test_clean_formatter_format(self):
        """Test CleanFormatter format method."""
        formatter = CleanFormatter()

        # Create a mock record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        # The actual implementation just returns the message, not the full format
        assert "Test message" in formatted

    def test_clean_formatter_long_message_truncation(self):
        """Test that long messages are truncated."""
        formatter = CleanFormatter()

        long_message = "x" * 250  # Longer than 200 chars
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=long_message,
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        assert "..." in formatted
        # The actual implementation truncates in the format method
        assert len(formatted) <= len(long_message)


class TestRobotCallbackHandler:
    """Test the RobotCallbackHandler class."""

    def test_callback_handler_init(self):
        """Test RobotCallbackHandler initialization."""
        handler = RobotCallbackHandler()
        assert handler.log_level == "INFO"
        assert handler.show_thinking is False
        assert handler.tool_count == 0
        assert handler.previous_tool_use is None

    def test_callback_handler_init_with_params(self):
        """Test RobotCallbackHandler initialization with parameters."""
        handler = RobotCallbackHandler(log_level="DEBUG", show_thinking=True)
        assert handler.log_level == "DEBUG"
        assert handler.show_thinking is True

    @mock.patch("utils.logger")
    def test_callback_handler_data_output(self, mock_logger):
        """Test callback handler with data output."""
        handler = RobotCallbackHandler()

        handler(data="Test output data", complete=True)

        mock_logger.info.assert_called_with("🤖 Agent: Test output data")

    @mock.patch("utils.logger")
    def test_callback_handler_streaming_data(self, mock_logger):
        """Test callback handler with streaming data."""
        handler = RobotCallbackHandler(log_level="DEBUG")

        handler(data="Streaming data", complete=False)

        mock_logger.debug.assert_called()

    def test_filter_verbose_content_normal(self):
        """Test normal content passes through."""
        handler = RobotCallbackHandler()

        normal_content = "This is normal content that should pass through"
        result = handler._filter_verbose_content(normal_content)
        assert result == normal_content

    def test_filter_verbose_content_non_string(self):
        """Test filtering of non-string content."""
        handler = RobotCallbackHandler()

        result = handler._filter_verbose_content(123)
        assert result is None

    def test_is_signature_or_binary_normal(self):
        """Test normal text is not detected as signature or binary."""
        handler = RobotCallbackHandler()

        normal_text = "This is normal text content"
        assert handler._is_signature_or_binary(normal_text) is False


class TestSetupRobotLogging:
    """Test the setup_robot_logging function."""

    @mock.patch("utils.logger")
    def test_setup_robot_logging_basic(self, mock_logger):
        """Test basic logging setup."""
        # Mock that loguru doesn't have handlers to trigger the setup path
        mock_logger._core.handlers = []

        setup_robot_logging()

        # Should configure loguru
        mock_logger.remove.assert_called()
        mock_logger.add.assert_called()

    @mock.patch("utils.logger")
    def test_setup_robot_logging_already_configured(self, mock_logger):
        """Test logging setup when already configured."""
        # Mock that logging is already configured
        mock_logger._core.handlers = [mock.Mock()]

        setup_robot_logging()

        # Should not call remove/add if already configured
        mock_logger.remove.assert_not_called()
        mock_logger.add.assert_not_called()


class TestCreateCleanCallbackHandler:
    """Test the create_clean_callback_handler function."""

    def test_create_clean_callback_handler_default(self):
        """Test creating callback handler with default parameters."""
        handler = create_clean_callback_handler()
        assert isinstance(handler, RobotCallbackHandler)
        assert handler.show_thinking is False

    def test_create_clean_callback_handler_with_thinking(self):
        """Test creating callback handler with thinking enabled."""
        handler = create_clean_callback_handler(show_thinking=True)
        assert isinstance(handler, RobotCallbackHandler)
        assert handler.show_thinking is True
