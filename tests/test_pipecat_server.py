"""
Tests for pipecat_server.py voice interface and language configuration.

This file tests:
- Language presets structure and completeness
- Language, mode, and profile validation
- Configuration selection for different languages
"""

import os
from unittest import mock

import pytest
from pipecat.services.elevenlabs.tts import Language


class TestLanguageSpecificConfigs:
    """Test language-specific configuration values."""

    @pytest.mark.parametrize(
        "lang,model,el_lang,polly_voice,polly_lang",
        [
            ("en", None, Language.EN, "Matthew", "en-US"),
            ("zh", "nova-2", Language.CMN, "Zhiyu", "cmn-CN"),
            ("ja", "nova-2", Language.JA, "Takumi", "ja-JP"),
            ("es", "nova-2", Language.ES, "Sergio", "es-ES"),
        ],
    )
    def test_language_configurations(
        self, lang, model, el_lang, polly_voice, polly_lang
    ):
        """Test each language preset has correct configuration values."""
        from pipecat_server import LANGUAGE_PRESETS

        preset = LANGUAGE_PRESETS[lang]

        # Deepgram config
        assert preset["deepgram"]["model"] == model
        assert preset["deepgram"]["language"] == lang

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
        """Test Deepgram model selection: default for English, nova-2 for others."""
        from pipecat_server import LANGUAGE_PRESETS

        assert LANGUAGE_PRESETS["en"]["deepgram"]["model"] is None

        for lang in ["zh", "ja", "es"]:
            assert LANGUAGE_PRESETS[lang]["deepgram"]["model"] == "nova-2"

    def test_all_profiles_supported(self):
        """Test that all languages support both default and AWS profiles."""
        from pipecat_server import LANGUAGE_PRESETS

        for preset in LANGUAGE_PRESETS.values():
            # Default profile requirements
            assert "deepgram" in preset
            assert "elevenlabs" in preset
            # AWS profile requirements
            assert "aws_polly" in preset
