"""Opt-in LIVE speech-service tests (real network + real API keys).

These tests hit the real ElevenLabs / Deepgram APIs and are therefore SKIPPED by
default — they only run when the relevant key is present AND the opt-in flag
``DUME_RUN_LIVE_SPEECH_TESTS=1`` is set. This keeps CI keyless and offline while
giving a way to catch failures that the keyless wiring tests cannot — most
importantly the "configured voice produces no audio" class of bug.

Background (Phase 2 UAT, test 2): the cascaded ``zh`` fallback was silently
producing no audio because the configured ElevenLabs voice was a paid *library*
voice. On a free-tier key the REST API returns ``402 payment_required`` and the
streaming websocket path fails silently (a ``final`` message with no audio
frames). The wiring tests all passed because the preset/model/language code were
correct — only a live synthesis call surfaces the entitlement failure. This test
makes that check automatable.

Run locally with:

    DUME_RUN_LIVE_SPEECH_TESTS=1 uv run pytest tests/test_speech_live.py -v
"""

import json
import os
import urllib.request

import pytest

try:
    # Mirror the app: the key normally lives in .env, not the shell environment.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is a project dependency
    pass

RUN_LIVE = os.getenv("DUME_RUN_LIVE_SPEECH_TESTS") == "1"
ELEVENLABS_KEY = os.getenv("ELEVENLABS_API_KEY")

pytestmark = pytest.mark.skipif(
    not (RUN_LIVE and ELEVENLABS_KEY),
    reason="live speech tests are opt-in: set DUME_RUN_LIVE_SPEECH_TESTS=1 and ELEVENLABS_API_KEY",
)


def _elevenlabs_tts_bytes(voice_id: str, text: str, model_id: str, language_code: str):
    """Synthesize via the ElevenLabs REST API. Returns (status, payload).

    status is the HTTP status code (200 on success). On success payload is the
    audio byte length; on error it is the decoded error body (JSON/text).
    """
    body = json.dumps(
        {"text": text, "model_id": model_id, "language_code": language_code}
    ).encode()
    req = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        data=body,
        headers={
            "xi-api-key": ELEVENLABS_KEY,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, len(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


class TestElevenLabsZhVoiceLive:
    """The configured zh ElevenLabs voice must actually synthesize Mandarin audio
    on the deployment's account — not just be wired correctly. This catches the
    paid-library-voice silent-failure (402) that produced no audio in UAT."""

    def test_configured_zh_voice_produces_audio(self):
        from pipecat_server import LANGUAGE_PRESETS

        cfg = LANGUAGE_PRESETS["zh"]["elevenlabs"]
        voice_id = cfg["voice_id"]
        model_id = cfg["model"]

        status, payload = _elevenlabs_tts_bytes(
            voice_id=voice_id,
            text="你好呀，主人",
            model_id=model_id,
            language_code="zh",
        )

        if status == 402:
            pytest.fail(
                f"zh voice {voice_id!r} requires a PAID ElevenLabs plan "
                f"(402 payment_required) — it produces NO audio on this account. "
                f"Use a premade voice or upgrade the plan. API said: {payload}"
            )
        assert status == 200, (
            f"zh ElevenLabs synthesis failed (HTTP {status}) for voice {voice_id!r} / "
            f"model {model_id!r}: {payload}"
        )
        # A real Mandarin clip is well over a few hundred bytes; guard against an
        # empty/truncated 200 body.
        assert isinstance(payload, int) and payload > 1000, (
            f"zh ElevenLabs synthesis returned only {payload} bytes — "
            "expected real audio (likely a silent/empty response)"
        )

    def test_configured_zh_model_is_streaming_multilingual(self):
        """The websocket path pipecat uses (multi-stream-input) only serves the
        streaming models, which are also the ones that accept a language_code.
        A non-streaming model (e.g. eleven_multilingual_v2) yields no audio there."""
        from pipecat_server import LANGUAGE_PRESETS

        model = LANGUAGE_PRESETS["zh"]["elevenlabs"]["model"]
        assert model in {"eleven_flash_v2_5", "eleven_turbo_v2_5"}, (
            f"zh model {model!r} is not streaming-compatible; the multi-stream-input "
            "websocket endpoint returns no audio for it"
        )
