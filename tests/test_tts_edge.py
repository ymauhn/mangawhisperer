"""Tests for the Edge-TTS engine + TTS factory (fake network — offline)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from mangawhisperer.engines.tts_edge import DEFAULT_EDGE_CAST, EdgeTTSEngine
from mangawhisperer.engines.tts_factory import create_tts_engine
from mangawhisperer.models import ContextualizedBlock

BLOCK = ContextualizedBlock(text="Eu vou sobreviver.", speaker_id="Guts", is_speech=True)


class FakeCommunicate:
    """Mimics edge_tts.Communicate: async .save() writing decodable audio."""

    calls: list[dict] = []

    def __init__(self, text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz") -> None:
        FakeCommunicate.calls.append(
            {"text": text, "voice": voice, "rate": rate, "pitch": pitch}
        )

    async def save(self, path: str) -> None:
        # Real Edge writes mp3; libsndfile sniffs content, not extension,
        # so writing WAV bytes to the .mp3 temp path decodes fine.
        tone = (np.sin(np.arange(12000) / 24000 * 2 * np.pi * 220) * 15000).astype(np.int16)
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(tone.tobytes())


@pytest.fixture(autouse=True)
def _clear_calls():
    FakeCommunicate.calls = []


class TestEdgeTTSEngine:
    def test_produces_pipeline_wav_and_metadata(self, tmp_path: Path) -> None:
        engine = EdgeTTSEngine(communicate_factory=FakeCommunicate)
        metadata = engine.synthesize(BLOCK, tmp_path / "seg.wav")

        with wave.open(str(metadata.file_path), "rb") as wav:
            assert wav.getframerate() == 24000
            assert wav.getnchannels() == 1
        assert metadata.duration_ms == pytest.approx(500, abs=5)
        assert not (tmp_path / "seg.edge.mp3").exists(), "temp mp3 cleaned up"

    def test_cast_gets_distinct_voice_specs(self, tmp_path: Path) -> None:
        engine = EdgeTTSEngine(communicate_factory=FakeCommunicate)
        engine.synthesize(BLOCK, tmp_path / "a.wav")
        engine.synthesize(
            ContextualizedBlock(text="Olá", speaker_id="Casca", is_speech=True),
            tmp_path / "b.wav",
        )

        guts, casca = FakeCommunicate.calls
        assert guts["voice"] == "pt-BR-AntonioNeural"
        assert casca["voice"] == "pt-BR-FranciscaNeural"
        assert guts["pitch"] == "-15Hz", "per-character pitch distinguishes shared voices"

    def test_final_period_normalized_before_synthesis(self, tmp_path: Path) -> None:
        EdgeTTSEngine(communicate_factory=FakeCommunicate).synthesize(BLOCK, tmp_path / "s.wav")
        assert FakeCommunicate.calls[0]["text"] == "Eu vou sobreviver"

    def test_speed_shifts_rate_globally(self, tmp_path: Path) -> None:
        engine = EdgeTTSEngine(communicate_factory=FakeCommunicate, speed=1.06)
        engine.synthesize(BLOCK, tmp_path / "s.wav")

        base_rate = DEFAULT_EDGE_CAST["Guts"][1]
        assert FakeCommunicate.calls[0]["rate"] == f"{base_rate + 6:+d}%"

    def test_uncast_speaker_is_deterministic(self) -> None:
        a = EdgeTTSEngine(communicate_factory=FakeCommunicate)
        b = EdgeTTSEngine(communicate_factory=FakeCommunicate)
        assert a.voice_for("Aldeã") == b.voice_for("Aldeã")


class TestTTSFactory:
    def test_backends_resolve(self) -> None:
        from mangawhisperer.engines.placeholders import SilentTTSEngine
        from mangawhisperer.engines.tts import XTTSEngine

        assert isinstance(create_tts_engine("xtts", synthesizer=object()), XTTSEngine)
        assert isinstance(
            create_tts_engine("edge", communicate_factory=FakeCommunicate), EdgeTTSEngine
        )
        assert isinstance(
            create_tts_engine("silent", speed=1.1, extra_synthesis_kwargs={}), SilentTTSEngine
        )

    def test_unknown_backend_rejected(self) -> None:
        with pytest.raises(ValueError, match="xtts"):
            create_tts_engine("kokoro")
