"""Tests for the XTTSv2 TTS engine (fake synthesizer — no model download)."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from mangawhisperer.engines.tts import DEFAULT_CAST_VOICES, FALLBACK_VOICE_POOL, XTTSEngine
from mangawhisperer.models import ContextualizedBlock

SAMPLE_RATE = 24000  # XTTSv2 output rate


class FakeSynthesizer:
    """Mimics ``TTS.api.TTS``: writes a silent WAV of fixed length."""

    def __init__(self, duration_ms: int = 600) -> None:
        self.duration_ms = duration_ms
        self.calls: list[dict] = []

    def tts_to_file(
        self, text: str, speaker: str, language: str, file_path: str,
        speed: float = 1.0, **kwargs,
    ) -> None:
        self.calls.append(
            {"text": text, "speaker": speaker, "language": language, "speed": speed, **kwargs}
        )
        frames = int(SAMPLE_RATE * self.duration_ms / 1000)
        with wave.open(file_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(b"\x00" * (frames * 2))


def make_engine(**kwargs) -> tuple[XTTSEngine, FakeSynthesizer]:
    fake = FakeSynthesizer()
    return XTTSEngine(synthesizer=fake, **kwargs), fake


BLOCK = ContextualizedBlock(text="Eu vou sobreviver!", speaker_id="Guts", is_speech=True)


class TestXTTSEngine:
    def test_cast_member_gets_curated_voice(self, tmp_path: Path) -> None:
        engine, fake = make_engine()
        engine.synthesize(BLOCK, tmp_path / "seg.wav")

        (call,) = fake.calls
        assert call["speaker"] == DEFAULT_CAST_VOICES["Guts"]
        assert call["language"] == "pt"
        assert call["text"] == "Eu vou sobreviver!"

    def test_uncast_speaker_maps_deterministically_to_fallback_pool(self) -> None:
        engine_a, _ = make_engine()
        engine_b, _ = make_engine()

        voice = engine_a.voice_for("Soldado Raso")
        assert voice in FALLBACK_VOICE_POOL
        assert engine_b.voice_for("Soldado Raso") == voice, "must be stable across instances"
        assert engine_a.voice_for("Soldado Raso") == voice, "must be stable across calls"

    def test_metadata_reports_real_wav_duration(self, tmp_path: Path) -> None:
        engine, _ = make_engine()
        metadata = engine.synthesize(BLOCK, tmp_path / "seg.wav")

        assert metadata.duration_ms == 600
        assert metadata.file_path == tmp_path / "seg.wav"
        assert metadata.speaker_id == "Guts"

    def test_block_index_increments_per_segment(self, tmp_path: Path) -> None:
        engine, _ = make_engine()
        first = engine.synthesize(BLOCK, tmp_path / "a.wav")
        second = engine.synthesize(BLOCK, tmp_path / "b.wav")

        assert (first.block_index, second.block_index) == (0, 1)

    def test_rejects_empty_fallback_pool(self) -> None:
        with pytest.raises(ValueError):
            XTTSEngine(fallback_voices=())

    def test_final_period_is_not_vocalized(self, tmp_path: Path) -> None:
        engine, fake = make_engine()
        block = ContextualizedBlock(
            text="Guts avança furioso, agarrando a criatura.", speaker_id="Narrator", is_speech=False
        )
        engine.synthesize(block, tmp_path / "seg.wav")

        assert fake.calls[0]["text"] == "Guts avança furioso, agarrando a criatura"

    def test_intonation_punctuation_is_preserved_and_collapsed(self, tmp_path: Path) -> None:
        engine, fake = make_engine()
        block = ContextualizedBlock(text='"Siiim!!! Você vai morrer?"', speaker_id="Guts", is_speech=True)
        engine.synthesize(block, tmp_path / "seg.wav")

        assert fake.calls[0]["text"] == "Siiim! Você vai morrer?"

    def test_ellipsis_becomes_short_pause(self, tmp_path: Path) -> None:
        engine, fake = make_engine()
        block = ContextualizedBlock(text="Griffith... por quê?", speaker_id="Casca", is_speech=True)
        engine.synthesize(block, tmp_path / "seg.wav")

        assert fake.calls[0]["text"] == "Griffith, por quê?"

    def test_expressiveness_kwargs_reach_the_synthesizer(self, tmp_path: Path) -> None:
        engine, fake = make_engine(
            speed=1.1, extra_synthesis_kwargs={"temperature": 0.75, "repetition_penalty": 4.0}
        )
        engine.synthesize(BLOCK, tmp_path / "seg.wav")

        (call,) = fake.calls
        assert call["speed"] == 1.1
        assert call["temperature"] == 0.75
        assert call["repetition_penalty"] == 4.0
