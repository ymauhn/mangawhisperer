"""Tests for the two-bus timeline mixer (narration/SFX gains + BGM)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from mangawhisperer.engines.mixing import SAMPLE_RATE, MixingStitcher
from mangawhisperer.models import AudioSegmentMetadata


def _write_tone(path: Path, duration_s: float, amplitude: float = 0.5,
                freq: float = 440.0, rate: int = SAMPLE_RATE) -> None:
    t = np.arange(int(rate * duration_s)) / rate
    pcm = (np.sin(2 * np.pi * freq * t) * amplitude * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


def _segment(path: Path, speaker: str, index: int) -> AudioSegmentMetadata:
    with wave.open(str(path), "rb") as wav:
        ms = round(wav.getnframes() * 1000 / wav.getframerate())
    return AudioSegmentMetadata(file_path=path, speaker_id=speaker, duration_ms=ms, block_index=index)


def _read_float(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnchannels() == 1
        return np.frombuffer(wav.readframes(wav.getnframes()), np.int16).astype(np.float64) / 32768


class TestMixingStitcher:
    def test_concatenates_with_gaps_like_the_plain_stitcher(self, tmp_path: Path) -> None:
        _write_tone(tmp_path / "a.wav", 1.0)
        _write_tone(tmp_path / "b.wav", 0.5)
        segments = [_segment(tmp_path / "a.wav", "Guts", 0), _segment(tmp_path / "b.wav", "Casca", 1)]

        final = MixingStitcher().stitch(segments, tmp_path / "out.wav", gap_ms=200)

        data = _read_float(final)
        expected = int(SAMPLE_RATE * (1.0 + 0.2 + 0.5))
        assert abs(len(data) - expected) <= 2

    def test_sfx_channel_gain_is_independent(self, tmp_path: Path) -> None:
        _write_tone(tmp_path / "voice.wav", 0.5, amplitude=0.5)
        _write_tone(tmp_path / "sfx.wav", 0.5, amplitude=0.5)
        segments = [
            _segment(tmp_path / "voice.wav", "Guts", 0),
            _segment(tmp_path / "sfx.wav", "SFX", 1),
        ]

        final = MixingStitcher(sfx_gain=0.0).stitch(segments, tmp_path / "out.wav", gap_ms=100)

        data = _read_float(final)
        voice_rms = float(np.sqrt(np.mean(data[: int(0.4 * SAMPLE_RATE)] ** 2)))
        sfx_region = data[int(0.65 * SAMPLE_RATE):]
        sfx_rms = float(np.sqrt(np.mean(sfx_region**2)))
        assert voice_rms > 0.2, "voice untouched"
        assert sfx_rms < 0.01, "sfx muted by its own gain"

    def test_bgm_loops_under_the_whole_timeline(self, tmp_path: Path) -> None:
        _write_tone(tmp_path / "bgm.wav", 0.25, amplitude=0.8, freq=110)
        silence = np.zeros(int(SAMPLE_RATE * 3.0), dtype=np.int16)
        with wave.open(str(tmp_path / "quiet.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(silence.tobytes())
        segments = [_segment(tmp_path / "quiet.wav", "Narrator", 0)]

        final = MixingStitcher(
            bgm_path=tmp_path / "bgm.wav", bgm_gain=0.5, bgm_fade_ms=300
        ).stitch(segments, tmp_path / "out.wav")

        data = _read_float(final)
        middle = data[int(1.0 * SAMPLE_RATE): int(2.0 * SAMPLE_RATE)]
        assert float(np.sqrt(np.mean(middle**2))) > 0.1, "looped bgm audible mid-track"
        assert abs(float(data[0])) < 0.01, "fade-in starts from silence"
        assert abs(float(data[-1])) < 0.01, "fade-out ends in silence"

    def test_bgm_resampled_from_other_rates(self, tmp_path: Path) -> None:
        _write_tone(tmp_path / "bgm44.wav", 0.5, amplitude=0.6, rate=44100)
        _write_tone(tmp_path / "voice.wav", 1.0, amplitude=0.1)
        segments = [_segment(tmp_path / "voice.wav", "Guts", 0)]

        final = MixingStitcher(bgm_path=tmp_path / "bgm44.wav", bgm_gain=0.4, bgm_fade_ms=100).stitch(
            segments, tmp_path / "out.wav"
        )
        assert len(_read_float(final)) == pytest.approx(SAMPLE_RATE, abs=2)

    def test_peak_normalization_prevents_clipping(self, tmp_path: Path) -> None:
        _write_tone(tmp_path / "loud.wav", 1.0, amplitude=0.95)
        _write_tone(tmp_path / "bgm.wav", 1.0, amplitude=0.95)
        segments = [_segment(tmp_path / "loud.wav", "Guts", 0)]

        final = MixingStitcher(bgm_path=tmp_path / "bgm.wav", bgm_gain=1.0, bgm_fade_ms=0).stitch(
            segments, tmp_path / "out.wav"
        )
        data = _read_float(final)
        assert float(np.max(np.abs(data))) <= 0.995

    def test_no_segments_produces_valid_empty_wav(self, tmp_path: Path) -> None:
        final = MixingStitcher().stitch([], tmp_path / "out.wav")
        assert len(_read_float(final)) == 0
