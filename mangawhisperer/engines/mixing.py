"""Timeline mixer: narration + SFX buses with independent gains, plus a
looping background-music (BGM) bed.

Replaces the concat-only ``WaveFileStitcher`` when channel volumes or
BGM are wanted. The narration timeline is built exactly like before
(segments in order, silence gaps between them); what's new:

* per-channel gain — segments with ``speaker_id == "SFX"`` go through
  ``sfx_gain``, everything else through ``voice_gain``;
* an optional BGM file, looped to cover the whole timeline, faded in
  and out, scaled by ``bgm_gain`` and summed underneath;
* a peak-normalization guard so the sum never clips.

Pure numpy — no ffmpeg, no external mixer.
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np

from mangawhisperer.engines.sfx import read_audio, resample_audio
from mangawhisperer.interfaces import AudioStitcher
from mangawhisperer.models import AudioSegmentMetadata

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000  # pipeline-wide narration rate (XTTS output)


class MixingStitcher(AudioStitcher):
    """Two-bus mixer implementing the :class:`AudioStitcher` contract."""

    def __init__(
        self,
        bgm_path: Path | None = None,
        voice_gain: float = 1.0,
        sfx_gain: float = 1.0,
        bgm_gain: float = 0.22,
        bgm_fade_ms: int = 2500,
    ) -> None:
        """
        Args:
            bgm_path: Optional music/ambience file (wav/ogg/mp3/flac);
                looped under the whole narration.
            voice_gain: Gain for narration/dialogue segments.
            sfx_gain: Gain for sound-effect segments.
            bgm_gain: Gain for the music bed (keep well under the voice).
            bgm_fade_ms: Fade-in/out applied to the music bed.
        """
        self._bgm_path = bgm_path
        self._voice_gain = voice_gain
        self._sfx_gain = sfx_gain
        self._bgm_gain = bgm_gain
        self._bgm_fade_ms = bgm_fade_ms

    def stitch(
        self,
        segments: list[AudioSegmentMetadata],
        output_path: Path,
        gap_ms: int = 350,
    ) -> Path:
        """Mix all buses and write a 16-bit mono WAV at 24 kHz."""
        timeline = self._narration_timeline(segments, gap_ms)
        if self._bgm_path is not None and len(timeline) > 0:
            timeline = timeline + self._bgm_bed(len(timeline))

        peak = float(np.max(np.abs(timeline))) if len(timeline) else 0.0
        if peak > 0.99:  # headroom guard: scale the mix instead of clipping
            timeline = timeline * (0.99 / peak)

        pcm = (np.clip(timeline, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes(pcm.tobytes())
        return output_path

    def _narration_timeline(
        self, segments: list[AudioSegmentMetadata], gap_ms: int
    ) -> np.ndarray:
        gap = np.zeros(int(SAMPLE_RATE * gap_ms / 1000), dtype=np.float32)
        parts: list[np.ndarray] = []
        for index, segment in enumerate(segments):
            samples, rate = read_audio(segment.file_path)
            samples = resample_audio(samples, rate, SAMPLE_RATE)
            gain = self._sfx_gain if segment.speaker_id == "SFX" else self._voice_gain
            if index:
                parts.append(gap)
            parts.append(samples.astype(np.float32) * gain)
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)

    def _bgm_bed(self, length: int) -> np.ndarray:
        samples, rate = read_audio(self._bgm_path)
        samples = resample_audio(samples, rate, SAMPLE_RATE).astype(np.float32)
        if len(samples) == 0:
            return np.zeros(length, dtype=np.float32)

        repeats = int(np.ceil(length / len(samples)))
        bed = np.tile(samples, repeats)[:length]

        fade = min(int(SAMPLE_RATE * self._bgm_fade_ms / 1000), length // 2)
        if fade > 0:
            envelope = np.ones(length, dtype=np.float32)
            envelope[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            envelope[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
            bed = bed * envelope
        logger.info(
            "BGM bed: %s looped %dx to %.1fs at gain %.2f",
            self._bgm_path.name, repeats, length / SAMPLE_RATE, self._bgm_gain,
        )
        return bed * self._bgm_gain
