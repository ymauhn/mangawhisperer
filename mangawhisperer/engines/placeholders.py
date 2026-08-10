"""Stdlib-only placeholder engines for the not-yet-real stages.

These let the pipeline run end-to-end on real pages while the VLM and
XTTSv2 integrations are pending: the VLM placeholder does no
diarization, and the TTS placeholder renders timed silence — but the
output is a valid, playable WAV with one silent beat per script block,
so the audio plumbing is exercised for real.
"""

from __future__ import annotations

import wave
from pathlib import Path

from mangawhisperer.interfaces import (
    AudioStitcher,
    Image,
    MultiSpeakerTTSEngine,
    VisionLanguageEngine,
)
from mangawhisperer.models import AudioSegmentMetadata, ContextualizedBlock, SpeechBubble

SAMPLE_RATE = 22050
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM
CHANNELS = 1


def _write_silence(path: Path, duration_ms: int) -> None:
    """Write ``duration_ms`` of 16-bit mono silence as a valid WAV."""
    frame_count = int(SAMPLE_RATE * duration_ms / 1000)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"\x00" * (frame_count * SAMPLE_WIDTH_BYTES * CHANNELS))


class PassthroughVLM(VisionLanguageEngine):
    """No-model scriptwriter: every non-empty bubble becomes a speech
    block for a single default speaker; no action descriptions."""

    def __init__(self, default_speaker: str = "Desconhecido") -> None:
        self._default_speaker = default_speaker

    def contextualize(
        self, panel_image: Image, bubbles: list[SpeechBubble]
    ) -> list[ContextualizedBlock]:
        return [
            ContextualizedBlock(text=b.text, speaker_id=self._default_speaker, is_speech=True)
            for b in bubbles
            if b.text.strip()
        ]


class SilentTTSEngine(MultiSpeakerTTSEngine):
    """Renders each block as silence whose length tracks the text length,
    so the stitched track has a realistic narration rhythm."""

    MS_PER_CHAR = 60
    MIN_DURATION_MS = 300

    def __init__(self) -> None:
        self._block_index = 0

    def synthesize(self, block: ContextualizedBlock, output_path: Path) -> AudioSegmentMetadata:
        duration_ms = max(self.MIN_DURATION_MS, self.MS_PER_CHAR * len(block.text))
        _write_silence(output_path, duration_ms)
        metadata = AudioSegmentMetadata(
            file_path=output_path,
            speaker_id=block.speaker_id,
            duration_ms=duration_ms,
            block_index=self._block_index,
        )
        self._block_index += 1
        return metadata


class WaveFileStitcher(AudioStitcher):
    """Concatenates WAV segments with silence gaps using only stdlib
    ``wave``. All segments must share the first segment's params."""

    def stitch(
        self,
        segments: list[AudioSegmentMetadata],
        output_path: Path,
        gap_ms: int = 350,
    ) -> Path:
        if not segments:
            _write_silence(output_path, 0)
            return output_path

        with wave.open(str(segments[0].file_path), "rb") as first:
            params = first.getparams()
        gap_frames = int(params.framerate * gap_ms / 1000)
        gap_bytes = b"\x00" * (gap_frames * params.sampwidth * params.nchannels)

        with wave.open(str(output_path), "wb") as out:
            out.setparams(params)
            for i, segment in enumerate(segments):
                if i > 0:
                    out.writeframes(gap_bytes)
                with wave.open(str(segment.file_path), "rb") as src:
                    if src.getparams()[:3] != params[:3]:
                        raise ValueError(
                            f"Segment {segment.file_path} has mismatched audio params"
                        )
                    out.writeframes(src.readframes(src.getnframes()))
        return output_path
