"""Edge-TTS engine: zero-VRAM pt-BR narration via Microsoft neural voices.

Fallback/alternative to XTTS when the GPU is busy or quality comparison
is wanted. Only three pt-BR voices exist (Antonio, Francisca, Thalita),
so cast distinction comes from per-character rate/pitch offsets.

Honest caveats: this uses the reverse-engineered Edge "Read Aloud"
endpoint (free, but a ToS gray area that can break without notice) and
requires network at synthesis time. Do not build the project's future
on it; it is a pragmatic option, not the foundation.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import wave
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from mangawhisperer.constants import AUDIO_SAMPLE_RATE as TARGET_RATE
from mangawhisperer.engines.casting import VoiceRegistry
from mangawhisperer.engines.sfx import read_audio, resample_audio
from mangawhisperer.engines.tts import SILENT_BEAT_MS, is_pronounceable, normalize_for_tts, write_silence
from mangawhisperer.interfaces import MultiSpeakerTTSEngine
from mangawhisperer.models import AudioSegmentMetadata, ContextualizedBlock

logger = logging.getLogger(__name__)

VoiceSpec = tuple[str, int, int]  # (voice id, rate offset %, pitch offset Hz)

DEFAULT_EDGE_CAST: dict[str, VoiceSpec] = {
    "Narrator": ("pt-BR-AntonioNeural", 0, 0),
    "Guts": ("pt-BR-AntonioNeural", -8, -15),
    "Griffith": ("pt-BR-AntonioNeural", 4, 12),
    "Casca": ("pt-BR-FranciscaNeural", 0, 0),
    "Puck": ("pt-BR-ThalitaMultilingualNeural", 12, 18),
    "Judeau": ("pt-BR-AntonioNeural", 2, 6),
    "Corkus": ("pt-BR-AntonioNeural", 6, -4),
    "Zodd": ("pt-BR-AntonioNeural", -12, -25),
    "Criatura": ("pt-BR-AntonioNeural", -14, -22),
    "Monstro": ("pt-BR-AntonioNeural", -14, -22),
    "Desconhecido": ("pt-BR-AntonioNeural", 0, -6),
}

EDGE_VOICE_BANK: dict[str, tuple[VoiceSpec, ...]] = {
    "homem": (("pt-BR-AntonioNeural", -4, -10), ("pt-BR-AntonioNeural", 8, 8), ("pt-BR-AntonioNeural", 0, -6)),
    "idoso": (("pt-BR-AntonioNeural", -14, -22), ("pt-BR-AntonioNeural", -10, -16)),
    "menino": (("pt-BR-AntonioNeural", 12, 20), ("pt-BR-ThalitaMultilingualNeural", 6, -4)),
    "mulher": (("pt-BR-FranciscaNeural", 0, 0), ("pt-BR-FranciscaNeural", -4, -6), ("pt-BR-ThalitaMultilingualNeural", 0, 0)),
    "idosa": (("pt-BR-FranciscaNeural", -12, -14),),
    "menina": (("pt-BR-ThalitaMultilingualNeural", 10, 14), ("pt-BR-FranciscaNeural", 8, 12)),
    "criatura": (("pt-BR-AntonioNeural", -16, -28),),
}
"""Voice profile -> Edge voices (three pt-BR voices, so profiles are offsets)."""

FALLBACK_EDGE_POOL: tuple[VoiceSpec, ...] = (
    ("pt-BR-AntonioNeural", -4, -10),
    ("pt-BR-AntonioNeural", 8, 8),
    ("pt-BR-FranciscaNeural", -4, -6),
    ("pt-BR-FranciscaNeural", 6, 10),
    ("pt-BR-ThalitaMultilingualNeural", 0, 0),
)


class EdgeTTSEngine(MultiSpeakerTTSEngine):
    """Multi-speaker synthesis over the ``edge-tts`` package."""

    def __init__(
        self,
        cast_voices: Mapping[str, VoiceSpec] | None = None,
        fallback_voices: tuple[VoiceSpec, ...] = FALLBACK_EDGE_POOL,
        speed: float = 1.0,
        extra_synthesis_kwargs: Mapping[str, float] | None = None,  # XTTS-only; ignored
        communicate_factory: Callable[..., Any] | None = None,
    ) -> None:
        """
        Args:
            cast_voices: Character -> (voice, rate%, pitchHz) mapping.
            fallback_voices: Deterministic pool for uncast speakers.
            speed: Global pace multiplier (1.06 -> +6% on every voice).
            extra_synthesis_kwargs: Accepted for interface parity with
                :class:`XTTSEngine`; Edge has no such knobs.
            communicate_factory: Injectable ``edge_tts.Communicate``
                replacement for tests.
        """
        if not fallback_voices:
            raise ValueError("fallback_voices must not be empty")
        self._cast = dict(DEFAULT_EDGE_CAST if cast_voices is None else cast_voices)
        self._fallback = tuple(fallback_voices)
        self._speed = speed
        self._communicate_factory = communicate_factory
        self._block_index = 0
        self._registry: VoiceRegistry[VoiceSpec] = VoiceRegistry(self._cast, EDGE_VOICE_BANK, self._fallback)

    @property
    def fingerprint(self) -> str:
        cast_digest = hashlib.sha1(repr(sorted(self._cast.items())).encode()).hexdigest()[:8]
        assigned = f":cast={self._registry.digest()}" if self._registry.assignments else ""
        return f"edge-tts:{cast_digest}:speed={self._speed}{assigned}"

    def configure(self, speed: float | None = None, extra_synthesis_kwargs=None) -> None:
        """Interface parity with :class:`XTTSEngine`."""
        if speed is not None:
            self._speed = speed

    def synthesize(self, block: ContextualizedBlock, output_path: Path) -> AudioSegmentMetadata:
        """Render one block via Edge and convert to pipeline WAV; a block
        with nothing to voice becomes a short beat of silence."""
        if not is_pronounceable(block.text):
            logger.info("Bloco sem texto pronunciável (%r): pausa de %d ms", block.text, SILENT_BEAT_MS)
            write_silence(output_path, SILENT_BEAT_MS, TARGET_RATE)
            metadata = AudioSegmentMetadata(
                file_path=output_path, speaker_id=block.speaker_id,
                duration_ms=SILENT_BEAT_MS, block_index=self._block_index,
            )
            self._block_index += 1
            return metadata
        voice, rate, pitch = self.voice_for(block.speaker_id, block.voice)
        rate += round((self._speed - 1.0) * 100)
        temp_mp3 = output_path.with_suffix(".edge.mp3")
        communicate = self._get_factory()(
            normalize_for_tts(block.text),
            voice,
            rate=f"{rate:+d}%",
            pitch=f"{pitch:+d}Hz",
        )
        asyncio.run(communicate.save(str(temp_mp3)))

        samples, source_rate = read_audio(temp_mp3)
        temp_mp3.unlink(missing_ok=True)
        samples = resample_audio(samples, source_rate, TARGET_RATE)
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        with wave.open(str(output_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(TARGET_RATE)
            wav.writeframes(pcm.tobytes())

        metadata = AudioSegmentMetadata(
            file_path=output_path,
            speaker_id=block.speaker_id,
            duration_ms=round(len(pcm) * 1000 / TARGET_RATE),
            block_index=self._block_index,
        )
        self._block_index += 1
        return metadata

    def assign_voices(self, profiles: Mapping[str, str | None], registry_path: Path | None = None) -> dict[str, VoiceSpec]:
        """Same contract as :meth:`XTTSEngine.assign_voices`."""
        return self._registry.assign(profiles, registry_path)

    def voice_for(self, speaker_id: str, profile: str | None = None) -> VoiceSpec:
        """Cast voice, the registered one, or a stable pick from the profile's bank."""
        return self._registry.voice_for(speaker_id, profile)

    def _get_factory(self) -> Callable[..., Any]:
        if self._communicate_factory is None:
            import edge_tts  # noqa: PLC0415 — deferred so tests need no package

            self._communicate_factory = edge_tts.Communicate
        return self._communicate_factory
