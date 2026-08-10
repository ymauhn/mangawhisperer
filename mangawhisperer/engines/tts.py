"""Real multi-speaker TTS via Coqui XTTSv2.

Implements the curated-cast + deterministic-fallback voice decision:
main characters map to hand-picked XTTSv2 studio voices; any other
speaker id hashes into a small fallback pool, so the same minor
character always gets the same voice across all 42 volumes.

The XTTS model (~1.8 GB) downloads on first use and needs a GPU to run
at a sane speed. Synthesis writes 24 kHz mono WAV files, which the
stdlib ``WaveFileStitcher`` concatenates without conversion.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

from mangawhisperer.interfaces import MultiSpeakerTTSEngine
from mangawhisperer.models import AudioSegmentMetadata, ContextualizedBlock

logger = logging.getLogger(__name__)

DEFAULT_CAST_VOICES: dict[str, str] = {
    "Guts": "Craig Gutsy",
    "Griffith": "Viktor Menelaos",
    "Casca": "Ana Florence",
    "Puck": "Andrew Chipper",
    "Judeau": "Ludvig Milivoj",
    "Corkus": "Dionisio Schuyler",
    "Pippin": "Kumar Dahl",
    "Rickert": "Ilkin Urbano",
    "Zodd": "Torcull Diarmuid",
    "Narrator": "Aaron Dreschner",
    "Desconhecido": "Abrahan Mack",
    # Rótulos descritivos comuns que o roteirista usa para não-elenco:
    "Criatura": "Baldur Sanjin",
    "Monstro": "Baldur Sanjin",
    "Soldado": "Gilberto Mathias",
    "Comandante": "Kazuhiko Atallah",
}
"""Curated cast: character id -> XTTSv2 built-in studio voice."""

FALLBACK_VOICE_POOL: tuple[str, ...] = (
    "Abrahan Mack",
    "Baldur Sanjin",
    "Gilberto Mathias",
    "Suad Qasim",
    "Brenda Stern",
    "Henriette Usha",
)
"""Voices assigned deterministically to uncast speakers."""

_STRIP_CHARS = str.maketrans("", "", '"“”*«»[]')


def normalize_for_tts(text: str) -> str:
    """Clean script text for XTTS so punctuation shapes prosody instead
    of being vocalized.

    Keeps ``!`` and ``?`` (they drive intonation), turns ellipses into a
    comma pause (ellipses are a known XTTS artifact trigger), collapses
    repeated punctuation, and drops the sentence-final period — XTTS
    sometimes reads a trailing "ponto" aloud.
    """
    cleaned = text.translate(_STRIP_CHARS)
    cleaned = cleaned.replace("…", "...")
    cleaned = re.sub(r"\s*\.{3,}\s*", ", ", cleaned)   # "..." -> breve pausa
    cleaned = re.sub(r"([!?])\1+", r"\1", cleaned)     # "!!!" -> "!"
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\s*,\s*$", "", cleaned)         # vírgula solta no fim
    cleaned = cleaned.rstrip(".").strip()              # ponto final não se lê
    return cleaned or text


class XTTSEngine(MultiSpeakerTTSEngine):
    """Coqui XTTSv2 synthesis with consistent per-character voices.

    The underlying model loads lazily on first synthesis (first-ever use
    also downloads the weights, gated by Coqui's CPML license — this
    engine auto-accepts it via ``COQUI_TOS_AGREED`` for non-commercial
    accessibility use).
    """

    MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

    def __init__(
        self,
        cast_voices: Mapping[str, str] | None = None,
        fallback_voices: Sequence[str] = FALLBACK_VOICE_POOL,
        language: str = "pt",
        speed: float = 1.0,
        extra_synthesis_kwargs: Mapping[str, float] | None = None,
        device: str | None = None,
        synthesizer: Any = None,
    ) -> None:
        """
        Args:
            cast_voices: Character id -> XTTS voice name; defaults to
                the Berserk cast in :data:`DEFAULT_CAST_VOICES`.
            fallback_voices: Pool for speakers not in the cast; chosen
                by a stable hash of the speaker id.
            language: XTTS language code (``pt`` covers PT-BR).
            device: ``"cuda"``/``"cpu"``; ``None`` auto-detects.
            synthesizer: Injectable object exposing ``tts_to_file`` for
                tests; ``None`` loads the real XTTS model lazily.
        """
        if not fallback_voices:
            raise ValueError("fallback_voices must not be empty")
        self._cast = dict(DEFAULT_CAST_VOICES if cast_voices is None else cast_voices)
        self._fallback = tuple(fallback_voices)
        self._language = language
        self._speed = speed
        # Expressiveness knobs forwarded to XTTS inference (idiap fork
        # forwards kwargs end-to-end). Community expressive recipe:
        # {"temperature": 0.75, "repetition_penalty": 4.0, "top_p": 0.85}.
        self._extra_kwargs = dict(extra_synthesis_kwargs or {})
        self._device = device
        self._synthesizer = synthesizer
        self._block_index = 0

    @property
    def fingerprint(self) -> str:
        """Checkpoint identity: real XTTS output, per model + language + pace."""
        extras = ",".join(f"{k}={v}" for k, v in sorted(self._extra_kwargs.items()))
        return f"xtts:{self.MODEL_NAME}:{self._language}:speed={self._speed}:{extras or 'default'}"

    def synthesize(self, block: ContextualizedBlock, output_path: Path) -> AudioSegmentMetadata:
        """Render one block to WAV with the speaker's assigned voice."""
        self._get_synthesizer().tts_to_file(
            text=normalize_for_tts(block.text),
            speaker=self.voice_for(block.speaker_id),
            language=self._language,
            speed=self._speed,
            file_path=str(output_path),
            **self._extra_kwargs,
        )
        with wave.open(str(output_path), "rb") as wav:
            duration_ms = round(wav.getnframes() * 1000 / wav.getframerate())

        metadata = AudioSegmentMetadata(
            file_path=output_path,
            speaker_id=block.speaker_id,
            duration_ms=duration_ms,
            block_index=self._block_index,
        )
        self._block_index += 1
        return metadata

    def configure(
        self,
        speed: float | None = None,
        extra_synthesis_kwargs: Mapping[str, float] | None = None,
    ) -> None:
        """Adjust delivery (pace/expressiveness) without reloading the
        model — used by the UI when switching narration styles."""
        if speed is not None:
            self._speed = speed
        if extra_synthesis_kwargs is not None:
            self._extra_kwargs = dict(extra_synthesis_kwargs)

    def voice_for(self, speaker_id: str) -> str:
        """Resolve a speaker id to an XTTS voice name.

        Cast members get their curated voice; anyone else hashes into
        the fallback pool (stable across runs and Python processes).
        """
        if speaker_id in self._cast:
            return self._cast[speaker_id]
        digest = hashlib.sha1(speaker_id.encode("utf-8")).digest()
        return self._fallback[digest[0] % len(self._fallback)]

    def _get_synthesizer(self) -> Any:
        if self._synthesizer is None:
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            import torch  # noqa: PLC0415 — heavy imports, deferred until needed
            from TTS.api import TTS  # noqa: PLC0415

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading XTTSv2 on %s (first run downloads ~1.8 GB)", device)
            self._synthesizer = TTS(self.MODEL_NAME).to(device)
        return self._synthesizer
