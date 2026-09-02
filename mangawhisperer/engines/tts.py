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

from mangawhisperer.constants import AUDIO_SAMPLE_RATE
from mangawhisperer.engines.casting import VoiceRegistry
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
"""Voices for uncast speakers whose voice profile is unknown."""

VOICE_BANK: dict[str, tuple[str, ...]] = {
    # XTTS v2 ships 58 studio voices. Gender follows the speaker names;
    # the age/creature groupings are judged by ear and meant to be tuned.
    "homem": (
        "Abrahan Mack", "Gilberto Mathias", "Suad Qasim", "Damien Black", "Zacharie Aimilios",
        "Filip Traverse", "Damjan Chapman", "Wulf Carlevaro", "Eugenio Mataracı", "Ferran Simen",
        "Xavier Hayasaka", "Luis Moray", "Marcos Rudaski", "Badr Odhiambo", "Viktor Eka",
        "Adde Michal", "Ige Behringer", "Kazuhiko Atallah", "Ludvig Milivoj", "Ilkin Urbano",
    ),
    "idoso": ("Torcull Diarmuid", "Kumar Dahl", "Dionisio Schuyler", "Viktor Menelaos", "Baldur Sanjin"),
    "menino": ("Andrew Chipper", "Royston Min", "Ilkin Urbano"),
    "mulher": (
        "Ana Florence", "Claribel Dervla", "Alison Dietlinde", "Annmarie Nele", "Asya Anara",
        "Gitta Nikolina", "Sofia Hellen", "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie",
        "Maja Ruoho", "Uta Obando", "Lidiya Szekeres", "Chandra MacFarland", "Szofi Granger",
        "Camilla Holmström", "Lilya Stainthorpe", "Zofija Kendrick", "Narelle Moon",
        "Alexandra Hisakawa", "Alma María",
    ),
    "idosa": ("Henriette Usha", "Brenda Stern", "Rosemary Okafor", "Barbora MacLean"),
    "menina": ("Daisy Studious", "Gracie Wise", "Tammie Ema", "Nova Hogarth"),
    "criatura": ("Baldur Sanjin", "Torcull Diarmuid", "Damien Black"),
}
"""Voice profile (``models.VOICE_PROFILES``) -> XTTS voices to cast from."""

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


def is_pronounceable(text: str) -> bool:
    """False for blocks with nothing to voice — a silent "......" bubble,
    a lone "!" — which the TTS models turn into zero-length audio (XTTS
    even crashes computing its real-time factor). Such blocks become a
    short beat of silence instead."""
    return any(char.isalnum() for char in normalize_for_tts(text))


SILENT_BEAT_MS = 450
"""Pause rendered for an unpronounceable block: the reader still "hears"
that the character stayed silent."""


def write_silence(path: Path, duration_ms: int, sample_rate: int = AUDIO_SAMPLE_RATE) -> None:
    """Write ``duration_ms`` of 16-bit mono silence as a valid WAV."""
    frames = int(sample_rate * duration_ms / 1000)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * frames)


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
        voice_bank: Mapping[str, Sequence[str]] | None = None,
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
        self._registry: VoiceRegistry[str] = VoiceRegistry(
            self._cast, voice_bank if voice_bank is not None else VOICE_BANK, self._fallback
        )

    @property
    def fingerprint(self) -> str:
        """Checkpoint identity: real XTTS output, per model + language + pace."""
        extras = ",".join(f"{k}={v}" for k, v in sorted(self._extra_kwargs.items()))
        cast = f":cast={self._registry.digest()}" if self._registry.assignments else ""
        return f"xtts:{self.MODEL_NAME}:{self._language}:speed={self._speed}:{extras or 'default'}{cast}"

    def synthesize(self, block: ContextualizedBlock, output_path: Path) -> AudioSegmentMetadata:
        """Render one block to WAV with the speaker's assigned voice; a
        block with nothing to voice becomes a short beat of silence."""
        if is_pronounceable(block.text):
            self._get_synthesizer().tts_to_file(
                text=normalize_for_tts(block.text),
                speaker=self.voice_for(block.speaker_id, block.voice),
                language=self._language,
                speed=self._speed,
                file_path=str(output_path),
                **self._extra_kwargs,
            )
        else:
            logger.info("Bloco sem texto pronunciável (%r): pausa de %d ms", block.text, SILENT_BEAT_MS)
            write_silence(output_path, SILENT_BEAT_MS, AUDIO_SAMPLE_RATE)
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

    def assign_voices(self, profiles: Mapping[str, str | None], registry_path: Path | None = None) -> dict[str, str]:
        """Cast every speaker once (curated cast, then the profile's bank,
        avoiding voices already taken) and persist to ``registry_path``
        so the same character keeps the same voice across runs."""
        assigned = self._registry.assign(profiles, registry_path)
        logger.info("Elenco de vozes: %s", ", ".join(f"{s}={v}" for s, v in sorted(assigned.items())))
        return assigned

    def voice_for(self, speaker_id: str, profile: str | None = None) -> str:
        """Resolve a speaker id to an XTTS voice name.

        Cast members get their curated voice; an assigned speaker keeps
        the registered one; anyone else gets a deterministic voice from
        the bank of their voice profile (or the fallback pool).
        """
        return self._registry.voice_for(speaker_id, profile)

    def _get_synthesizer(self) -> Any:
        if self._synthesizer is None:
            os.environ.setdefault("COQUI_TOS_AGREED", "1")
            import torch  # noqa: PLC0415 — heavy imports, deferred until needed
            from TTS.api import TTS  # noqa: PLC0415

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("Loading XTTSv2 on %s (first run downloads ~1.8 GB)", device)
            self._synthesizer = TTS(self.MODEL_NAME).to(device)
        return self._synthesizer
