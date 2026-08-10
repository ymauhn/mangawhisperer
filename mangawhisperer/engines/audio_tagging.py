"""Automatic tagging of uploaded sound effects via CLAP (zero-shot).

Uses ``laion/larger_clap_general`` through the transformers
``zero-shot-audio-classification`` pipeline: open-vocabulary labels, so
we classify directly against manga-relevant sounds ("sword clash",
"monster roar") and map the winners to the library's PT tags. ~776 MB
of weights, loads lazily, and ``release()`` frees the VRAM afterwards —
on an 8 GB GPU it must not linger next to XTTS.

Audio is decoded by our own readers (no ffmpeg dependency) and
resampled to CLAP's 48 kHz mono input.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from mangawhisperer.engines.sfx import read_audio, resample_audio

logger = logging.getLogger(__name__)

MODEL_ID = "laion/larger_clap_general"
CLAP_RATE = 48000

# English CLAP labels -> PT tags of the SFX library.
LABEL_TO_TAG: dict[str, str] = {
    "sword clash": "espada",
    "sword slash": "espada",
    "metal blade": "espada",
    "explosion": "explosao",
    "cannon blast": "explosao",
    "punch impact": "soco",
    "body hit": "soco",
    "monster roar": "monstro",
    "animal growl": "monstro",
    "fire crackling": "fogo",
    "wind blowing": "vento",
    "human scream": "grito",
    "footsteps": "passos",
    "thunder": "trovao",
    "door creaking": "porta",
    "rain falling": "chuva",
    "water splash": "agua",
    "horse galloping": "cavalo",
    "crowd murmur": "multidao",
    "bell tolling": "sino",
    "arrow whoosh": "flecha",
}


class ClapAudioTagger:
    """Zero-shot SFX tagger with lazy model lifecycle."""

    def __init__(self, model_id: str = MODEL_ID, classifier: Any = None) -> None:
        """
        Args:
            model_id: Hugging Face CLAP checkpoint.
            classifier: Injectable pipeline-compatible callable for tests.
        """
        self._model_id = model_id
        self._classifier = classifier

    def suggest(
        self, audio_path: Path, top_k: int = 3, threshold: float = 0.30
    ) -> list[tuple[str, str, float]]:
        """Suggest PT tags for one audio file.

        Returns:
            Up to ``top_k`` tuples ``(pt_tag, english_label, score)``,
            deduplicated by tag, best score first. Empty when nothing
            clears ``threshold``.
        """
        samples, rate = read_audio(audio_path)
        samples = resample_audio(samples, rate, CLAP_RATE)
        results = self._get_classifier()(
            samples,
            candidate_labels=list(LABEL_TO_TAG),
            hypothesis_template="This is a sound of {}.",
        )

        suggestions: list[tuple[str, str, float]] = []
        seen: set[str] = set()
        for item in sorted(results, key=lambda r: r["score"], reverse=True):
            tag = LABEL_TO_TAG.get(item["label"])
            score = float(item["score"])
            if tag is None or tag in seen or score < threshold:
                continue
            suggestions.append((tag, item["label"], score))
            seen.add(tag)
            if len(suggestions) >= top_k:
                break
        return suggestions

    def release(self) -> None:
        """Free the model (GPU memory is precious next to XTTS)."""
        if self._classifier is None:
            return
        self._classifier = None
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("CLAP tagger released from memory")

    def _get_classifier(self) -> Any:
        if self._classifier is None:
            import torch  # noqa: PLC0415 — heavy imports, deferred
            from transformers import pipeline  # noqa: PLC0415

            device = 0 if torch.cuda.is_available() else -1
            logger.info("Loading CLAP (%s) for audio tagging...", self._model_id)
            self._classifier = pipeline(
                "zero-shot-audio-classification", model=self._model_id, device=device
            )
        return self._classifier
