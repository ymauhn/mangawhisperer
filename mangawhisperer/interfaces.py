"""Abstract interfaces for every MangaWhisperer pipeline stage.

Each ABC is a deployment-agnostic contract: an implementation may run a
local model (comic-text-detector, EasyOCR, XTTSv2), call a cloud API, or
be a test mock — the orchestrator cannot tell the difference. Batching,
retries and device management are implementation concerns and must stay
hidden behind these signatures.

Images cross these boundaries as ``numpy.ndarray`` (H, W, C) uint8
arrays — the native currency of OpenCV and EasyOCR — rather than
framework-specific tensors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mangawhisperer.models import (
    AudioSegmentMetadata,
    BoundingBox,
    ContextualizedBlock,
    PanelData,
    SpeechBubble,
)

Image = NDArray[np.uint8]
"""An (H, W, C) uint8 image array."""


class PDFPageExtractor(ABC):
    """Rasterizes a manga PDF into one image file per page.

    Real implementation: PyMuPDF/pdf2image at print resolution.
    """

    @abstractmethod
    def extract_pages(self, pdf_path: Path, output_dir: Path) -> list[Path]:
        """Render every page of ``pdf_path`` into ``output_dir``.

        Args:
            pdf_path: Source manga volume PDF.
            output_dir: Existing directory to write page images into.

        Returns:
            Page image paths in ascending page order (index 0 = page 1).
        """


class MangaLayoutParser(ABC):
    """Isolates panel and speech-bubble regions in reading order.

    Both methods contractually return boxes already sorted by the
    best-effort manga reading heuristic: right-to-left, top-to-bottom.
    Real implementation: comic-text-detector for bubble masks combined
    with Manga-Panel-Extractor-style panel splitting.
    """

    @abstractmethod
    def extract_panels(self, page_image: Image) -> list[BoundingBox]:
        """Detect panel regions on a full page.

        Args:
            page_image: Full page as an (H, W, C) uint8 array.

        Returns:
            Panel boxes normalized to the page, in reading order.
        """

    @abstractmethod
    def extract_bubbles(self, panel_image: Image) -> list[BoundingBox]:
        """Detect speech-bubble regions inside a single panel.

        Args:
            panel_image: Cropped panel as an (H, W, C) uint8 array.

        Returns:
            Bubble boxes normalized to the panel, in reading order.
        """


class PortugueseOCREngine(ABC):
    """Extracts PT-BR text from a cropped bubble region.

    Real implementation: EasyOCR with the ``pt`` recognizer, which
    handles Latin diacritics (á, ç, õ) that Japanese-centric manga OCR
    models garble.
    """

    @abstractmethod
    def recognize(self, region_image: Image) -> str:
        """Run OCR on one bubble crop.

        Args:
            region_image: Bubble region as an (H, W, C) uint8 array.

        Returns:
            The recognized text, whitespace-normalized. Empty string if
            the region contains no legible text.
        """


class VisionLanguageEngine(ABC):
    """Turns a panel image plus its OCR text into a narratable script.

    The VLM acts as scriptwriter: it attributes each bubble to a speaker
    (diarization) and interleaves short PT-BR action descriptions for
    purely visual beats. May be backed by a local VLM or a cloud API.
    """

    @abstractmethod
    def contextualize(
        self, panel_image: Image, bubbles: list[SpeechBubble]
    ) -> list[ContextualizedBlock]:
        """Produce the ordered script for one panel.

        Args:
            panel_image: Cropped panel as an (H, W, C) uint8 array.
            bubbles: OCR results for the panel, in reading order.

        Returns:
            Script blocks in narration order. Real dialogue appears as
            speech blocks (``is_speech=True``) preserving bubble order;
            OCR noise (onomatopoeia fragments, garbled text) may be
            dropped or lightly corrected. Action descriptions
            (``is_speech=False``, speaker ``"Narrator"``) may be
            interleaved anywhere.
        """


class ScriptReviewer(ABC):
    """Second-pass quality layer over the scriptwriter's output.

    Reviews the whole volume at once (a global view the per-panel
    writer never has): speaker-label consistency, read-aloud text
    normalization, continuity, and SFX sanity. Implementations must be
    conservative — never invent dialogue, and return the input
    unchanged when unsure.
    """

    @abstractmethod
    def review(self, panels: list[PanelData]) -> list[PanelData]:
        """Return the reviewed script (same panels, corrected blocks).

        Args:
            panels: The full script in reading order.

        Returns:
            Panels with the same structure; only block contents may
            change. On any internal failure, the original list.
        """


class MultiSpeakerTTSEngine(ABC):
    """Synthesizes PT-BR speech, mapping speaker ids to voice profiles.

    Real implementation: Coqui XTTSv2 voice cloning with a curated cast
    config (character -> reference clip) and a deterministic fallback
    voice pool for uncast speakers.
    """

    @abstractmethod
    def synthesize(self, block: ContextualizedBlock, output_path: Path) -> AudioSegmentMetadata:
        """Render one script block to an audio file.

        Args:
            block: The text and speaker to synthesize.
            output_path: Exact file path to write the audio to.

        Returns:
            Metadata for the written segment; ``file_path`` must equal
            ``output_path`` and ``duration_ms`` must reflect the
            rendered audio length.
        """


class AudioStitcher(ABC):
    """Concatenates audio segments into the final immersive track.

    Real implementation: pydub concatenation with silence padding.
    """

    @abstractmethod
    def stitch(
        self,
        segments: list[AudioSegmentMetadata],
        output_path: Path,
        gap_ms: int = 350,
    ) -> Path:
        """Join segments in order, separated by silence.

        Args:
            segments: Segments in narration order.
            output_path: Exact file path to write the final audio to.
            gap_ms: Milliseconds of silence inserted between segments.

        Returns:
            ``output_path``, once the file has been written.
        """
