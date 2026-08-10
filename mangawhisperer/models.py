"""Pydantic v2 data contracts for the MangaWhisperer pipeline.

These models are the interchange format between every pipeline stage
(layout parsing -> OCR -> VLM scripting -> TTS -> stitching). They are
also the on-disk checkpoint format: each stage persists its output as
JSON under the per-volume workspace so long runs can be inspected,
corrected, and eventually resumed.

All models are immutable (``frozen=True``) except :class:`PanelData`,
whose ``blocks`` list is filled progressively as OCR and VLM stages run.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(BaseModel):
    """An axis-aligned rectangle in normalized coordinates.

    Coordinates are fractions of the enclosing image's dimensions, in
    the range ``[0.0, 1.0]``, with the origin at the top-left corner.
    Normalized coordinates make boxes resolution-independent: the same
    box is valid against a thumbnail or a print-quality scan.
    """

    model_config = ConfigDict(frozen=True)

    x_min: float = Field(ge=0.0, le=1.0, description="Left edge, fraction of image width.")
    y_min: float = Field(ge=0.0, le=1.0, description="Top edge, fraction of image height.")
    x_max: float = Field(ge=0.0, le=1.0, description="Right edge, fraction of image width.")
    y_max: float = Field(ge=0.0, le=1.0, description="Bottom edge, fraction of image height.")

    @model_validator(mode="after")
    def _check_extents(self) -> "BoundingBox":
        """Reject degenerate or inverted boxes."""
        if self.x_max <= self.x_min:
            raise ValueError(f"x_max ({self.x_max}) must be greater than x_min ({self.x_min})")
        if self.y_max <= self.y_min:
            raise ValueError(f"y_max ({self.y_max}) must be greater than y_min ({self.y_min})")
        return self

    @property
    def width(self) -> float:
        """Normalized width of the box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Normalized height of the box."""
        return self.y_max - self.y_min

    @property
    def center(self) -> tuple[float, float]:
        """Normalized ``(x, y)`` center point, used by reading-order sorts."""
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)

    def to_absolute(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        """Convert to integer pixel coordinates ``(x0, y0, x1, y1)``.

        Args:
            image_width: Width in pixels of the image this box refers to.
            image_height: Height in pixels of the image this box refers to.

        Returns:
            Pixel coordinates suitable for numpy slicing:
            ``image[y0:y1, x0:x1]``.
        """
        return (
            int(round(self.x_min * image_width)),
            int(round(self.y_min * image_height)),
            int(round(self.x_max * image_width)),
            int(round(self.y_max * image_height)),
        )


class SpeechBubble(BaseModel):
    """A single segmented speech bubble with its OCR-extracted text.

    Produced by the OCR stage from the bubble regions the layout parser
    detected; consumed by the VLM stage for speaker attribution. The
    ``bbox`` is normalized against the *panel* image the bubble was
    cropped from, not the full page.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(description="Raw OCR text extracted from the bubble region.")
    bbox: BoundingBox = Field(description="Bubble location, normalized to its parent panel.")
    language: str = Field(default="pt-BR", description="BCP-47 tag of the bubble's text.")


class ContextualizedBlock(BaseModel):
    """A narratable unit of script emitted by the VLM stage.

    Either a speech bubble attributed to a character, or an
    AI-generated action description voiced by the narrator.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, description="The text to be synthesized.")
    speaker_id: str = Field(
        min_length=1,
        description='Character name (e.g. "Guts") or "Narrator" for action descriptions.',
    )
    is_speech: bool = Field(
        description="True for character dialogue from a bubble; False for an "
        "AI-generated action/scene description."
    )
    sfx: str | None = Field(
        default=None,
        description="Optional sound-effect tag from the SFX library, played "
        "right before this block (e.g. 'espada', 'explosao').",
    )


class PanelData(BaseModel):
    """A single comic panel and its chronologically ordered script.

    The position of each block in ``blocks`` *is* its reading order
    (best-effort right-to-left, top-to-bottom heuristic) — there is no
    separate ordering field. Mutable so the pipeline can attach blocks
    after construction.
    """

    image_path: Path = Field(description="Path of the cropped panel image inside the workspace.")
    bbox: BoundingBox = Field(description="Panel location, normalized to its parent page.")
    page_number: int = Field(
        ge=1,
        description="1-based position within the processed page range (equals the "
        "source PDF page number only when processing starts at page 1).",
    )
    panel_index: int = Field(ge=0, description="0-based reading-order index of the panel on its page.")
    blocks: list[ContextualizedBlock] = Field(
        default_factory=list,
        description="Script units in reading order (list position = narration order).",
    )


class AudioSegmentMetadata(BaseModel):
    """Bookkeeping for one synthesized audio cut.

    Produced by the TTS stage, consumed by the stitcher and persisted as
    the ``segments.json`` checkpoint so a run can be audited (which
    voice said what, and for how long) without re-listening.
    """

    model_config = ConfigDict(frozen=True)

    file_path: Path = Field(description="Path of the synthesized audio file.")
    speaker_id: str = Field(min_length=1, description="Voice profile this segment was rendered with.")
    duration_ms: int = Field(ge=0, description="Segment duration in milliseconds.")
    block_index: int = Field(ge=0, description="0-based global narration index across the whole volume.")
