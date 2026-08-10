"""Tests for the real input-stage engines and stdlib placeholders.

Fast by default: synthetic images and generated mini-PDFs only. Tests
that download real model weights (EasyOCR) are marked ``realmodels``
and deselected unless you run ``pytest -m realmodels``.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from mangawhisperer.models import BoundingBox, ContextualizedBlock, SpeechBubble

cv2 = pytest.importorskip("cv2")
pymupdf = pytest.importorskip("pymupdf")

from mangawhisperer.engines.layout import ClassicalLayoutParser, sort_reading_order  # noqa: E402
from mangawhisperer.engines.pdf import PyMuPDFPageExtractor  # noqa: E402
from mangawhisperer.engines.placeholders import (  # noqa: E402
    SAMPLE_RATE,
    PassthroughVLM,
    SilentTTSEngine,
    WaveFileStitcher,
)

# ---------------------------------------------------------------------------
# PyMuPDFPageExtractor
# ---------------------------------------------------------------------------


@pytest.fixture()
def mini_pdf(tmp_path: Path) -> Path:
    """A generated 3-page PDF (300x400 pt pages with a line of text)."""
    doc = pymupdf.open()
    for label in ("PÁGINA UM", "PÁGINA DOIS", "PÁGINA TRÊS"):
        page = doc.new_page(width=300, height=400)
        page.insert_text((72, 72), label, fontsize=20)
    pdf = tmp_path / "mini.pdf"
    doc.save(pdf)
    doc.close()
    return pdf


class TestPyMuPDFPageExtractor:
    def test_extracts_all_pages_in_order(self, mini_pdf: Path, tmp_path: Path) -> None:
        out = tmp_path / "pages"
        out.mkdir()
        paths = PyMuPDFPageExtractor(dpi=100).extract_pages(mini_pdf, out)

        assert [p.name for p in paths] == ["page_001.png", "page_002.png", "page_003.png"]
        assert all(p.is_file() and p.stat().st_size > 0 for p in paths)

    def test_renders_at_requested_dpi(self, mini_pdf: Path, tmp_path: Path) -> None:
        out = tmp_path / "pages"
        out.mkdir()
        paths = PyMuPDFPageExtractor(dpi=144).extract_pages(mini_pdf, out)

        image = cv2.imread(str(paths[0]))
        # 300x400 pt at 144 dpi -> 2x scale over 72 dpi baseline.
        assert image.shape[:2] == (800, 600)

    def test_page_range_selection(self, mini_pdf: Path, tmp_path: Path) -> None:
        out = tmp_path / "pages"
        out.mkdir()
        paths = PyMuPDFPageExtractor(dpi=72, first_page=2, max_pages=1).extract_pages(mini_pdf, out)

        assert [p.name for p in paths] == ["page_002.png"]

    def test_rejects_invalid_configuration(self) -> None:
        with pytest.raises(ValueError):
            PyMuPDFPageExtractor(dpi=0)
        with pytest.raises(ValueError):
            PyMuPDFPageExtractor(first_page=0)


# ---------------------------------------------------------------------------
# Reading-order sort
# ---------------------------------------------------------------------------


def test_sort_reading_order_is_rtl_top_to_bottom() -> None:
    top_right = BoundingBox(x_min=0.55, y_min=0.05, x_max=0.95, y_max=0.45)
    top_left = BoundingBox(x_min=0.05, y_min=0.08, x_max=0.45, y_max=0.48)
    bottom = BoundingBox(x_min=0.05, y_min=0.55, x_max=0.95, y_max=0.95)

    assert sort_reading_order([bottom, top_left, top_right]) == [top_right, top_left, bottom]


# ---------------------------------------------------------------------------
# ClassicalLayoutParser
# ---------------------------------------------------------------------------


def _synthetic_page() -> np.ndarray:
    """White page with two black-bordered panels side by side."""
    page = np.full((600, 800, 3), 255, dtype=np.uint8)
    for x0, x1 in ((430, 760), (40, 370)):  # right panel, left panel
        cv2.rectangle(page, (x0, 40), (x1, 560), (0, 0, 0), thickness=6)
        cv2.line(page, (x0 + 30, 100), (x1 - 30, 500), (0, 0, 0), thickness=3)
    return page


def _synthetic_panel_with_bubble() -> np.ndarray:
    """Dark-toned panel with one white, black-outlined bubble holding text."""
    panel = np.full((400, 400, 3), 90, dtype=np.uint8)
    cv2.ellipse(panel, (200, 150), (120, 70), 0, 0, 360, (255, 255, 255), thickness=-1)
    cv2.ellipse(panel, (200, 150), (120, 70), 0, 0, 360, (0, 0, 0), thickness=3)
    for row, y in enumerate(range(125, 190, 22)):
        cv2.putText(panel, "OLA MUNDO", (130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return panel


class TestClassicalLayoutParser:
    def test_detects_two_panels_in_rtl_order(self) -> None:
        panels = ClassicalLayoutParser().extract_panels(_synthetic_page())

        assert len(panels) == 2
        right, left = panels
        assert right.center[0] > left.center[0], "right panel must be read first"
        assert right.x_min == pytest.approx(430 / 800, abs=0.03)
        assert left.x_max == pytest.approx(370 / 800, abs=0.03)

    def test_full_bleed_page_falls_back_to_single_panel(self) -> None:
        # A page that thresholds to nothing (uniform near-white noise-free art).
        blank = np.full((600, 800, 3), 250, dtype=np.uint8)
        panels = ClassicalLayoutParser().extract_panels(blank)

        assert panels == [BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)]

    def test_detects_bubble_with_text(self) -> None:
        bubbles = ClassicalLayoutParser().extract_bubbles(_synthetic_panel_with_bubble())

        assert len(bubbles) == 1
        (bubble,) = bubbles
        cx, cy = bubble.center
        assert cx == pytest.approx(200 / 400, abs=0.05)
        assert cy == pytest.approx(150 / 400, abs=0.05)

    def test_ignores_bright_background_without_text(self) -> None:
        plain_white = np.full((400, 400, 3), 255, dtype=np.uint8)
        assert ClassicalLayoutParser().extract_bubbles(plain_white) == []


# ---------------------------------------------------------------------------
# Placeholder engines
# ---------------------------------------------------------------------------


class TestPlaceholders:
    def test_passthrough_vlm_skips_empty_bubbles(self) -> None:
        box = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
        bubbles = [
            SpeechBubble(text="Berserk!", bbox=box),
            SpeechBubble(text="   ", bbox=box),
        ]
        blocks = PassthroughVLM().contextualize(np.zeros((10, 10, 3), np.uint8), bubbles)

        assert [(b.text, b.speaker_id, b.is_speech) for b in blocks] == [
            ("Berserk!", "Desconhecido", True)
        ]

    def test_silent_tts_duration_tracks_text_length(self, tmp_path: Path) -> None:
        engine = SilentTTSEngine()
        block = ContextualizedBlock(text="x" * 10, speaker_id="Guts", is_speech=True)
        metadata = engine.synthesize(block, tmp_path / "seg.wav")

        assert metadata.duration_ms == 10 * SilentTTSEngine.MS_PER_CHAR
        with wave.open(str(metadata.file_path), "rb") as wav:
            frames_ms = wav.getnframes() * 1000 / wav.getframerate()
        assert frames_ms == pytest.approx(metadata.duration_ms, abs=1)

    def test_silent_tts_enforces_minimum_duration(self, tmp_path: Path) -> None:
        block = ContextualizedBlock(text="a", speaker_id="Guts", is_speech=True)
        metadata = SilentTTSEngine().synthesize(block, tmp_path / "seg.wav")

        assert metadata.duration_ms == SilentTTSEngine.MIN_DURATION_MS

    def test_wave_stitcher_joins_with_gaps(self, tmp_path: Path) -> None:
        engine = SilentTTSEngine()
        segments = [
            engine.synthesize(
                ContextualizedBlock(text="x" * 10, speaker_id="Guts", is_speech=True),
                tmp_path / f"seg{i}.wav",
            )
            for i in range(2)
        ]
        final = WaveFileStitcher().stitch(segments, tmp_path / "final.wav", gap_ms=100)

        with wave.open(str(final), "rb") as wav:
            total_ms = wav.getnframes() * 1000 / wav.getframerate()
            assert wav.getframerate() == SAMPLE_RATE
        expected_ms = sum(s.duration_ms for s in segments) + 100
        assert total_ms == pytest.approx(expected_ms, abs=2)


# ---------------------------------------------------------------------------
# EasyOCR (downloads real weights — deselected by default)
# ---------------------------------------------------------------------------


def test_easyocr_returns_empty_for_tiny_regions() -> None:
    from mangawhisperer.engines.ocr import EasyOCREngine

    # Guard path only: never touches the (undownloaded) reader.
    assert EasyOCREngine().recognize(np.zeros((4, 4, 3), np.uint8)) == ""


@pytest.mark.realmodels
def test_easyocr_reads_portuguese_text() -> None:
    from mangawhisperer.engines.ocr import EasyOCREngine

    canvas = np.full((120, 640, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "VOCE VAI MORRER", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)

    text = EasyOCREngine(gpu=False).recognize(canvas)
    assert "MORRER" in text.upper().replace(".", "")
