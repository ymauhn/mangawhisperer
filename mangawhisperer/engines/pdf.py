"""Real PDF rasterization via PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from mangawhisperer.interfaces import PDFPageExtractor


class PyMuPDFPageExtractor(PDFPageExtractor):
    """Renders PDF pages to RGB PNG files with PyMuPDF.

    Page files are named ``page_<NNN>.png`` using the 1-based page
    number from the source document, so a partial extraction (via
    ``first_page``/``max_pages``) stays traceable to the original PDF.
    """

    def __init__(self, dpi: int = 200, first_page: int = 1, max_pages: int | None = None) -> None:
        """
        Args:
            dpi: Render resolution. 200 dpi keeps speech-bubble text
                crisp enough for OCR without producing huge files.
            first_page: 1-based page to start rendering from.
            max_pages: Render at most this many pages; ``None`` renders
                through the end of the document.
        """
        if dpi <= 0:
            raise ValueError(f"dpi must be positive, got {dpi}")
        if first_page < 1:
            raise ValueError(f"first_page is 1-based, got {first_page}")
        self._dpi = dpi
        self._first_page = first_page
        self._max_pages = max_pages

    @property
    def fingerprint(self) -> str:
        """Checkpoint identity: page range + dpi define the artifacts."""
        return f"pymupdf:dpi={self._dpi}:first={self._first_page}:max={self._max_pages}"

    def extract_pages(self, pdf_path: Path, output_dir: Path) -> list[Path]:
        """Render the configured page range of ``pdf_path`` into ``output_dir``."""
        paths: list[Path] = []
        with pymupdf.open(pdf_path) as doc:
            start = self._first_page - 1
            end = len(doc) if self._max_pages is None else min(len(doc), start + self._max_pages)
            for index in range(start, end):
                pixmap = doc[index].get_pixmap(dpi=self._dpi, colorspace=pymupdf.csRGB)
                page_path = output_dir / f"page_{index + 1:03d}.png"
                pixmap.save(page_path)
                paths.append(page_path)
        return paths
