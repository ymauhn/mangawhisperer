"""Real PT-BR OCR via EasyOCR."""

from __future__ import annotations

from typing import Any, Sequence

import cv2

from mangawhisperer.interfaces import Image, PortugueseOCREngine


class EasyOCREngine(PortugueseOCREngine):
    """EasyOCR-backed recognizer for Latin-alphabet manga text.

    The underlying ``easyocr.Reader`` is created lazily on first use
    (its construction downloads ~110 MB of detection/recognition
    weights), so instantiating this engine stays cheap.
    """

    def __init__(
        self,
        languages: Sequence[str] = ("pt",),
        gpu: bool | None = None,
        min_region_px: int = 8,
    ) -> None:
        """
        Args:
            languages: EasyOCR language codes; ``pt`` covers PT-BR
                diacritics (á, ç, õ).
            gpu: Force GPU on/off; ``None`` auto-detects CUDA.
            min_region_px: Regions narrower/shorter than this are
                assumed to be segmentation noise and return "".
        """
        self._languages = list(languages)
        self._gpu = gpu
        self._min_region_px = min_region_px
        self._reader: Any = None

    def _get_reader(self) -> Any:
        if self._reader is None:
            import easyocr  # noqa: PLC0415 — heavy import, deferred until needed
            import torch  # noqa: PLC0415

            gpu = self._gpu if self._gpu is not None else torch.cuda.is_available()
            self._reader = easyocr.Reader(self._languages, gpu=gpu, verbose=False)
        return self._reader

    def recognize(self, region_image: Image) -> str:
        """OCR one bubble crop; returns "" for regions too small to read."""
        height, width = region_image.shape[:2]
        if height < self._min_region_px or width < self._min_region_px:
            return ""

        # Grayscale sidesteps RGB-vs-BGR channel-order ambiguity in EasyOCR.
        gray = (
            cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY)
            if region_image.ndim == 3
            else region_image
        )
        fragments: list[str] = self._get_reader().readtext(gray, detail=0, paragraph=True)
        return " ".join(" ".join(fragments).split())
