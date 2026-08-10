"""Classical computer-vision layout parsing (no learned models).

This is the first real :class:`MangaLayoutParser`: contour-based panel
splitting (in the spirit of adenzu/Manga-Panel-Extractor) and
bright-blob speech-bubble detection. It is deliberately model-free so
the input pipeline works today; a comic-text-detector-backed parser can
replace it later behind the same ABC.
"""

from __future__ import annotations

import cv2
import numpy as np

from mangawhisperer.interfaces import Image, MangaLayoutParser
from mangawhisperer.models import BoundingBox


def sort_reading_order(boxes: list[BoundingBox], overlap_ratio: float = 0.5) -> list[BoundingBox]:
    """Sort boxes into manga reading order: rows top-to-bottom, and
    right-to-left within a row.

    Boxes are grouped into a row when their vertical overlap with the
    row's span exceeds ``overlap_ratio`` of the smaller height — a
    best-effort heuristic per the project's reading-order decision.
    """
    rows: list[dict] = []  # each: {"y0": float, "y1": float, "boxes": list[BoundingBox]}
    for box in sorted(boxes, key=lambda b: b.y_min):
        for row in rows:
            overlap = min(row["y1"], box.y_max) - max(row["y0"], box.y_min)
            min_height = min(box.height, row["y1"] - row["y0"])
            if min_height > 0 and overlap / min_height > overlap_ratio:
                row["boxes"].append(box)
                row["y0"] = min(row["y0"], box.y_min)
                row["y1"] = max(row["y1"], box.y_max)
                break
        else:
            rows.append({"y0": box.y_min, "y1": box.y_max, "boxes": [box]})

    ordered: list[BoundingBox] = []
    for row in sorted(rows, key=lambda r: r["y0"]):
        ordered.extend(sorted(row["boxes"], key=lambda b: b.center[0], reverse=True))
    return ordered


class ClassicalLayoutParser(MangaLayoutParser):
    """Panel and bubble detection with thresholding + contours.

    Panels: ink pixels are dilated until each panel's artwork merges
    into one blob; blob bounding boxes above a minimum area are panels.
    Falls back to the full page when nothing qualifies (full-bleed art).

    Bubbles: bright connected regions (bubble interiors are near-white,
    enclosed by their dark outline) that are convex-ish, reasonably
    sized, and actually contain dark pixels (i.e. text).
    """

    def __init__(
        self,
        ink_threshold: int = 220,
        bright_threshold: int = 220,
        min_panel_area_ratio: float = 0.02,
        min_bubble_area_ratio: float = 0.004,
        max_bubble_area_ratio: float = 0.5,
        min_bubble_solidity: float = 0.75,
        text_pixel_ratio_range: tuple[float, float] = (0.01, 0.5),
    ) -> None:
        """
        Args:
            ink_threshold: Gray level below which a pixel counts as ink
                (panel content) on a page.
            bright_threshold: Gray level above which a pixel counts as
                bubble interior inside a panel.
            min_panel_area_ratio: Minimum blob area, as a fraction of
                the page, to qualify as a panel.
            min_bubble_area_ratio: Minimum blob area, as a fraction of
                the panel, to qualify as a bubble.
            max_bubble_area_ratio: Maximum bubble fraction of the panel;
                filters out bright page backgrounds.
            min_bubble_solidity: Minimum contour-area / hull-area ratio;
                bubbles are convex-ish.
            text_pixel_ratio_range: Accepted fraction of dark pixels
                inside the bubble box — too few means an empty blob, too
                many means artwork, not a bubble.
        """
        self._ink_threshold = ink_threshold
        self._bright_threshold = bright_threshold
        self._min_panel_area_ratio = min_panel_area_ratio
        self._min_bubble_area_ratio = min_bubble_area_ratio
        self._max_bubble_area_ratio = max_bubble_area_ratio
        self._min_bubble_solidity = min_bubble_solidity
        self._text_pixel_ratio_range = text_pixel_ratio_range

    def extract_panels(self, page_image: Image) -> list[BoundingBox]:
        """Detect panels on a page; falls back to one full-page panel."""
        gray = self._to_gray(page_image)
        height, width = gray.shape
        ink = (gray < self._ink_threshold).astype(np.uint8) * 255

        kernel_size = max(3, min(height, width) // 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        merged = cv2.dilate(ink, kernel, iterations=2)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_area = float(height * width)
        boxes: list[BoundingBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w * h) / page_area >= self._min_panel_area_ratio:
                boxes.append(self._normalize(x, y, w, h, width, height))

        if not boxes:
            boxes = [BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)]
        return sort_reading_order(boxes)

    def extract_bubbles(self, panel_image: Image) -> list[BoundingBox]:
        """Detect speech-bubble interiors inside one panel."""
        gray = self._to_gray(panel_image)
        height, width = gray.shape
        bright = (gray >= self._bright_threshold).astype(np.uint8) * 255
        dark = gray < 128

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        panel_area = float(height * width)
        boxes: list[BoundingBox] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            area_ratio = area / panel_area
            if not (self._min_bubble_area_ratio <= area_ratio <= self._max_bubble_area_ratio):
                continue

            hull_area = cv2.contourArea(cv2.convexHull(contour))
            if hull_area <= 0 or area / hull_area < self._min_bubble_solidity:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            text_ratio = float(np.count_nonzero(dark[y : y + h, x : x + w])) / float(w * h)
            low, high = self._text_pixel_ratio_range
            if not (low <= text_ratio <= high):
                continue

            boxes.append(self._normalize(x, y, w, h, width, height))
        return sort_reading_order(boxes)

    @staticmethod
    def _to_gray(image: Image) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image

    @staticmethod
    def _normalize(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> BoundingBox:
        return BoundingBox(
            x_min=x / img_w,
            y_min=y / img_h,
            x_max=min(1.0, (x + w) / img_w),
            y_max=min(1.0, (y + h) / img_h),
        )
