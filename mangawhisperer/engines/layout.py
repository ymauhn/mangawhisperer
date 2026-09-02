"""Classical computer-vision layout parsing (no learned models).

Zero-token heuristics (ticket #10), in the order they run:

* **Scan-border trim** — thin near-solid strips along the image edges
  (scanner shadow) are cropped away before anything else.
* **Spread split** — a landscape image whose centre carries a bright
  band (the two inner margins around the fold) is cut there and each
  half is parsed on its own, in reading order.
* **Gutter network** — the paper connected to the page margin (margin +
  gutters) is flooded; the holes left behind are panels. Survives art
  bleeding into a gutter and bubbles overlapping a border, which defeat
  straight-line cuts on real Berserk pages.
* **Projection-profile X-Y cut** — inside each region (or the whole page
  when the network finds nothing), recursively cut along rows/columns
  whose ink count is ~zero (gutters). Runs as thin as ``min_gutter_px``
  survive, which matters for Berserk's tight gutters. A cut crossed by
  a straight line (a border) is a panel interior, never a gutter.
  Undersized leaves are merged into their nearest neighbour, never
  dropped. Layouts without any straight gutter (diagonal borders,
  borderless art) fall back to contour blobs, then to the whole page.
* **Reading order** — recursive: split at clean horizontal gaps first
  (rows top-to-bottom), then vertical ones (right-to-left for manga,
  ``reading_order="rtl"``; left-to-right otherwise); staggered layouts
  without a clean gap fall back to overlap grouping.

Known limit (measured on Berserk vol. 1): bubbles whose outline crosses
a gutter block the straight cut on that axis. A second pass that erased
thin strokes recovered some of those gutters but also sliced through
bubble columns inside panels — losing dialogue is worse than merging
panels, so it was dropped; those pages wait for the learned detector.

Bubbles: bright convex-ish blobs that contain dark pixels (text).
"""

from __future__ import annotations

from typing import Callable, Literal

import cv2
import numpy as np

from mangawhisperer.interfaces import Image, MangaLayoutParser
from mangawhisperer.models import BoundingBox

ReadingOrder = Literal["rtl", "ltr"]
PixelRect = tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive)
Span = Callable[[BoundingBox], tuple[float, float]]


def sort_reading_order(
    boxes: list[BoundingBox], overlap_ratio: float = 0.5, rtl: bool = True
) -> list[BoundingBox]:
    """Sort boxes into reading order: rows top-to-bottom, then
    right-to-left within a row (``rtl=True``, manga) or left-to-right,
    recursively — a tall panel beside a 2×2 grid reads the tall panel,
    then the grid row by row.

    Groups are split at clean gaps (no box spans the cut). When a set
    of boxes has no clean gap on either axis (staggered layouts), they
    are grouped by overlap — ``overlap_ratio`` of the smaller extent —
    as a best-effort fallback per the project's reading-order decision.
    """
    return _order_recursive(list(boxes), overlap_ratio, rtl)


def _order_recursive(boxes: list[BoundingBox], overlap_ratio: float, rtl: bool) -> list[BoundingBox]:
    if len(boxes) <= 1:
        return boxes
    rows = _split_by_gap(boxes, lambda b: (b.y_min, b.y_max))
    if len(rows) > 1:
        return [b for row in rows for b in _order_recursive(row, overlap_ratio, rtl)]
    # No full-width gap: cut once at the widest vertical gap so a grid
    # beside a tall panel is separated from it first and then read row
    # by row, instead of column by column.
    columns = _split_by_gap(boxes, lambda b: (b.x_min, b.x_max), widest_only=True)
    if len(columns) > 1:
        if rtl:
            columns.reverse()
        return [b for column in columns for b in _order_recursive(column, overlap_ratio, rtl)]

    ordered: list[BoundingBox] = []
    for row in _group_by_overlap(boxes, lambda b: (b.y_min, b.y_max), overlap_ratio):
        cols = _group_by_overlap(row, lambda b: (b.x_min, b.x_max), overlap_ratio)
        cols.sort(key=lambda col: sum(b.center[0] for b in col) / len(col), reverse=rtl)
        for column in cols:
            ordered.extend(sorted(column, key=lambda b: b.y_min))
    return ordered


def _split_by_gap(
    boxes: list[BoundingBox], span: Span, widest_only: bool = False
) -> list[list[BoundingBox]]:
    """Split at positions no box spans (every gap, or only the widest
    one); groups come back in axis order."""
    groups: list[list[BoundingBox]] = []
    gaps: list[float] = []  # width of the gap before each group after the first
    current: list[BoundingBox] = []
    current_hi = float("-inf")
    for box in sorted(boxes, key=lambda b: span(b)[0]):
        lo, hi = span(box)
        if current and lo >= current_hi - 1e-9:
            groups.append(current)
            gaps.append(lo - current_hi)
            current = []
        current.append(box)
        current_hi = max(current_hi, hi)
    groups.append(current)
    if widest_only and len(groups) > 2:
        cut = gaps.index(max(gaps)) + 1
        groups = [[b for g in groups[:cut] for b in g], [b for g in groups[cut:] for b in g]]
    return groups


def _group_by_overlap(boxes, span: Span, overlap_ratio: float) -> list[list[BoundingBox]]:
    """Cluster boxes whose ``span`` (lo, hi) overlaps the cluster's by
    more than ``overlap_ratio`` of the smaller extent; clusters come
    back ordered by their start."""
    groups: list[dict] = []  # each: {"lo": float, "hi": float, "boxes": list[BoundingBox]}
    for box in sorted(boxes, key=lambda b: span(b)[0]):
        lo, hi = span(box)
        for group in groups:
            overlap = min(group["hi"], hi) - max(group["lo"], lo)
            smaller = min(hi - lo, group["hi"] - group["lo"])
            if smaller > 0 and overlap / smaller > overlap_ratio:
                group["boxes"].append(box)
                group["lo"], group["hi"] = min(group["lo"], lo), max(group["hi"], hi)
                break
        else:
            groups.append({"lo": lo, "hi": hi, "boxes": [box]})
    return [g["boxes"] for g in sorted(groups, key=lambda g: g["lo"])]


class ClassicalLayoutParser(MangaLayoutParser):
    """Panel and bubble detection with thresholding, projections and contours."""

    def __init__(
        self,
        ink_threshold: int = 220,
        bright_threshold: int = 220,
        min_panel_area_ratio: float = 0.02,
        min_bubble_area_ratio: float = 0.004,
        max_bubble_area_ratio: float = 0.5,
        min_bubble_solidity: float = 0.75,
        text_pixel_ratio_range: tuple[float, float] = (0.01, 0.5),
        reading_order: ReadingOrder = "rtl",
        gutter_ink_ratio: float = 0.02,
        min_gutter_px: int = 3,
        min_panel_side_ratio: float = 0.04,
        spread_min_aspect: float = 1.15,
        spread_fold_ratio: float = 0.03,
        max_cut_depth: int = 8,
        min_ink_coverage: float = 0.7,
        scan_border_ratio: float = 0.03,
        scan_border_min_px: int = 6,
        min_hole_ink_ratio: float = 0.0005,
    ) -> None:
        """
        Args:
            ink_threshold: Gray level below which a pixel counts as ink
                (panel content) on a page.
            bright_threshold: Gray level above which a pixel counts as
                bubble interior inside a panel (and as gutter/margin
                paper when looking for a spread's fold).
            min_panel_area_ratio: Minimum region area, as a fraction of
                the page, to qualify as a panel on its own; smaller
                projection leaves are merged into a neighbour.
            min_bubble_area_ratio: Minimum blob area, as a fraction of
                the panel, to qualify as a bubble.
            max_bubble_area_ratio: Maximum bubble fraction of the panel;
                filters out bright page backgrounds.
            min_bubble_solidity: Minimum contour-area / hull-area ratio;
                bubbles are convex-ish.
            text_pixel_ratio_range: Accepted fraction of dark pixels
                inside the bubble box — too few means an empty blob, too
                many means artwork, not a bubble.
            reading_order: ``"rtl"`` for manga, ``"ltr"`` for western comics.
            gutter_ink_ratio: A row/column is a gutter candidate when its
                ink pixels are at most this fraction of its length
                (tolerates scan speckle and bubble tails crossing the
                gutter).
            min_gutter_px: Shortest blank run accepted as a gutter.
            min_panel_side_ratio: Every panel side must be at least this
                fraction of the page's extent (rejects hairline slivers
                such as a scanner strip); cuts must leave that much
                content on both sides.
            spread_min_aspect: Width/height above which the image may be
                a two-page spread.
            spread_fold_ratio: The fold must be a bright band at least
                this fraction of the width (two inner margins); a single
                landscape page's gutter is narrower.
            max_cut_depth: Recursion bound for the X-Y cut.
            min_ink_coverage: The gutter-network result is trusted only
                when its boxes cover at least this fraction of the
                page's ink (guards against a flooded bleed panel).
            scan_border_ratio: Near-solid strips along the edges up to
                this fraction of the extent are cropped as scanner
                shadow; anything thicker is art and stays.
            scan_border_min_px: Strips thinner than this are frame
                lines at the page edge, not scanner shadow.
            min_hole_ink_ratio: Undersized gutter-network holes carrying
                at least this fraction of the page area in ink (a small
                inset panel) are merged into the nearest panel; lighter
                ones (page numbers, specks) are dropped.
        """
        self._ink_threshold = ink_threshold
        self._bright_threshold = bright_threshold
        self._min_panel_area_ratio = min_panel_area_ratio
        self._min_bubble_area_ratio = min_bubble_area_ratio
        self._max_bubble_area_ratio = max_bubble_area_ratio
        self._min_bubble_solidity = min_bubble_solidity
        self._text_pixel_ratio_range = text_pixel_ratio_range
        self._rtl = reading_order == "rtl"
        self._gutter_ink_ratio = gutter_ink_ratio
        self._min_gutter_px = min_gutter_px
        self._min_panel_side_ratio = min_panel_side_ratio
        self._spread_min_aspect = spread_min_aspect
        self._spread_fold_ratio = spread_fold_ratio
        self._max_cut_depth = max_cut_depth
        self._min_ink_coverage = min_ink_coverage
        self._scan_border_ratio = scan_border_ratio
        self._scan_border_min_px = scan_border_min_px
        self._min_hole_ink_ratio = min_hole_ink_ratio

    @property
    def fingerprint(self) -> str:
        return f"classical-layout:v3-gutters:{'rtl' if self._rtl else 'ltr'}"

    # ── Panels ────────────────────────────────────────────────────────

    def extract_panels(self, page_image: Image) -> list[BoundingBox]:
        """Detect panels on a page (or spread); falls back to one full-page panel."""
        gray = self._to_gray(page_image)
        if gray.size == 0:
            return []
        page_h, page_w = gray.shape
        y_off, x_off, gray = self._trim_scan_borders(gray)
        boxes: list[BoundingBox] = []
        for half_off, half in self.split_spread(gray):
            half_boxes = [
                self._normalize(x0 + x_off + half_off, y0 + y_off, x1 - x0, y1 - y0, page_w, page_h)
                for x0, y0, x1, y1 in self._panels_in_page(half)
            ]
            boxes.extend(sort_reading_order(half_boxes, rtl=self._rtl))
        return boxes

    def _trim_scan_borders(self, gray: np.ndarray) -> tuple[int, int, np.ndarray]:
        """Crop thin near-solid strips along the edges (scanner shadow).
        Returns ``(y_offset, x_offset, cropped)``."""
        ink = gray < self._ink_threshold
        page_h, page_w = ink.shape

        def leading(mask: np.ndarray, limit: int) -> int:
            count = 0
            while count < min(mask.size, limit) and mask[count]:
                count += 1
            # Thinner than a scanner shadow it is a frame line; thicker it is art.
            return count if self._scan_border_min_px <= count < limit else 0

        dark_rows, dark_cols = ink.mean(axis=1) > 0.9, ink.mean(axis=0) > 0.9
        limit_y, limit_x = int(self._scan_border_ratio * page_h), int(self._scan_border_ratio * page_w)
        top, bottom = leading(dark_rows, limit_y), leading(dark_rows[::-1], limit_y)
        left, right = leading(dark_cols, limit_x), leading(dark_cols[::-1], limit_x)
        return top, left, gray[top:page_h - bottom, left:page_w - right]

    def split_spread(self, gray: np.ndarray) -> list[tuple[int, np.ndarray]]:
        """Return ``[(x_offset, sub_image)]`` in reading order: the whole
        image for a single page, two halves for a landscape spread whose
        centre carries a bright band (the inner margins around the
        fold) and whose halves are both inked."""
        page_h, page_w = gray.shape
        if page_h == 0 or page_w / page_h < self._spread_min_aspect:
            return [(0, gray)]
        centre = page_w // 2
        band = max(1, int(page_w * 0.08))
        strip = gray[:, centre - band : centre + band]
        paper = (strip >= self._bright_threshold).mean(axis=0) >= 0.98
        start, length = self._longest_run(paper)
        if length < self._spread_fold_ratio * page_w:
            return [(0, gray)]
        fold = strip[:, start : start + length] < self._ink_threshold
        if (fold.mean(axis=1) > 0.9).any():  # a border crosses it: a panel interior, not the fold
            return [(0, gray)]
        cut = centre - band + start + length // 2
        ink = gray < self._ink_threshold
        if cut <= 0 or cut >= page_w or not (ink[:, :cut].any() and ink[:, cut:].any()):
            return [(0, gray)]
        halves = [(cut, gray[:, cut:]), (0, gray[:, :cut])]
        return halves if self._rtl else halves[::-1]

    @staticmethod
    def _longest_run(mask: np.ndarray) -> tuple[int, int]:
        """``(start, length)`` of the longest run of True values."""
        best_start = best_len = 0
        index = 0
        while index < mask.size:
            if not mask[index]:
                index += 1
                continue
            end = index
            while end < mask.size and mask[end]:
                end += 1
            if end - index > best_len:
                best_start, best_len = index, end - index
            index = end
        return best_start, best_len

    def _panels_in_page(self, gray: np.ndarray) -> list[PixelRect]:
        """Heuristic ladder: gutter network ➜ projection cut inside each
        region ➜ blob fallback ➜ the whole page. Stages only ever split
        further; none merges what an earlier stage separated."""
        page_h, page_w = gray.shape
        regions = self._panels_by_gutter_network(gray)
        if len(regions) > 1:
            rects: list[PixelRect] = []
            for x0, y0, x1, y1 in regions:
                inner = self._panels_by_projection(gray[y0:y1, x0:x1]) or [(0, 0, x1 - x0, y1 - y0)]
                rects.extend((x0 + a, y0 + b, x0 + c, y0 + d) for a, b, c, d in inner)
            return rects
        rects = self._panels_by_projection(gray)
        if len(rects) <= 1:  # no straight gutter: blobs may still separate staggered panels
            blobs = self._panels_by_contours(gray)
            if len(blobs) > len(rects):
                rects = blobs
        return rects or [(0, 0, page_w, page_h)]

    def _panels_by_gutter_network(self, gray: np.ndarray) -> list[PixelRect]:
        """Panels as the holes of the paper connected to the page margin.

        Gutters are white and reach the margin, so flooding the paper
        from the image border paints margin + gutters; what is left are
        the panels (framed or solid black alike). Art bleeding into a
        gutter merely narrows it, and a bubble overlapping a border
        keeps its own outline, so neither leaks. Fragments inside one
        panel (a white background touching the page edge gets flooded)
        are merged by box overlap; if the boxes still miss too much
        ink, the result is rejected and the caller falls back.
        """
        page_h, page_w = gray.shape
        paper = (gray >= self._bright_threshold).astype(np.uint8)
        _, labels = cv2.connectedComponents(paper, connectivity=4)
        edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        margin_labels = np.unique(edge[edge != 0])
        if margin_labels.size == 0:
            return []
        interior = (~np.isin(labels, margin_labels)).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(interior, connectivity=4)

        ink = gray < self._ink_threshold
        min_area = self._min_panel_area_ratio * page_h * page_w
        min_w, min_h = self._min_panel_side_ratio * page_w, self._min_panel_side_ratio * page_h
        min_hole_ink = self._min_hole_ink_ratio * page_h * page_w
        holes = [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, _area in stats[1:]]
        rects = [r for r in holes if (r[2] - r[0]) * (r[3] - r[1]) >= min_area
                 and r[2] - r[0] >= min_w and r[3] - r[1] >= min_h]
        rects = self._merge_overlapping(rects)
        if rects:  # small inked holes (an inset panel) join their nearest panel; specks are dropped
            for small in (r for r in holes if r not in rects):
                if int(ink[small[1]:small[3], small[0]:small[2]].sum()) >= min_hole_ink:
                    nearest = min(range(len(rects)), key=lambda i: _rect_gap(rects[i], small))
                    rects[nearest] = _union(rects[nearest], small)

        covered = np.zeros_like(ink)
        for x0, y0, x1, y1 in rects:
            covered[y0:y1, x0:x1] = True
        total_ink = int(ink.sum())
        if total_ink and int((ink & covered).sum()) / total_ink < self._min_ink_coverage:
            return []
        return rects

    @staticmethod
    def _merge_overlapping(rects: list[PixelRect], ratio: float = 0.3) -> list[PixelRect]:
        """Union boxes whose intersection exceeds ``ratio`` of the smaller one."""
        rects = list(rects)
        merged = True
        while merged:
            merged = False
            for i in range(len(rects)):
                for j in range(i + 1, len(rects)):
                    a, b = rects[i], rects[j]
                    iw = min(a[2], b[2]) - max(a[0], b[0])
                    ih = min(a[3], b[3]) - max(a[1], b[1])
                    if iw <= 0 or ih <= 0:
                        continue
                    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
                    if iw * ih / smaller > ratio:
                        rects[i] = _union(a, b)
                        del rects[j]
                        merged = True
                        break
                if merged:
                    break
        return rects

    def _panels_by_projection(self, gray: np.ndarray) -> list[PixelRect]:
        """X-Y cut leaves; undersized leaves are merged into the nearest
        panel-sized one so no content is ever dropped."""
        ink = gray < self._ink_threshold
        page_h, page_w = ink.shape
        leaves: list[PixelRect] = []
        self._xy_cut(ink, 0, 0, page_w, page_h, leaves, depth=0)
        min_area = self._min_panel_area_ratio * page_h * page_w
        panels = [r for r in leaves if (r[2] - r[0]) * (r[3] - r[1]) >= min_area]
        if not panels:
            return []
        for small in (r for r in leaves if (r[2] - r[0]) * (r[3] - r[1]) < min_area):
            nearest = min(range(len(panels)), key=lambda i: _rect_gap(panels[i], small))
            panels[nearest] = _union(panels[nearest], small)
        return panels

    def _xy_cut(
        self, ink: np.ndarray, x0: int, y0: int, x1: int, y1: int,
        out: list[PixelRect], depth: int,
    ) -> None:
        """Trim the region to its content, then split it at the first
        axis with a gutter run; leaves become panel candidates."""
        region = ink[y0:y1, x0:x1]
        rows = np.flatnonzero(~self._blank_profile(region, axis=0))
        cols = np.flatnonzero(~self._blank_profile(region, axis=1))
        if rows.size == 0 or cols.size == 0:
            return
        y0, y1 = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
        x0, x1 = x0 + int(cols[0]), x0 + int(cols[-1]) + 1

        if depth < self._max_cut_depth:
            region = ink[y0:y1, x0:x1]
            for axis in (0, 1):  # 0: cut between rows, 1: cut between columns
                cuts = self._find_gutters(region, axis)
                if not cuts:
                    continue
                bounds = [0, *cuts, region.shape[axis]]
                for start, end in zip(bounds, bounds[1:]):
                    if axis == 0:
                        self._xy_cut(ink, x0, y0 + start, x1, y0 + end, out, depth + 1)
                    else:
                        self._xy_cut(ink, x0 + start, y0, x0 + end, y1, out, depth + 1)
                return
        out.append((x0, y0, x1, y1))

    def _blank_profile(self, region: np.ndarray, axis: int) -> np.ndarray:
        """Per row (``axis=0``) / column (``axis=1``): ink at or below
        the tolerance — a bubble tail or a page number crossing a
        gutter must not hide it."""
        profile = region.sum(axis=1 - axis)
        return profile <= max(1, int(self._gutter_ink_ratio * region.shape[1 - axis]))

    def _find_gutters(self, ink: np.ndarray, axis: int) -> list[int]:
        """Centres of the internal gutter runs along ``axis`` (0 = rows,
        1 = columns), leaving at least ``min_side`` of content between
        consecutive gutters and at both ends."""
        blank = self._blank_profile(ink, axis)
        length = int(blank.size)
        min_side = max(self._min_gutter_px, int(self._min_panel_side_ratio * length))
        cuts: list[int] = []
        last_end = 0  # end of the last accepted gutter = start of the current content chunk
        index = 0
        while index < length:
            if not blank[index]:
                index += 1
                continue
            end = index
            while end < length and blank[end]:
                end += 1
            internal = index > 0 and end < length
            if (
                internal
                and end - index >= self._min_gutter_px
                and index - last_end >= min_side
                and not self._crossed_by_line(ink, axis, index, end)
            ):
                cuts.append((index + end) // 2)
                last_end = end
            index = end
        if cuts and length - last_end < min_side:
            cuts.pop()
        return cuts

    @staticmethod
    def _crossed_by_line(ink: np.ndarray, axis: int, start: int, end: int) -> bool:
        """A run crossed by a straight line is white space *inside* a
        frame, not a gutter: either the run is wide (a panel interior,
        spanned by its own border) or the line continues contiguously
        beyond a narrow run on both sides (a border passing a T-junction
        or a panel's own frame). Bubble outlines crossing a narrow
        gutter curve away within a few pixels and pass."""
        run = end - start
        wide = run >= 0.1 * ink.shape[axis]
        reach = max(20, 2 * run)
        band = ink[start:end] if axis == 0 else ink[:, start:end]
        for idx in np.flatnonzero(band.mean(axis=axis) > 0.9):
            if wide:
                return True
            line = ink[:, idx] if axis == 0 else ink[idx, :]
            before = _contiguous_ink(line[:start][::-1])
            after = _contiguous_ink(line[end:])
            if before and after and before + after >= reach:
                return True
        return False

    def _panels_by_contours(self, gray: np.ndarray) -> list[PixelRect]:
        """Legacy blob method: dilate ink until each panel's art merges."""
        height, width = gray.shape
        ink = (gray < self._ink_threshold).astype(np.uint8) * 255
        kernel_size = max(3, min(height, width) // 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        merged = cv2.dilate(ink, kernel, iterations=2)

        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        page_area = float(height * width)
        rects: list[PixelRect] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if (w * h) / page_area >= self._min_panel_area_ratio:
                rects.append((x, y, x + w, y + h))
        return rects

    # ── Bubbles ───────────────────────────────────────────────────────

    def extract_bubbles(self, panel_image: Image) -> list[BoundingBox]:
        """Detect speech-bubble interiors inside one panel."""
        gray = self._to_gray(panel_image)
        if gray.size == 0:
            return []
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
        return sort_reading_order(boxes, rtl=self._rtl)

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_gray(image: Image) -> np.ndarray:
        if image.ndim == 3:
            if image.shape[2] == 1:
                return image[:, :, 0]
            if image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
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


def _union(a: PixelRect, b: PixelRect) -> PixelRect:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _rect_gap(a: PixelRect, b: PixelRect) -> int:
    """Manhattan distance between two rectangles' edges (0 when they touch/overlap)."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return dx + dy


def _contiguous_ink(line: np.ndarray) -> int:
    """Length of the ink run at the start of ``line``."""
    gaps = np.flatnonzero(~line)
    return int(gaps[0]) if gaps.size else int(line.size)
