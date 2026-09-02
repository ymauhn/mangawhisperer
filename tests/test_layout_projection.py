"""Projection-profile panel cuts, gutter network, spread splitting and
reading order (ticket #10). Pages are synthetic: white paper, black
panel borders."""

import numpy as np
import pytest

from mangawhisperer.engines.layout import ClassicalLayoutParser, sort_reading_order
from mangawhisperer.models import BoundingBox


def _page(width: int, height: int) -> np.ndarray:
    return np.full((height, width), 255, dtype=np.uint8)


def _draw_panel(page: np.ndarray, x0: int, y0: int, x1: int, y1: int, border: int = 3) -> None:
    page[y0:y1, x0:x0 + border] = 0
    page[y0:y1, x1 - border:x1] = 0
    page[y0:y0 + border, x0:x1] = 0
    page[y1 - border:y1, x0:x1] = 0


def _rect(box: BoundingBox, width: int, height: int) -> tuple[int, int, int, int]:
    return box.to_absolute(width, height)


def _box(x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
    return BoundingBox(x_min=x0, y_min=y0, x_max=x1, y_max=y1)


def test_thin_gutter_survives_and_rows_come_before_columns():
    width, height = 1000, 1400
    page = _page(width, height)
    _draw_panel(page, 520, 40, 960, 600)  # top-right
    _draw_panel(page, 40, 40, 516, 600)  # top-left — 4 px gutter to its right
    _draw_panel(page, 40, 640, 960, 1360)  # bottom, full width

    boxes = ClassicalLayoutParser().extract_panels(page)

    assert len(boxes) == 3
    rects = [_rect(b, width, height) for b in boxes]
    assert rects[0] == (520, 40, 960, 600)  # manga: right first...
    assert rects[1] == (40, 40, 516, 600)  # ...then left...
    assert rects[2] == (40, 640, 960, 1360)  # ...then the next row


def test_ltr_reading_order_flag_flips_rows():
    width, height = 1000, 1400
    page = _page(width, height)
    _draw_panel(page, 520, 40, 960, 600)
    _draw_panel(page, 40, 40, 516, 600)

    boxes = ClassicalLayoutParser(reading_order="ltr").extract_panels(page)

    assert [_rect(b, width, height)[0] for b in boxes] == [40, 520]


def test_page_number_in_margin_does_not_become_a_panel():
    width, height = 1000, 1400
    page = _page(width, height)
    _draw_panel(page, 40, 40, 960, 700)
    _draw_panel(page, 40, 740, 960, 1300)
    page[1340:1352, 490:510] = 0  # "page number" ink at the bottom margin

    boxes = ClassicalLayoutParser().extract_panels(page)

    assert [_rect(b, width, height) for b in boxes] == [(40, 40, 960, 700), (40, 740, 960, 1300)]


def test_landscape_spread_is_split_and_right_page_read_first():
    width, height = 2000, 1400  # two portrait pages side by side
    page = _page(width, height)
    _draw_panel(page, 60, 60, 940, 1340)  # left page
    _draw_panel(page, 1060, 60, 1940, 1340)  # right page

    parser = ClassicalLayoutParser()
    halves = parser.split_spread(page)
    boxes = parser.extract_panels(page)

    assert [offset for offset, _ in halves] == [1000, 0]
    assert [_rect(b, width, height) for b in boxes] == [(1060, 60, 1940, 1340), (60, 60, 940, 1340)]


def test_landscape_single_page_with_a_grid_is_not_a_spread():
    width, height = 1400, 1000
    page = _page(width, height)
    _draw_panel(page, 40, 40, 688, 480)
    _draw_panel(page, 712, 40, 1360, 480)  # 24 px gutter: far narrower than a fold band
    _draw_panel(page, 40, 520, 688, 960)
    _draw_panel(page, 712, 520, 1360, 960)

    parser = ClassicalLayoutParser()
    assert [offset for offset, _ in parser.split_spread(page)] == [0]
    boxes = parser.extract_panels(page)

    assert [_rect(b, width, height) for b in boxes] == [
        (712, 40, 1360, 480), (40, 40, 688, 480), (712, 520, 1360, 960), (40, 520, 688, 960),
    ]


def test_art_bleeding_into_a_gutter_is_handled_by_the_gutter_network():
    width, height = 1000, 1400
    page = _page(width, height)
    _draw_panel(page, 520, 0, 960, 1360)  # right panel bleeds off the top edge...
    page[0:35, 500:960] = 0  # ...and its art spills into the gutter up there
    _draw_panel(page, 40, 40, 500, 600)  # left column: two framed panels
    _draw_panel(page, 40, 640, 500, 1360)

    parser = ClassicalLayoutParser()
    assert len(parser._panels_by_gutter_network(page)) == 3  # holes in the flooded paper
    boxes = parser.extract_panels(page)

    assert [_rect(b, width, height) for b in boxes] == [
        (500, 0, 960, 1360), (40, 40, 500, 600), (40, 640, 500, 1360),
    ]


@pytest.mark.parametrize("band", [(60, 130), (300, 370), (480, 550)])  # near the frame or not
def test_white_band_inside_a_frame_is_never_a_gutter(band):
    frame = np.zeros((400, 600), dtype=bool)  # a framed panel with sparse art
    frame[:3, :] = frame[-3:, :] = frame[:, :3] = frame[:, -3:] = True
    frame[150:250, 20:600] = True  # art everywhere...
    frame[150:250, band[0]:band[1]] = False  # ...except one white vertical band

    assert ClassicalLayoutParser()._find_gutters(frame, axis=1) == []


def test_undersized_leaf_is_merged_into_its_neighbour_not_dropped():
    width, height = 1000, 1400
    page = _page(width, height)
    _draw_panel(page, 40, 40, 960, 700)
    _draw_panel(page, 40, 740, 800, 1360)
    _draw_panel(page, 840, 740, 960, 860)  # tiny inset panel (0.9% of the page)

    boxes = ClassicalLayoutParser().extract_panels(page)

    rects = [_rect(b, width, height) for b in boxes]
    assert len(rects) == 2  # the inset joins a neighbour (either is 40 px away) instead of vanishing
    assert any(r[0] <= 840 and r[2] >= 960 and r[1] <= 740 and r[3] >= 860 for r in rects)


def test_scanner_strip_along_the_edge_is_not_a_panel():
    width, height = 1000, 1400
    page = _page(width, height)
    _draw_panel(page, 40, 40, 940, 700)
    _draw_panel(page, 40, 740, 940, 1360)
    page[:, 975:1000] = 0  # dark scanner shadow on the right edge

    boxes = ClassicalLayoutParser().extract_panels(page)

    assert [_rect(b, width, height) for b in boxes] == [(40, 40, 940, 700), (40, 740, 940, 1360)]


def test_diagonal_layout_falls_back_to_contours_then_page():
    width, height = 800, 1000
    page = _page(width, height)
    for i in range(0, 800):  # one diagonal stroke: no straight gutter anywhere
        page[i + 100:i + 106, i:i + 6] = 0
    _draw_panel(page, 0, 0, 800, 1000)

    boxes = ClassicalLayoutParser().extract_panels(page)

    assert boxes == [BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)]


def test_blank_page_is_one_full_page_panel():
    page = np.full((600, 400), 250, dtype=np.uint8)
    assert ClassicalLayoutParser().extract_panels(page) == [
        BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
    ]


def test_degenerate_images_do_not_crash():
    parser = ClassicalLayoutParser()
    assert parser.extract_panels(np.zeros((0, 10), dtype=np.uint8)) == []
    assert parser.extract_panels(np.full((2, 2, 1), 255, dtype=np.uint8)) == [_box(0.0, 0.0, 1.0, 1.0)]
    assert len(parser.extract_panels(np.zeros((300, 200), dtype=np.uint8))) == 1  # all black


def test_sort_reading_order_supports_ltr():
    left = _box(0.0, 0.0, 0.4, 0.5)
    right = _box(0.6, 0.0, 1.0, 0.5)
    below = _box(0.0, 0.6, 1.0, 1.0)
    assert sort_reading_order([below, left, right]) == [right, left, below]
    assert sort_reading_order([below, left, right], rtl=False) == [left, right, below]


def test_stacked_panels_beside_a_tall_one_read_top_to_bottom():
    tall_right = _box(0.4, 0.0, 1.0, 0.5)
    top_left = _box(0.1, 0.05, 0.36, 0.22)
    bottom_left = _box(0.1, 0.26, 0.37, 0.46)  # a hair wider
    below = _box(0.1, 0.55, 1.0, 1.0)

    assert sort_reading_order([bottom_left, below, top_left, tall_right]) == [
        tall_right, top_left, bottom_left, below,
    ]
    assert sort_reading_order([bottom_left, below, top_left, tall_right], rtl=False) == [
        top_left, bottom_left, tall_right, below,
    ]


def test_grid_beside_a_tall_panel_reads_row_by_row():
    tall = _box(0.55, 0.0, 1.0, 1.0)
    a, b = _box(0.28, 0.0, 0.5, 0.45), _box(0.0, 0.0, 0.25, 0.45)  # top row (right, left)
    c, d = _box(0.28, 0.55, 0.5, 1.0), _box(0.0, 0.55, 0.25, 1.0)  # bottom row

    assert sort_reading_order([d, b, tall, c, a]) == [tall, a, b, c, d]
    assert sort_reading_order([d, b, tall, c, a], rtl=False) == [b, a, d, c, tall]


def test_fingerprint_tracks_reading_order():
    assert ClassicalLayoutParser().fingerprint != ClassicalLayoutParser(reading_order="ltr").fingerprint
