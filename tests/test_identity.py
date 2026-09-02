"""Character identity (ticket #22): YOLO decoding, gallery gate, geometry
and the leave-one-out evaluation — all without a model."""

import numpy as np
import pytest

from mangawhisperer.engines.identity import (
    UNKNOWN,
    CharacterDetector,
    Detection,
    Gallery,
    assign_to_panels,
    containment,
    crop_with_margin,
    decode_yolo_output,
    format_hint_block,
    l2_normalize,
    leave_one_out,
    letterbox,
    nearest_character,
    preprocess_crop,
    summarize_naming,
)

LABELS = ("body", "face", "frame", "text")


def _raw(rows):
    """Build a (1, 4+nc, N) YOLO output from (cx, cy, w, h, scores...) rows."""
    return np.asarray(rows, dtype=np.float32).T[None]


def test_letterbox_keeps_aspect_and_centres():
    page = np.full((2000, 1000, 3), 200, dtype=np.uint8)
    canvas, scale, left, top = letterbox(page, 640)
    assert canvas.shape == (640, 640, 3) and scale == pytest.approx(0.32)
    assert (left, top) == (160, 0)
    assert canvas[0, 0].tolist() == [114, 114, 114] and canvas[320, 320].tolist() == [200, 200, 200]


def test_decode_filters_nms_and_maps_back_to_page_pixels():
    raw = _raw([
        [100, 100, 40, 40, 0.90, 0.0, 0.0, 0.0],  # body
        [102, 101, 40, 40, 0.60, 0.0, 0.0, 0.0],  # duplicate body -> suppressed
        [300, 200, 20, 20, 0.0, 0.70, 0.0, 0.0],  # face
        [400, 400, 30, 30, 0.20, 0.0, 0.0, 0.0],  # below threshold
    ])
    detections = decode_yolo_output(raw, LABELS, threshold=0.383, scale=0.5, offset=(0, 0), image_size=(1000, 1000))
    assert [(d.label, d.box) for d in detections] == [("body", (160, 160, 240, 240)), ("face", (580, 380, 620, 420))]
    assert detections[0].confidence == pytest.approx(0.90)


def test_decode_clips_to_the_image_and_handles_empty_output():
    raw = _raw([[10, 10, 100, 100, 0.9, 0, 0, 0]])
    (only,) = decode_yolo_output(raw, LABELS, 0.383, scale=1.0, offset=(0, 0), image_size=(50, 50))
    assert only.box == (0, 0, 50, 50)
    assert decode_yolo_output(_raw([[10, 10, 5, 5, 0.1, 0, 0, 0]]), LABELS, 0.383, 1.0, (0, 0), (50, 50)) == []


class _FakeSession:
    def __init__(self):
        self.calls = []

    def get_inputs(self):
        class _Input:
            name = "images"
        return [_Input()]

    def run(self, _outputs, feeds):
        self.calls.append(feeds["images"].shape)
        return [_raw([[320, 320, 64, 64, 0.95, 0, 0, 0]])]


def test_detector_runs_the_session_on_a_letterboxed_tensor():
    session = _FakeSession()
    detector = CharacterDetector(session=session, labels=LABELS, input_size=640, threshold=0.5)
    page = np.full((1280, 1280, 3), 255, dtype=np.uint8)

    detections = detector.detect(page)

    assert session.calls == [(1, 3, 640, 640)]
    assert [(d.label, d.box) for d in detections] == [("body", (576, 576, 704, 704))]
    assert "size=640" in detector.fingerprint and "thr=0.500" in detector.fingerprint
    roundtrip = Detection.from_dict(detections[0].to_dict())
    assert (roundtrip.label, roundtrip.box) == ("body", (576, 576, 704, 704))
    assert roundtrip.confidence == pytest.approx(0.95, abs=1e-4)  # to_dict rounds to 4 decimals


def test_preprocess_crop_shapes_and_normalizes():
    crop = np.random.default_rng(0).integers(0, 255, size=(50, 30, 3), dtype=np.uint8)
    tensor = preprocess_crop(crop)
    assert tensor.shape == (3, 224, 224)
    from mangawhisperer.engines.identity import _IMAGENET_MEAN, _IMAGENET_STD

    denorm = tensor * _IMAGENET_STD[:, None, None] + _IMAGENET_MEAN[:, None, None]
    assert np.allclose(denorm[0], denorm[1], atol=1e-5) and np.allclose(denorm[1], denorm[2], atol=1e-5)  # grayscale replicated
    assert preprocess_crop(crop, grayscale=False).shape == (3, 224, 224)
    assert np.allclose(np.linalg.norm(l2_normalize(np.array([[3.0, 4.0]])), axis=1), 1.0)


def _unit(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)


def test_gallery_names_only_confident_and_unambiguous_matches():
    gallery = Gallery(accept=0.9, margin=0.05, strategy="nearest")
    gallery.add("Guts", _unit(0))
    gallery.add("Guts", _unit(10))
    gallery.add("Casca", _unit(90))

    close = gallery.match(_unit(5))  # cos 5° = 0.996 vs Casca 0.087
    assert (close.name, close.best, close.runner_up) == ("Guts", "Guts", "Casca")
    far = gallery.match(_unit(40))  # cos 30° = 0.87 to Guts@10 < accept; Casca cos 50° = 0.64
    assert far.name is None and far.best == "Guts"
    ambiguous = Gallery(accept=0.5, margin=0.2)
    ambiguous.add("Guts", _unit(0)); ambiguous.add("Casca", _unit(90))
    assert ambiguous.match(_unit(45)).name is None  # tie within the margin
    assert gallery.names == ["Casca", "Guts"] and gallery.count("Guts") == 2
    assert Gallery.from_json(gallery.to_json()).match(_unit(5)).name == "Guts"


def test_gallery_strategies_differ_on_spread_exemplars():
    nearest = Gallery(accept=0.0, margin=0.0, strategy="nearest")
    prototype = Gallery(accept=0.0, margin=0.0, strategy="prototype")
    for gallery in (nearest, prototype):
        gallery.add("Guts", _unit(-40)); gallery.add("Guts", _unit(40))
    assert nearest.scores(_unit(40))["Guts"] == pytest.approx(1.0)
    assert prototype.scores(_unit(40))["Guts"] == pytest.approx(np.cos(np.deg2rad(40)))
    with pytest.raises(ValueError):
        Gallery(strategy="magic")


def test_leave_one_out_and_summary_count_unknowns_and_confusions():
    labelled = [
        ("Guts", _unit(0)), ("Guts", _unit(8)), ("Guts", _unit(-8)),
        ("Casca", _unit(90)), ("Casca", _unit(84)),
        ("Puck", _unit(200)),  # lone exemplar: cannot be named from the others
        (UNKNOWN, _unit(300)),  # an extra: must be rejected
    ]
    results = leave_one_out(labelled, accept=0.9, margin=0.05)
    summary = summarize_naming(results)

    assert summary["known_crops"] == 6 and summary["extras"] == 1
    assert summary["per_name"]["Guts"] == {"total": 3, "correct": 3, "unknown": 0}
    assert summary["per_name"]["Casca"]["correct"] == 2
    assert summary["per_name"]["Puck"] == {"total": 1, "correct": 0, "unknown": 1}
    assert summary["accuracy"] == pytest.approx(5 / 6)
    assert summary["extras_rejected"] == 1.0
    assert summary["confusions"] == {}


def test_geometry_helpers():
    assert containment((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert containment((5, 5, 15, 15), (0, 0, 10, 10)) == pytest.approx(0.25)
    body = Detection("body", 0.9, (10, 10, 50, 90))
    stray = Detection("body", 0.9, (200, 200, 260, 300))
    panels = [(0, 0, 100, 100), (100, 0, 200, 100)]
    assert assign_to_panels([body, stray], panels) == {0: [body]}
    assert nearest_character((0, 0, 10, 10), [(100, 100, 120, 120), (12, 0, 30, 10)]) == 1
    assert nearest_character((0, 0, 10, 10), []) is None
    page = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    assert crop_with_margin(page, (40, 40, 60, 60), margin=0.5).shape == (40, 40, 3)
    assert crop_with_margin(page, (0, 0, 20, 20), margin=1.0).shape == (40, 40, 3)  # clipped at the edge


def test_hint_block_is_advisory_and_numbered():
    from mangawhisperer.engines.identity import Match

    text = format_hint_block([Match("Guts", 0.83, "Guts", "Casca", 0.4), Match(None, 0.5, "Casca", None, 0.0)])
    assert text.startswith("Personagens detectados neste painel (dica automática")
    assert "[1] Guts (0.83)" in text and "[2] desconhecido" in text and "marcas sobre a imagem" in text
    assert format_hint_block([]) == ""
