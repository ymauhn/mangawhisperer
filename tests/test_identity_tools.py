"""Pure parts of the identity tools (detection dump/overlay, evaluation report)."""

import json

import numpy as np

from mangawhisperer.engines.identity import UNKNOWN, Detection
from tools.detect_characters import detections_to_json, draw_overlay, run
from tools.eval_identity import labelled_crops, report_markdown, sweep


def _unit(angle_deg: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    return np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)


def test_detections_json_roundtrip_and_overlay_shape():
    dets = [Detection("body", 0.9, (10, 10, 50, 90)), Detection("text", 0.7, (60, 60, 80, 70))]
    payload = detections_to_json("manga109-yolo:test", {"page_031": (100, 200, dets)})
    assert payload["pages"]["page_031"]["detections"][0] == {"label": "body", "confidence": 0.9, "box": [10, 10, 50, 90]}
    page = np.full((200, 100, 3), 255, dtype=np.uint8)
    overlay = draw_overlay(page, dets)
    assert overlay.shape == page.shape and not np.array_equal(overlay, page)


class _FakeDetector:
    fingerprint = "fake-detector"

    def detect(self, page):
        return [Detection("body", 0.8, (0, 0, page.shape[1] // 2, page.shape[0] // 2))]


def test_run_writes_detections_and_overlays(tmp_path):
    from PIL import Image

    (tmp_path / "pages").mkdir()
    Image.fromarray(np.full((40, 30, 3), 255, dtype=np.uint8)).save(tmp_path / "pages" / "page_008.png")

    output = run(tmp_path, input_size=64, overlays=True, detector=_FakeDetector())

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["detector"] == "fake-detector"
    assert data["pages"]["page_008"]["detections"][0]["box"] == [0, 0, 15, 20]
    assert (tmp_path / "identity" / "overlays" / "page_008.png").is_file()


def test_labelled_crops_keeps_extras_and_names_with_enough_exemplars():
    gabarito = {"characters": {
        "page_031/body00": {"label": "Guts", "box": [0, 0, 10, 10]},
        "page_031/body01": {"label": "Guts", "box": [0, 0, 10, 10]},
        "page_032/body00": {"label": "Puck", "box": [0, 0, 10, 10]},  # lone exemplar: excluded
        "page_032/body01": {"label": UNKNOWN, "box": [0, 0, 10, 10]},
        "page_032/body02": {"label": "__lixo__", "box": []},  # no box: excluded
    }}
    kept = labelled_crops(gabarito, min_per_name=2)
    assert [e["key"] for e in kept] == ["page_031/body00", "page_031/body01", "page_032/body01"]


def test_sweep_and_report_render():
    labelled = [("Guts", _unit(0)), ("Guts", _unit(5)), ("Casca", _unit(90)), ("Casca", _unit(95)), (UNKNOWN, _unit(200))]
    rows = sweep(labelled, [0.5, 0.999], margin=0.05, strategy="nearest")
    assert rows[0]["accuracy"] == 1.0 and rows[1]["accuracy"] == 0.0  # cos 5 graus = 0.996 < 0.999
    from mangawhisperer.engines.identity import leave_one_out, summarize_naming

    summary = summarize_naming(leave_one_out(labelled, accept=0.5, margin=0.05))
    report = report_markdown("vol", "fake", summary, rows, "nearest", 0.05)
    assert "Acurácia de nomeação: 100.0%" in report and "| Guts | 2 | 2 | 0 |" in report and "| 1.00 |" in report
