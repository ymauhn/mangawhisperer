"""Pure parts of the gabarito labeler (the matplotlib loop is not tested)."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter

from mangawhisperer.models import BoundingBox, ContextualizedBlock, PanelData
from tools.label_benchmark import (
    JUNK_LABEL,
    NARRATOR_LABEL,
    Gabarito,
    UNKNOWN_LABEL,
    ask_text,
    character_items,
    LabelItem,
    first_pdf_page,
    hotkey_legend,
    label_for_key,
    page_items,
    speaker_items,
)

CAST = ("Guts", "Casca")


def _item(key: str, text: str = "") -> LabelItem:
    return LabelItem(key=key, image_path=Path("a.npy"), prompt="?", predicted="Desconhecido", text=text)


def test_label_for_key_maps_cast_narrator_junk_and_ignores_controls():
    assert label_for_key("1", CAST) == "Guts"
    assert label_for_key("2", CAST) == "Casca"
    assert label_for_key("3", CAST) is None  # beyond the cast
    assert label_for_key("0", CAST) == NARRATOR_LABEL
    assert label_for_key("x", CAST) == JUNK_LABEL
    assert label_for_key("s", CAST) is None
    assert label_for_key("ctrl+c", CAST) is None


def test_legend_lists_cast_and_controls():
    legend = hotkey_legend("speakers", CAST)
    assert "1=Guts" in legend and "2=Casca" in legend and "0=Narrador" in legend and "q=sair" in legend
    assert "u=under" in hotkey_legend("panels", CAST)


def test_gabarito_records_resumes_and_reports_pending(tmp_path):
    path = tmp_path / "vol.json"
    items = [_item("page_008_panel00/b0", "Oi"), _item("page_008_panel00/b1", "Tchau")]
    gabarito = Gabarito(path, "vol")
    gabarito.record("speakers", items[0], label="Guts")

    reopened = Gabarito(path, "vol")
    assert reopened.pending("speakers", items) == [items[1]]
    assert reopened.count("speakers") == 1
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["speakers"]["page_008_panel00/b0"] == {"predicted": "Desconhecido", "label": "Guts", "text": "Oi"}
    assert stored["panels"] == {}
    assert not path.with_suffix(".json.tmp").exists()  # atomic save leaves no temp file


def test_gabarito_requeues_a_key_whose_text_changed(tmp_path):
    gabarito = Gabarito(tmp_path / "vol.json", "vol")
    gabarito.record("speakers", _item("page_008_panel00/b0", "Oi"), label="Guts")

    same = _item("page_008_panel00/b0", "Oi")
    changed = _item("page_008_panel00/b0", "Olá")  # a re-run OCR'd the bubble differently
    assert gabarito.pending("speakers", [same]) == []
    assert gabarito.pending("speakers", [changed]) == [changed]


def _write_workspace(workspace: Path, first_page: int = 8) -> None:
    (workspace / "script").mkdir(parents=True)
    (workspace / "panels").mkdir()
    (workspace / "pages").mkdir()
    (workspace / "run_config.json").write_text(json.dumps(
        {"script": {"page_extractor": f"pymupdf:dpi=200:first={first_page}:max=2"}}
    ), encoding="utf-8")
    for stem in ("page_008", "page_009", "page_004"):  # page_004: stale file from an older run
        (workspace / "pages" / f"{stem}.png").write_bytes(b"")

    def panel(page_number: int, panel_index: int, blocks: list[ContextualizedBlock]) -> PanelData:
        path = workspace / "panels" / f"page{page_number:03d}_panel{panel_index:02d}.npy"
        np.save(path, np.zeros((4, 4, 3), dtype=np.uint8))
        return PanelData(
            image_path=path, bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0),
            page_number=page_number, panel_index=panel_index, blocks=blocks,
        )

    panels = [
        panel(1, 0, [
            ContextualizedBlock(text="Guts entra.", speaker_id="Narrator", is_speech=False),
            ContextualizedBlock(text="Vamos.", speaker_id="Desconhecido", is_speech=True),
        ]),
        panel(1, 1, [ContextualizedBlock(text="Agora!", speaker_id="Guts", is_speech=True)]),
        panel(2, 0, [ContextualizedBlock(text="Espere.", speaker_id="Casca", is_speech=True)]),
    ]
    (workspace / "script" / "panels.json").write_bytes(TypeAdapter(list[PanelData]).dump_json(panels))


def test_first_pdf_page_comes_from_the_run_config(tmp_path):
    _write_workspace(tmp_path, first_page=8)
    assert first_pdf_page(tmp_path) == 8
    assert first_pdf_page(tmp_path / "missing") == 1


def test_speaker_items_use_true_page_keys_and_only_speech_blocks(tmp_path):
    _write_workspace(tmp_path)
    items = speaker_items(tmp_path)
    assert [i.key for i in items] == ["page_008_panel00/b1", "page_008_panel01/b0", "page_009_panel00/b0"]
    assert items[0].text == "Vamos." and items[0].predicted == "Desconhecido"
    assert items[0].image_path.is_file()


def test_speaker_items_requires_a_script(tmp_path):
    with pytest.raises(FileNotFoundError):
        speaker_items(tmp_path)


def test_page_items_count_panels_from_the_checkpoint_not_the_directory(tmp_path):
    _write_workspace(tmp_path)
    (tmp_path / "panels" / "page003_panel00.npy").write_bytes(b"")  # stale file from an older run

    items = page_items(tmp_path)

    assert [(i.key, i.predicted) for i in items] == [("page_008", "2"), ("page_009", "1")]
    assert items[0].image_path == tmp_path / "pages" / "page_008.png"


def test_ask_text_falls_back_to_the_console_without_a_tk_window():
    assert ask_text("Nome:", window=None, fallback=lambda prompt: "  Guts ") == "Guts"

    def no_console(prompt):
        raise EOFError

    assert ask_text("Nome:", window=None, fallback=no_console) == ""


def test_character_items_come_from_the_detections_file(tmp_path):
    (tmp_path / "pages").mkdir()
    (tmp_path / "identity").mkdir()
    (tmp_path / "pages" / "page_031.png").write_bytes(b"")
    (tmp_path / "identity" / "detections.json").write_text(json.dumps({
        "detector": "manga109-yolo:test",
        "pages": {"page_031": {"width": 100, "height": 200, "detections": [
            {"label": "body", "confidence": 0.9, "box": [10, 10, 50, 90]},
            {"label": "face", "confidence": 0.8, "box": [20, 15, 40, 35]},
            {"label": "body", "confidence": 0.6, "box": [60, 100, 95, 190]},
        ]}, "page_099": {"width": 1, "height": 1, "detections": [{"label": "body", "confidence": 0.5, "box": [0, 0, 1, 1]}]}},
    }), encoding="utf-8")

    items = character_items(tmp_path)

    assert [i.key for i in items] == ["page_031/body00", "page_031/body01"]  # faces skipped, missing page skipped
    assert items[0].box == (10, 10, 50, 90) and items[0].confidence == 0.9
    assert "corpo 1/2" in items[0].prompt
    with pytest.raises(FileNotFoundError):
        character_items(tmp_path / "elsewhere")


def test_characters_legend_and_gabarito_section():
    assert "0=Desconhecido" in hotkey_legend("characters", CAST) and "x=não é personagem" in hotkey_legend("characters", CAST)
    assert UNKNOWN_LABEL == "Desconhecido"
