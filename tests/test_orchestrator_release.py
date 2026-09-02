"""The VLM is released even when the script stage fails: a spawned
llama-server (ADR-0004) must not survive an exception holding the GPU."""

from pathlib import Path

import numpy as np
import pytest

from mangawhisperer.models import BoundingBox
from mangawhisperer.orchestrator import MangaAudioOrchestrator


class _Extractor:
    def extract_pages(self, pdf_path: Path, output_dir: Path) -> list[Path]:
        page = output_dir / "page_001.png"
        page.write_bytes(b"")
        return [page]


class _Layout:
    def extract_panels(self, page_image):
        return [BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)]

    def extract_bubbles(self, panel_image):
        return []


class _ExplodingVLM:
    released = False

    def contextualize(self, panel_image, bubbles):
        raise RuntimeError("servidor caiu no meio do painel")

    def release(self) -> None:
        self.released = True


def test_release_runs_even_when_the_script_stage_raises(tmp_path):
    pdf = tmp_path / "vol.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    vlm = _ExplodingVLM()
    orchestrator = MangaAudioOrchestrator(
        page_extractor=_Extractor(),
        layout_parser=_Layout(),
        ocr_engine=object(),
        vlm_engine=vlm,
        tts_engine=object(),
        stitcher=object(),
        workspace_root=tmp_path / "ws",
        image_loader=lambda path: np.full((8, 8, 3), 255, dtype=np.uint8),
    )

    with pytest.raises(RuntimeError, match="servidor caiu"):
        orchestrator.run(pdf)

    assert vlm.released
