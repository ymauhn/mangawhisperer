"""Run the real input stages on an actual Berserk volume.

PDF (PyMuPDF) -> panels/bubbles (classical CV) -> PT-BR text (EasyOCR),
with the VLM and TTS stages stubbed by stdlib placeholders, so the run
produces a real workspace: page PNGs, panel crops, a `panels.json`
script with genuine OCR text, and a playable (silent-beat) WAV.

Usage:
    python scripts/demo_input_stages.py --pages 3 --start 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import TypeAdapter

from mangawhisperer.engines.layout import ClassicalLayoutParser
from mangawhisperer.engines.ocr import EasyOCREngine
from mangawhisperer.engines.pdf import PyMuPDFPageExtractor
from mangawhisperer.engines.placeholders import (
    PassthroughVLM,
    SilentTTSEngine,
    WaveFileStitcher,
)
from mangawhisperer.models import PanelData
from mangawhisperer.orchestrator import MangaAudioOrchestrator, slugify

DEFAULT_PDF = (
    Path(__file__).resolve().parents[1]
    / "data_berserk_samples"
    / "Berserk-20260706T192902Z-3-002"
    / "Berserk"
    / "BERSERK VOL.01.pdf"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Volume PDF to process.")
    parser.add_argument("--start", type=int, default=1, help="1-based first page to render.")
    parser.add_argument("--pages", type=int, default=3, help="Number of pages to process.")
    parser.add_argument("--dpi", type=int, default=200, help="Page render resolution.")
    parser.add_argument(
        "--workspace", type=Path, default=Path("workspace"), help="Workspace root directory."
    )
    parser.add_argument(
        "--vlm",
        choices=("passthrough", "claude"),
        default="passthrough",
        help="Scriptwriter engine: 'claude' does real speaker diarization + action "
        "descriptions via the Claude API (needs credentials); 'passthrough' is offline.",
    )
    args = parser.parse_args()

    if args.vlm == "claude":
        from mangawhisperer.engines.vlm import ClaudeVisionLanguageEngine

        vlm_engine = ClaudeVisionLanguageEngine()
    else:
        vlm_engine = PassthroughVLM()

    orchestrator = MangaAudioOrchestrator(
        page_extractor=PyMuPDFPageExtractor(
            dpi=args.dpi, first_page=args.start, max_pages=args.pages
        ),
        layout_parser=ClassicalLayoutParser(),
        ocr_engine=EasyOCREngine(),
        vlm_engine=vlm_engine,
        tts_engine=SilentTTSEngine(),
        stitcher=WaveFileStitcher(),
        workspace_root=args.workspace,
    )

    print(f"Processing {args.pdf.name} (pages {args.start}..{args.start + args.pages - 1})...")
    final_path = orchestrator.run(args.pdf)

    script_path = args.workspace / slugify(args.pdf.stem) / "script" / "panels.json"
    panels = TypeAdapter(list[PanelData]).validate_json(script_path.read_bytes())

    total_blocks = sum(len(p.blocks) for p in panels)
    print(f"\n{len(panels)} panels, {total_blocks} text blocks extracted:\n")
    for panel in panels:
        header = f"page {panel.page_number:>3}, panel {panel.panel_index}"
        if not panel.blocks:
            print(f"  [{header}] (no text found)")
        for block in panel.blocks:
            print(f"  [{header}] {block.speaker_id}: {block.text}")

    print(f"\nScript checkpoint: {script_path}")
    print(f"Final audio:       {final_path}")
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
