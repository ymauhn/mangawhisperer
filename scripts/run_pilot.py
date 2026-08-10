"""Narrate one full manga volume with the real engine stack.

PDF (PyMuPDF) -> panels/bubbles (classical CV) -> PT-BR OCR (EasyOCR)
-> scriptwriter VLM (Claude API, or local Qwen-VL as plan B)
-> XTTSv2 multi-voice synthesis -> stitched WAV.

Resume is ON: if the run crashes or the session disconnects, rerunning
the same command picks up after the last completed stage — paid VLM
calls are never repeated once `script/panels.json` exists.

Usage:
    python scripts/run_pilot.py --pdf ".../BERSERK VOL.01.pdf" --vlm auto
    python scripts/run_pilot.py --pages 10          # quick smoke first
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import sys
import time
from pathlib import Path

from pydantic import TypeAdapter

from mangawhisperer.engines.layout import ClassicalLayoutParser
from mangawhisperer.engines.ocr import EasyOCREngine
from mangawhisperer.engines.pdf import PyMuPDFPageExtractor
from mangawhisperer.engines.placeholders import PassthroughVLM, WaveFileStitcher
from mangawhisperer.engines.tts import XTTSEngine
from mangawhisperer.interfaces import VisionLanguageEngine
from mangawhisperer.models import PanelData
from mangawhisperer.orchestrator import MangaAudioOrchestrator, slugify

DEFAULT_PDF = (
    Path(__file__).resolve().parents[1]
    / "data_berserk_samples"
    / "Berserk-20260706T192902Z-3-002"
    / "Berserk"
    / "BERSERK VOL.01.pdf"
)


def build_vlm(choice: str, qwen_model: str) -> VisionLanguageEngine:
    """Resolve the scriptwriter engine via the shared factory."""
    from mangawhisperer.engines.factory import create_vlm_engine

    if choice == "qwen":  # historical alias: this script's 'qwen' meant local
        choice = "qwen-local"
    model = qwen_model if choice == "qwen-local" else None
    return create_vlm_engine(choice, model=model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="Volume PDF to narrate.")
    parser.add_argument("--start", type=int, default=1, help="1-based first page to render.")
    parser.add_argument("--pages", type=int, default=None, help="Page count; default = whole volume.")
    parser.add_argument("--dpi", type=int, default=200, help="Page render resolution.")
    parser.add_argument("--workspace", type=Path, default=Path("workspace"), help="Workspace root.")
    parser.add_argument("--gap-ms", type=int, default=350, help="Silence between narration segments.")
    parser.add_argument(
        "--vlm",
        choices=("auto", "claude", "qwen", "passthrough"),
        default="auto",
        help="Scriptwriter: 'auto' picks claude when ANTHROPIC_API_KEY is set, else qwen.",
    )
    parser.add_argument(
        "--qwen-model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Local VLM model id (use Qwen/Qwen2.5-VL-3B-Instruct on small GPUs).",
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore existing checkpoints instead of resuming.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    orchestrator = MangaAudioOrchestrator(
        page_extractor=PyMuPDFPageExtractor(dpi=args.dpi, first_page=args.start, max_pages=args.pages),
        layout_parser=ClassicalLayoutParser(),
        ocr_engine=EasyOCREngine(),
        vlm_engine=build_vlm(args.vlm, args.qwen_model),
        tts_engine=XTTSEngine(),
        stitcher=WaveFileStitcher(),
        workspace_root=args.workspace,
        panel_gap_ms=args.gap_ms,
        resume=not args.fresh,
    )

    started = time.monotonic()
    print(f"Narrating {args.pdf.name} "
          f"({'whole volume' if args.pages is None else f'pages {args.start}..{args.start + args.pages - 1}'})...")
    final_path = orchestrator.run(args.pdf)
    elapsed_min = (time.monotonic() - started) / 60

    script_path = args.workspace / slugify(args.pdf.stem) / "script" / "panels.json"
    panels = TypeAdapter(list[PanelData]).validate_json(script_path.read_bytes())
    blocks = [b for p in panels for b in p.blocks]
    by_speaker = collections.Counter(b.speaker_id for b in blocks)

    print(f"\nDone in {elapsed_min:.1f} min: {len(panels)} panels, {len(blocks)} narration blocks")
    print("Blocks per speaker:")
    for speaker, count in by_speaker.most_common():
        print(f"  {speaker:>14}: {count}")
    print(f"\nScript checkpoint: {script_path}")
    print(f"Final audio:       {final_path}")
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
