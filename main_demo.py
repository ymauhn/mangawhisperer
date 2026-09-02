"""MangaWhisperer — demo ponta a ponta (1 a 3 páginas).

Orquestra o pipeline completo: PDF -> painéis/balões (visão clássica)
-> OCR PT-BR (EasyOCR, GPU) -> diarização + descrição de ação (VLM
plugável) -> revisão -> TTS multi-voz -> mixagem (SFX + BGM) -> WAV.

Toda a montagem vive em `mangawhisperer.config.build_pipeline`; este
arquivo só traduz flags de linha de comando em um `PipelineConfig`.

    python main_demo.py                              # 3 páginas, --vlm auto
    python main_demo.py --vlm anthropic --style sombrio --pages 2
    python main_demo.py --vlm passthrough --tts silent  # 100% offline
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from mangawhisperer.config import PROJECT_ROOT, PipelineConfig, build_pipeline
from mangawhisperer.engines.factory import ALL_PROVIDERS
from mangawhisperer.engines.styles import STYLES
from mangawhisperer.engines.tts_factory import TTS_BACKENDS
from mangawhisperer.reporting import load_script, summarize_script

DEFAULT_PDF = (
    PROJECT_ROOT / "data_berserk_samples" / "Berserk-20260706T192902Z-3-002"
    / "Berserk" / "BERSERK VOL.01.pdf"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF do volume.")
    parser.add_argument("--start", type=int, default=8, help="Primeira página (1-based).")
    parser.add_argument("--pages", type=int, default=3, help="Quantidade de páginas do teste.")
    parser.add_argument("--dpi", type=int, default=200, help="Resolução de renderização.")
    parser.add_argument("--reading-order", choices=("rtl", "ltr"), default="rtl",
                        help="Ordem de leitura do volume (mangá = rtl).")
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "workspace",
                        help="Raiz do workspace.")
    parser.add_argument("--style", choices=tuple(STYLES), default="neutro",
                        help="Preset de narração: roteiro, ritmo, pausas e BGM sugerida.")
    parser.add_argument("--vlm", choices=(*ALL_PROVIDERS, "auto"), default="auto",
                        help="Provedor do roteirista (auto: chave de API disponível, senão local).")
    parser.add_argument("--model", default=None, help="Override do modelo do provedor.")
    parser.add_argument("--prefer-local", action="store_true",
                        help="Com --vlm auto, escolhe o VLM local antes de qualquer API (ADR-0001).")
    parser.add_argument("--tts", choices=TTS_BACKENDS, default="xtts", help="Motor de voz.")
    parser.add_argument("--speed", type=float, default=None, help="Velocidade (padrão: do estilo).")
    parser.add_argument("--gap-ms", type=int, default=None, help="Silêncio entre segmentos (padrão: do estilo).")
    parser.add_argument("--sfx-intensity", type=int, choices=(0, 1, 2, 3), default=2,
                        help="Frequência de efeitos: 0=off, 1=raro, 2=normal, 3=agressivo.")
    parser.add_argument("--no-sfx", action="store_true", help="Desliga a sonoplastia (= intensidade 0).")
    parser.add_argument("--no-announce", action="store_true", help="Não anunciar o personagem antes das falas.")
    parser.add_argument("--no-review", action="store_true", help="Pula a revisão do roteiro.")
    parser.add_argument("--bgm", default=None, help="Trilha de assets/bgm, 'off', ou vazio = sugestão do estilo.")
    parser.add_argument("--gain-voice", type=float, default=1.0, help="Volume da narração.")
    parser.add_argument("--gain-sfx", type=float, default=1.0, help="Volume dos efeitos.")
    parser.add_argument("--gain-bgm", type=float, default=0.22, help="Volume da trilha de fundo.")
    parser.add_argument("--fresh", action="store_true", help="Ignora checkpoints existentes.")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        pdf_path=args.pdf,
        first_page=args.start,
        max_pages=args.pages,
        dpi=args.dpi,
        reading_order=args.reading_order,
        workspace_root=args.workspace,
        resume=not args.fresh,
        vlm_provider=args.vlm,
        vlm_model=args.model,
        prefer_local=args.prefer_local,
        review=not args.no_review,
        style=args.style,
        tts_backend=args.tts,
        speed=args.speed,
        gap_ms=args.gap_ms,
        announce_speakers=not args.no_announce,
        sfx_intensity=0 if args.no_sfx else args.sfx_intensity,
        bgm=args.bgm,
        gain_voice=args.gain_voice,
        gain_sfx=args.gain_sfx,
        gain_bgm=args.gain_bgm,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = config_from_args(args)
    try:
        pipeline = build_pipeline(config)
    except (RuntimeError, FileNotFoundError) as exc:  # missing credentials / BGM: no traceback
        raise SystemExit(str(exc)) from exc
    print("\n".join(pipeline.report))

    last = config.first_page + (config.max_pages or 0) - 1
    print(f"Narrando {config.pdf_path.name} (páginas {config.first_page}..{last})...")
    started = time.monotonic()
    final_path = pipeline.run()
    elapsed_min = (time.monotonic() - started) / 60

    summary = summarize_script(load_script(config.script_path))
    print(f"\nConcluído em {elapsed_min:.1f} min: " + summary.format(sfx_available=bool(pipeline.sfx_tags)))
    print(f"\nRoteiro (checkpoint): {config.script_path}")
    print(f"Áudio final:          {final_path}")
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
