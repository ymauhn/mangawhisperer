"""MangaWhisperer — demo ponta a ponta (1 a 3 páginas).

Orquestra o pipeline completo: PDF -> painéis/balões (visão clássica)
-> OCR PT-BR (EasyOCR, GPU) -> diarização + descrição de ação (VLM
plugável: Qwen API por padrão; OpenAI/Kimi/Anthropic/local à escolha)
-> XTTSv2 multi-voz (GPU) -> WAV final costurado.

Comece pequeno (o padrão são 3 páginas) e escale só depois de validar
o resultado e o custo:

    python main_demo.py                              # 3 páginas, --vlm auto
    python main_demo.py --vlm qwen --pages 2
    python main_demo.py --vlm openai --model gpt-4.1-mini
    python main_demo.py --vlm passthrough            # 100% offline, sem VLM
"""

from __future__ import annotations

import argparse
import collections
import logging
import sys
import time
from pathlib import Path

from pydantic import TypeAdapter

from mangawhisperer.device import cuda_report
from mangawhisperer.engines.factory import ALL_PROVIDERS, create_reviewer, create_vlm_engine
from mangawhisperer.engines.layout import ClassicalLayoutParser
from mangawhisperer.engines.ocr import EasyOCREngine
from mangawhisperer.engines.mixing import MixingStitcher
from mangawhisperer.engines.pdf import PyMuPDFPageExtractor
from mangawhisperer.engines.sfx import SFXLibrary
from mangawhisperer.engines.styles import STYLES, get_style
from mangawhisperer.engines.tts_factory import TTS_BACKENDS, create_tts_engine
from mangawhisperer.models import PanelData
from mangawhisperer.orchestrator import MangaAudioOrchestrator, slugify

SFX_DIR = Path(__file__).resolve().parent / "assets" / "sfx"
BGM_DIR = Path(__file__).resolve().parent / "assets" / "bgm"


def resolve_bgm(name: str | None) -> Path | None:
    """Resolve --bgm <nome> para um arquivo em assets/bgm/."""
    if not name or name.lower() == "off":
        return None
    for suffix in (".wav", ".ogg", ".mp3", ".flac"):
        candidate = BGM_DIR / f"{name}{suffix}"
        if candidate.is_file():
            return candidate
    available = sorted(p.stem for p in BGM_DIR.glob("*.*")) if BGM_DIR.is_dir() else []
    raise SystemExit(f"BGM '{name}' não encontrada. Disponíveis: {', '.join(available) or 'nenhuma'}")

DEFAULT_PDF = (
    Path(__file__).resolve().parent
    / "data_berserk_samples"
    / "Berserk-20260706T192902Z-3-002"
    / "Berserk"
    / "BERSERK VOL.01.pdf"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="PDF do volume.")
    parser.add_argument("--start", type=int, default=8, help="Primeira página (1-based).")
    parser.add_argument("--pages", type=int, default=3, help="Quantidade de páginas do teste.")
    parser.add_argument("--dpi", type=int, default=200, help="Resolução de renderização.")
    parser.add_argument("--workspace", type=Path, default=Path("workspace"), help="Raiz do workspace.")
    parser.add_argument(
        "--style", choices=tuple(STYLES), default="neutro",
        help="Preset de narração: ajusta roteiro, ritmo do TTS, pausas e BGM sugerida.",
    )
    parser.add_argument("--gap-ms", type=int, default=None,
                        help="Silêncio entre segmentos (padrão: do estilo).")
    parser.add_argument(
        "--vlm", choices=(*ALL_PROVIDERS, "auto"), default="auto",
        help="Provedor do motor de contexto (padrão: auto — Qwen API se houver DASHSCOPE_API_KEY).",
    )
    parser.add_argument("--model", default=None, help="Override do modelo do provedor escolhido.")
    parser.add_argument("--fresh", action="store_true", help="Ignora checkpoints existentes.")
    parser.add_argument("--no-sfx", action="store_true", help="Desliga a sonoplastia (= intensidade 0).")
    parser.add_argument(
        "--sfx-intensity", type=int, choices=(0, 1, 2, 3), default=2,
        help="Frequência de efeitos: 0=off, 1=raro, 2=normal, 3=agressivo.",
    )
    parser.add_argument(
        "--no-announce", action="store_true",
        help="Não anunciar o nome do personagem antes das falas.",
    )
    parser.add_argument(
        "--no-review", action="store_true",
        help="Pula a 2ª passada de revisão do roteiro (Reviewer Agent).",
    )
    parser.add_argument("--speed", type=float, default=None,
                        help="Velocidade da narração (padrão: do estilo).")
    parser.add_argument("--tts", choices=TTS_BACKENDS, default="xtts",
                        help="Motor de voz: xtts (GPU local), edge (nuvem MS) ou silent.")
    parser.add_argument("--bgm", default=None,
                        help="Trilha de assets/bgm, 'off', ou vazio = sugestão do estilo.")
    parser.add_argument("--gain-voice", type=float, default=1.0, help="Volume da narração.")
    parser.add_argument("--gain-sfx", type=float, default=1.0, help="Volume dos efeitos.")
    parser.add_argument("--gain-bgm", type=float, default=0.22, help="Volume da trilha de fundo.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(f"[GPU] {cuda_report()}")

    # 1) Estilo primeiro: ele fornece os padrões (ritmo, pausas, BGM sugerida).
    style = get_style(args.style)
    speed = args.speed if args.speed is not None else style.tts_speed
    gap_ms = args.gap_ms if args.gap_ms is not None else style.gap_ms
    bgm_name = args.bgm if args.bgm is not None else style.suggested_bgm
    print(f"[STY] estilo: {style.label} (speed={speed}, gap={gap_ms}ms)")

    # 2) Sonoplastia.
    sfx_intensity = 0 if args.no_sfx else args.sfx_intensity
    sfx_library = None if sfx_intensity == 0 else SFXLibrary(SFX_DIR)
    sfx_tags = tuple(sfx_library.tags()) if sfx_library else ()
    if sfx_library and not sfx_tags:
        sfx_library = None
        sfx_tags = ()
    print(f"[SFX] sonoplastia: "
          f"{f'nível {sfx_intensity} — ' + ', '.join(sfx_tags) if sfx_tags else 'desligada'}")

    # 3) Trilha de fundo (a sugestão do estilo pode não existir na pasta).
    try:
        bgm_path = resolve_bgm(bgm_name)
    except SystemExit:
        if args.bgm is not None:
            raise  # o usuário pediu explicitamente uma trilha que não existe
        print(f"[BGM] sugestão do estilo ('{bgm_name}') não encontrada — seguindo sem trilha")
        bgm_path = None
    print(f"[BGM] trilha de fundo: {bgm_path.stem if bgm_path else 'desligada'}"
          f" (ganhos voz={args.gain_voice} sfx={args.gain_sfx} bgm={args.gain_bgm})")

    vlm_engine = create_vlm_engine(
        args.vlm, model=args.model, sfx_tags=sfx_tags, sfx_intensity=sfx_intensity,
        style_addendum=style.prompt_addendum,
    )
    provider_label = getattr(vlm_engine, "provider", type(vlm_engine).__name__)
    print(f"[VLM] motor de contexto: {provider_label}"
          f" (modelo: {getattr(vlm_engine, 'model', 'n/a')})")

    preflight = getattr(vlm_engine, "preflight", None)
    if callable(preflight):
        preflight()  # falha AQUI (com mensagem clara) se faltar a chave de API

    from mangawhisperer.engines.tts import DEFAULT_CAST_VOICES
    reviewer = None
    if not args.no_review:
        reviewer = create_reviewer(
            args.vlm, model=args.model,
            known_characters=tuple(DEFAULT_CAST_VOICES), sfx_tags=sfx_tags,
        )
    print(f"[REV] revisor de roteiro: "
          f"{getattr(reviewer, 'model', 'desligado') if reviewer else 'desligado'}")

    orchestrator = MangaAudioOrchestrator(
        page_extractor=PyMuPDFPageExtractor(dpi=args.dpi, first_page=args.start, max_pages=args.pages),
        layout_parser=ClassicalLayoutParser(),
        ocr_engine=EasyOCREngine(),
        vlm_engine=vlm_engine,
        tts_engine=create_tts_engine(
            args.tts, speed=speed, extra_synthesis_kwargs=style.synthesis_kwargs
        ),
        stitcher=MixingStitcher(
            bgm_path=bgm_path,
            voice_gain=args.gain_voice,
            sfx_gain=args.gain_sfx,
            bgm_gain=args.gain_bgm,
        ),
        workspace_root=args.workspace,
        panel_gap_ms=gap_ms,
        resume=not args.fresh,
        sfx_library=sfx_library,
        sfx_intensity=sfx_intensity,
        announce_speakers=not args.no_announce,
        reviewer=reviewer,
    )

    started = time.monotonic()
    print(f"Narrando {args.pdf.name} (páginas {args.start}..{args.start + args.pages - 1})...")
    final_path = orchestrator.run(args.pdf)
    elapsed_min = (time.monotonic() - started) / 60

    script_path = args.workspace / slugify(args.pdf.stem) / "script" / "panels.json"
    panels = TypeAdapter(list[PanelData]).validate_json(script_path.read_bytes())
    blocks = [b for p in panels for b in p.blocks]
    by_speaker = collections.Counter(b.speaker_id for b in blocks)

    print(f"\nConcluído em {elapsed_min:.1f} min: {len(panels)} painéis, {len(blocks)} blocos")
    print("Blocos por personagem:")
    for speaker, count in by_speaker.most_common():
        print(f"  {speaker:>14}: {count}")

    sfx_used = collections.Counter(b.sfx for b in blocks if b.sfx)
    if sfx_used:
        print("Efeitos sonoros no roteiro: "
              + ", ".join(f"{tag}×{n}" for tag, n in sfx_used.most_common()))
    elif sfx_tags:
        print("Efeitos sonoros no roteiro: nenhum (nem o roteirista nem o "
              "tagger automático encontraram cena compatível)")
    print(f"\nRoteiro (checkpoint): {script_path}")
    print(f"Áudio final:          {final_path}")
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
