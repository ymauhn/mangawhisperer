"""MangaWhisperer — interface web (Gradio).

Duas abas:
* **Narração** — upload do PDF, provedor VLM (Qwen API, OpenAI, Kimi,
  Anthropic, Qwen local ou passthrough), estilo, sonoplastia, revisão,
  mixagem (BGM + volumes) e player com o áudio final.
* **Biblioteca de sons** — upload de efeitos próprios com sugestão
  automática de tags via CLAP (zero-shot, local).

    python app.py            # abre em http://127.0.0.1:7860
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import gradio as gr

from mangawhisperer.device import cuda_report
from mangawhisperer.engines.factory import ALL_PROVIDERS, create_reviewer, create_vlm_engine
from mangawhisperer.engines.layout import ClassicalLayoutParser
from mangawhisperer.engines.mixing import MixingStitcher
from mangawhisperer.engines.ocr import EasyOCREngine
from mangawhisperer.engines.pdf import PyMuPDFPageExtractor
from mangawhisperer.engines.sfx import SFXLibrary
from mangawhisperer.engines.styles import STYLES, get_style
from mangawhisperer.engines.tts import DEFAULT_CAST_VOICES, XTTSEngine
from mangawhisperer.engines.vlm_api import PROVIDER_PRESETS
from mangawhisperer.orchestrator import MangaAudioOrchestrator, slugify

WORKSPACE_ROOT = Path("workspace_ui")
SFX_LIBRARY = SFXLibrary(Path(__file__).resolve().parent / "assets" / "sfx")
BGM_DIR = Path(__file__).resolve().parent / "assets" / "bgm"
AUDIO_TYPES = [".wav", ".mp3", ".ogg", ".flac"]


def bgm_choices() -> list[str]:
    tracks = sorted(p.stem for p in BGM_DIR.glob("*.*")) if BGM_DIR.is_dir() else []
    return ["off", *tracks]


def bgm_path_for(name: str):
    if not name or name == "off":
        return None
    matches = list(BGM_DIR.glob(f"{name}.*"))
    return matches[0] if matches else None


# Heavy local models load once and are reused across requests.
layout_parser = ClassicalLayoutParser()
ocr_engine = EasyOCREngine()
tts_engine = XTTSEngine()


@functools.lru_cache(maxsize=8)
def _local_qwen_engine(sfx_tags: tuple, sfx_intensity: int, style_addendum: str):
    return create_vlm_engine(
        "qwen-local", sfx_tags=sfx_tags, sfx_intensity=sfx_intensity,
        style_addendum=style_addendum,
    )


def _vlm_for(provider: str, model: str, api_key: str, sfx_tags: tuple,
             sfx_intensity: int, style_addendum: str):
    model = model.strip() or None
    if provider == "qwen-local":
        return _local_qwen_engine(sfx_tags, sfx_intensity, style_addendum)
    if api_key.strip():
        env = "ANTHROPIC_API_KEY" if provider == "anthropic" else (
            PROVIDER_PRESETS[provider].api_key_env if provider in PROVIDER_PRESETS else None
        )
        if env:
            os.environ[env] = api_key.strip()
    return create_vlm_engine(
        provider, model=model, sfx_tags=sfx_tags, sfx_intensity=sfx_intensity,
        style_addendum=style_addendum,
    )


def narrate(pdf_file, provider, model, api_key, style_name, tts_backend, start_page,
            num_pages, use_sfx, sfx_intensity, announce, use_review, bgm_name,
            gain_voice, gain_sfx, gain_bgm, progress=gr.Progress()):
    if pdf_file is None:
        raise gr.Error("Envie um PDF primeiro.")
    pdf_path = Path(pdf_file)
    start, pages = int(start_page), int(num_pages)
    style = get_style(style_name)
    intensity = int(sfx_intensity) if use_sfx else 0
    sfx_library = SFX_LIBRARY if (intensity > 0 and SFX_LIBRARY.tags()) else None
    sfx_tags = tuple(sfx_library.tags()) if sfx_library else ()

    progress(0.05, desc="Preparando pipeline...")
    if tts_backend == "edge":
        from mangawhisperer.engines.tts_factory import create_tts_engine

        active_tts = create_tts_engine(
            "edge", speed=style.tts_speed, extra_synthesis_kwargs=style.synthesis_kwargs
        )
    else:
        tts_engine.configure(speed=style.tts_speed, extra_synthesis_kwargs=style.synthesis_kwargs)
        active_tts = tts_engine
    vlm_engine = _vlm_for(provider, model, api_key, sfx_tags, intensity, style.prompt_addendum)
    preflight = getattr(vlm_engine, "preflight", None)
    if callable(preflight):
        try:
            preflight()
        except RuntimeError as exc:
            raise gr.Error(str(exc)) from exc

    orchestrator = MangaAudioOrchestrator(
        page_extractor=PyMuPDFPageExtractor(dpi=200, first_page=start, max_pages=pages),
        layout_parser=layout_parser,
        ocr_engine=ocr_engine,
        vlm_engine=vlm_engine,
        tts_engine=active_tts,
        stitcher=MixingStitcher(
            bgm_path=bgm_path_for(bgm_name),
            voice_gain=float(gain_voice),
            sfx_gain=float(gain_sfx),
            bgm_gain=float(gain_bgm),
        ),
        workspace_root=WORKSPACE_ROOT,
        resume=False,  # cada clique é uma execução nova e explícita
        sfx_library=sfx_library,
        sfx_intensity=intensity,
        announce_speakers=bool(announce),
        reviewer=(
            create_reviewer(provider, model=model.strip() or None,
                            known_characters=tuple(DEFAULT_CAST_VOICES), sfx_tags=sfx_tags)
            if use_review else None
        ),
    )

    progress(0.15, desc="Narrando (layout -> OCR -> VLM -> TTS)...")
    final_path = orchestrator.run(pdf_path)

    script_path = WORKSPACE_ROOT / slugify(pdf_path.stem) / "script" / "panels.json"
    panels = json.loads(script_path.read_text(encoding="utf-8"))
    lines = []
    for panel in panels:
        for block in panel["blocks"]:
            icon = "🗣️" if block["is_speech"] else "🎬"
            sfx_note = f" 🔊`{block['sfx']}`" if block.get("sfx") else ""
            lines.append(f"- {icon} **{block['speaker_id']}**{sfx_note}: {block['text']}")
    script_md = "\n".join(lines) or "*Nenhum texto encontrado nas páginas selecionadas.*"

    progress(1.0, desc="Pronto!")
    return str(final_path), script_md


def analyze_uploads(files, progress=gr.Progress()):
    """Roda o CLAP nos uploads e sugere uma tag por arquivo."""
    if not files:
        raise gr.Error("Envie arquivos de áudio primeiro.")
    from mangawhisperer.engines.audio_tagging import ClapAudioTagger

    progress(0.1, desc="Carregando CLAP (1ª vez baixa ~780 MB)...")
    tagger = ClapAudioTagger()
    rows: list[list[str]] = []
    try:
        for index, file_path in enumerate(files):
            path = Path(file_path)
            progress(0.2 + 0.7 * index / len(files), desc=f"Analisando {path.name}...")
            suggestions = tagger.suggest(path)
            if suggestions:
                tag, label, score = suggestions[0]
                rows.append([path.name, tag, f"{label} ({score:.0%})", str(path)])
            else:
                rows.append([path.name, "", "sem sugestão confiável — digite a tag", str(path)])
    finally:
        tagger.release()
    return rows


def add_to_library(rows):
    """Confirma as tags (editáveis) e registra os arquivos na biblioteca."""
    added: list[str] = []
    for row in rows or []:
        name, tag, _note, full_path = (str(c or "").strip() for c in row)
        if not tag or not full_path:
            continue
        SFX_LIBRARY.add_entry(tag.lower(), Path(full_path))
        added.append(f"`{tag.lower()}` ← {name}")
    if not added:
        return "*Nada adicionado — preencha a coluna de tag dos arquivos desejados.*"
    tags = SFX_LIBRARY.tags()
    return (
        "**Adicionados:**\n" + "\n".join(f"- {item}" for item in added)
        + f"\n\nBiblioteca agora tem **{len(tags)} tags**: {', '.join(tags)}"
    )


with gr.Blocks(title="MangaWhisperer") as demo:
    gr.Markdown("# 🎧 MangaWhisperer\nNarração imersiva multi-voz de mangás em PT-BR "
                "para leitores com deficiência visual.")
    gr.Markdown(f"**GPU:** {cuda_report()}")

    with gr.Tab("🎙️ Narração"):
        with gr.Row():
            with gr.Column():
                pdf_in = gr.File(label="PDF do mangá", file_types=[".pdf"], type="filepath")
                provider_in = gr.Dropdown(
                    choices=list(ALL_PROVIDERS), value="qwen", label="Motor de contexto (VLM)",
                    info="qwen/openai/kimi/anthropic usam API (precisa de chave); "
                         "qwen-local roda na sua GPU; passthrough é offline sem diarização.",
                )
                style_in = gr.Dropdown(
                    choices=[(s.label, s.name) for s in STYLES.values()], value="neutro",
                    label="Estilo de narração",
                    info="Ajusta o roteiro (atmosfera), o ritmo do TTS e as pausas.",
                )
                tts_in = gr.Dropdown(
                    choices=[("XTTS v2 (GPU local, vozes do elenco)", "xtts"),
                             ("Edge-TTS (nuvem Microsoft, zero VRAM)", "edge")],
                    value="xtts", label="Motor de voz",
                )
                model_in = gr.Textbox(label="Modelo (opcional)",
                                      placeholder="ex.: qwen-vl-plus, gpt-4.1-mini, claude-haiku-4-5")
                key_in = gr.Textbox(label="API key (opcional — senão usa a variável de ambiente)",
                                    type="password")
                with gr.Row():
                    start_in = gr.Number(value=8, precision=0, label="Primeira página")
                    pages_in = gr.Slider(1, 20, value=3, step=1, label="Nº de páginas")
                sfx_in = gr.Checkbox(
                    value=bool(SFX_LIBRARY.tags()),
                    label=f"Sonoplastia ({len(SFX_LIBRARY.tags())} efeitos disponíveis)",
                    interactive=bool(SFX_LIBRARY.tags()),
                )
                sfx_intensity_in = gr.Slider(
                    1, 3, value=2, step=1, label="Intensidade dos efeitos",
                    info="1 = raros e marcantes · 2 = equilibrado · 3 = agressivo",
                )
                announce_in = gr.Checkbox(
                    value=True, label="Anunciar personagem antes da fala (voz do narrador)"
                )
                review_in = gr.Checkbox(
                    value=True,
                    label="Revisão do roteiro (2ª passada do LLM — consistência e pronúncia)",
                )
                with gr.Accordion("🎚️ Mixagem (BGM e volumes)", open=False):
                    bgm_in = gr.Dropdown(
                        choices=bgm_choices(), value="off", label="Trilha de fundo (BGM)",
                        info="Arquivos de assets/bgm — loop automático sob a narração.",
                    )
                    gain_voice_in = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Volume: narração")
                    gain_sfx_in = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Volume: efeitos")
                    gain_bgm_in = gr.Slider(0.0, 1.0, value=0.22, step=0.02, label="Volume: trilha")
                run_btn = gr.Button("▶ Narrar", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Áudio final", type="filepath")
                script_out = gr.Markdown(label="Roteiro gerado")

    with gr.Tab("📚 Biblioteca de sons"):
        gr.Markdown(
            "Envie seus próprios efeitos (.wav/.mp3/.ogg/.flac). O CLAP analisa o áudio "
            "localmente e sugere a tag; revise/edite a coluna **tag** e confirme. "
            "Tags novas ficam disponíveis para o roteirista na próxima narração."
        )
        uploads_in = gr.Files(label="Arquivos de áudio", file_types=AUDIO_TYPES, type="filepath")
        analyze_btn = gr.Button("🔍 Analisar e sugerir tags")
        suggestions_table = gr.Dataframe(
            headers=["arquivo", "tag (edite se quiser)", "análise CLAP", "caminho"],
            datatype=["str", "str", "str", "str"],
            interactive=True, type="array", label="Sugestões",
        )
        add_btn = gr.Button("➕ Adicionar à biblioteca", variant="primary")
        library_status = gr.Markdown()

    run_btn.click(
        narrate,
        inputs=[pdf_in, provider_in, model_in, key_in, style_in, tts_in, start_in, pages_in,
                sfx_in, sfx_intensity_in, announce_in, review_in, bgm_in,
                gain_voice_in, gain_sfx_in, gain_bgm_in],
        outputs=[audio_out, script_out],
    )
    analyze_btn.click(analyze_uploads, inputs=[uploads_in], outputs=[suggestions_table])
    add_btn.click(add_to_library, inputs=[suggestions_table], outputs=[library_status])

if __name__ == "__main__":
    print("Abrindo o MangaWhisperer no navegador... (Ctrl+C para encerrar)")
    demo.launch(inbrowser=True)  # abre o navegador automaticamente
