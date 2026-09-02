"""MangaWhisperer — interface web (Gradio).

Duas abas:
* **Narração** — upload do PDF, provedor VLM, estilo, sonoplastia, revisão,
  mixagem (BGM + volumes) e player com o áudio final. Toda a montagem vem de
  `mangawhisperer.config.build_pipeline` (a mesma do CLI).
* **Biblioteca de sons** — upload de efeitos próprios com sugestão automática
  de tags via CLAP (zero-shot, local).

    python app.py            # abre em http://127.0.0.1:7860
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from mangawhisperer.config import (
    PROJECT_ROOT,
    HardwareProfile,
    PipelineConfig,
    PipelineResources,
    build_pipeline,
)
from mangawhisperer.engines.factory import ALL_PROVIDERS
from mangawhisperer.engines.sfx import SFXLibrary
from mangawhisperer.engines.styles import STYLES
from mangawhisperer.engines.vlm_api import PROVIDER_PRESETS
from mangawhisperer.reporting import load_script, script_markdown

WORKSPACE_ROOT = PROJECT_ROOT / "workspace_ui"
SFX_LIBRARY = SFXLibrary(PROJECT_ROOT / "assets" / "sfx")
BGM_DIR = PROJECT_ROOT / "assets" / "bgm"
AUDIO_TYPES = [".wav", ".mp3", ".ogg", ".flac"]
HARDWARE = HardwareProfile.detect()
RESOURCES = PipelineResources()  # OCR e XTTS carregam uma vez e são reaproveitados


def bgm_choices() -> list[str]:
    tracks = sorted(p.stem for p in BGM_DIR.glob("*.*")) if BGM_DIR.is_dir() else []
    return ["off", *tracks]


def _apply_api_key(provider: str, api_key: str) -> None:
    key = api_key.strip()
    if not key:
        return
    env = "ANTHROPIC_API_KEY" if provider == "anthropic" else (
        PROVIDER_PRESETS[provider].api_key_env if provider in PROVIDER_PRESETS else None
    )
    if env:
        os.environ[env] = key


def narrate(pdf_file, provider, model, api_key, style_name, tts_backend, start_page,
            num_pages, use_sfx, sfx_intensity, announce, use_review, bgm_name,
            gain_voice, gain_sfx, gain_bgm, progress=gr.Progress()):
    if pdf_file is None:
        raise gr.Error("Envie um PDF primeiro.")
    _apply_api_key(provider, api_key)

    config = PipelineConfig(
        pdf_path=Path(pdf_file),
        first_page=int(start_page),
        max_pages=int(num_pages),
        workspace_root=WORKSPACE_ROOT,
        resume=False,  # cada clique é uma execução nova e explícita
        vlm_provider=provider,
        vlm_model=model.strip() or None,
        review=bool(use_review),
        style=style_name,
        tts_backend=tts_backend,
        announce_speakers=bool(announce),
        sfx_intensity=int(sfx_intensity) if use_sfx else 0,
        bgm=bgm_name or "off",
        gain_voice=float(gain_voice),
        gain_sfx=float(gain_sfx),
        gain_bgm=float(gain_bgm),
        hardware=HARDWARE,
    )

    progress(0.05, desc="Montando o pipeline...")
    try:
        pipeline = build_pipeline(config, RESOURCES)
    except (RuntimeError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc

    progress(0.15, desc="Narrando (layout -> OCR -> VLM -> revisão -> TTS -> mix)...")
    final_path = pipeline.run()
    script_md = script_markdown(load_script(config.script_path))
    progress(1.0, desc="Pronto!")
    return str(final_path), script_md, "\n".join(pipeline.report)


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
        tagger.release()  # nunca deixar o CLAP residente ao lado do XTTS
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
    return ("**Adicionados:**\n" + "\n".join(f"- {item}" for item in added)
            + f"\n\nBiblioteca agora tem **{len(tags)} tags**: {', '.join(tags)}")


with gr.Blocks(title="MangaWhisperer") as demo:
    gr.Markdown("# 🎧 MangaWhisperer\nNarração imersiva multi-voz de mangás em PT-BR "
                "para leitores com deficiência visual.")
    gr.Markdown(f"**GPU:** {HARDWARE.describe()}")

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
                report_out = gr.Textbox(label="Configuração usada", lines=7, interactive=False)

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
        outputs=[audio_out, script_out, report_out],
    )
    analyze_btn.click(analyze_uploads, inputs=[uploads_in], outputs=[suggestions_table])
    add_btn.click(add_to_library, inputs=[suggestions_table], outputs=[library_status])

if __name__ == "__main__":
    print("Abrindo o MangaWhisperer no navegador... (Ctrl+C para encerrar)")
    demo.launch(inbrowser=True)
