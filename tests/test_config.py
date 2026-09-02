"""PipelineConfig (ticket #9): style-derived defaults, shared invariants
and the single assembly path used by CLI and UI."""

import pytest
from pydantic import ValidationError

from mangawhisperer.config import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    HardwareProfile,
    PipelineConfig,
    PipelineResources,
    build_pipeline,
    local_provider_for,
    xtts_can_stay_resident,
)
from mangawhisperer.constants import AUDIO_SAMPLE_RATE as LEAF_SAMPLE_RATE
from mangawhisperer.engines.styles import get_style

CPU = HardwareProfile()


def _config(tmp_path, **overrides) -> PipelineConfig:
    defaults = dict(
        pdf_path=tmp_path / "BERSERK VOL.01.pdf",
        workspace_root=tmp_path / "ws",
        sfx_dir=tmp_path / "sfx",
        bgm_dir=tmp_path / "bgm",
        hardware=CPU,
    )
    return PipelineConfig(**{**defaults, **overrides})


def test_audio_invariants_are_shared_by_every_engine():
    from mangawhisperer.engines import mixing, placeholders, sfx, tts_edge

    assert AUDIO_SAMPLE_RATE == LEAF_SAMPLE_RATE == 24_000
    assert AUDIO_CHANNELS == 1
    assert (mixing.SAMPLE_RATE, sfx.TARGET_RATE, tts_edge.TARGET_RATE, placeholders.SAMPLE_RATE) == (
        AUDIO_SAMPLE_RATE,
    ) * 4


def test_style_supplies_defaults_but_explicit_values_win(tmp_path):
    sombrio = get_style("sombrio")
    config = _config(tmp_path, style="sombrio")
    assert config.effective_speed == sombrio.tts_speed
    assert config.effective_gap_ms == sombrio.gap_ms
    assert config.effective_bgm_name == sombrio.suggested_bgm

    custom = _config(tmp_path, style="sombrio", speed=1.3, gap_ms=100, bgm="off")
    assert (custom.effective_speed, custom.effective_gap_ms, custom.effective_bgm_name) == (1.3, 100, None)


def test_volume_paths_follow_the_pdf_name(tmp_path):
    config = _config(tmp_path)
    assert config.volume_slug == "berserk_vol_01"
    assert config.script_path == tmp_path / "ws" / "berserk_vol_01" / "script" / "panels.json"


@pytest.mark.parametrize(
    "field, value",
    [("sfx_intensity", 4), ("reading_order", "top"), ("first_page", 0), ("gain_bgm", -1.0)],
)
def test_invalid_values_are_rejected(tmp_path, field, value):
    with pytest.raises(ValidationError):
        _config(tmp_path, **{field: value})


def test_config_is_immutable(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValidationError):
        config.style = "epico"


def test_explicit_missing_bgm_is_an_error_but_style_suggestion_is_soft(tmp_path):
    with pytest.raises(FileNotFoundError):
        _config(tmp_path, bgm="inexistente").resolve_bgm_path()
    assert _config(tmp_path, style="sombrio").resolve_bgm_path() is None
    assert _config(tmp_path, bgm="off").resolve_bgm_path() is None

    (tmp_path / "bgm").mkdir()
    (tmp_path / "bgm" / "tensa.wav").write_bytes(b"")
    assert _config(tmp_path, bgm="tensa").resolve_bgm_path() == tmp_path / "bgm" / "tensa.wav"


def test_hardware_profile_describes_the_8gb_rule():
    laptop = HardwareProfile(device="cuda", gpu_name="RTX 5060 Laptop", vram_gb=8.0)
    assert not laptop.heavy_models_coexist
    assert "NÃO podem" in laptop.describe()
    assert HardwareProfile().describe().startswith("CPU")


def test_resources_call_each_factory_once():
    resources = PipelineResources()
    calls: list[int] = []
    first = resources.get("k", lambda: calls.append(1) or object())
    assert resources.get("k", lambda: calls.append(2) or object()) is first
    assert calls == [1]


def test_resources_evict_releases_the_engine():
    class Engine:
        released = False

        def release(self):
            self.released = True

    resources = PipelineResources()
    engine = resources.get("tts:xtts", Engine)
    assert resources.evict("tts:xtts") is True
    assert engine.released
    assert resources.evict("tts:xtts") is False  # already gone
    assert resources.get("tts:xtts", Engine) is not engine  # rebuilt fresh


def test_xtts_never_stays_resident_beside_a_local_vlm_on_8gb():
    laptop = HardwareProfile(device="cuda", gpu_name="RTX 5060 Laptop", vram_gb=8.0)
    workstation = HardwareProfile(device="cuda", gpu_name="A6000", vram_gb=48.0, heavy_models_coexist=True)
    assert not xtts_can_stay_resident(laptop, "qwen-local")
    assert not xtts_can_stay_resident(laptop, "llamacpp")  # spawned llama-server holds VRAM too
    assert xtts_can_stay_resident(laptop, "anthropic")  # API scriptwriter: nothing else on the GPU
    assert xtts_can_stay_resident(workstation, "qwen-local")


def test_prefer_local_picks_llama_server_when_a_gguf_or_server_is_configured():
    assert local_provider_for({}) == "qwen-local"
    assert local_provider_for({"LLAMA_MODEL_GGUF": "C:/models/qwen3-vl-8b-q4.gguf"}) == "llamacpp"
    assert local_provider_for({"LLAMA_SERVER_URL": "http://127.0.0.1:8080"}) == "llamacpp"
    assert local_provider_for({}, vlm_model="models/gemma-4-e4b.GGUF") == "llamacpp"
    assert local_provider_for({"LLAMA_MODEL_GGUF": ""}, vlm_model="qwen2.5-vl-7b") == "qwen-local"


def test_build_pipeline_offline_assembles_in_order_and_reuses_resources(tmp_path):
    resources = PipelineResources()
    fake_ocr = object()
    resources.get("ocr", lambda: fake_ocr)
    config = _config(
        tmp_path, vlm_provider="passthrough", tts_backend="silent", review=False,
        bgm="off", sfx_intensity=0, style="sombrio",
    )

    pipeline = build_pipeline(config, resources)

    assert [line.split(" ")[0] for line in pipeline.report] == [
        "[GPU]", "[STY]", "[SFX]", "[BGM]", "[VLM]", "[REV]", "[TTS]",
    ]
    assert pipeline.report[1].startswith(f"[STY] estilo: {get_style('sombrio').label}")
    assert pipeline.report[2] == "[SFX] sonoplastia: desligada"
    assert pipeline.report[3].startswith("[BGM] trilha de fundo: desligada")
    assert pipeline.report[5] == "[REV] revisor: desligado"
    assert pipeline.sfx_tags == () and pipeline.bgm_path is None
    assert pipeline.orchestrator._ocr_engine is fake_ocr
    assert pipeline.orchestrator._layout_parser.fingerprint.endswith("rtl")


def test_build_pipeline_fails_fast_without_api_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config = _config(tmp_path, vlm_provider="qwen", tts_backend="silent", sfx_intensity=0)
    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        build_pipeline(config, PipelineResources())
