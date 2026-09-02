"""Single source of truth for assembling the pipeline (ticket #9).

Every entrypoint (CLI, web UI, notebooks) builds a :class:`PipelineConfig`
and hands it to :func:`build_pipeline`. The assembly order — style ➜ sound
effects ➜ background music ➜ engines ➜ orchestrator — lives here exactly
once, so the "used before assigned" class of bug can't come back through a
second copy of the wiring.

Also home of the project invariants that several engines silently shared:
the audio format (24 kHz mono 16-bit) and the hardware profile (8 GB VRAM
floor; heavy models never coexist on the GPU).
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mangawhisperer.constants import (  # noqa: F401 — re-exported audio invariants
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    AUDIO_SAMPLE_WIDTH_BYTES,
)
from mangawhisperer.orchestrator import MangaAudioOrchestrator, slugify

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HardwareProfile(BaseModel):
    """What the local machine can hold — drives model residency decisions."""

    model_config = ConfigDict(frozen=True)

    device: Literal["cuda", "cpu"] = "cpu"
    gpu_name: str = "n/a"
    vram_gb: float = 0.0
    heavy_models_coexist: bool = Field(
        default=False,
        description="False on ≤8 GB: the VLM must be released before the TTS loads.",
    )
    offload_dir: Path = Field(
        default=PROJECT_ROOT / "workspace" / "_offload",
        description="Scratch dir for CPU/disk offload of model layers (llama.cpp, accelerate).",
    )

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Probe torch/CUDA; falls back to a CPU profile without torch."""
        try:
            import torch  # noqa: PLC0415 — keep module import light

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / 1024**3
                return cls(
                    device="cuda",
                    gpu_name=torch.cuda.get_device_name(0),
                    vram_gb=round(vram_gb, 1),
                    heavy_models_coexist=vram_gb >= 20,
                )
        except Exception:  # torch missing or CUDA probe failed
            pass
        return cls()

    def describe(self) -> str:
        if self.device == "cuda":
            return (f"CUDA: {self.gpu_name} ({self.vram_gb:.0f} GB) — "
                    f"modelos pesados {'podem' if self.heavy_models_coexist else 'NÃO podem'} coexistir")
        return "CPU (sem CUDA) — caminho degradado"


class PipelineConfig(BaseModel):
    """Everything a run needs, validated once, with style-derived defaults.

    ``speed``/``gap_ms``/``bgm`` accept ``None`` meaning "take it from the
    style preset"; explicit values always win. ``bgm="off"`` disables the
    music bed even if the style suggests one.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Source
    pdf_path: Path
    first_page: int = Field(default=1, ge=1)
    max_pages: int | None = Field(default=None, ge=1)
    dpi: int = Field(default=200, gt=0)
    reading_order: Literal["rtl", "ltr"] = "rtl"

    # Where things live
    workspace_root: Path = PROJECT_ROOT / "workspace"
    sfx_dir: Path = PROJECT_ROOT / "assets" / "sfx"
    bgm_dir: Path = PROJECT_ROOT / "assets" / "bgm"
    resume: bool = True

    # Scriptwriter / reviewer
    vlm_provider: str = "auto"
    vlm_model: str | None = None
    prefer_local: bool = Field(
        default=False,
        description="ADR-0001 policy switch: True makes 'auto' pick the local VLM before any "
        "API key. Default stays False until the local scriptwriter passes the benchmark (ticket #17).",
    )
    review: bool = True

    # Narration
    style: str = "neutro"
    tts_backend: str = "xtts"
    speed: float | None = None
    gap_ms: int | None = Field(default=None, ge=0)
    announce_speakers: bool = True
    sfx_intensity: int = Field(default=2, ge=0, le=3)

    # Mix
    bgm: str | None = None
    gain_voice: float = Field(default=1.0, ge=0.0)
    gain_sfx: float = Field(default=1.0, ge=0.0)
    gain_bgm: float = Field(default=0.22, ge=0.0)

    hardware: HardwareProfile = Field(default_factory=HardwareProfile.detect)

    # ── derived ────────────────────────────────────────────────────────
    @property
    def style_preset(self):
        from mangawhisperer.engines.styles import get_style  # noqa: PLC0415

        return get_style(self.style)

    @property
    def effective_speed(self) -> float:
        return self.speed if self.speed is not None else self.style_preset.tts_speed

    @property
    def effective_gap_ms(self) -> int:
        return self.gap_ms if self.gap_ms is not None else self.style_preset.gap_ms

    @property
    def effective_bgm_name(self) -> str | None:
        """``None`` = no music; the style's suggestion is only a soft default."""
        if self.bgm is None:
            return self.style_preset.suggested_bgm
        return None if self.bgm.lower() == "off" else self.bgm

    @property
    def volume_slug(self) -> str:
        return slugify(self.pdf_path.stem)

    @property
    def script_path(self) -> Path:
        return self.workspace_root / self.volume_slug / "script" / "panels.json"

    def resolve_bgm_path(self) -> Path | None:
        """Locate the music file; an explicit missing name is an error, a
        missing style suggestion just means no music."""
        name = self.effective_bgm_name
        if not name:
            return None
        for suffix in (".wav", ".ogg", ".mp3", ".flac"):
            candidate = self.bgm_dir / f"{name}{suffix}"
            if candidate.is_file():
                return candidate
        if self.bgm is not None:  # user asked for it explicitly
            available = sorted(p.stem for p in self.bgm_dir.glob("*.*")) if self.bgm_dir.is_dir() else []
            raise FileNotFoundError(
                f"BGM '{name}' não encontrada em {self.bgm_dir}. Disponíveis: "
                f"{', '.join(available) or 'nenhuma'}"
            )
        logger.info("Sugestão de BGM do estilo ('%s') não encontrada — sem trilha", name)
        return None


class PipelineResources:
    """Cache of heavy, reusable engines for long-lived processes (the UI).

    The CLI builds fresh engines per run; the web app keeps one OCR reader
    and one XTTS model alive across requests and only re-configures them.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, key: str, factory):
        if key not in self._cache:
            self._cache[key] = factory()
        return self._cache[key]

    def evict(self, key: str) -> bool:
        """Drop a cached engine and free its GPU memory (``release()`` when
        it has one, then a CUDA cache flush). Returns whether it existed."""
        engine = self._cache.pop(key, None)
        if engine is None:
            return False
        release = getattr(engine, "release", None)
        if callable(release):
            release()
        del engine
        gc.collect()
        try:
            import torch  # noqa: PLC0415 — optional at runtime

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # torch absent or no CUDA: nothing to flush
            pass
        return True


def xtts_can_stay_resident(hardware: HardwareProfile, vlm_provider: str) -> bool:
    """The 8 GB rule: a cached XTTS may not sit beside a local VLM unless
    the GPU can hold both. XTTS loads lazily, so a fresh engine built
    here only touches the GPU after the orchestrator released the VLM."""
    return hardware.heavy_models_coexist or vlm_provider != "qwen-local"


@dataclass
class BuiltPipeline:
    """A ready-to-run orchestrator plus the facts worth showing the user."""

    config: PipelineConfig
    orchestrator: MangaAudioOrchestrator
    sfx_tags: tuple[str, ...]
    bgm_path: Path | None
    report: list[str] = field(default_factory=list)

    def run(self) -> Path:
        return self.orchestrator.run(self.config.pdf_path)


def build_pipeline(config: PipelineConfig, resources: PipelineResources | None = None) -> BuiltPipeline:
    """Assemble every engine in the one correct order.

    Args:
        config: The validated run configuration.
        resources: Optional cache so a UI can reuse OCR/TTS across runs.

    Raises:
        RuntimeError: When the chosen API provider has no credentials
            (fails here, before any expensive work).
        FileNotFoundError: When an explicitly requested BGM is missing.
    """
    from mangawhisperer.engines.factory import create_reviewer, create_vlm_engine
    from mangawhisperer.engines.layout import ClassicalLayoutParser
    from mangawhisperer.engines.mixing import MixingStitcher
    from mangawhisperer.engines.ocr import EasyOCREngine
    from mangawhisperer.engines.pdf import PyMuPDFPageExtractor
    from mangawhisperer.engines.sfx import SFXLibrary
    from mangawhisperer.engines.tts import DEFAULT_CAST_VOICES
    from mangawhisperer.engines.tts_factory import create_tts_engine

    resources = resources or PipelineResources()
    report: list[str] = [f"[GPU] {config.hardware.describe()}"]

    # 1) Style — provides the defaults everything below may inherit.
    style = config.style_preset
    report.append(f"[STY] estilo: {style.label} (speed={config.effective_speed}, "
                  f"gap={config.effective_gap_ms}ms)")

    # 2) Sound effects.
    sfx_library = SFXLibrary(config.sfx_dir) if config.sfx_intensity > 0 else None
    sfx_tags = tuple(sfx_library.tags()) if sfx_library else ()
    if sfx_library and not sfx_tags:
        sfx_library = None
    report.append("[SFX] sonoplastia: "
                  + (f"nível {config.sfx_intensity} — {', '.join(sfx_tags)}" if sfx_tags else "desligada"))

    # 3) Background music (needs style + explicit override resolved).
    bgm_path = config.resolve_bgm_path()
    report.append(f"[BGM] trilha de fundo: {bgm_path.stem if bgm_path else 'desligada'} "
                  f"(ganhos voz={config.gain_voice} sfx={config.gain_sfx} bgm={config.gain_bgm})")

    # 4) Scriptwriter + reviewer (fail fast on missing credentials).
    provider = config.vlm_provider
    if provider == "auto" and config.prefer_local and config.hardware.device == "cuda":
        provider = "qwen-local"
    vlm_engine = create_vlm_engine(
        provider, model=config.vlm_model, sfx_tags=sfx_tags,
        sfx_intensity=config.sfx_intensity, style_addendum=style.prompt_addendum,
    )
    preflight = getattr(vlm_engine, "preflight", None)
    if callable(preflight):
        preflight()
    report.append(f"[VLM] roteirista: {getattr(vlm_engine, 'provider', type(vlm_engine).__name__)} "
                  f"(modelo: {getattr(vlm_engine, 'model', 'n/a')})")

    reviewer = None
    if config.review:
        reviewer = create_reviewer(
            provider, model=config.vlm_model,
            known_characters=tuple(DEFAULT_CAST_VOICES), sfx_tags=sfx_tags,
        )
    report.append(f"[REV] revisor: {getattr(reviewer, 'model', 'desligado') if reviewer else 'desligado'}")

    # 5) Voice — the UI keeps one XTTS alive and re-configures it, except
    #    beside a local VLM on a small GPU (8 GB rule): then it is evicted
    #    and rebuilt per run, loading only after the VLM is released.
    tts_note = ""
    if config.tts_backend == "xtts" and xtts_can_stay_resident(config.hardware, provider):
        tts_engine = resources.get("tts:xtts", lambda: create_tts_engine("xtts"))
        tts_engine.configure(speed=config.effective_speed, extra_synthesis_kwargs=style.synthesis_kwargs)
    else:
        if config.tts_backend == "xtts" and resources.evict("tts:xtts"):
            tts_note = " (XTTS liberado: não coexiste com o VLM local em 8 GB)"
        tts_engine = create_tts_engine(
            config.tts_backend, speed=config.effective_speed,
            extra_synthesis_kwargs=style.synthesis_kwargs,
        )
    report.append(f"[TTS] voz: {config.tts_backend}{tts_note}")

    # 6) Orchestrator.
    orchestrator = MangaAudioOrchestrator(
        page_extractor=PyMuPDFPageExtractor(
            dpi=config.dpi, first_page=config.first_page, max_pages=config.max_pages
        ),
        layout_parser=resources.get(
            f"layout:{config.reading_order}",
            lambda: ClassicalLayoutParser(reading_order=config.reading_order),
        ),
        ocr_engine=resources.get("ocr", EasyOCREngine),
        vlm_engine=vlm_engine,
        tts_engine=tts_engine,
        stitcher=MixingStitcher(
            bgm_path=bgm_path, voice_gain=config.gain_voice,
            sfx_gain=config.gain_sfx, bgm_gain=config.gain_bgm,
        ),
        workspace_root=config.workspace_root,
        panel_gap_ms=config.effective_gap_ms,
        resume=config.resume,
        sfx_library=sfx_library,
        sfx_intensity=config.sfx_intensity,
        announce_speakers=config.announce_speakers,
        reviewer=reviewer,
    )
    return BuiltPipeline(config=config, orchestrator=orchestrator, sfx_tags=sfx_tags,
                         bgm_path=bgm_path, report=report)
