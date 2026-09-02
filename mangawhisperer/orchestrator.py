"""Pipeline coordinator: PDF volume in, immersive audio file out.

The :class:`MangaAudioOrchestrator` owns the per-volume workspace on
disk and drives the abstract engines in sequence:

    PDF -> pages -> panels -> bubbles -> OCR -> VLM script -> TTS -> stitch

Every stage's output is persisted under ``workspace/<slug>/`` both as
raw artifacts (page images, panel crops, segment WAVs) and as JSON
checkpoints (``script/panels.json``, ``audio/segments.json``) so long
runs are inspectable and human-correctable between stages.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import TypeAdapter

from mangawhisperer.engines.narration import auto_tag_sfx, plan_narration
from mangawhisperer.engines.text_cleaning import clean_ocr_text, is_ocr_junk
from mangawhisperer.interfaces import (
    AudioStitcher,
    Image,
    MangaLayoutParser,
    MultiSpeakerTTSEngine,
    PDFPageExtractor,
    PortugueseOCREngine,
    VisionLanguageEngine,
)
from mangawhisperer.models import (
    AudioSegmentMetadata,
    ContextualizedBlock,
    PanelData,
    SpeechBubble,
)

logger = logging.getLogger(__name__)

ImageLoader = Callable[[Path], Image]
"""Reads an image file into an (H, W, C) uint8 array."""

_PANELS_ADAPTER: TypeAdapter[list[PanelData]] = TypeAdapter(list[PanelData])
_SEGMENTS_ADAPTER: TypeAdapter[list[AudioSegmentMetadata]] = TypeAdapter(list[AudioSegmentMetadata])


def _default_image_loader(path: Path) -> Image:
    """Load an image via Pillow. Imported lazily so the mocked pipeline
    (which injects its own loader) has no imaging dependency."""
    from PIL import Image as PILImage  # noqa: PLC0415 — optional dependency

    with PILImage.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _engine_fingerprint(engine: object) -> str:
    """Identity string for checkpoint compatibility checks.

    Engines may expose a ``fingerprint`` attribute (e.g. including the
    provider/model); the class name is the fallback. A checkpoint made
    by one fingerprint is never resumed by a run with another — this is
    what stops a placeholder-engine artifact (silent TTS, passthrough
    VLM) from silently masquerading as real output.
    """
    fingerprint = getattr(engine, "fingerprint", None)
    return fingerprint if isinstance(fingerprint, str) else type(engine).__name__


def slugify(name: str) -> str:
    """Normalize a volume name into a filesystem-safe workspace slug.

    ``"BERSERK VOL.01"`` -> ``"berserk_vol_01"``.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug:
        raise ValueError(f"Cannot derive a workspace slug from {name!r}")
    return slug


class MangaAudioOrchestrator:
    """Drives one manga volume end-to-end through the pipeline.

    All heavy lifting is delegated to the injected engine
    implementations; the orchestrator only owns sequencing, cropping,
    workspace layout and checkpointing.
    """

    STAGE_DIRS: tuple[str, ...] = ("pages", "panels", "script", "audio", "final")

    def __init__(
        self,
        page_extractor: PDFPageExtractor,
        layout_parser: MangaLayoutParser,
        ocr_engine: PortugueseOCREngine,
        vlm_engine: VisionLanguageEngine,
        tts_engine: MultiSpeakerTTSEngine,
        stitcher: AudioStitcher,
        workspace_root: Path,
        panel_gap_ms: int = 350,
        resume: bool = False,
        sfx_library: object | None = None,
        sfx_intensity: int = 2,
        announce_speakers: bool = False,
        reviewer: object | None = None,
        image_loader: ImageLoader = _default_image_loader,
    ) -> None:
        """
        Args:
            page_extractor: Rasterizes the PDF into page images.
            layout_parser: Detects panels and bubbles in reading order.
            ocr_engine: Extracts PT-BR text from bubble crops.
            vlm_engine: Attributes speakers and writes action descriptions.
            tts_engine: Renders script blocks to audio segments.
            stitcher: Joins segments into the final track.
            workspace_root: Root directory for all per-volume workspaces.
            panel_gap_ms: Silence inserted between narration segments.
            resume: If True, reuse existing stage checkpoints in the
                workspace instead of recomputing — a crashed run picks
                up after the last completed stage (protecting paid VLM
                calls). Leave False when inputs or engines changed.
            sfx_library: Optional :class:`~mangawhisperer.engines.sfx.
                SFXLibrary`; when given, blocks carrying an ``sfx`` tag
                get the effect inserted before their narration.
            announce_speakers: When True, the Narrator voice announces
                the character's name whenever the speaker changes
                (audiobook style: "Guts" — then Guts' line).
            image_loader: Reads page images from disk; injectable so
                tests can run without an imaging library.
        """
        self._page_extractor = page_extractor
        self._layout_parser = layout_parser
        self._ocr_engine = ocr_engine
        self._vlm_engine = vlm_engine
        self._tts_engine = tts_engine
        self._stitcher = stitcher
        self._workspace_root = workspace_root
        self._panel_gap_ms = panel_gap_ms
        self._resume = resume
        self._sfx_library = None if sfx_intensity <= 0 else sfx_library
        self._sfx_intensity = max(0, min(sfx_intensity, 3))
        self._announce_speakers = announce_speakers
        self._reviewer = reviewer
        self._image_loader = image_loader

    def run(self, pdf_path: Path) -> Path:
        """Process one volume PDF into a single immersive audio file.

        Args:
            pdf_path: Path to the manga volume PDF.

        Returns:
            Path of the final stitched audio file under
            ``workspace/<slug>/final/``.

        Raises:
            FileNotFoundError: If ``pdf_path`` does not exist.
        """
        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        slug = slugify(pdf_path.stem)
        workspace = self._prepare_workspace(slug)
        logger.info("Processing %s in workspace %s", pdf_path.name, workspace)

        panels = self._resume_script(workspace)
        script_resumed = panels is not None
        try:
            if panels is None:
                page_paths = self._page_extractor.extract_pages(pdf_path, workspace / "pages")
                panels = self._build_script(page_paths, workspace)
        finally:
            # Free VLM VRAM before TTS loads — matters on single small GPUs —
            # and never leave a spawned model server behind when the script
            # stage fails (a long-lived UI process would find it on the next run).
            release = getattr(self._vlm_engine, "release", None)
            if callable(release):
                release()

        # A freshly rebuilt script always invalidates the audio: segments
        # from a previous script must never be stitched under a new one.
        segments = self._resume_segments(workspace) if script_resumed else None
        if segments is None:
            segments = self._synthesize_audio(panels, workspace)
        final_path = self._stitcher.stitch(
            segments, workspace / "final" / f"{slug}.wav", gap_ms=self._panel_gap_ms
        )
        logger.info("Finished %s: %d panels, %d segments -> %s",
                    pdf_path.name, len(panels), len(segments), final_path)
        return final_path

    def _resume_script(self, workspace: Path) -> list[PanelData] | None:
        """Load the script checkpoint if resuming, it exists, AND it was
        produced by the same engine configuration as this run."""
        checkpoint = workspace / "script" / "panels.json"
        if not (self._resume and checkpoint.is_file()):
            return None
        stored = self._load_run_config(workspace).get("script")
        if stored != self._script_fingerprints():
            logger.warning(
                "Script checkpoint ignored: engine config changed (was %s, now %s) — recomputing",
                stored, self._script_fingerprints(),
            )
            return None
        panels = _PANELS_ADAPTER.validate_json(checkpoint.read_bytes())
        logger.info("Resumed %d panels from %s (layout/OCR/VLM skipped)", len(panels), checkpoint)
        return panels

    def _resume_segments(self, workspace: Path) -> list[AudioSegmentMetadata] | None:
        """Load the audio checkpoint if resuming, the same engines
        produced it (script AND TTS), and every segment file exists."""
        checkpoint = workspace / "audio" / "segments.json"
        if not (self._resume and checkpoint.is_file()):
            return None
        config = self._load_run_config(workspace)
        if config.get("script") != self._script_fingerprints() or config.get("audio") != self._audio_fingerprints():
            logger.warning("Audio checkpoint ignored: engine config changed — re-running TTS")
            return None
        segments = _SEGMENTS_ADAPTER.validate_json(checkpoint.read_bytes())
        if not all(s.file_path.is_file() for s in segments):
            logger.warning("Audio checkpoint has missing segment files; re-running TTS")
            return None
        logger.info("Resumed %d audio segments from %s (TTS skipped)", len(segments), checkpoint)
        return segments

    def _script_fingerprints(self) -> dict[str, str]:
        return {
            "page_extractor": _engine_fingerprint(self._page_extractor),
            "layout_parser": _engine_fingerprint(self._layout_parser),
            "ocr_engine": _engine_fingerprint(self._ocr_engine),
            "vlm_engine": _engine_fingerprint(self._vlm_engine),
            "reviewer": _engine_fingerprint(self._reviewer) if self._reviewer else "none",
        }

    def _audio_fingerprints(self) -> dict[str, str]:
        tags = ",".join(self._sfx_library.tags()) if self._sfx_library else ""
        return {
            "tts_engine": _engine_fingerprint(self._tts_engine),
            "sfx_tags": tags or "none",
            "announce_speakers": str(self._announce_speakers),
        }

    def _auto_tag_sfx(self, blocks: list[ContextualizedBlock]) -> list[ContextualizedBlock]:
        """Keyword safety net for effects the scriptwriter skipped (rules
        live in :func:`mangawhisperer.engines.narration.auto_tag_sfx`)."""
        tagged = auto_tag_sfx(blocks, set(self._sfx_library.tags()), self._sfx_intensity)
        for before, after in zip(blocks, tagged):
            if before.sfx is None and after.sfx:
                logger.info("Auto-tagged SFX %r for: %.60s", after.sfx, after.text)
        return tagged

    def _sfx_segment(
        self, tag: str | None, block_index: int, seed: str = ""
    ) -> AudioSegmentMetadata | None:
        """Resolve a block's sound-effect tag into a stitchable segment.

        ``seed`` (the block text) picks the variant deterministically
        when the tag has several files — stable across resumes.
        """
        if not tag or self._sfx_library is None:
            return None
        path = self._sfx_library.path_for(tag, seed)
        if path is None:
            logger.warning("Unknown SFX tag %r requested by the scriptwriter; skipping", tag)
            return None
        return AudioSegmentMetadata(
            file_path=path,
            speaker_id="SFX",
            duration_ms=self._sfx_library.duration_ms(tag, seed),
            block_index=block_index,
        )

    @staticmethod
    def _load_run_config(workspace: Path) -> dict:
        path = workspace / "run_config.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _record_run_config(self, workspace: Path, section: str, fingerprints: dict[str, str]) -> None:
        config = self._load_run_config(workspace)
        config[section] = fingerprints
        if section == "script":
            # New script ⇒ any recorded audio belongs to the old script.
            config.pop("audio", None)
        (workspace / "run_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    def _prepare_workspace(self, slug: str) -> Path:
        """Create ``workspace/<slug>/`` and all stage subdirectories."""
        workspace = self._workspace_root / slug
        for stage in self.STAGE_DIRS:
            (workspace / stage).mkdir(parents=True, exist_ok=True)
        return workspace

    def _build_script(self, page_paths: list[Path], workspace: Path) -> list[PanelData]:
        """Layout + OCR + VLM: turn page images into scripted panels.

        Persists panel crops under ``panels/`` and the full script
        checkpoint at ``script/panels.json``.
        """
        panels: list[PanelData] = []
        for page_index, page_path in enumerate(page_paths):
            page_number = page_index + 1
            page_image = self._image_loader(page_path)
            page_h, page_w = page_image.shape[:2]

            for panel_index, panel_box in enumerate(self._layout_parser.extract_panels(page_image)):
                x0, y0, x1, y1 = panel_box.to_absolute(page_w, page_h)
                panel_image = page_image[y0:y1, x0:x1]
                panel_path = workspace / "panels" / f"page{page_number:03d}_panel{panel_index:02d}.npy"
                np.save(panel_path, panel_image)

                bubbles = self._read_bubbles(panel_image)
                blocks = self._vlm_engine.contextualize(panel_image, bubbles)
                if self._sfx_library is not None:
                    blocks = self._auto_tag_sfx(blocks)
                panel = PanelData(
                    image_path=panel_path,
                    bbox=panel_box,
                    page_number=page_number,
                    panel_index=panel_index,
                    blocks=blocks,
                )
                panels.append(panel)

        if self._reviewer is not None:
            self._write_checkpoint(workspace / "script" / "panels_raw.json",
                                   _PANELS_ADAPTER.dump_json(panels, indent=2))
            try:
                panels = self._reviewer.review(panels)
            except Exception as exc:
                logger.warning("Reviewer failed (%s); using the unreviewed script", exc)

        self._write_checkpoint(workspace / "script" / "panels.json",
                               _PANELS_ADAPTER.dump_json(panels, indent=2))
        self._record_run_config(workspace, "script", self._script_fingerprints())
        return panels

    def _read_bubbles(self, panel_image: Image) -> list[SpeechBubble]:
        """Detect and OCR every bubble in one panel, in reading order.

        OCR junk (stray glyphs, page numbers, symbol soup) is dropped
        here, before any token is spent on it downstream.
        """
        panel_h, panel_w = panel_image.shape[:2]
        bubbles: list[SpeechBubble] = []
        for bubble_box in self._layout_parser.extract_bubbles(panel_image):
            x0, y0, x1, y1 = bubble_box.to_absolute(panel_w, panel_h)
            text = clean_ocr_text(self._ocr_engine.recognize(panel_image[y0:y1, x0:x1]))
            if is_ocr_junk(text):
                logger.info("Dropped OCR junk: %r", text)  # visible: a lost line must be diagnosable
                continue
            bubbles.append(SpeechBubble(text=text, bbox=bubble_box))
        return bubbles

    def _synthesize_audio(self, panels: list[PanelData], workspace: Path) -> list[AudioSegmentMetadata]:
        """Render the narration plan: TTS for speech and speaker
        announcements, library lookup for sound effects. What goes where
        is decided by :func:`mangawhisperer.engines.narration.plan_narration`.

        Persists segment audio under ``audio/`` and the metadata
        checkpoint at ``audio/segments.json``.
        """
        plan = plan_narration(
            panels,
            announce_speakers=self._announce_speakers,
            sfx_enabled=self._sfx_library is not None,
        )
        segments: list[AudioSegmentMetadata] = []
        for item in plan:
            # The global narration index is the orchestrator's to assign
            # (engines can't see interleaved SFX/announcement segments).
            block_index = len(segments)
            if item.kind == "sfx":
                segment = self._sfx_segment(item.sfx_tag, block_index, seed=item.seed)
                if segment is None:
                    continue
            else:
                segment_path = workspace / "audio" / f"seg{block_index:05d}.wav"
                metadata = self._tts_engine.synthesize(item.block, segment_path)
                segment = metadata.model_copy(update={"block_index": block_index})
            segments.append(segment)

        self._write_checkpoint(workspace / "audio" / "segments.json",
                               _SEGMENTS_ADAPTER.dump_json(segments, indent=2))
        self._record_run_config(workspace, "audio", self._audio_fingerprints())
        return segments

    @staticmethod
    def _write_checkpoint(path: Path, payload: bytes) -> None:
        """Persist a JSON stage checkpoint."""
        path.write_bytes(payload)
        logger.debug("Checkpoint written: %s", path)
