"""TDD harness for the MangaWhisperer pipeline skeleton.

Every deep-learning stage is replaced by a lightweight, deterministic
mock so the end-to-end data flow — PDF -> pages -> panels -> bubbles ->
OCR -> VLM script -> TTS -> stitched track — is verified without
loading a single model weight.

Mock geometry: 2 pages x 2 panels/page x 2 bubbles/panel; the mock VLM
adds 1 action description per panel, so the run produces
4 panels x 3 blocks = 12 audio segments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

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
    BoundingBox,
    ContextualizedBlock,
    PanelData,
    SpeechBubble,
)
from mangawhisperer.orchestrator import MangaAudioOrchestrator, slugify

# ---------------------------------------------------------------------------
# Mock engines
# ---------------------------------------------------------------------------

PAGE_COUNT = 2
PANELS_PER_PAGE = 2
BUBBLES_PER_PANEL = 2

SCRIPTED_LINES = [
    "Eu vou sobreviver!",
    "Griffith...!",
    "A Marca está sangrando.",
    "Corra, Casca!",
]
ACTION_TEXT = "[Ação]: Guts ergue sua espada gigante contra a escuridão."


class MockPDFPageExtractor(PDFPageExtractor):
    """Writes stub page files instead of rasterizing a real PDF."""

    def __init__(self, page_count: int = PAGE_COUNT) -> None:
        self.page_count = page_count
        self.calls: list[Path] = []

    def extract_pages(self, pdf_path: Path, output_dir: Path) -> list[Path]:
        self.calls.append(pdf_path)
        paths = []
        for i in range(1, self.page_count + 1):
            page = output_dir / f"page_{i:03d}.png"
            page.write_bytes(b"stub-page-image")
            paths.append(page)
        return paths


class MockLayoutParser(MangaLayoutParser):
    """Returns a fixed right-to-left layout: right box first, then left."""

    RTL_PANELS = [
        BoundingBox(x_min=0.55, y_min=0.05, x_max=0.95, y_max=0.95),  # right = read first
        BoundingBox(x_min=0.05, y_min=0.05, x_max=0.45, y_max=0.95),  # left = read second
    ]
    RTL_BUBBLES = [
        BoundingBox(x_min=0.60, y_min=0.10, x_max=0.90, y_max=0.45),  # right = read first
        BoundingBox(x_min=0.10, y_min=0.10, x_max=0.40, y_max=0.45),  # left = read second
    ]

    def __init__(self) -> None:
        self.panel_calls = 0
        self.bubble_calls = 0

    def extract_panels(self, page_image: Image) -> list[BoundingBox]:
        self.panel_calls += 1
        return list(self.RTL_PANELS)

    def extract_bubbles(self, panel_image: Image) -> list[BoundingBox]:
        self.bubble_calls += 1
        return list(self.RTL_BUBBLES)


class MockOCREngine(PortugueseOCREngine):
    """Emits scripted PT-BR lines in a deterministic cycle."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def recognize(self, region_image: Image) -> str:
        text = SCRIPTED_LINES[len(self.calls) % len(SCRIPTED_LINES)]
        self.calls.append(region_image.shape)
        return text


class MockVisionLanguageEngine(VisionLanguageEngine):
    """Attributes every bubble to 'Guts' and appends one action block."""

    def __init__(self) -> None:
        self.calls: list[list[SpeechBubble]] = []

    def contextualize(
        self, panel_image: Image, bubbles: list[SpeechBubble]
    ) -> list[ContextualizedBlock]:
        self.calls.append(bubbles)
        blocks = [
            ContextualizedBlock(text=b.text, speaker_id="Guts", is_speech=True)
            for b in bubbles
        ]
        blocks.append(
            ContextualizedBlock(text=ACTION_TEXT, speaker_id="Narrator", is_speech=False)
        )
        return blocks


class MockTTSEngine(MultiSpeakerTTSEngine):
    """Writes stub audio bytes and reports a fixed 1200 ms duration."""

    SEGMENT_BYTES = b"RIFF-mock-wave-data"
    FIXED_DURATION_MS = 1200

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []  # (speaker_id, text) in synthesis order

    def synthesize(self, block: ContextualizedBlock, output_path: Path) -> AudioSegmentMetadata:
        output_path.write_bytes(self.SEGMENT_BYTES)
        metadata = AudioSegmentMetadata(
            file_path=output_path,
            speaker_id=block.speaker_id,
            duration_ms=self.FIXED_DURATION_MS,
            block_index=len(self.calls),
        )
        self.calls.append((block.speaker_id, block.text))
        return metadata


class MockAudioStitcher(AudioStitcher):
    """Concatenates raw segment bytes with a 1-byte gap marker."""

    GAP_MARKER = b"\x00"

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []  # (segment_count, gap_ms)

    def stitch(
        self,
        segments: list[AudioSegmentMetadata],
        output_path: Path,
        gap_ms: int = 350,
    ) -> Path:
        self.calls.append((len(segments), gap_ms))
        payload = self.GAP_MARKER.join(s.file_path.read_bytes() for s in segments)
        output_path.write_bytes(payload)
        return output_path


def fake_image_loader(path: Path) -> Image:
    """Stands in for PIL: every 'image file' is a 100x80 black RGB frame."""
    return np.zeros((100, 80, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class EngineSet:
    """Bundles the mock engines so tests can assert against each one."""

    def __init__(self) -> None:
        self.page_extractor = MockPDFPageExtractor()
        self.layout_parser = MockLayoutParser()
        self.ocr_engine = MockOCREngine()
        self.vlm_engine = MockVisionLanguageEngine()
        self.tts_engine = MockTTSEngine()
        self.stitcher = MockAudioStitcher()


@pytest.fixture()
def engines() -> EngineSet:
    return EngineSet()


@pytest.fixture()
def workspace_root(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


@pytest.fixture()
def orchestrator(engines: EngineSet, workspace_root: Path) -> MangaAudioOrchestrator:
    return MangaAudioOrchestrator(
        page_extractor=engines.page_extractor,
        layout_parser=engines.layout_parser,
        ocr_engine=engines.ocr_engine,
        vlm_engine=engines.vlm_engine,
        tts_engine=engines.tts_engine,
        stitcher=engines.stitcher,
        workspace_root=workspace_root,
        panel_gap_ms=500,
        image_loader=fake_image_loader,
    )


@pytest.fixture()
def pdf_path(tmp_path: Path) -> Path:
    pdf = tmp_path / "BERSERK VOL.01.pdf"
    pdf.write_bytes(b"%PDF-1.4 mock berserk volume")
    return pdf


# ---------------------------------------------------------------------------
# Data model unit tests
# ---------------------------------------------------------------------------


class TestBoundingBox:
    def test_rejects_inverted_extents(self) -> None:
        with pytest.raises(ValidationError, match="x_max"):
            BoundingBox(x_min=0.8, y_min=0.1, x_max=0.2, y_max=0.9)
        with pytest.raises(ValidationError, match="y_max"):
            BoundingBox(x_min=0.1, y_min=0.9, x_max=0.8, y_max=0.2)

    def test_rejects_out_of_range_coordinates(self) -> None:
        with pytest.raises(ValidationError):
            BoundingBox(x_min=-0.1, y_min=0.1, x_max=0.5, y_max=0.9)
        with pytest.raises(ValidationError):
            BoundingBox(x_min=0.1, y_min=0.1, x_max=1.5, y_max=0.9)

    def test_geometry_helpers(self) -> None:
        box = BoundingBox(x_min=0.2, y_min=0.1, x_max=0.6, y_max=0.5)
        assert box.width == pytest.approx(0.4)
        assert box.height == pytest.approx(0.4)
        assert box.center == (pytest.approx(0.4), pytest.approx(0.3))

    def test_to_absolute_pixel_conversion(self) -> None:
        box = BoundingBox(x_min=0.25, y_min=0.5, x_max=0.75, y_max=1.0)
        assert box.to_absolute(image_width=200, image_height=100) == (50, 50, 150, 100)

    def test_is_immutable(self) -> None:
        box = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)
        with pytest.raises(ValidationError):
            box.x_min = 0.5  # type: ignore[misc]


class TestModels:
    def test_speech_bubble_defaults_to_ptbr(self) -> None:
        bubble = SpeechBubble(
            text="Olá", bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
        )
        assert bubble.language == "pt-BR"

    def test_contextualized_block_requires_nonempty_text(self) -> None:
        with pytest.raises(ValidationError):
            ContextualizedBlock(text="", speaker_id="Guts", is_speech=True)

    def test_audio_segment_rejects_negative_duration(self) -> None:
        with pytest.raises(ValidationError):
            AudioSegmentMetadata(
                file_path=Path("seg.wav"), speaker_id="Guts", duration_ms=-1, block_index=0
            )

    def test_panel_data_json_round_trip(self) -> None:
        panel = PanelData(
            image_path=Path("panels/page001_panel00.npy"),
            bbox=BoundingBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9),
            page_number=1,
            panel_index=0,
            blocks=[ContextualizedBlock(text="Oi", speaker_id="Guts", is_speech=True)],
        )
        restored = PanelData.model_validate_json(panel.model_dump_json())
        assert restored == panel


def test_slugify_normalizes_volume_names() -> None:
    assert slugify("BERSERK VOL.01") == "berserk_vol_01"
    assert slugify("  Berserk---Vol 2 ") == "berserk_vol_2"
    with pytest.raises(ValueError):
        slugify("!!!")


# ---------------------------------------------------------------------------
# Orchestrator integration tests
# ---------------------------------------------------------------------------

TOTAL_PANELS = PAGE_COUNT * PANELS_PER_PAGE
TOTAL_BUBBLES = TOTAL_PANELS * BUBBLES_PER_PANEL
TOTAL_BLOCKS = TOTAL_PANELS * (BUBBLES_PER_PANEL + 1)  # +1 action block per panel


def _make_resuming_orchestrator(engines: EngineSet, workspace_root: Path) -> MangaAudioOrchestrator:
    return MangaAudioOrchestrator(
        page_extractor=engines.page_extractor,
        layout_parser=engines.layout_parser,
        ocr_engine=engines.ocr_engine,
        vlm_engine=engines.vlm_engine,
        tts_engine=engines.tts_engine,
        stitcher=engines.stitcher,
        workspace_root=workspace_root,
        panel_gap_ms=500,
        resume=True,
        image_loader=fake_image_loader,
    )


class TestPipelineIntegration:
    def test_missing_pdf_raises(self, orchestrator: MangaAudioOrchestrator, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            orchestrator.run(tmp_path / "does_not_exist.pdf")

    def test_run_produces_final_audio_file(
        self,
        orchestrator: MangaAudioOrchestrator,
        pdf_path: Path,
        workspace_root: Path,
    ) -> None:
        final_path = orchestrator.run(pdf_path)

        assert final_path == workspace_root / "berserk_vol_01" / "final" / "berserk_vol_01.wav"
        assert final_path.is_file()
        assert final_path.stat().st_size > 0

    def test_workspace_layout_and_artifacts(
        self,
        orchestrator: MangaAudioOrchestrator,
        pdf_path: Path,
        workspace_root: Path,
    ) -> None:
        orchestrator.run(pdf_path)
        volume_dir = workspace_root / "berserk_vol_01"

        for stage in MangaAudioOrchestrator.STAGE_DIRS:
            assert (volume_dir / stage).is_dir(), f"missing stage dir: {stage}"
        assert len(list((volume_dir / "pages").glob("*.png"))) == PAGE_COUNT
        assert len(list((volume_dir / "panels").glob("*.npy"))) == TOTAL_PANELS
        assert len(list((volume_dir / "audio").glob("*.wav"))) == TOTAL_BLOCKS

    def test_every_stage_called_with_expected_cardinality(
        self,
        orchestrator: MangaAudioOrchestrator,
        engines: EngineSet,
        pdf_path: Path,
    ) -> None:
        orchestrator.run(pdf_path)

        assert engines.page_extractor.calls == [pdf_path]
        assert engines.layout_parser.panel_calls == PAGE_COUNT
        assert engines.layout_parser.bubble_calls == TOTAL_PANELS
        assert len(engines.ocr_engine.calls) == TOTAL_BUBBLES
        assert len(engines.vlm_engine.calls) == TOTAL_PANELS
        assert len(engines.tts_engine.calls) == TOTAL_BLOCKS
        assert engines.stitcher.calls == [(TOTAL_BLOCKS, 500)]

    def test_script_checkpoint_is_parseable_and_attributed(
        self,
        orchestrator: MangaAudioOrchestrator,
        pdf_path: Path,
        workspace_root: Path,
    ) -> None:
        orchestrator.run(pdf_path)
        checkpoint = workspace_root / "berserk_vol_01" / "script" / "panels.json"

        panels = TypeAdapter(list[PanelData]).validate_json(checkpoint.read_bytes())
        assert len(panels) == TOTAL_PANELS
        assert [p.page_number for p in panels] == [1, 1, 2, 2]
        assert [p.panel_index for p in panels] == [0, 1, 0, 1]
        for panel in panels:
            speech = [b for b in panel.blocks if b.is_speech]
            actions = [b for b in panel.blocks if not b.is_speech]
            assert len(speech) == BUBBLES_PER_PANEL
            assert all(b.speaker_id == "Guts" for b in speech)
            assert [a.speaker_id for a in actions] == ["Narrator"]
            assert actions[0].text == ACTION_TEXT

    def test_audio_checkpoint_tracks_all_segments(
        self,
        orchestrator: MangaAudioOrchestrator,
        pdf_path: Path,
        workspace_root: Path,
    ) -> None:
        orchestrator.run(pdf_path)
        checkpoint = workspace_root / "berserk_vol_01" / "audio" / "segments.json"

        segments = TypeAdapter(list[AudioSegmentMetadata]).validate_json(checkpoint.read_bytes())
        assert len(segments) == TOTAL_BLOCKS
        assert [s.block_index for s in segments] == list(range(TOTAL_BLOCKS))
        assert all(s.duration_ms == MockTTSEngine.FIXED_DURATION_MS for s in segments)
        assert all(s.file_path.is_file() for s in segments)

    def test_blocks_reach_tts_in_reading_order(
        self,
        orchestrator: MangaAudioOrchestrator,
        engines: EngineSet,
        pdf_path: Path,
    ) -> None:
        """Bubbles must be narrated in the parser's RTL order, panel by
        panel, with the action description closing each panel."""
        orchestrator.run(pdf_path)

        expected: list[tuple[str, str]] = []
        line_cursor = 0
        for _ in range(TOTAL_PANELS):
            for _ in range(BUBBLES_PER_PANEL):
                expected.append(("Guts", SCRIPTED_LINES[line_cursor % len(SCRIPTED_LINES)]))
                line_cursor += 1
            expected.append(("Narrator", ACTION_TEXT))

        assert engines.tts_engine.calls == expected

    def test_resume_skips_completed_stages(self, workspace_root: Path, pdf_path: Path) -> None:
        """A second resumed run must not repeat extraction, layout, OCR,
        VLM, or TTS — only the cheap final stitch."""
        _make_resuming_orchestrator(EngineSet(), workspace_root).run(pdf_path)

        second = EngineSet()
        final_path = _make_resuming_orchestrator(second, workspace_root).run(pdf_path)

        assert second.page_extractor.calls == []
        assert second.layout_parser.panel_calls == 0
        assert len(second.ocr_engine.calls) == 0
        assert len(second.vlm_engine.calls) == 0
        assert len(second.tts_engine.calls) == 0
        assert second.stitcher.calls == [(TOTAL_BLOCKS, 500)]
        assert final_path.is_file()

    def test_resume_reruns_tts_when_a_segment_file_is_missing(
        self, workspace_root: Path, pdf_path: Path
    ) -> None:
        """Missing audio invalidates the audio checkpoint but the paid
        script checkpoint (VLM output) must still be reused."""
        _make_resuming_orchestrator(EngineSet(), workspace_root).run(pdf_path)
        victim = next((workspace_root / "berserk_vol_01" / "audio").glob("*.wav"))
        victim.unlink()

        second = EngineSet()
        _make_resuming_orchestrator(second, workspace_root).run(pdf_path)

        assert len(second.vlm_engine.calls) == 0, "script checkpoint must survive"
        assert len(second.tts_engine.calls) == TOTAL_BLOCKS, "TTS must rerun in full"

    def test_resume_invalidated_when_tts_engine_changes(
        self, workspace_root: Path, pdf_path: Path
    ) -> None:
        """The silent-audio bug: segments made by a different TTS engine
        (e.g. the silent placeholder) must NOT be resumed."""
        first = EngineSet()
        first.tts_engine.fingerprint = "silent-placeholder"
        _make_resuming_orchestrator(first, workspace_root).run(pdf_path)

        second = EngineSet()
        second.tts_engine.fingerprint = "xtts:real"
        _make_resuming_orchestrator(second, workspace_root).run(pdf_path)

        assert len(second.vlm_engine.calls) == 0, "script unchanged — still resumed"
        assert len(second.tts_engine.calls) == TOTAL_BLOCKS, "TTS must be redone"

    def test_resume_invalidated_when_vlm_engine_changes(
        self, workspace_root: Path, pdf_path: Path
    ) -> None:
        """A script written by a different VLM (e.g. passthrough) must be
        recomputed — and the dependent audio with it."""
        first = EngineSet()
        first.vlm_engine.fingerprint = "passthrough"
        _make_resuming_orchestrator(first, workspace_root).run(pdf_path)

        second = EngineSet()
        second.vlm_engine.fingerprint = "vlm-api:qwen:qwen3-vl-plus"
        _make_resuming_orchestrator(second, workspace_root).run(pdf_path)

        assert len(second.vlm_engine.calls) == TOTAL_PANELS, "script must be redone"
        assert len(second.tts_engine.calls) == TOTAL_BLOCKS, "audio depends on the script"

    def test_legacy_workspace_without_run_config_is_not_resumed(
        self, workspace_root: Path, pdf_path: Path
    ) -> None:
        """Workspaces created before fingerprinting have no run_config —
        treat them as incompatible instead of trusting them."""
        first = EngineSet()
        _make_resuming_orchestrator(first, workspace_root).run(pdf_path)
        (workspace_root / "berserk_vol_01" / "run_config.json").unlink()

        second = EngineSet()
        _make_resuming_orchestrator(second, workspace_root).run(pdf_path)

        assert len(second.vlm_engine.calls) == TOTAL_PANELS
        assert len(second.tts_engine.calls) == TOTAL_BLOCKS

    def test_sound_effects_are_interleaved_before_their_blocks(
        self, workspace_root: Path, pdf_path: Path, tmp_path: Path
    ) -> None:
        """A block tagged with sfx gets the effect inserted right before
        its narration segment, with sequential global indices."""
        import wave

        import numpy as np

        from mangawhisperer.engines.sfx import SFXLibrary

        # Tag "porta" on purpose: its keywords don't appear in the mock
        # action text, so this test isolates VLM-chosen effects from the
        # keyword auto-tagger (covered by its own test below).
        sfx_dir = tmp_path / "sfx"
        sfx_dir.mkdir()
        tone = (np.sin(np.arange(2400) / 24000 * 2 * np.pi * 440) * 20000).astype(np.int16)
        with wave.open(str(sfx_dir / "porta.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(tone.tobytes())

        class SfxVLM(MockVisionLanguageEngine):
            def contextualize(self, panel_image, bubbles):
                blocks = super().contextualize(panel_image, bubbles)
                tagged = blocks[0].model_copy(update={"sfx": "porta"})
                return [tagged, *blocks[1:]]

        engines = EngineSet()
        engines.vlm_engine = SfxVLM()
        orchestrator = MangaAudioOrchestrator(
            page_extractor=engines.page_extractor,
            layout_parser=engines.layout_parser,
            ocr_engine=engines.ocr_engine,
            vlm_engine=engines.vlm_engine,
            tts_engine=engines.tts_engine,
            stitcher=engines.stitcher,
            workspace_root=workspace_root,
            sfx_library=SFXLibrary(sfx_dir),
            image_loader=fake_image_loader,
        )
        orchestrator.run(pdf_path)

        checkpoint = workspace_root / "berserk_vol_01" / "audio" / "segments.json"
        segments = TypeAdapter(list[AudioSegmentMetadata]).validate_json(checkpoint.read_bytes())

        sfx_segments = [s for s in segments if s.speaker_id == "SFX"]
        assert len(sfx_segments) == TOTAL_PANELS, "one tagged block per panel"
        assert len(segments) == TOTAL_BLOCKS + TOTAL_PANELS
        assert [s.block_index for s in segments] == list(range(len(segments)))
        for sfx_seg in sfx_segments:
            assert sfx_seg.file_path.is_file()
            following = segments[sfx_seg.block_index + 1]
            assert following.speaker_id != "SFX", "effect must precede a narration block"

    def test_keyword_fallback_tags_action_blocks(
        self, workspace_root: Path, pdf_path: Path, tmp_path: Path
    ) -> None:
        """When the VLM chooses no effect, the keyword safety net tags
        obvious action beats — the mock action text mentions 'espada'."""
        import wave

        import numpy as np

        from mangawhisperer.engines.sfx import SFXLibrary

        sfx_dir = tmp_path / "sfx"
        sfx_dir.mkdir()
        tone = (np.sin(np.arange(2400) / 24000 * 2 * np.pi * 440) * 20000).astype(np.int16)
        with wave.open(str(sfx_dir / "espada.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(tone.tobytes())

        engines = EngineSet()  # plain mock VLM: never sets sfx itself
        orchestrator = MangaAudioOrchestrator(
            page_extractor=engines.page_extractor,
            layout_parser=engines.layout_parser,
            ocr_engine=engines.ocr_engine,
            vlm_engine=engines.vlm_engine,
            tts_engine=engines.tts_engine,
            stitcher=engines.stitcher,
            workspace_root=workspace_root,
            sfx_library=SFXLibrary(sfx_dir),
            image_loader=fake_image_loader,
        )
        orchestrator.run(pdf_path)

        panels = TypeAdapter(list[PanelData]).validate_json(
            (workspace_root / "berserk_vol_01" / "script" / "panels.json").read_bytes()
        )
        auto_tagged = [b for p in panels for b in p.blocks if b.sfx == "espada"]
        assert len(auto_tagged) == TOTAL_PANELS, "every panel's action block mentions espada"
        assert all(not b.is_speech for b in auto_tagged), "only action blocks are auto-tagged"

    def test_unknown_sfx_tag_is_skipped_with_no_segment(
        self, workspace_root: Path, pdf_path: Path, tmp_path: Path
    ) -> None:
        from mangawhisperer.engines.sfx import SFXLibrary

        class BadSfxVLM(MockVisionLanguageEngine):
            def contextualize(self, panel_image, bubbles):
                blocks = super().contextualize(panel_image, bubbles)
                return [blocks[0].model_copy(update={"sfx": "laser"}), *blocks[1:]]

        engines = EngineSet()
        engines.vlm_engine = BadSfxVLM()
        empty_dir = tmp_path / "sfx_vazio"
        empty_dir.mkdir()
        orchestrator = MangaAudioOrchestrator(
            page_extractor=engines.page_extractor,
            layout_parser=engines.layout_parser,
            ocr_engine=engines.ocr_engine,
            vlm_engine=engines.vlm_engine,
            tts_engine=engines.tts_engine,
            stitcher=engines.stitcher,
            workspace_root=workspace_root,
            sfx_library=SFXLibrary(empty_dir),
            image_loader=fake_image_loader,
        )
        orchestrator.run(pdf_path)

        checkpoint = workspace_root / "berserk_vol_01" / "audio" / "segments.json"
        segments = TypeAdapter(list[AudioSegmentMetadata]).validate_json(checkpoint.read_bytes())
        assert len(segments) == TOTAL_BLOCKS, "unknown tag adds nothing"
        assert all(s.speaker_id != "SFX" for s in segments)

    def test_reviewer_stage_rewrites_script_and_keeps_raw_checkpoint(
        self, workspace_root: Path, pdf_path: Path
    ) -> None:
        class RenamingReviewer:
            fingerprint = "reviewer:test"

            def review(self, panels):
                return [
                    p.model_copy(update={"blocks": [
                        b.model_copy(update={"speaker_id": "Revisado"}) for b in p.blocks
                    ]})
                    for p in panels
                ]

        engines = EngineSet()
        orchestrator = MangaAudioOrchestrator(
            page_extractor=engines.page_extractor,
            layout_parser=engines.layout_parser,
            ocr_engine=engines.ocr_engine,
            vlm_engine=engines.vlm_engine,
            tts_engine=engines.tts_engine,
            stitcher=engines.stitcher,
            workspace_root=workspace_root,
            reviewer=RenamingReviewer(),
            image_loader=fake_image_loader,
        )
        orchestrator.run(pdf_path)

        script_dir = workspace_root / "berserk_vol_01" / "script"
        final = TypeAdapter(list[PanelData]).validate_json((script_dir / "panels.json").read_bytes())
        raw = TypeAdapter(list[PanelData]).validate_json((script_dir / "panels_raw.json").read_bytes())

        assert all(b.speaker_id == "Revisado" for p in final for b in p.blocks)
        assert any(b.speaker_id == "Guts" for p in raw for b in p.blocks), "raw preserved"

    def test_broken_reviewer_falls_back_to_raw_script(
        self, workspace_root: Path, pdf_path: Path
    ) -> None:
        class ExplodingReviewer:
            fingerprint = "reviewer:boom"

            def review(self, panels):
                raise RuntimeError("reviewer down")

        engines = EngineSet()
        orchestrator = MangaAudioOrchestrator(
            page_extractor=engines.page_extractor,
            layout_parser=engines.layout_parser,
            ocr_engine=engines.ocr_engine,
            vlm_engine=engines.vlm_engine,
            tts_engine=engines.tts_engine,
            stitcher=engines.stitcher,
            workspace_root=workspace_root,
            reviewer=ExplodingReviewer(),
            image_loader=fake_image_loader,
        )
        final_path = orchestrator.run(pdf_path)

        assert final_path.is_file()
        panels = TypeAdapter(list[PanelData]).validate_json(
            (workspace_root / "berserk_vol_01" / "script" / "panels.json").read_bytes()
        )
        assert any(b.speaker_id == "Guts" for p in panels for b in p.blocks)

    def test_speaker_announcements_on_speaker_change_only(
        self, workspace_root: Path, pdf_path: Path
    ) -> None:
        """With announce_speakers on, the Narrator says the character's
        name when the speaker changes — not before every line."""
        engines = EngineSet()
        orchestrator = MangaAudioOrchestrator(
            page_extractor=engines.page_extractor,
            layout_parser=engines.layout_parser,
            ocr_engine=engines.ocr_engine,
            vlm_engine=engines.vlm_engine,
            tts_engine=engines.tts_engine,
            stitcher=engines.stitcher,
            workspace_root=workspace_root,
            announce_speakers=True,
            image_loader=fake_image_loader,
        )
        orchestrator.run(pdf_path)

        # Mock VLM: every panel = 2 speech blocks by "Guts" + 1 Narrator
        # action. Only the very first Guts line gets an announcement.
        announcements = [
            call for call in engines.tts_engine.calls if call == ("Narrator", "Guts")
        ]
        assert len(announcements) == 1, "same speaker throughout -> announce once"
        assert engines.tts_engine.calls[0] == ("Narrator", "Guts"), "announcement comes first"
        assert len(engines.tts_engine.calls) == TOTAL_BLOCKS + 1

    def test_announcements_off_by_default(self, workspace_root: Path, pdf_path: Path) -> None:
        engines = EngineSet()
        _make_resuming_orchestrator(engines, workspace_root).run(pdf_path)
        assert len(engines.tts_engine.calls) == TOTAL_BLOCKS

    def test_ocr_receives_cropped_bubble_regions(
        self,
        orchestrator: MangaAudioOrchestrator,
        engines: EngineSet,
        pdf_path: Path,
    ) -> None:
        """OCR must see bubble crops, not whole pages or whole panels."""
        orchestrator.run(pdf_path)

        page_shape = fake_image_loader(pdf_path).shape
        for region_shape in engines.ocr_engine.calls:
            assert region_shape[0] < page_shape[0]
            assert region_shape[1] < page_shape[1]
            assert region_shape[0] > 0 and region_shape[1] > 0
