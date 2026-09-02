"""Tests for narration style presets and their pipeline wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from mangawhisperer.engines.styles import STYLES, get_style
from mangawhisperer.engines.tts import XTTSEngine
from mangawhisperer.engines.vlm import build_scriptwriter_prompt
from mangawhisperer.models import ContextualizedBlock


class TestStylePresets:
    def test_all_presets_exist(self) -> None:
        assert set(STYLES) == {"neutro", "sobrio", "sombrio", "epico"}

    def test_get_style_is_case_insensitive(self) -> None:
        assert get_style("  Sombrio ").name == "sombrio"

    def test_unknown_style_raises_with_options(self) -> None:
        with pytest.raises(ValueError, match="neutro"):
            get_style("cyberpunk")

    def test_presets_differ_on_every_lever(self) -> None:
        sombrio, epico, neutro = get_style("sombrio"), get_style("epico"), get_style("neutro")
        assert sombrio.tts_speed < neutro.tts_speed < epico.tts_speed
        assert sombrio.gap_ms > neutro.gap_ms > epico.gap_ms
        assert sombrio.prompt_addendum != epico.prompt_addendum
        assert neutro.prompt_addendum == ""

    def test_addendum_lands_at_the_end_of_the_prompt(self) -> None:
        style = get_style("sombrio")
        prompt = build_scriptwriter_prompt(("Guts",), style_addendum=style.prompt_addendum)
        assert prompt.endswith(style.prompt_addendum)
        assert "SOMBRIO" in prompt

    def test_neutral_addendum_changes_nothing(self) -> None:
        base = build_scriptwriter_prompt(("Guts",))
        neutral = build_scriptwriter_prompt(("Guts",), style_addendum=get_style("neutro").prompt_addendum)
        assert base == neutral


class TestTTSConfigure:
    def test_configure_updates_delivery_without_reload(self, tmp_path: Path) -> None:
        from tests.test_tts import FakeSynthesizer

        fake = FakeSynthesizer()
        engine = XTTSEngine(synthesizer=fake)
        style = get_style("sombrio")
        engine.configure(speed=style.tts_speed, extra_synthesis_kwargs=style.synthesis_kwargs)

        block = ContextualizedBlock(text="A noite cai", speaker_id="Narrator", is_speech=False)
        engine.synthesize(block, tmp_path / "seg.wav")

        (call,) = fake.calls
        assert call["speed"] == style.tts_speed
        assert call["temperature"] == 0.7

    def test_configure_changes_fingerprint(self) -> None:
        engine = XTTSEngine(synthesizer=object())
        before = engine.fingerprint
        engine.configure(speed=0.92, extra_synthesis_kwargs={"temperature": 0.7})
        assert engine.fingerprint != before
