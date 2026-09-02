"""Narration planning is pure: announcements, SFX placement and the
keyword tagger are verified without touching TTS or audio files."""

from pathlib import Path

from mangawhisperer.engines.narration import (
    NARRATOR,
    announcement_for,
    auto_tag_sfx,
    plan_narration,
)
from mangawhisperer.models import BoundingBox, ContextualizedBlock, PanelData


def _panel(index: int, blocks: list[ContextualizedBlock]) -> PanelData:
    return PanelData(
        image_path=Path(f"panel{index}.npy"),
        bbox=BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0),
        page_number=1,
        panel_index=index,
        blocks=blocks,
    )


def _speech(speaker: str, text: str, sfx: str | None = None) -> ContextualizedBlock:
    return ContextualizedBlock(text=text, speaker_id=speaker, is_speech=True, sfx=sfx)


def _desc(text: str, sfx: str | None = None) -> ContextualizedBlock:
    return ContextualizedBlock(text=text, speaker_id=NARRATOR, is_speech=False, sfx=sfx)


def test_plan_announces_only_on_speaker_change_and_keeps_sfx_before_block():
    panels = [
        _panel(0, [_desc("Guts caminha."), _speech("Guts", "Vamos."), _speech("Guts", "Agora.")]),
        _panel(1, [_speech("Casca", "Espera!", sfx="espada"), _desc("Lâminas se cruzam."),
                   _speech("Guts", "Saia daí.")]),
    ]
    plan = plan_narration(panels, announce_speakers=True, sfx_enabled=True)
    kinds = [(item.kind, item.block.speaker_id if item.block else item.sfx_tag) for item in plan]
    assert kinds == [
        ("speech", NARRATOR),
        ("announcement", NARRATOR),  # "Guts"
        ("speech", "Guts"),
        ("speech", "Guts"),  # same speaker: not re-announced
        ("sfx", "espada"),  # effect precedes the block that carries it
        ("announcement", NARRATOR),  # "Casca"
        ("speech", "Casca"),
        ("speech", NARRATOR),  # narrator interlude does not reset memory...
        ("announcement", NARRATOR),  # ...but Guts after Casca is a change
        ("speech", "Guts"),
    ]
    assert plan[1].block.text == "Guts"
    assert plan[4].seed == "Espera!"


def test_plan_can_disable_announcements_and_sfx():
    panels = [_panel(0, [_speech("Guts", "Oi", sfx="soco"), _speech("Casca", "Olá")])]
    plan = plan_narration(panels, announce_speakers=False, sfx_enabled=False)
    assert [item.kind for item in plan] == ["speech", "speech"]


def test_narrator_speech_is_never_announced():
    panels = [_panel(0, [ContextualizedBlock(text="Era uma vez", speaker_id=NARRATOR, is_speech=True)])]
    plan = plan_narration(panels, announce_speakers=True, sfx_enabled=True)
    assert [item.kind for item in plan] == ["speech"]


def test_announcement_block_is_narrator_non_speech():
    block = announcement_for("Zodd")
    assert (block.text, block.speaker_id, block.is_speech) == ("Zodd", NARRATOR, False)


def test_auto_tag_respects_intensity_cap_and_scriptwriter_choices():
    blocks = [
        _desc("Uma explosão devasta o campo.", sfx="fogo"),  # scriptwriter's own pick counts
        _desc("A espada corta o ar."),
        _desc("Um grito ecoa."),
    ]
    available = {"explosao", "espada", "grito", "fogo"}
    assert [b.sfx for b in auto_tag_sfx(blocks, available, intensity=0)] == ["fogo", None, None]
    assert [b.sfx for b in auto_tag_sfx(blocks, available, intensity=1)] == ["fogo", None, None]
    assert [b.sfx for b in auto_tag_sfx(blocks, available, intensity=2)] == ["fogo", "espada", None]
    assert [b.sfx for b in auto_tag_sfx(blocks, available, intensity=3)] == ["fogo", "espada", "grito"]


def test_auto_tag_only_touches_dialogue_at_max_intensity():
    blocks = [_speech("Guts", "Minha espada vai te cortar!")]
    assert auto_tag_sfx(blocks, {"espada"}, intensity=2)[0].sfx is None
    assert auto_tag_sfx(blocks, {"espada"}, intensity=3)[0].sfx == "espada"


def test_auto_tag_ignores_unavailable_tags_and_leaves_input_untouched():
    blocks = [_desc("A espada corta o ar.")]
    assert auto_tag_sfx(blocks, {"trovao"}, intensity=2)[0].sfx is None
    assert auto_tag_sfx(blocks, set(), intensity=2)[0].sfx is None
    assert blocks[0].sfx is None
