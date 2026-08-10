"""Tests for the tolerant VLM output parser (the Colab bug-fix)."""

from __future__ import annotations

import json

from mangawhisperer.engines.script_parsing import parse_script_blocks, passthrough_blocks
from mangawhisperer.models import BoundingBox, SpeechBubble

VALID = [
    {"text": "Eu vou sobreviver!", "speaker_id": "Guts", "is_speech": True},
    {"text": "Guts ergue a espada.", "speaker_id": "Narrator", "is_speech": False},
]


class TestParseScriptBlocks:
    def test_parses_clean_array(self) -> None:
        blocks = parse_script_blocks(json.dumps(VALID, ensure_ascii=False))
        assert [(b.text, b.speaker_id, b.is_speech) for b in blocks] == [
            ("Eu vou sobreviver!", "Guts", True),
            ("Guts ergue a espada.", "Narrator", False),
        ]

    def test_drops_empty_text_blocks_keeps_the_rest(self) -> None:
        """The exact Colab failure: Qwen emitted '"text": ""' blocks and
        the strict parser threw away the whole panel."""
        dirty = [{"text": "", "speaker_id": "Narrator", "is_speech": False}, *VALID]
        blocks = parse_script_blocks(json.dumps(dirty, ensure_ascii=False))
        assert len(blocks) == 2
        assert blocks[0].text == "Eu vou sobreviver!"

    def test_salvages_truncated_array(self) -> None:
        """The other Colab failure: output cut mid-array by max_new_tokens."""
        truncated = json.dumps(VALID, ensure_ascii=False)[:-30]  # cut inside last object
        assert "]" not in truncated
        blocks = parse_script_blocks(truncated)
        assert [b.text for b in blocks] == ["Eu vou sobreviver!"]

    def test_missing_speaker_becomes_desconhecido(self) -> None:
        blocks = parse_script_blocks('[{"text": "Quem está aí?", "is_speech": true}]')
        assert blocks[0].speaker_id == "Desconhecido"

    def test_no_json_at_all_returns_none(self) -> None:
        assert parse_script_blocks("desculpe, não consigo ajudar") is None

    def test_all_blocks_invalid_returns_empty_list(self) -> None:
        blocks = parse_script_blocks('[{"text": ""}, {"speaker_id": "Guts"}, 42]')
        assert blocks == []

    def test_prose_around_array_is_tolerated(self) -> None:
        raw = f"Claro! Aqui está:\n{json.dumps(VALID, ensure_ascii=False)}\nEspero que ajude."
        assert len(parse_script_blocks(raw)) == 2


def test_passthrough_skips_blank_bubbles() -> None:
    box = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    bubbles = [SpeechBubble(text="Berserk!", bbox=box), SpeechBubble(text="  ", bbox=box)]
    blocks = passthrough_blocks(bubbles)
    assert [(b.text, b.speaker_id) for b in blocks] == [("Berserk!", "Desconhecido")]
