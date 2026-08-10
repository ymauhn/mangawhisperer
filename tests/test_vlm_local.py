"""Tests for the local Qwen-VL engine (fake generator — no weights)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mangawhisperer.models import BoundingBox, SpeechBubble

pytest.importorskip("cv2")  # vlm_local imports the shared prompt from engines.vlm

from mangawhisperer.engines.vlm_local import QwenVisionLanguageEngine  # noqa: E402

BBOX = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
PANEL = np.full((100, 80, 3), 200, dtype=np.uint8)
BUBBLES = [
    SpeechBubble(text="Eu vou sobreviver!", bbox=BBOX),
    SpeechBubble(text="Griffith...!", bbox=BBOX),
]

VALID_SCRIPT_JSON = json.dumps(
    [
        {"text": "Eu vou sobreviver!", "speaker_id": "Guts", "is_speech": True},
        {"text": "Guts ergue a espada.", "speaker_id": "Narrator", "is_speech": False},
    ],
    ensure_ascii=False,
)


class RecordingGenerator:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[np.ndarray, str]] = []

    def __call__(self, image: np.ndarray, prompt: str) -> str:
        self.calls.append((image, prompt))
        return self.output


class TestQwenVisionLanguageEngine:
    def test_parses_json_script_into_blocks(self) -> None:
        engine = QwenVisionLanguageEngine(generator=RecordingGenerator(VALID_SCRIPT_JSON))
        blocks = engine.contextualize(PANEL, BUBBLES)

        assert [(b.text, b.speaker_id, b.is_speech) for b in blocks] == [
            ("Eu vou sobreviver!", "Guts", True),
            ("Guts ergue a espada.", "Narrator", False),
        ]

    def test_tolerates_prose_around_the_json_array(self) -> None:
        chatty = f"Claro! Aqui está o roteiro:\n{VALID_SCRIPT_JSON}\nEspero que ajude."
        engine = QwenVisionLanguageEngine(generator=RecordingGenerator(chatty))

        assert len(engine.contextualize(PANEL, BUBBLES)) == 2

    def test_malformed_output_falls_back_to_passthrough(self) -> None:
        engine = QwenVisionLanguageEngine(generator=RecordingGenerator("desculpe, não sei"))
        blocks = engine.contextualize(PANEL, BUBBLES)

        assert [(b.text, b.speaker_id, b.is_speech) for b in blocks] == [
            ("Eu vou sobreviver!", "Desconhecido", True),
            ("Griffith...!", "Desconhecido", True),
        ]

    def test_truncated_output_salvages_complete_blocks(self) -> None:
        """The Colab regression: output cut mid-array by max_new_tokens
        must keep the complete blocks instead of dropping the panel."""
        truncated = VALID_SCRIPT_JSON[:-25]
        assert "]" not in truncated
        engine = QwenVisionLanguageEngine(generator=RecordingGenerator(truncated))

        blocks = engine.contextualize(PANEL, BUBBLES)
        assert [b.speaker_id for b in blocks] == ["Guts"], "salvage, not passthrough"

    def test_empty_text_blocks_are_dropped_not_fatal(self) -> None:
        """Colab regression #2: '"text": ""' blocks poisoned the whole
        array under strict validation."""
        dirty = json.dumps(
            [
                {"text": "", "speaker_id": "Narrator", "is_speech": False},
                {"text": "Eu vou sobreviver!", "speaker_id": "Guts", "is_speech": True},
            ],
            ensure_ascii=False,
        )
        engine = QwenVisionLanguageEngine(generator=RecordingGenerator(dirty))

        blocks = engine.contextualize(PANEL, BUBBLES)
        assert [(b.text, b.speaker_id) for b in blocks] == [("Eu vou sobreviver!", "Guts")]

    def test_schema_violation_falls_back_to_passthrough(self) -> None:
        bad_schema = json.dumps([{"text": "", "speaker_id": "Guts", "is_speech": True}])
        engine = QwenVisionLanguageEngine(generator=RecordingGenerator(bad_schema))

        blocks = engine.contextualize(PANEL, BUBBLES)
        assert all(b.speaker_id == "Desconhecido" for b in blocks)

    def test_prompt_carries_bubble_texts_and_image(self) -> None:
        generator = RecordingGenerator(VALID_SCRIPT_JSON)
        QwenVisionLanguageEngine(generator=generator).contextualize(PANEL, BUBBLES)

        ((image, prompt),) = generator.calls
        assert image is PANEL
        assert "Eu vou sobreviver!" in prompt
        assert "Griffith...!" in prompt
