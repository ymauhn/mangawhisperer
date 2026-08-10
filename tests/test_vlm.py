"""Tests for the Claude-backed VisionLanguageEngine.

All tests inject a fake Anthropic client, so they run offline with no
API key and without the ``anthropic`` package installed.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mangawhisperer.models import BoundingBox, ContextualizedBlock, SpeechBubble

cv2 = pytest.importorskip("cv2")

from mangawhisperer.engines.vlm import ClaudeVisionLanguageEngine  # noqa: E402

BBOX = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)


class FakeMessages:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.response


class FakeAnthropicClient:
    def __init__(self, response: SimpleNamespace) -> None:
        self.messages = FakeMessages(response)


def make_response(
    blocks: list[ContextualizedBlock] | None, stop_reason: str = "end_turn"
) -> SimpleNamespace:
    parsed = SimpleNamespace(blocks=blocks) if blocks is not None else None
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason)


def make_engine(response: SimpleNamespace, **kwargs) -> tuple[ClaudeVisionLanguageEngine, FakeAnthropicClient]:
    client = FakeAnthropicClient(response)
    return ClaudeVisionLanguageEngine(client=client, **kwargs), client


PANEL = np.full((200, 160, 3), 200, dtype=np.uint8)
BUBBLES = [
    SpeechBubble(text="Eu vou sobreviver!", bbox=BBOX),
    SpeechBubble(text="Griffith...!", bbox=BBOX),
]


class TestClaudeVisionLanguageEngine:
    def test_returns_blocks_from_parsed_output(self) -> None:
        expected = [
            ContextualizedBlock(text="Eu vou sobreviver!", speaker_id="Guts", is_speech=True),
            ContextualizedBlock(text="Guts ergue a espada.", speaker_id="Narrator", is_speech=False),
        ]
        engine, _ = make_engine(make_response(expected))

        assert engine.contextualize(PANEL, BUBBLES) == expected

    def test_sends_base64_png_image_block(self) -> None:
        engine, client = make_engine(make_response([]))
        engine.contextualize(PANEL, BUBBLES)

        (call,) = client.messages.calls
        image_block = call["messages"][0]["content"][0]
        assert image_block["type"] == "image"
        assert image_block["source"]["type"] == "base64"
        assert image_block["source"]["media_type"] == "image/png"
        png_bytes = base64.standard_b64decode(image_block["source"]["data"])
        assert png_bytes.startswith(b"\x89PNG")

    def test_prompt_carries_bubble_texts_in_reading_order(self) -> None:
        engine, client = make_engine(make_response([]))
        engine.contextualize(PANEL, BUBBLES)

        (call,) = client.messages.calls
        text_block = call["messages"][0]["content"][1]
        assert text_block["type"] == "text"
        payload_json = text_block["text"].split("\n", 1)[1]
        assert json.loads(payload_json) == ["Eu vou sobreviver!", "Griffith...!"]

    def test_system_prompt_names_the_cast(self) -> None:
        engine, client = make_engine(
            make_response([]), known_characters=("Guts", "Griffith", "Zodd")
        )
        engine.contextualize(PANEL, BUBBLES)

        (call,) = client.messages.calls
        assert call["model"] == "claude-opus-4-8"
        for name in ("Guts", "Griffith", "Zodd", "Desconhecido"):
            assert name in call["system"]

    def test_refusal_falls_back_to_passthrough_blocks(self) -> None:
        engine, _ = make_engine(make_response(None, stop_reason="refusal"))

        blocks = engine.contextualize(PANEL, BUBBLES)
        assert [(b.text, b.speaker_id, b.is_speech) for b in blocks] == [
            ("Eu vou sobreviver!", "Desconhecido", True),
            ("Griffith...!", "Desconhecido", True),
        ]

    def test_downscales_oversized_panels_before_upload(self) -> None:
        engine, client = make_engine(make_response([]), max_image_edge=512)
        big_panel = np.full((400, 2048, 3), 128, dtype=np.uint8)
        engine.contextualize(big_panel, [])

        (call,) = client.messages.calls
        png_bytes = base64.standard_b64decode(
            call["messages"][0]["content"][0]["source"]["data"]
        )
        decoded = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert max(decoded.shape[:2]) == 512
        assert decoded.shape[:2] == (100, 512)  # aspect ratio preserved

    def test_textless_panel_still_requests_scene_description(self) -> None:
        """Empty panels must still hit the VLM — visual-only beats are
        exactly what the accessibility narration exists for."""
        action = ContextualizedBlock(
            text="O vento varre o campo de batalha silencioso.",
            speaker_id="Narrator",
            is_speech=False,
        )
        engine, client = make_engine(make_response([action]))

        blocks = engine.contextualize(PANEL, [])
        assert blocks == [action]
        assert len(client.messages.calls) == 1
