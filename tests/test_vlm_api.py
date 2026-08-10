"""Tests for the multi-provider OpenAI-compatible VLM engine + factory.

All tests inject a fake client — no API keys, no network, no `openai`
package needed.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mangawhisperer.models import BoundingBox, SpeechBubble

pytest.importorskip("cv2")

from mangawhisperer.engines.factory import create_vlm_engine  # noqa: E402
from mangawhisperer.engines.placeholders import PassthroughVLM  # noqa: E402
from mangawhisperer.engines.vlm_api import (  # noqa: E402
    PROVIDER_PRESETS,
    OpenAICompatibleVisionLanguageEngine,
)

BBOX = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
PANEL = np.full((200, 160, 3), 200, dtype=np.uint8)
BUBBLES = [
    SpeechBubble(text="Eu vou sobreviver!", bbox=BBOX),
    SpeechBubble(text="Griffith...!", bbox=BBOX),
]
SCRIPT_JSON = json.dumps(
    [{"text": "Eu vou sobreviver!", "speaker_id": "Guts", "is_speech": True}],
    ensure_ascii=False,
)


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeOpenAIClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


def make_engine(content: str = SCRIPT_JSON, **kwargs):
    client = FakeOpenAIClient(content)
    engine = OpenAICompatibleVisionLanguageEngine(client=client, **kwargs)
    return engine, client.chat.completions


class TestOpenAICompatibleEngine:
    def test_defaults_to_qwen_preset(self) -> None:
        engine, completions = make_engine()
        engine.contextualize(PANEL, BUBBLES)

        assert engine.provider == "qwen"
        (call,) = completions.calls
        assert call["model"] == PROVIDER_PRESETS["qwen"].default_model

    def test_sends_data_uri_png_and_bubble_texts(self) -> None:
        engine, completions = make_engine()
        engine.contextualize(PANEL, BUBBLES)

        (call,) = completions.calls
        image_part, text_part = call["messages"][1]["content"]
        assert image_part["type"] == "image_url"
        url = image_part["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.standard_b64decode(url.split(",", 1)[1]).startswith(b"\x89PNG")
        assert "Eu vou sobreviver!" in text_part["text"]
        assert call["messages"][0]["role"] == "system"

    def test_parses_blocks_from_response(self) -> None:
        engine, _ = make_engine()
        blocks = engine.contextualize(PANEL, BUBBLES)
        assert [(b.text, b.speaker_id) for b in blocks] == [("Eu vou sobreviver!", "Guts")]

    def test_garbage_response_falls_back_to_passthrough(self) -> None:
        engine, _ = make_engine(content="sem json aqui")
        blocks = engine.contextualize(PANEL, BUBBLES)
        assert all(b.speaker_id == "Desconhecido" for b in blocks)
        assert len(blocks) == 2

    def test_model_override_wins_over_preset(self) -> None:
        engine, completions = make_engine(provider="openai", model="gpt-5-mini")
        engine.contextualize(PANEL, BUBBLES)
        assert completions.calls[0]["model"] == "gpt-5-mini"

    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            OpenAICompatibleVisionLanguageEngine(provider="gemini")

    def test_fingerprint_tracks_prompt_changes(self) -> None:
        """Adding SFX tags (or any prompt change) must alter the
        fingerprint so stale scripts get invalidated — the bug that hid
        the sound effects in the first SFX run."""
        plain, _ = make_engine()
        with_sfx, _ = make_engine(sfx_tags=("espada", "explosao"))

        assert plain.fingerprint != with_sfx.fingerprint
        assert plain.fingerprint.startswith("vlm-api:qwen:")

    def test_missing_api_key_raises_actionable_error(self, monkeypatch) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        engine = OpenAICompatibleVisionLanguageEngine(provider="qwen")
        with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
            engine.contextualize(PANEL, BUBBLES)


class TestFactory:
    def test_api_providers_resolve_to_compatible_engine(self) -> None:
        for provider in ("qwen", "openai", "kimi"):
            engine = create_vlm_engine(provider)
            assert isinstance(engine, OpenAICompatibleVisionLanguageEngine)
            assert engine.provider == provider

    def test_passthrough_provider(self) -> None:
        assert isinstance(create_vlm_engine("passthrough"), PassthroughVLM)

    def test_anthropic_provider(self) -> None:
        from mangawhisperer.engines.vlm import ClaudeVisionLanguageEngine

        engine = create_vlm_engine("anthropic", model="claude-haiku-4-5")
        assert isinstance(engine, ClaudeVisionLanguageEngine)
        assert engine.model == "claude-haiku-4-5"

    def test_auto_prefers_qwen_key(self, monkeypatch) -> None:
        for var in ("DASHSCOPE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MOONSHOT_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test2")

        engine = create_vlm_engine("auto")
        assert isinstance(engine, OpenAICompatibleVisionLanguageEngine)
        assert engine.provider == "qwen"

    def test_auto_without_keys_falls_back_to_local(self, monkeypatch) -> None:
        for var in ("DASHSCOPE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "MOONSHOT_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        from mangawhisperer.engines.vlm_local import QwenVisionLanguageEngine

        assert isinstance(create_vlm_engine("auto"), QwenVisionLanguageEngine)

    def test_unknown_provider_rejected(self) -> None:
        with pytest.raises(ValueError):
            create_vlm_engine("gemini")
