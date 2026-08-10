"""Tests for the script Reviewer layer (fake clients — offline)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cv2")  # reviewer imports provider presets from vlm_api

from mangawhisperer.engines.factory import create_reviewer  # noqa: E402
from mangawhisperer.engines.reviewer import (  # noqa: E402
    LLMScriptReviewer,
    ReviewedPanel,
    ReviewedScript,
)
from mangawhisperer.models import BoundingBox, ContextualizedBlock, PanelData  # noqa: E402

BBOX = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.9, y_max=0.9)


def make_panel(index: int, *blocks: ContextualizedBlock) -> PanelData:
    return PanelData(
        image_path=Path(f"panels/p{index}.npy"), bbox=BBOX,
        page_number=index + 1, panel_index=0, blocks=list(blocks),
    )


GUTS = ContextualizedBlock(text="Eu vou sobreviver!", speaker_id="Guts", is_speech=True)
MONSTRO = ContextualizedBlock(text="Você é meu!", speaker_id="Monstro", is_speech=True)
CRIATURA = ContextualizedBlock(text="Você é meu!", speaker_id="Criatura", is_speech=True)


class FakeAnthropicMessages:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self.response


def make_anthropic_reviewer(parsed, stop_reason: str = "end_turn", **kwargs):
    response = SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason)
    client = SimpleNamespace(messages=FakeAnthropicMessages(response))
    return LLMScriptReviewer(provider="anthropic", client=client, **kwargs), client


class TestLLMScriptReviewer:
    def test_applies_speaker_consistency_fix(self) -> None:
        panels = [make_panel(0, GUTS), make_panel(1, MONSTRO)]
        reviewed_payload = ReviewedScript(panels=[
            ReviewedPanel(panel=0, blocks=[GUTS]),
            ReviewedPanel(panel=1, blocks=[CRIATURA]),  # Monstro -> Criatura
        ])
        reviewer, _ = make_anthropic_reviewer(reviewed_payload)

        result = reviewer.review(panels)
        assert result[1].blocks[0].speaker_id == "Criatura"
        assert result[0].blocks == [GUTS]
        assert result[1].image_path == panels[1].image_path, "non-block fields untouched"

    def test_missing_panel_in_response_keeps_original(self) -> None:
        panels = [make_panel(0, GUTS), make_panel(1, MONSTRO)]
        reviewer, _ = make_anthropic_reviewer(
            ReviewedScript(panels=[ReviewedPanel(panel=0, blocks=[GUTS])])
        )

        result = reviewer.review(panels)
        assert result[1].blocks[0].speaker_id == "Monstro"

    def test_refusal_keeps_everything(self) -> None:
        panels = [make_panel(0, GUTS)]
        reviewer, _ = make_anthropic_reviewer(None, stop_reason="refusal")
        assert reviewer.review(panels) == panels

    def test_client_exception_keeps_everything(self) -> None:
        class ExplodingMessages:
            def parse(self, **kwargs):
                raise RuntimeError("api down")

        reviewer = LLMScriptReviewer(
            provider="anthropic", client=SimpleNamespace(messages=ExplodingMessages())
        )
        panels = [make_panel(0, GUTS)]
        assert reviewer.review(panels) == panels

    def test_openai_compatible_path_parses_json_content(self, monkeypatch) -> None:
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
        payload = json.dumps({
            "panels": [{"panel": 0, "blocks": [
                {"text": "Você é meu!", "speaker_id": "Criatura", "is_speech": True}
            ]}]
        }, ensure_ascii=False)
        completions = SimpleNamespace(
            create=lambda **kw: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=f"Claro!\n{payload}"))]
            )
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        reviewer = LLMScriptReviewer(provider="qwen", client=client)

        result = reviewer.review([make_panel(0, MONSTRO)])
        assert result[0].blocks[0].speaker_id == "Criatura"

    def test_labels_seen_accumulate_across_chunks(self) -> None:
        panels = [make_panel(i, GUTS) for i in range(3)]
        reviewer, client = make_anthropic_reviewer(
            ReviewedScript(panels=[]), chunk_size=1
        )
        reviewer.review(panels)

        assert len(client.messages.calls) == 3
        later_context = client.messages.calls[2]["messages"][0]["content"]
        assert "Guts" in later_context, "labels from earlier chunks inform later ones"

    def test_fingerprint_covers_provider_model_and_prompt(self) -> None:
        a = LLMScriptReviewer(provider="anthropic", client=object())
        b = LLMScriptReviewer(provider="anthropic", client=object(), sfx_tags=("espada",))
        assert a.fingerprint != b.fingerprint
        assert a.fingerprint.startswith("reviewer:anthropic:claude-opus-4-8")


class TestReviewerFactory:
    def test_local_and_passthrough_have_no_reviewer(self) -> None:
        assert create_reviewer("passthrough") is None
        assert create_reviewer("qwen-local") is None

    def test_api_providers_build_reviewer(self) -> None:
        reviewer = create_reviewer("anthropic", model="claude-haiku-4-5")
        assert isinstance(reviewer, LLMScriptReviewer)
        assert reviewer.model == "claude-haiku-4-5"
