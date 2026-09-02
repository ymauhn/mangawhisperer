"""Multi-provider API Vision-Language engine (OpenAI-compatible protocol).

One engine class covers every provider that speaks the OpenAI
``chat.completions`` protocol with vision — which is how Qwen
(DashScope), OpenAI and Kimi (Moonshot) all expose their VLMs. Each
provider is a preset (base URL + key env var + default model); adding a
new one is a dictionary entry, not a new class. Anthropic has richer
native structured outputs, so it keeps its dedicated engine in
``engines/vlm.py``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Sequence

import cv2

from mangawhisperer.engines.script_parsing import parse_script_blocks, passthrough_blocks
from mangawhisperer.engines.vlm import DEFAULT_CAST, build_scriptwriter_prompt
from mangawhisperer.interfaces import Image, VisionLanguageEngine
from mangawhisperer.models import ContextualizedBlock, SpeechBubble

logger = logging.getLogger(__name__)

JSON_INSTRUCTION = """

Responda APENAS com um array JSON, sem markdown e sem texto extra, no formato:
[{"text": "...", "speaker_id": "...", "is_speech": true, "voice": "homem"}, ...]
"voice" só nos blocos de fala: homem, mulher, idoso, idosa, menino, menina ou criatura.
Nunca inclua blocos com "text" vazio — simplesmente omita-os.
No máximo 2 blocos de ação (is_speech=false) por painel.\
"""
"""Prompt tail shared by the OpenAI-protocol engines (API and llama-server)."""
_JSON_INSTRUCTION = JSON_INSTRUCTION


def bubble_request_text(bubbles: Sequence[SpeechBubble]) -> str:
    """The user-turn text: the OCR'd bubble texts as a JSON list."""
    return (
        "Textos das bolhas de fala, em ordem de leitura (lista JSON):\n"
        + json.dumps([b.text for b in bubbles], ensure_ascii=False)
    )


def encode_panel_png(image: Image, max_edge: int) -> str:
    """Downscale (if the long edge exceeds ``max_edge``) and encode the
    panel as base64 PNG — the image currency of every OpenAI-protocol
    engine."""
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge > max_edge:
        scale = max_edge / long_edge
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise ValueError("Failed to encode panel image as PNG")
    return base64.standard_b64encode(buffer.tobytes()).decode("ascii")


def png_data_url(image: Image, max_edge: int) -> str:
    """``data:image/png;base64,...`` for an ``image_url`` content part."""
    return "data:image/png;base64," + encode_panel_png(image, max_edge)


@dataclass(frozen=True)
class ProviderPreset:
    """Connection defaults for one OpenAI-compatible provider."""

    base_url: str | None
    api_key_env: str
    default_model: str


PROVIDER_PRESETS: dict[str, ProviderPreset] = {
    # Alibaba Model Studio international. This is the legacy endpoint —
    # still live, but Alibaba is migrating to workspace-scoped URLs
    # (https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/...); pass
    # base_url= to override when your account requires it.
    "qwen": ProviderPreset(
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        api_key_env="DASHSCOPE_API_KEY",
        default_model="qwen3-vl-plus",  # qwen3-vl-flash is the budget option
    ),
    "openai": ProviderPreset(
        base_url=None,  # SDK default
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-5.6-luna",  # cheap vision tier; gpt-5-mini also solid
    ),
    "kimi": ProviderPreset(
        base_url="https://api.moonshot.ai/v1",
        api_key_env="MOONSHOT_API_KEY",
        default_model="kimi-k2.6",  # moonshot-v1-*-vision sunsets 2026-08-31
    ),
}


class OpenAICompatibleVisionLanguageEngine(VisionLanguageEngine):
    """Scriptwriter backed by any OpenAI-compatible vision API.

    The client is created lazily (injectable for tests). A missing API
    key raises a clear error naming the env var to set.
    """

    def __init__(
        self,
        provider: str = "qwen",
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        known_characters: Sequence[str] = DEFAULT_CAST,
        sfx_tags: Sequence[str] = (),
        sfx_intensity: int = 2,
        style_addendum: str = "",
        max_tokens: int = 2048,
        max_image_edge: int = 1568,
        client: Any = None,
    ) -> None:
        """
        Args:
            provider: One of :data:`PROVIDER_PRESETS` (``qwen``,
                ``openai``, ``kimi``).
            model: Override the preset's default model id.
            api_key: Explicit key; defaults to the preset's env var.
            base_url: Override the preset's endpoint.
            known_characters: Cast names for speaker attribution.
            max_tokens: Output budget per panel.
            max_image_edge: Long-edge downscale cap before upload
                (controls image-token cost).
            client: Injectable OpenAI-compatible client for tests.
        """
        if provider not in PROVIDER_PRESETS:
            raise ValueError(
                f"Unknown provider {provider!r}; expected one of {sorted(PROVIDER_PRESETS)}"
            )
        self.provider = provider
        self._preset = PROVIDER_PRESETS[provider]
        self.model = model or self._preset.default_model
        self._api_key = api_key
        self._base_url = base_url if base_url is not None else self._preset.base_url
        self._max_tokens = max_tokens
        self._max_image_edge = max_image_edge
        self._client = client
        self._system_prompt = (
            build_scriptwriter_prompt(known_characters, sfx_tags, sfx_intensity, style_addendum)
            + JSON_INSTRUCTION
        )

    @property
    def fingerprint(self) -> str:
        """Checkpoint identity: provider + model + prompt define the output."""
        digest = hashlib.sha1(self._system_prompt.encode("utf-8")).hexdigest()[:8]
        return f"vlm-api:{self.provider}:{self.model}:prompt={digest}"

    def preflight(self) -> None:
        """Fail fast (before any pipeline work) if credentials are missing."""
        if self._client is None and not (self._api_key or os.environ.get(self._preset.api_key_env)):
            raise RuntimeError(
                f"Sem chave de API para o provedor '{self.provider}': defina a variável "
                f"de ambiente {self._preset.api_key_env} (ou passe api_key=...)."
            )

    def contextualize(
        self, panel_image: Image, bubbles: list[SpeechBubble]
    ) -> list[ContextualizedBlock]:
        """Produce the ordered narration script for one panel."""
        request_text = bubble_request_text(bubbles)
        response = self._get_client().chat.completions.create(
            model=self.model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": png_data_url(panel_image, self._max_image_edge)},
                        },
                        {"type": "text", "text": request_text},
                    ],
                },
            ],
        )
        raw = response.choices[0].message.content or ""

        blocks = parse_script_blocks(raw)
        if blocks is None or (not blocks and bubbles):
            logger.warning(
                "%s/%s output yielded no usable blocks (%r...); falling back to "
                "passthrough for %d bubbles",
                self.provider,
                self.model,
                raw[:80],
                len(bubbles),
            )
            return passthrough_blocks(bubbles)
        return blocks

    def _get_client(self) -> Any:
        if self._client is None:
            self.preflight()
            from openai import OpenAI  # noqa: PLC0415 — deferred so tests need no SDK

            api_key = self._api_key or os.environ.get(self._preset.api_key_env)
            self._client = OpenAI(api_key=api_key, base_url=self._base_url)
        return self._client

    def _encode_png(self, image: Image) -> str:
        """Downscale (if needed) and encode the panel as base64 PNG."""
        return encode_panel_png(image, self._max_image_edge)

