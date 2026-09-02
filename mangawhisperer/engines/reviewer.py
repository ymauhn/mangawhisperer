"""LLM-backed script reviewer — the pipeline's quality layer.

Text-only (no images), so it costs roughly a tenth of the writer stage
while seeing what the writer never does: the whole volume at once.
Fixes speaker-label inconsistencies, normalizes text for read-aloud
delivery, sanity-checks SFX tags and continuity. Conservative by
construction: any parsing/API failure returns the original script.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Sequence

from pydantic import BaseModel, TypeAdapter, ValidationError

from mangawhisperer.engines.vlm_api import PROVIDER_PRESETS
from mangawhisperer.interfaces import ScriptReviewer
from mangawhisperer.models import ContextualizedBlock, PanelData

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """\
You are the script REVIEWER for MangaWhisperer, an accessibility audio-drama
pipeline. You receive consecutive panels of an existing PT-BR narration
script as JSON and return the corrected version. Apply ONLY these fixes:

1. Speaker consistency: the same character must use the SAME label in every
panel. Known cast: {cast}. Descriptive labels ("Criatura", "Soldado") must
be reused consistently; replace "Desconhecido" with a cast name or an
existing descriptive label when the context makes the speaker obvious.
2. Read-aloud normalization: numbers, abbreviations and symbols become
spoken Brazilian Portuguese words ("Cap. 3" -> "Capítulo três").
3. Continuity: fix obvious contradictions between consecutive action
descriptions; keep them short and vivid.
4. Expressive punctuation: dialogue keeps ! and ? that convey intonation.
{sfx_rule}
Never: invent new dialogue, reorder or merge blocks/panels, change what is
said, or output any language other than Brazilian Portuguese.

Return JSON in EXACTLY this shape, including EVERY panel you received
(changed or not): {{"panels": [{{"panel": <index>, "blocks": [{{"text": ...,
"speaker_id": ..., "is_speech": ..., "sfx": ...}}]}}]}}\
"""

_SFX_RULE = """5. SFX sanity: the "sfx" field must be one of [{tags}] or
omitted; remove invalid tags; keep at most 2 effects per panel."""


class ReviewedPanel(BaseModel):
    panel: int
    blocks: list[ContextualizedBlock]


class ReviewedScript(BaseModel):
    panels: list[ReviewedPanel]


_REVIEWED_ADAPTER: TypeAdapter[ReviewedScript] = TypeAdapter(ReviewedScript)


class LLMScriptReviewer(ScriptReviewer):
    """Reviewer over any supported LLM provider (text-only chat)."""

    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        known_characters: Sequence[str] = (),
        sfx_tags: Sequence[str] = (),
        chunk_size: int = 40,
        max_tokens: int = 8192,
        client: Any = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        """
        Args:
            provider: ``anthropic``, an OpenAI-compatible preset
                (``qwen``/``openai``/``kimi``), or ``llamacpp`` — the
                local llama-server the scriptwriter runs on, addressed
                by ``base_url`` (no API key needed).
            model: Override the provider's default model.
            known_characters: Cast names for label consistency.
            sfx_tags: Valid effect tags (enables the SFX sanity rule).
            chunk_size: Panels per review request.
            max_tokens: Output budget per chunk.
            client: Injectable client for tests.
            base_url: OpenAI-compatible endpoint override (required for
                ``llamacpp``, e.g. ``http://127.0.0.1:8080/v1``).
            api_key: Explicit key; defaults to the preset's env var
                (``llamacpp`` uses a placeholder).
        """
        self.provider = provider
        if provider in ("anthropic", "claude"):
            self.provider = "anthropic"
            self.model = model or "claude-opus-4-8"
        elif provider in PROVIDER_PRESETS:
            self.model = model or PROVIDER_PRESETS[provider].default_model
        elif provider == "llamacpp":
            self.model = model or "llama-server"
        else:
            raise ValueError(f"Reviewer: unknown provider {provider!r}")
        self._sfx_tags = tuple(sfx_tags)
        self._chunk_size = chunk_size
        self._max_tokens = max_tokens
        self._client = client
        self._base_url = base_url
        self._api_key = api_key
        sfx_rule = _SFX_RULE.format(tags=", ".join(sfx_tags)) if sfx_tags else ""
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            cast=", ".join(known_characters) or "(nenhum)", sfx_rule=sfx_rule
        )

    @property
    def base_url(self) -> str | None:
        """The OpenAI-compatible endpoint this reviewer talks to (if overridden)."""
        return self._base_url

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha1(self._system_prompt.encode("utf-8")).hexdigest()[:8]
        return f"reviewer:{self.provider}:{self.model}:prompt={digest}"

    def review(self, panels: list[PanelData]) -> list[PanelData]:
        """Review the script chunk by chunk; failures keep originals."""
        reviewed = list(panels)
        labels_seen: set[str] = set()
        for start in range(0, len(panels), self._chunk_size):
            chunk = panels[start : start + self._chunk_size]
            try:
                result = self._review_chunk(chunk, start, labels_seen)
            except Exception as exc:
                logger.warning("Reviewer chunk %d failed (%s); keeping original", start, exc)
                result = None
            if result is not None:
                by_index = {p.panel: p.blocks for p in result.panels}
                for offset, panel in enumerate(chunk):
                    blocks = by_index.get(start + offset)
                    if blocks:  # missing/empty -> keep the original panel
                        reviewed[start + offset] = panel.model_copy(update={"blocks": blocks})
            for panel in reviewed[start : start + len(chunk)]:
                labels_seen.update(b.speaker_id for b in panel.blocks)
        changed = sum(
            1 for old, new in zip(panels, reviewed, strict=True) if old.blocks != new.blocks
        )
        logger.info("Reviewer: %d/%d panels adjusted", changed, len(panels))
        return reviewed

    def _review_chunk(
        self, chunk: list[PanelData], start: int, labels_seen: set[str]
    ) -> ReviewedScript | None:
        payload = {
            "panels": [
                {
                    "panel": start + offset,
                    "blocks": [b.model_dump(exclude_none=True) for b in panel.blocks],
                }
                for offset, panel in enumerate(chunk)
            ]
        }
        context = (
            "Rótulos de personagens já usados em painéis anteriores: "
            + (", ".join(sorted(labels_seen)) if labels_seen else "(nenhum ainda)")
        )
        user_text = context + "\n\nRoteiro para revisar:\n" + json.dumps(
            payload, ensure_ascii=False
        )

        if self.provider == "anthropic":
            return self._review_anthropic(user_text)
        return self._review_openai_compatible(user_text)

    def _review_anthropic(self, user_text: str) -> ReviewedScript | None:
        client = self._get_anthropic_client()
        response = client.messages.parse(
            model=self.model,
            max_tokens=self._max_tokens,
            thinking={"type": "adaptive"},
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_text}],
            output_format=ReviewedScript,
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            return None
        return response.parsed_output

    def _review_openai_compatible(self, user_text: str) -> ReviewedScript | None:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self._max_tokens,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_text},
            ],
        )
        raw = response.choices[0].message.content or ""
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            return _REVIEWED_ADAPTER.validate_python(json.loads(raw[start : end + 1]))
        except (json.JSONDecodeError, ValidationError):
            return None

    def _get_anthropic_client(self) -> Any:
        if self._client is None:
            import anthropic  # noqa: PLC0415 — deferred

            self._client = anthropic.Anthropic()
        return self._client

    def _get_openai_client(self) -> Any:
        if self._client is None:
            if self.provider == "llamacpp":
                from openai import OpenAI  # noqa: PLC0415 — deferred

                # A local server checks no key, but the SDK insists on one.
                self._client = OpenAI(api_key=self._api_key or "sk-local", base_url=self._base_url)
                return self._client
            preset = PROVIDER_PRESETS[self.provider]
            api_key = self._api_key or os.environ.get(preset.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"Reviewer sem chave: defina {preset.api_key_env} ou use --no-review."
                )
            from openai import OpenAI  # noqa: PLC0415 — deferred

            self._client = OpenAI(api_key=api_key, base_url=self._base_url or preset.base_url)
        return self._client
