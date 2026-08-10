"""Tolerant parsing of VLM script output shared by all API/local engines.

Real models misbehave in predictable ways — they emit blocks with empty
text, wrap the JSON in prose, or get truncated mid-array by the token
limit. Rejecting the whole panel for one bad block (the original strict
behavior) threw away every good block with it; this parser salvages
instead: recover every complete JSON object, drop invalid ones, keep
the rest.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from mangawhisperer.models import ContextualizedBlock, SpeechBubble

logger = logging.getLogger(__name__)


def parse_script_blocks(raw: str) -> list[ContextualizedBlock] | None:
    """Extract narration blocks from raw model text, salvaging what's valid.

    Returns ``None`` only when no JSON array/objects can be found at
    all; otherwise returns the valid blocks (possibly an empty list —
    e.g. a panel whose blocks were all empty-text noise).
    """
    start = raw.find("[")
    if start == -1:
        return None
    candidate = raw[start:]

    items: list[Any] | None = None
    end = candidate.rfind("]")
    if end != -1:
        try:
            parsed = json.loads(candidate[: end + 1])
            if isinstance(parsed, list):
                items = parsed
        except json.JSONDecodeError:
            items = None
    if items is None:
        # Truncated or dirty array: recover every complete {...} object.
        items = _salvage_objects(candidate)
        if not items:
            return None

    blocks: list[ContextualizedBlock] = []
    dropped = 0
    for item in items:
        block = _coerce_block(item)
        if block is None:
            dropped += 1
        else:
            blocks.append(block)
    if dropped:
        logger.info("Dropped %d invalid/empty script blocks, kept %d", dropped, len(blocks))
    return blocks


def passthrough_blocks(bubbles: list[SpeechBubble]) -> list[ContextualizedBlock]:
    """Last-resort script: every non-empty bubble as unattributed speech."""
    return [
        ContextualizedBlock(text=b.text, speaker_id="Desconhecido", is_speech=True)
        for b in bubbles
        if b.text.strip()
    ]


def _coerce_block(item: Any) -> ContextualizedBlock | None:
    """Build a block from one parsed item; ``None`` if unsalvageable.

    Empty-text blocks (a common Qwen-VL failure mode) are dropped rather
    than failing validation for the whole panel.
    """
    if not isinstance(item, dict):
        return None
    text = str(item.get("text") or "").strip()
    if not text:
        return None
    speaker = str(item.get("speaker_id") or "").strip() or "Desconhecido"
    sfx_raw = item.get("sfx")
    sfx = sfx_raw.strip().lower() if isinstance(sfx_raw, str) and sfx_raw.strip() else None
    try:
        return ContextualizedBlock(
            text=text,
            speaker_id=speaker,
            is_speech=bool(item.get("is_speech", True)),
            sfx=sfx,
        )
    except ValidationError:
        return None


def _salvage_objects(text: str) -> list[Any]:
    """Decode consecutive complete JSON objects, stopping at truncation."""
    decoder = json.JSONDecoder()
    items: list[Any] = []
    index = 0
    while True:
        brace = text.find("{", index)
        if brace == -1:
            break
        try:
            obj, consumed = decoder.raw_decode(text[brace:])
        except json.JSONDecodeError:
            break  # incomplete trailing object — the truncation point
        items.append(obj)
        index = brace + consumed
    return items
