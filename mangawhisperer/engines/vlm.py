"""Claude-backed Vision-Language engine: the pipeline's scriptwriter.

Sends the panel image (base64 PNG) plus the OCR'd bubble texts to the
Claude API and gets back a validated, speaker-attributed script via
structured outputs (``client.messages.parse`` against a Pydantic
schema). Deployment-agnostic per the project's ABC contract — swapping
in a local VLM later only requires another ``VisionLanguageEngine``
implementation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from typing import Any, Sequence

import cv2
import numpy as np
from pydantic import BaseModel

from mangawhisperer.interfaces import Image, VisionLanguageEngine
from mangawhisperer.models import ContextualizedBlock, SpeechBubble

logger = logging.getLogger(__name__)

DEFAULT_CAST: tuple[str, ...] = (
    "Guts",
    "Griffith",
    "Casca",
    "Puck",
    "Judeau",
    "Pippin",
    "Corkus",
    "Rickert",
    "Zodd",
)

_SYSTEM_PROMPT_TEMPLATE = """\
You are the scriptwriter for MangaWhisperer, an accessibility pipeline that \
turns Portuguese manga into immersive audio for visually impaired listeners.

You receive one manga panel image and the OCR text of its speech bubbles in \
reading order (right-to-left, top-to-bottom). Produce the narration script \
for this panel as structured blocks:

1. Dialogue blocks (is_speech=true): one per real speech bubble, preserving \
the given order. Keep the Brazilian Portuguese text as spoken; you may fix \
obvious OCR mistakes (e.g. a '0' misread for an 'O'). Attribute each block \
to the character who speaks it, using exactly one name from the known cast \
when you can identify them: {cast}. When the speaker is NOT in the cast, \
invent a SHORT descriptive Portuguese label instead — "Criatura", "Soldado", \
"Aldeã", "Comandante" — and reuse the SAME label for the same character in \
every panel. Use "Desconhecido" only as a last resort.
2. Onomatopoeia — interpret, never copy literally:
   - Elongated emphatic speech ("SIMMMMM!!!", "NÃAAOO") IS real dialogue: \
normalize it into a natural expressive form ("Siiim!", "Nããão!") that a \
voice actor would read, keeping the emphatic punctuation.
   - Written sound effects that nobody speaks ("CRASH", "VROOM", "GROAAR") \
are NOT dialogue: drop them, or fold what they convey into an action block \
("um rugido ensurdecedor ecoa").
   - Drop illegible OCR fragments entirely.
3. Expressive punctuation: the text-to-speech engine uses punctuation for \
intonation. Preserve exclamation marks, question marks and ellipses in \
dialogue; use commas to mark dramatic pauses in action descriptions.
4. Action blocks (is_speech=false, speaker_id="Narrator"): audio description \
for a blind listener, in Brazilian Portuguese. Describe ONLY what is drawn in \
the panel — who is present, where they are, what they do, what changes. Never \
invent atmosphere, sounds, smells, thoughts or feelings that are not visibly \
drawn (a drawn expression may be named plainly: "Guts cerra os dentes"). Do \
not restate what the dialogue already says. Be brief: one action block per \
panel, two at most when the action changes mid-panel, each about 20 words or \
fewer, in plain direct sentences — no metaphors, no stacked adjectives. Place \
each one where it belongs in the narration order. If the panel has no \
dialogue, describe the scene in one plain sentence.
5. Voice profile: for every dialogue block set "voice" to how the speaker \
looks as drawn, exactly one of: homem, mulher, idoso, idosa, menino, menina, \
criatura. Keep it identical for the same character across panels. Leave it \
null for narration blocks.\
{sfx_section}

Every output text must be in Brazilian Portuguese.\
"""

_SFX_SCENE_MAP = """Typical scene mapping: blades/cuts -> espada; physical \
impacts/blows -> soco; destruction/blasts -> explosao; a creature appearing, \
roaring or attacking -> monstro; flames -> fogo; wind or fast movement -> \
vento; screams of pain or terror -> grito; approaching steps -> passos; \
storms/dramatic reveals -> trovao; doors -> porta."""

_SFX_SECTION_LEVELS = {
    1: """
6. Sound effects: any block may include an "sfx" field naming ONE tag played \
right before it. Use one only when the scene truly demands it — a single \
striking moment per panel at most. {scene_map} Available tags: {tags}.\
""",
    2: """
6. Sound effects bring the audio drama to life — actively look for the \
chance to use one in every panel that shows physical action. Any block may \
include an "sfx" field naming ONE tag played right before it (usually on an \
action block). {scene_map} When an action matches a tag, USE it — prefer \
adding an effect over skipping it. Up to 2 effects per panel. Available \
tags: {tags}.\
""",
    3: """
6. Sound effects are a core part of this production — EVERY panel with any \
physical action, movement or atmosphere should carry at least one. Any \
block may include an "sfx" field naming ONE tag played right before it. \
{scene_map} Be generous: if any tag remotely fits the scene, use it. Up to \
3 effects per panel. Available tags: {tags}.\
""",
}


def build_scriptwriter_prompt(
    cast: Sequence[str],
    sfx_tags: Sequence[str] = (),
    sfx_intensity: int = 2,
    style_addendum: str = "",
) -> str:
    """Compose the shared scriptwriter system prompt for every VLM engine.

    ``sfx_intensity`` (0-3) picks the SFX-rule wording: 0 removes the
    section entirely, 1 is sparing, 2 encouraging, 3 aggressive.
    ``style_addendum`` appends a narration-style directive (see
    :mod:`mangawhisperer.engines.styles`).
    """
    sfx_section = ""
    if sfx_tags and sfx_intensity > 0:
        template = _SFX_SECTION_LEVELS[min(max(sfx_intensity, 1), 3)]
        sfx_section = template.format(scene_map=_SFX_SCENE_MAP, tags=", ".join(sfx_tags))
    prompt = _SYSTEM_PROMPT_TEMPLATE.format(cast=", ".join(cast), sfx_section=sfx_section)
    return prompt + style_addendum


class PanelScript(BaseModel):
    """Structured-output contract returned by the Claude scriptwriter."""

    blocks: list[ContextualizedBlock]


class ClaudeVisionLanguageEngine(VisionLanguageEngine):
    """Speaker diarization + audio description via the Claude API.

    The Anthropic client is created lazily on first use, so constructing
    this engine is free and tests can inject a fake client. Credentials
    resolve from the environment (``ANTHROPIC_API_KEY`` or an
    ``ant auth login`` profile).
    """

    def __init__(
        self,
        model: str = "claude-opus-4-8",
        known_characters: Sequence[str] = DEFAULT_CAST,
        sfx_tags: Sequence[str] = (),
        sfx_intensity: int = 2,
        style_addendum: str = "",
        max_tokens: int = 8192,
        max_image_edge: int = 1568,
        client: Any = None,
    ) -> None:
        """
        Args:
            model: Claude model ID.
            known_characters: Cast names the model should attribute
                dialogue to (curated-cast decision); anything else
                becomes "Desconhecido".
            max_tokens: Output budget per panel (thinking included).
            max_image_edge: Panels larger than this on their long edge
                are downscaled before upload to control image-token
                cost; manga art keeps its legibility well below the
                model's 2576px ceiling.
            client: Injectable Anthropic-compatible client for tests;
                ``None`` creates a real one lazily.
        """
        self.provider = "anthropic"
        self.model = model
        self._max_tokens = max_tokens
        self._max_image_edge = max_image_edge
        self._client = client
        self._system_prompt = build_scriptwriter_prompt(
            known_characters, sfx_tags, sfx_intensity, style_addendum
        )

    @property
    def fingerprint(self) -> str:
        """Checkpoint identity: model + prompt (cast/SFX/rules) define
        the output — a prompt change must invalidate old scripts."""
        digest = hashlib.sha1(self._system_prompt.encode("utf-8")).hexdigest()[:8]
        return f"anthropic:{self.model}:prompt={digest}"

    def contextualize(
        self, panel_image: Image, bubbles: list[SpeechBubble]
    ) -> list[ContextualizedBlock]:
        """Produce the ordered narration script for one panel.

        Falls back to unattributed passthrough blocks if the request is
        refused or the structured output cannot be parsed, so one bad
        panel never aborts a volume-length run.
        """
        bubble_texts = [b.text for b in bubbles]
        request_text = (
            "Textos das bolhas de fala, em ordem de leitura (lista JSON):\n"
            + json.dumps(bubble_texts, ensure_ascii=False)
        )

        response = self._get_client().messages.parse(
            model=self.model,
            max_tokens=self._max_tokens,
            thinking={"type": "adaptive"},
            system=self._system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": self._encode_png(panel_image),
                            },
                        },
                        {"type": "text", "text": request_text},
                    ],
                }
            ],
            output_format=PanelScript,
        )

        if response.stop_reason == "refusal" or response.parsed_output is None:
            logger.warning(
                "VLM did not return a script (stop_reason=%s); falling back to "
                "passthrough blocks for %d bubbles",
                response.stop_reason,
                len(bubbles),
            )
            return self._fallback_blocks(bubbles)
        return list(response.parsed_output.blocks)

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic  # noqa: PLC0415 — deferred so tests need no SDK

            self._client = anthropic.Anthropic()
        return self._client

    def _encode_png(self, image: Image) -> str:
        """Downscale (if needed) and encode the panel as base64 PNG."""
        height, width = image.shape[:2]
        long_edge = max(height, width)
        if long_edge > self._max_image_edge:
            scale = self._max_image_edge / long_edge
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

    @staticmethod
    def _fallback_blocks(bubbles: list[SpeechBubble]) -> list[ContextualizedBlock]:
        return [
            ContextualizedBlock(text=b.text, speaker_id="Desconhecido", is_speech=True)
            for b in bubbles
            if b.text.strip()
        ]
