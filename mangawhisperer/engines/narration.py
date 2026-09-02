"""Pure narration planning — decoupled from synthesis (ticket #10).

Given a script, decide *what* gets rendered in *which order*: speech
blocks, narrator announcements when the speaker changes, and the sound
effects attached to blocks. Nothing here touches audio, models or the
filesystem, so every rule is unit-testable in milliseconds. The
orchestrator only renders the resulting plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from mangawhisperer.engines.casting import majority_profile
from mangawhisperer.engines.sfx import suggest_tag
from mangawhisperer.models import ContextualizedBlock, PanelData

NARRATOR = "Narrator"


@dataclass(frozen=True)
class NarrationItem:
    """One entry of the render plan."""

    kind: Literal["sfx", "announcement", "speech"]
    block: ContextualizedBlock | None = None
    sfx_tag: str | None = None
    seed: str = ""


def announcement_for(speaker_id: str) -> ContextualizedBlock:
    """Narrator block that speaks the character's label before their line."""
    return ContextualizedBlock(text=speaker_id, speaker_id=NARRATOR, is_speech=False)


def plan_narration(
    panels: Iterable[PanelData],
    *,
    announce_speakers: bool,
    sfx_enabled: bool,
) -> list[NarrationItem]:
    """Flatten panels into the ordered render plan.

    Rules: an effect plays right before the block that carries it; the
    narrator announces a speaker only when the speaker *changes*
    (consecutive lines by the same character are not re-announced, and
    narrator interludes do not reset the memory); the narrator is never
    announced.
    """
    plan: list[NarrationItem] = []
    last_speaker: str | None = None
    for panel in panels:
        for block in panel.blocks:
            if sfx_enabled and block.sfx:
                plan.append(NarrationItem(kind="sfx", sfx_tag=block.sfx, seed=block.text))
            if (
                announce_speakers
                and block.is_speech
                and block.speaker_id != NARRATOR
                and block.speaker_id != last_speaker
            ):
                plan.append(NarrationItem(kind="announcement", block=announcement_for(block.speaker_id)))
            if block.is_speech:
                last_speaker = block.speaker_id
            plan.append(NarrationItem(kind="speech", block=block))
    return plan


def consolidate_voice_profiles(*panel_lists: Iterable[PanelData]) -> dict[str, str | None]:
    """One voice profile per speaker: the majority of the scriptwriter's
    per-block votes (several script versions may be passed — e.g. the raw
    and the reviewed one — so a reviewer that drops the field costs
    nothing). Speakers without any vote map to None."""
    votes: dict[str, list[str | None]] = {}
    for panels in panel_lists:
        for panel in panels:
            for block in panel.blocks:
                if block.is_speech and block.speaker_id != NARRATOR:
                    votes.setdefault(block.speaker_id, []).append(block.voice)
    return {speaker: majority_profile(v) for speaker, v in votes.items()}


def auto_tag_sfx(
    blocks: list[ContextualizedBlock],
    available_tags: set[str],
    intensity: int,
) -> list[ContextualizedBlock]:
    """Keyword safety net for effects the scriptwriter skipped.

    The per-panel cap equals ``intensity`` (1–3) and counts the
    scriptwriter's own choices; level 3 also considers dialogue lines.
    ``intensity`` 0 returns the blocks untouched.
    """
    cap = max(0, min(intensity, 3))
    if cap == 0 or not available_tags:
        return list(blocks)
    used = sum(1 for b in blocks if b.sfx)
    tagged: list[ContextualizedBlock] = []
    for block in blocks:
        eligible = not block.is_speech or intensity >= 3
        if used < cap and block.sfx is None and eligible:
            tag = suggest_tag(block.text, available_tags)
            if tag is not None:
                block = block.model_copy(update={"sfx": tag})
                used += 1
        tagged.append(block)
    return tagged
