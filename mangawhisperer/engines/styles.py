"""Narration style presets: one knob that tunes script AND delivery.

A style bundles the levers that already exist across the pipeline —
a scriptwriter prompt addendum, XTTS pacing/expressiveness parameters,
the silence gap between blocks and a suggested BGM — so "Sombrio" is a
single choice instead of five flags.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class NarrationStyle:
    """A cohesive narration direction applied across stages."""

    name: str
    label: str
    prompt_addendum: str
    tts_speed: float = 1.0
    synthesis_kwargs: dict[str, float] = field(default_factory=dict)
    gap_ms: int = 350
    suggested_bgm: str | None = None


STYLES: dict[str, NarrationStyle] = {
    "neutro": NarrationStyle(
        name="neutro",
        label="Pragmático / Neutro",
        prompt_addendum="",
    ),
    "sobrio": NarrationStyle(
        name="sobrio",
        label="Sóbrio / Audiodescrição",
        prompt_addendum="""

Narration style directive (SÓBRIO): audio description at its most restrained.
At most one action block per panel, twelve words or fewer, naming only what
is drawn: who, where, the action. No mood adjectives, no sound words, no
interpretation. Skip the action block entirely when the dialogue already
makes the action clear.""",
        tts_speed=1.0,
        gap_ms=450,
    ),
    "sombrio": NarrationStyle(
        name="sombrio",
        label="Sombrio / Dark",
        prompt_addendum="""

Narration style directive (SOMBRIO): action descriptions must dwell on
atmosphere, dread and suspense — shadows, sounds, what lurks unseen. Use
weighty, deliberate sentences with commas marking slow, ominous pauses.
Dialogue keeps its text, but descriptions favor tension over motion.""",
        tts_speed=0.92,
        synthesis_kwargs={"temperature": 0.7},
        gap_ms=550,
        suggested_bgm="ambiente_sombrio",
    ),
    "epico": NarrationStyle(
        name="epico",
        label="Épico / Alta ênfase",
        prompt_addendum="""

Narration style directive (ÉPICO): action descriptions are dynamic and
charged — short, punchy sentences, strong verbs, exclamation where the art
earns it. Mark emotional peaks explicitly and let momentum carry from one
block into the next.""",
        tts_speed=1.06,
        synthesis_kwargs={"temperature": 0.8, "repetition_penalty": 4.0},
        gap_ms=250,
        suggested_bgm="batalha",
    ),
}


MAX_EXAMPLE_CHARS = 6000


def style_examples_addendum(excerpts: str, max_chars: int = MAX_EXAMPLE_CHARS) -> str:
    """Wrap excerpts of a human narrator (e.g. ``style_excerpts.md`` from
    ``tools/transcribe_reference.py``) as a tone reference for the
    scriptwriter: imitate rhythm and focus, never copy sentences."""
    text = excerpts.strip()
    if not text:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n[...]"
    return (
        "\n\nTone reference — excerpts from a human narrator of this manga for blind "
        "listeners. Imitate the rhythm, the plain vocabulary and the focus on what is "
        "drawn; do NOT copy or paraphrase these sentences into the script:\n" + text
    )


def get_style(name: str) -> NarrationStyle:
    """Look up a preset by name (case-insensitive)."""
    try:
        return STYLES[name.lower().strip()]
    except KeyError as exc:
        raise ValueError(f"Estilo desconhecido {name!r}; opções: {', '.join(STYLES)}") from exc
