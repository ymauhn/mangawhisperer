"""Post-run summaries of a volume's script — shared by CLI and UI."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import TypeAdapter

from mangawhisperer.models import PanelData

_PANELS = TypeAdapter(list[PanelData])


@dataclass
class ScriptSummary:
    panel_count: int
    block_count: int
    blocks_per_speaker: dict[str, int] = field(default_factory=dict)
    sfx_usage: dict[str, int] = field(default_factory=dict)

    def format(self, sfx_available: bool) -> str:
        lines = [f"{self.panel_count} painéis, {self.block_count} blocos", "Blocos por personagem:"]
        lines += [f"  {speaker:>14}: {count}" for speaker, count in self.blocks_per_speaker.items()]
        if self.sfx_usage:
            lines.append("Efeitos sonoros no roteiro: "
                         + ", ".join(f"{tag}×{n}" for tag, n in self.sfx_usage.items()))
        elif sfx_available:
            lines.append("Efeitos sonoros no roteiro: nenhum (nem o roteirista nem o tagger "
                         "automático encontraram cena compatível)")
        return "\n".join(lines)


def load_script(path: Path) -> list[PanelData]:
    return _PANELS.validate_json(path.read_bytes())


def summarize_script(panels: list[PanelData]) -> ScriptSummary:
    blocks = [b for p in panels for b in p.blocks]
    speakers = collections.Counter(b.speaker_id for b in blocks)
    sfx = collections.Counter(b.sfx for b in blocks if b.sfx)
    return ScriptSummary(
        panel_count=len(panels),
        block_count=len(blocks),
        blocks_per_speaker=dict(speakers.most_common()),
        sfx_usage=dict(sfx.most_common()),
    )


def script_markdown(panels: list[PanelData]) -> str:
    """Roteiro renderizado para a UI (um bloco por linha, com ícones)."""
    lines = []
    for panel in panels:
        for block in panel.blocks:
            icon = "🗣️" if block.is_speech else "🎬"
            sfx_note = f" 🔊`{block.sfx}`" if block.sfx else ""
            lines.append(f"- {icon} **{block.speaker_id}**{sfx_note}: {block.text}")
    return "\n".join(lines) or "*Nenhum texto encontrado nas páginas selecionadas.*"
