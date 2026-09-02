"""Gabarito labeler — hotkey-driven labelling of pipeline output (ticket #3).

    python -m tools.label_benchmark --volume berserk_vol_01 --mode speakers
    python -m tools.label_benchmark --volume berserk_vol_01 --mode panels

Reads a pipeline workspace (``workspace/<volume>/``), shows each item in
a matplotlib window and writes the label to
``tests/benchmark/gabarito/<volume>.json`` on every key press, so a
labelling session can be interrupted and resumed at any time. Items are
keyed by the *true* PDF page (``page_008_panel00/b1``), taken from the
run's ``run_config.json``, so gabaritos from runs with different
``--start`` values line up. Nothing here is part of the runtime pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mangawhisperer.config import PROJECT_ROOT
from mangawhisperer.reporting import load_script

GABARITO_DIR = PROJECT_ROOT / "tests" / "benchmark" / "gabarito"
DEFAULT_CAST = ("Guts", "Griffith", "Casca", "Puck", "Judeau", "Corkus", "Zodd", "Criatura")
NARRATOR_LABEL = "Narrador"
JUNK_LABEL = "__lixo__"
MODES = ("speakers", "panels")
PANEL_VERDICTS = {"y": "ok", "u": "under", "v": "over", "n": "wrong"}
SKIP_KEY = "s"
QUIT_KEY = "q"
OTHER_KEY = "o"
JUNK_KEY = "x"
COUNT_KEY = "c"
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class LabelItem:
    """One thing to label: a bubble (speakers) or a page (panels)."""

    key: str
    image_path: Path
    prompt: str
    predicted: str
    text: str = ""


def label_for_key(key: str, cast: Sequence[str]) -> str | None:
    """Map a hotkey to a speaker label; ``None`` for control keys."""
    if key == "0":
        return NARRATOR_LABEL
    if key == JUNK_KEY:
        return JUNK_LABEL
    if len(key) == 1 and key.isdigit():
        index = int(key) - 1
        return cast[index] if index < len(cast) else None
    return None


def hotkey_legend(mode: str, cast: Sequence[str]) -> str:
    if mode == "speakers":
        keys = [f"{i + 1}={name}" for i, name in enumerate(cast[:9])]
        keys += [f"0={NARRATOR_LABEL}", f"{OTHER_KEY}=outro", f"{JUNK_KEY}=lixo"]
    else:
        keys = [f"{k}={v}" for k, v in PANEL_VERDICTS.items()] + [f"{COUNT_KEY}=contagem"]
    keys += [f"{SKIP_KEY}=pular", f"{QUIT_KEY}=sair"]
    return "  ".join(keys)


class Gabarito:
    """Resume-safe label store: one JSON per volume, saved atomically on every record."""

    def __init__(self, path: Path, volume: str) -> None:
        self.path = path
        self.volume = volume
        self.data: dict = {"volume": volume, "speakers": {}, "panels": {}}
        if path.is_file():
            stored = json.loads(path.read_text(encoding="utf-8"))
            for mode in MODES:
                self.data[mode] = dict(stored.get(mode, {}))

    def is_labeled(self, mode: str, item: LabelItem) -> bool:
        """Labeled *for this text*: a re-run that OCR'd the bubble
        differently re-queues it instead of trusting the old label."""
        entry = self.data[mode].get(item.key)
        if entry is None:
            return False
        return not (item.text and entry.get("text") and entry["text"] != item.text)

    def pending(self, mode: str, items: Sequence[LabelItem]) -> list[LabelItem]:
        return [item for item in items if not self.is_labeled(mode, item)]

    def count(self, mode: str) -> int:
        return len(self.data[mode])

    def record(self, mode: str, item: LabelItem, **fields) -> None:
        entry = {"predicted": item.predicted, **fields}
        if item.text:
            entry["text"] = item.text
        self.data[mode][item.key] = entry
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temp, self.path)  # never leave a half-written gabarito behind


def first_pdf_page(workspace: Path) -> int:
    """The ``--start`` the run used, from the extractor fingerprint in
    ``run_config.json`` (``pymupdf:dpi=200:first=31:max=2``); 1 if unknown."""
    config_path = workspace / "run_config.json"
    if not config_path.is_file():
        return 1
    try:
        fingerprint = json.loads(config_path.read_text(encoding="utf-8"))["script"]["page_extractor"]
    except (OSError, ValueError, KeyError, TypeError):
        return 1
    match = re.search(r"first=(\d+)", str(fingerprint))
    return int(match.group(1)) if match else 1


def page_stem(page_number: int, first_page: int) -> str:
    """Run-relative page number -> the rendered page file stem (``page_008``)."""
    return f"page_{first_page + page_number - 1:03d}"


def _page_file(workspace: Path, stem: str) -> Path | None:
    for suffix in _IMAGE_SUFFIXES:
        candidate = workspace / "pages" / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _script(workspace: Path):
    script = workspace / "script" / "panels.json"
    if not script.is_file():
        raise FileNotFoundError(f"Roteiro não encontrado: {script} — rode o pipeline primeiro.")
    return load_script(script)


def speaker_items(workspace: Path) -> list[LabelItem]:
    """One item per speech block of the volume's script checkpoint."""
    first = first_pdf_page(workspace)
    items: list[LabelItem] = []
    for panel in _script(workspace):
        image_path = panel.image_path
        if not image_path.is_file():  # workspace moved: resolve by name
            image_path = workspace / "panels" / image_path.name
        stem = page_stem(panel.page_number, first)
        for index, block in enumerate(panel.blocks):
            if not block.is_speech:
                continue
            items.append(LabelItem(
                key=f"{stem}_panel{panel.panel_index:02d}/b{index}",
                image_path=image_path,
                prompt=f"Quem fala: “{block.text}”",
                predicted=block.speaker_id,
                text=block.text,
            ))
    return items


def page_items(workspace: Path) -> list[LabelItem]:
    """One item per page of the script checkpoint, with the pipeline's
    panel count (the checkpoint, not the directory, is the truth —
    ``pages/`` and ``panels/`` accumulate files across runs)."""
    first = first_pdf_page(workspace)
    counts = Counter(panel.page_number for panel in _script(workspace))
    items: list[LabelItem] = []
    for page_number in sorted(counts):
        stem = page_stem(page_number, first)
        image_path = _page_file(workspace, stem)
        if image_path is None:
            print(f"  (pulando {stem}: imagem da página não encontrada em {workspace / 'pages'})")
            continue
        items.append(LabelItem(
            key=stem,
            image_path=image_path,
            prompt=f"Painéis detectados: {counts[page_number]}",
            predicted=str(counts[page_number]),
        ))
    if not items:
        raise FileNotFoundError(f"Nenhuma página renderizada em {workspace / 'pages'} — rode o pipeline primeiro.")
    return items


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    from PIL import Image  # noqa: PLC0415 — only needed for page files

    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def run_session(mode: str, items: Sequence[LabelItem], gabarito: Gabarito, cast: Sequence[str]) -> int:
    """Interactive matplotlib loop; returns how many labels were recorded."""
    import matplotlib.pyplot as plt  # noqa: PLC0415 — UI dependency only here

    state = {"index": 0, "recorded": 0}
    fig, ax = plt.subplots(figsize=(9, 10))
    fig.canvas.manager.set_window_title(f"MangaWhisperer · gabarito · {gabarito.volume} · {mode}")
    legend = hotkey_legend(mode, cast)

    def show() -> None:
        ax.clear()
        item = items[state["index"]]
        ax.imshow(load_image(item.image_path), cmap="gray")
        ax.set_axis_off()
        ax.set_title(
            f"[{state['index'] + 1}/{len(items)}] {item.prompt}\n"
            f"previsto: {item.predicted}\n{legend}",
            fontsize=9, loc="left",
        )
        fig.canvas.draw_idle()

    def advance() -> None:
        state["index"] += 1
        if state["index"] >= len(items):
            plt.close(fig)
        else:
            show()

    def on_key(event) -> None:
        key = event.key or ""
        if key == QUIT_KEY:
            plt.close(fig)
            return
        if key == SKIP_KEY:
            advance()
            return
        item = items[state["index"]]
        if mode == "speakers":
            label = label_for_key(key, cast)
            if key == OTHER_KEY:
                label = input("Nome do personagem: ").strip() or None
            if label is None:
                return
            gabarito.record(mode, item, label=label)
        else:
            if key == COUNT_KEY:
                raw = input("Contagem real de painéis: ").strip()
                if not raw.isdigit():
                    return
                true_count = int(raw)
                verdict = "ok" if true_count == int(item.predicted) else (
                    "under" if true_count > int(item.predicted) else "over")
                gabarito.record(mode, item, verdict=verdict, true_count=true_count)
            elif key in PANEL_VERDICTS:
                gabarito.record(mode, item, verdict=PANEL_VERDICTS[key])
            else:
                return
        state["recorded"] += 1
        print(f"  {item.key}: {gabarito.data[mode][item.key]}")
        advance()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show()
    plt.show()
    return state["recorded"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volume", required=True, help="Slug do workspace (ex.: berserk_vol_01).")
    parser.add_argument("--mode", choices=MODES, default="speakers")
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "workspace")
    parser.add_argument("--gabarito-dir", type=Path, default=GABARITO_DIR)
    parser.add_argument("--cast", default=",".join(DEFAULT_CAST),
                        help="Personagens para as teclas 1..9, separados por vírgula.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cast = tuple(name.strip() for name in args.cast.split(",") if name.strip())
    workspace = args.workspace / args.volume
    items = speaker_items(workspace) if args.mode == "speakers" else page_items(workspace)
    gabarito = Gabarito(args.gabarito_dir / f"{args.volume}.json", args.volume)
    pending = gabarito.pending(args.mode, items)
    print(f"{len(items)} itens · {len(items) - len(pending)} já rotulados · {len(pending)} pendentes")
    print(f"Gabarito: {gabarito.path}")
    if not pending:
        print("Nada a rotular.")
        return 0
    recorded = run_session(args.mode, pending, gabarito, cast)
    print(f"Sessão encerrada: {recorded} rótulos novos, {gabarito.count(args.mode)} no total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
