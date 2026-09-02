"""Measure character naming from the human gabarito (ticket #22).

    python -m tools.eval_identity --volume berserk_vol_01 --workspace workspace/bench --embedder magiv2

Takes the crops the author named in ``label_benchmark --mode characters``
(labels + boxes in ``tests/benchmark/gabarito/<volume>.json``), embeds
them, and names each one from a gallery built from all the *other*
crops (leave-one-out) — the honest estimate for "3–10 exemplars per
cast member". Reports accuracy per character, the unknown rate, how
many extras were correctly rejected, the confusions, and a sweep of
acceptance thresholds. Writes ``identity/report_<embedder>.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mangawhisperer.config import PROJECT_ROOT
from mangawhisperer.engines.identity import (
    UNKNOWN,
    CropEmbedder,
    create_embedder,
    crop_with_margin,
    leave_one_out,
    summarize_naming,
)

GABARITO_DIR = PROJECT_ROOT / "tests" / "benchmark" / "gabarito"


def labelled_crops(gabarito: dict, min_per_name: int = 2) -> list[dict[str, Any]]:
    """Character labels usable for evaluation: names with at least
    ``min_per_name`` crops, plus every extra (``Desconhecido``)."""
    entries = [{"key": k, **v} for k, v in gabarito.get("characters", {}).items() if v.get("label") and v.get("box")]
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["label"]] = counts.get(entry["label"], 0) + 1
    return [e for e in entries if e["label"] == UNKNOWN or counts[e["label"]] >= min_per_name]


def sweep(labelled: Sequence[tuple[str, np.ndarray]], accepts: Sequence[float], margin: float, strategy: str) -> list[dict[str, Any]]:
    rows = []
    for accept in accepts:
        summary = summarize_naming(leave_one_out(labelled, accept=accept, margin=margin, strategy=strategy))
        rows.append({"accept": accept, "accuracy": summary["accuracy"], "unknown_rate": summary["unknown_rate"],
                     "extras_rejected": summary["extras_rejected"]})
    return rows


def report_markdown(volume: str, embedder: str, summary: dict[str, Any], rows: Sequence[dict[str, Any]], strategy: str, margin: float) -> str:
    lines = [f"# Identidade de personagens — {volume} — {embedder}", "",
             f"Estratégia `{strategy}`, margem {margin:.2f}, leave-one-out sobre {summary['known_crops']} recortes "
             f"nomeados + {summary['extras']} figurantes.", "",
             f"**Acurácia de nomeação: {summary['accuracy']:.1%}** · desconhecido: {summary['unknown_rate']:.1%}"
             + (f" · figurantes rejeitados: {summary['extras_rejected']:.1%}" if summary["extras_rejected"] is not None else ""),
             "", "| Personagem | recortes | acertos | desconhecido |", "|---|---|---|---|"]
    for name, bucket in sorted(summary["per_name"].items()):
        lines.append(f"| {name} | {bucket['total']} | {bucket['correct']} | {bucket['unknown']} |")
    if summary["confusions"]:
        lines += ["", "Confusões: " + ", ".join(f"{k} ({v})" for k, v in summary["confusions"].items())]
    lines += ["", "| aceite | acurácia | desconhecido | figurantes rejeitados |", "|---|---|---|---|"]
    for row in rows:
        rejected = f"{row['extras_rejected']:.0%}" if row["extras_rejected"] is not None else "—"
        lines.append(f"| {row['accept']:.2f} | {row['accuracy']:.1%} | {row['unknown_rate']:.1%} | {rejected} |")
    return "\n".join(lines) + "\n"


def evaluate(workspace: Path, volume: str, embedder: CropEmbedder, accept: float, margin: float,
             strategy: str, min_per_name: int, gabarito_dir: Path = GABARITO_DIR) -> dict[str, Any]:
    from PIL import Image  # noqa: PLC0415

    gabarito = json.loads((gabarito_dir / f"{volume}.json").read_text(encoding="utf-8"))
    entries = labelled_crops(gabarito, min_per_name)
    if len(entries) < 2:
        raise SystemExit("Poucos recortes rotulados: rode `label_benchmark --mode characters` primeiro.")
    pages: dict[str, np.ndarray] = {}
    crops = []
    for entry in entries:
        stem = entry["key"].split("/")[0]
        if stem not in pages:
            for suffix in (".png", ".jpg", ".jpeg"):
                candidate = workspace / "pages" / f"{stem}{suffix}"
                if candidate.is_file():
                    pages[stem] = np.asarray(Image.open(candidate).convert("RGB"))
                    break
        crops.append(crop_with_margin(pages[stem], tuple(int(v) for v in entry["box"])))
    embeddings = embedder.embed(crops)
    labelled = [(e["label"], v) for e, v in zip(entries, embeddings)]
    summary = summarize_naming(leave_one_out(labelled, accept=accept, margin=margin, strategy=strategy))
    rows = sweep(labelled, [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], margin, strategy)
    report = report_markdown(volume, embedder.name, summary, rows, strategy, margin)
    out = workspace / "identity" / f"report_{embedder.name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"-> {out}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "workspace")
    parser.add_argument("--embedder", choices=("magiv2", "dinov2"), default="magiv2")
    parser.add_argument("--accept", type=float, default=0.75)
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--strategy", choices=("nearest", "prototype", "mixed"), default="nearest")
    parser.add_argument("--min-per-name", type=int, default=2)
    parser.add_argument("--device", default=None, help="cuda|cpu (padrão: cuda se disponível)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    embedder = create_embedder(args.embedder, device=args.device)
    try:
        evaluate(args.workspace / args.volume, args.volume, embedder, args.accept, args.margin,
                 args.strategy, args.min_per_name)
    finally:
        embedder.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
