"""Run the manga109 character detector over a workspace's pages (ticket #22).

    python -m tools.detect_characters --volume berserk_vol_01 --workspace workspace/bench

Writes ``<workspace>/<volume>/identity/detections.json`` (every body/face/
frame/text box per page, in page pixels) and, with ``--overlays``, a
preview PNG per page under ``identity/overlays/`` so the boxes can be
eyeballed. The labelling tool's ``characters`` mode reads the JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from mangawhisperer.config import PROJECT_ROOT
from mangawhisperer.engines.identity import CharacterDetector, Detection

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_COLOURS = {"body": (220, 30, 30), "face": (30, 90, 220), "frame": (30, 160, 60), "text": (200, 160, 0)}


def page_files(workspace: Path) -> list[Path]:
    return sorted(p for p in (workspace / "pages").glob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)


def detections_to_json(detector_fingerprint: str, pages: dict[str, tuple[int, int, list[Detection]]]) -> dict:
    return {
        "detector": detector_fingerprint,
        "pages": {
            stem: {"width": width, "height": height, "detections": [d.to_dict() for d in dets]}
            for stem, (width, height, dets) in pages.items()
        },
    }


def draw_overlay(page: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Boxes coloured by class; bodies numbered in reading order."""
    import cv2  # noqa: PLC0415

    canvas = np.ascontiguousarray(page.copy())
    thickness = max(2, page.shape[1] // 600)
    body_index = 0
    for detection in detections:
        x0, y0, x1, y1 = detection.box
        colour = _COLOURS.get(detection.label, (0, 0, 0))
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thickness)
        if detection.label == "body":
            body_index += 1
            cv2.putText(canvas, str(body_index), (x0 + 4, y0 + 28 + thickness * 4), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, colour, thickness)
    return canvas


def run(workspace: Path, input_size: int, overlays: bool, detector: CharacterDetector | None = None) -> Path:
    from PIL import Image  # noqa: PLC0415

    files = page_files(workspace)
    if not files:
        raise FileNotFoundError(f"Nenhuma página em {workspace / 'pages'} — rode o pipeline primeiro.")
    detector = detector or CharacterDetector(input_size=input_size)
    out_dir = workspace / "identity"
    out_dir.mkdir(parents=True, exist_ok=True)
    pages: dict[str, tuple[int, int, list[Detection]]] = {}
    started = time.monotonic()
    for page_path in files:
        page = np.asarray(Image.open(page_path).convert("RGB"))
        detections = detector.detect(page)
        pages[page_path.stem] = (page.shape[1], page.shape[0], detections)
        counts = {label: sum(1 for d in detections if d.label == label) for label in ("body", "face", "frame", "text")}
        print(f"  {page_path.stem}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
        if overlays:
            (out_dir / "overlays").mkdir(exist_ok=True)
            Image.fromarray(draw_overlay(page, detections)).save(out_dir / "overlays" / f"{page_path.stem}.png")
    output = out_dir / "detections.json"
    output.write_text(json.dumps(detections_to_json(detector.fingerprint, pages), indent=1), encoding="utf-8")
    print(f"{len(files)} páginas em {time.monotonic() - started:.1f} s -> {output}")
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volume", required=True)
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT / "workspace")
    parser.add_argument("--input-size", type=int, default=1024, help="Lado do letterbox (múltiplo de 32).")
    parser.add_argument("--overlays", action="store_true", help="Salva PNGs com as caixas para conferência.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args.workspace / args.volume, args.input_size, args.overlays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
