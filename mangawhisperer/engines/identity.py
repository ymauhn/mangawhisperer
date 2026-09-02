"""Zero-token character identity (ticket #22): who is in the panel.

Pipeline from research R5 (``docs/research/identidade-personagens.md``):

    page image ─► CharacterDetector (manga109_yolo, ONNX, ~40 ms on CPU)
               ─► body/face crops ─► CropEmbedder (Magiv2 crop embedder,
                  or DINOv2 as the licence-clean fallback)
               ─► Gallery.match (cosine against 3–10 named exemplars,
                  with an "unknown" threshold and a margin)
               ─► advisory hint text for the scriptwriter (never imposed).

Everything heavy loads lazily and is injectable, so the geometry, the
YOLO decoding and the gallery logic are tested without any model. The
Magiv2 weights are loaded straight into a stock ``ViTMAEModel`` — the
repository's remote code was read and does exactly that (grayscale →
RGB, 224 px, ImageNet mean/std, CLS token), so ``trust_remote_code`` is
not needed. Magi weights are non-commercial (ADR-0002).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

DETECTOR_REPO = "deepghs/manga109_yolo"
DETECTOR_VARIANT = "v2023.12.07_s_yv11"
DETECTOR_CLASSES: tuple[str, ...] = ("body", "face", "frame", "text")
DETECTOR_INPUT_SIZE = 1024  # multiple of 32; the export accepts any size
DEFAULT_THRESHOLD = 0.383  # the repo's F1-optimal threshold for this variant
MAGIV2_REPO = "ragavsachdeva/magiv2-crop-embedder"
DINOV2_REPO = "facebook/dinov2-base"
UNKNOWN = "Desconhecido"

PixelBox = tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive), page pixels


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: PixelBox

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "confidence": round(self.confidence, 4), "box": list(self.box)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Detection":
        x0, y0, x1, y1 = (int(v) for v in data["box"])
        return cls(label=str(data["label"]), confidence=float(data["confidence"]), box=(x0, y0, x1, y1))


# ── YOLO pre/post-processing (pure numpy + cv2) ─────────────────────


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping the aspect ratio onto a ``size``×``size`` gray canvas.
    Returns ``(canvas, scale, left, top)`` so boxes can be mapped back."""
    import cv2  # noqa: PLC0415

    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_h, new_w = max(1, round(height * scale)), max(1, round(width * scale))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - new_h) // 2, (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return canvas, scale, left, top


def decode_yolo_output(
    raw: np.ndarray,
    labels: Sequence[str],
    threshold: float,
    scale: float,
    offset: tuple[int, int],
    image_size: tuple[int, int],
    iou: float = 0.7,
) -> list[Detection]:
    """Turn a YOLOv8/11 export's ``(1, 4+nc, N)`` output into page-pixel
    detections: confidence filter, per-class NMS, un-letterbox, clip."""
    import cv2  # noqa: PLC0415

    pred = np.asarray(raw)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] == 4 + len(labels):
        pred = pred.T  # (N, 4+nc)
    boxes_cxcywh, class_scores = pred[:, :4], pred[:, 4:4 + len(labels)]
    confidences = class_scores.max(axis=1)
    classes = class_scores.argmax(axis=1)
    keep = confidences >= threshold
    if not keep.any():
        return []
    boxes_cxcywh, confidences, classes = boxes_cxcywh[keep], confidences[keep], classes[keep]
    x0 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y0 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    xywh = np.stack([x0, y0, boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]], axis=1)

    width, height = image_size
    left, top = offset
    detections: list[Detection] = []
    for class_index in np.unique(classes):
        members = np.flatnonzero(classes == class_index)
        kept = cv2.dnn.NMSBoxes(xywh[members].tolist(), confidences[members].tolist(), float(threshold), float(iou))
        for k in np.asarray(kept).reshape(-1):
            i = members[int(k)]
            bx0 = int(round((xywh[i, 0] - left) / scale))
            by0 = int(round((xywh[i, 1] - top) / scale))
            bx1 = int(round((xywh[i, 0] + xywh[i, 2] - left) / scale))
            by1 = int(round((xywh[i, 1] + xywh[i, 3] - top) / scale))
            bx0, by0 = max(0, min(bx0, width)), max(0, min(by0, height))
            bx1, by1 = max(0, min(bx1, width)), max(0, min(by1, height))
            if bx1 > bx0 and by1 > by0:
                detections.append(Detection(str(labels[class_index]), float(confidences[i]), (bx0, by0, bx1, by1)))
    detections.sort(key=lambda d: (d.box[1], -d.box[0]))  # top-to-bottom, right-to-left
    return detections


class CharacterDetector:
    """manga109_yolo (body/face/frame/text on manga pages) through ONNX Runtime."""

    def __init__(
        self,
        variant: str = DETECTOR_VARIANT,
        repo: str = DETECTOR_REPO,
        input_size: int = DETECTOR_INPUT_SIZE,
        threshold: float | None = None,
        iou: float = 0.7,
        session: Any = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            variant: Model folder in the repo (size ``n``/``s``/``m``/``l``/``x``).
            input_size: Square inference size; the page is letterboxed to it.
                640 is the training size, 1024 keeps small faces on a
                2300 px page (about 4× the compute, still ~0.2 s on CPU).
            threshold: Confidence floor; ``None`` reads the repo's F1-optimal one.
            session: Injectable ONNX session (``run``/``get_inputs``) for tests.
            labels: Class names, injectable with ``session``.
        """
        self._variant = variant
        self._repo = repo
        self._input_size = input_size
        self._threshold = threshold
        self._iou = iou
        self._session = session
        self._labels = tuple(labels) if labels else None

    @property
    def fingerprint(self) -> str:
        return f"manga109-yolo:{self._variant}:size={self._input_size}:thr={self.threshold:.3f}:iou={self._iou}"

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            self._threshold = self._load_threshold()
        return self._threshold

    @property
    def labels(self) -> tuple[str, ...]:
        if self._labels is None:
            self._labels = tuple(json.loads(Path(self._download("labels.json")).read_text(encoding="utf-8")))
        return self._labels

    def detect(self, page_rgb: np.ndarray) -> list[Detection]:
        """All boxes on a page image (H, W, 3 uint8), in reading order."""
        canvas, scale, left, top = letterbox(page_rgb, self._input_size)
        tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        session = self._get_session()
        raw = session.run(None, {session.get_inputs()[0].name: tensor})[0]
        height, width = page_rgb.shape[:2]
        return decode_yolo_output(raw, self.labels, self.threshold, scale, (left, top), (width, height), self._iou)

    def release(self) -> None:
        self._session = None

    def _download(self, filename: str) -> str:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        return hf_hub_download(self._repo, f"{self._variant}/{filename}")

    def _load_threshold(self) -> float:
        if self._session is not None and self._labels is not None:
            return DEFAULT_THRESHOLD  # injected session: no repo access
        try:
            return float(json.loads(Path(self._download("threshold.json")).read_text(encoding="utf-8"))["threshold"])
        except Exception:  # missing file: the variant's documented value
            return DEFAULT_THRESHOLD

    def _get_session(self) -> Any:
        if self._session is None:
            import onnxruntime as ort  # noqa: PLC0415

            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
            self._session = ort.InferenceSession(self._download("model.onnx"), providers=providers)
            logger.info("Detector %s carregado (%s)", self._variant, ", ".join(providers))
        return self._session


# ── Crop embedders ──────────────────────────────────────────────────


class CropEmbedder(Protocol):
    name: str

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray: ...

    def release(self) -> None: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_crop(crop: np.ndarray, size: int = 224, grayscale: bool = True) -> np.ndarray:
    """Magi's recipe: grayscale → RGB, resize to 224², ImageNet mean/std, CHW."""
    import cv2  # noqa: PLC0415

    image = np.asarray(crop)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if grayscale:
        gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        image = np.stack([gray] * 3, axis=-1)
    image = cv2.resize(image[:, :, :3], (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
    return image.transpose(2, 0, 1)


class Magiv2CropEmbedder:
    """``ragavsachdeva/magiv2-crop-embedder`` (ViT-B, SupCon on manga
    crops) loaded into a stock ``ViTMAEModel`` with ``mask_ratio=0``.
    With no masking the CLS token is deterministic: patch tokens carry
    their position embeddings before MAE's shuffle, and attention is
    permutation-invariant."""

    name = "magiv2-crop-embedder"

    def __init__(self, device: str | None = None, batch_size: int = 64, model: Any = None, repo: str = MAGIV2_REPO) -> None:
        self._device = device
        self._batch_size = batch_size
        self._model = model
        self._repo = repo

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 768), dtype=np.float32)
        import torch  # noqa: PLC0415

        model = self._get_model()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(crops), self._batch_size):
                batch = np.stack([preprocess_crop(c) for c in crops[start:start + self._batch_size]])
                pixels = torch.from_numpy(batch).to(model.device, dtype=model.dtype)
                hidden = model(pixel_values=pixels).last_hidden_state[:, 0]
                outputs.append(hidden.float().cpu().numpy())
        return l2_normalize(np.concatenate(outputs))

    def release(self) -> None:
        self._model = None
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _get_model(self) -> Any:
        if self._model is None:
            import torch  # noqa: PLC0415
            from huggingface_hub import hf_hub_download  # noqa: PLC0415
            from safetensors.torch import load_file  # noqa: PLC0415
            from transformers import ViTMAEConfig, ViTMAEModel  # noqa: PLC0415

            config_dict = json.loads(Path(hf_hub_download(self._repo, "config.json")).read_text(encoding="utf-8"))
            vit = config_dict["crop_embedding_model_config"]
            known = ViTMAEConfig().to_dict()
            config = ViTMAEConfig(**{k: v for k, v in vit.items() if k in known and k != "model_type"})
            config.mask_ratio = 0.0
            model = ViTMAEModel(config)
            state = load_file(hf_hub_download(self._repo, "model.safetensors"))
            prefix = "crop_embedding_model."
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                logger.warning("Magiv2 embedder: %d pesos faltando, %d inesperados", len(missing), len(unexpected))
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = model.eval().to(device)
            logger.info("Embedder %s carregado em %s", self.name, device)
        return self._model


class Dinov2CropEmbedder:
    """``facebook/dinov2-base`` (Apache-2.0): the licence-clean fallback,
    weaker on manga (CoMix AMI 0.29 vs Magi's ~0.65 in-domain)."""

    name = "dinov2-base"

    def __init__(self, device: str | None = None, batch_size: int = 64, model: Any = None, repo: str = DINOV2_REPO) -> None:
        self._device = device
        self._batch_size = batch_size
        self._model = model
        self._repo = repo

    def embed(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 768), dtype=np.float32)
        import torch  # noqa: PLC0415

        model = self._get_model()
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(crops), self._batch_size):
                batch = np.stack([preprocess_crop(c, grayscale=False) for c in crops[start:start + self._batch_size]])
                pixels = torch.from_numpy(batch).to(model.device, dtype=model.dtype)
                outputs.append(model(pixel_values=pixels).pooler_output.float().cpu().numpy())
        return l2_normalize(np.concatenate(outputs))

    def release(self) -> None:
        self._model = None

    def _get_model(self) -> Any:
        if self._model is None:
            import torch  # noqa: PLC0415
            from transformers import AutoModel  # noqa: PLC0415

            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = AutoModel.from_pretrained(self._repo).eval().to(device)
            logger.info("Embedder %s carregado em %s", self.name, device)
        return self._model


def create_embedder(name: str, **kwargs: Any) -> CropEmbedder:
    if name in ("magiv2", Magiv2CropEmbedder.name):
        return Magiv2CropEmbedder(**kwargs)
    if name in ("dinov2", Dinov2CropEmbedder.name):
        return Dinov2CropEmbedder(**kwargs)
    raise ValueError(f"Embedder desconhecido {name!r}; use 'magiv2' ou 'dinov2'.")


# ── Gallery matching ────────────────────────────────────────────────


@dataclass(frozen=True)
class Match:
    """Outcome of comparing one crop against the gallery."""

    name: str | None  # None = unknown (below the acceptance rules)
    score: float
    best: str | None
    runner_up: str | None
    runner_up_score: float

    @property
    def margin(self) -> float:
        return self.score - self.runner_up_score


class Gallery:
    """Named exemplar embeddings with cosine matching and an unknown gate.

    ``strategy``: ``nearest`` = best single exemplar per name (k-NN 1),
    ``prototype`` = cosine to the normalized mean of the exemplars,
    ``mixed`` = average of both. A crop is named only when its best
    score reaches ``accept`` and beats the runner-up name by ``margin``.
    """

    def __init__(self, accept: float = 0.80, margin: float = 0.05, strategy: str = "mixed") -> None:
        # Defaults measured on Berserk vol. 1 (99 labelled crops, leave-one-out, Magiv2):
        # mixed @0.80 names the principal cast 85.7% of the time, rejects 89% of extras.
        if strategy not in ("nearest", "prototype", "mixed"):
            raise ValueError(f"Estratégia desconhecida {strategy!r}")
        self.accept = accept
        self.margin = margin
        self.strategy = strategy
        self._exemplars: dict[str, list[np.ndarray]] = {}

    def add(self, name: str, embedding: np.ndarray) -> None:
        self._exemplars.setdefault(name, []).append(l2_normalize(np.asarray(embedding, dtype=np.float32)))

    @property
    def names(self) -> list[str]:
        return sorted(self._exemplars)

    def count(self, name: str) -> int:
        return len(self._exemplars.get(name, []))

    def scores(self, embedding: np.ndarray) -> dict[str, float]:
        query = l2_normalize(np.asarray(embedding, dtype=np.float32))
        result: dict[str, float] = {}
        for name, vectors in self._exemplars.items():
            matrix = np.stack(vectors)
            nearest = float((matrix @ query).max())
            prototype = float(l2_normalize(matrix.mean(axis=0)) @ query)
            result[name] = {"nearest": nearest, "prototype": prototype, "mixed": (nearest + prototype) / 2}[self.strategy]
        return result

    def match(self, embedding: np.ndarray) -> Match:
        scores = self.scores(embedding)
        if not scores:
            return Match(None, 0.0, None, None, 0.0)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best, best_score = ranked[0]
        runner_up, runner_up_score = ranked[1] if len(ranked) > 1 else (None, 0.0)
        accepted = best_score >= self.accept and (best_score - runner_up_score) >= self.margin
        return Match(best if accepted else None, best_score, best, runner_up, runner_up_score)

    def to_json(self) -> dict[str, Any]:
        return {
            "accept": self.accept, "margin": self.margin, "strategy": self.strategy,
            "exemplars": [{"name": n, "embedding": v.tolist()} for n, vs in self._exemplars.items() for v in vs],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Gallery":
        gallery = cls(accept=data.get("accept", 0.80), margin=data.get("margin", 0.05), strategy=data.get("strategy", "mixed"))
        for item in data.get("exemplars", []):
            gallery.add(item["name"], np.asarray(item["embedding"], dtype=np.float32))
        return gallery


def leave_one_out(
    labelled: Sequence[tuple[str, np.ndarray]],
    accept: float = 0.80,
    margin: float = 0.05,
    strategy: str = "mixed",
    unknown_label: str = UNKNOWN,
) -> list[tuple[str, str | None, float]]:
    """Name every labelled crop from a gallery built from all the *other*
    crops. Crops labelled ``unknown_label`` never enter the gallery: the
    right answer for them is ``None``. Returns ``(truth, predicted, score)``."""
    results: list[tuple[str, str | None, float]] = []
    for index, (truth, embedding) in enumerate(labelled):
        gallery = Gallery(accept=accept, margin=margin, strategy=strategy)
        for other, (name, vector) in enumerate(labelled):
            if other != index and name != unknown_label:
                gallery.add(name, vector)
        match = gallery.match(embedding)
        results.append((truth, match.name, match.score))
    return results


def summarize_naming(results: Sequence[tuple[str, str | None, float]], unknown_label: str = UNKNOWN) -> dict[str, Any]:
    """Per-name accuracy, unknown handling and confusions from leave-one-out results."""
    per_name: dict[str, dict[str, int]] = {}
    confusions: dict[str, int] = {}
    for truth, predicted, _score in results:
        bucket = per_name.setdefault(truth, {"total": 0, "correct": 0, "unknown": 0})
        bucket["total"] += 1
        expected = None if truth == unknown_label else truth
        if predicted == expected:
            bucket["correct"] += 1
        elif predicted is None:
            bucket["unknown"] += 1
        else:
            key = f"{truth} -> {predicted}"
            confusions[key] = confusions.get(key, 0) + 1
    known = [r for r in results if r[0] != unknown_label]
    accuracy = sum(1 for t, p, _ in known if p == t) / len(known) if known else 0.0
    unknown_rate = sum(1 for _, p, _ in known if p is None) / len(known) if known else 0.0
    extras = [r for r in results if r[0] == unknown_label]
    rejected = sum(1 for _, p, _ in extras if p is None) / len(extras) if extras else None
    return {
        "known_crops": len(known), "accuracy": accuracy, "unknown_rate": unknown_rate,
        "extras": len(extras), "extras_rejected": rejected,
        "per_name": per_name, "confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1])),
    }


# ── Geometry: boxes ↔ panels ↔ bubbles ──────────────────────────────


def box_area(box: PixelBox) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def containment(inner: PixelBox, outer: PixelBox) -> float:
    """Fraction of ``inner`` that lies inside ``outer``."""
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = box_area(inner)
    return inter / area if area else 0.0


def assign_to_panels(detections: Sequence[Detection], panels: Sequence[PixelBox], min_inside: float = 0.5) -> dict[int, list[Detection]]:
    """Give each detection to the panel holding most of it (at least
    ``min_inside``); detections in no panel are dropped."""
    assigned: dict[int, list[Detection]] = {}
    for detection in detections:
        best, best_fraction = None, 0.0
        for index, panel in enumerate(panels):
            fraction = containment(detection.box, panel)
            if fraction > best_fraction:
                best, best_fraction = index, fraction
        if best is not None and best_fraction >= min_inside:
            assigned.setdefault(best, []).append(detection)
    return assigned


def nearest_character(bubble: PixelBox, characters: Sequence[PixelBox]) -> int | None:
    """Index of the character box closest to the bubble (centre-to-centre,
    ties to the larger box) — the geometric speaker guess."""
    if not characters:
        return None
    bx, by = (bubble[0] + bubble[2]) / 2, (bubble[1] + bubble[3]) / 2

    def key(index: int) -> tuple[float, int]:
        box = characters[index]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        return ((cx - bx) ** 2 + (cy - by) ** 2, -box_area(box))

    return min(range(len(characters)), key=key)


def format_hint_block(matches: Sequence[Match], numbered: bool = True) -> str:
    """Advisory PT-BR block for the scriptwriter's user message (research §E)."""
    if not matches:
        return ""
    parts = []
    for index, match in enumerate(matches, start=1):
        label = f"{match.name} ({match.score:.2f})" if match.name else "desconhecido"
        parts.append(f"[{index}] {label}" if numbered else label)
    lines = ["Personagens detectados neste painel (dica automática, pode estar errada; confie na imagem se discordar): "
             + ", ".join(parts) + "."]
    if numbered and len(matches) > 1:
        lines.append("Os números correspondem às marcas sobre a imagem.")
    return "\n".join(lines)


def crop_with_margin(page: np.ndarray, box: PixelBox, margin: float = 0.0) -> np.ndarray:
    """Cut ``box`` out of the page, optionally padded by ``margin`` × its size."""
    height, width = page.shape[:2]
    x0, y0, x1, y1 = box
    pad_x, pad_y = int((x1 - x0) * margin), int((y1 - y0) * margin)
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(width, x1 + pad_x), min(height, y1 + pad_y)
    return page[y0:y1, x0:x1]
