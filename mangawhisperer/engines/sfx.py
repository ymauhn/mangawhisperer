"""Sound-effect (sonoplastia) library for scene ambience.

Two complementary layers:

* **Folder convention** (always on): drop audio files into
  ``assets/sfx/`` — the file stem IS the tag (``espada.wav`` -> tag
  ``espada``).
* **Dictionary** (optional): ``sfx_dictionary.json`` in the same folder
  maps a tag to MULTIPLE variant files with per-variant gain and
  metadata. Variants are picked deterministically from a seed (the
  narration block's text), so the same scene always gets the same
  sound — resume-safe variety.

Files are normalized lazily to the pipeline format (24 kHz mono 16-bit
WAV) into ``assets/sfx/_normalized/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import wave
from pathlib import Path

import numpy as np

from mangawhisperer.constants import AUDIO_SAMPLE_RATE as TARGET_RATE

logger = logging.getLogger(__name__)
_SUPPORTED = (".wav", ".mp3", ".ogg", ".flac")
_GAIN = 0.75  # effects slightly under narration level

TAG_KEYWORDS: dict[str, tuple[str, ...]] = {
    # Ordered by dramatic priority — first match wins.
    "espada": ("espada", "lâmina", "lamina", "aço", "aco", "corta", "corte",
               "desembainha", "crava", "golpe"),
    "explosao": ("explosão", "explosao", "explode", "estrondo", "destroça",
                 "destroca", "canhão", "canhao"),
    "monstro": ("monstro", "criatura", "besta", "ruge", "rugido", "demônio",
                "demonio", "apóstolo", "apostolo", "mandíbulas", "mandibulas",
                "garras", "tentáculos", "tentaculos"),
    "soco": ("soco", "punho", "esmurra", "golpeia", "pancada", "impacto", "atinge"),
    "fogo": ("fogo", "chamas", "queima", "arde", "fogueira", "incêndio", "incendio"),
    "grito": ("grita", "grito", "berra", "urra", "uivo"),
    "trovao": ("trovão", "trovao", "relâmpago", "relampago", "tempestade"),
    "vento": ("vento", "rajada", "ventania", "sopra"),
    "passos": ("passos", "caminha", "aproxima"),
    "porta": ("porta", "portão", "portao"),
}
"""PT-BR keywords that map an action description to an effect tag."""


def suggest_tag(text: str, available_tags: set[str]) -> str | None:
    """Deterministic fallback tagger: match an action description to the
    first available tag whose keywords appear in the text.

    Used when the VLM scriptwriter did not request an effect — it
    guarantees that obvious action beats ("uma explosão devasta o
    campo") still trigger sonoplastia.
    """
    haystack = text.lower()
    for tag, keywords in TAG_KEYWORDS.items():
        if tag in available_tags and any(keyword in haystack for keyword in keywords):
            return tag
    return None


class SFXLibrary:
    """Tag-addressed sound effects, normalized on demand."""

    def __init__(self, root: Path) -> None:
        """
        Args:
            root: Directory holding the effect files (tag = file stem)
                and optionally ``sfx_dictionary.json`` with variants.
        """
        self._root = root
        self._normalized_dir = root / "_normalized"
        self._dictionary_path = root / "sfx_dictionary.json"

    def tags(self) -> list[str]:
        """Available effect tags (folder stems + dictionary entries).

        Files registered as dictionary variants belong to their tag and
        are not surfaced as tags of their own.
        """
        dictionary = self._load_dictionary()
        variant_files = {
            str(v.get("file", "")).lower()
            for variants in dictionary.values()
            for v in variants
            if isinstance(v, dict)
        }
        found: set[str] = set()
        if self._root.is_dir():
            found.update(
                p.stem.lower() for p in self._root.iterdir()
                if p.suffix.lower() in _SUPPORTED and p.is_file()
                and p.name.lower() not in variant_files
            )
        for tag, variants in dictionary.items():
            if any((self._root / str(v.get("file", ""))).is_file()
                   for v in variants if isinstance(v, dict)):
                found.add(tag.lower())
        return sorted(found)

    def path_for(self, tag: str, seed: str = "") -> Path | None:
        """Normalized WAV for ``tag``; with multiple variants, ``seed``
        picks one deterministically. ``None`` if unknown/broken."""
        variant = self._pick_variant(tag, seed)
        if variant is None:
            return None
        source, gain = variant
        normalized = self._normalized_dir / f"{tag.lower()}__{source.stem.lower()}.wav"
        if not normalized.is_file() or normalized.stat().st_mtime < source.stat().st_mtime:
            try:
                self._normalized_dir.mkdir(parents=True, exist_ok=True)
                self._normalize(source, normalized, gain)
            except Exception as exc:
                logger.warning("Could not normalize SFX %s: %s", source.name, exc)
                return None
        return normalized

    def duration_ms(self, tag: str, seed: str = "") -> int:
        """Duration of the (seed-selected) normalized effect, in ms."""
        path = self.path_for(tag, seed)
        if path is None:
            return 0
        with wave.open(str(path), "rb") as wav:
            return round(wav.getnframes() * 1000 / wav.getframerate())

    def add_entry(
        self,
        tag: str,
        source_file: Path,
        auto_tags: list[str] | None = None,
        gain: float = 1.0,
    ) -> Path:
        """Register an uploaded file as a variant of ``tag``.

        Copies the file into the library root (as ``tag.ext`` or
        ``tag_N.ext``) and records it in ``sfx_dictionary.json``.
        """
        tag = tag.lower().strip()
        self._root.mkdir(parents=True, exist_ok=True)
        suffix = source_file.suffix.lower()
        destination = self._root / f"{tag}{suffix}"
        counter = 2
        while destination.exists():
            destination = self._root / f"{tag}_{counter}{suffix}"
            counter += 1
        shutil.copyfile(source_file, destination)

        dictionary = self._load_dictionary()
        dictionary.setdefault(tag, []).append(
            {"file": destination.name, "gain": gain, "auto_tags": auto_tags or []}
        )
        self._dictionary_path.write_text(
            json.dumps(dictionary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("SFX '%s': variant %s registered", tag, destination.name)
        return destination

    def _load_dictionary(self) -> dict[str, list[dict]]:
        if not self._dictionary_path.is_file():
            return {}
        try:
            data = json.loads(self._dictionary_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring broken sfx_dictionary.json: %s", exc)
            return {}

    def _pick_variant(self, tag: str, seed: str) -> tuple[Path, float] | None:
        """All variants for a tag = dictionary entries + the stem file."""
        tag = tag.lower().strip()
        variants: list[tuple[Path, float]] = []
        for entry in self._load_dictionary().get(tag, []):
            path = self._root / str(entry.get("file", ""))
            if path.is_file():
                variants.append((path, float(entry.get("gain", 1.0))))
        stem_file = self._find_source(tag)
        if stem_file is not None and all(stem_file != p for p, _ in variants):
            variants.append((stem_file, 1.0))
        if not variants:
            return None
        if len(variants) == 1:
            return variants[0]
        digest = hashlib.sha1(f"{tag}:{seed}".encode("utf-8")).digest()
        return variants[int.from_bytes(digest[:4], "big") % len(variants)]

    def _find_source(self, tag: str) -> Path | None:
        tag = tag.lower().strip()
        for suffix in _SUPPORTED:
            candidate = self._root / f"{tag}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def _normalize(self, source: Path, destination: Path, gain: float = 1.0) -> None:
        if source.suffix.lower() == ".wav":
            try:
                samples, rate = _read_wav(source)
            except Exception:  # float/ADPCM WAVs the stdlib reader rejects
                samples, rate = _read_compressed(source)
        else:
            samples, rate = _read_compressed(source)

        if rate != TARGET_RATE:
            samples = _resample_linear(samples, rate, TARGET_RATE)

        peak = float(np.max(np.abs(samples))) or 1.0
        samples = samples / peak * _GAIN * gain
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)

        with wave.open(str(destination), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(TARGET_RATE)
            wav.writeframes(pcm.tobytes())


def read_audio(path: Path) -> tuple[np.ndarray, int]:
    """Public audio reader: mono float32 in [-1, 1] + sample rate.

    WAV via stdlib (fast path); ogg/mp3/flac/odd-WAVs via soundfile.
    Shared by the SFX normalizer and the audio mixer.
    """
    if path.suffix.lower() == ".wav":
        try:
            return _read_wav(path)
        except Exception:
            return _read_compressed(path)
    return _read_compressed(path)


def resample_audio(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Public linear resampler (adequate for SFX/BGM material)."""
    if src_rate == dst_rate:
        return samples
    return _resample_linear(samples, src_rate, dst_rate)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV as mono float32 in [-1, 1] using only the stdlib."""
    with wave.open(str(path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {width}")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def _read_compressed(path: Path) -> tuple[np.ndarray, int]:
    """Decode ogg/mp3/flac: soundfile first (bundled libsndfile, no
    FFmpeg needed on Windows), torchaudio as last resort."""
    try:
        import soundfile as sf  # noqa: PLC0415 — deferred; ships with coqui-tts

        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return data.mean(axis=1).astype(np.float32), int(rate)
    except Exception:
        import torchaudio  # noqa: PLC0415 — heavy import, deferred

        tensor, rate = torchaudio.load(str(path))
        return tensor.mean(dim=0).numpy().astype(np.float32), int(rate)


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-interpolation resample — plenty for short sound effects."""
    duration = len(samples) / src_rate
    target_len = max(1, int(round(duration * dst_rate)))
    positions = np.linspace(0.0, len(samples) - 1, target_len)
    return np.interp(positions, np.arange(len(samples)), samples).astype(np.float32)
