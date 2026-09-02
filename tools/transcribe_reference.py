"""Reference-narration transcriber: mine a human narrator's style (developer tool).

    python -m tools.transcribe_reference --audio assets/reference/narracao.m4a --limit-seconds 60
    python -m tools.transcribe_reference --audio assets/reference/narracao.m4a --arcs arcos.txt --arc "Era de Ouro"

Given a LOCAL audio file of a human narrating a manga in PT-BR, transcribes
it with faster-whisper (segment + word timestamps) and writes, into
``assets/reference/<audio stem>/`` (git-ignored):

* ``segments.json``     -- the segments as plain dicts (start/end/text/words);
* ``transcript.md``     -- one ``[HH:MM:SS] text`` line per segment, with a
  heading whenever the story arc changes (``--arcs`` takes the YouTube
  description format: one ``HH:MM:SS - Title`` per line);
* ``pacing.json``       -- words per minute, pause statistics, segment lengths;
* ``style_excerpts.md`` -- windows of narration spread over the file and
  ranked by a narration-likeness heuristic, to become few-shot examples in
  the scriptwriter prompt (the ``style_addendum`` of
  :func:`mangawhisperer.engines.vlm.build_scriptwriter_prompt`).

Obtaining the audio is out of scope (see ``docs/reference-narration.md``):
nothing here downloads, uploads or publishes anything. The narration -- and
therefore every transcript and excerpt derived from it -- is the narrator's
copyrighted work: keep the outputs local, private and non-commercial, and
never redistribute them. Nothing here is part of the runtime pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import unicodedata
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from mangawhisperer.config import PROJECT_ROOT

REFERENCE_DIR = PROJECT_ROOT / "assets" / "reference"
DEFAULT_MODEL = "large-v3-turbo"  # faster-whisper alias -> mobiuslabsgmbh/faster-whisper-large-v3-turbo (1.6 GB)
DEFAULT_LANGUAGE = "pt"
DEFAULT_EXCERPTS = 8
DEFAULT_WINDOW_S = 90.0
LONG_PAUSE_S = 1.5
LONG_PAUSE_LIST_CAP = 50
MIN_EXCERPT_WORDS = 20
SAMPLE_RATE = 16_000  # faster-whisper decodes every input to 16 kHz mono
OUTPUT_FILES = ("segments.json", "transcript.md", "pacing.json", "style_excerpts.md")

Segment = dict[str, Any]
Clip = tuple[float, float | None]  # (start_s, end_s); end None = up to the end of the audio
Transcriber = Callable[..., tuple[Iterable[Any], Any]]


def _r(value: float) -> float:
    return round(float(value), 3)


# ── story arcs ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Arc:
    """A story-arc / chapter marker: where it starts and how it is called."""

    start_s: float
    title: str


_TIMESTAMP = r"\d{1,3}:\d{2}(?::\d{2})?"
_ARC_LINE = re.compile(rf"^\s*(?P<ts>{_TIMESTAMP})\s*(?:[-–—|]\s*)?(?P<title>\S.*?)\s*$")


def parse_timestamp(text: str) -> float:
    """``HH:MM:SS``, ``H:MM:SS`` or ``MM:SS`` -> seconds."""
    parts = text.strip().split(":")
    if not 2 <= len(parts) <= 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Timestamp inválido: {text!r} (esperado HH:MM:SS ou MM:SS)")
    numbers = [int(part) for part in parts]
    while len(numbers) < 3:
        numbers.insert(0, 0)
    hours, minutes, seconds = numbers
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_arc_list(text: str) -> list[Arc]:
    """Arcs from a chapter list in the YouTube description format.

    Reads lines like ``01:07:35 - A Era de Ouro`` (also ``1:07:35``,
    ``MM:SS``, an en/em dash or no separator at all); every other line is
    ignored, so a whole ``.description`` file can be passed as is. Sorted by
    start time.
    """
    arcs: list[Arc] = []
    for line in text.splitlines():
        match = _ARC_LINE.match(line)
        if not match:
            continue
        title = match.group("title").strip(" \t-–—|:")
        if title:
            arcs.append(Arc(parse_timestamp(match.group("ts")), title))
    return sorted(arcs, key=lambda arc: arc.start_s)


def shift_arcs(arcs: Sequence[Arc], offset_s: float) -> list[Arc]:
    """Re-base arc starts on an audio slice that begins at ``offset_s`` of
    the original video (``--download-sections`` keeps the video's chapter
    times in the description, not the slice's)."""
    return [Arc(arc.start_s - offset_s, arc.title) for arc in arcs]


def arc_for_time(arcs: Sequence[Arc], t_s: float) -> str | None:
    """Title of the arc that started last at ``t_s``; ``None`` before the first one."""
    ordered = sorted(arcs, key=lambda arc: arc.start_s)
    index = bisect_right([arc.start_s for arc in ordered], t_s) - 1
    return ordered[index].title if index >= 0 else None


def _fold(text: str) -> str:
    """Case- and accent-insensitive key (``Guardião`` == ``guardiao``)."""
    stripped = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    return " ".join(stripped.casefold().split())


def find_arc(arcs: Sequence[Arc], title: str) -> Arc:
    """The arc whose title contains ``title`` (case/accent-insensitive; an
    exact title wins over substring matches, then the earliest)."""
    ordered = sorted(arcs, key=lambda arc: arc.start_s)
    wanted = _fold(title)
    matches = [arc for arc in ordered if wanted and wanted in _fold(arc.title)]
    if not matches:
        known = ", ".join(arc.title for arc in ordered) or "(nenhum)"
        raise ValueError(f"Arco {title!r} não encontrado; arcos conhecidos: {known}")
    exact = [arc for arc in matches if _fold(arc.title) == wanted]
    return (exact or matches)[0]


def arc_bounds(arcs: Sequence[Arc], title: str, total_duration: float | None) -> tuple[float, float | None]:
    """``(start, end)`` of the arc matching ``title``: the end is the next
    arc's start, or ``total_duration`` for the last arc (``None`` = open-ended)."""
    ordered = sorted(arcs, key=lambda arc: arc.start_s)
    arc = find_arc(ordered, title)
    index = ordered.index(arc)
    end = ordered[index + 1].start_s if index + 1 < len(ordered) else total_duration
    return arc.start_s, end


# ── pacing ────────────────────────────────────────────────────────────


def count_words(text: str) -> int:
    """Whitespace tokens with at least one letter or digit (``...`` is not a word)."""
    return sum(1 for token in text.split() if any(ch.isalnum() for ch in token))


def _percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default); 0.0 for no values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def pacing_stats(
    segments: Sequence[Mapping[str, Any]],
    total_duration_s: float | None = None,
    long_pause_s: float = LONG_PAUSE_S,
) -> dict[str, Any]:
    """Delivery statistics of a transcript: words per minute (over speech time
    and over the whole ``total_duration_s`` -- the segments' span when not
    given), pauses between consecutive segments (overlaps count as 0), long
    pauses (>= ``long_pause_s``, listed longest first and capped) and segment
    lengths. Safe for zero or one segment."""
    ordered = sorted(segments, key=lambda seg: (seg["start"], seg["end"]))
    durations = [max(seg["end"] - seg["start"], 0.0) for seg in ordered]
    speech = sum(durations)
    words = sum(count_words(seg["text"]) for seg in ordered)
    if total_duration_s is None:
        total = (ordered[-1]["end"] - ordered[0]["start"]) if ordered else 0.0
    else:
        total = float(total_duration_s)
    gaps = [(prev["end"], max(nxt["start"] - prev["end"], 0.0)) for prev, nxt in zip(ordered, ordered[1:])]
    gap_values = [gap for _, gap in gaps]
    long_gaps = sorted((item for item in gaps if item[1] >= long_pause_s), key=lambda item: item[1], reverse=True)
    return {
        "segments": len(ordered),
        "total_duration_s": _r(total),
        "speech_duration_s": _r(speech),
        "speech_ratio": _r(speech / total) if total > 0 else 0.0,
        "words": words,
        "words_per_minute_speech": _r(words * 60.0 / speech) if speech > 0 else 0.0,
        "words_per_minute_total": _r(words * 60.0 / total) if total > 0 else 0.0,
        "words_per_segment_mean": _r(words / len(ordered)) if ordered else 0.0,
        "segment_duration_s": {
            "p10": _r(_percentile(durations, 10)),
            "p50": _r(_percentile(durations, 50)),
            "p90": _r(_percentile(durations, 90)),
        },
        "pauses": {
            "count": len(gap_values),
            "mean_s": _r(sum(gap_values) / len(gap_values)) if gap_values else 0.0,
            "median_s": _r(_percentile(gap_values, 50)),
            "p90_s": _r(_percentile(gap_values, 90)),
        },
        "long_pauses": {
            "threshold_s": long_pause_s,
            "count": len(long_gaps),
            "list": [{"at_s": _r(at), "gap_s": _r(gap)} for at, gap in long_gaps[:LONG_PAUSE_LIST_CAP]],
        },
    }


# ── style excerpts ────────────────────────────────────────────────────

# PT-BR words a narrator uses when describing what the panel shows (atmosphere,
# stage directions) and rarely appear in dialogue read aloud.
NARRATION_CUES: frozenset[str] = frozenset({
    "silêncio", "sombra", "sombras", "escuridão", "trevas", "névoa", "vento", "chuva", "noite", "lua",
    "sangue", "lâmina", "espada", "chamas", "fogo", "brilho", "olhar", "olhos", "rosto", "corpo", "corpos",
    "passos", "grito", "gritos", "ruído", "eco", "cheiro", "ferida", "feridas",
    "enquanto", "lentamente", "subitamente", "súbito",
    "surge", "surgem", "avança", "avançam", "ergue", "atravessa", "observa", "encara", "aponta",
    "respira", "sussurra", "ecoa", "ecoam", "cai", "caem", "corta", "salta", "recua",
})
NARRATION_PHRASES: tuple[str, ...] = (
    "de repente", "naquele momento", "nesse instante", "por um instante", "ao longe", "em meio",
    "diante de", "sem dizer", "ao mesmo tempo", "em silêncio", "a passos", "no meio da",
)
_VERB_SUFFIXES = ("ou", "eu", "iu", "ava", "avam", "iam", "ando", "endo", "indo")
_MIN_VERB_LEN = 4  # keeps "eu", "sou", "vou", "meu" out
_ONOMATOPOEIA = re.compile(r"(.)\1\1")  # a letter three times in a row: "booom", "shhh"
_LETTERS = re.compile(r"[^\W\d_]+")


def narration_score(text: str, min_words: int = 1) -> float:
    """Narration-likeness of a stretch of transcript, as cue hits per word.

    Deliberately simple: counts (a) description/atmosphere cue words and
    phrases (``NARRATION_CUES``, ``NARRATION_PHRASES``), (b) tokens shaped
    like 3rd-person narrative verbs -- at least 4 letters, ending in
    -ou/-eu/-iu (past), -ava/-avam/-iam (imperfect) or -ando/-endo/-indo
    (gerund) -- and (c) onomatopoeia-like tokens with a letter repeated three
    times ("booom"). Divided by the word count, floored at ``min_words`` so a
    two-word window cannot outrank a real paragraph. Dialogue read aloud
    ("Eu vou com você!") scores near zero; stage-direction prose scores high.
    """
    lowered = text.casefold()
    tokens = _LETTERS.findall(lowered)
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in NARRATION_CUES)
    hits += sum(lowered.count(phrase) for phrase in NARRATION_PHRASES)
    hits += sum(1 for token in tokens if len(token) >= _MIN_VERB_LEN and token.endswith(_VERB_SUFFIXES))
    hits += sum(1 for token in tokens if _ONOMATOPOEIA.search(token))
    return round(hits / max(count_words(text), min_words, 1), 4)


@dataclass(frozen=True)
class Excerpt:
    """A contiguous window of segments chosen as a style example."""

    start: float
    end: float
    text: str
    score: float


def select_excerpts(
    segments: Sequence[Mapping[str, Any]],
    n: int = DEFAULT_EXCERPTS,
    window_s: float = DEFAULT_WINDOW_S,
) -> list[Excerpt]:
    """Up to ``n`` non-overlapping windows of ~``window_s`` seconds spread
    over the file: the timeline is cut into ``n`` equal bins and each bin
    contributes the window (a run of consecutive segments starting there)
    with the highest :func:`narration_score`, earliest on ties."""
    ordered = sorted(segments, key=lambda seg: (seg["start"], seg["end"]))
    if not ordered or n <= 0:
        return []
    windows: list[Excerpt] = []
    for i, first in enumerate(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1]["end"] - first["start"] <= window_s:
            j += 1
        chunk = ordered[i:j + 1]
        text = " ".join(seg["text"].strip() for seg in chunk if seg["text"].strip())
        windows.append(Excerpt(first["start"], chunk[-1]["end"], text, narration_score(text, MIN_EXCERPT_WORDS)))

    t0, t1 = ordered[0]["start"], ordered[-1]["end"]
    span = max(t1 - t0, 1e-9)
    bins: list[list[Excerpt]] = [[] for _ in range(n)]
    for window in windows:
        bins[min(int((window.start - t0) / span * n), n - 1)].append(window)
    chosen: list[Excerpt] = []
    for pool in bins:
        limit = chosen[-1].end if chosen else -math.inf
        candidates = [window for window in pool if window.start >= limit]
        if candidates:
            chosen.append(max(candidates, key=lambda window: (window.score, -window.start)))
    return chosen


# ── formatting ────────────────────────────────────────────────────────


def format_timestamp(seconds: float) -> str:
    """Seconds -> ``HH:MM:SS`` (floored; negatives clamp to zero)."""
    total = max(int(seconds), 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def slugify(text: str) -> str:
    """Folder-safe ASCII name for an arc title."""
    ascii_text = _fold(text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_") or "arco"


def write_transcript_md(segments: Sequence[Mapping[str, Any]], arcs: Sequence[Arc]) -> str:
    """One ``[HH:MM:SS] text`` line per segment, with a ``## Arc`` heading
    whenever the arc changes (none before the first arc)."""
    lines: list[str] = []
    current: str | None = None
    for segment in sorted(segments, key=lambda seg: (seg["start"], seg["end"])):
        arc = arc_for_time(arcs, segment["start"])
        if arc is not None and arc != current:
            if lines:
                lines.append("")
            lines.append(f"## {arc}")
            current = arc
        lines.append(f"[{format_timestamp(segment['start'])}] {segment['text']}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_style_excerpts_md(excerpts: Sequence[Excerpt], arcs: Sequence[Arc], source: str) -> str:
    """The excerpts as a Markdown file with a header explaining their purpose
    (few-shot material for the scriptwriter prompt) and their legal status."""
    lines = [
        f"# Excertos de estilo — {source}",
        "",
        "Trechos da narração humana escolhidos automaticamente (janelas contíguas espalhadas pelo",
        "arquivo, ordenadas pela densidade de descrição narrativa) para servir de exemplos few-shot",
        "no prompt do roteirista (`style_addendum` de `build_scriptwriter_prompt`).",
        "",
        "Uso local e privado: o texto é obra do narrador original — não redistribua, não publique",
        "e não o inclua em nada que vá sair desta máquina além do provedor de LLM que você mesmo",
        "escolher.",
        "",
    ]
    for index, excerpt in enumerate(excerpts, start=1):
        arc = arc_for_time(arcs, excerpt.start)
        span = f"[{format_timestamp(excerpt.start)} - {format_timestamp(excerpt.end)}]"
        title = f"## {index}. {span}" + (f" · {arc}" if arc else "") + f" · score {excerpt.score:.3f}"
        lines += [title, "", excerpt.text, ""]
    if not excerpts:
        lines += ["(nenhum excerto: a transcrição está vazia)", ""]
    return "\n".join(lines)


# ── transcription ─────────────────────────────────────────────────────


def resolve_device(device: str, compute_type: str, cuda_available: bool) -> tuple[str, str]:
    """``auto`` device -> cuda when available, else cpu. ``auto`` compute type
    -> float16 on cuda (int8 kernels on RTX 50 need ctranslate2 >= 4.7) and
    int8 on cpu (float16 is not supported there). Explicit values pass through."""
    if device == "auto":
        device = "cuda" if cuda_available else "cpu"
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415 — heavy, only for the auto device choice
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _preload_cuda_libraries() -> None:
    """Windows: ctranslate2 loads ``cublas64_12.dll`` (and, before 4.6.3,
    cuDNN 9) by name through PATH. Importing torch first pulls the copies
    bundled in ``torch/lib`` into the process, and the directory goes on PATH
    as a fallback (see docs/reference-narration.md)."""
    if sys.platform != "win32":
        return
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return
    lib_dir = Path(torch.__file__).resolve().parent / "lib"
    if lib_dir.is_dir():
        os.environ["PATH"] = str(lib_dir) + os.pathsep + os.environ.get("PATH", "")


def _faster_whisper_transcriber(
    audio_path: Path, *, model: str, device: str, compute_type: str, language: str, clip: Clip | None,
) -> tuple[Iterable[Any], Any]:
    """Default transcriber: faster-whisper with word timestamps and VAD.

    ``clip`` slices the decoded waveform ourselves so the VAD stays on
    (faster-whisper's own ``clip_timestamps`` disables it); timestamps then
    come back relative to the clip start, as they do for any array.
    """
    if device == "cuda":
        _preload_cuda_libraries()
    try:
        from faster_whisper import WhisperModel, decode_audio  # noqa: PLC0415 — optional dependency
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper não está instalado. Rode: pip install faster-whisper "
            "(ou pip install -e \".[reference]\"); veja docs/reference-narration.md."
        ) from exc

    audio: Any = str(audio_path)
    if clip is not None:
        start, end = clip
        first = max(start, 0.0)
        if end is not None and end <= first:
            # never let a negative ``end`` reach numpy: as a stop index it would count from the
            # end of the waveform and quietly transcribe almost the whole file as "the clip"
            raise ValueError(f"Trecho inválido: o fim ({end:g} s) não é depois do início ({start:g} s).")
        waveform = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        stop = None if end is None else int(end * SAMPLE_RATE)
        audio = waveform[int(first * SAMPLE_RATE):stop]
        if len(audio) == 0:
            raise ValueError(f"O trecho pedido ({format_timestamp(start)} em diante) começa depois do fim do áudio.")
    print(f"Carregando {model} ({device}, {compute_type})...")
    whisper = WhisperModel(model, device=device, compute_type=compute_type)
    return whisper.transcribe(
        audio, language=language, beam_size=5, word_timestamps=True,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
    )


def _segment_to_dict(segment: Any, offset: float) -> Segment:
    words = [
        {"start": _r(word.start + offset), "end": _r(word.end + offset), "word": word.word,
         "probability": _r(getattr(word, "probability", 0.0))}
        for word in (getattr(segment, "words", None) or [])
    ]
    return {"start": _r(segment.start + offset), "end": _r(segment.end + offset),
            "text": segment.text.strip(), "words": words}


def _info_to_dict(info: Any, model: str, device: str, compute_type: str, clip: Clip | None) -> dict[str, Any]:
    probability = getattr(info, "language_probability", None)
    return {
        "model": model,
        "device": device,
        "compute_type": compute_type,
        "language": getattr(info, "language", None),
        "language_probability": _r(probability) if probability is not None else None,
        "duration_s": _r(getattr(info, "duration", 0.0) or 0.0),
        "duration_after_vad_s": _r(getattr(info, "duration_after_vad", 0.0) or 0.0),
        "clip": list(clip) if clip is not None else None,
    }


def transcribe(
    audio_path: Path,
    model: str = DEFAULT_MODEL,
    device: str = "auto",
    compute_type: str = "auto",
    language: str = DEFAULT_LANGUAGE,
    transcriber: Transcriber | None = None,
    clip: Clip | None = None,
    progress: Callable[[int, float], None] | None = None,
) -> tuple[list[Segment], dict[str, Any]]:
    """Transcribe ``audio_path`` (or its ``clip`` = ``(start_s, end_s | None)``)
    into plain segment dicts (``start``/``end``/``text``/``words``, absolute
    seconds, empty segments dropped) plus an info dict.

    ``transcriber`` is the model wrapper -- faster-whisper by default; tests
    inject a fake -- called with clip-relative semantics: its timestamps are
    offset by the clip start here. ``progress(done, end_s)`` is called after
    each kept segment (faster-whisper decodes lazily while we iterate).
    """
    device, compute_type = resolve_device(device, compute_type, _cuda_available() if device == "auto" else False)
    run = transcriber or _faster_whisper_transcriber
    raw_segments, raw_info = run(
        Path(audio_path), model=model, device=device, compute_type=compute_type, language=language, clip=clip,
    )
    offset = clip[0] if clip is not None else 0.0
    segments: list[Segment] = []
    for raw in raw_segments:
        converted = _segment_to_dict(raw, offset)
        if not converted["text"]:
            continue
        segments.append(converted)
        if progress is not None:
            progress(len(segments), converted["end"])
    segments.sort(key=lambda seg: (seg["start"], seg["end"]))
    return segments, _info_to_dict(raw_info, model, device, compute_type, clip)


# ── CLI ───────────────────────────────────────────────────────────────


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio", type=Path, required=True, help="Arquivo de áudio LOCAL (m4a, mp3, opus, wav...).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Pasta de saída (padrão: assets/reference/<nome do áudio>/; com --arc, uma subpasta do arco).")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Modelo faster-whisper: alias (large-v3-turbo, distil-large-v3, medium, small) ou repo do HF.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--compute-type", default="auto",
                        help="Tipo de cálculo do CTranslate2 (auto = float16 na GPU, int8 na CPU).")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE, help="Código ISO 639-1 do idioma falado.")
    parser.add_argument("--arcs", type=Path, default=None,
                        help="Lista de arcos: 'HH:MM:SS - Título' por linha (as demais linhas são ignoradas).")
    parser.add_argument("--arcs-offset", default=None,
                        help="Se o áudio é um recorte do vídeo, o ponto do vídeo onde ele começa (HH:MM:SS ou segundos).")
    parser.add_argument("--arc", default=None,
                        help="Restringe a saída a um arco (trecho do título; ignora maiúsculas e acentos).")
    parser.add_argument("--excerpts", type=int, default=DEFAULT_EXCERPTS, help="Quantos excertos de estilo escolher.")
    parser.add_argument("--window", type=float, default=DEFAULT_WINDOW_S,
                        help="Duração aproximada de cada excerto, em segundos.")
    parser.add_argument("--limit-seconds", type=float, default=None,
                        help="Transcreve só os primeiros N segundos (do arco, se houver): teste rápido.")
    args = parser.parse_args(argv)
    if not args.audio.is_file():
        parser.error(f"áudio não encontrado: {args.audio}")
    if args.arc and args.arcs is None:
        parser.error("--arc exige --arcs (a lista de arcos).")
    if args.arcs is not None and not args.arcs.is_file():
        parser.error(f"lista de arcos não encontrada: {args.arcs}")
    if args.limit_seconds is not None and not (math.isfinite(args.limit_seconds) and args.limit_seconds > 0):
        parser.error(f"--limit-seconds deve ser um número de segundos maior que zero (recebido {args.limit_seconds!r}).")
    if args.arcs_offset is not None:
        try:
            args.arcs_offset = parse_timestamp(args.arcs_offset)
        except ValueError:
            try:
                args.arcs_offset = float(args.arcs_offset)
            except ValueError:
                parser.error(f"--arcs-offset inválido: {args.arcs_offset!r} (use HH:MM:SS ou segundos).")
    return args


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None, transcriber: Transcriber | None = None) -> int:
    args = parse_args(argv)
    arcs = parse_arc_list(args.arcs.read_text(encoding="utf-8-sig")) if args.arcs else []
    if args.arcs_offset is not None:
        arcs = shift_arcs(arcs, args.arcs_offset)

    clip: Clip | None = None
    arc_title: str | None = None
    if args.arc:
        arc_title = find_arc(arcs, args.arc).title
        start, end = arc_bounds(arcs, args.arc, None)
        clip = (max(start, 0.0), end)
        if end is not None and end <= clip[0]:
            # nothing of the arc is in this audio: refuse now, before an output folder exists and before a
            # negative end could reach the waveform slice (where it would mean "up to N s before the end")
            if args.arcs_offset is not None and end <= 0.0:
                reason = (f"termina em {format_timestamp(end + args.arcs_offset)} do vídeo e o recorte só começa "
                          f"em {format_timestamp(args.arcs_offset)} (--arcs-offset)")
            else:
                reason = "o arco seguinte começa no mesmo instante (confira a lista em --arcs)"
            raise SystemExit(f"O arco '{arc_title}' não está neste áudio: {reason}.")
    if args.limit_seconds is not None:
        start = clip[0] if clip else 0.0
        end = start + args.limit_seconds
        if clip and clip[1] is not None:
            end = min(end, clip[1])
        clip = (start, end)

    out_dir = args.out_dir or (REFERENCE_DIR / args.audio.stem)
    if arc_title:
        out_dir = out_dir / slugify(arc_title)
    out_dir.mkdir(parents=True, exist_ok=True)

    scope = f" - arco '{arc_title}'" if arc_title else ""
    if clip:
        until = format_timestamp(clip[1]) if clip[1] is not None else "fim"
        scope += f" - trecho {format_timestamp(clip[0])} a {until}"
    print(f"Transcrevendo {args.audio.name}{scope}...")

    def report(done: int, t_s: float) -> None:
        if done % 50 == 0:
            print(f"  ... {format_timestamp(t_s)} ({done} segmentos)")

    segments, info = transcribe(
        args.audio, model=args.model, device=args.device, compute_type=args.compute_type,
        language=args.language, transcriber=transcriber, clip=clip, progress=report,
    )
    stats = pacing_stats(segments, total_duration_s=info["duration_s"] or None)
    excerpts = select_excerpts(segments, args.excerpts, args.window)

    _write_json(out_dir / "segments.json", {
        "audio": args.audio.name, "arc": arc_title, "info": info,
        "arcs": [{"start_s": arc.start_s, "title": arc.title} for arc in arcs], "segments": segments,
    })
    header = (f"<!-- Transcrição automática ({info['model']}) de {args.audio.name}. "
              "Obra do narrador original: uso local e privado, não redistribuir. -->\n\n")
    (out_dir / "transcript.md").write_text(header + write_transcript_md(segments, arcs), encoding="utf-8")
    _write_json(out_dir / "pacing.json", {"audio": args.audio.name, "arc": arc_title, "clip": info["clip"], **stats})
    (out_dir / "style_excerpts.md").write_text(
        write_style_excerpts_md(excerpts, arcs, source=args.audio.name), encoding="utf-8",
    )

    language = info.get("language") or "?"
    print(f"Transcrição concluída: {format_timestamp(stats['total_duration_s'])} de áudio, "
          f"{len(segments)} segmentos, idioma {language}.")
    print(f"Ritmo: {stats['words_per_minute_speech']:.0f} palavras/min na fala "
          f"({stats['words_per_minute_total']:.0f} no tempo total), {stats['words']} palavras, "
          f"{stats['long_pauses']['count']} pausas longas (>= {LONG_PAUSE_S} s).")
    print(f"{len(excerpts)} excertos de estilo. Saída em {out_dir}:")
    for name in OUTPUT_FILES:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
