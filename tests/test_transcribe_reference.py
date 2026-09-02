"""Pure parts of the reference-narration transcriber (no model is ever loaded here)."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tools.transcribe_reference import (
    LONG_PAUSE_S,
    SAMPLE_RATE,
    Arc,
    _faster_whisper_transcriber,
    arc_bounds,
    arc_for_time,
    count_words,
    format_timestamp,
    main,
    narration_score,
    pacing_stats,
    parse_arc_list,
    parse_timestamp,
    resolve_device,
    select_excerpts,
    shift_arcs,
    slugify,
    transcribe,
    write_style_excerpts_md,
    write_transcript_md,
)

ARCS = [Arc(0.0, "Abertura"), Arc(330.0, "Guardião dos Desejos"), Arc(4055.0, "A Era de Ouro")]
NARRATION = "Guts avançou lentamente pela sombra enquanto o vento uivava."
DIALOGUE = "Eu vou com você, não me deixe aqui sozinho."


def _seg(start: float, end: float, text: str) -> dict:
    return {"start": start, "end": end, "text": text, "words": []}


# ── arcs ──────────────────────────────────────────────────────────────


def test_parse_timestamp_accepts_hms_and_ms():
    assert parse_timestamp("01:07:35") == 4055.0
    assert parse_timestamp("1:07:35") == 4055.0
    assert parse_timestamp("05:30") == 330.0
    with pytest.raises(ValueError):
        parse_timestamp("abc")


def test_parse_arc_list_reads_description_lines_and_sorts_by_start():
    text = """Capítulos do vídeo:

01:07:35 - A Era de Ouro
05:30 – Guardião dos Desejos
uma linha de prosa sem timestamp, nem 2024: nada
0:00 Abertura
12:30 -
"""
    assert parse_arc_list(text) == ARCS
    assert parse_arc_list("") == []


def test_shift_arcs_moves_every_start_by_the_slice_origin():
    assert shift_arcs(ARCS, 330.0) == [
        Arc(-330.0, "Abertura"), Arc(0.0, "Guardião dos Desejos"), Arc(3725.0, "A Era de Ouro"),
    ]
    assert arc_for_time(shift_arcs(ARCS, 330.0), 0.0) == "Guardião dos Desejos"


def test_arc_for_time_returns_the_arc_that_started_last():
    assert arc_for_time(ARCS, 0.0) == "Abertura"
    assert arc_for_time(ARCS, 329.9) == "Abertura"
    assert arc_for_time(ARCS, 330.0) == "Guardião dos Desejos"
    assert arc_for_time(ARCS, 99999.0) == "A Era de Ouro"
    assert arc_for_time([], 10.0) is None
    assert arc_for_time(ARCS[1:], 10.0) is None  # before the first arc


def test_arc_bounds_matches_a_title_substring_case_and_accent_insensitively():
    assert arc_bounds(ARCS, "guardiao", 5000.0) == (330.0, 4055.0)
    assert arc_bounds(ARCS, "Era de Ouro", 5000.0) == (4055.0, 5000.0)
    assert arc_bounds(ARCS, "era de ouro", None) == (4055.0, None)  # last arc, unknown total: open-ended
    with pytest.raises(ValueError, match="Abertura"):  # the error lists the known titles
        arc_bounds(ARCS, "Falcão", 5000.0)


# ── pacing ────────────────────────────────────────────────────────────


def test_count_words_ignores_punctuation_only_tokens():
    assert count_words("De repente, silêncio... — ...") == 3
    assert count_words("") == 0


def test_pacing_stats_counts_words_pauses_and_percentiles():
    segments = [
        _seg(0.0, 2.0, "Guts avança pela floresta"),
        _seg(2.5, 4.5, "O vento uiva"),
        _seg(7.0, 9.0, "De repente, silêncio."),
    ]
    stats = pacing_stats(segments)
    assert stats["segments"] == 3
    assert stats["words"] == 10
    assert stats["speech_duration_s"] == 6.0
    assert stats["total_duration_s"] == 9.0  # span of the segments when no duration is given
    assert stats["words_per_minute_speech"] == 100.0
    assert stats["words_per_minute_total"] == pytest.approx(66.667, abs=1e-3)
    assert stats["pauses"] == {"count": 2, "mean_s": 1.5, "median_s": 1.5, "p90_s": 2.3}
    assert stats["long_pauses"]["count"] == 1
    assert stats["long_pauses"]["list"] == [{"at_s": 4.5, "gap_s": 2.5}]
    assert stats["segment_duration_s"] == {"p10": 2.0, "p50": 2.0, "p90": 2.0}
    assert stats["words_per_segment_mean"] == pytest.approx(3.333, abs=1e-3)
    assert pacing_stats(segments, total_duration_s=12.0)["words_per_minute_total"] == 50.0


def test_pacing_stats_is_robust_to_zero_and_one_segment():
    empty = pacing_stats([])
    assert empty["segments"] == 0 and empty["words"] == 0
    assert empty["words_per_minute_speech"] == 0.0 and empty["words_per_minute_total"] == 0.0
    assert empty["pauses"]["count"] == 0
    assert empty["long_pauses"] == {"threshold_s": LONG_PAUSE_S, "count": 0, "list": []}
    assert empty["segment_duration_s"] == {"p10": 0.0, "p50": 0.0, "p90": 0.0}

    single = pacing_stats([_seg(3.0, 5.0, "Silêncio.")])
    assert single["words"] == 1 and single["pauses"]["count"] == 0
    assert single["total_duration_s"] == 2.0 and single["words_per_minute_speech"] == 30.0


def test_pacing_stats_clamps_overlaps_and_caps_the_long_pause_list():
    overlapping = [_seg(0.0, 3.0, "a b"), _seg(2.0, 4.0, "c d")]  # whisper occasionally overlaps
    assert pacing_stats(overlapping)["pauses"] == {"count": 1, "mean_s": 0.0, "median_s": 0.0, "p90_s": 0.0}

    segments, t = [], 0.0
    for i in range(61):  # 60 gaps growing from 1.5 s to 7.4 s
        segments.append(_seg(t, t + 1.0, "palavra"))
        t += 1.0 + 1.5 + i * 0.1
    stats = pacing_stats(segments)
    assert stats["long_pauses"]["count"] == 60
    assert len(stats["long_pauses"]["list"]) == 50
    assert stats["long_pauses"]["list"][0]["gap_s"] == pytest.approx(7.4, abs=1e-6)  # longest first


# ── excerpts ──────────────────────────────────────────────────────────


def test_narration_score_prefers_description_over_dialogue():
    assert narration_score(NARRATION) > narration_score(DIALOGUE)
    assert narration_score(DIALOGUE) == 0.0
    assert narration_score("De repente, booom!") > 0.0  # phrase cue + onomatopoeia
    assert narration_score("") == 0.0


def test_select_excerpts_spreads_windows_over_the_file_and_prefers_narration():
    segments = []
    for i in range(60):  # 10 s cadence: 5 s of speech, 5 s of silence; bins are ~198 s wide
        narrative = 25 <= i <= 30 or 45 <= i <= 50  # one narration block inside bins 2 and 3
        segments.append(_seg(i * 10.0, i * 10.0 + 5.0, NARRATION if narrative else DIALOGUE))

    excerpts = select_excerpts(segments, n=3, window_s=60.0)

    assert [(e.start, e.end) for e in excerpts] == [(0.0, 55.0), (250.0, 305.0), (450.0, 505.0)]
    assert all(e.end - e.start <= 60.0 for e in excerpts)
    assert excerpts[1].text.startswith("Guts avançou") and excerpts[1].score > excerpts[0].score
    assert excerpts[0].score == 0.0


def test_select_excerpts_handles_empty_short_and_zero_requests():
    assert select_excerpts([], n=3, window_s=60.0) == []
    single = [_seg(10.0, 12.0, "Guts avança.")]
    assert select_excerpts(single, n=0, window_s=60.0) == []
    (only,) = select_excerpts(single, n=8, window_s=60.0)
    assert (only.start, only.end, only.text) == (10.0, 12.0, "Guts avança.")


# ── formatting ────────────────────────────────────────────────────────


def test_format_timestamp_and_slugify():
    assert format_timestamp(0) == "00:00:00"
    assert format_timestamp(4055.7) == "01:07:35"
    assert format_timestamp(-1) == "00:00:00"
    assert slugify("A Era de Ouro: Guardião!") == "a_era_de_ouro_guardiao"


def test_write_transcript_md_adds_a_heading_when_the_arc_changes():
    segments = [_seg(0.0, 2.0, "Abre."), _seg(100.0, 102.0, "Segue."), _seg(400.0, 402.0, "Muda.")]
    assert write_transcript_md(segments, ARCS).splitlines() == [
        "## Abertura",
        "[00:00:00] Abre.",
        "[00:01:40] Segue.",
        "",
        "## Guardião dos Desejos",
        "[00:06:40] Muda.",
    ]
    assert write_transcript_md(segments, []).splitlines() == [
        "[00:00:00] Abre.", "[00:01:40] Segue.", "[00:06:40] Muda.",
    ]
    assert write_transcript_md(segments[:1], ARCS[1:]).splitlines() == ["[00:00:00] Abre."]  # before the first arc


def test_write_style_excerpts_md_has_a_local_use_header_and_timestamped_excerpts():
    excerpts = select_excerpts([_seg(4060.0, 4070.0, NARRATION)], n=1, window_s=30.0)
    text = write_style_excerpts_md(excerpts, ARCS, source="narracao.m4a")
    assert "few-shot" in text and "uso local" in text.lower()
    assert "[01:07:40 - 01:07:50]" in text and "A Era de Ouro" in text and NARRATION in text


# ── transcription (fake model) ────────────────────────────────────────


class FakeTranscriber:
    """Stands in for faster-whisper: clip-relative segments, as the real
    model returns for a sliced waveform."""

    def __init__(self, duration: float = 30.0):
        self.calls: list[dict] = []
        self.duration = duration

    def __call__(self, audio_path, *, model, device, compute_type, language, clip):
        self.calls.append({"audio_path": audio_path, "model": model, "device": device,
                           "compute_type": compute_type, "language": language, "clip": clip})
        words = [SimpleNamespace(start=0.0, end=0.4, word=" Guts", probability=0.98),
                 SimpleNamespace(start=0.5, end=1.0, word=" avança.", probability=0.91)]
        segments = [
            SimpleNamespace(start=0.0, end=1.0, text=" Guts avança. ", words=words),
            SimpleNamespace(start=3.0, end=4.0, text="   ", words=None),  # empty: dropped
            SimpleNamespace(start=5.0, end=7.0, text="O vento uiva.", words=None),
        ]
        info = SimpleNamespace(language=language, language_probability=0.99,
                               duration=self.duration, duration_after_vad=self.duration - 2)
        return iter(segments), info


def test_transcribe_converts_segments_to_dicts_and_offsets_them_by_the_clip(tmp_path):
    fake = FakeTranscriber()
    seen: list[tuple[int, float]] = []

    segments, info = transcribe(
        tmp_path / "a.m4a", model="small", device="cpu", compute_type="int8", language="pt",
        transcriber=fake, clip=(100.0, 130.0), progress=lambda done, t: seen.append((done, t)),
    )

    assert fake.calls == [{"audio_path": tmp_path / "a.m4a", "model": "small", "device": "cpu",
                           "compute_type": "int8", "language": "pt", "clip": (100.0, 130.0)}]
    assert [(s["start"], s["end"], s["text"]) for s in segments] == [
        (100.0, 101.0, "Guts avança."), (105.0, 107.0, "O vento uiva."),
    ]
    assert segments[0]["words"][0] == {"start": 100.0, "end": 100.4, "word": " Guts", "probability": 0.98}
    assert segments[1]["words"] == []
    assert seen == [(1, 101.0), (2, 107.0)]
    assert info == {
        "model": "small", "device": "cpu", "compute_type": "int8", "language": "pt",
        "language_probability": 0.99, "duration_s": 30.0, "duration_after_vad_s": 28.0,
        "clip": [100.0, 130.0],
    }


def test_resolve_device_auto_picks_cuda_float16_or_cpu_int8():
    assert resolve_device("auto", "auto", cuda_available=True) == ("cuda", "float16")
    assert resolve_device("auto", "auto", cuda_available=False) == ("cpu", "int8")
    assert resolve_device("cuda", "auto", cuda_available=False) == ("cuda", "float16")  # explicit wins
    assert resolve_device("cpu", "int8_float32", cuda_available=True) == ("cpu", "int8_float32")


def test_transcribe_without_faster_whisper_raises_a_friendly_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)  # makes `import faster_whisper` fail
    with pytest.raises(RuntimeError, match="pip install faster-whisper"):
        transcribe(tmp_path / "a.m4a", device="cpu")


class _Waveform:
    """Stands in for the decoded float32 array: records the slice it is asked
    for and answers with a range of the same length (numpy semantics, so a
    negative stop counts from the end)."""

    def __init__(self, n_samples: int):
        self.n_samples = n_samples
        self.slices: list[slice] = []

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, item: slice) -> range:
        self.slices.append(item)
        return range(*item.indices(self.n_samples))


def _fake_faster_whisper(monkeypatch, waveform: _Waveform) -> list:
    """A `faster_whisper` module whose decoder returns ``waveform`` and whose
    model only logs what it receives; returns the log."""
    log: list = []
    module = ModuleType("faster_whisper")

    def decode_audio(path, sampling_rate):
        log.append(("decode", path, sampling_rate))
        return waveform

    class WhisperModel:
        def __init__(self, model, device, compute_type):
            log.append(("load", model, device, compute_type))

        def transcribe(self, audio, **kwargs):
            log.append(("transcribe", audio))
            return iter([]), SimpleNamespace(language="pt", duration=0.0)

    module.decode_audio = decode_audio
    module.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return log


def _run_slicer(tmp_path, clip):
    return _faster_whisper_transcriber(tmp_path / "a.m4a", model="small", device="cpu", compute_type="int8",
                                       language="pt", clip=clip)


def test_faster_whisper_transcriber_slices_the_decoded_waveform(tmp_path, monkeypatch):
    waveform = _Waveform(1000 * SAMPLE_RATE)  # a 1000 s file
    log = _fake_faster_whisper(monkeypatch, waveform)

    _run_slicer(tmp_path, clip=(10.0, 30.0))

    assert log[:2] == [("decode", str(tmp_path / "a.m4a"), SAMPLE_RATE), ("load", "small", "cpu", "int8")]
    assert waveform.slices == [slice(10 * SAMPLE_RATE, 30 * SAMPLE_RATE)]
    assert log[2][0] == "transcribe" and len(log[2][1]) == 20 * SAMPLE_RATE  # the 20 s clip reached the model

    log.clear()
    _run_slicer(tmp_path, clip=None)  # no clip: the path goes straight to the model, nothing is decoded here
    assert log == [("load", "small", "cpu", "int8"), ("transcribe", str(tmp_path / "a.m4a"))]


@pytest.mark.parametrize("clip", [(0.0, -285.0), (0.0, 0.0), (10.0, 10.0), (40.0, 30.0)])
def test_faster_whisper_transcriber_refuses_a_clip_that_does_not_end_after_it_starts(tmp_path, monkeypatch, clip):
    # (0.0, -285.0) is what an arc ending 285 s before an --arcs-offset slice used to produce: numpy reads
    # the negative stop as "all but the last 285 s" and the whole file gets transcribed under that arc's name
    waveform = _Waveform(1000 * SAMPLE_RATE)
    log = _fake_faster_whisper(monkeypatch, waveform)

    with pytest.raises(ValueError, match="não é depois do início"):
        _run_slicer(tmp_path, clip=clip)

    assert log == [] and waveform.slices == []  # refused before decoding anything or loading the model


def test_faster_whisper_transcriber_refuses_a_clip_past_the_end_of_the_audio(tmp_path, monkeypatch):
    log = _fake_faster_whisper(monkeypatch, _Waveform(1000 * SAMPLE_RATE))
    with pytest.raises(ValueError, match="depois do fim do áudio"):
        _run_slicer(tmp_path, clip=(2000.0, None))
    assert [entry[0] for entry in log] == ["decode"]  # decoded, found empty, no model loaded


# ── CLI ───────────────────────────────────────────────────────────────


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    audio = tmp_path / "narracao.m4a"
    audio.write_bytes(b"not really audio")
    arcs = tmp_path / "arcos.txt"  # BOM: pasted from Notepad
    arcs.write_text(chr(0xFEFF) + "00:00 - Abertura\n05:30 - Guardião dos Desejos\n01:07:35 - A Era de Ouro\n",
                    encoding="utf-8")
    return audio, arcs


def test_main_writes_the_four_outputs_and_a_summary(tmp_path, capsys):
    audio, _ = _write_inputs(tmp_path)
    fake = FakeTranscriber(duration=30.0)
    out = tmp_path / "out"

    code = main(["--audio", str(audio), "--out-dir", str(out), "--device", "cpu",
                 "--excerpts", "2", "--window", "30"], transcriber=fake)

    assert code == 0
    call = fake.calls[0]
    assert call["clip"] is None and call["model"] == "large-v3-turbo" and call["language"] == "pt"
    assert (call["device"], call["compute_type"]) == ("cpu", "int8")
    assert sorted(p.name for p in out.iterdir()) == ["pacing.json", "segments.json", "style_excerpts.md", "transcript.md"]
    stored = json.loads((out / "segments.json").read_text(encoding="utf-8"))
    assert stored["audio"] == "narracao.m4a" and stored["info"]["duration_s"] == 30.0
    assert [s["text"] for s in stored["segments"]] == ["Guts avança.", "O vento uiva."]
    pacing = json.loads((out / "pacing.json").read_text(encoding="utf-8"))
    assert pacing["words"] == 5 and pacing["total_duration_s"] == 30.0  # the audio duration, not the span
    assert pacing["arc"] is None
    assert "[00:00:00] Guts avança." in (out / "transcript.md").read_text(encoding="utf-8")
    assert "Guts avança." in (out / "style_excerpts.md").read_text(encoding="utf-8")
    summary = capsys.readouterr().out
    assert "palavras/min" in summary and str(out) in summary


def test_main_restricts_to_one_arc_and_writes_into_an_arc_subfolder(tmp_path):
    audio, arcs = _write_inputs(tmp_path)
    fake = FakeTranscriber(duration=600.0)
    out = tmp_path / "out"

    main(["--audio", str(audio), "--out-dir", str(out), "--device", "cpu", "--arcs", str(arcs),
          "--arc", "era de ouro", "--limit-seconds", "600"], transcriber=fake)

    assert fake.calls[0]["clip"] == (4055.0, 4655.0)  # last arc: open-ended, then capped by --limit-seconds
    arc_dir = out / "a_era_de_ouro"
    transcript = (arc_dir / "transcript.md").read_text(encoding="utf-8")
    assert "## A Era de Ouro" in transcript and "[01:07:35] Guts avança." in transcript
    pacing = json.loads((arc_dir / "pacing.json").read_text(encoding="utf-8"))
    assert pacing["arc"] == "A Era de Ouro"


def test_main_limit_seconds_and_arcs_offset_shift_the_clip(tmp_path):
    audio, arcs = _write_inputs(tmp_path)
    fake = FakeTranscriber()
    main(["--audio", str(audio), "--out-dir", str(tmp_path / "out"), "--device", "cpu",
          "--limit-seconds", "60"], transcriber=fake)
    assert fake.calls[-1]["clip"] == (0.0, 60.0)

    # the audio is a slice that starts at 05:30 of the video: arc times shift accordingly
    main(["--audio", str(audio), "--out-dir", str(tmp_path / "out2"), "--device", "cpu",
          "--arcs", str(arcs), "--arcs-offset", "05:30", "--arc", "guardi"], transcriber=fake)
    assert fake.calls[-1]["clip"] == (0.0, 3725.0)


def test_main_refuses_an_arc_that_ends_before_the_slice_starts(tmp_path):
    audio, arcs = _write_inputs(tmp_path)
    fake = FakeTranscriber(duration=1000.0)
    out = tmp_path / "out"
    base = ["--audio", str(audio), "--out-dir", str(out), "--device", "cpu", "--arcs", str(arcs)]

    # the audio is a slice starting at 10:15 of the video: "Abertura" (00:00-05:30) is not in it at all.
    # Shifted, its bounds are (-615, -285); a (0.0, -285.0) clip used to reach the waveform slice as a
    # negative stop index (everything but the last 285 s), silently labelled "Abertura".
    for extra in ([], ["--limit-seconds", "60"]):  # min(60, -285) = -285: the limit did not help either
        with pytest.raises(SystemExit) as excinfo:
            main(base + ["--arcs-offset", "10:15", "--arc", "Abertura"] + extra, transcriber=fake)
        message = str(excinfo.value)
        assert "Abertura" in message and "00:05:30" in message and "00:10:15" in message
    # an arc that ends exactly where the slice starts is just as absent
    with pytest.raises(SystemExit, match="Abertura"):
        main(base + ["--arcs-offset", "05:30", "--arc", "Abertura"], transcriber=fake)
    assert fake.calls == [] and not out.exists()  # nothing transcribed, no arc folder left behind

    # whereas an arc that starts before the slice and ends inside it is transcribed from 00:00
    main(base + ["--arcs-offset", "03:00", "--arc", "Abertura"], transcriber=fake)
    assert fake.calls[-1]["clip"] == (0.0, 150.0)


def test_main_refuses_an_arc_with_no_duration(tmp_path):
    audio, _ = _write_inputs(tmp_path)
    arcs = tmp_path / "duplicados.txt"  # two chapters on the same timestamp: the first one is empty
    arcs.write_text("05:30 - Guardião dos Desejos\n05:30 - O Falcão\n", encoding="utf-8")
    fake = FakeTranscriber()
    base = ["--audio", str(audio), "--out-dir", str(tmp_path / "out"), "--device", "cpu", "--arcs", str(arcs)]

    with pytest.raises(SystemExit, match="mesmo instante"):
        main(base + ["--arc", "Guardi"], transcriber=fake)
    assert fake.calls == []

    main(base + ["--arc", "Falc"], transcriber=fake)  # the last one is open-ended and fine
    assert fake.calls[-1]["clip"] == (330.0, None)


def test_main_rejects_a_non_positive_limit_seconds(tmp_path):
    audio, _ = _write_inputs(tmp_path)
    fake = FakeTranscriber()
    for limit in ("0", "-5", "nan"):  # (0.0, -5.0) would be the same negative stop index
        with pytest.raises(SystemExit):
            main(["--audio", str(audio), "--out-dir", str(tmp_path / "out"), "--device", "cpu",
                  "--limit-seconds", limit], transcriber=fake)
    assert fake.calls == []


def test_main_rejects_a_missing_audio_file_and_arc_without_arcs(tmp_path):
    with pytest.raises(SystemExit):
        main(["--audio", str(tmp_path / "missing.m4a")])
    audio, _ = _write_inputs(tmp_path)
    with pytest.raises(SystemExit):
        main(["--audio", str(audio), "--arc", "Era"])
