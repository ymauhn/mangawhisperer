"""Tests for the CLAP-based SFX auto-tagger (fake classifier — offline)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from mangawhisperer.engines.audio_tagging import LABEL_TO_TAG, ClapAudioTagger
from mangawhisperer.engines.sfx import TAG_KEYWORDS


def _write_wav(path: Path, rate: int = 44100) -> None:
    tone = (np.sin(np.arange(rate) / rate * 2 * np.pi * 220) * 15000).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(tone.tobytes())


class RecordingClassifier:
    def __init__(self, results) -> None:
        self.results = results
        self.calls: list[dict] = []

    def __call__(self, audio, candidate_labels, hypothesis_template):
        self.calls.append({
            "samples": len(audio), "labels": candidate_labels, "template": hypothesis_template,
        })
        return self.results


class TestClapAudioTagger:
    def test_maps_english_labels_to_pt_tags(self, tmp_path: Path) -> None:
        _write_wav(tmp_path / "boom.wav")
        classifier = RecordingClassifier([
            {"label": "explosion", "score": 0.82},
            {"label": "thunder", "score": 0.41},
            {"label": "rain falling", "score": 0.05},  # below threshold
        ])
        tagger = ClapAudioTagger(classifier=classifier)

        suggestions = tagger.suggest(tmp_path / "boom.wav")
        assert [(tag, round(score, 2)) for tag, _lbl, score in suggestions] == [
            ("explosao", 0.82), ("trovao", 0.41),
        ]

    def test_resamples_input_to_48k(self, tmp_path: Path) -> None:
        _write_wav(tmp_path / "boom.wav", rate=44100)  # 1s of audio
        classifier = RecordingClassifier([{"label": "explosion", "score": 0.9}])
        ClapAudioTagger(classifier=classifier).suggest(tmp_path / "boom.wav")

        assert classifier.calls[0]["samples"] == pytest.approx(48000, abs=5)
        assert set(classifier.calls[0]["labels"]) == set(LABEL_TO_TAG)

    def test_dedupes_labels_pointing_to_the_same_tag(self, tmp_path: Path) -> None:
        _write_wav(tmp_path / "clang.wav")
        classifier = RecordingClassifier([
            {"label": "sword clash", "score": 0.7},
            {"label": "sword slash", "score": 0.6},  # same tag: espada
        ])
        suggestions = ClapAudioTagger(classifier=classifier).suggest(tmp_path / "clang.wav")
        assert [s[0] for s in suggestions] == ["espada"]

    def test_release_without_load_is_safe(self) -> None:
        ClapAudioTagger().release()  # no model was loaded; must not raise


def test_label_map_covers_every_keyword_tag() -> None:
    """Every tag the keyword fallback knows must be reachable by CLAP too."""
    clap_tags = set(LABEL_TO_TAG.values())
    assert set(TAG_KEYWORDS).issubset(clap_tags)
