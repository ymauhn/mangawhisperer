"""Tests for the sound-effect library and its pipeline integration."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from mangawhisperer.engines.script_parsing import parse_script_blocks
from mangawhisperer.engines.sfx import TARGET_RATE, SFXLibrary
from mangawhisperer.engines.vlm import build_scriptwriter_prompt


def _write_wav(path: Path, rate: int = 44100, channels: int = 2, duration: float = 0.3) -> None:
    t = np.arange(int(rate * duration)) / rate
    tone = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)
    frames = np.repeat(tone[:, None], channels, axis=1).reshape(-1) if channels > 1 else tone
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames.tobytes())


class TestSFXLibrary:
    def test_tags_come_from_file_stems(self, tmp_path: Path) -> None:
        _write_wav(tmp_path / "espada.wav")
        _write_wav(tmp_path / "Explosao.wav")
        (tmp_path / "notas.txt").write_text("não é áudio")

        assert SFXLibrary(tmp_path).tags() == ["espada", "explosao"]

    def test_missing_directory_is_empty_not_fatal(self, tmp_path: Path) -> None:
        library = SFXLibrary(tmp_path / "nao_existe")
        assert library.tags() == []
        assert library.path_for("espada") is None

    def test_normalizes_stereo_44k_to_pipeline_format(self, tmp_path: Path) -> None:
        _write_wav(tmp_path / "espada.wav", rate=44100, channels=2, duration=0.3)
        library = SFXLibrary(tmp_path)

        normalized = library.path_for("espada")
        assert normalized is not None
        with wave.open(str(normalized), "rb") as wav:
            assert wav.getframerate() == TARGET_RATE
            assert wav.getnchannels() == 1
            assert wav.getsampwidth() == 2
        assert library.duration_ms("espada") == pytest.approx(300, abs=10)

    def test_unknown_tag_returns_none(self, tmp_path: Path) -> None:
        _write_wav(tmp_path / "espada.wav")
        assert SFXLibrary(tmp_path).path_for("laser") is None


class TestSFXInScript:
    def test_parser_carries_sfx_tag(self) -> None:
        raw = json.dumps(
            [{"text": "Guts ataca!", "speaker_id": "Narrator", "is_speech": False, "sfx": "Espada "}]
        )
        (block,) = parse_script_blocks(raw)
        assert block.sfx == "espada"

    def test_parser_defaults_sfx_to_none(self) -> None:
        (block,) = parse_script_blocks('[{"text": "Olá", "speaker_id": "Guts", "is_speech": true}]')
        assert block.sfx is None

    def test_prompt_lists_tags_only_when_library_present(self) -> None:
        with_sfx = build_scriptwriter_prompt(("Guts",), ("espada", "explosao"))
        without = build_scriptwriter_prompt(("Guts",))

        assert "espada, explosao" in with_sfx
        assert "Sound effects" in with_sfx
        assert "Sound effects" not in without
        assert "{sfx_section}" not in without

    def test_prompt_asks_for_descriptive_unknown_labels(self) -> None:
        prompt = build_scriptwriter_prompt(("Guts",))
        assert "Criatura" in prompt
        assert "last resort" in prompt


class TestSFXDictionary:
    def test_dictionary_variants_are_deterministic_by_seed(self, tmp_path: Path) -> None:
        _write_wav(tmp_path / "espada_a.wav")
        _write_wav(tmp_path / "espada_b.wav")
        (tmp_path / "sfx_dictionary.json").write_text(json.dumps({
            "espada": [{"file": "espada_a.wav"}, {"file": "espada_b.wav"}]
        }), encoding="utf-8")
        library = SFXLibrary(tmp_path)

        assert library.tags() == ["espada"]
        first = library.path_for("espada", seed="Guts corta a criatura")
        again = library.path_for("espada", seed="Guts corta a criatura")
        assert first == again, "same seed -> same variant (resume-safe)"

        seeds = [f"cena {i}" for i in range(12)]
        chosen = {library.path_for("espada", seed=s).name for s in seeds}
        assert len(chosen) == 2, "different seeds exercise both variants"

    def test_add_entry_registers_upload_as_variant(self, tmp_path: Path) -> None:
        upload = tmp_path / "upload.wav"
        _write_wav(upload)
        library_dir = tmp_path / "lib"
        library_dir.mkdir()
        library = SFXLibrary(library_dir)

        library.add_entry("chuva", upload, auto_tags=["rain"])
        library.add_entry("chuva", upload)

        assert library.tags() == ["chuva"]
        dictionary = json.loads((library_dir / "sfx_dictionary.json").read_text(encoding="utf-8"))
        assert len(dictionary["chuva"]) == 2
        assert library.path_for("chuva", seed="x") is not None

    def test_broken_dictionary_is_ignored(self, tmp_path: Path) -> None:
        _write_wav(tmp_path / "espada.wav")
        (tmp_path / "sfx_dictionary.json").write_text("{corrompido", encoding="utf-8")

        library = SFXLibrary(tmp_path)
        assert library.tags() == ["espada"]
        assert library.path_for("espada") is not None


class TestIntensityLevels:
    def test_intensity_zero_removes_sfx_section(self) -> None:
        prompt = build_scriptwriter_prompt(("Guts",), ("espada",), sfx_intensity=0)
        assert "sfx" not in prompt.lower()

    def test_levels_have_distinct_wording(self) -> None:
        sparing = build_scriptwriter_prompt(("Guts",), ("espada",), sfx_intensity=1)
        normal = build_scriptwriter_prompt(("Guts",), ("espada",), sfx_intensity=2)
        aggressive = build_scriptwriter_prompt(("Guts",), ("espada",), sfx_intensity=3)

        assert "truly demands" in sparing
        assert "prefer adding an effect" in normal
        assert "Be generous" in aggressive
        assert len({sparing, normal, aggressive}) == 3


class TestSuggestTag:
    AVAILABLE = {"espada", "explosao", "monstro", "fogo", "vento"}

    def test_matches_action_text_to_tag(self) -> None:
        from mangawhisperer.engines.sfx import suggest_tag

        assert suggest_tag("Uma explosão devasta o acampamento", self.AVAILABLE) == "explosao"
        assert suggest_tag("Guts crava a lâmina na garganta", self.AVAILABLE) == "espada"
        assert suggest_tag("A criatura ruge entre as chamas", self.AVAILABLE) == "monstro"

    def test_only_suggests_available_tags(self) -> None:
        from mangawhisperer.engines.sfx import suggest_tag

        assert suggest_tag("Um trovão ecoa ao longe", self.AVAILABLE) is None

    def test_no_match_returns_none(self) -> None:
        from mangawhisperer.engines.sfx import suggest_tag

        assert suggest_tag("Griffith sorri em silêncio", self.AVAILABLE) is None
