"""Voice profiles end to end: declared by the scriptwriter, consolidated per
speaker, cast from profile-aware banks and pinned per volume."""

import json
from pathlib import Path

import pytest

from mangawhisperer.engines.casting import VoiceRegistry, majority_profile
from mangawhisperer.engines.narration import consolidate_voice_profiles
from mangawhisperer.engines.script_parsing import parse_script_blocks
from mangawhisperer.engines.styles import get_style, style_examples_addendum
from mangawhisperer.engines.tts import DEFAULT_CAST_VOICES, FALLBACK_VOICE_POOL, VOICE_BANK, XTTSEngine
from mangawhisperer.engines.tts_edge import EDGE_VOICE_BANK, EdgeTTSEngine
from mangawhisperer.engines.vlm import build_scriptwriter_prompt
from mangawhisperer.engines.vlm_llamacpp import build_panel_schema
from mangawhisperer.models import VOICE_PROFILES, BoundingBox, ContextualizedBlock, PanelData


def test_block_voice_is_normalized_and_lenient():
    assert ContextualizedBlock(text="x", speaker_id="a", is_speech=True, voice=" Homem ").voice == "homem"
    assert ContextualizedBlock(text="x", speaker_id="a", is_speech=True, voice="robô").voice is None
    assert ContextualizedBlock(text="x", speaker_id="a", is_speech=True).voice is None
    assert "criatura" in VOICE_PROFILES


def test_parser_keeps_the_voice_field():
    raw = '[{"text": "Já chega!", "speaker_id": "Sacerdote", "is_speech": true, "voice": "idoso"},'
    raw += ' {"text": "...", "speaker_id": "Narrator", "is_speech": false}]'
    blocks = parse_script_blocks(raw)
    assert [b.voice for b in blocks] == ["idoso", None]


def test_prompt_and_schema_carry_the_sober_rules_and_the_voice_profile():
    prompt = build_scriptwriter_prompt(("Guts",), ("espada",), 2, "")
    assert "Describe ONLY what is drawn" in prompt and "Never" in prompt
    assert '"voice"' in prompt and "homem, mulher, idoso, idosa, menino, menina" in prompt
    assert "6. Sound effects" in prompt and "5. Voice profile" in prompt
    schema = build_panel_schema(("espada",))
    assert schema["items"]["properties"]["voice"]["enum"] == list(VOICE_PROFILES)
    assert "voice" not in schema["items"]["required"]


def test_majority_profile_and_consolidation_across_script_versions():
    assert majority_profile(["homem", None, "idoso", "idoso"]) == "idoso"
    assert majority_profile(["homem", "idoso"]) == "homem"  # tie: first seen
    assert majority_profile([None, None]) is None

    def panel(blocks):
        return PanelData(image_path=Path("p.npy"), bbox=BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1),
                         page_number=1, panel_index=0, blocks=blocks)

    reviewed = [panel([ContextualizedBlock(text="Já chega!", speaker_id="Sacerdote", is_speech=True)])]  # field dropped
    raw = [panel([
        ContextualizedBlock(text="Já chega!", speaker_id="Sacerdote", is_speech=True, voice="idoso"),
        ContextualizedBlock(text="Sim", speaker_id="Soldado", is_speech=True, voice="homem"),
        ContextualizedBlock(text="cena", speaker_id="Narrator", is_speech=False),
    ])]
    assert consolidate_voice_profiles(reviewed, raw) == {"Sacerdote": "idoso", "Soldado": "homem"}


def test_registry_prefers_cast_then_unused_bank_voices_and_persists(tmp_path):
    registry = VoiceRegistry(cast={"Guts": "Craig Gutsy"}, bank={"homem": ("A", "B"), "mulher": ("C",)}, fallback=("Z",))
    path = tmp_path / "cast_voices.json"

    assigned = registry.assign({"Guts": "homem", "Sacerdote": "idoso", "Aldeã": "mulher", "Soldado": "homem", "Recruta": "homem"}, path)

    assert assigned["Guts"] == "Craig Gutsy"
    assert assigned["Aldeã"] == "C"
    assert assigned["Sacerdote"] == "Z"  # no bank for idoso here: fallback
    assert {assigned["Soldado"], assigned["Recruta"]} == {"A", "B"}  # distinct voices while the bank lasts
    assert json.loads(path.read_text(encoding="utf-8"))["voices"]["Aldeã"] == "C"

    reopened = VoiceRegistry(cast={}, bank={"homem": ("B", "A")}, fallback=("Z",))
    reopened.load(path)
    assert reopened.voice_for("Soldado") == assigned["Soldado"]  # pinned, whatever the bank order now
    assert reopened.assign({"Soldado": "homem", "Novo": "homem"}, path)["Soldado"] == assigned["Soldado"]
    assert registry.digest() != VoiceRegistry(cast={}, bank={}, fallback=("Z",)).digest()


def test_registry_is_deterministic_without_a_file():
    a = VoiceRegistry(cast={}, bank={"mulher": ("C", "D", "E")}, fallback=("Z",))
    b = VoiceRegistry(cast={}, bank={"mulher": ("C", "D", "E")}, fallback=("Z",))
    assert a.voice_for("Aldeã", "mulher") == b.voice_for("Aldeã", "mulher") in ("C", "D", "E")
    with pytest.raises(ValueError):
        VoiceRegistry(cast={}, bank={}, fallback=())


def test_xtts_casts_by_profile_and_never_gives_a_priest_a_female_voice(tmp_path):
    engine = XTTSEngine(synthesizer=object())
    assigned = engine.assign_voices({"Sacerdote": "idoso", "Aldeã": "mulher", "Guts": "homem"}, tmp_path / "cast_voices.json")
    assert assigned["Guts"] == DEFAULT_CAST_VOICES["Guts"]
    assert assigned["Sacerdote"] in VOICE_BANK["idoso"]
    assert assigned["Aldeã"] in VOICE_BANK["mulher"]
    assert engine.voice_for("Sacerdote") == assigned["Sacerdote"]  # pinned even without the profile
    assert engine.voice_for("Soldado Raso") in FALLBACK_VOICE_POOL  # unknown profile: legacy pool
    assert engine.voice_for("Menina da vila", "menina") in VOICE_BANK["menina"]
    assert "cast=" in engine.fingerprint
    male = set(VOICE_BANK["homem"]) | set(VOICE_BANK["idoso"]) | set(VOICE_BANK["menino"]) | set(VOICE_BANK["criatura"])
    female = set(VOICE_BANK["mulher"]) | set(VOICE_BANK["idosa"]) | set(VOICE_BANK["menina"])
    assert not (male & female)


def test_edge_casts_by_profile_with_offsets():
    engine = EdgeTTSEngine(communicate_factory=lambda *a, **k: None)
    assert engine.voice_for("Sacerdote", "idoso") in EDGE_VOICE_BANK["idoso"]
    assert engine.voice_for("Aldeã", "mulher")[0] in ("pt-BR-FranciscaNeural", "pt-BR-ThalitaMultilingualNeural")
    assigned = engine.assign_voices({"Sacerdote": "idoso"})
    assert assigned["Sacerdote"][0] == "pt-BR-AntonioNeural"


def test_sober_style_and_examples_addendum():
    sober = get_style("sobrio")
    assert "SÓBRIO" in sober.prompt_addendum and sober.suggested_bgm is None
    addendum = style_examples_addendum("Guts avança. Silêncio.\nPuck voa.")
    assert addendum.startswith("\n\nTone reference") and "do NOT copy" in addendum and "Puck voa." in addendum
    assert style_examples_addendum("   ") == ""
    long = style_examples_addendum("linha\n" * 5000, max_chars=100)
    assert long.endswith("[...]") and len(long) < 400
