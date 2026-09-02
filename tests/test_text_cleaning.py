"""Deterministic OCR junk filter (ticket #10) — zero tokens spent on noise,
and no dialogue lost to it."""

import pytest

from mangawhisperer.engines.text_cleaning import clean_ocr_text, is_ocr_junk


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "{",
        "@",
        "|",
        "x",  # a lone consonant is noise ("É" is not — see below)
        "Jbl@jw",  # EasyOCR glyph soup with a foreign symbol
        "S#2",
        "XKRTP QWRTS",  # long words without vowels mixing several consonants
        "}{[]",
        "= = =",
    ],
)
def test_junk_is_rejected(text):
    assert is_ocr_junk(text)


@pytest.mark.parametrize(
    "text",
    [
        "Você está bem?",
        "Guts!",
        "?!",  # reaction mark — narratable
        "...",
        "Não.",
        "AAAH",
        "Hm",
        "Grrr",
        "Hmph",
        "Grrrrr!",  # grunts repeat one letter: not glyph soup
        "Grrrr! Você!",
        "Hmmmm, sim",
        "Shhhh",
        "Zzzzz",
        "Zodd... é você?",
        "Ei, espera aí — o que é isso?!",
        "Ryu",
        "É",  # a whole utterance in PT-BR
        "1... 2... 3!",  # countdown
        "10!",
        "100% certo",
        "R$ 50",
        "30°",
    ],
)
def test_dialogue_passes(text):
    assert not is_ocr_junk(text)


def test_clean_strips_bracket_noise_and_normalizes_whitespace():
    assert clean_ocr_text("{Guts|") == "Guts"
    assert clean_ocr_text("  olá \n  mundo ") == "olá mundo"
    assert clean_ocr_text("[Vamos!]") == "Vamos!"


def test_clean_keeps_inner_punctuation():
    assert clean_ocr_text("Não... eu não!") == "Não... eu não!"
