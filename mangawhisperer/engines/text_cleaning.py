"""Zero-token OCR hygiene: drop junk before it costs LLM tokens (ticket #10).

EasyOCR on manga emits stray brackets, glyph soup ("Jbl@jw"), lone
symbols and page numbers. Sending those to the scriptwriter wastes
tokens and invites hallucinated dialogue. These checks are
deterministic and deliberately conservative: a false "junk" verdict
loses real dialogue, so anything that plausibly reads as Portuguese
speech — grunts ("Grrrr!"), countdowns ("1... 2... 3!"), a lone "É" —
passes through.
"""

from __future__ import annotations

import re

_EDGE_NOISE = "{}[]|_~^`<>\\/"
_FOREIGN = re.compile(r"[^\w\sÀ-ÿ.,;:!?'\"()\-–—…%$°ºª]", re.UNICODE)  # outside the PT-BR set
_WORD = re.compile(r"[A-Za-zÀ-ÿ]+")
_VOWEL = re.compile(r"[aeiouyAEIOUYáéíóúâêôãõàÁÉÍÓÚÂÊÔÃÕÀ]")
_PUNCT_ONLY_OK = set("?!….")
_SINGLE_LETTER_OK = set("aeoáéóâêôãõàAEOÁÉÓÂÊÔÃÕÀ")  # "É" is a whole utterance


def clean_ocr_text(text: str) -> str:
    """Normalize whitespace and strip bracket/pipe noise at the edges."""
    cleaned = " ".join(text.split())
    return cleaned.strip(_EDGE_NOISE + " ")


def is_ocr_junk(text: str) -> bool:
    """True when the string carries no narratable content.

    Rules (any hit = junk): empty; neither letters nor digits — unless
    it is a short reaction mark like "?!" or "..."; a single character
    that is not a vowel; foreign symbols (@, #, …) inside a short
    token; symbol-heavy strings; or glyph soup — every long word
    lacking a vowel while mixing several different consonants (grunts
    repeat one letter and pass).
    """
    stripped = clean_ocr_text(text)
    if not stripped:
        return True

    letters = sum(char.isalpha() for char in stripped)
    digits = sum(char.isdigit() for char in stripped)
    if letters == 0 and digits == 0:
        return not (len(stripped) <= 3 and set(stripped) <= _PUNCT_ONLY_OK)
    if len(stripped) < 2:
        return stripped not in _SINGLE_LETTER_OK

    foreign = len(_FOREIGN.findall(stripped))
    if foreign and letters <= 6:
        return True  # "Jbl@jw", "S#2"
    if foreign / len(stripped) > 0.2:
        return True

    long_words = [w for w in _WORD.findall(stripped) if len(w) >= 5]
    soup = [w for w in long_words if not _VOWEL.search(w) and len(set(w.lower())) >= 3]
    return bool(long_words) and len(soup) == len(long_words)  # "XKRTP QWRTS"
