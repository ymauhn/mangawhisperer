"""Voice casting shared by every TTS engine: profile-aware banks and a
per-volume registry so a character keeps the same voice forever.

The scriptwriter declares a *voice profile* per speaker as drawn
(``homem``, ``mulher``, ``idoso``, ``idosa``, ``menino``, ``menina``,
``criatura``). The registry turns speakers into concrete voices once —
curated cast first, then an unused voice from the profile's bank,
deterministically — and persists the choice in the workspace
(``cast_voices.json``), so the "Sacerdote" of page 31 sounds the same on
page 200 and on next month's re-run, and never gets a woman's voice.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Generic, Iterable, Mapping, Sequence, TypeVar

logger = logging.getLogger(__name__)

Voice = TypeVar("Voice")


def _stable_index(speaker_id: str, size: int) -> int:
    digest = hashlib.sha1(speaker_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "big") % max(1, size)


class VoiceRegistry(Generic[Voice]):
    """Speaker -> voice assignments, profile-aware and persistent.

    ``voices`` must be JSON-serialisable (a string for XTTS, a
    ``[voice, rate, pitch]`` list for Edge); ``key`` turns a voice into
    something hashable for the "already in use" check.
    """

    def __init__(
        self,
        cast: Mapping[str, Voice],
        bank: Mapping[str, Sequence[Voice]],
        fallback: Sequence[Voice],
        key=lambda voice: json.dumps(voice, sort_keys=True) if not isinstance(voice, str) else voice,
    ) -> None:
        if not fallback:
            raise ValueError("fallback voices must not be empty")
        self._cast = dict(cast)
        self._bank = {profile: tuple(voices) for profile, voices in bank.items() if voices}
        self._fallback = tuple(fallback)
        self._key = key
        self._assignments: dict[str, Voice] = {}
        self._path: Path | None = None

    @property
    def assignments(self) -> dict[str, Voice]:
        return dict(self._assignments)

    def load(self, path: Path) -> None:
        """Adopt the assignments persisted for a workspace (if any)."""
        self._path = path
        self._assignments = {}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._assignments = {str(k): self._from_json(v) for k, v in data.get("voices", {}).items()}
            except (OSError, ValueError, TypeError) as exc:
                logger.warning("Registro de vozes ilegível (%s): recomeçando", exc)

    def assign(self, profiles: Mapping[str, str | None], path: Path | None = None) -> dict[str, Voice]:
        """Give every speaker a voice (keeping existing ones), persist, return the map."""
        if path is not None and path != self._path:
            self.load(path)
        for speaker in sorted(profiles):
            if speaker not in self._assignments:
                self._assignments[speaker] = self._pick(speaker, profiles.get(speaker))
        if self._path is not None:
            self.save()
        return self.assignments

    def voice_for(self, speaker_id: str, profile: str | None = None) -> Voice:
        """The assigned voice, else an on-the-fly deterministic pick."""
        if speaker_id in self._assignments:
            return self._assignments[speaker_id]
        return self._pick(speaker_id, profile)

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"voices": {k: self._to_json(v) for k, v in sorted(self._assignments.items())}}
        self._path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def digest(self) -> str:
        """Short identity of the current assignments (for fingerprints)."""
        blob = json.dumps({k: self._to_json(v) for k, v in sorted(self._assignments.items())}, sort_keys=True)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]

    def _pick(self, speaker_id: str, profile: str | None) -> Voice:
        if speaker_id in self._cast:
            return self._cast[speaker_id]
        pool = self._bank.get(profile or "", self._fallback)
        used = {self._key(v) for v in self._assignments.values()}
        start = _stable_index(speaker_id, len(pool))
        for offset in range(len(pool)):  # prefer a voice nobody else has yet
            candidate = pool[(start + offset) % len(pool)]
            if self._key(candidate) not in used:
                return candidate
        return pool[start]

    @staticmethod
    def _to_json(voice: Any) -> Any:
        return list(voice) if isinstance(voice, tuple) else voice

    @staticmethod
    def _from_json(value: Any) -> Any:
        return tuple(value) if isinstance(value, list) else value


def majority_profile(votes: Iterable[str | None]) -> str | None:
    """Most frequent non-null vote; ties go to the first seen."""
    counts: dict[str, int] = {}
    order: list[str] = []
    for vote in votes:
        if vote:
            if vote not in counts:
                order.append(vote)
            counts[vote] = counts.get(vote, 0) + 1
    if not counts:
        return None
    return max(order, key=lambda v: (counts[v], -order.index(v)))
