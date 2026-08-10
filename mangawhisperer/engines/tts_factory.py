"""TTS engine factory — one place to swap the voice backend.

    create_tts_engine("xtts")    # padrão: XTTSv2 local na GPU (CPML)
    create_tts_engine("edge")    # Microsoft Edge neural pt-BR (nuvem, zero VRAM)
    create_tts_engine("silent")  # placeholder de silêncio (testes/protótipos)
"""

from __future__ import annotations

from mangawhisperer.interfaces import MultiSpeakerTTSEngine

TTS_BACKENDS: tuple[str, ...] = ("xtts", "edge", "silent")


def create_tts_engine(backend: str = "xtts", **kwargs: object) -> MultiSpeakerTTSEngine:
    """Build the TTS engine for ``backend`` (kwargs forwarded)."""
    backend = backend.lower()
    if backend == "xtts":
        from mangawhisperer.engines.tts import XTTSEngine

        return XTTSEngine(**kwargs)
    if backend == "edge":
        from mangawhisperer.engines.tts_edge import EdgeTTSEngine

        return EdgeTTSEngine(**kwargs)
    if backend == "silent":
        from mangawhisperer.engines.placeholders import SilentTTSEngine

        kwargs.pop("speed", None)
        kwargs.pop("extra_synthesis_kwargs", None)
        return SilentTTSEngine(**kwargs)
    raise ValueError(f"Backend de TTS desconhecido {backend!r}; opções: {TTS_BACKENDS}")
