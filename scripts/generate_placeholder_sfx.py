"""Gera efeitos sonoros procedurais em assets/sfx/ (placeholders CC0-like).

São efeitos sintéticos simples (estilo radionovela/8-bit refinado) para a
sonoplastia funcionar imediatamente. Para qualidade profissional, basta
substituir/adicionar arquivos .wav/.mp3 na mesma pasta — o nome do
arquivo é a tag que o modelo enxerga (ex.: `espada.wav` -> tag `espada`).

    python scripts/generate_placeholder_sfx.py
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

RATE = 24000
SFX_DIR = Path(__file__).resolve().parents[1] / "assets" / "sfx"

rng = np.random.default_rng(42)  # determinístico: mesmos sons a cada geração


def _t(duration: float) -> np.ndarray:
    return np.arange(int(RATE * duration)) / RATE


def _decay(t: np.ndarray, rate: float) -> np.ndarray:
    return np.exp(-rate * t)


def espada() -> np.ndarray:
    """Aço contra aço: transiente + ressonâncias metálicas agudas."""
    t = _t(0.45)
    ring = sum(
        amp * np.sin(2 * np.pi * freq * t) * _decay(t, dec)
        for freq, amp, dec in ((2500, 0.5, 9), (3700, 0.35, 11), (5200, 0.25, 14), (6800, 0.15, 18))
    )
    click = rng.normal(0, 1, len(t)) * _decay(t, 90) * 0.8
    return ring + click


def soco() -> np.ndarray:
    """Impacto surdo: sub-grave com queda de pitch + estalo."""
    t = _t(0.28)
    pitch = 110 * np.exp(-8 * t) + 45
    body = np.sin(2 * np.pi * np.cumsum(pitch) / RATE) * _decay(t, 12)
    snap = rng.normal(0, 1, len(t)) * _decay(t, 60) * 0.5
    return body + snap


def explosao() -> np.ndarray:
    """Estrondo: ruído grave filtrado + sub-boom longo."""
    t = _t(1.4)
    noise = rng.normal(0, 1, len(t))
    kernel = np.ones(96) / 96  # passa-baixa rústico
    rumble = np.convolve(noise, kernel, mode="same") * _decay(t, 2.8)
    boom = np.sin(2 * np.pi * (55 * np.exp(-1.5 * t) + 28) * t) * _decay(t, 3.5) * 0.9
    return rumble + boom


def monstro() -> np.ndarray:
    """Rugido gutural: harmônicos graves com vibrato + soprosidade."""
    t = _t(1.0)
    vibrato = 1 + 0.04 * np.sin(2 * np.pi * 6 * t)
    growl = sum(
        amp * np.sin(2 * np.pi * harm * 82 * vibrato * t)
        for harm, amp in ((1, 0.6), (2, 0.45), (3, 0.3), (5, 0.15))
    )
    breath = np.convolve(rng.normal(0, 1, len(t)), np.ones(24) / 24, mode="same") * 0.35
    envelope = np.minimum(t / 0.12, 1.0) * _decay(t, 1.6)
    return (growl + breath) * envelope


def vento() -> np.ndarray:
    """Rajada: ruído suavizado com respiração lenta."""
    t = _t(1.8)
    noise = np.convolve(rng.normal(0, 1, len(t)), np.ones(160) / 160, mode="same")
    swell = 0.5 * (1 + np.sin(2 * np.pi * 0.6 * t - np.pi / 2))
    return noise * swell * _decay(t, 0.7)


def fogo() -> np.ndarray:
    """Crepitar: cama de ruído + estalos esparsos."""
    t = _t(1.5)
    bed = np.convolve(rng.normal(0, 1, len(t)), np.ones(48) / 48, mode="same") * 0.35
    crackle = np.zeros(len(t))
    for pos in rng.integers(0, len(t) - 600, size=26):
        burst = rng.normal(0, 1, 600) * _decay(_t(0.025), 160)
        crackle[pos : pos + 600] += burst * rng.uniform(0.3, 1.0)
    return (bed + crackle) * _decay(t, 0.8)


EFFECTS = {
    "espada": espada,
    "soco": soco,
    "explosao": explosao,
    "monstro": monstro,
    "vento": vento,
    "fogo": fogo,
}


def main() -> None:
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    for tag, build in EFFECTS.items():
        samples = build()
        peak = float(np.max(np.abs(samples))) or 1.0
        pcm = (np.clip(samples / peak * 0.85, -1, 1) * 32767).astype(np.int16)
        path = SFX_DIR / f"{tag}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(pcm.tobytes())
        print(f"  {tag}.wav ({len(pcm) / RATE:.2f}s)")
    print(f"\n{len(EFFECTS)} efeitos gerados em {SFX_DIR}")


if __name__ == "__main__":
    main()
