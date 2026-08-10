"""Gera trilhas de fundo (BGM) procedurais loopáveis em assets/bgm/.

Placeholders dark-fantasy para validar o mixer hoje; substitua depois
por loops CC0 (OpenGameArt: "Loopable Dungeon Ambience", "Dark Shrine
Loop"; Pixabay Music) — basta soltar os arquivos na mesma pasta.

    python scripts/generate_placeholder_bgm.py
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

RATE = 24000
BGM_DIR = Path(__file__).resolve().parents[1] / "assets" / "bgm"

rng = np.random.default_rng(7)


def _t(duration: float) -> np.ndarray:
    return np.arange(int(RATE * duration)) / RATE


def _loopable(samples: np.ndarray, crossfade_s: float = 1.0) -> np.ndarray:
    """Make the tail crossfade into the head so the loop seam is smooth."""
    n = int(RATE * crossfade_s)
    if n * 2 >= len(samples):
        return samples
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    head = samples[:n].copy()
    body = samples[n:].copy()
    body[-n:] = body[-n:] * (1 - ramp) + head * ramp
    return body


def ambiente_sombrio() -> np.ndarray:
    """Drone grave e lento — cavernas, ruínas, presságio."""
    t = _t(18.0)
    drone = (
        0.5 * np.sin(2 * np.pi * 55 * t)
        + 0.3 * np.sin(2 * np.pi * 82.5 * t + 0.5)
        + 0.2 * np.sin(2 * np.pi * 110 * t + 1.1)
    )
    breath = np.convolve(rng.normal(0, 1, len(t)), np.ones(400) / 400, mode="same") * 0.12
    slow_lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.05 * t)
    return (drone * slow_lfo + breath) * 0.5


def tensao() -> np.ndarray:
    """Segunda menor pulsante — suspense crescente."""
    t = _t(14.0)
    dissonance = 0.45 * np.sin(2 * np.pi * 110 * t) + 0.45 * np.sin(2 * np.pi * 116.5 * t)
    tremolo = 0.65 + 0.35 * np.sin(2 * np.pi * 1.5 * t)
    shimmer = 0.08 * np.sin(2 * np.pi * 880 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.21 * t))
    return (dissonance * tremolo + shimmer) * 0.5


def batalha() -> np.ndarray:
    """Pulso percussivo grave a ~92 bpm sob um drone — combate."""
    t = _t(15.652)  # 24 batidas a 92 bpm fecham o loop
    beat_period = 60.0 / 92.0
    phase = (t % beat_period) / beat_period
    thump = np.sin(2 * np.pi * 60 * t) * np.exp(-phase * 18) * 0.8
    accent = np.sin(2 * np.pi * 45 * t) * np.exp(-((t % (beat_period * 4)) / beat_period) * 10) * 0.5
    drone = 0.25 * np.sin(2 * np.pi * 73.4 * t)
    return (thump + accent + drone) * 0.55


TRACKS = {"ambiente_sombrio": ambiente_sombrio, "tensao": tensao, "batalha": batalha}


def main() -> None:
    BGM_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in TRACKS.items():
        samples = _loopable(build().astype(np.float32))
        peak = float(np.max(np.abs(samples))) or 1.0
        pcm = (np.clip(samples / peak * 0.8, -1, 1) * 32767).astype(np.int16)
        path = BGM_DIR / f"{name}.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(pcm.tobytes())
        print(f"  {name}.wav ({len(pcm) / RATE:.1f}s, loopável)")
    print(f"\n{len(TRACKS)} trilhas geradas em {BGM_DIR}")


if __name__ == "__main__":
    main()
