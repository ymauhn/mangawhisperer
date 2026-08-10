"""End-to-end smoke of the CLI entrypoint — offline, real wiring.

Executes ``main_demo.main()`` for real (argparse -> style -> sfx/bgm ->
engines -> orchestrator -> final wav) with the offline backends
(passthrough VLM, silent TTS). This is the regression net for
initialization-order bugs that ``--help`` checks can never catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("pymupdf")

import main_demo  # noqa: E402


@pytest.mark.skipif(not main_demo.DEFAULT_PDF.is_file(), reason="sample PDF not present")
def test_main_demo_runs_end_to_end_offline(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", [
        "main_demo.py",
        "--vlm", "passthrough",       # sem API
        "--tts", "silent",            # sem GPU/modelo de voz
        "--style", "sombrio",         # exercita defaults do estilo (incl. BGM sugerida)
        "--bgm", "off",               # e o override explícito por cima
        "--pages", "1", "--start", "8",
        "--no-review",
        "--workspace", str(tmp_path / "ws"),
    ])

    assert main_demo.main() == 0

    out = capsys.readouterr().out
    assert "[STY] estilo: Sombrio" in out
    assert "[BGM] trilha de fundo: desligada" in out
    final = tmp_path / "ws" / "berserk_vol_01" / "final" / "berserk_vol_01.wav"
    assert final.is_file() and final.stat().st_size > 0
