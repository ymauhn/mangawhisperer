"""Baixa packs de efeitos sonoros CC0 e monta a biblioteca de sonoplastia.

Fontes: OpenGameArt + Kenney — todas CC0 (domínio público), URLs diretas
verificadas, sem login. Total ~23 MB. Os packs são extraídos em
``assets/sfx/_packs/`` e o melhor arquivo de cada categoria é promovido a
tag da biblioteca (``assets/sfx/<tag>.wav``), substituindo os placeholders
procedurais.

    python scripts/download_sfx.py
"""

from __future__ import annotations

import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

SFX_DIR = Path(__file__).resolve().parents[1] / "assets" / "sfx"
PACKS_DIR = SFX_DIR / "_packs"
USER_AGENT = {"User-Agent": "Mozilla/5.0 (MangaWhisperer SFX fetcher)"}

# (nome, url, página de fallback para re-descobrir o zip se a URL mudar)
PACKS: list[tuple[str, str, str | None]] = [
    ("swords_attack", "https://opengameart.org/sites/default/files/sword_-_starninjas_1.zip", None),
    ("swords_clash", "https://opengameart.org/sites/default/files/sword_clash_-_starninjas_0.zip", None),
    ("rpg_sound_pack", "https://opengameart.org/sites/default/files/rpg_sound_pack.zip", None),
    ("creature_sfx", "https://opengameart.org/sites/default/files/80-CC0-creature-SFX_0.zip", None),
    ("rpg_sfx", "https://opengameart.org/sites/default/files/80-CC0-RPG-SFX_0.zip", None),
    ("bang_sfx", "https://opengameart.org/sites/default/files/25-CC0-bang-sfx.zip", None),
    ("sfx_100_v2", "https://opengameart.org/sites/default/files/sfx_100_v2.zip", None),
    (
        "kenney_impact",
        "https://kenney.nl/media/pages/assets/impact-sounds/87b4ddecda-1677589768/kenney_impact-sounds.zip",
        "https://kenney.nl/assets/impact-sounds",  # o hash do CDN pode mudar
    ),
]

# tag -> palavras-chave em ordem de prioridade (case-insensitive)
TAG_RULES: dict[str, list[str]] = {
    "espada": ["sword_clash", "clash", "sword", "blade", "swing"],
    "explosao": ["explosion", "bang", "boom", "cannon"],
    "soco": ["punch", "impact", "slam", "hit"],
    "monstro": ["roar", "growl", "monster", "beast", "ogre", "giant"],
    "fogo": ["fire", "flame", "burn"],
    "vento": ["wind", "air", "whoosh"],
    "grito": ["scream", "shout", "yell"],
    "passos": ["footstep", "foot_step", "steps", "walk"],
    "trovao": ["thunder"],
    "porta": ["door"],
}


def download(name: str, url: str, fallback_page: str | None) -> Path | None:
    target = PACKS_DIR / f"{name}.zip"
    if target.is_file():
        print(f"  [{name}] já baixado")
        return target
    for attempt_url in filter(None, (url, _rediscover(fallback_page))):
        try:
            print(f"  [{name}] baixando {attempt_url}")
            request = urllib.request.Request(attempt_url, headers=USER_AGENT)
            with urllib.request.urlopen(request, timeout=120) as response, open(target, "wb") as out:
                shutil.copyfileobj(response, out)
            return target
        except Exception as exc:
            print(f"  [{name}] falhou: {exc}")
    return None


def _rediscover(page: str | None) -> str | None:
    if not page:
        return None
    try:
        request = urllib.request.Request(page, headers=USER_AGENT)
        html = urllib.request.urlopen(request, timeout=60).read().decode("utf-8", "replace")
        match = re.search(r'href="(https://kenney\.nl/media/[^"]+\.zip)"', html)
        return match.group(1) if match else None
    except Exception:
        return None


def extract(archive: Path) -> None:
    destination = PACKS_DIR / archive.stem
    if destination.is_dir():
        return
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)


def promote_tags() -> dict[str, Path]:
    """Pick the best candidate file for each tag from the extracted packs."""
    candidates = [
        p for p in PACKS_DIR.rglob("*")
        if p.suffix.lower() in (".wav", ".ogg", ".mp3") and p.stat().st_size > 4096
    ]
    chosen: dict[str, Path] = {}
    for tag, keywords in TAG_RULES.items():
        best: tuple[int, int, int, Path] | None = None  # (kw_rank, not_wav, -size)
        for path in candidates:
            haystack = path.name.lower().replace("-", "_")
            for rank, keyword in enumerate(keywords):
                if keyword in haystack:
                    key = (rank, 0 if path.suffix.lower() == ".wav" else 1, -path.stat().st_size, path)
                    if best is None or key < best:
                        best = key
                    break
        if best is not None:
            source = best[3]
            for stale in SFX_DIR.glob(f"{tag}.*"):
                stale.unlink()
            target = SFX_DIR / f"{tag}{source.suffix.lower()}"
            shutil.copyfile(source, target)
            chosen[tag] = source
            print(f"  {tag:<10} <- {source.name}")
    return chosen


def main() -> int:
    SFX_DIR.mkdir(parents=True, exist_ok=True)
    PACKS_DIR.mkdir(parents=True, exist_ok=True)

    print("Baixando packs CC0 (~23 MB no total)...")
    archives = [a for name, url, page in PACKS if (a := download(name, url, page))]
    print(f"\n{len(archives)}/{len(PACKS)} packs disponíveis. Extraindo...")
    for archive in archives:
        extract(archive)

    print("\nPromovendo os melhores arquivos para tags da biblioteca:")
    chosen = promote_tags()

    normalized = SFX_DIR / "_normalized"
    if normalized.is_dir():
        shutil.rmtree(normalized)  # força renormalização dos novos arquivos

    from mangawhisperer.engines.sfx import SFXLibrary  # noqa: PLC0415

    tags = SFXLibrary(SFX_DIR).tags()
    print(f"\nBiblioteca final ({len(tags)} tags): {', '.join(tags)}")
    if not chosen:
        print("Nenhum arquivo promovido — os placeholders procedurais continuam valendo.")
    return 0


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
