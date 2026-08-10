"""GPU/CUDA diagnostics shared by the CLI entrypoints and the web UI."""

from __future__ import annotations


def cuda_report() -> str:
    """One-line human-readable summary of local GPU acceleration."""
    try:
        import torch  # noqa: PLC0415 — keep module import light
    except ImportError:
        return "torch não instalado — estágios locais rodarão em CPU"
    if not torch.cuda.is_available():
        build = "+cpu" if "+cpu" in torch.__version__ else ""
        hint = (
            " (build CPU-only instalada — reinstale com "
            "--index-url https://download.pytorch.org/whl/cu128)"
            if build
            else ""
        )
        return f"CUDA indisponível — torch {torch.__version__}{hint}"
    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return f"CUDA ativa: {name} ({vram_gb:.0f} GB) — torch {torch.__version__}"
