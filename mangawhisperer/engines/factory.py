"""Engine factory: one place to resolve a provider name into a VLM engine.

Providers:
    ``qwen`` / ``openai`` / ``kimi``  -> OpenAI-compatible API engine
    ``anthropic`` (alias ``claude``)  -> native Claude engine
    ``qwen-local`` (alias ``local``)  -> local Qwen-VL on the GPU
    ``passthrough``                   -> offline no-model stub
    ``auto``                          -> first provider with credentials:
                                         Qwen API > Anthropic > OpenAI >
                                         Kimi > local Qwen-VL
"""

from __future__ import annotations

import logging
import os

from mangawhisperer.engines.placeholders import PassthroughVLM
from mangawhisperer.engines.vlm_api import PROVIDER_PRESETS
from mangawhisperer.interfaces import VisionLanguageEngine

logger = logging.getLogger(__name__)

API_PROVIDERS: tuple[str, ...] = tuple(PROVIDER_PRESETS)
ALL_PROVIDERS: tuple[str, ...] = (*API_PROVIDERS, "anthropic", "qwen-local", "passthrough")

_AUTO_ORDER: tuple[tuple[str, str], ...] = (
    ("qwen", "DASHSCOPE_API_KEY"),
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
    ("kimi", "MOONSHOT_API_KEY"),
)


def create_vlm_engine(
    provider: str, model: str | None = None, **kwargs: object
) -> VisionLanguageEngine:
    """Build the scriptwriter engine for ``provider``.

    Args:
        provider: One of :data:`ALL_PROVIDERS`, or ``auto``.
        model: Optional model id override (provider-specific).
        kwargs: Passed through to the engine constructor.
    """
    provider = provider.lower()
    if provider == "auto":
        provider = _resolve_auto()

    if provider in PROVIDER_PRESETS:
        from mangawhisperer.engines.vlm_api import OpenAICompatibleVisionLanguageEngine

        return OpenAICompatibleVisionLanguageEngine(provider=provider, model=model, **kwargs)
    if provider in ("anthropic", "claude"):
        from mangawhisperer.engines.vlm import ClaudeVisionLanguageEngine

        if model is not None:
            kwargs["model"] = model
        return ClaudeVisionLanguageEngine(**kwargs)
    if provider in ("qwen-local", "local"):
        from mangawhisperer.engines.vlm_local import QwenVisionLanguageEngine

        if model is not None:
            kwargs["model_name"] = model
        return QwenVisionLanguageEngine(**kwargs)
    if provider == "passthrough":
        return PassthroughVLM()
    raise ValueError(f"Unknown VLM provider {provider!r}; expected one of {ALL_PROVIDERS + ('auto',)}")


def create_reviewer(provider: str, model: str | None = None, **kwargs: object):
    """Build the script-reviewer engine, or ``None`` when the provider
    can't support it (local/passthrough runs review nothing)."""
    provider = provider.lower()
    if provider == "auto":
        provider = _resolve_auto()
    if provider in ("qwen-local", "local", "passthrough"):
        return None
    from mangawhisperer.engines.reviewer import LLMScriptReviewer

    return LLMScriptReviewer(provider=provider, model=model, **kwargs)


def _resolve_auto() -> str:
    for provider, env_var in _AUTO_ORDER:
        if os.environ.get(env_var):
            logger.info("VLM provider 'auto' -> '%s' (%s is set)", provider, env_var)
            return provider
    logger.info("VLM provider 'auto' -> 'qwen-local' (no API key found)")
    return "qwen-local"
