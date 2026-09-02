"""Engine factory: one place to resolve a provider name into a VLM engine.

Providers:
    ``qwen`` / ``openai`` / ``kimi``  -> OpenAI-compatible API engine
    ``anthropic`` (alias ``claude``)  -> native Claude engine
    ``qwen-local`` (alias ``local``)  -> local Qwen-VL in this process (transformers)
    ``llamacpp`` (alias ``llama-server``) -> local GGUF served out of process by
                                         llama-server (ADR-0004); configured by
                                         LLAMA_SERVER_BIN / LLAMA_MODEL_GGUF /
                                         LLAMA_MMPROJ_GGUF / LLAMA_SERVER_URL /
                                         LLAMA_SERVER_ARGS, ``model=`` = GGUF path
    ``passthrough``                   -> offline no-model stub
    ``auto``                          -> first provider with credentials:
                                         Qwen API > Anthropic > OpenAI >
                                         Kimi > local Qwen-VL
"""

from __future__ import annotations

import logging
import os
import shlex

from mangawhisperer.engines.placeholders import PassthroughVLM
from mangawhisperer.engines.vlm_api import PROVIDER_PRESETS
from mangawhisperer.interfaces import VisionLanguageEngine

logger = logging.getLogger(__name__)

API_PROVIDERS: tuple[str, ...] = tuple(PROVIDER_PRESETS)
ALL_PROVIDERS: tuple[str, ...] = (*API_PROVIDERS, "anthropic", "qwen-local", "llamacpp", "passthrough")

_LLAMACPP_ALIASES: tuple[str, ...] = ("llamacpp", "llama-server", "llama.cpp")

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
    if provider in _LLAMACPP_ALIASES:
        from mangawhisperer.engines.vlm_llamacpp import LlamaServerVisionLanguageEngine

        return LlamaServerVisionLanguageEngine(**_llamacpp_kwargs(model, kwargs))
    if provider == "passthrough":
        return PassthroughVLM()
    raise ValueError(f"Unknown VLM provider {provider!r}; expected one of {ALL_PROVIDERS + ('auto',)}")


def create_reviewer(provider: str, model: str | None = None, **kwargs: object):
    """Build the script-reviewer engine, or ``None`` when the provider
    can't support it (the in-process local VLM and passthrough review
    nothing). ``llamacpp`` reviews text-only through the same
    llama-server the scriptwriter uses, with a chunk size that fits
    its context window."""
    provider = provider.lower()
    if provider == "auto":
        provider = _resolve_auto()
    if provider in ("qwen-local", "local", "passthrough"):
        return None
    from mangawhisperer.engines.reviewer import LLMScriptReviewer

    if provider in _LLAMACPP_ALIASES:
        from mangawhisperer.engines.vlm_llamacpp import model_label, resolve_server_url

        kwargs.setdefault("base_url", resolve_server_url(os.environ) + "/v1")
        kwargs.setdefault("chunk_size", 8)  # ~1.3k tokens in + out fits a 4k context
        kwargs.setdefault("max_tokens", 2048)
        label = model_label(model or os.environ.get("LLAMA_MODEL_GGUF") or None)
        return LLMScriptReviewer(provider="llamacpp", model=label, **kwargs)
    return LLMScriptReviewer(provider=provider, model=model, **kwargs)


def _llamacpp_kwargs(model: str | None, kwargs: dict[str, object]) -> dict[str, object]:
    """Fill the llama-server engine's inputs from the environment; explicit
    kwargs (and ``model=`` as the GGUF path) always win."""
    env = os.environ
    kwargs.setdefault("model_path", model or env.get("LLAMA_MODEL_GGUF") or None)
    kwargs.setdefault("mmproj_path", env.get("LLAMA_MMPROJ_GGUF") or None)
    kwargs.setdefault("server_binary", env.get("LLAMA_SERVER_BIN") or None)
    kwargs.setdefault("server_url", env.get("LLAMA_SERVER_URL") or None)
    extra = env.get("LLAMA_SERVER_ARGS", "").strip()
    if extra and "extra_server_args" not in kwargs:
        # posix=False on Windows keeps backslashes in paths intact.
        kwargs["extra_server_args"] = tuple(shlex.split(extra, posix=os.name != "nt"))
    return kwargs


def _resolve_auto() -> str:
    for provider, env_var in _AUTO_ORDER:
        if os.environ.get(env_var):
            logger.info("VLM provider 'auto' -> '%s' (%s is set)", provider, env_var)
            return provider
    logger.info("VLM provider 'auto' -> 'qwen-local' (no API key found)")
    return "qwen-local"
