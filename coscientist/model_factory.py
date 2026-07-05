"""
Central model-provider layer for the Co-Scientist system.

Every chat, vision and embedding model used anywhere in the system is created
here and routed through **RouterAI**, an OpenAI-compatible API.

Design rules
------------
* No network client is instantiated at import time. Everything is built lazily
  inside the ``get_*`` functions, so importing this module without any API keys
  never raises and never performs I/O.
* Model ids, pools, base URLs and embedding settings are all overridable through
  environment variables, so the same code runs against RouterAI defaults, a
  self-hosted OpenAI-compatible gateway, or a local deployment.

Environment variables
----------------------
Chat / vision (RouterAI):
    ROUTER_AI_API_KEY        API key for RouterAI (required to use hosted models).
    ROUTER_AI_BASE_URL       Base URL, default ``https://routerai.ai/api/v1``.
    ROUTER_AI_MODEL_MAP      Optional JSON object merged over the default alias map,
                              e.g. ``{"o3": "openai/o3", "fast": "google/gemini-2.5-flash"}``.
    COSCIENTIST_SMARTER_MODELS  Comma-separated alias list for the "smarter" pool.
    COSCIENTIST_CHEAPER_MODELS  Comma-separated alias list for the "cheaper" pool.
    COSCIENTIST_DEFAULT_MODEL   Alias or raw model id used when no spec is given.
    ROUTER_AI_APP_URL / ROUTER_AI_APP_TITLE  Optional attribution headers.

Embeddings (RouterAI-compatible API):
    ROUTER_AI_EMBEDDING_MODEL       default ``text-embedding-3-small``.
    ROUTER_AI_EMBEDDING_DIMENSIONS  default ``256``.
    ROUTER_AI_EMBEDDING_BASE_URL    optional override; defaults to ``ROUTER_AI_BASE_URL``.
    ROUTER_AI_EMBEDDING_API_KEY     optional override; defaults to ``ROUTER_AI_API_KEY``.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ROUTER_AI_BASE_URL = "https://routerai.ai/api/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 256
DEFAULT_MAX_TOKENS = 50_000
DEFAULT_MAX_RETRIES = 3
# A vision-capable RouterAI model for parsing images (flotation schemes,
# regulations, equipment lists). Overridable via COSCIENTIST_VISION_MODEL.
DEFAULT_VISION_MODEL = "google/gemini-2.5-pro"

# Logical alias -> RouterAI model slug plus any per-model default kwargs.
# The aliases intentionally match the keys used historically by the framework
# so that the rest of the system stays backward compatible.
_DEFAULT_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "o3": {"model": "openai/o3"},
    "o4-mini": {"model": "openai/o4-mini"},
    "gemini-2.5-pro": {"model": "google/gemini-2.5-pro", "temperature": 1.0},
    "gemini-2.5-flash": {"model": "google/gemini-2.5-flash", "temperature": 1.0},
    "claude-sonnet-4-20250514": {"model": "anthropic/claude-sonnet-4"},
}

_DEFAULT_SMARTER_ALIASES = ["gemini-2.5-pro", "claude-sonnet-4-20250514"]
_DEFAULT_CHEAPER_ALIASES = ["gemini-2.5-flash", "claude-sonnet-4-20250514"]


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------


def router_ai_base_url() -> str:
    return os.environ.get("ROUTER_AI_BASE_URL", DEFAULT_ROUTER_AI_BASE_URL)


def _router_ai_api_key() -> str:
    # RouterAI is OpenAI-compatible; the underlying client would otherwise fall
    # back to OPENAI_API_KEY, which is wrong here, so we resolve it explicitly.
    key = os.environ.get("ROUTER_AI_API_KEY")
    if not key:
        raise RuntimeError(
            "ROUTER_AI_API_KEY is not set. Every chat model in Co-Scientist is "
            "routed through RouterAI; export ROUTER_AI_API_KEY to use them."
        )
    return key


def _router_ai_default_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    app_url = os.environ.get("ROUTER_AI_APP_URL")
    app_title = os.environ.get("ROUTER_AI_APP_TITLE")
    if app_url:
        headers["HTTP-Referer"] = app_url
    if app_title:
        headers["X-Title"] = app_title
    return headers


def model_registry() -> dict[str, dict[str, Any]]:
    """Return the alias registry with any ``ROUTER_AI_MODEL_MAP`` overrides applied."""
    registry = {alias: dict(cfg) for alias, cfg in _DEFAULT_MODEL_REGISTRY.items()}
    raw = os.environ.get("ROUTER_AI_MODEL_MAP")
    if raw:
        try:
            overrides = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ROUTER_AI_MODEL_MAP is not valid JSON: {exc}") from exc
        for alias, model_id in overrides.items():
            entry = registry.get(alias, {})
            entry["model"] = model_id
            registry[alias] = entry
    return registry


def resolve_model_id(spec: str | None) -> str:
    """Resolve a spec (alias or raw model id) to a concrete RouterAI model id."""
    if spec is None:
        spec = os.environ.get("COSCIENTIST_DEFAULT_MODEL", "claude-sonnet-4-20250514")
    registry = model_registry()
    if spec in registry:
        return registry[spec]["model"]
    # Not a known alias: treat it as a raw RouterAI model id (e.g. "openai/o3").
    return spec


def _pool_aliases(env_var: str, default: list[str]) -> list[str]:
    raw = os.environ.get(env_var)
    if not raw:
        return list(default)
    return [alias.strip() for alias in raw.split(",") if alias.strip()]


# ---------------------------------------------------------------------------
# Chat models
# ---------------------------------------------------------------------------


def _reasoning_extra_body(reasoning: str | None) -> dict | None:
    """
    Translate a friendly thinking level into a RouterAI ``reasoning`` body.

    RouterAI normalizes this across providers and safely IGNORES it when a
    model cannot honor it (e.g. asking a non-reasoning model to think, or asking
    an always-thinking model to stop), so callers never get an error for a
    mismatched capability — the model just runs in its natural mode.
    """
    if reasoning is None:
        return None
    level = str(reasoning).strip().lower()
    if level in ("", "default", "auto"):
        return None
    if level in ("off", "none", "disabled"):
        return {"reasoning": {"enabled": False}}
    if level in ("low", "medium", "high"):
        return {"reasoning": {"effort": level}}
    return None


def get_chat_model(
    spec: str | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float | None = None,
    reasoning: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Build a chat model routed through RouterAI.

    Parameters
    ----------
    spec : str | None
        A registry alias (e.g. ``"gemini-2.5-pro"``) or a raw RouterAI model id
        (e.g. ``"openai/o3"``). ``None`` uses ``COSCIENTIST_DEFAULT_MODEL``.
    max_tokens, max_retries, temperature :
        Standard generation controls. ``temperature`` is only sent when set, so
        reasoning models that reject it are handled correctly. A registry alias
        may supply a default temperature.
    """
    # Imported lazily so that merely importing this module never pulls a client.
    from langchain_openai import ChatOpenAI

    registry = model_registry()
    entry = registry.get(spec, {}) if spec is not None else {}
    model_id = resolve_model_id(spec)

    if temperature is None:
        temperature = entry.get("temperature")

    client_kwargs: dict[str, Any] = {
        "model": model_id,
        "base_url": router_ai_base_url(),
        "api_key": _router_ai_api_key(),
        "max_tokens": max_tokens,
        "max_retries": max_retries,
    }
    if temperature is not None:
        client_kwargs["temperature"] = temperature
    headers = _router_ai_default_headers()
    if headers:
        client_kwargs["default_headers"] = headers

    # Thinking / reasoning control (RouterAI). Merged with any caller extra_body.
    extra_body = dict(kwargs.pop("extra_body", None) or {})
    reasoning_body = _reasoning_extra_body(reasoning)
    if reasoning_body:
        extra_body.update(reasoning_body)
    if extra_body:
        client_kwargs["extra_body"] = extra_body

    client_kwargs.update(kwargs)

    return ChatOpenAI(**client_kwargs)


def get_vision_model(
    spec: str | None = None,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Build a vision-capable chat model routed through RouterAI.

    Uses ``COSCIENTIST_VISION_MODEL`` (or a sensible default) when ``spec`` is
    None. The returned client accepts OpenAI-style multimodal messages, so no
    extra SDK is needed to caption/parse images.
    """
    if spec is None:
        spec = os.environ.get("COSCIENTIST_VISION_MODEL", DEFAULT_VISION_MODEL)
    return get_chat_model(spec, max_tokens=max_tokens, **kwargs)


def get_chat_model_pool(
    aliases: list[str] | None = None,
    *,
    env_var: str | None = None,
    default_aliases: list[str] | None = None,
    **model_kwargs: Any,
) -> dict[str, BaseChatModel]:
    """Build a ``{alias: chat_model}`` pool. Alias list is env-overridable."""
    if aliases is None:
        aliases = _pool_aliases(
            env_var or "", default_aliases or list(_DEFAULT_MODEL_REGISTRY.keys())
        )
    return {alias: get_chat_model(alias, **model_kwargs) for alias in aliases}


def smarter_pool(**model_kwargs: Any) -> dict[str, BaseChatModel]:
    """The pool of stronger reasoning models (backward-compatible keys)."""
    return get_chat_model_pool(
        env_var="COSCIENTIST_SMARTER_MODELS",
        default_aliases=_DEFAULT_SMARTER_ALIASES,
        **model_kwargs,
    )


def cheaper_pool(**model_kwargs: Any) -> dict[str, BaseChatModel]:
    """The pool of cheaper / faster models (backward-compatible keys)."""
    return get_chat_model_pool(
        env_var="COSCIENTIST_CHEAPER_MODELS",
        default_aliases=_DEFAULT_CHEAPER_ALIASES,
        **model_kwargs,
    )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def embedding_dimensions() -> int:
    return int(
        os.environ.get("ROUTER_AI_EMBEDDING_DIMENSIONS")
        or os.environ.get("COSCIENTIST_EMBEDDING_DIMENSIONS")
        or DEFAULT_EMBEDDING_DIMENSIONS
    )


def get_embeddings(
    *,
    model: str | None = None,
    dimensions: int | None = None,
    **kwargs: Any,
) -> Embeddings:
    """
    Build a RouterAI-compatible embeddings client.

    By default embeddings use the same RouterAI key and OpenAI-compatible base
    URL as chat/vision. Separate embedding env vars are still supported for
    deployments that split chat and embedding traffic.
    """
    from langchain_openai import OpenAIEmbeddings

    model = (
        model
        or os.environ.get("ROUTER_AI_EMBEDDING_MODEL")
        or os.environ.get("COSCIENTIST_EMBEDDING_MODEL")
        or DEFAULT_EMBEDDING_MODEL
    )
    if dimensions is None:
        dimensions = embedding_dimensions()

    base_url = (
        os.environ.get("ROUTER_AI_EMBEDDING_BASE_URL")
        or os.environ.get("COSCIENTIST_EMBEDDING_BASE_URL")
        or router_ai_base_url()
    )
    api_key = (
        os.environ.get("ROUTER_AI_EMBEDDING_API_KEY")
        or os.environ.get("COSCIENTIST_EMBEDDING_API_KEY")
        or os.environ.get("ROUTER_AI_API_KEY")
    )

    client_kwargs: dict[str, Any] = {
        "model": model,
        "dimensions": dimensions,
        "base_url": base_url,
    }
    if api_key:
        client_kwargs["api_key"] = api_key
    client_kwargs.update(kwargs)

    return OpenAIEmbeddings(**client_kwargs)
