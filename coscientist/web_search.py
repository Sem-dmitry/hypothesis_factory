"""
RouterAI web-search retriever and GPT Researcher provider bridge.

RouterAI is used through the same OpenAI-compatible protocol as the chat,
vision and embedding model clients. The web retriever calls a RouterAI chat
model with the ``web`` plugin and returns GPT Researcher-compatible
``{"href", "body"}`` search results extracted from URL annotations.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

from coscientist.model_factory import DEFAULT_ROUTER_AI_BASE_URL, router_ai_base_url

ROUTER_AI_RETRIEVER = "routerai"
DEFAULT_WEBSEARCH_MODEL = "google/gemini-2.5-flash"


def router_ai_chat_url() -> str:
    return router_ai_base_url().rstrip("/") + "/chat/completions"


def researcher_config_path() -> str:
    """
    Path to the GPT Researcher config JSON for the current run.

    Honours ``COSCIENTIST_RESEARCHER_CONFIG`` (set per-run by the studio to pick
    the lite vs full config), falling back to the full ``researcher_config.json``
    next to this package.
    """
    try:
        from coscientist.env_utils import bridge_provider_aliases

        bridge_provider_aliases()
        register_routerai_gpt_researcher()
    except Exception:
        pass
    override = os.environ.get("COSCIENTIST_RESEARCHER_CONFIG")
    if override and os.path.exists(override):
        return override
    return os.path.join(os.path.dirname(__file__), "researcher_config.json")


def _default_post(url: str, headers: dict, json_body: dict, timeout: float) -> dict:
    import requests

    resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def extract_citations(response: dict) -> list[dict]:
    """
    Pull ``{"href", "body"}`` results from a RouterAI chat response.

    Reads the assistant message's ``annotations`` (url_citation entries). Falls
    back to an empty list when there are none.
    """
    results: list[dict] = []
    seen: set[str] = set()
    for choice in (response or {}).get("choices", []):
        message = choice.get("message", {}) or {}
        for ann in message.get("annotations", []) or []:
            if ann.get("type") != "url_citation":
                continue
            cit = ann.get("url_citation", {}) or {}
            url = cit.get("url")
            if not url or url in seen:
                continue
            seen.add(url)
            body = (cit.get("content") or cit.get("title") or "").strip()
            results.append({"href": url, "body": body})
    return results


class RouterAIWebSearch:
    """GPT Researcher-compatible retriever backed by RouterAI web search."""

    def __init__(
        self,
        query: str,
        headers: Optional[dict] = None,
        topic: str = "general",
        query_domains: Optional[list[str]] = None,
        *,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        post_fn: Optional[Callable[..., dict]] = None,
        **kwargs: Any,
    ):
        self.query = query
        self.query_domains = query_domains or []
        self.model = model or os.environ.get(
            "COSCIENTIST_WEBSEARCH_MODEL", DEFAULT_WEBSEARCH_MODEL
        )
        self.api_key = (
            api_key
            or (headers or {}).get("router_ai_api_key")
            or (headers or {}).get("ROUTER_AI_API_KEY")
            or os.environ.get("ROUTER_AI_API_KEY")
        )
        self._post = post_fn or _default_post

    def search(self, max_results: int = 7) -> list[dict]:
        """Return up to ``max_results`` web results for the query."""
        if not self.api_key:
            return []
        domains = ""
        if self.query_domains:
            domains = " Restrict results to these domains: " + ", ".join(self.query_domains)
        body = {
            "model": self.model,
            "plugins": [{"id": "web", "max_results": max_results}],
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Search the web for authoritative sources on the following and "
                        "cite them. Be concise." + domains + "\n\n" + self.query
                    ),
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            data = self._post(router_ai_chat_url(), headers, body, 120.0)
        except Exception:
            return []
        return extract_citations(data)[:max_results]


def register_routerai_retriever() -> bool:
    """
    Make GPT Researcher accept and resolve ``RETRIEVER="routerai"``.

    Two things must be patched, both idempotently:
    1. the retriever factory so the name resolves to :class:`RouterAIWebSearch`;
    2. the name validation list so config loading does not fall back to Tavily.
    """
    patched_any = False

    try:
        from gpt_researcher.actions import retriever as _r

        original_get = getattr(_r, "_coscientist_original_get_retriever", None) or _r.get_retriever

        def _patched_get(retriever: str):
            if retriever == ROUTER_AI_RETRIEVER:
                return RouterAIWebSearch
            return original_get(retriever)

        _r._coscientist_original_get_retriever = original_get
        _r.get_retriever = _patched_get
        patched_any = True
    except Exception:
        pass

    try:
        from gpt_researcher.retrievers import utils as _u

        original_names = (
            getattr(_u, "_coscientist_original_get_all_retriever_names", None)
            or _u.get_all_retriever_names
        )

        def _patched_names():
            names = list(original_names() or [])
            if ROUTER_AI_RETRIEVER not in names:
                names.append(ROUTER_AI_RETRIEVER)
            return names

        _u._coscientist_original_get_all_retriever_names = original_names
        _u.get_all_retriever_names = _patched_names
        try:
            if ROUTER_AI_RETRIEVER not in _u.VALID_RETRIEVERS:
                _u.VALID_RETRIEVERS.append(ROUTER_AI_RETRIEVER)
        except Exception:
            pass
        patched_any = True
    except Exception:
        pass

    return patched_any


def register_routerai_llm_provider() -> bool:
    """Teach GPT Researcher the ``routerai:MODEL`` LLM provider prefix."""
    try:
        from gpt_researcher.llm_provider.generic import base as _base
        from gpt_researcher.llm_provider.generic.base import GenericLLMProvider

        original_from_provider = (
            getattr(_base, "_coscientist_original_from_provider", None)
            or GenericLLMProvider.from_provider.__func__
        )

        def _patched_from_provider(cls, provider: str, chat_log: str | None = None, verbose: bool = True, **kwargs: Any):
            if provider != ROUTER_AI_RETRIEVER:
                return original_from_provider(cls, provider, chat_log=chat_log, verbose=verbose, **kwargs)

            _base._check_pkg("langchain_openai")
            from langchain_core.rate_limiters import InMemoryRateLimiter
            from langchain_openai import ChatOpenAI

            rps = float(os.environ.get("ROUTER_AI_LIMIT_RPS", "1.0"))
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=rps,
                check_every_n_seconds=0.1,
                max_bucket_size=10,
            )
            llm = ChatOpenAI(
                openai_api_base=os.environ.get("ROUTER_AI_BASE_URL", DEFAULT_ROUTER_AI_BASE_URL),
                request_timeout=180,
                openai_api_key=os.environ["ROUTER_AI_API_KEY"],
                rate_limiter=rate_limiter,
                **kwargs,
            )
            return cls(llm, chat_log, verbose=verbose)

        _base._coscientist_original_from_provider = original_from_provider
        GenericLLMProvider.from_provider = classmethod(_patched_from_provider)
        return True
    except Exception:
        return False


def register_routerai_embedding_provider() -> bool:
    """Teach GPT Researcher the ``routerai:MODEL`` embedding provider prefix."""
    try:
        from gpt_researcher.memory import embeddings as _emb

        if ROUTER_AI_RETRIEVER not in _emb._SUPPORTED_PROVIDERS:
            _emb._SUPPORTED_PROVIDERS.add(ROUTER_AI_RETRIEVER)

        original_init = (
            getattr(_emb.Memory, "_coscientist_original_init", None)
            or _emb.Memory.__init__
        )

        def _patched_init(self, embedding_provider: str, model: str, **embedding_kwargs: Any):
            if embedding_provider != ROUTER_AI_RETRIEVER:
                return original_init(self, embedding_provider, model, **embedding_kwargs)

            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=model,
                openai_api_key=(
                    os.getenv("ROUTER_AI_EMBEDDING_API_KEY")
                    or os.getenv("ROUTER_AI_API_KEY")
                ),
                openai_api_base=(
                    os.getenv("ROUTER_AI_EMBEDDING_BASE_URL")
                    or os.getenv("ROUTER_AI_BASE_URL", DEFAULT_ROUTER_AI_BASE_URL)
                ),
                **embedding_kwargs,
            )
            return None

        _emb.Memory._coscientist_original_init = original_init
        _emb.Memory.__init__ = _patched_init
        return True
    except Exception:
        return False


def register_routerai_gpt_researcher() -> bool:
    """Register RouterAI retriever, LLM and embedding provider hooks."""
    retriever_ok = register_routerai_retriever()
    provider_ok = register_routerai_llm_provider()
    embedding_ok = register_routerai_embedding_provider()
    return bool(retriever_ok or provider_ok or embedding_ok)
