# -*- coding: utf-8 -*-
"""Offline tests for the RouterAI web-search retriever."""

import os

from coscientist.web_search import RouterAIWebSearch, extract_citations


_FAKE_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": "Here are sources.",
                "annotations": [
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "url": "https://a.example/paper",
                            "title": "Ni flotation",
                            "content": "Pentlandite liberation study.",
                        },
                    },
                    {
                        "type": "url_citation",
                        "url_citation": {
                            "url": "https://b.example/report",
                            "title": "Tailings",
                        },
                    },
                    {
                        "type": "url_citation",
                        "url_citation": {"url": "https://a.example/paper", "title": "dup"},
                    },
                    {"type": "other", "url_citation": {"url": "https://c.example/skip"}},
                ],
            }
        }
    ]
}


def test_extract_citations_dedupes_and_filters():
    out = extract_citations(_FAKE_RESPONSE)
    urls = [r["href"] for r in out]
    assert urls == ["https://a.example/paper", "https://b.example/report"]
    assert out[0]["body"] == "Pentlandite liberation study."
    assert out[1]["body"] == "Tailings"


def test_extract_citations_empty():
    assert extract_citations({}) == []
    assert extract_citations({"choices": [{"message": {}}]}) == []


def test_search_uses_injected_post_and_respects_max():
    calls = {}

    def fake_post(url, headers, json_body, timeout):
        calls["url"] = url
        calls["model"] = json_body["model"]
        calls["plugins"] = json_body["plugins"]
        calls["auth"] = headers["Authorization"]
        return _FAKE_RESPONSE

    r = RouterAIWebSearch(
        "длинный запрос про пентландит " * 20,
        model="google/gemini-2.5-flash",
        api_key="sk-test",
        post_fn=fake_post,
    )
    results = r.search(max_results=1)
    assert calls["url"].endswith("/chat/completions")
    assert calls["model"] == "google/gemini-2.5-flash"
    assert calls["plugins"] == [{"id": "web", "max_results": 1}]
    assert calls["auth"] == "Bearer sk-test"
    assert len(results) == 1


def test_search_without_key_returns_empty(monkeypatch):
    monkeypatch.delenv("ROUTER_AI_API_KEY", raising=False)
    r = RouterAIWebSearch("q", api_key=None, post_fn=lambda *a, **k: _FAKE_RESPONSE)
    assert r.search() == []


def test_search_swallows_http_errors():
    def boom(*a, **k):
        raise RuntimeError("400 Bad Request")

    r = RouterAIWebSearch("q", api_key="k", post_fn=boom)
    assert r.search() == []


def test_configure_web_retriever_always_routerai(monkeypatch):
    """RouterAI is the only backend: any legacy web_retriever value is ignored."""
    from coscientist.studio import RunSettings, _configure_web_retriever

    monkeypatch.delenv("RETRIEVER", raising=False)
    _configure_web_retriever(RunSettings(web_search_model="google/gemini-2.5-flash"))
    assert os.environ["RETRIEVER"] == "routerai"
    assert os.environ["COSCIENTIST_WEBSEARCH_MODEL"] == "google/gemini-2.5-flash"

    _configure_web_retriever(RunSettings(web_retriever="tavily"))
    assert os.environ["RETRIEVER"] == "routerai"


def test_registration_makes_routerai_a_valid_retriever():
    import pytest

    pytest.importorskip("gpt_researcher")
    from coscientist.web_search import RouterAIWebSearch, register_routerai_retriever

    assert register_routerai_retriever()
    from gpt_researcher.retrievers.utils import get_all_retriever_names

    assert "routerai" in get_all_retriever_names()
    from gpt_researcher.actions.retriever import get_retriever

    assert get_retriever("routerai") is RouterAIWebSearch


def test_registration_makes_routerai_a_valid_llm_provider(monkeypatch):
    import pytest

    pytest.importorskip("gpt_researcher")
    from coscientist.web_search import register_routerai_llm_provider
    from gpt_researcher.llm_provider.generic.base import GenericLLMProvider

    monkeypatch.setenv("ROUTER_AI_API_KEY", "dummy-routerai")
    monkeypatch.setenv("ROUTER_AI_BASE_URL", "https://routerai.internal/v1")
    assert register_routerai_llm_provider()
    provider = GenericLLMProvider.from_provider(
        "routerai",
        model="google/gemini-2.5-flash",
        verbose=False,
    )
    assert provider.llm.openai_api_base == "https://routerai.internal/v1"


def test_registration_makes_routerai_a_valid_embedding_provider(monkeypatch):
    import pytest

    pytest.importorskip("gpt_researcher")
    from coscientist.web_search import register_routerai_embedding_provider
    from gpt_researcher.memory.embeddings import Memory, _SUPPORTED_PROVIDERS

    monkeypatch.setenv("ROUTER_AI_API_KEY", "dummy-routerai")
    monkeypatch.setenv("ROUTER_AI_BASE_URL", "https://routerai.internal/v1")
    assert register_routerai_embedding_provider()
    assert "routerai" in _SUPPORTED_PROVIDERS
    memory = Memory("routerai", "text-embedding-3-small")
    emb = memory.get_embeddings()
    assert emb.openai_api_base == "https://routerai.internal/v1"
