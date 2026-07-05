"""
Unit tests for the central RouterAI / API model-provider layer.

These tests never hit the network and require no real API keys: constructing a
langchain ``ChatOpenAI`` / ``OpenAIEmbeddings`` client only configures an HTTP
client, it does not call out. We assert on the resulting client configuration.
"""

import importlib

import pytest

MODEL_ENV_VARS = [
    "ROUTER_AI_API_KEY",
    "ROUTER_AI_BASE_URL",
    "ROUTER_AI_MODEL_MAP",
    "ROUTER_AI_EMBEDDING_MODEL",
    "ROUTER_AI_EMBEDDING_DIMENSIONS",
    "ROUTER_AI_EMBEDDING_BASE_URL",
    "ROUTER_AI_EMBEDDING_API_KEY",
    "COSCIENTIST_SMARTER_MODELS",
    "COSCIENTIST_CHEAPER_MODELS",
    "COSCIENTIST_DEFAULT_MODEL",
    "COSCIENTIST_EMBEDDING_MODEL",
    "COSCIENTIST_EMBEDDING_DIMENSIONS",
    "COSCIENTIST_EMBEDDING_BASE_URL",
    "COSCIENTIST_EMBEDDING_API_KEY",
    "OPENAI_API_KEY",
]


@pytest.fixture()
def clean_env(monkeypatch):
    """Start from a known env: dummy keys, no overrides."""
    for var in MODEL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ROUTER_AI_API_KEY", "test-routerai-key")
    return monkeypatch


@pytest.fixture()
def mf(clean_env):
    from coscientist import model_factory

    return importlib.reload(model_factory)


def test_get_chat_model_targets_routerai(mf):
    client = mf.get_chat_model("gemini-2.5-pro")
    assert type(client).__name__ == "ChatOpenAI"
    assert client.openai_api_base == mf.DEFAULT_ROUTER_AI_BASE_URL
    # alias resolves to the RouterAI model slug
    assert client.model_name == "google/gemini-2.5-pro"


def test_raw_model_id_passthrough(mf):
    client = mf.get_chat_model("openai/o3")
    assert client.model_name == "openai/o3"
    assert client.openai_api_base == mf.DEFAULT_ROUTER_AI_BASE_URL


def test_base_url_env_override(mf, clean_env):
    clean_env.setenv("ROUTER_AI_BASE_URL", "https://gateway.internal/v1")
    client = mf.get_chat_model("o3")
    assert client.openai_api_base == "https://gateway.internal/v1"


def test_model_map_env_override(mf, clean_env):
    clean_env.setenv("ROUTER_AI_MODEL_MAP", '{"o3": "openai/o3-custom"}')
    assert mf.resolve_model_id("o3") == "openai/o3-custom"
    assert mf.get_chat_model("o3").model_name == "openai/o3-custom"


def test_missing_api_key_raises_only_when_used(clean_env, monkeypatch):
    monkeypatch.delenv("ROUTER_AI_API_KEY", raising=False)
    from coscientist import model_factory

    mf = importlib.reload(model_factory)
    with pytest.raises(RuntimeError, match="ROUTER_AI_API_KEY"):
        mf.get_chat_model("o3")


def test_temperature_only_when_set(mf):
    # A reasoning alias without a registry temperature must not carry one.
    reasoning = mf.get_chat_model("o3")
    assert reasoning.temperature is None
    # An alias with a registry default temperature carries it.
    gemini = mf.get_chat_model("gemini-2.5-pro")
    assert gemini.temperature == 1.0


def test_pools_have_backward_compatible_keys(mf):
    assert set(mf.smarter_pool().keys()) == {
        "gemini-2.5-pro",
        "claude-sonnet-4-20250514",
    }
    assert set(mf.cheaper_pool().keys()) == {
        "gemini-2.5-flash",
        "claude-sonnet-4-20250514",
    }


def test_pool_alias_env_override(mf, clean_env):
    clean_env.setenv("COSCIENTIST_SMARTER_MODELS", "o3, gemini-2.5-pro")
    assert set(mf.smarter_pool().keys()) == {"o3", "gemini-2.5-pro"}


def test_get_embeddings_is_api_based(mf):
    emb = mf.get_embeddings()
    assert type(emb).__name__ == "OpenAIEmbeddings"
    assert emb.model == mf.DEFAULT_EMBEDDING_MODEL
    assert emb.dimensions == mf.DEFAULT_EMBEDDING_DIMENSIONS
    assert emb.openai_api_base == mf.DEFAULT_ROUTER_AI_BASE_URL


def test_get_embeddings_env_overrides(mf, clean_env):
    clean_env.setenv("ROUTER_AI_EMBEDDING_MODEL", "text-embedding-3-large")
    clean_env.setenv("ROUTER_AI_EMBEDDING_DIMENSIONS", "1024")
    clean_env.setenv("ROUTER_AI_EMBEDDING_BASE_URL", "https://emb.internal/v1")
    clean_env.setenv("ROUTER_AI_EMBEDDING_API_KEY", "embedding-key")
    emb = mf.get_embeddings()
    assert emb.model == "text-embedding-3-large"
    assert emb.dimensions == 1024
    assert emb.openai_api_base == "https://emb.internal/v1"


def test_import_does_no_network_and_light_deps(clean_env):
    # Importing the lightweight modules must not require the heavy agent stack
    # (LangGraph / gpt-researcher) and must not raise with only a dummy key set.
    mf = importlib.import_module("coscientist.model_factory")
    prox = importlib.import_module("coscientist.proximity_agent")
    assert hasattr(mf, "get_chat_model")
    assert hasattr(prox, "create_embedding")


def test_reasoning_extra_body(mf):
    high = mf.get_chat_model("google/gemini-2.5-flash", reasoning="high")
    assert high.extra_body == {"reasoning": {"effort": "high"}}
    off = mf.get_chat_model("google/gemini-2.5-flash", reasoning="off")
    assert off.extra_body == {"reasoning": {"enabled": False}}
    default = mf.get_chat_model("google/gemini-2.5-flash", reasoning="default")
    assert not getattr(default, "extra_body", None)
    none = mf.get_chat_model("google/gemini-2.5-flash")
    assert not getattr(none, "extra_body", None)
