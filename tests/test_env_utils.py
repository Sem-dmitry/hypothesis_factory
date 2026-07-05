"""Offline tests for the .env auto-loader."""

import os

from coscientist.env_utils import load_env_file, parse_env_text


def test_parse_env_text_variants():
    text = (
        "# comment\n"
        "\n"
        "ROUTER_AI_API_KEY=sk-or-123\n"
        "export COSCIENTIST_WEBSEARCH_MODEL=google/gemini-2.5-flash\n"
        'ROUTER_AI_EMBEDDING_API_KEY="sk-embed"\n'
        "EMPTY=\n"
        "no_equals_line\n"
    )
    parsed = parse_env_text(text)
    assert parsed["ROUTER_AI_API_KEY"] == "sk-or-123"
    assert parsed["COSCIENTIST_WEBSEARCH_MODEL"] == "google/gemini-2.5-flash"  # export prefix stripped
    assert parsed["ROUTER_AI_EMBEDDING_API_KEY"] == "sk-embed"  # quotes stripped
    assert parsed["EMPTY"] == ""
    assert "no_equals_line" not in parsed


def test_load_env_file_sets_and_respects_existing(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "ROUTER_AI_API_KEY=from-file\nROUTER_AI_EMBEDDING_API_KEY=emb-file\n",
        encoding="utf-8",
    )
    # a real env var must win over the file
    monkeypatch.setenv("ROUTER_AI_API_KEY", "from-shell")
    monkeypatch.delenv("ROUTER_AI_EMBEDDING_API_KEY", raising=False)

    applied = load_env_file(str(tmp_path))
    assert "ROUTER_AI_EMBEDDING_API_KEY" in applied
    assert "ROUTER_AI_API_KEY" not in applied  # not overridden
    assert os.environ["ROUTER_AI_API_KEY"] == "from-shell"
    assert os.environ["ROUTER_AI_EMBEDDING_API_KEY"] == "emb-file"


def test_load_env_file_prefers_dotenv_over_env(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("K=dot\n", encoding="utf-8")
    (tmp_path / "env").write_text("K=plain\n", encoding="utf-8")
    monkeypatch.delenv("K", raising=False)
    load_env_file(str(tmp_path))
    assert os.environ["K"] == "dot"


def test_load_env_file_missing_returns_empty(tmp_path):
    assert load_env_file(str(tmp_path)) == []


def test_bridge_provider_aliases(monkeypatch):
    from coscientist.env_utils import bridge_provider_aliases

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("ROUTER_AI_EMBEDDING_API_KEY", "emb-key")
    monkeypatch.setenv("ROUTER_AI_EMBEDDING_BASE_URL", "https://emb.example/v1")

    applied = bridge_provider_aliases()
    assert "OPENAI_API_KEY" in applied and "OPENAI_BASE_URL" in applied
    assert os.environ["OPENAI_API_KEY"] == "emb-key"
    assert os.environ["OPENAI_BASE_URL"] == "https://emb.example/v1"


def test_bridge_does_not_override_existing(monkeypatch):
    from coscientist.env_utils import bridge_provider_aliases

    monkeypatch.setenv("OPENAI_API_KEY", "real-openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("ROUTER_AI_EMBEDDING_API_KEY", "emb-key")
    applied = bridge_provider_aliases()
    assert applied == []
    assert os.environ["OPENAI_API_KEY"] == "real-openai"  # not overridden
    assert "OPENAI_BASE_URL" not in os.environ


def test_bridge_leaves_plain_openai_key_alone(monkeypatch):
    from coscientist.env_utils import bridge_provider_aliases

    monkeypatch.setenv("OPENAI_API_KEY", "real-openai")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ROUTER_AI_API_KEY", raising=False)
    monkeypatch.delenv("ROUTER_AI_EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("COSCIENTIST_EMBEDDING_API_KEY", raising=False)
    applied = bridge_provider_aliases()
    assert applied == []
    assert "OPENAI_BASE_URL" not in os.environ


def test_apply_llm_runtime_defaults(monkeypatch):
    from coscientist.env_utils import apply_llm_runtime_defaults

    monkeypatch.delenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", raising=False)
    applied = apply_llm_runtime_defaults()
    assert "LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S" in applied
    assert float(os.environ["LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S"]) >= 300

    # respects an explicit user value
    monkeypatch.setenv("LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S", "42")
    apply_llm_runtime_defaults()
    assert os.environ["LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S"] == "42"


def test_silence_optional_warnings():
    import logging
    from coscientist.env_utils import silence_optional_warnings

    logging.getLogger("gpt_researcher.retrievers.mcp").setLevel(logging.WARNING)
    silence_optional_warnings()
    assert logging.getLogger("gpt_researcher.retrievers.mcp").level == logging.ERROR
