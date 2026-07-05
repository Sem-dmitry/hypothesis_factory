"""
Tiny, dependency-free ``.env`` loader.

The Studio reads credentials from ``os.environ``. To spare users from exporting
variables by hand, we auto-load a ``.env`` (or ``env``) file from the repo root
at startup. Existing environment variables are never overridden.
"""

from __future__ import annotations

import os
from typing import Optional

# Candidate filenames, in priority order.
_ENV_FILENAMES = (".env", "env")


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=VALUE`` lines (dotenv style) into a dict."""
    result: dict[str, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip matching surrounding quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        result[key] = value
    return result


def silence_optional_warnings() -> None:
    """
    Quiet harmless startup noise from optional gpt_researcher retrievers we do
    not use. The MCP retriever needs ``langchain_mcp_adapters`` (an undeclared
    optional dep); when absent, gpt_researcher logs
    ``Failed to import MCPRetriever`` at import time. We use RouterAI web
    search, not MCP, so we raise that logger's level to ERROR (called before
    gpt_researcher is imported).
    """
    import logging

    logging.getLogger("gpt_researcher.retrievers.mcp").setLevel(logging.ERROR)


def apply_llm_runtime_defaults() -> list[str]:
    """
    Set generous defaults for long-running LLM calls if the user hasn't.

    Deep-mode web research (gpt_researcher) generates long reports; the default
    120s per-chunk streaming timeout in this environment's langchain_openai
    kills calls when the model goes content-silent for a while (slow upstream /
    RouterAI routing), triggering many retries. Raise it so long syntheses
    finish. Returns the names that were applied.
    """
    applied: list[str] = []
    defaults = {
        # Seconds of content-silence tolerated on a streaming response.
        "LANGCHAIN_OPENAI_STREAM_CHUNK_TIMEOUT_S": "600",
    }
    for key, value in defaults.items():
        if not os.environ.get(key):
            os.environ[key] = value
            applied.append(key)
    return applied


def bridge_provider_aliases() -> list[str]:
    """
    Map our credential variable names onto the ones third-party libraries expect.

    ``gpt_researcher`` (deep-mode web research) uses the ``openai:`` embedding
    provider, which reads ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``. RouterAI uses
    the same OpenAI-compatible protocol, so alias RouterAI embedding credentials
    to those third-party variable names only when the user has not set them.
    Returns the names that were applied.
    """
    applied: list[str] = []
    key_sources = (
        "ROUTER_AI_EMBEDDING_API_KEY",
        "COSCIENTIST_EMBEDDING_API_KEY",
        "ROUTER_AI_API_KEY",
    )
    base_sources = (
        "ROUTER_AI_EMBEDDING_BASE_URL",
        "COSCIENTIST_EMBEDDING_BASE_URL",
        "ROUTER_AI_BASE_URL",
    )
    key_value = next((os.environ.get(source) for source in key_sources if os.environ.get(source)), None)
    using_routerai_alias = False
    if key_value and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = key_value
        applied.append("OPENAI_API_KEY")
        using_routerai_alias = True

    base_value = next((os.environ.get(source) for source in base_sources if os.environ.get(source)), None)
    if not base_value and using_routerai_alias:
        base_value = "https://routerai.ai/api/v1"
    if base_value and using_routerai_alias and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = base_value
        applied.append("OPENAI_BASE_URL")
    return applied


def load_env_file(
    repo_root: Optional[str] = None,
    *,
    override: bool = False,
) -> list[str]:
    """
    Load the first existing ``.env``/``env`` file under ``repo_root`` into
    ``os.environ``. Returns the list of variable names applied.

    Existing environment variables win unless ``override`` is True.
    """
    if repo_root is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for name in _ENV_FILENAMES:
        path = os.path.join(repo_root, name)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                pairs = parse_env_text(fh.read())
            applied = []
            for key, value in pairs.items():
                if override or not os.environ.get(key):
                    os.environ[key] = value
                    applied.append(key)
            return applied
    return []
