# -*- coding: utf-8 -*-
"""Utilities for exposing Literature-agent web links as cited evidence.

These helpers intentionally do not add web pages to the private corpus/RAG
index. They only normalize URLs already present in Literature reports so later
agents can cite them and the web UI can render clickable source links.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class WebLiteratureSource:
    """A web source mentioned by the Literature agent."""

    title: str
    url: str
    snippet: str = ""


_MD_LINK_RE = re.compile(r"\[([^\]\n]{1,240})\]\((https?://[^)\s]+)\)")
_RAW_URL_RE = re.compile(r"https?://[^\s<>\]\)\"']+")
_TRAILING_URL_PUNCT = ".,;:"


def extract_web_literature_sources(
    reports: str | list[str] | tuple[str, ...] | None,
    *,
    max_sources: int = 20,
) -> list[WebLiteratureSource]:
    """Extract deterministic, de-duplicated web sources from Literature text."""
    if reports is None:
        return []
    if isinstance(reports, str):
        texts = [reports]
    else:
        texts = [str(r) for r in reports if str(r).strip()]

    sources: list[WebLiteratureSource] = []
    seen: set[str] = set()

    for text in texts:
        if len(sources) >= max_sources:
            break
        occupied: list[tuple[int, int]] = []

        for match in _MD_LINK_RE.finditer(text):
            url = _clean_url(match.group(2))
            if not url or url in seen:
                continue
            title = _clean_title(match.group(1)) or _title_from_url(url)
            sources.append(
                WebLiteratureSource(
                    title=title,
                    url=url,
                    snippet=_snippet_around(text, match.start(), match.end()),
                )
            )
            seen.add(url)
            occupied.append(match.span(2))
            if len(sources) >= max_sources:
                break

        if len(sources) >= max_sources:
            break

        for match in _RAW_URL_RE.finditer(text):
            if _inside_any(match.span(), occupied):
                continue
            url = _clean_url(match.group(0))
            if not url or url in seen:
                continue
            sources.append(
                WebLiteratureSource(
                    title=_title_near_url(text, match.start(), match.end(), url),
                    url=url,
                    snippet=_snippet_around(text, match.start(), match.end()),
                )
            )
            seen.add(url)
            if len(sources) >= max_sources:
                break

    return sources


def format_web_reference(source: WebLiteratureSource, index: int) -> str:
    """Format one web literature reference as ``[Wn] title — url (web)``."""
    return f"[W{index}] {source.title} — {source.url} (web)"


def format_web_references(sources: list[WebLiteratureSource]) -> list[str]:
    """Return numbered web reference lines."""
    return [format_web_reference(src, i) for i, src in enumerate(sources, start=1)]


def format_web_context_block(
    sources: list[WebLiteratureSource],
    *,
    max_chars_per_source: int = 700,
) -> str:
    """Build a prompt block for web sources found by Literature."""
    if not sources:
        return ""

    blocks: list[str] = []
    for i, src in enumerate(sources, start=1):
        snippet = re.sub(r"\s+", " ", src.snippet or "").strip()
        if len(snippet) > max_chars_per_source:
            snippet = snippet[:max_chars_per_source].rstrip() + " ..."
        blocks.append(
            f"[W{i}] {src.title}\n"
            f"URL: {src.url}\n"
            f"Literature context: {snippet or src.title}"
        )
    return "\n\n".join(blocks) + "\n\nSources:\n" + "\n".join(
        format_web_references(sources)
    )


def append_web_references_to_review(literature_review: str) -> str:
    """Append normalized web references to literature review text for Generation."""
    sources = extract_web_literature_sources(literature_review)
    if not sources:
        return literature_review
    return (
        f"{literature_review.rstrip()}\n\n"
        "# Web literature references found by Literature\n"
        "Use these [Wn] references when they directly support a hypothesis. "
        "They are web links from the Literature agent, not private-corpus chunks.\n\n"
        f"{format_web_context_block(sources)}"
    )


def _clean_url(url: str) -> str:
    cleaned = (url or "").strip().rstrip(_TRAILING_URL_PUNCT)
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return cleaned


def _clean_title(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", (title or "")).strip(" -:;.,")
    return cleaned[:240]


def _title_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc.removeprefix("www.")
    path = parsed.path.strip("/")
    if path:
        tail = path.split("/")[-1].replace("-", " ").replace("_", " ")
        tail = re.sub(r"\.\w+$", "", tail).strip()
        if tail:
            return f"{host}: {tail[:120]}"
    return host or url


def _line_at(text: str, start: int, end: int) -> str:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end]


def _title_near_url(text: str, start: int, end: int, url: str) -> str:
    line = _line_at(text, start, end)
    line_without_url = _RAW_URL_RE.sub("", line)
    line_without_url = re.sub(r"^\s*(?:[-*]|\d+[.)]|\[\d+\])\s*", "", line_without_url)
    title = _clean_title(line_without_url)
    if title and len(title) >= 4:
        return title
    return _title_from_url(url)


def _snippet_around(text: str, start: int, end: int, *, radius: int = 360) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right]
    return re.sub(r"\s+", " ", snippet).strip()


def _inside_any(span: tuple[int, int], occupied: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start >= left and end <= right for left, right in occupied)
