"""
Citation formatting for retrieved corpus chunks.

Turns retrieval hits into (1) a compact numbered reference list and (2) a
grounded context block with inline ``[n]`` markers, so a downstream agent or
report can attach precise, verifiable references (source name + locator).
"""

from __future__ import annotations

from typing import Sequence

from coscientist.corpus.store import RetrievedChunk

_MODALITY_LABEL = {
    "pdf": "PDF",
    "docx": "DOCX",
    "xlsx": "XLSX",
    "image": "image (VLM)",
}


def format_reference(chunk_hit: RetrievedChunk, index: int) -> str:
    """Format a single ``[n] source_name — locator (modality)`` reference line."""
    chunk = chunk_hit.chunk
    modality = _MODALITY_LABEL.get(chunk.modality, chunk.modality)
    return f"[{index}] {chunk.source_name} — {chunk.locator} ({modality})"


def format_references(hits: Sequence[RetrievedChunk]) -> str:
    """Format a numbered reference list for a set of retrieval hits."""
    if not hits:
        return "No sources retrieved."
    return "\n".join(format_reference(hit, i) for i, hit in enumerate(hits, start=1))


def format_context_block(
    hits: Sequence[RetrievedChunk],
    *,
    max_chars_per_chunk: int = 800,
) -> str:
    """
    Build a grounded context block: each excerpt is prefixed with its ``[n]``
    marker and followed by a numbered reference list, ready to paste into a
    generation/reflection prompt.
    """
    if not hits:
        return "No supporting sources were retrieved from the private corpus."

    excerpts = []
    for i, hit in enumerate(hits, start=1):
        text = hit.chunk.text.strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + " …"
        excerpts.append(f"[{i}] {text}")

    body = "\n\n".join(excerpts)
    refs = format_references(hits)
    return f"{body}\n\nSources:\n{refs}"
