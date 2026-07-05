"""
Retrieval helper on top of :class:`CorpusIndex`.

Turns a query into a grounded, cited context block ready to drop into a
generation / reflection / assessment prompt. This is the bridge between the
Phase-2 corpus index and the Phase-3 grounded agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from coscientist.corpus import citations
from coscientist.corpus.store import CorpusIndex, EmbedFn, RetrievedChunk
from coscientist.tailings_domain import is_tailings_related, tailings_guide_block


@dataclass
class GroundedContext:
    """A cited context block plus the raw retrieval hits it was built from."""

    query: str
    context_block: str
    hits: list[RetrievedChunk] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.hits

    def citation_lines(self) -> list[str]:
        """Structural citations (source_name + locator) for each hit, in order."""
        return [
            citations.format_reference(hit, i)
            for i, hit in enumerate(self.hits, start=1)
        ]


class CorpusRetriever:
    """Query the private corpus and return grounded, cited context."""

    def __init__(self, index: CorpusIndex, *, default_k: int = 6):
        self.index = index
        self.default_k = default_k

    @classmethod
    def from_path(
        cls,
        index_path: str,
        *,
        default_k: int = 6,
        embed_fn: Optional[EmbedFn] = None,
    ) -> "CorpusRetriever":
        """Build a retriever from a saved index on disk."""
        index = CorpusIndex.load(index_path, embed_fn=embed_fn)
        return cls(index, default_k=default_k)

    def ground(self, query: str, k: Optional[int] = None) -> GroundedContext:
        """Retrieve top-k corpus chunks for ``query`` and format a cited block."""
        k = k if k is not None else self.default_k
        hits: Sequence[RetrievedChunk] = self._search_grounding_hits(query, k)
        block = citations.format_context_block(hits)
        if self._needs_tailings_guide(query, hits):
            block = f"{tailings_guide_block()}\n\n{block}"
        return GroundedContext(query=query, context_block=block, hits=list(hits))

    def _search_grounding_hits(self, query: str, k: int) -> list[RetrievedChunk]:
        """Search with source-aware tailings handling while preserving defaults."""
        if not self.index.chunks:
            return []
        tailings_query = is_tailings_related(query)
        if not tailings_query:
            return self.index.search(query, k=k)

        # Tailings spreadsheets are often small (few chunks) and can be drowned
        # out by large PDF textbooks. For tailings goals we inspect the full
        # ranked list, then force source diversity and at least one relevant
        # XLSX hit when available.
        candidates = self.index.search(query, k=len(self.index.chunks))
        return _select_tailings_hits(candidates, k)

    @staticmethod
    def _needs_tailings_guide(query: str, hits: Sequence[RetrievedChunk]) -> bool:
        if is_tailings_related(query):
            return True
        return any(
            bool(hit.chunk.metadata.get("tailings_report"))
            or is_tailings_related(hit.chunk.source_name, hit.chunk.text[:500])
            for hit in hits
        )


def _is_tailings_xlsx(hit: RetrievedChunk) -> bool:
    chunk = hit.chunk
    return chunk.modality == "xlsx" and (
        bool(chunk.metadata.get("tailings_report"))
        or is_tailings_related(chunk.source_name, chunk.locator, chunk.text[:1200])
    )


def _select_tailings_hits(candidates: Sequence[RetrievedChunk], k: int) -> list[RetrievedChunk]:
    if not candidates:
        return []
    k = max(1, min(k, len(candidates)))
    selected: list[RetrievedChunk] = []
    selected_ids: set[str] = set()
    source_counts: dict[str, int] = {}

    def add(hit: RetrievedChunk) -> bool:
        if hit.chunk.chunk_id in selected_ids or len(selected) >= k:
            return False
        selected.append(hit)
        selected_ids.add(hit.chunk.chunk_id)
        source_counts[hit.chunk.source_name] = source_counts.get(hit.chunk.source_name, 0) + 1
        return True

    # Force the best tailings spreadsheet into the context first. It may not be
    # the highest vector hit, but for this domain it is primary plant evidence.
    for hit in candidates:
        if _is_tailings_xlsx(hit):
            add(hit)
            break

    # First pass: keep the strongest evidence while preventing one PDF from
    # occupying the whole context if other sources are available.
    for hit in candidates:
        if len(selected) >= k:
            break
        if source_counts.get(hit.chunk.source_name, 0) >= 2:
            continue
        add(hit)

    # Second pass: fill any remaining slots with the original vector order.
    for hit in candidates:
        if len(selected) >= k:
            break
        add(hit)

    # Re-rank display order by original retrieval rank after forced inclusion,
    # while preserving the selected set.
    selected.sort(key=lambda h: h.rank)
    return [
        RetrievedChunk(chunk=hit.chunk, score=hit.score, rank=i)
        for i, hit in enumerate(selected, start=1)
    ]
