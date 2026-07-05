"""
Vector index for the private corpus.

A small, dependency-light index: chunk embeddings are stored in a NumPy matrix
and queried by cosine similarity (scikit-learn). Sufficient at case scale and
trivially persistable. Embeddings are produced by an injectable ``embed_fn`` so
the search/persistence logic is testable without network access.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from coscientist.corpus.loaders import CorpusChunk

EmbedFn = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@dataclass
class RetrievedChunk:
    """A search hit: the matched chunk plus its similarity score and rank."""

    chunk: CorpusChunk
    score: float
    rank: int


def _default_embed_fn() -> EmbedFn:
    """Build the default embedding function via the central model factory."""
    from coscientist.model_factory import get_embeddings

    embeddings = get_embeddings()
    return embeddings.embed_documents


class CorpusIndex:
    """An in-memory cosine-similarity index over corpus chunks."""

    def __init__(self, embed_fn: Optional[EmbedFn] = None):
        # embed_fn resolution is deferred so constructing an empty index needs no keys.
        self._embed_fn = embed_fn
        self.chunks: list[CorpusChunk] = []
        self._vectors: Optional[np.ndarray] = None

    # -- embedding ---------------------------------------------------------

    @property
    def embed_fn(self) -> EmbedFn:
        if self._embed_fn is None:
            self._embed_fn = _default_embed_fn()
        return self._embed_fn

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self.embed_fn(list(texts)), dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embed_fn must return a 2D sequence of vectors")
        return vectors

    # -- building ----------------------------------------------------------

    def add(self, chunks: Sequence[CorpusChunk]) -> int:
        """Embed and add chunks to the index. Returns the number added."""
        chunks = [c for c in chunks if c.text and c.text.strip()]
        if not chunks:
            return 0
        new_vectors = self._embed([c.text for c in chunks])
        if new_vectors.shape[0] != len(chunks):
            raise ValueError("embed_fn returned a mismatched number of vectors")
        self.chunks.extend(chunks)
        if self._vectors is None:
            self._vectors = new_vectors
        else:
            if new_vectors.shape[1] != self._vectors.shape[1]:
                raise ValueError("embedding dimensionality mismatch")
            self._vectors = np.vstack([self._vectors, new_vectors])
        return len(chunks)

    # -- querying ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        """Return the top-``k`` chunks most similar to ``query``."""
        if not self.chunks or self._vectors is None:
            return []
        k = max(1, min(k, len(self.chunks)))
        query_vec = self._embed([query])
        sims = cosine_similarity(query_vec, self._vectors)[0]
        top_idx = np.argsort(-sims)[:k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(sims[i]), rank=rank)
            for rank, i in enumerate(top_idx, start=1)
        ]

    # -- persistence -------------------------------------------------------

    def save(self, path: str) -> str:
        """Persist the index to ``<path>.npy`` + ``<path>.json`` (deterministic)."""
        base = _strip_ext(path)
        vectors = (
            self._vectors
            if self._vectors is not None
            else np.zeros((0, 0), dtype=np.float32)
        )
        np.save(base + ".npy", vectors)
        payload = {
            "version": 1,
            "chunks": [_chunk_to_dict(c) for c in self.chunks],
        }
        with open(base + ".json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return base

    @classmethod
    def load(cls, path: str, *, embed_fn: Optional[EmbedFn] = None) -> "CorpusIndex":
        """Load an index previously written by :meth:`save`."""
        base = _strip_ext(path)
        index = cls(embed_fn=embed_fn)
        with open(base + ".json", encoding="utf-8") as fh:
            payload = json.load(fh)
        index.chunks = [_chunk_from_dict(d) for d in payload.get("chunks", [])]
        vectors = np.load(base + ".npy")
        index._vectors = vectors if vectors.size else None
        return index


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _strip_ext(path: str) -> str:
    for ext in (".json", ".npy"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def _chunk_to_dict(chunk: CorpusChunk) -> dict:
    return {
        "text": chunk.text,
        "source_path": chunk.source_path,
        "source_name": chunk.source_name,
        "modality": chunk.modality,
        "locator": chunk.locator,
        "chunk_id": chunk.chunk_id,
        "metadata": chunk.metadata,
    }


def _chunk_from_dict(d: dict) -> CorpusChunk:
    return CorpusChunk(
        text=d["text"],
        source_path=d["source_path"],
        source_name=d["source_name"],
        modality=d["modality"],
        locator=d["locator"],
        chunk_id=d.get("chunk_id", ""),
        metadata=d.get("metadata", {}),
    )
