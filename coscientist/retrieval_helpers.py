"""
Shared helper for obtaining a :class:`CorpusRetriever`.

Both the one-command runner (`scripts/run_pipeline.py`) and the multi-agent
framework use this so the whole system points at a single corpus. Either load a
prebuilt index from disk, or build one from a documents directory on the fly.
"""

from __future__ import annotations

import os
from typing import Optional

from coscientist.corpus.retrieval import CorpusRetriever
from coscientist.corpus.store import EmbedFn


def build_retriever(
    index_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    *,
    default_k: int = 6,
    embed_fn: Optional[EmbedFn] = None,
    include_images: bool = True,
    vlm_fn=None,
) -> CorpusRetriever:
    """
    Return a retriever, loading a saved index or building one from documents.

    Parameters
    ----------
    index_path : str | None
        Path to a saved index (``.json`` sidecar). If it exists, it is loaded.
        If given but missing and ``data_dir`` is set, the index is built here.
    data_dir : str | None
        Directory of documents to ingest when no saved index is available.
    embed_fn, vlm_fn : optional
        Injected embedding / image-to-text callables (offline-testable).

    Raises
    ------
    ValueError
        If neither a loadable index nor a ``data_dir`` to build from is given.
    """
    if index_path and os.path.exists(_json_sidecar(index_path)):
        return CorpusRetriever.from_path(
            index_path, default_k=default_k, embed_fn=embed_fn
        )

    if data_dir:
        # Local import avoids importing the build CLI unless we actually build.
        from coscientist.corpus.build import build_corpus_index

        index = build_corpus_index(
            data_dir=data_dir,
            index_path=index_path or "",
            include_images=include_images,
            embed_fn=embed_fn,
            vlm_fn=vlm_fn,
        )
        return CorpusRetriever(index, default_k=default_k)

    raise ValueError(
        "build_retriever needs either an existing index_path or a data_dir to build from."
    )


def _json_sidecar(index_path: str) -> str:
    return index_path if index_path.endswith(".json") else index_path + ".json"
