"""
Build a persistent corpus index from a directory of documents.

Usage
-----
    python -m coscientist.corpus.build --data-dir data --index-path ~/.coscientist/corpus/index
    python -m coscientist.corpus.build --data-dir data --no-images   # skip VLM image parsing

Requires ``ROUTER_AI_API_KEY`` when run for real; both the embedder and the VLM
are injectable for offline testing.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable, Optional, Sequence

from coscientist.corpus.loaders import CorpusChunk, iter_corpus_files, load_file
from coscientist.corpus.store import CorpusIndex, EmbedFn


def build_corpus_index(
    data_dir: str,
    index_path: str,
    *,
    include_images: bool = True,
    embed_fn: Optional[EmbedFn] = None,
    vlm_fn: Optional[Callable[[str], str]] = None,
    chunk_size: int = 1200,
    files: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> CorpusIndex:
    """
    Walk ``data_dir``, load every supported file into chunks, embed them, and
    save the index to ``index_path``.

    Parameters
    ----------
    include_images : bool
        Whether to parse images with the VLM.
    embed_fn, vlm_fn : optional
        Injected embedding / image-to-text callables. When omitted, the central
        model factory (RouterAI / API) is used. Injecting stubs keeps builds
        offline for tests.
    files : optional
        Explicit list of files to ingest instead of walking ``data_dir``.
    """
    paths = list(files) if files is not None else list(iter_corpus_files(data_dir))
    index = CorpusIndex(embed_fn=embed_fn)

    total_chunks = 0
    for path in paths:
        chunks: list[CorpusChunk] = load_file(
            path,
            include_images=include_images,
            vlm_fn=vlm_fn,
            chunk_size=chunk_size,
        )
        added = index.add(chunks)
        total_chunks += added
        if verbose:
            print(f"  {os.path.basename(path)}: +{added} chunks")

    if index_path:
        os.makedirs(os.path.dirname(os.path.abspath(index_path)), exist_ok=True)
        index.save(index_path)
        if verbose:
            print(f"Saved index with {total_chunks} chunks to {index_path}")
    return index


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m coscientist.corpus.build",
        description="Build a private-corpus RAG index from a document directory.",
    )
    parser.add_argument(
        "--data-dir", default="data", help="Directory of documents to ingest."
    )
    parser.add_argument(
        "--index-path",
        default=os.path.join(
            os.path.expanduser(os.environ.get("COSCIENTIST_DIR", "~/.coscientist")),
            "corpus",
            "index",
        ),
        help="Output index path (without extension).",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip VLM image parsing.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1200, help="Target characters per chunk."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    index = build_corpus_index(
        data_dir=args.data_dir,
        index_path=args.index_path,
        include_images=not args.no_images,
        chunk_size=args.chunk_size,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print(f"Done. Index contains {len(index)} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
