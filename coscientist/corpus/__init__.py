"""
Private-corpus RAG subsystem for the Co-Scientist "Фабрика гипотез" system.

Ingests a scientist-provided document repository (PDF / DOCX / XLSX / images)
into a searchable, citable index. Images are parsed with a RouterAI vision
model. Everything routes through ``coscientist.model_factory`` (RouterAI / API)
and is dependency-injectable so the whole subsystem is testable offline.
"""

from coscientist.corpus.loaders import CorpusChunk, chunk_text, load_file
from coscientist.corpus.store import CorpusIndex, RetrievedChunk

__all__ = [
    "CorpusChunk",
    "CorpusIndex",
    "RetrievedChunk",
    "chunk_text",
    "load_file",
]
