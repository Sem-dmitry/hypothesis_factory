"""
Unit tests for the private-corpus RAG subsystem.

No network and no real API keys: embeddings and the VLM are injected as
deterministic fakes, and the document loaders run on the real files under
``data/``. Run with only a dummy ROUTER_AI_API_KEY set.
"""

import glob
import math
import os

import pytest

from coscientist.corpus import citations, loaders, store
from coscientist.corpus.build import build_corpus_index
from coscientist.corpus.loaders import CorpusChunk
from coscientist.corpus.store import CorpusIndex

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

# A tiny fixed vocabulary makes the fake embedder produce meaningful cosine
# similarities driven by word overlap.
_VOCAB = [
    "flotation",
    "grinding",
    "hydrocyclone",
    "nickel",
    "tailings",
    "mill",
    "reagent",
    "classifier",
]


def fake_embed(texts):
    """Deterministic bag-of-words embedding over a fixed vocabulary."""
    vectors = []
    for text in texts:
        low = text.lower()
        vec = [float(low.count(word)) for word in _VOCAB]
        # add a tiny constant so all-zero texts still have a defined direction
        vec.append(1.0)
        vectors.append(vec)
    return vectors


def _smallest(pattern):
    matches = glob.glob(os.path.join(DATA_DIR, "**", pattern), recursive=True)
    matches = [m for m in matches if os.path.isfile(m)]
    if not matches:
        return None
    return min(matches, key=os.path.getsize)


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def test_chunk_text_basic():
    assert loaders.chunk_text("") == []
    assert loaders.chunk_text("short") == ["short"]
    text = ("word " * 1000).strip()
    chunks = loaders.chunk_text(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # size + a little slack from separators


def test_corpus_chunk_id_is_stable():
    c1 = CorpusChunk(text="abc", source_path="/x/y.pdf", source_name="y.pdf",
                     modality="pdf", locator="p.1")
    c2 = CorpusChunk(text="abc", source_path="/x/y.pdf", source_name="y.pdf",
                     modality="pdf", locator="p.1")
    assert c1.chunk_id == c2.chunk_id and c1.chunk_id


# ---------------------------------------------------------------------------
# loaders on real data
# ---------------------------------------------------------------------------


def test_load_docx_real():
    path = _smallest("*.docx")
    assert path, "expected a .docx under data/"
    chunks = loaders.load_docx(path)
    assert chunks, "docx produced no chunks"
    c = chunks[0]
    assert c.modality == "docx"
    assert c.source_name == os.path.basename(path)
    assert c.text.strip()


def test_load_xlsx_real():
    path = _smallest("*.xlsx")
    assert path, "expected a .xlsx under data/"
    chunks = loaders.load_xlsx(path)
    assert chunks, "xlsx produced no chunks"
    c = chunks[0]
    assert c.modality == "xlsx"
    assert c.locator.startswith("sheet ")
    assert "row" in c.metadata.get("sheet", "") or "sheet" in c.metadata


def test_load_pdf_real():
    path = _smallest("*.pdf")
    assert path, "expected a .pdf under data/"
    chunks = loaders.load_pdf(path)
    assert chunks, "pdf produced no chunks"
    c = chunks[0]
    assert c.modality == "pdf"
    assert c.locator.startswith("p.")
    assert c.metadata.get("page", 0) >= 1


# ---------------------------------------------------------------------------
# VLM (injected fake client)
# ---------------------------------------------------------------------------


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeVisionClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _FakeMessage(self.reply)


def test_describe_image_with_injected_client():
    from coscientist.corpus import vlm

    path = _smallest("*.png")
    assert path, "expected a .png under data/"
    client = _FakeVisionClient("A flotation flowsheet with mills and hydrocyclones.")
    desc = vlm.describe_image(path, client=client)
    assert "flotation" in desc.lower()
    # a multimodal message with an image_url part was constructed
    parts = client.calls[0][0]["content"]
    assert any(p.get("type") == "image_url" for p in parts)
    assert parts[-1]["image_url"]["url"].startswith("data:image/")


def test_load_image_uses_vlm_fn():
    path = _smallest("*.png")
    chunks = loaders.load_image(path, vlm_fn=lambda p: "grinding mill schematic")
    assert len(chunks) == 1
    assert chunks[0].modality == "image"
    assert chunks[0].text == "grinding mill schematic"


# ---------------------------------------------------------------------------
# index: add / search / save / load
# ---------------------------------------------------------------------------


def _sample_chunks():
    return [
        CorpusChunk(text="flotation reagent dosing in the mill", source_path="a",
                    source_name="a.pdf", modality="pdf", locator="p.1"),
        CorpusChunk(text="hydrocyclone classifier tuning for tailings", source_path="b",
                    source_name="b.pdf", modality="pdf", locator="p.2"),
        CorpusChunk(text="nickel recovery from grinding circuit", source_path="c",
                    source_name="c.pdf", modality="pdf", locator="p.3"),
    ]


def test_index_add_and_search():
    idx = CorpusIndex(embed_fn=fake_embed)
    added = idx.add(_sample_chunks())
    assert added == 3
    assert len(idx) == 3
    hits = idx.search("hydrocyclone classifier for tailings", k=2)
    assert len(hits) == 2
    assert hits[0].rank == 1
    assert hits[0].chunk.locator == "p.2"  # the hydrocyclone/classifier chunk
    assert hits[0].score >= hits[1].score


def test_index_skips_empty_text():
    idx = CorpusIndex(embed_fn=fake_embed)
    added = idx.add([CorpusChunk(text="   ", source_path="a", source_name="a",
                                 modality="pdf", locator="p.1")])
    assert added == 0
    assert idx.search("anything", k=3) == []


def test_index_save_load_roundtrip(tmp_path):
    idx = CorpusIndex(embed_fn=fake_embed)
    idx.add(_sample_chunks())
    base = idx.save(str(tmp_path / "index"))
    assert os.path.exists(base + ".npy") and os.path.exists(base + ".json")

    loaded = CorpusIndex.load(str(tmp_path / "index"), embed_fn=fake_embed)
    assert len(loaded) == 3
    hits = loaded.search("nickel grinding circuit", k=1)
    assert hits[0].chunk.locator == "p.3"


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------


def test_citation_formatting():
    idx = CorpusIndex(embed_fn=fake_embed)
    idx.add(_sample_chunks())
    hits = idx.search("flotation reagent", k=2)
    refs = citations.format_references(hits)
    assert "[1]" in refs and "—" in refs
    block = citations.format_context_block(hits, max_chars_per_chunk=50)
    assert "Sources:" in block
    assert block.count("[1]") >= 2  # marker in both excerpt and reference list


# ---------------------------------------------------------------------------
# end-to-end build with fakes on real files
# ---------------------------------------------------------------------------


def test_build_corpus_index_end_to_end(tmp_path):
    files = [p for p in (_smallest("*.docx"), _smallest("*.xlsx")) if p]
    assert files, "need at least one real doc under data/"
    index_path = str(tmp_path / "corpus" / "index")
    idx = build_corpus_index(
        data_dir=DATA_DIR,
        index_path=index_path,
        include_images=False,
        embed_fn=fake_embed,
        files=files,
    )
    assert len(idx) > 0
    assert os.path.exists(index_path + ".json")
    # reload and query
    reloaded = CorpusIndex.load(index_path, embed_fn=fake_embed)
    assert len(reloaded) == len(idx)
    hits = reloaded.search("tailings", k=1)
    assert len(hits) == 1
