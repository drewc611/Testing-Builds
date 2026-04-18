"""Hermetic tests for the RAG pipeline.

These run without network or API keys: they use the EchoEmbeddings hashing
provider and the InMemoryStore fallback, so the full retrieve → prompt path
is exercised in CI without heavy dependencies.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.rag.chunker import load_kb, as_context_block
from backend.rag.embeddings import EchoEmbeddings
from backend.rag.retrieval import InMemoryStore


DATA = Path(__file__).resolve().parent.parent / "data"


def test_kb_loads_nonempty():
    chunks = load_kb(DATA)
    assert len(chunks) > 5
    # Every chunk has provenance
    for c in chunks:
        assert c.chunk_id
        assert c.url.startswith("https://")
        assert c.text.strip()


def test_chunks_have_expected_sections():
    chunks = load_kb(DATA)
    ids = {c.chunk_id for c in chunks}
    # known excerpts we grounded in real USPS sources
    assert any(i.startswith("pub28-213") for i in ids), "missing 213 Secondary Unit Designators"
    assert any(i.startswith("pub28-appb") for i in ids), "missing Appendix B state abbreviations"
    assert any(i.startswith("ams-api") for i in ids), "missing AMS API overview"


def test_context_block_renders_citation_headers():
    chunks = load_kb(DATA)[:3]
    block = as_context_block(chunks)
    for c in chunks:
        assert f"[{c.chunk_id}]" in block
        assert c.url in block


def test_inmemory_store_roundtrip():
    chunks = load_kb(DATA)
    emb = EchoEmbeddings(dim=128)
    vecs = emb.embed([c.text for c in chunks])
    store = InMemoryStore(dim=128)
    store.add(chunks, vecs)
    assert store.size == len(chunks)

    qv = emb.embed(["secondary unit designator suite apartment"])[0]
    hits = store.search(qv, k=5)
    assert len(hits) == 5
    # Top hits for a secondary-unit query should include section 213
    top_ids = {c.chunk_id for c, _ in hits}
    assert any("213" in cid for cid in top_ids), f"expected 213 chunk in {top_ids}"


def test_every_doc_has_source_url():
    chunks = load_kb(DATA)
    doc_urls = {c.doc_id: c.url for c in chunks}
    for doc_id, url in doc_urls.items():
        assert url.startswith("https://"), f"{doc_id} has no source URL"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
