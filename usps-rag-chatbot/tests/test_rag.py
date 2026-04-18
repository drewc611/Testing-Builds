"""Hermetic tests for the RAG pipeline.

These run without network or API keys: they use the EchoEmbeddings hashing
provider and the InMemoryStore fallback, so the full retrieve → prompt path
is exercised in CI without heavy dependencies.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.ingest.pdf_loader import build_pdf_chunks, load_sidecar
from backend.rag.chunker import load_kb, as_context_block
from backend.rag.embeddings import EchoEmbeddings
from backend.rag.pipeline import SYSTEM_PROMPT
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


def test_pdf_chunks_carry_provenance():
    meta = {
        "doc_id": "pub28-full",
        "title": "Publication 28 — Postal Addressing Standards",
        "url": "https://pe.usps.com/text/pub28/welcome.htm",
    }
    pages = [
        (1, "Standardized Address\nAll addresses must follow the format…"),
        (2, "Secondary Unit Designators\nUse APT, STE, BLDG…"),
        (3, ""),  # blank page — should be skipped
    ]
    chunks = build_pdf_chunks(pages, meta, default_doc_id="pub28-full")
    assert len(chunks) == 2, "blank pages must not produce chunks"
    assert chunks[0].chunk_id == "pub28-full-p001"
    assert chunks[1].chunk_id == "pub28-full-p002"
    for c in chunks:
        assert c.doc_id == "pub28-full"
        assert c.url.startswith("https://")
        assert c.section.startswith("p.")
        assert c.heading  # first-line heading extracted


def test_pdf_long_page_splits_with_suffix():
    from backend.rag.chunker import MAX_CHUNK_CHARS

    long_text = ("Section A. " + "word " * 400 + "End.")
    assert len(long_text) > MAX_CHUNK_CHARS
    chunks = build_pdf_chunks([(7, long_text)], meta={}, default_doc_id="doc")
    assert len(chunks) >= 2
    ids = [c.chunk_id for c in chunks]
    assert ids[0] == "doc-p007-0"
    assert ids[1].startswith("doc-p007-1")


def test_pdf_sidecar_defaults_when_missing(tmp_path):
    pdf = tmp_path / "some_doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")  # contents irrelevant; sidecar loader doesn't read
    assert load_sidecar(pdf) == {}

    (tmp_path / "some_doc.meta.json").write_text(
        '{"doc_id": "x", "title": "X", "url": "https://example.com"}'
    )
    assert load_sidecar(pdf)["doc_id"] == "x"


def test_load_kb_merges_json_and_pdf_chunks():
    chunks = load_kb(DATA)
    json_chunks = [c for c in chunks if not c.section.startswith("p.")]
    pdf_chunks = [c for c in chunks if c.section.startswith("p.")]
    # Always have curated JSON excerpts
    assert len(json_chunks) > 5
    # PDFs are optional; when present, both paths contribute and IDs don't collide
    if pdf_chunks:
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "chunk ids must be unique across JSON + PDF"


def test_build_llm_wires_ollama_provider():
    from backend.llm.base import build_llm

    llm = build_llm(
        "ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1:8b",
    )
    assert llm.name == "ollama"
    assert llm.model == "llama3.1:8b"


def test_system_prompt_enforces_grounded_and_general_split():
    p = SYSTEM_PROMPT
    # Citation requirement
    assert "[pub28-213-02]" in p or "[pub28-" in p
    assert "[cass-cycle-o-" in p, "prompt must reference CASS chunk id format"
    # Explicit grounded/general labels
    assert "GROUNDED" in p and "GENERAL" in p
    assert "General guidance" in p
    # Authoritative source pointers for fallbacks
    assert "pe.usps.com" in p
    assert "postalpro.usps.com" in p
    # Technical vocabulary guidance
    for term in ("DPV", "LACSLink", "ZIP+4", "NDC", "IMb"):
        assert term in p, f"prompt missing technical term: {term}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
