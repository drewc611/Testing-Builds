"""RAG pipeline: retrieve → rerank (optional) → generate with citations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from backend.config import Settings
from backend.llm.base import LLMProvider, LLMMessage
from backend.rag.chunker import Chunk, as_context_block, load_kb
from backend.rag.embeddings import EmbeddingProvider, build_embedder
from backend.rag.retrieval import VectorStore, build_store
from backend.schemas import Citation


SYSTEM_PROMPT = """You are a technical assistant for the United States Postal Service.

You serve USPS employees, business mailers, software developers, and the public
with questions across:
  • USPS Publication 28 — Postal Addressing Standards (full text + curated excerpts)
  • CASS Technical Guide — the coding-accuracy standards that AMS-certified
    software must implement (ZIP+4, DPV, LACSLink, SuiteLink, eLOT return codes)
  • USPS Address Management System (AMS) API and address-quality products
  • Broader USPS technical topics: mail classes, routing hierarchy (NDC/ADC/SCF),
    barcodes (IMb, IMpb), Intelligent Mail, addressing workflows

Answer rules:
  1. GROUNDED claims — Pub 28 sections, CASS/DPV/LACSLink/SuiteLink specifics,
     AMS return codes, exact addressing standards — must come from the provided
     context. Cite inline with the chunk id in brackets, e.g. [pub28-213-02] or
     [cass-cycle-o-p045]. Never invent section numbers, URLs, return codes, or
     rate amounts.
  2. If the context does not contain a needed fact, say so plainly and point to
     the authoritative source: pe.usps.com for Pub 28 / DMM, postalpro.usps.com
     for CASS and AMS API, about.usps.com for operational handbooks.
  3. GENERAL guidance (current rates, hours, tracking flows, non-addressing ops)
     may draw on general knowledge. When it does, label that portion on its own
     line: "General guidance (not from Pub 28 / CASS / AMS): …".
  4. When an answer mixes both, structure it as:
        **Grounded (Pub 28 / CASS / AMS):** …citations…
        **General guidance:** …
  5. Be technically precise. Use correct terminology (ZIP+4, DPV confirmation,
     LACSLink indicator, carrier route, NDC, ADC, SCF, IMb, CRID, MID).
     Prefer bulleted lists for rules and tables for abbreviations or return codes.
"""


@dataclass
class PipelineState:
    embedder: EmbeddingProvider
    store: VectorStore
    settings: Settings


def bootstrap(settings: Settings) -> PipelineState:
    embedder = build_embedder(settings.embeddings_provider, settings.embeddings_model)
    store = build_store(settings.vector_store, embedder.dim)
    # Try cached index first
    if not store.load(settings.index_path):
        chunks = load_kb(settings.data_root)
        if not chunks:
            raise RuntimeError(f"No KB chunks found under {settings.data_root}")
        texts = [c.text for c in chunks]
        vecs = embedder.embed(texts)
        store.add(chunks, vecs)
        store.save(settings.index_path)
    return PipelineState(embedder=embedder, store=store, settings=settings)


def retrieve(state: PipelineState, query: str, k: int | None = None) -> List[Tuple[Chunk, float]]:
    k = k or state.settings.top_k
    qv = state.embedder.embed([query])[0]
    hits = state.store.search(qv, k)
    return [(c, s) for c, s in hits if s >= state.settings.min_score] or hits[:1]


def build_prompt(query: str, hits: List[Tuple[Chunk, float]], max_chars: int) -> List[LLMMessage]:
    context = as_context_block([c for c, _ in hits])
    if len(context) > max_chars:
        context = context[:max_chars] + "\n…[truncated]"
    user = (
        f"Question: {query}\n\n"
        f"Context (retrieved passages — cite with chunk ids in brackets):\n"
        f"{context}\n\n"
        f"Answer the question above using ONLY the context when the question is about Pub 28 or AMS. "
        f"Cite chunk ids inline. If insufficient information, say so."
    )
    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=user),
    ]


def to_citations(hits: List[Tuple[Chunk, float]]) -> List[Citation]:
    out: List[Citation] = []
    for c, s in hits:
        snippet = c.text if len(c.text) <= 240 else c.text[:237] + "…"
        out.append(
            Citation(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                title=c.title,
                section=c.section,
                heading=c.heading,
                url=c.url,
                score=round(s, 4),
                snippet=snippet,
            )
        )
    return out


def answer(state: PipelineState, llm: LLMProvider, query: str, k: int | None) -> Tuple[str, List[Citation], int, int]:
    from backend.telemetry import timer

    with timer() as rt:
        hits = retrieve(state, query, k)
    messages = build_prompt(query, hits, state.settings.max_context_chars)
    with timer() as gt:
        text = llm.complete(messages)
    return text, to_citations(hits), rt["ms"], gt["ms"]
