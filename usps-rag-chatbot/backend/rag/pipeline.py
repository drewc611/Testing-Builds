"""RAG pipeline: retrieve → rerank (optional) → generate with citations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from backend.config import Settings
from backend.llm.base import LLMProvider, LLMMessage
from backend.rag.chunker import Chunk, as_context_block, load_kb
from backend.rag.embeddings import EmbeddingProvider, build_embedder
from backend.rag.retrieval import VectorStore, build_store
from backend.schemas import Citation


SYSTEM_PROMPT = """You are a technical assistant for the United States Postal Service.

You help USPS employees, business mailers, developers, and the general public
with questions about:
  • USPS Publication 28 — Postal Addressing Standards
  • The USPS Address Management System (AMS) and the AMS API
  • Related address-quality products (DPV, LACSLink, SuiteLink, eLOT, CASS)
  • General USPS operational and addressing questions

Rules:
  1. Answer ONLY from the provided context when the question is about Pub 28 or AMS specifics.
     If the context does not contain the answer, say so clearly and recommend the authoritative source.
  2. Cite every factual claim with the chunk id in square brackets, e.g. [pub28-213-02].
  3. When a user asks about general USPS topics NOT in the context (rates, hours, tracking),
     answer with general knowledge and clearly label it as "general guidance — not from Pub 28/AMS".
  4. Be precise. Prefer bulleted lists for rules and abbreviation tables.
  5. Never invent section numbers, URLs, or AMS return codes not present in the context.
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
