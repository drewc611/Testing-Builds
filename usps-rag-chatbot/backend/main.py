"""FastAPI entry point.

Endpoints:
  GET  /health          — liveness and configuration
  POST /chat            — single-shot chat with citations
  POST /chat/stream     — server-sent events, streamed tokens + trailing citations
  GET  /                — serves the bundled frontend (single-file HTML)
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from backend.config import get_settings
from backend.llm.base import build_llm
from backend.rag.pipeline import answer, bootstrap, build_prompt, retrieve, to_citations
from backend.schemas import ChatRequest, ChatResponse, HealthResponse
from backend.telemetry import audit, configure_logging, timer


settings = get_settings()
configure_logging(settings.log_level)
_log = logging.getLogger("usps_rag")

app = FastAPI(title="USPS RAG Chatbot", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Build pipeline and LLM once at startup.
_state = bootstrap(settings)
_llm = build_llm(
    settings.llm_provider,
    anthropic_api_key=settings.anthropic_api_key,
    anthropic_model=settings.anthropic_model,
    openai_api_key=settings.openai_api_key,
    openai_model=settings.openai_model,
    ollama_base_url=settings.ollama_base_url,
    ollama_model=settings.ollama_model,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_provider=_llm.name,
        embeddings_provider=settings.embeddings_provider,
        vector_store=settings.vector_store,
        index_ready=_state.store.size > 0,
        num_chunks=_state.store.size,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(400, "messages required")
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
    if not user_msg:
        raise HTTPException(400, "at least one user message required")

    conv_id = req.conversation_id or str(uuid.uuid4())
    text, citations, r_ms, g_ms = answer(_state, _llm, user_msg, req.top_k)
    audit(
        "chat",
        conversation_id=conv_id,
        model=_llm.model,
        retrieval_ms=r_ms,
        generation_ms=g_ms,
        chunks=[c.chunk_id for c in citations],
    )
    return ChatResponse(
        answer=text,
        citations=citations,
        conversation_id=conv_id,
        model=_llm.model,
        retrieval_ms=r_ms,
        generation_ms=g_ms,
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if not req.messages:
        raise HTTPException(400, "messages required")
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
    if not user_msg:
        raise HTTPException(400, "at least one user message required")
    conv_id = req.conversation_id or str(uuid.uuid4())

    def gen() -> AsyncIterator[bytes]:
        with timer() as rt:
            hits = retrieve(_state, user_msg, req.top_k)
        messages = build_prompt(user_msg, hits, settings.max_context_chars)
        citations = to_citations(hits)
        yield _sse("meta", {"conversation_id": conv_id, "retrieval_ms": rt["ms"], "model": _llm.model})
        try:
            for delta in _llm.stream(messages):
                if delta:
                    yield _sse("token", {"text": delta})
        except Exception:
            # Log full details server-side; return a generic message to the client
            # to avoid leaking internal errors, stack traces, or secrets.
            _log.exception("stream generation failed", extra={"conversation_id": conv_id})
            yield _sse("error", {"message": "generation failed"})
            return
        yield _sse("citations", [c.model_dump() for c in citations])
        yield _sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(event: str, data) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


# Frontend bundle (single-file HTML)
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "index.html"


@app.get("/")
def root():
    if _FRONTEND.exists():
        return FileResponse(_FRONTEND)
    return {"message": "frontend not found"}
