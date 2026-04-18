"""Pydantic request/response schemas for the chat API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    top_k: Optional[int] = None
    stream: bool = False
    conversation_id: Optional[str] = None


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    section: str
    heading: str
    url: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    conversation_id: Optional[str] = None
    model: str
    retrieval_ms: int
    generation_ms: int


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embeddings_provider: str
    vector_store: str
    index_ready: bool
    num_chunks: int
