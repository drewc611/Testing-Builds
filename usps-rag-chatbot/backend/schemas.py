"""Pydantic request/response schemas for the chat API."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


MAX_MESSAGE_CHARS = 16_000
MAX_MESSAGES = 64
MAX_CONVERSATION_ID_CHARS = 128


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=MAX_MESSAGE_CHARS)


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., max_length=MAX_MESSAGES)
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    stream: bool = False
    conversation_id: Optional[str] = Field(default=None, max_length=MAX_CONVERSATION_ID_CHARS)


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
