"""Embedding provider interface + implementations.

Default: sentence-transformers (local, no network, free).
Echo: deterministic hashed bag-of-words; used for CI / offline testing when
sentence-transformers is not installed.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingProvider(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return (n, dim) float32 array of unit-normalized vectors."""


class SentenceTransformersEmbeddings(EmbeddingProvider):
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return vecs.astype("float32")


class EchoEmbeddings(EmbeddingProvider):
    """Hash-based embedding, deterministic, no deps. Not semantically meaningful;
    use only for offline wiring tests."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for token in t.lower().split():
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


def build_embedder(provider: str, model: str) -> EmbeddingProvider:
    if provider == "sentence-transformers":
        return SentenceTransformersEmbeddings(model)
    if provider == "echo":
        return EchoEmbeddings()
    raise ValueError(f"Unknown embeddings provider: {provider}")
