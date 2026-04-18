"""Vector store abstraction + FAISS implementation + lightweight in-memory fallback.

Cosine similarity over unit-normalized vectors (FAISS IndexFlatIP). The
abstraction keeps production swaps to Pinecone, pgvector, OpenSearch, etc.,
to a single class.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np

from backend.rag.chunker import Chunk


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[Chunk, float]]: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> bool: ...

    @property
    @abstractmethod
    def size(self) -> int: ...


class FaissStore(VectorStore):
    def __init__(self, dim: int):
        import faiss

        self._faiss = faiss
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("chunks/vectors length mismatch")
        self._index.add(vectors.astype("float32"))
        self._chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[Chunk, float]]:
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_vec.reshape(1, -1).astype("float32"), k)
        return [
            (self._chunks[i], float(scores[0][rank]))
            for rank, i in enumerate(indices[0])
            if i >= 0
        ]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self._chunks], f)
        with open(path / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"dim": self._dim, "count": len(self._chunks)}, f)

    def load(self, path: Path) -> bool:
        if not (path / "index.faiss").exists():
            return False
        chunks_path = path / "chunks.json"
        if not chunks_path.exists():
            return False
        self._index = self._faiss.read_index(str(path / "index.faiss"))
        with open(chunks_path, "r", encoding="utf-8") as f:
            dicts = json.load(f)
        self._chunks = [Chunk(**d) for d in dicts]
        return True

    @property
    def size(self) -> int:
        return self._index.ntotal


class InMemoryStore(VectorStore):
    """Pure-numpy fallback so the app runs even without FAISS installed."""

    def __init__(self, dim: int):
        self._dim = dim
        self._vecs: np.ndarray = np.zeros((0, dim), dtype="float32")
        self._chunks: List[Chunk] = []

    def add(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        self._vecs = np.vstack([self._vecs, vectors.astype("float32")]) if self._vecs.size else vectors.astype("float32")
        self._chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[Chunk, float]]:
        if not self._chunks:
            return []
        scores = self._vecs @ query_vec.astype("float32")
        idx = np.argsort(-scores)[:k]
        return [(self._chunks[i], float(scores[i])) for i in idx]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vecs.npy", self._vecs)
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self._chunks], f)

    def load(self, path: Path) -> bool:
        vp = path / "vecs.npy"
        cp = path / "chunks.json"
        if not (vp.exists() and cp.exists()):
            return False
        self._vecs = np.load(vp, allow_pickle=False)
        with open(cp, "r", encoding="utf-8") as f:
            self._chunks = [Chunk(**d) for d in json.load(f)]
        return True

    @property
    def size(self) -> int:
        return len(self._chunks)


def build_store(kind: str, dim: int) -> VectorStore:
    if kind == "faiss":
        try:
            return FaissStore(dim)
        except ImportError:
            return InMemoryStore(dim)
    if kind == "memory":
        return InMemoryStore(dim)
    raise ValueError(f"Unknown vector store: {kind}")
