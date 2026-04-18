"""One-shot ingestion CLI.

Usage:
    python -m backend.ingest.ingest

Rebuilds the vector index from ./data/**/*.json and writes it to INDEX_PATH.
Run this whenever the knowledge base changes.
"""
from __future__ import annotations

import shutil
import sys

from backend.config import get_settings
from backend.rag.chunker import load_kb
from backend.rag.embeddings import build_embedder
from backend.rag.retrieval import build_store


def main() -> int:
    s = get_settings()
    print(f"Loading KB from {s.data_root} …")
    chunks = load_kb(s.data_root)
    print(f"  {len(chunks)} chunks loaded")
    if not chunks:
        print("ERROR: no chunks found — check data_root path", file=sys.stderr)
        return 1

    print(f"Building embedder: {s.embeddings_provider} / {s.embeddings_model}")
    embedder = build_embedder(s.embeddings_provider, s.embeddings_model)

    print(f"Embedding {len(chunks)} chunks (dim={embedder.dim}) …")
    vecs = embedder.embed([c.text for c in chunks])

    if s.index_path.exists():
        shutil.rmtree(s.index_path)
    store = build_store(s.vector_store, embedder.dim)
    store.add(chunks, vecs)
    store.save(s.index_path)
    print(f"Index written to {s.index_path} (size={store.size})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
