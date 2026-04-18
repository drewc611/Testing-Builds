"""Convert knowledge-base JSON documents into retrieval chunks.

The KB docs are already pre-chunked by USPS section, which is the natural unit
for Pub 28. We enrich each chunk with document-level metadata so retrieval can
cite provenance. For very long sections (not present today but possible as the
KB grows), we fall back to sliding-window splitting on sentence boundaries.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Dict, Any


MAX_CHUNK_CHARS = 1400
OVERLAP_CHARS = 200


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    heading: str
    url: str
    text: str
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "title": self.title,
            "section": self.section,
            "heading": self.heading,
            "url": self.url,
            "text": self.text,
            "keywords": self.keywords,
        }


def _split_long(text: str) -> List[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 > MAX_CHUNK_CHARS:
            if buf:
                out.append(buf.strip())
            buf = (buf[-OVERLAP_CHARS:] + " " + s) if buf else s
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf:
        out.append(buf.strip())
    return out


def load_kb(data_root: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(Path(data_root).rglob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        url = doc.get("source", {}).get("url", "")
        title = doc.get("title", doc.get("doc_id", path.stem))
        doc_id = doc.get("doc_id", path.stem)
        for raw in doc.get("chunks", []):
            pieces = _split_long(raw["text"])
            for i, piece in enumerate(pieces):
                suffix = f"#{i}" if len(pieces) > 1 else ""
                chunks.append(
                    Chunk(
                        chunk_id=f"{raw['chunk_id']}{suffix}",
                        doc_id=doc_id,
                        title=title,
                        section=raw.get("section", ""),
                        heading=raw.get("heading", ""),
                        url=url,
                        text=piece,
                        keywords=raw.get("keywords", []),
                    )
                )
    return chunks


def as_context_block(chunks: Iterable[Chunk]) -> str:
    """Render retrieved chunks for the LLM prompt, with stable citation keys."""
    blocks = []
    for c in chunks:
        blocks.append(
            f"[{c.chunk_id}] Source: {c.title} — {c.heading} (section {c.section})\n"
            f"URL: {c.url}\n"
            f"{c.text}"
        )
    return "\n\n---\n\n".join(blocks)
