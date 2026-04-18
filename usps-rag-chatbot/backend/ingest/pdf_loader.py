"""PDF ingestion: extract text per page, split into Chunks with provenance.

Drop a PDF under ``data/`` (e.g. ``data/pub28/source/pub28.pdf``). Optionally
place a sidecar ``<stem>.meta.json`` next to it to set ``doc_id``, ``title``,
and ``url`` for citations:

    { "doc_id": "pub28-full", "title": "Publication 28 …", "url": "https://…" }

Without a sidecar, metadata falls back to the filename stem and an empty URL.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from backend.rag.chunker import Chunk, _split_long


def extract_pages(path: Path) -> List[Tuple[int, str]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(i + 1, (page.extract_text() or "")) for i, page in enumerate(reader.pages)]


def load_sidecar(pdf_path: Path) -> Dict:
    meta_path = pdf_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_line(text: str, max_len: int = 120) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:max_len]
    return ""


def build_pdf_chunks(
    pages: List[Tuple[int, str]],
    meta: Dict,
    default_doc_id: str,
) -> List[Chunk]:
    doc_id = meta.get("doc_id", default_doc_id)
    title = meta.get("title", default_doc_id)
    url = meta.get("url", "")

    out: List[Chunk] = []
    for page_num, raw in pages:
        text = (raw or "").strip()
        if not text:
            continue
        pieces = _split_long(text)
        for i, piece in enumerate(pieces):
            suffix = f"-{i}" if len(pieces) > 1 else ""
            out.append(
                Chunk(
                    chunk_id=f"{doc_id}-p{page_num:03d}{suffix}",
                    doc_id=doc_id,
                    title=title,
                    section=f"p.{page_num}",
                    heading=_first_line(piece),
                    url=url,
                    text=piece,
                )
            )
    return out


def load_pdfs(data_root: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(Path(data_root).rglob("*.pdf")):
        meta = load_sidecar(path)
        pages = extract_pages(path)
        chunks.extend(build_pdf_chunks(pages, meta, default_doc_id=path.stem))
    return chunks
