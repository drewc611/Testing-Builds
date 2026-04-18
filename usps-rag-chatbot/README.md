# USPS Assistant — Pub 28 & AMS RAG Chatbot

Enterprise-grade Retrieval-Augmented Generation (RAG) chatbot for the United States Postal Service. It answers technical questions grounded in real excerpts from:

- **USPS Publication 28 — Postal Addressing Standards** (October 2024 edition)
- **USPS Address Management System (AMS) API** and related address-quality products (DPV, LACSLink, SuiteLink, eLOT, CASS)

Every factual claim is cited inline with a source chunk id that links back to the original USPS URL.

---

## Project layout

```
usps-rag-chatbot/
├── backend/
│   ├── main.py               FastAPI app: /chat, /chat/stream, /health, static frontend
│   ├── config.py             env-driven settings
│   ├── schemas.py            Pydantic request/response
│   ├── telemetry.py          structured audit logging
│   ├── rag/
│   │   ├── chunker.py        KB JSON → retrieval chunks
│   │   ├── embeddings.py     EmbeddingProvider + sentence-transformers impl
│   │   ├── retrieval.py      VectorStore interface + FAISS + in-memory fallback
│   │   └── pipeline.py       retrieve → prompt → generate
│   ├── llm/
│   │   ├── base.py           LLMProvider interface
│   │   ├── anthropic_provider.py
│   │   ├── openai_provider.py
│   │   └── echo_provider.py  offline stub for CI / no-key dev
│   └── ingest/ingest.py      one-shot index builder CLI
├── data/
│   ├── pub28/                real, cited excerpts from Publication 28
│   └── ams/                  AMS API overview, glossary, coding workflow
├── frontend/
│   └── index.html            single-file Tailwind + vanilla JS chat UI
├── docs/
│   └── ARCHITECTURE.docx     full design document
├── tests/
│   └── test_rag.py           unit tests for chunker, retrieval, pipeline wiring
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick start

```bash
# 1. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Config
cp .env.example .env
# (optional) add ANTHROPIC_API_KEY=... and set LLM_PROVIDER=anthropic
#            — without keys, LLM_PROVIDER=echo runs offline and shows retrieval results

# 3. Build index (first run only — also builds automatically on startup)
python -m backend.ingest.ingest

# 4. Run
uvicorn backend.main:app --reload --port 8000
# open http://localhost:8000
```

`GET /health` reports the active providers and the indexed chunk count.

---

## Switching providers

All provider choices are environment variables — no code change required.

| Variable | Values | Default |
|---|---|---|
| `LLM_PROVIDER` | `anthropic`, `openai`, `echo` | `echo` |
| `EMBEDDINGS_PROVIDER` | `sentence-transformers`, `echo` | `sentence-transformers` |
| `VECTOR_STORE` | `faiss`, `memory` | `faiss` |

The pipeline builds against the `LLMProvider`, `EmbeddingProvider`, and `VectorStore` interfaces in `backend/`, so adding Pinecone, pgvector, Bedrock, etc., is a new class plus a factory line.

---

## How the RAG works

1. **Ingest.** `backend/ingest/ingest.py` walks `data/**/*.json`, emits retrieval chunks with full provenance (section, heading, URL), embeds them, and persists a FAISS index plus pickled metadata.
2. **Retrieve.** User queries are embedded with the same model, FAISS returns top-K hits ranked by cosine similarity, weak hits below `MIN_SCORE` are dropped (with a fallback so the LLM can say "coverage thin" rather than nothing).
3. **Prompt.** Retrieved chunks become labeled blocks prefixed with their chunk id so the LLM can cite them inline (`[pub28-213-02]`). The system prompt restricts Pub 28/AMS claims to the provided context and requires citations.
4. **Generate.** The LLM streams tokens via Server-Sent Events; the frontend renders tokens live, then renders citation cards linking back to `pe.usps.com` / `postalpro.usps.com`.

---

## Adding full-document PDFs

The KB also ingests PDFs dropped anywhere under `data/`. Pair each PDF with a
sidecar `<stem>.meta.json` so citations link back to the authoritative URL:

```
data/pub28/source/pub28.pdf
data/pub28/source/pub28.meta.json
```

```json
{
  "doc_id": "pub28-full",
  "title": "Publication 28 — Postal Addressing Standards",
  "url": "https://pe.usps.com/text/pub28/welcome.htm"
}
```

Each page becomes one chunk (long pages split on sentence boundaries, keyed
`<doc_id>-p###-<n>`). Re-run `python -m backend.ingest.ingest` to re-index.
PDFs without a sidecar still index, but citations show the filename instead
of a URL.

---

## Testing

```bash
pytest -q
```

The tests exercise KB loading, chunking, the in-memory vector store, and the prompt-assembly path — no network or API keys needed.

---

## Sources (APA)

U.S. Postal Service. (2024, October). *Publication 28: Postal addressing standards* (PSN 7610-03-000-3688). Postal Explorer. https://pe.usps.com/text/pub28/welcome.htm

U.S. Postal Service. (2024). Chapter 2: Postal addressing standards. In *Publication 28: Postal addressing standards*. https://pe.usps.com/text/pub28/28c2_001.htm

U.S. Postal Service. (2024). Section 213: Secondary address unit designators. In *Publication 28: Postal addressing standards*. https://pe.usps.com/text/pub28/28c2_003.htm

U.S. Postal Service. (2024). Appendix B: Two-letter state and possession abbreviations. In *Publication 28: Postal addressing standards*. https://pe.usps.com/text/pub28/28apb.htm

U.S. Postal Service. (n.d.). *Address Matching System API (AMS API)*. PostalPro. https://postalpro.usps.com/address-quality/ams-api

U.S. Postal Service. (n.d.). *Address quality solutions*. PostalPro. https://postalpro.usps.com/address-quality

U.S. Postal Service. (n.d.). *Address management products*. RIBBS. https://ribbs.usps.gov/address_manage_products/address_manage_products_print.htm

---

## License and attribution

This prototype is not affiliated with or endorsed by the United States Postal Service. All USPS documentation is the property of the U.S. Postal Service; excerpts here are reproduced for grounded retrieval and cite back to the authoritative source.
