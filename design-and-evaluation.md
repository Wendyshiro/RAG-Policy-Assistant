# Design and Evaluation

## 1. Architecture Overview

The system is a Retrieval-Augmented Generation (RAG) pipeline built over a corpus of 13 HR policy documents. A user question is embedded, matched against a ChromaDB vector index, and the top-k retrieved chunks are injected into a prompt sent to an LLM (Groq) to generate a grounded, cited answer.

```
User Question
     │
     ▼
[Embedding Model]  ←─ all-MiniLM-L6-v2 (local, no API)
     │
     ▼
[ChromaDB Vector Store]  ←─ cosine similarity, persistent on disk
     │  top-k chunks
     ▼
[Prompt Builder]  ←─ chunks + question + system instructions
     │
     ▼
[Groq LLM]  ←─ llama-3.1-70b-versatile via OpenAI-compatible API
     │
     ▼
Answer + Citations
```

---

## 2. Design Decisions

### 2.1 Embedding Model — `all-MiniLM-L6-v2`

`all-MiniLM-L6-v2` from the `sentence-transformers` library was chosen as the embedding model. It runs fully locally with no API key, downloads once (~90 MB) from HuggingFace, and produces 384-dimensional vectors. It is fast enough to embed 233 chunks in under 25 seconds on CPU, which is acceptable for an offline indexing step. For a corpus of this size (13 policy documents), a larger model would not meaningfully improve retrieval quality and would add latency and cost.

### 2.2 Chunking Strategy — Merge-Then-Split

A naive heading-boundary split produced 533 chunks averaging only 259 characters — too small for the retriever to find meaningful signal. The final strategy uses three passes:

1. **Split** the document at `##` and `###` Markdown headings into (heading, body) pairs.
2. **Merge** consecutive sections whose body is under `MIN_CHUNK_CHARS` (400 chars) into a single chunk, joining headings with ` / ` so section context is preserved in metadata.
3. **Sliding-window split** any section still exceeding `CHUNK_CHARS` (1600 chars), with an `OVERLAP` of 200 characters between windows to avoid cutting sentences at boundaries.

This produced **233 chunks averaging 596 characters**, a better balance between retrieval precision and context richness.

| Parameter | Value | Rationale |
|---|---|---|
| `CHUNK_CHARS` | 1600 | ~400 tokens, fits comfortably in context with other chunks |
| `OVERLAP` | 200 | Prevents sentence truncation at window boundaries |
| `MIN_CHUNK_CHARS` | 400 | Merges subsections too small to carry standalone meaning |

### 2.3 Vector Store — ChromaDB (local, persistent)

ChromaDB was chosen for its zero-configuration local mode (`PersistentClient`). It requires no API key, no server process, and no cloud account — the index is written to `./chroma_db/` on disk. The collection is configured with cosine similarity (`hnsw:space: cosine`), which is the correct distance metric for normalised sentence-transformer embeddings. For a corpus of this scale, a local store is sufficient; a managed service would only be warranted at tens of millions of chunks.

### 2.4 LLM — Groq (`llama-3.1-70b-versatile`)

Groq was chosen for its free tier and low inference latency. The model is accessed via Groq's OpenAI-compatible API endpoint, which means the standard `openai` Python SDK can be used by simply pointing `base_url` at `https://api.groq.com/openai/v1`. `llama-3.1-70b-versatile` provides strong instruction-following for grounded Q&A tasks within a policy corpus.

### 2.5 Retrieval — Top-k with Cosine Similarity

The retriever queries ChromaDB for the top `k` chunks (default `k=5`, configurable via `TOP_K` in `.env`) most similar to the embedded user question. No re-ranking step is used at this stage. The retrieved chunk texts and their source metadata are passed directly to the prompt builder.

---

## 3. Ingest Pipeline — Verification Results

The index was verified after build using four spot-check queries. All four returned the correct source document as the top result:

| Query | Top Result Source | Correct? |
|---|---|---|
| How many PTO days does an employee get? | `pto_policy.md` | ✓ |
| What is the remote work internet stipend? | `remote_work_policy.md` | ✓ |
| Password requirements for company systems | `information_security.md` | ✓ |
| Workers compensation coverage | `workplace_safety.md` | ✓ |

**Index summary:**
- Documents ingested: 13
- Total chunks: 233
- Average chunk size: 596 characters
- Embedding time (CPU): ~21 seconds
- Storage: ChromaDB persistent collection `policies`

---
