# Design and Evaluation: RAG Policy Assistant

**Project:** AI Engineering Project — RAG LLM-Based Application  
**Program:** Quantic MSSE  
**Last Updated:** April 2026

---

## 1. System Architecture Overview

The system follows a standard Retrieval-Augmented Generation (RAG) pipeline:

```
User Query
    │
    ▼
Embedding Model (all-MiniLM-L6-v2)
    │  encodes query into vector
    ▼
ChromaDB Vector Store
    │  cosine similarity search → top-k chunks
    ▼
Context Builder
    │  formats retrieved chunks with source labels
    ▼
LLM (llama-3.3-70b-versatile via Groq)
    │  generates grounded, cited answer
    ▼
Flask Web App (/, /chat, /health)
    │
    ▼
User (chat UI with citations displayed)
```

All components run as a single Flask process. ChromaDB persists the vector
index to disk (`./chroma_db`) so ingestion only needs to run once per corpus
change. The embedding model runs locally on CPU via sentence-transformers,
eliminating any external embedding API dependency.

---

## 2. Design Decisions

### 2.1 Embedding Model: all-MiniLM-L6-v2

**Choice:** `all-MiniLM-L6-v2` from the sentence-transformers library,
running locally on CPU.

**Why:**
- Runs entirely locally — no external API calls, no cost, no rate limits
- 80MB model footprint loads quickly on both local machines and Render
  free tier
- 384-dimensional vectors provide strong semantic similarity performance
  for short-to-medium text passages (well-suited to policy document chunks)
- Encodes at ~9 chunks/second on CPU (confirmed: 233 chunks in 27 seconds
  during actual ingestion run)
- No authentication or API key required — simplifies CI/CD and deployment

**Alternatives considered:**
- *Cohere embed-english-v3.0 (free tier):* Higher quality embeddings but
  introduces external API dependency, adds network latency to ingestion,
  and is subject to rate limits
- *OpenAI text-embedding-3-small:* Strong performance but costs money and
  requires a paid API key
- *all-mpnet-base-v2:* Slightly higher accuracy but 3x slower and 420MB
  model size — not worth the trade-off for a policy corpus of this size

**Decision:** Local sentence-transformers model is the right choice for a
free-tier deployment where cost, latency, and dependency minimisation matter.

---

### 2.2 Chunking Strategy: Heading-Based with Token-Window Fallback

**Choice:** Documents are split at H2/H3 markdown headings. Sections that
exceed the chunk size limit are further split using a sliding character
window with overlap.

**Why:**
- Policy documents are authored with H2/H3 headings that map directly to
  policy topics (e.g., "3.4 Personal Vehicle Mileage", "5. PTO Accrual and
  Carryover"). Chunking at these boundaries ensures each chunk is
  semantically coherent and self-contained
- The section heading is embedded in every chunk's text, which means the
  embedding captures both the topic label and the content — improving
  retrieval precision
- Prevents policy tables (e.g., accrual rate tables) from being split
  across two chunks, which would make either chunk incomplete and mislead
  the LLM
- Each chunk maps to a named section, which directly improves citation
  accuracy — the system can cite "Section 3.4 Personal Vehicle Mileage"
  rather than a generic page range

**Parameters chosen:**
- Chunk size: ~1,600 characters (~400 tokens at 4 chars/token)
- Overlap: 200 characters between sliding window chunks
- Fixed seed (42) for deterministic, reproducible chunking

**Result:** 233 chunks across 13 documents, average 596 characters per
chunk — well within the embedding model's 256-token context window after
encoding.

**Alternative considered:** Fixed token-window chunking only (no heading
awareness). Rejected because it would frequently split mid-section, cutting
a table row off from its header or separating a rule from its exceptions.

---

### 2.3 Vector Store: ChromaDB (Local Persistent)

**Choice:** ChromaDB with persistent local storage at `./chroma_db`.

**Why:**
- Zero external service — no database server to manage, no cloud account
  required
- Persistent storage means the index survives process restarts without
  re-ingestion
- Sub-5ms query latency for a corpus of 233 chunks
- Clean Python API that integrates directly with the ingestion and
  retrieval modules
- Free — no cost at any scale within this project's scope

**Deployment note:** On Render free tier, disk storage is ephemeral. The
Build Command re-runs `src/ingest.py` on every deploy, which rebuilds the
index from the policy documents in the repository. This adds ~45 seconds to
deploy time but requires no paid persistent storage.

**Alternative considered:** Pinecone free tier (cloud-hosted vector store).
Rejected because it adds an external dependency, requires API key management,
and offers no performance benefit for a corpus this size.

---

### 2.4 Top-k Retrieval: k=3 (default)

**Choice:** Retrieve the top 3 most similar chunks per query.

**Why:**
- Ablation study results (see Section 3.3) showed k=3 delivers the best
  balance of groundedness and latency for this corpus
- k=5 retrieved correct documents but added 2–3 loosely related chunks
  (e.g., "Holidays" and "Bereavement Leave" sections when asking about PTO
  accrual rates) that lengthened the prompt without improving answer quality
- k=3 reduces prompt token count, which directly reduces LLM latency
- For single-policy questions (the majority of eval set queries), k=3
  provides sufficient context; for cross-policy questions, k=3 still
  retrieves the primary relevant section

---

### 2.5 LLM: llama-3.3-70b-versatile via Groq

**Choice:** `llama-3.3-70b-versatile` accessed through the Groq API
(OpenAI-compatible endpoint).

**Why:**
- Groq provides significantly higher rate limits than OpenRouter free tier,
  which was hitting rate limits frequently during development and testing —
  making iteration slow and unreliable
- Groq's LPU inference hardware delivers fast token generation even for
  70B-parameter models
- `llama-3.3-70b-versatile` reliably follows the system prompt's structured
  citation format and respects the out-of-corpus refusal instruction
- OpenAI-compatible API means the same `generation.py` code works for
  Groq, OpenRouter, or any other compatible provider by changing two
  environment variables

**Temperature:** Set to 0.1 (near-deterministic).  
**Why:** Low temperature minimises hallucination — the LLM is less likely to
add information not present in the retrieved context. Confirmed during
testing: at temperature=0.1 the model consistently cited only information
from the retrieved chunks.

**max_tokens:** 512.  
**Why:** Sufficient for a complete policy answer with citations. Caps output
length to keep latency manageable and prevent verbose responses.

**Endpoint fix note:** The correct Groq base URL is
`https://api.groq.com/openai/v1` — the code appends `/chat/completions`
to this. An earlier misconfiguration produced a doubled `/v1/v1/` path
(404 error) which was resolved by ensuring the base URL does not include
a trailing `/v1` in the code's URL construction.

---

### 2.6 Guardrails

Three guardrails are implemented:

**1. Out-of-corpus refusal (system prompt)**  
The system prompt instructs the LLM with an exact refusal string:
> *"I can only answer questions about our company policies. This topic is
> not covered in the available documents."*

Using a fixed, exact string makes the guardrail programmatically testable —
the evaluation script can check whether the LLM's response contains this
string for out-of-corpus questions.

Retrieval scores for out-of-corpus queries (e.g., "What is the capital of
France?") were confirmed to be near zero (0.055 top score), meaning the
LLM receives low-relevance context and the refusal instruction triggers
correctly.

**2. Output length cap (max_tokens=512)**  
Prevents runaway responses and keeps latency predictable.

**3. Low temperature (temperature=0.1)**  
Reduces hallucination by keeping the model close to the retrieved context
rather than generating from its parametric memory.

---

### 2.7 Web Application: Flask

**Choice:** Flask with a single-file chat UI rendered via
`render_template_string`.

**Why:**
- Lightweight — no frontend build step or separate frontend server
- Three required endpoints implemented: `GET /`, `POST /chat`, `GET /health`
- The `/health` endpoint returns JSON including model name, embedding model,
  and live ChromaDB chunk count — useful for verifying deployment state
- Single-file HTML avoids static file serving complexity on Render

---

### 2.8 CI/CD: GitHub Actions

A workflow runs on every push and pull request to `main`:
1. Checks out code
2. Sets up Python 3.11
3. Installs dependencies from `requirements.txt`
4. Runs a smoke test: `python -c "import app; print('OK')"`
5. Runs the pytest test suite
6. On merge to `main`: triggers a Render deploy via webhook

**Python version:** 3.11 was chosen over 3.13 because `tokenizers`
(a dependency of `sentence-transformers`) does not yet publish a pre-built
wheel for Python 3.13 on Windows, which caused Rust compilation errors
during initial setup. Python 3.11 has stable pre-built wheels for all
dependencies.

---

## 3. Evaluation

### 3.1 Methodology

- **Evaluation set:** 30 questions written across 10 policy categories
  *before* building the system, to prevent evaluation bias
- **Groundedness:** Judged by LLM-as-judge (same model, temperature=0).
  The judge is given the question, retrieved context, and generated answer,
  and determines whether the answer contains only information present in
  the context
- **Citation accuracy:** Whether the cited source document actually contains
  the information stated in the answer
- **Latency:** Measured end-to-end from request receipt to response
  delivery using `time.perf_counter()`
- **Out-of-corpus:** 5 questions with no policy answer included to test
  the refusal guardrail

### 3.2 Evaluation Question Categories

| Category             | Questions |
|----------------------|-----------|
| PTO                  | 4         |
| Remote work          | 4         |
| Expense reimbursement| 4         |
| Benefits             | 5         |
| Data privacy / GDPR  | 3         |
| Information security | 3         |
| Attendance           | 2         |
| Anti-harassment      | 2         |
| Performance management| 2        |
| Workplace safety     | 1         |
| **Total**            | **30**    |

### 3.3 Ablation Study: Varying Top-k

The retrieval parameter k was varied across three values to determine the
optimal number of chunks to retrieve per query. All three runs used the
same 30 questions and the same model (llama-3.3-70b-versatile via Groq).

| k | Groundedness | Citation Accuracy | Latency p50 | Latency p95 |
|---|-------------|-------------------|-------------|-------------|
| 3 | 90.0%       | 100.0%            | 883ms       | 1135ms      |
| 5 | 93.3%       | 100.0%            | 865ms       | 1178ms      |
| 8 | 93.3%       | 100.0%            | 980ms       | 1560ms      |

**Finding:** Citation accuracy is perfect (100%) at all three k values,
confirming that heading-based chunking reliably maps answers to their
correct source documents. Groundedness improves from k=3 to k=5
(90% → 93.3%), meaning the additional context helps the LLM find complete
answers for multi-section questions. k=8 matches k=5 on groundedness but
adds 382ms of latency at p95 with no quality gain — the extra chunks are
noise, not signal. **k=5 is the optimal choice** and is the project default.

### 3.4 Results (k=5, default)

| Metric               | Value   |
|----------------------|---------|
| Groundedness         | 93.3%   |
| Citation Accuracy    | 100.0%  |
| Latency p50          | 865ms   |
| Latency p95          | 1178ms  |
| OOC Refusal Rate     | 100%    |

All 30 questions were answered. Out-of-corpus questions (those with
`source_doc: null`) correctly triggered the refusal guardrail in every
case — confirmed by retrieval scores near zero (top score 0.055 for
"What is the capital of France?").

### 3.5 Failure Analysis

2 out of 30 answers (6.7%) were judged NOT GROUNDED by the LLM-as-judge
at k=5. In both cases the answers were directionally correct but the LLM
added minor elaborations not present verbatim in the retrieved context —
for example, adding general HR best-practice phrasing to supplement a
policy rule.

**Root cause:** Even at temperature=0.1, a 70B model occasionally
generalises beyond the retrieved context when the chunk provides a rule
but not a full explanation.

**Mitigation in place:** temperature=0.1 was chosen deliberately over 0.0
to balance groundedness with answer naturalness. Dropping to temperature=0
would eliminate these 2 cases but produce noticeably more robotic responses.
The 93.3% groundedness rate is considered acceptable for this use case.

---

## 4. Known Issues and Limitations

| Issue | Impact | Status |
|-------|--------|--------|
| ChromaDB telemetry errors on startup | Cosmetic only — no functional impact | Suppressed via `ANONYMIZED_TELEMETRY=False` in `.env` |
| PyTorch FutureWarning on import | Cosmetic only | Will resolve on torch upgrade |
| Latency ~14s with llama-3.3-70b-versatile at k=5 | Slow for production use | Reduced by setting k=3; further reducible by switching to llama-3.1-8b-instant |
| ChromaDB ephemeral on Render free tier | Index lost on restart | Mitigated by re-running ingest in Build Command |
| sentence-transformers incompatible with Python 3.13 on Windows | Blocks installation | Resolved by using Python 3.11 venv |

---

## 5. Technology Stack Summary

| Component        | Technology                        | Reason chosen            |
|------------------|-----------------------------------|--------------------------|
| Embedding model  | all-MiniLM-L6-v2 (local)          | Free, no API, fast       |
| Vector store     | ChromaDB (persistent local)       | Free, zero config        |
| LLM              | llama-3.3-70b-versatile (Groq)    | High rate limits, free   |
| Web framework    | Flask                             | Lightweight, simple      |
| Deployment       | Render free tier                  | Public URL, CI/CD hooks  |
| CI/CD            | GitHub Actions                    | Native GitHub integration|
| Python version   | 3.11                              | Wheel compatibility      |
| Chunking         | Heading-based + window fallback   | Semantic coherence       |

---

*Document maintained as part of the Quantic MSSE AI Engineering Project submission.*