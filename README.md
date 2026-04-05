# RAG Policy Assistant

An AI-powered HR policy Q&A system built with Retrieval-Augmented Generation (RAG).
Ask questions about company policies in plain English and get accurate, cited answers
grounded in the actual policy documents.

**Live Demo:** [https://rag-policy-app.onrender.com](https://rag-policy-app.onrender.com)  
**Built for:** Quantic MSSE AI Engineering Project

---

## What It Does

- Answers HR policy questions with cited sources
- Refuses to answer questions outside the policy corpus
- Shows which document and section each answer comes from
- Runs fully in the browser — no login required

**Example questions it can answer:**
- *"How many PTO days does an employee with 4 years of tenure get?"*
- *"What is the mileage reimbursement rate?"*
- *"Can I work remotely if I'm on a PIP?"*
- *"How quickly must a data breach be reported?"*

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Embedding model | `all-MiniLM-L6-v2` via ChromaDB/onnxruntime | Free, no API, runs locally |
| Vector store | ChromaDB (persistent local) | Zero config, sub-5ms queries |
| LLM | `llama-3.3-70b-versatile` via Groq | Free tier, high rate limits |
| Web framework | Flask | Lightweight, single-file UI |
| Deployment | Render free tier | Public URL, CI/CD webhooks |
| CI/CD | GitHub Actions | Runs on every push/PR |
| Python | 3.11 | Stable wheel compatibility |

---

## Project Structure

```
RAG-Policy-App/
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI/CD
├── data/
│   └── policies/                 # 13 markdown policy documents
│       ├── pto_policy.md
│       ├── remote_work_policy.md
│       ├── expense_reimbursement.md
│       ├── benefits_policy.md
│       ├── information_security.md
│       ├── data_privacy_gdpr.md
│       ├── anti_harassment_policy.md
│       ├── attendance_policy.md
│       ├── performance_management.md
│       ├── code_of_conduct.md
│       ├── drug_alcohol_testing.md
│       ├── social_media_internet_use.md
│       └── workplace_safety.md
├── src/
│   ├── __init__.py
│   ├── ingest.py                 # Parse, chunk, embed, store in ChromaDB
│   ├── retrieval.py              # ChromaDB top-k semantic search
│   ├── generation.py             # Prompt builder + Groq LLM call
│   └── evaluate.py               # Full evaluation suite + ablations
├── eval/
│   ├── questions.json            # 30 evaluation Q&A pairs
│   ├── results.json              # Evaluation output (generated)
│   └── ablation_results.json     # Ablation study output (generated)
├── frontend/
│   └── index.html                # Chat UI
├── tests/
│   ├── __init__.py
│   └── test_app.py               # Pytest smoke tests (mocked)
├── app.py                        # Flask server: /, /chat, /health
├── conftest.py                   # Pytest path configuration
├── pytest.ini                    # Pytest settings
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── .python-version               # Pins Python 3.11 for Render
├── deployed.md                   # Live deployment URL
├── design-and-evaluation.md      # Architecture decisions + eval results
└── ai-tooling.md                 # AI tools used during development
```

---

## Quick Start

### Prerequisites

- Python 3.11 (required — see note below)
- A free [Groq API key](https://console.groq.com)
- Git

> **Why Python 3.11?** The `sentence-transformers` and `tokenizers` packages
> do not yet have pre-built wheels for Python 3.13/3.14 on Windows.
> Python 3.11 has stable wheels for all dependencies.

### 1. Clone the repository

```bash
git clone https://github.com/Wendyshiro/RAG-Policy-App.git
cd RAG-Policy-App
```

### 2. Create a virtual environment with Python 3.11

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
LLM_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxx   # Your Groq API key
LLM_BASE_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PATH=./chroma_db
TOP_K=5
ANONYMIZED_TELEMETRY=False
```

### 5. Build the vector index

```bash
python -W ignore -m src.ingest
```

This parses all 13 policy documents, chunks them by heading, embeds the
chunks, and stores them in ChromaDB. Takes about 30-60 seconds on first run.
Only needs to re-run if you change the policy documents.

Expected output:
```
19:57:14 [INFO] === Building RAG index ===
19:57:14 [INFO] Documents  : 13
19:57:14 [INFO] Chunks     : 233
19:57:14 [INFO] Avg chunk  : 596 chars
19:57:15 [INFO] Embedding 233 chunks…
19:57:43 [INFO] ✓ Indexed 233 chunks into collection 'policies' at ./chroma_db
```

### 6. Start the server

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## API Reference

### `GET /`
Serves the web chat UI.

### `POST /chat`
Accepts a question, returns a cited answer.

**Request:**
```json
{ "question": "How many PTO days do I get?" }
```

**Response:**
```json
{
  "answer": "Full-time employees receive:\n- Years 0-2: 15 days/year\n- Years 3-5: 20 days/year\n- Years 6+: 25 days/year\n[Source: PTO Policy -- Annual PTO Accrual]",
  "citations": [
    {
      "title": "Paid Time Off (PTO) Policy",
      "section": "Annual PTO Accrual",
      "source": "pto_policy.md",
      "snippet": "Years 0-2: 15 days per year (120 hours)..."
    }
  ],
  "latency_ms": 865,
  "model": "llama-3.3-70b-versatile"
}
```

### `GET /health`
Returns system status — useful for verifying deployment.

**Response:**
```json
{
  "status": "ok",
  "embedding_model": "all-MiniLM-L6-v2",
  "llm_model": "llama-3.3-70b-versatile",
  "top_k": 5,
  "chroma_chunks": 233,
  "chroma_status": "ok"
}
```

---

## Running Tests

```bash
pytest tests/test_app.py -v
```

Tests are fully mocked — no API key or built index required. Covers:
- `GET /health` returns 200 with status field
- `POST /chat` with no question returns 400
- `POST /chat` with empty string returns 400
- `POST /chat` with valid question returns answer + citations
- `GET /` returns 200 with chat UI

---

## Manual Testing Tools

Two additional scripts are provided for manual pipeline testing.
Run these from the project root — they require a built index and API key.

### Test retrieval only (no LLM call)

```bash
# Run 4 built-in test queries
python -W ignore -m src.retrieval

# Test a specific question
python -W ignore -m src.retrieval "what is the password minimum length?"
```

Good output shows scores above 0.5 for in-corpus questions and near 0.0
for out-of-corpus questions like "what is the capital of France?".

### Test the full RAG pipeline (retrieval + LLM)

```bash
# Run one demo question with full verbose output
python -W ignore tests/test_generation.py

# Run full test suite (10 in-corpus + 3 out-of-corpus)
python -W ignore tests/test_generation.py --all

# Test only the out-of-corpus guardrail
python -W ignore tests/test_generation.py --guardrail

# Test any custom question
python -W ignore tests/test_generation.py "what is the expense receipt limit?"
```

---

## Running the Evaluation Suite

The evaluation suite runs all 30 Q&A pairs through the full pipeline,
scores groundedness using an LLM-as-judge, measures citation accuracy,
and runs an ablation study across k=3, k=5, k=8.

```bash
python -W ignore -m src.evaluate
```

Takes approximately 10-15 minutes due to rate limit delays between questions.
Results are saved to `eval/results.json` and `eval/ablation_results.json`.

**Results (k=5 default):**

| Metric | Value |
|--------|-------|
| Groundedness | 93.3% |
| Citation Accuracy | 100.0% |
| Latency p50 | 865ms |
| Latency p95 | 1178ms |
| OOC Refusal Rate | 100% |

**Ablation study — varying k:**

| k | Groundedness | Citation Acc | p50 | p95 |
|---|-------------|--------------|-----|-----|
| 3 | 90.0% | 100.0% | 883ms | 1135ms |
| 5 | 93.3% | 100.0% | 865ms | 1178ms |
| 8 | 93.3% | 100.0% | 980ms | 1560ms |

k=5 is optimal: better groundedness than k=3, same quality as k=8 with lower latency.

---

## Policy Documents

The system answers questions from 13 company policy documents:

| Document | Topics Covered |
|----------|---------------|
| `pto_policy.md` | Vacation accrual, carryover, sick leave, bereavement |
| `remote_work_policy.md` | Eligibility, equipment, stipends, core hours |
| `expense_reimbursement.md` | Travel, mileage ($0.67/mile), meals, receipts |
| `benefits_policy.md` | Health insurance, 401k, EAP, maternity/paternity leave |
| `information_security.md` | Passwords, MFA, device policy, incident reporting |
| `data_privacy_gdpr.md` | GDPR rights, data retention, breach notification |
| `anti_harassment_policy.md` | Reporting procedures, investigation timeline |
| `attendance_policy.md` | Absence procedures, progressive discipline |
| `performance_management.md` | Review cycle, ratings, PIP process |
| `code_of_conduct.md` | Expected behaviour, conflicts of interest |
| `drug_alcohol_testing.md` | Testing types, consequences, EAP referral |
| `social_media_internet_use.md` | Acceptable use, monitoring, streaming |
| `workplace_safety.md` | PPE, incident reporting, emergency procedures |

---

## Deployment (Render)

The app is deployed on Render free tier. See [deployed.md](deployed.md) for the live URL.

**Render configuration:**
- **Build Command:** `pip install -r requirements.txt && python -W ignore src/ingest.py`
- **Start Command:** `gunicorn app:app`
- **Python Version:** 3.11.9 (pinned via `.python-version`)
- **Environment Variables:** Set in Render dashboard (same keys as `.env.example`)

> **Note:** Render free tier has ephemeral disk storage. The vector index is
> rebuilt on every deploy via the Build Command. This adds ~60 seconds to
> deploy time but requires no paid persistent storage.

**Auto-deploy:** Enabled via GitHub Actions webhook. Every push to `main` that
passes CI triggers an automatic Render deploy.

---

## CI/CD

GitHub Actions workflow runs on every push and pull request to `main`:

1. Checkout code
2. Set up Python 3.11
3. Install dependencies from `requirements.txt`
4. Smoke test: `python -c "import app; print('OK')"`
5. Run pytest test suite
6. On merge to `main`: trigger Render deploy via webhook

View CI runs: [GitHub Actions tab](https://github.com/Wendyshiro/RAG-Policy-App/actions)

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | Yes | — | Groq API key (starts with `gsk_`) |
| `LLM_BASE_URL` | Yes | — | `https://api.groq.com/openai/v1` |
| `LLM_MODEL` | Yes | — | e.g. `llama-3.3-70b-versatile` |
| `EMBEDDING_MODEL` | No | `all-MiniLM-L6-v2` | Sentence embedding model name |
| `CHROMA_PATH` | No | `./chroma_db` | ChromaDB storage path |
| `TOP_K` | No | `5` | Number of chunks to retrieve per query |
| `ANONYMIZED_TELEMETRY` | No | `True` | Set to `False` to silence ChromaDB logs |
| `FLASK_DEBUG` | No | `true` | Set to `false` in production |
| `PORT` | No | `5000` | Port for Flask dev server |

---

## How RAG Works in This App

```
User types a question
        │
        ▼
Question is embedded into a vector (384 dimensions)
        │
        ▼
ChromaDB finds the top-5 most similar chunks
from 233 indexed policy document sections
        │
        ▼
Retrieved chunks are injected into the LLM prompt
along with the system instruction to cite sources
and refuse out-of-corpus questions
        │
        ▼
Groq LLM generates a grounded, cited answer
        │
        ▼
Answer + citations displayed in the chat UI
```

**Guardrails:**
- Out-of-corpus refusal: LLM instructed with exact refusal wording
- Temperature 0.1: reduces hallucination
- max_tokens 512: caps response length
- Citation required: system prompt demands source attribution

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`
Always run scripts from the project root using `-m`:
```bash
# Correct
python -m src.ingest
python -m src.retrieval

# Wrong
python src/ingest.py
```

### `429 Too Many Requests` from Groq
Groq free tier allows ~30 requests/minute. The evaluation suite automatically
waits 4 seconds between questions. If you hit limits during normal use,
wait 60 seconds and try again.

### ChromaDB telemetry errors
Add to your `.env`:
```
ANONYMIZED_TELEMETRY=False
```

### `tokenizers` fails to install (Windows)
You are likely on Python 3.13 or 3.14. Switch to Python 3.11:
```bash
py -3.11 -m venv .venv
```

### Corrupted ONNX model on Windows
Delete the ChromaDB model cache and re-run ingest:
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\chroma"
python -W ignore -m src.ingest
```

---

## Design and Evaluation

See [design-and-evaluation.md](design-and-evaluation.md) for full documentation of:
- All architecture and technology choices with justifications
- Chunking strategy rationale
- Evaluation methodology
- Ablation study results and findings
- Known limitations

## AI Tooling

See [ai-tooling.md](ai-tooling.md) for a description of which AI tools were
used during development and what worked well vs what didn't.

---

## Grader Access

GitHub collaborator `quantic-grader` has been added to this repository.

**Submission email:** msse+projects@quantic.edu

---

*Quantic MSSE AI Engineering Project — April 2026*