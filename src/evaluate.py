"""
src/evaluate.py — Full evaluation suite for the RAG Policy Assistant.

Measures:
    - Groundedness      : LLM-as-judge checks answer is grounded in context
    - Citation accuracy : does cited source doc actually contain the answer?
    - Latency           : p50 and p95 end-to-end response times
    - Ablations         : compares k=3, k=5, k=8 retrieval

Run from project root:
    python -W ignore -m src.evaluate
"""

# ── Suppress deprecation warnings (torch / transformers / pydantic) ───────────
# Must come before all other imports
import warnings
warnings.filterwarnings("ignore")

import json
import statistics
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.generation import answer as rag_answer
from src.retrieval import retrieve

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
EVAL_FILE    = Path("eval/questions.json")
RESULTS_FILE = Path("eval/results.json")

API_KEY  = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL    = os.getenv("LLM_MODEL")

# Groq free tier: ~30 req/min. Each question makes 2 LLM calls
# (answer + judge) = 15 questions/min max safely.
# 4s sleep between questions keeps us under the limit.
SLEEP_BETWEEN_QUESTIONS = 4   # seconds — increase to 6 if still hitting 429
MAX_RETRIES             = 3   # retry up to 3x on 429
RETRY_WAIT              = 30  # seconds to wait after a 429

JUDGE_PROMPT = """You are an evaluation assistant.
Given a question, context (policy excerpts), and an AI answer,
decide if the answer is grounded.
Grounded = contains ONLY information from the context, nothing invented.
Respond with ONLY valid JSON: {"grounded": true/false, "reason": "..."}"""


# ── Rate-limit-aware LLM call ─────────────────────────────────────────────────

def llm_call_with_retry(payload: dict) -> dict:
    """POST to LLM endpoint with automatic retry on 429."""
    for attempt in range(1, MAX_RETRIES + 1):
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type":  "application/json"
            },
            json=payload,
            timeout=60
        )
        if r.status_code == 429:
            print(f"  [429] Rate limited — waiting {RETRY_WAIT}s "
                  f"(attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(RETRY_WAIT)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries (429).")


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

def judge_groundedness(question: str, chunks: list, answer_text: str) -> dict:
    """
    Ask the LLM whether answer_text is grounded in the retrieved chunks.
    Returns {"grounded": True/False/None, "reason": "..."}
    """
    ctx = "\n\n".join(c["text"] for c in chunks)
    try:
        data = llm_call_with_retry({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user",   "content":
                    f"Question: {question}\n"
                    f"Context:\n{ctx}\n"
                    f"Answer: {answer_text}"}
            ],
            "max_tokens": 200,
            "temperature": 0
        })
        raw = data["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {"grounded": None, "reason": str(e)}


# ── Citation accuracy ─────────────────────────────────────────────────────────

def citation_ok(result: dict, meta: dict):
    """
    True  — cited source matches expected doc.
    None  — out-of-corpus question, skip check.
    False — wrong citation.
    """
    if meta.get("source_doc") is None:
        return None
    return any(
        c["source"] == meta["source_doc"]
        for c in result.get("citations", [])
    )


# ── Single eval run ───────────────────────────────────────────────────────────

def run_eval(k: int = 3) -> dict:
    """
    Run all questions through the RAG pipeline at top-k=k.
    Saves full results to RESULTS_FILE. Returns summary dict.
    """
    questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    records, latencies = [], []

    print(f"\nRunning evaluation — k={k}, {len(questions)} questions")
    print(f"Delay between questions: {SLEEP_BETWEEN_QUESTIONS}s (rate limit guard)")
    print("-" * 55)

    for i, q in enumerate(questions, 1):
        print(f"[{i:02}/{len(questions)}] {q['question'][:60]}...")

        # ── RAG answer with retry on 429 ──────────────────────────────────────
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = rag_answer(q["question"], k=k)
                break
            except Exception as e:
                if "429" in str(e) and attempt < MAX_RETRIES:
                    print(f"  [429] Rate limited on answer — "
                          f"waiting {RETRY_WAIT}s...")
                    time.sleep(RETRY_WAIT)
                else:
                    print(f"  [ERROR] {e}")
                    result = {
                        "answer":     f"ERROR: {e}",
                        "citations":  [],
                        "latency_ms": 0,
                        "model":      MODEL
                    }
                    break

        latencies.append(result["latency_ms"])

        # ── Judge groundedness ────────────────────────────────────────────────
        chunks = retrieve(q["question"], k=k)
        g      = judge_groundedness(q["question"], chunks, result["answer"])
        ca     = citation_ok(result, q)

        records.append({
            **q,
            "generated_answer": result["answer"],
            "citations":        result["citations"],
            "latency_ms":       result["latency_ms"],
            "grounded":         g.get("grounded"),
            "grounded_reason":  g.get("reason"),
            "citation_correct": ca
        })

        g_str  = "GROUNDED"  if g.get("grounded") else "NOT GROUNDED"
        ca_str = "cit OK"    if ca else ("cit WRONG" if ca is False else "cit N/A")
        print(f"         {g_str} | {ca_str} | {result['latency_ms']}ms")

        # Sleep between questions (skip after the last one)
        if i < len(questions):
            time.sleep(SLEEP_BETWEEN_QUESTIONS)

    # ── Summary ───────────────────────────────────────────────────────────────
    ls        = sorted(latencies)
    n         = len(ls)
    in_corpus = [r for r in records if r["citation_correct"] is not None]

    summary = {
        "k":                 k,
        "groundedness_pct":  round(
            sum(1 for r in records if r["grounded"]) / n * 100, 1),
        "citation_acc_pct":  round(
            sum(1 for r in in_corpus if r["citation_correct"])
            / len(in_corpus) * 100, 1
        ) if in_corpus else 0,
        "latency_p50_ms":    ls[n // 2],
        "latency_p95_ms":    ls[int(n * 0.95)],
        "latency_mean_ms":   round(statistics.mean(latencies)),
        "total_questions":   n,
        "results":           records
    }

    RESULTS_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n  Groundedness    : {summary['groundedness_pct']}%")
    print(f"  Citation Acc    : {summary['citation_acc_pct']}%")
    print(f"  Latency p50/p95 : {summary['latency_p50_ms']}ms / "
          f"{summary['latency_p95_ms']}ms")
    print(f"  Results saved   : {RESULTS_FILE}")

    return summary


# ── Ablation study ────────────────────────────────────────────────────────────

def run_ablations() -> None:
    """Run eval at k=3, k=5, k=8 and print a comparison table."""
    print("\n=== Ablation Study: varying k ===")
    rows = []

    for k in [3, 5, 8]:
        print(f"\n--- k={k} ---")
        s = run_eval(k=k)
        rows.append({
            "k":                k,
            "groundedness_pct": s["groundedness_pct"],
            "citation_acc_pct": s["citation_acc_pct"],
            "latency_p50_ms":   s["latency_p50_ms"],
            "latency_p95_ms":   s["latency_p95_ms"]
        })

    Path("eval/ablation_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    print("\n k  | Groundedness | Citation Acc | p50     | p95")
    print(" ---|-------------|--------------|---------|--------")
    for r in rows:
        print(
            f" {r['k']}  | {r['groundedness_pct']:5.1f}%       "
            f"| {r['citation_acc_pct']:5.1f}%        "
            f"| {r['latency_p50_ms']}ms   | {r['latency_p95_ms']}ms"
        )
    print("\nAblation results saved to eval/ablation_results.json")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_eval()       # k=3 → eval/results.json
    run_ablations()  # k=3,5,8 → eval/ablation_results.json