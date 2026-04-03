"""
app.py — Flask web server for the RAG Policy Assistant.

Routes:
    GET  /          → Serve the chat UI (frontend/index.html)
    POST /chat      → JSON {question} → {answer, citations, latency_ms, model}
    GET  /health    → JSON status check

Usage:
    python app.py                    # development (debug=True)
    gunicorn app:app --bind 0.0.0.0:5000   # production
"""

import os
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, abort
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the chat UI."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        abort(404, description="frontend/index.html not found. "
              "Make sure the file is placed in the frontend/ directory.")
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Accept a JSON body: { "question": "..." }
    Return:            { "answer": "...", "citations": [...], "latency_ms": int, "model": "..." }
    """
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' field in request body."}), 400

    log.info("Question: %s", question[:120])

    try:
        from src.generation import answer
        result = answer(question)
    except Exception as exc:
        log.exception("Generation error")
        return jsonify({"error": str(exc)}), 500

    log.info("Answered in %d ms", result.get("latency_ms", -1))
    return jsonify(result)


@app.route("/health")
def health():
    """Return system status — useful for deployment verification."""
    try:
        import chromadb
        from dotenv import dotenv_values

        chroma_path = os.getenv("CHROMA_PATH", "./chroma_db")
        client      = chromadb.PersistentClient(path=chroma_path)
        col         = client.get_collection("policies")
        chunk_count = col.count()
        db_status   = "ok"
    except Exception as exc:
        chunk_count = 0
        db_status   = f"error: {exc}"

    return jsonify({
        "status":          "ok",
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "llm_model":       os.getenv("LLM_MODEL", "unknown"),
        "llm_base_url":    os.getenv("LLM_BASE_URL", "unknown"),
        "top_k":           int(os.getenv("TOP_K", "3")),
        "chroma_chunks":   chunk_count,
        "chroma_status":   db_status,
    })


# Warm up the embedder and ChromaDB connection at startup
with app.app_context():
    try:
        from src.retrieval import retrieve
        retrieve("warmup", k=1)
        log.info("Retrieval pipeline warmed up.")
    except Exception as e:
        log.warning("Warmup failed: %s", e)

# ── Dev server ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    log.info("Starting dev server on http://localhost:%d  (debug=%s)", port, debug)
    app.run(host="0.0.0.0", port=port, debug=debug)