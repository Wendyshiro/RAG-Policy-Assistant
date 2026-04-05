"""
tests/test_app.py — Smoke tests for the Flask app.
Mocks ChromaDB and the LLM so no API keys or built index are needed.
Run from project root: pytest tests/test_app.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    """
    Create a Flask test client with all external dependencies mocked.
    - src.retrieval._get_collection  → mocked ChromaDB collection
    - src.retrieval._get_embedder    → mocked sentence-transformers model
    - src.generation.answer          → mocked in individual tests that need it
    """
    with patch("src.retrieval._get_collection") as mock_col, \
         patch("src.retrieval._get_embedder") as mock_emb:

        # Mock embedder: returns a dummy 384-dim vector
        mock_emb.return_value.encode = MagicMock(
            return_value=[[0.1] * 384]
        )

        # Mock ChromaDB collection: returns one fake chunk
        mock_col.return_value.query = MagicMock(return_value={
            "documents": [["Employees accrue 1.25 PTO days per month."]],
            "metadatas": [[{
                "source":  "pto_policy.md",
                "title":   "PTO Policy",
                "section": "Accrual"
            }]],
            "distances": [[0.15]]
        })
        mock_col.return_value.count = MagicMock(return_value=233)

        import app
        app.app.config["TESTING"] = True
        with app.app.test_client() as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────

def test_health(client):
    """GET /health should return 200 with a status field."""
    r = client.get("/health")
    assert r.status_code == 200
    data = json.loads(r.data)
    assert "status" in data
    assert data["status"] == "ok"


# ── /chat validation ──────────────────────────────────────────────────────────

def test_chat_requires_question(client):
    """POST /chat with no question should return 400."""
    r = client.post(
        "/chat",
        data=json.dumps({}),
        content_type="application/json"
    )
    assert r.status_code == 400
    data = json.loads(r.data)
    assert "error" in data


def test_chat_empty_question(client):
    """POST /chat with an empty string question should return 400."""
    r = client.post(
        "/chat",
        data=json.dumps({"question": "   "}),
        content_type="application/json"
    )
    assert r.status_code == 400


# ── /chat answer ──────────────────────────────────────────────────────────────

def test_chat_returns_answer(client):
    """
    POST /chat with a valid question should return 200 with answer + citations.
    Patches src.generation.answer because app.py imports it inside the route
    function (lazy import), so patching at the source is required.
    """
    mock_result = {
        "answer":     "You accrue 1.25 PTO days per month.",
        "citations":  [{
            "title":   "PTO Policy",
            "section": "Accrual",
            "source":  "pto_policy.md",
            "snippet": "Employees accrue 1.25 PTO days per month..."
        }],
        "latency_ms": 320,
        "model":      "llama-3.3-70b-versatile"
    }

    # Patch at the source module, not at app level, because app.py uses
    # a lazy import: `from src.generation import answer` inside the route
    with patch("src.generation.answer", return_value=mock_result):
        r = client.post(
            "/chat",
            data=json.dumps({"question": "How does PTO accrue?"}),
            content_type="application/json"
        )

    assert r.status_code == 200
    data = json.loads(r.data)
    assert "answer"    in data
    assert "citations" in data
    assert "latency_ms" in data
    assert data["answer"] == "You accrue 1.25 PTO days per month."


# ── / home route ──────────────────────────────────────────────────────────────

def test_home_route(client):
    """GET / should return 200 and serve the chat UI."""
    r = client.get("/")
    assert r.status_code == 200