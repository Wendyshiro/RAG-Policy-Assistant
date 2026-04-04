"""
ingest.py — Parse, chunk, embed, and store policy documents into ChromaDB.

Usage:
    python src/ingest.py                  # index all policies
    python src/ingest.py --dry-run        # preview chunks without writing
    python src/ingest.py --verify         # verify the index after build
"""

import warnings
warnings.filterwarnings("ignore")



import os
import sys
import hashlib
import random
import argparse
import logging
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config (all overridable via env vars)
# ---------------------------------------------------------------------------
POLICY_DIR  = Path(os.getenv("POLICY_DIR",  "data/policies"))
CHROMA_PATH = os.getenv("CHROMA_PATH",      "./chroma_db")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL",  "all-MiniLM-L6-v2")
COLLECTION  = os.getenv("CHROMA_COLLECTION","policies")
CHUNK_CHARS     = int(os.getenv("CHUNK_CHARS",    "1600"))  # ~400 tokens @ 4 chars/token
OVERLAP         = int(os.getenv("CHUNK_OVERLAP",  "200"))   # char overlap between windows
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "400"))  # merge sections smaller than this
BATCH_SIZE      = int(os.getenv("EMBED_BATCH",     "64"))   # embeddings per batch

random.seed(42)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Load documents
# ---------------------------------------------------------------------------
def load_documents(policy_dir: Path) -> list[dict]:
    """Read every .md file; extract title from first H1 or filename."""
    if not policy_dir.exists():
        log.error("Policy directory not found: %s", policy_dir)
        sys.exit(1)

    docs = []
    for path in sorted(policy_dir.glob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("Skipping %s — read error: %s", path.name, e)
            continue

        # Try to pull the first H1 as the document title
        title = path.stem.replace("_", " ").title()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break

        docs.append({"text": text, "source": path.name, "title": title})
        log.info("  Loaded %-40s  (%d chars)", path.name, len(text))

    if not docs:
        log.error("No .md files found in %s", policy_dir)
        sys.exit(1)

    return docs


# ---------------------------------------------------------------------------
# 2. Chunk documents  (merge-then-split strategy)
# ---------------------------------------------------------------------------
def chunk_document(doc: dict) -> list[dict]:
    """
    Three-pass chunking strategy:
      1. Split document into (heading, body) sections at ## / ### boundaries.
      2. Merge consecutive small sections so no chunk is tiny (<MIN_CHUNK_CHARS).
      3. Sliding-window split any section still over CHUNK_CHARS, with OVERLAP
         to preserve context at window boundaries.
    """
    sections = _split_into_sections(doc)
    sections = _merge_small_sections(sections)
    chunks = []
    for heading, body in sections:
        chunks.extend(_window_split(doc, heading, body))
    return chunks


def _split_into_sections(doc: dict) -> list[tuple[str, str]]:
    """Walk lines and return list of (heading, body_text) pairs."""
    sections: list[tuple[str, str]] = []
    cur_heading = doc["title"]
    cur_lines: list[str] = []

    for line in doc["text"].splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("## ") or stripped.startswith("### "):
            body = "".join(cur_lines).strip()
            if body:
                sections.append((cur_heading, body))
            cur_heading = stripped.lstrip("#").strip()
            cur_lines = []
        else:
            cur_lines.append(line)

    # Flush the final section
    body = "".join(cur_lines).strip()
    if body:
        sections.append((cur_heading, body))

    return sections


def _merge_small_sections(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Merge consecutive sections whose combined body is still under CHUNK_CHARS.
    Headings are joined with ' / ' so context isn't silently lost.
    This prevents the retriever from seeing lots of isolated 50-word fragments.
    """
    if not sections:
        return sections

    merged: list[tuple[str, str]] = []
    acc_heading, acc_body = sections[0]

    for heading, body in sections[1:]:
        combined = acc_body + "\n\n" + body
        if len(acc_body) < MIN_CHUNK_CHARS and len(combined) <= CHUNK_CHARS:
            # Current accumulator is small and merging still fits — keep going
            acc_heading = acc_heading + " / " + heading
            acc_body = combined
        else:
            merged.append((acc_heading, acc_body))
            acc_heading, acc_body = heading, body

    merged.append((acc_heading, acc_body))
    return merged


def _window_split(doc: dict, heading: str, body: str) -> list[dict]:
    """Sliding-window split for bodies that exceed CHUNK_CHARS."""
    if len(body) <= CHUNK_CHARS:
        return [_make_chunk(doc, heading, body, 0)]

    chunks: list[dict] = []
    step = max(1, CHUNK_CHARS - OVERLAP)
    for part, start in enumerate(range(0, len(body), step)):
        window = body[start : start + CHUNK_CHARS]
        if window.strip():
            chunks.append(_make_chunk(doc, heading, window, part))
    return chunks


def _make_chunk(doc: dict, section: str, body: str, part: int) -> dict:
    """Build a chunk dict with a stable, deterministic ID."""
    key = f"{doc['source']}:{section}:{part}"
    chunk_id = hashlib.md5(key.encode()).hexdigest()[:12]
    return {
        "text":    f"{section}\n\n{body}",
        "source":  doc["source"],
        "title":   doc["title"],
        "section": section,
        "part":    part,
        "id":      chunk_id,
    }


# ---------------------------------------------------------------------------
# 3. Embed
# ---------------------------------------------------------------------------
def embed_chunks(chunks: list[dict], model_name: str) -> list[list[float]]:
    """Embed all chunks in batches; returns embeddings aligned with chunks."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        log.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    log.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    log.info("Embedding %d chunks (batch size %d)…", len(texts), BATCH_SIZE)

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()
    elapsed = time.time() - t0

    log.info("Embedding complete in %.1fs  (%.0f chunks/s)", elapsed, len(chunks) / elapsed)
    return embeddings


# ---------------------------------------------------------------------------
# 4. Store in ChromaDB
# ---------------------------------------------------------------------------
def build_index(dry_run: bool = False, verify: bool = False) -> None:
    log.info("=== Building RAG index ===")
    log.info("Policy dir : %s", POLICY_DIR)
    log.info("ChromaDB   : %s", CHROMA_PATH)
    log.info("Model      : %s", EMBED_MODEL)

    # Load
    docs   = load_documents(POLICY_DIR)
    chunks = [c for d in docs for c in chunk_document(d)]

    # Stats
    log.info("Documents  : %d", len(docs))
    log.info("Chunks     : %d", len(chunks))
    avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
    log.info("Avg chunk  : %.0f chars", avg_len)

    if dry_run:
        log.info("[DRY RUN] Skipping embedding and storage.")
        _print_sample(chunks)
        return

    # Embed
    embeddings = embed_chunks(chunks, EMBED_MODEL)

    # Store
    try:
        import chromadb
    except ImportError:
        log.error("chromadb not installed. Run: pip install chromadb")
        sys.exit(1)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Idempotent: drop and recreate
    try:
        client.delete_collection(COLLECTION)
        log.info("Dropped existing collection '%s'", COLLECTION)
    except Exception:
        pass

    col = client.create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )

    col.add(
        documents  = [c["text"]    for c in chunks],
        embeddings = embeddings,
        metadatas  = [
            {
                "source":  c["source"],
                "title":   c["title"],
                "section": c["section"],
                "part":    c["part"],
            }
            for c in chunks
        ],
        ids = [c["id"] for c in chunks],
    )

    log.info("✓ Indexed %d chunks into collection '%s' at %s",
             len(chunks), COLLECTION, CHROMA_PATH)

    if verify:
        _verify_index(col, chunks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _print_sample(chunks: list[dict], n: int = 3) -> None:
    log.info("--- Sample chunks ---")
    for c in random.sample(chunks, min(n, len(chunks))):
        preview = c["text"][:120].replace("\n", " ")
        log.info("  [%s] %s | %s\n    %s…", c["id"], c["source"], c["section"], preview)


def _verify_index(col, chunks: list[dict]) -> None:
    """Spot-check the index with a few keyword queries."""
    log.info("=== Verifying index ===")

    test_queries = [
        "How many PTO days does an employee get?",
        "What is the remote work internet stipend?",
        "Password requirements for company systems",
        "Workers compensation coverage",
    ]

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(EMBED_MODEL)
    except ImportError:
        log.warning("Cannot verify without sentence-transformers.")
        return

    for query in test_queries:
        q_emb = model.encode(query).tolist()
        results = col.query(query_embeddings=[q_emb], n_results=2)
        docs_returned = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        log.info("  Q: %s", query)
        for doc, meta in zip(docs_returned, metas):
            preview = doc[:80].replace("\n", " ")
            log.info("    → [%s | %s] %s…", meta["source"], meta["section"], preview)

    # Count check
    count = col.count()
    assert count == len(chunks), f"Count mismatch: {count} vs {len(chunks)}"
    log.info("✓ Verification passed — %d documents in collection", count)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest policy docs into ChromaDB.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only; do not embed or store.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run spot-check queries after indexing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_index(dry_run=args.dry_run, verify=args.verify)