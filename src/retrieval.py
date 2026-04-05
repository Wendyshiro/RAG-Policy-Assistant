# chromadb top-k query
import warnings
warnings.filterwarnings("ignore")

import os
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
TOP_K       = int(os.getenv("TOP_K", "3"))

# Singletons — loaded once at startup
_embedder   = None
_collection = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = DefaultEmbeddingFunction()
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client      = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection("policies")
    return _collection


def retrieve(query: str, k: int = TOP_K) -> list:
    embedder   = _get_embedder()
    collection = _get_collection()

    # DefaultEmbeddingFunction takes a list of strings
    query_embedding = embedder([query])

    res = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )

    return [
        {
            "text":    doc,
            "source":  meta["source"],
            "title":   meta["title"],
            "section": meta["section"],
            "score":   round(1 - dist, 4)  # cosine distance → similarity
        }
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0]
        )
    ]


if __name__ == "__main__":
    import sys
    queries = sys.argv[1:] or [
        "How many PTO days do I get?",
        "What is the mileage reimbursement rate?",
        "Can I work remotely?",
        "What is the capital of France?",
    ]
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)
        results = retrieve(query, k=5)
        for i, r in enumerate(results, 1):
            print(f"\n  Result {i}  score={r['score']:.4f}  [{r['source']}]")
            print(f"  {r['title']} -- {r['section']}")
            snippet = r["text"].replace("\n", " ")[:180]
            print(f"  {snippet}...")