#chromadb top-k query
import warnings
warnings.filterwarnings("ignore")

import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv('TOP_K', '3'))

_embedder, _collection = None, None

def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection('policies')#check if the correct path is data/policies
    return _collection

def retrieve(query: str, k: int = TOP_K) -> list:
    emb = _get_embedder().encode([query]).tolist()
    res = _get_collection().query(
        query_embeddings=emb, n_results=k,
        include=['documents', 'metadatas', 'distances']
    )
    return [
        {'text': doc,
         'source': meta['source'],
         'title': meta['title'],
         'section': meta['section'],
         'score': round(1 - dist, 4) } # Convert distance to similarity score}
         for doc, meta, dist in zip(
             res['documents'][0],
             res['metadatas'][0],
             res['distances'][0]
         )
    ]
