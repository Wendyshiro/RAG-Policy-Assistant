# Deployment

## Live URL
**https://rag-policy-assistant-production.up.railway.app/**


## Notes
- Python 3.13 (Railway resolves compatible wheels automatically)
- ChromaDB index (233 chunks) is rebuilt on each deploy
- Embedding model: `all-MiniLM-L6-v2` running on CPU
- LLM: `llama-3.3-70b-versatile` via Groq API