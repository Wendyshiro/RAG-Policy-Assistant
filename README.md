# RAG Policy App

AI-powered Q&A over company HR policies using RAG

## Quick Start
```bash
git clone https://github/com/YOUR_NAME/rag-policy-app
cd rag-policy-app
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API key
python sec/ingest.py # Build vector index (run once)
python app.py #Start at http://localhost:5000
```

## Endpoints
```bash
GET / Web chat ui
POST /chat JSON {question} -> {answers, citations, latency_ms}
GET /heath JSON status check
```

## Run Tests
```bash
pytest tests/ -v
```
## Run evaluation
```bash
python src/evaluate.py
```
