#RAG Policy Assistant
AI-powered Q&A over company HR policies using RAG

## Quick Start
```bash
git clone https://github/com/YOUR_NAME/rag-policy-app
```
```bash
cd rag-policy-app
```
```bash
python -m venv venv && source venv/bin/activate
```
```bash
pip install -r requirements.txt
```
```.env.example
cp .env.example .env  # Add your API key
```
```bash
python sec/ingest.py # Build vecore index (run once)
```
```bash
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