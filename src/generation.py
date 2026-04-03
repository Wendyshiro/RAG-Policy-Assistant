import os 
import time 
import requests
from dotenv import load_dotenv
from src.retrieval import retrieve

load_dotenv()
API_KEY = os.getenv('LLM_API_KEY')
BASE_URL = os.getenv('LLM_BASE_URL')
MODEL = os.getenv('LLM_MODEL')
TOP_K = int(os.getenv('LLM_TOP_K', 3))

SYSTEM = """You are a helpful HR policy assistant for Acme Group.
ANSWER questions ONLY using policy excerpts provided below.

RULES:
1. If the answer is not in the excerpts, respond EXACTLY:
'Sorry, I can only answer questions about our company polcies.
 This topic is not covered in the available documents.'
2. End your answer with [Source: Document Title -- Section Name]
3. Keep answer under 200 words unless a list is required.
4. Never add information not in the context below.

CONTEXT:
{context}"""

def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c['title']} -- {c['section']}\n{c['text']}")
    return '\n\n---\n\n'.join(parts)

def answer(question: str, k: int = TOP_K)-> dict:
    t0 = time.perf_counter()
    chunks = retrieve(question, k=k)
    system = SYSTEM.replace('{context}', build_context(chunks))

    resp = requests.post(
        f'{BASE_URL}/chat/completions',
        headers={'Authorization': f'Bearer {API_KEY}',
                 'Content-Type': 'application/json'},
        json={'model': MODEL,
              'messages': [{'role': 'system', 'content':system},
                           {'role': 'user', 'content':question}],
                'max_tokens': 512,
                'temperature': 0.1},
        timeout=30
    )
    resp.raise_for_status()
    text_out = resp.json()['choices'][0]['message']['content'].strip()
    latency_ms = round((time.perf_counter() -t0) * 1000)

    citations = []
    for c in chunks:
        entry = {'title' : c['title'], 'section': c['section'],
                 'source': c['source'], 'snippet': c['text'][:220] + '....'}
        if entry not in citations:
            citations.append(entry)
    return {'answer': text_out, 'citations': citations,
            'latency_ms': latency_ms, 'model': MODEL}
    
