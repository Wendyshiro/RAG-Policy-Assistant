#pytest smoke tests
import pytest 
import json
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    with patch('src.retrieval._get_collection') as mc, \
         patch('src.retrieval._get_embedder') as me:
        
        me.return_value.encode = MagicMock(return_value=[[0.1]*384])  # Mock embedding output
        mc.return_value.query = MagicMock(return_value={
            'documents': [['Employees accrue 1.25 PTO days per month.']],
            'metadatas': [[{'source': 'pto_policy.md',
                            'title': 'PTO Policy','section':'Accural'}]],
            'distances': [[0.15]]
        })
        import app
        app.app.config['TESTING'] = True
        with app.app.test_client() as c:
            yield c

def test_health(client):
     r = client.get('health')
     assert r.status_code == 200
     assert 'status' in json.loads(r.data)
   
def test_chat_requires_question(client):
    r = client.post('/chat', data= json.dumps({}),
                    content_type='application/json')
    assert r.status_code == 400

def  test_chat_returns_answer(client):
    with patch('app.rag_answer') as mock:
        mock.return_value = {
            'answer': 'You accrue 1.25 days/month',
            'citations': [{'title':'PTO Policy', 'section': 'Accrual',
                           'source': 'pto_policy.md', 'snippet': '....'}],
            'latency_ms': 300, 'model' :'test'
        }
        r = client.post('/chat',
                        data=json.dumps({'question':'How does PTO work'}),
                                        content_type='application/json')
        assert r.status_code == 200
        assert 'answer' in json.loads(r.data)

def test_home_route(client):
    r = client.get('/')
    assert r.status_code == 200
    assert b'Policy Assistant' in r.data