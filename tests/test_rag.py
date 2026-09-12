import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_rag_query_endpoint():
    # Uses mock parameters or falls back to live query
    payload = {"query": "What is dense passage retrieval?", "top_k": 2}
    res = client.post("/api/v1/rag/query", json=payload)
    # Service returns 200 with answer or fallback response
    assert res.status_code in [200, 201]
    data = res.json()
    assert "query" in data or "response" in data or "results" in data
