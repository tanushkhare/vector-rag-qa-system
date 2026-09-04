import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

@patch("backend.app.services.rag_service.rag_engine.query")
def test_rag_query_endpoint(mock_query):
    mock_query.return_value = {
        "answer": "FastAPI performs data serialization using Pydantic.",
        "chunks": [{"text": "Pydantic models enforce schema validation.", "score": 0.92, "source": "tech_specs"}],
        "latency_ms": 12.4
    }
    payload = {"query": "How does validation work?", "top_k": 3}
    res = client.post("/api/v1/rag/query", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "Pydantic" in data["answer"]
    assert len(data["chunks"]) > 0
