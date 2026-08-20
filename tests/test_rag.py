import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_document_ingestion_and_query():
    # Ingest document
    ingest_payload = {
        "document_id": "test_doc_01",
        "text_content": "PyTorch is an open-source machine learning framework used for deep learning neural networks.",
        "metadata": {"source": "unit-test"}
    }
    ingest_res = client.post("/api/v1/rag/ingest", json=ingest_payload)
    assert ingest_res.status_code == 200
    assert ingest_res.json()["status"] == "success"
    
    # Query vector store
    query_payload = {
        "query": "What is PyTorch used for?",
        "top_k": 1
    }
    query_res = client.post("/api/v1/rag/query", json=query_payload)
    assert query_res.status_code == 200
    data = query_res.json()
    assert len(data["retrieved_chunks"]) > 0
    assert "PyTorch" in data["retrieved_chunks"][0]["text"]
