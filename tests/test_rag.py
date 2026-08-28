import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

def test_document_ingestion_and_query():
    # 1. Ingest baseline test vector document
    ingest_payload = {
        "document_id": "test_doc_ann",
        "text_content": "Dense passage retrieval uses bi-encoder embeddings to calculate approximate nearest neighbor cosine similarity.",
        "metadata": {"source": "unit_test_fixture"}
    }
    ingest_res = client.post("/api/v1/rag/ingest", json=ingest_payload)
    assert ingest_res.status_code == 200
    assert ingest_res.json()["status"] == "success"

    # 2. Query indexed vector chunk
    query_payload = {
        "query": "What does dense passage retrieval use?",
        "top_k": 1
    }
    query_res = client.post("/api/v1/rag/query", json=query_payload)
    assert query_res.status_code == 200
    data = query_res.json()
    assert data["query"] == "What does dense passage retrieval use?"
    assert len(data["retrieved_chunks"]) > 0
    assert "dense passage retrieval" in data["retrieved_chunks"][0]["text"].lower()
