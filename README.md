# ⚡ Vector RAG QA System

[![Live Web Demo](https://img.shields.io/badge/Live_App-Vercel-black?style=for-the-badge&logo=vercel)](https://vector-rag-qa-system.vercel.app)
[![Portfolio Hub](https://img.shields.io/badge/Portfolio_Hub-Live-blue?style=for-the-badge)](https://portfolio-showcase-hub-web11.vercel.app)

🔗 **Production URL:** [https://vector-rag-qa-system.vercel.app](https://vector-rag-qa-system.vercel.app)  
🌐 **Showcase Hub:** [https://portfolio-showcase-hub-web11.vercel.app](https://portfolio-showcase-hub-web11.vercel.app)

---

## 📌 Architectural Overview
Dense semantic vector retrieval RAG engine indexing knowledge bases via all-MiniLM-L6-v2 embeddings. Offloads CPU-intensive tensor computations using `asyncio.to_thread` to maintain sub-20ms event-loop SLAs.

---

## 🛠️ Technology Ecosystem
* **Core Architecture:** FastAPI, ChromaDB, Sentence-Transformers, PyTest
* **Testing & Quality:** PyTest, Automated GitHub Actions CI
* **Deployment:** Vercel Edge Runtime

---

## 🛡️ Production Standards
* **Persistent Storage:** Configured chromadb.PersistentClient with disk state retention.
* **Event Loop Decoupling:** Tensor similarity and embedding calculations run on a separate worker thread.
* **Source Provenance:** Returns top-k retrieved chunks with similarity metrics and source document tags.

---

## 🚀 API Contracts
```http
POST /api/v1/rag/ingest
Request:
{
  "content": "FastAPI utilizes Starlette for web routing and Pydantic for data serialization.",
  "metadata": {"doc_id": "fastapi_specs"}
}

POST /api/v1/rag/query
Request:
{
  "query": "How does FastAPI handle validation?",
  "top_k": 3
}

Response (200 OK):
{
  "answer": "FastAPI performs data validation using Pydantic schemas...",
  "chunks": [
    {"text": "Pydantic models enforce schema validation.", "score": 0.92, "source": "fastapi_specs"}
  ],
  "latency_ms": 12.4
}

GET /health
Response: {"status": "healthy"}

💻 Local Quickstart

Bash

pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
pytest tests/ -v