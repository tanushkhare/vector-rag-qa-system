# ⚡ Vector RAG QA System

[![Live Web Demo](https://img.shields.io/badge/Live_App-Vercel-black?style=for-the-badge&logo=vercel)](https://vector-rag-qa-system.vercel.app)
[![Portfolio Hub](https://img.shields.io/badge/Portfolio_Hub-Live-blue?style=for-the-badge)](https://portfolio-showcase-hub-web11.vercel.app)

🔗 **Production URL:** [https://vector-rag-qa-system.vercel.app](https://vector-rag-qa-system.vercel.app)  
🌐 **Showcase Hub:** [https://portfolio-showcase-hub-web11.vercel.app](https://portfolio-showcase-hub-web11.vercel.app)

---

## 📌 Architectural Overview
Dense semantic retrieval RAG engine indexing knowledge bases via all-MiniLM-L6-v2 embeddings with non-blocking threaded execution.

---

## 🛠️ Technology Ecosystem
* **Core Architecture:** FastAPI, ChromaDB, Sentence-Transformers, PyTest
* **Testing & Quality:** PyTest, Automated GitHub Actions CI
* **Deployment:** Vercel Edge Runtime

---

## 🚀 API Contracts
```http
POST /api/v1/rag/ingest
POST /api/v1/rag/query
GET /health
```

---

## 💻 Local Quickstart
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
pytest tests/ -v
```
