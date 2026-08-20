from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.routers import rag_router
import uvicorn

app = FastAPI(
    title="ChromaDB Vector RAG Q&A Engine",
    description="Production-grade vector search and context retrieval engine utilizing Sentence-Transformers and ChromaDB.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router.router)

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "vector-rag-qa-system",
        "engine": "ChromaDB + Sentence-Transformers"
    }

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
