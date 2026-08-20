from fastapi import APIRouter, HTTPException
from backend.app.schemas.rag_schema import (
    DocumentIngestRequest, DocumentIngestResponse,
    QueryRequest, QueryResponse, RetrievedChunk
)
from backend.app.services.rag_service import rag_engine

router = APIRouter(prefix="/api/v1/rag", tags=["Vector RAG Engine"])

@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document(payload: DocumentIngestRequest):
    try:
        chunks = rag_engine.ingest_document(
            doc_id=payload.document_id,
            text=payload.text_content,
            metadata=payload.metadata
        )
        return DocumentIngestResponse(
            status="success",
            document_id=payload.document_id,
            chunks_created=chunks,
            message=f"Document successfully indexed into {chunks} vector chunks."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_rag(payload: QueryRequest):
    try:
        chunks = rag_engine.query_context(query_text=payload.query, top_k=payload.top_k)
        
        # Synthesize extracted context into an answer summary
        if chunks:
            context_summary = " ".join([c["text"] for c in chunks])
            synthesized = f"Based on retrieved documentation: {context_summary[:350]}..."
        else:
            synthesized = "No relevant context found in vector database matching your query."

        retrieved_objs = [
            RetrievedChunk(text=c["text"], distance=c["distance"], metadata=c["metadata"])
            for c in chunks
        ]

        return QueryResponse(
            query=payload.query,
            retrieved_chunks=retrieved_objs,
            synthesized_answer=synthesized
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
