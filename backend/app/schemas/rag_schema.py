from pydantic import BaseModel, Field
from typing import List, Optional

class DocumentIngestRequest(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    text_content: str = Field(..., min_length=10, description="Raw text content to chunk and embed")
    metadata: Optional[dict] = Field(default_factory=dict, description="Metadata tags (source, author, category)")

class DocumentIngestResponse(BaseModel):
    status: str
    document_id: str
    chunks_created: int
    message: str

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="User search query")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of relevant chunks to retrieve")

class RetrievedChunk(BaseModel):
    text: str
    distance: float
    metadata: dict

class QueryResponse(BaseModel):
    query: str
    retrieved_chunks: List[RetrievedChunk]
    synthesized_answer: str
