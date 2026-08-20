import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class VectorRAGEngine:
    def __init__(self, collection_name: str = "portfolio_kb"):
        # Initialize in-memory persistent ChromaDB client
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        # Load local embedding model
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]

    def ingest_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> int:
        chunks = self._chunk_text(text)
        embeddings = self.embed_model.encode(chunks).tolist()
        
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{**metadata, "chunk_index": i, "doc_id": doc_id} for i in range(len(chunks))]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        return len(chunks)

    def query_context(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_embedding = self.embed_model.encode([query_text]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            
            for doc, dist, meta in zip(docs, dists, metas):
                retrieved.append({
                    "text": doc,
                    "distance": float(dist),
                    "metadata": meta
                })
        return retrieved

# Singleton instance
rag_engine = VectorRAGEngine()
