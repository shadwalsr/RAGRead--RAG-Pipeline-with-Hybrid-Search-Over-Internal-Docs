"""
Phase 5: API Layer
FastAPI service exposing the RAG pipeline as a production-grade web service.
"""

import os
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

# Import our RAG modules
from storage import HybridStore
from retriever import HybridRetriever
from generator import RAGGenerator
from chunker import chunk_document
from loader import DocumentLoader

load_dotenv()

app = FastAPI(
    title="RAGRead API",
    description="Production-Grade Hybrid RAG Service with Citation Verification",
    version="1.0.0"
)

# Initialize shared components
store = HybridStore()
retriever = HybridRetriever(store)
generator = RAGGenerator()
loader = DocumentLoader()

# --- Pydantic Models ---

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5
    use_reranker: bool = True
    strategy: Optional[str] = None

class Source(BaseModel):
    id: str
    content_preview: str
    source: str
    score: float

class Refusal(BaseModel):
    what_is_found: Optional[str] = None
    what_is_missing: Optional[str] = None
    suggested_documents: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    confidence_score: int
    citation_coverage: int
    can_answer: bool
    sources: List[Source]
    refusal: Optional[Refusal] = None

class DocumentInfo(BaseModel):
    filename: str
    size_kb: float

# --- Endpoints ---

@app.get("/")
async def root():
    return {"message": "RAGRead API is online. Visit /docs for documentation."}

@app.get("/v1/documents", response_model=List[DocumentInfo])
async def list_documents():
    """Returns a list of all raw documents indexed in the system."""
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        return []
    
    docs = []
    for f in os.listdir(raw_dir):
        if f.startswith("."): continue
        path = os.path.join(raw_dir, f)
        docs.append(DocumentInfo(
            filename=f,
            size_kb=round(os.path.getsize(path) / 1024, 2)
        ))
    return docs

@app.post("/v1/ask", response_model=AskResponse)
async def ask(request: QueryRequest):
    """
    Asks a question against the document corpus.
    Runs the full Hybrid Retrieval -> Rerank -> Grounded Generation -> Verification pipeline.
    """
    try:
        # 1. Retrieve chunks
        chunks = retriever.retrieve(
            request.query, 
            top_k=request.top_k, 
            alpha=request.alpha, 
            use_reranker=request.use_reranker,
            strategy=request.strategy
        )
        
        # 2. Generate and Verify
        result = generator.generate_comprehensive_response(request.query, chunks)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))

        # 3. Format response
        sources = []
        for c in chunks:
            sources.append(Source(
                id=c["id"],
                content_preview=c["content"][:200] + "...",
                source=c["metadata"].get("source", "Unknown"),
                score=c.get("rrf_score", 0.0)
            ))

        refusal_data = result.get("structured_refusal", {})
        refusal = None
        if not result.get("can_answer"):
            refusal = Refusal(
                what_is_found=refusal_data.get("what_is_found"),
                what_is_missing=refusal_data.get("what_is_missing"),
                suggested_documents=refusal_data.get("suggested_documents")
            )

        return AskResponse(
            answer=result.get("verified_answer") or result.get("answer", ""),
            confidence_score=result.get("retrieval_confidence_score", 0),
            citation_coverage=result.get("citation_coverage_pct", 0),
            can_answer=result.get("can_answer", True),
            sources=sources,
            refusal=refusal
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    strategy: str = Query("structure_aware", enum=["fixed", "structural", "semantic"])
):
    """
    Uploads a new document and indexes it immediately.
    Supported formats: PDF, Markdown, TXT.
    """
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    file_path = os.path.join(raw_dir, file.filename)
    
    # 1. Save file to disk
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Load text
        ext = os.path.splitext(file.filename)[1].lower()
        if ext == ".pdf":
            docs = loader.load_pdf(file_path)
        elif ext in [".md", ".html", ".htm", ".txt"]:
            docs = loader.load_unstructured(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if not docs:
            raise HTTPException(status_code=400, detail="File is empty or could not be read")

        full_text = "\n\n".join(doc["content"] for doc in docs)
            
        # 3. Chunk
        # Map API strategy names to internal chunker strategy names
        strat_map = {
            "fixed": "fixed_size",
            "structural": "structure_aware",
            "semantic": "semantic"
        }
        internal_strat = strat_map.get(strategy, "structure_aware")
        
        chunks = chunk_document(full_text, file.filename, strategy=internal_strat)
        
        # 4. Ingest into HybridStore
        stats = store.ingest_chunks(chunks)
        
        return {
            "filename": file.filename,
            "strategy_used": internal_strat,
            "chunks_created": stats["total"],
            "chunks_added": stats["added"],
            "duplicates_skipped": stats["duplicates_skipped"]
        }
        
    except Exception as e:
        # Cleanup file if ingestion fails
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
