import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv

from storage import HybridStore
from retriever import HybridRetriever
from generator import RAGGenerator
from chunker import split_text
from loader import DocumentLoader

load_dotenv()

app = FastAPI(
    title="RAGRead API",
    description="Hybrid RAG pipeline with citation verification",
    version="1.0.0"
)

# init everything once at startup - don't want to reload models per request
db = HybridStore()
retriever = HybridRetriever(db)
gen = RAGGenerator()
doc_loader = DocumentLoader()

# the API names for strategies don't match the internal names in chunker.py
# so keeping a map here - easier than renaming everything
STRATEGY_MAP = {
    "fixed": "fixed_size",
    "structural": "structure_aware",
    "semantic": "semantic"
}


# --- request/response shapes ---

class AskRequest(BaseModel):
    query: str
    top_k: int = 5
    alpha: float = 0.5
    use_reranker: bool = True
    strategy: Optional[str] = None

class SourceChunk(BaseModel):
    id: str
    content_preview: str
    source: str
    score: float

class RefusalDetail(BaseModel):
    what_is_found: Optional[str] = None
    what_is_missing: Optional[str] = None
    suggested_documents: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    confidence_score: int
    citation_coverage: int
    can_answer: bool
    sources: List[SourceChunk]
    refusal: Optional[RefusalDetail] = None

class DocInfo(BaseModel):
    filename: str
    size_kb: float


# --- routes ---

@app.get("/")
async def root():
    return {"status": "online", "docs": "/docs"}


@app.get("/v1/documents", response_model=List[DocInfo])
async def list_docs():
    raw_dir = "data/raw"
    if not os.path.exists(raw_dir):
        return []

    file_list = []
    for fname in os.listdir(raw_dir):
        if fname.startswith("."):
            continue
        full_path = os.path.join(raw_dir, fname)
        file_list.append(DocInfo(
            filename=fname,
            size_kb=round(os.path.getsize(full_path) / 1024, 2)
        ))
    return file_list


@app.post("/v1/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    chunks = retriever.retrieve(
        req.query,
        top_k=req.top_k,
        alpha=req.alpha,
        use_reranker=req.use_reranker,
        strategy=req.strategy
    )

    result = gen.generate_comprehensive_response(req.query, chunks)

    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))

    # build source list - just grab 200 chars as preview
    source_list = []
    for c in chunks:
        source_list.append(SourceChunk(
            id=c["id"],
            content_preview=c["content"][:200] + "...",
            source=c["metadata"].get("source", "unknown"),
            score=c.get("rrf_score", 0.0)
        ))

    # only populate refusal info if the model couldn't answer
    refusal = None
    if not result.get("can_answer"):
        rd = result.get("structured_refusal", {})
        refusal = RefusalDetail(
            what_is_found=rd.get("what_is_found"),
            what_is_missing=rd.get("what_is_missing"),
            suggested_documents=rd.get("suggested_documents")
        )

    final_answer = result.get("verified_answer") or result.get("answer", "")

    return AskResponse(
        answer=final_answer,
        confidence_score=result.get("retrieval_confidence_score", 0),
        citation_coverage=result.get("citation_coverage_pct", 0),
        can_answer=result.get("can_answer", True),
        sources=source_list,
        refusal=refusal
    )


@app.post("/v1/ingest")
async def ingest_doc(
    file: UploadFile = File(...),
    strategy: str = Query("structural", enum=["fixed", "structural", "semantic"])
):
    save_dir = "data/raw"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file.filename)

    with open(save_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    ext = os.path.splitext(file.filename)[1].lower()

    if ext == ".pdf":
        pages = doc_loader.load_pdf(save_path)
    elif ext in [".md", ".html", ".htm", ".txt"]:
        pages = doc_loader.load_unstructured(save_path)
    else:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    if not pages:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail="File was empty or unreadable")

    # join all pages into one text block before chunking
    full_text = "\n\n".join(p["content"] for p in pages)
    internal_strat = STRATEGY_MAP.get(strategy, "structure_aware")
    chunks = split_text(full_text, file.filename, strategy=internal_strat)

    stats = db.ingest_chunks(chunks)

    return {
        "filename": file.filename,
        "strategy": internal_strat,
        "chunks_created": stats["total"],
        "chunks_added": stats["added"],
        "skipped_duplicates": stats["duplicates_skipped"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
