from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import pathlib
import uvicorn
import logging
from rag_brain.main import OpenStudioBrain, OpenStudioConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGBrainAPI")


def _validate_ingest_path(path: str) -> str:
    """Validate that path is within the allowed base directory."""
    allowed_base = os.path.abspath(os.getenv("RAG_INGEST_BASE", "."))
    abs_path = os.path.abspath(path)
    if not abs_path.startswith(allowed_base):
        raise HTTPException(
            status_code=403,
            detail=f"Path '{path}' is outside the allowed ingest directory. Set RAG_INGEST_BASE to permit additional paths."
        )
    return abs_path

app = FastAPI(
    title="Local RAG Brain API",
    description="REST API for agent orchestration and document retrieval",
    version="2.0.0"
)

# Global Brain Instance
brain: Optional[OpenStudioBrain] = None

@app.on_event("startup")
async def startup_event():
    global brain
    try:
        config = OpenStudioConfig.load()
        brain = OpenStudioBrain(config)
        logger.info("✅ RAG Brain initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG Brain: {e}")

# Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question or prompt to ask the brain")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters for retrieval")
    stream: bool = Field(False, description="Whether to stream the response (not fully implemented in REST yet)")
    # Per-request RAG overrides (match CLI flags exactly)
    top_k: Optional[int] = Field(None, description="Override number of chunks to retrieve")
    threshold: Optional[float] = Field(None, description="Override similarity threshold (0.0–1.0)")
    hybrid: Optional[bool] = Field(None, description="Override hybrid BM25+vector search toggle")
    rerank: Optional[bool] = Field(None, description="Override cross-encoder re-ranking toggle")
    hyde: Optional[bool] = Field(None, description="Override HyDE query transformation toggle")
    multi_query: Optional[bool] = Field(None, description="Override multi-query retrieval toggle")
    step_back: Optional[bool] = Field(None, description="Override step-back prompting toggle")
    decompose: Optional[bool] = Field(None, description="Override query decomposition toggle")
    compress: Optional[bool] = Field(None, description="Override LLM context compression toggle")
    discuss: Optional[bool] = Field(None, description="Override discussion fallback toggle")
    cite: Optional[bool] = Field(None, description="Override inline source citation toggle")

class SearchRequest(BaseModel):
    query: str = Field(..., description="The query string for semantic search")
    top_k: Optional[int] = Field(None, description="Number of documents to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")

class IngestRequest(BaseModel):
    path: str = Field(..., description="Path to a file or directory to ingest. Must be within RAG_INGEST_BASE (default: current working directory).")

class TextIngestRequest(BaseModel):
    text: str = Field(..., description="The content to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata for the document")
    doc_id: Optional[str] = Field(None, description="Optional unique ID for the document")

class DeleteRequest(BaseModel):
    doc_ids: List[str] = Field(..., description="List of document IDs to delete")

class SearchResult(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]

# Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "brain_ready": brain is not None}

@app.post("/query")
async def query_brain(request: QueryRequest):
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        overrides = {
            "top_k": request.top_k,
            "similarity_threshold": request.threshold,
            "hybrid_search": request.hybrid,
            "rerank": request.rerank,
            "hyde": request.hyde,
            "multi_query": request.multi_query,
            "step_back": request.step_back,
            "query_decompose": request.decompose,
            "compress_context": request.compress,
            "discussion_fallback": request.discuss,
            "cite_sources": request.cite,
        }
        response = brain.query(request.query, filters=request.filters, overrides=overrides)
        cfg = brain._apply_overrides(overrides)
        settings = {
            "top_k": cfg.top_k,
            "hybrid": cfg.hybrid_search,
            "rerank": cfg.rerank,
            "hyde": cfg.hyde,
            "multi_query": cfg.multi_query,
            "step_back": cfg.step_back,
            "decompose": cfg.query_decompose,
            "compress": cfg.compress_context,
            "discuss": cfg.discussion_fallback,
            "cite": cfg.cite_sources,
        }
        return {"query": request.query, "response": response, "settings": settings}
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def query_brain_stream(request: QueryRequest):
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    overrides = {
        "top_k": request.top_k,
        "similarity_threshold": request.threshold,
        "hybrid_search": request.hybrid,
        "rerank": request.rerank,
        "hyde": request.hyde,
        "multi_query": request.multi_query,
        "step_back": request.step_back,
        "query_decompose": request.decompose,
        "compress_context": request.compress,
        "discussion_fallback": request.discuss,
        "cite_sources": request.cite,
    }

    def generate():
        try:
            import json
            for chunk in brain.query_stream(request.query, filters=request.filters, overrides=overrides):
                if isinstance(chunk, dict):
                    yield f"data: {json.dumps(chunk)}\n\n"
                else:
                    yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/search", response_model=List[SearchResult])
async def search_brain(request: SearchRequest):
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        # We need to expose the search method from OpenStudioBrain more directly
        # or use the vector_store directly.
        # For agentic use, we'll implement a search flow here.
        query_embedding = brain.embedding.embed_query(request.query)
        top_k = request.top_k or brain.config.top_k
        
        results = brain.vector_store.search(
            query_embedding, 
            top_k=top_k, 
            filter_dict=request.filters
        )
        return results
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_data(request: IngestRequest, background_tasks: BackgroundTasks):
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")

    validated_path = _validate_ingest_path(request.path)

    try:
        requested_path = pathlib.Path(validated_path).resolve()
        if not requested_path.exists():
            raise HTTPException(status_code=404, detail="Path does not exist")
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid path: {str(e)}")

    def process_ingestion():
        import asyncio
        try:
            if requested_path.is_dir():
                asyncio.run(brain.load_directory(str(requested_path)))
            else:
                from rag_brain.loaders import DirectoryLoader
                ext = requested_path.suffix.lower()
                loader_mgr = DirectoryLoader()
                if ext in loader_mgr.loaders:
                    docs = loader_mgr.loaders[ext].load(str(requested_path))
                    brain.ingest(docs)
                else:
                    logger.warning(f"Unsupported file type: {ext}")
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")

    background_tasks.add_task(process_ingestion)
    return {"message": f"Ingestion started for {validated_path}", "status": "processing"}

@app.post("/add_text")
async def add_text(request: TextIngestRequest):
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    import uuid
    doc_id = request.doc_id or f"agent_doc_{uuid.uuid4().hex[:8]}"
    
    doc = {
        "id": doc_id,
        "text": request.text,
        "metadata": request.metadata or {"source": "api_agent", "type": "agent_input"}
    }
    
    try:
        brain.ingest([doc])
        return {"status": "success", "doc_id": doc_id}
    except Exception as e:
        logger.error(f"Error adding text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collection")
async def get_collection():
    """List all unique sources in the knowledge base with chunk counts."""
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        docs = brain.list_documents()
        return {
            "total_files": len(docs),
            "total_chunks": sum(d["chunks"] for d in docs),
            "files": [{"source": d["source"], "chunks": d["chunks"]} for d in docs],
        }
    except Exception as e:
        logger.error(f"Error listing collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete")
async def delete_documents(request: DeleteRequest):
    if brain is None:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    try:
        # Delete from ChromaDB if provider is chroma
        if brain.vector_store.provider == "chroma":
            brain.vector_store.collection.delete(ids=request.doc_ids)
        # Delete from BM25
        if brain.bm25 is not None:
            brain.bm25.delete_documents(request.doc_ids)
        return {"status": "success", "deleted": len(request.doc_ids), "doc_ids": request.doc_ids}
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point for rag-brain-api command."""
    host = os.getenv("RAG_BRAIN_HOST", "0.0.0.0")
    port = int(os.getenv("RAG_BRAIN_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
