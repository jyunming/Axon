from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import os
import uvicorn
import logging
from rag_brain.main import OpenStudioBrain, OpenStudioConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAGBrainAPI")

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

class SearchRequest(BaseModel):
    query: str = Field(..., description="The query string for semantic search")
    top_k: Optional[int] = Field(None, description="Number of documents to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Metadata filters")

class IngestRequest(BaseModel):
    path: str = Field(..., description="Path to a file or directory to ingest")

class TextIngestRequest(BaseModel):
    text: str = Field(..., description="The content to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata for the document")
    doc_id: Optional[str] = Field(None, description="Optional unique ID for the document")

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
        response = brain.query(request.query, filters=request.filters)
        return {"query": request.query, "response": response}
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
    
    if not os.path.exists(request.path):
        raise HTTPException(status_code=404, detail="Path does not exist")

    def process_ingestion():
        import asyncio
        if os.path.isdir(request.path):
            asyncio.run(brain.load_directory(request.path))
        else:
            from rag_brain.loaders import DirectoryLoader
            ext = os.path.splitext(request.path)[1].lower()
            loader_mgr = DirectoryLoader()
            if ext in loader_mgr.loaders:
                docs = loader_mgr.loaders[ext].load(request.path)
                brain.ingest(docs)

    background_tasks.add_task(process_ingestion)
    return {"message": f"Ingestion started for {request.path}", "status": "processing"}

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

def main():
    """Main entry point for rag-brain-api command."""
    host = os.getenv("RAG_BRAIN_HOST", "0.0.0.0")
    port = int(os.getenv("RAG_BRAIN_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
