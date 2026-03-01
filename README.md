# Local RAG Brain

**A robust, general-purpose local RAG platform for humans and AI agents.**

This project provides a fully open-source, local-first retrieval-augmented generation (RAG) system. It supports hybrid search (Vector + BM25), multimodal ingestion (BMP images), and is designed to serve as a central knowledge hub for both direct human interaction and automated agent orchestration.

## 🎯 Key Features

- **Local First:** Runs entirely on your hardware using Ollama and Sentence Transformers.
- **Hybrid Search:** Combines semantic vector search with keyword-based BM25 for maximum precision.
- **Multimodal Support:** Automatically captions and indexes BMP images via local Vision-Language Models (VLM).
- **Rich Document Support:** Ingest PDF, DOCX, HTML, CSV, Markdown, JSON, and plain text files.
- **Agent Orchestration Ready:** Standardized FastAPI service with specialized tools for agentic reasoning and self-learning.
- **Secure Ingestion:** Path traversal protection with configurable base directory via `RAG_INGEST_BASE`.
- **Async Ingestion:** High-performance asynchronous processing for directories and files.
- **Modern UI:** Interactive Streamlit interface for chat, ingestion, and parameter tuning.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Setup Ollama (Local Models)

```bash
# Pull standard models
ollama pull llama3.1
ollama pull nomic-embed-text

# Pull Vision model (optional, for image support)
ollama pull llava
```

### 3. Deployment Options

#### Option A: Docker Compose (Recommended)
```bash
docker-compose up --build
```
*Launches the Knowledge Service (port 8000) and Web UI (port 8501).*

#### Option B: Local CLI
```bash
# Launch the Web UI
rag-brain-ui

# Launch the Knowledge API (for Agents)
rag-brain-api

# Ingest data via CLI
rag-brain --ingest ./my_documents/
```

## 🤖 AI Agent Integration

Agents can use this brain as a "Collective Memory."

### Knowledge API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/query` | POST | Synthesized answer based on context. |
| `/query/stream` | POST | Streaming synthesized answer via Server-Sent Events. |
| `/search` | POST | Raw document chunks (perfect for multi-step reasoning). |
| `/add_text` | POST | Direct string ingestion (allows agents to "learn" new facts in real-time). |
| `/delete` | POST | Delete documents by ID. Request body: `{"doc_ids": ["id1","id2"]}`. |

### Tool Definitions
Standardized JSON schemas for tool-calling are provided in `src/rag_brain/tools.py`. The `get_rag_tool_definition()` function returns 6 OpenAI-compatible tools: `query_knowledge_base`, `search_documents`, `add_knowledge`, `delete_documents`, `ingest_directory`, and `stream_query`. See `examples/agent_simple.py` for a minimal integration, or `examples/agent_orchestration.py` for a richer multi-step planner-critic loop.

## ⚙️ Configuration

Customize behavior in `config.yaml`:
- **Hybrid Search:** Toggle BM25 + Vector fusion.
- **Re-ranking:** Enable Cross-Encoders for second-stage accuracy.
- **Chunking:** Adjust fragment size and overlap.

## 🔒 Security

### Path Traversal Protection
The `/ingest` endpoint and Streamlit UI validate file paths against a configurable base directory:
- **Environment variable:** `RAG_INGEST_BASE` (default: current working directory)
- **API behavior:** Paths outside the base directory are rejected with a 403 Forbidden response
- **Streamlit UI:** The sidebar ingestion input enforces the same boundary check, rejecting out-of-scope paths with an error message
- **Example:** `export RAG_INGEST_BASE=/home/user/documents` to restrict ingestion to that directory only

### BM25 Index Persistence
- **Previous format:** Pickled index (`bm25_index.pkl`)
- **New format:** JSON corpus (`bm25_corpus.json`) — safer and more portable
- **Auto-migration:** On first run, any legacy pickle files are automatically converted to JSON format

## 📊 Observability

The API emits structured JSON logs for monitoring and debugging:
- **`query_complete`** event: Logs query execution with latency (ms), retrieved document count, and top match scores
- **`ingest_complete`** event: Logs ingestion with latency (ms), total documents processed, and collection size

Example log output:
```json
{"event": "query_complete", "query": "...", "latency_ms": 234, "num_results": 5, "scores": [0.92, 0.87]}
{"event": "ingest_complete", "path": "...", "latency_ms": 1200, "docs_added": 42, "collection_size": 987}
```

## 🧪 Evaluation

RAGAS (Retrieval-Augmented Generation Assessment) evaluation is available in `scripts/evaluate.py`:

```bash
python scripts/evaluate.py --testset examples/eval_testset.jsonl --output results.csv
```

Requires optional evaluation dependencies:
```bash
pip install ragas langchain-community datasets
```

Includes a sample test set in `examples/eval_testset.jsonl` with ground-truth QA pairs for benchmarking retrieval quality.

## 📄 License

MIT License - See [LICENSE](LICENSE) file.
