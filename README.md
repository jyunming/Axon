# Local RAG Brain

**A robust, general-purpose local RAG platform for humans and AI agents.**

This project provides a fully open-source, local-first retrieval-augmented generation (RAG) system. It supports hybrid search (Vector + BM25), multimodal ingestion (BMP images), and is designed to serve as a central knowledge hub for both direct human interaction and automated agent orchestration.

## 🎯 Key Features

- **Local First:** Runs entirely on your hardware using Ollama and Sentence Transformers.
- **Hybrid Search:** Combines semantic vector search with keyword-based BM25 for maximum precision.
- **Multimodal Support:** Automatically captions and indexes BMP images via local Vision-Language Models (VLM).
- **Agent Orchestration Ready:** Standardized FastAPI service with specialized tools for agentic reasoning and self-learning.
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
- `POST /query`: Synthesized answer based on context.
- `POST /search`: Raw document chunks (perfect for multi-step reasoning).
- `POST /add_text`: Direct string ingestion (allows agents to "learn" new facts in real-time).

### Tool Definitions
Standardized JSON schemas for tool-calling are provided in `src/rag_brain/tools.py`. See `examples/agent_simple.py` for a reference implementation.

## ⚙️ Configuration

Customize behavior in `config.yaml`:
- **Hybrid Search:** Toggle BM25 + Vector fusion.
- **Re-ranking:** Enable Cross-Encoders for second-stage accuracy.
- **Chunking:** Adjust fragment size and overlap.

## 📄 License

MIT License - See [LICENSE](LICENSE) file.
