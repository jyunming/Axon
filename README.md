# Axon

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**A robust, general-purpose local RAG platform for humans and AI agents.**

This project provides a fully open-source, local-first retrieval-augmented generation (RAG) system. It supports hybrid search (Vector + BM25), multimodal ingestion (BMP, PNG, TIF/TIFF, PGM images), and is designed to serve as a central knowledge hub for both direct human interaction and automated agent orchestration.

---

## 📚 Documentation

- **[Getting Started](GETTING_STARTED.md)** - New here? Start with this
- **[Setup Guide](SETUP.md)** - Detailed step-by-step setup for LLM and embedding models
- **[Quick Reference](QUICKREF.md)** - Common commands and examples
- **[Model Guide](MODEL_GUIDE.md)** - Supported models and configuration
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues and fixes
- **[Development Guide](DEVELOPMENT.md)** - Setup and development workflow
- **[Contributing](CONTRIBUTING.md)** - How to contribute
- **[Security Policy](SECURITY.md)** - Security best practices
- **[SOTA Analysis](SOTA_ANALYSIS.md)** - Gap analysis vs. state-of-the-art RAG systems and roadmap

---

## 🎯 Key Features

- **Local First:** Runs entirely on your hardware using Ollama and Sentence Transformers. **No Docker required** — `pip install -e .` and `axon` gets you started.
- **Multi-LLM & Embedding Support:** Switch LLM provider (Ollama, Gemini, OpenAI, Ollama Cloud, vLLM) and embedding provider (sentence-transformers, Ollama, FastEmbed) live from the REPL or web UI sidebar.
- **Hybrid Search:** Combines semantic vector search with keyword-based BM25 for maximum precision.
- **Truth Grounding (Web Search):** Agentic Brave Search fallback — fires automatically only when local knowledge is insufficient (max cosine score < similarity threshold).
- **Project-Based Knowledge Bases:** Create named projects (`/project new research`) with completely isolated vector stores, BM25 indexes, and sessions. Switch instantly with `/project switch`.
- **Knowledge Base Viewer:** Browse all ingested documents and chunk counts from the web UI sidebar or via `axon --list`.
- **Chat Sessions:** Create, switch, and delete independent conversation sessions with full history persistence (auto-saved, persists across restarts).
- **Rich Interactive REPL:** Markdown rendering, animated spinners, tab completion, `@file` context attachment, `!cmd` shell passthrough, `/retry`, and a pinned status bar.
- **Conversational Memory:** The LLM remembers your previous messages within a session for natural follow-up questions.
- **Multimodal Support:** Automatically captions and indexes BMP, PNG, TIF/TIFF, and PGM images via local Vision-Language Models (VLM).
- **Rich Document Support:** Ingest PDF, DOCX, HTML, CSV/TSV, Markdown, JSON, and plain text files.
- **Agent Orchestration Ready:** Standardized FastAPI service with specialized tools for agentic reasoning and self-learning.
- **Secure Ingestion:** Path traversal protection with configurable base directory via `RAG_INGEST_BASE`.
- **Async Ingestion:** High-performance asynchronous processing for directories and files.
- **Modern UI:** Interactive Streamlit interface for chat, ingestion, and parameter tuning.

## 🖥️ Preview

![Axon REPL](docs/assets/repl-demo.png)

*The interactive REPL — always-visible header box with model, search and discuss settings; live Tab autocomplete for all slash commands; pinned two-line status toolbar.*

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install package with dependencies
pip install -e .

# Or with development tools
pip install -e ".[dev]"

# Or with all optional features
pip install -e ".[all]"
```

### 2. Setup Ollama (Local Models)

```bash
# Install Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Start the Ollama server (Linux; macOS/Windows start automatically)
ollama serve

# Verify Ollama is running
curl http://localhost:11434/api/tags

# Pull LLM (choose one based on your hardware)
ollama pull llama3.1:8b       # Recommended (4.7 GB, ~6-8 GB RAM)
ollama pull phi3:mini         # Minimal (2.3 GB, ~4-6 GB RAM)

# Pull embedding model (optional; sentence-transformers is used by default)
ollama pull nomic-embed-text

# Pull Vision model (optional, for image support)
ollama pull llava
```

> For detailed setup instructions including embedding model options, configuration examples, and troubleshooting, see the **[Setup Guide](SETUP.md)**.

### 3. Deployment Options

#### Option A: Docker Compose (Recommended for API + UI)
```bash
docker-compose up --build
```
*Launches the Knowledge Service (port 8000) and Web UI (port 8501).*

#### Option B: Local (No Docker Required)

> **No Docker needed!** Run everything directly on your machine.

```bash
# 1. Install the package
pip install -e .

# 2. Start Ollama and pull a model (one-time setup)
ollama pull gemma           # default LLM
ollama pull nomic-embed-text  # optional — higher-quality embeddings

# 3. Launch what you need
axon                   # interactive REPL (no server required)
axon-api               # FastAPI server on :8000 (optional)
axon-ui                # Streamlit web UI on :8501 (optional)

# Or use make shortcuts
make run-cli                # alias for axon
make run-all                # starts both API and UI
```

No ports, no containers, no compose file — just `pip install` and go.

> **Windows users:** For correct rendering of box-drawing characters and emoji, use
> [Windows Terminal](https://aka.ms/terminal) (recommended) or add `$env:PYTHONUTF8=1`
> to your PowerShell profile (`notepad $PROFILE`).

#### Option C: CLI Commands
```bash
# Interactive REPL (default — no args)
axon

# Single-shot query
axon "What is the main topic?"

# Stream response token-by-token
axon --stream "Summarise my documents"

# Switch model at runtime (auto-pulls if not available locally)
axon --model gemma:2b "Your question"
axon --model gemini/gemini-2.5-flash-lite "Your question"
axon --model openai/gpt-4o "Your question"

# Pull a model explicitly
axon --pull gemma:2b

# List available providers and locally installed Ollama models
axon --list-models

# List all ingested documents in the knowledge base
axon --list

# Ingest data
axon --ingest ./my_documents/

# Embedding model override
axon --embed ollama/nomic-embed-text "Your question"

# RAG behaviour flags (all also settable live from REPL)
axon --no-discuss "Your question"   # refuse to answer outside documents
axon --discuss "Your question"      # allow general knowledge fallback (default)
axon --search "Your question"       # enable Brave web search (requires BRAVE_API_KEY)
axon --top-k 5 "Your question"      # retrieve 5 chunks (default: 10)
axon --threshold 0.5 "Your question" # stricter similarity cutoff (default: 0.3)
axon --no-hybrid "Your question"    # vector-only search (disable BM25)
axon --rerank "Your question"       # enable cross-encoder reranking
axon --hyde "Your question"         # enable HyDE query expansion
axon --multi-query "Your question"  # enable multi-query retrieval

# Project management (all also available from REPL /project)
axon --project-list                 # list all projects
axon --project-new research --ingest ./papers/   # create project + ingest in one step
axon --project research "Your question"          # query within a project
axon --project-delete research                   # delete a project

# Quiet mode (for pipes and CI)
echo "What is X?" | axon -q
```

**Interactive REPL slash commands:**
| Command | Description |
|---|---|
| `/help [cmd]` | Show all commands or detailed help for a specific command (model, embed, ingest, rag, sessions) |
| `/list` | List ingested documents with chunk counts |
| `/ingest <path\|glob>` | Ingest files or directories using glob patterns |
| `/model [provider/model]` | Switch LLM provider and model on the fly; bare `/model <name>` auto-detects provider (auto-pulls from Ollama if needed) |
| `/embed [provider/model]` | Switch embedding provider and model |
| `/pull <name>` | Pull an Ollama model with progress indicator |
| `/vllm-url [url]` | Show or set the vLLM server base URL at runtime (e.g. `http://localhost:8000/v1`) |
| `/search` | Toggle Brave web search (truth_grounding) for knowledge grounding |
| `/discuss` | Toggle discuss mode — when OFF the LLM refuses queries with no document match |
| `/rag [option]` | Show or modify RAG settings: `topk`, `threshold`, `hybrid`, `rerank`, `rerank-model`, `hyde`, `multi`, `step-back`, `decompose`, `compress`, `cite`, `raptor`, `graph-rag` |
| `/project [list\|new\|switch\|delete\|folder]` | Manage named projects with isolated knowledge bases |
| `/keys [set provider]` | Show API key status for all providers, or interactively set a key |
| `/compact` | Summarize chat history via LLM to free context window |
| `/context` | Display token usage bar, model info, RAG settings, chat history, and retrieved sources |
| `/sessions` | List recent saved sessions |
| `/resume <id>` | Load a previous session by ID |
| `/retry` | Re-send the last query (useful after switching model or settings) |
| `/clear` | Clear chat history |
| `/quit`, `/exit` | Exit the REPL |
| `!<cmd>` | Run a shell command without leaving the REPL (e.g. `!ls ./docs`) |
| `@<path>` | Attach file contents inline to your query (e.g. `review @./config.yaml`) |

**REPL Features:**
- **Live Tab Completion:** Slash commands, `@file` paths, project names, and Ollama model names auto-complete as you type
- **Animated Spinners:** Braille spinner (⠋⠙⠹…) shows status during initialization and while the LLM generates responses
- **Rich Markdown Rendering:** Responses are rendered with syntax highlighting and word-wrap via the `rich` library
- **Session Persistence:** Chat history auto-saves to `~/.axon/sessions/session_<timestamp>.json`; resume any past session on startup
- **Input History Persistence:** ↑↓ arrows cycle through history across sessions (saved to `~/.axon/repl_history`)
- **Project-Based Knowledge Isolation:** Each project has its own vector store, BM25 index, and sessions under `~/.axon/projects/<name>/`
- **Pinned Status Bar:** LLM model, embedding model, discuss, search, and hybrid settings always visible at the terminal bottom even during streaming
- **Context Window Visibility:** `/context` shows exact token counts, model context limits, and retrieved sources with scores
- **Keyboard shortcuts:** `Tab` complete · `↑↓` history · `Ctrl+C` cancel generation · `Ctrl+L` clear screen · `Ctrl+D` exit

## 🤖 AI Agent Integration

Agents can use Axon as a "Collective Memory."

### Knowledge API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/query` | POST | Synthesized answer based on context. |
| `/query/stream` | POST | Streaming synthesized answer via Server-Sent Events. |
| `/search` | POST | Raw document chunks (perfect for multi-step reasoning). |
| `/add_text` | POST | Direct string ingestion (allows agents to "learn" new facts in real-time). |
| `/delete` | POST | Delete documents by ID. Request body: `{"doc_ids": ["id1","id2"]}`. |
| `/ingest` | POST | Background directory/file ingestion. Request body: `{"path": "/path/to/docs"}`. Path must be within `RAG_INGEST_BASE`. |
| `/collection` | GET | List all ingested sources with chunk counts. Returns `{total_files, total_chunks, files:[{source,chunks}]}`. |
| `/collection/stale` | GET | List ingested sources that have not been re-ingested within `days` (default 7) based on the in-memory dedup cache; results only cover activity since the last server restart. Optional query: `?days=N`. |
| `/add_texts` | POST | Batch string ingestion (multiple texts at once). |
| `/ingest_url` | POST | Ingest a document from a URL. Body: `{"url": "https://..."}`. |
| `/ingest/status/{job_id}` | GET | Poll background ingest job status by ID. |
| `/health` | GET | Health check — returns `{"status": "healthy", "axon_ready": <bool>}`. |
| `/projects` | GET | List all projects and their metadata. |
| `/project/switch` | POST | Switch the active project. Body: `{"project": "name"}`. |

### Tool Definitions
Standardized JSON schemas for tool-calling are provided in `src/axon/tools.py`. The `get_rag_tool_definition()` function returns 6 OpenAI-compatible tools: `query_knowledge_base`, `search_documents`, `add_knowledge`, `delete_documents`, `ingest_directory`, and `stream_query`. See `examples/agent_simple.py` for a minimal integration, or `examples/agent_orchestration.py` for a richer multi-step planner-critic loop.

## ⚙️ Configuration

Customize behavior in `config.yaml`:
- **LLM Providers:** Ollama (local), Gemini, OpenAI, Ollama Cloud, or vLLM.
- **Embedding Providers:** Sentence Transformers (default), Ollama, FastEmbed, or OpenAI.
- **Vector Stores:** ChromaDB (default), Qdrant, or LanceDB.
- **Hybrid Search:** Combine vector + BM25 keyword search for maximum precision.
- **Re-ranking:** Optional Cross-Encoder or LLM-based second-stage filtering.
- **Query Transformations:** Enable Multi-Query expansion or HyDE for improved retrieval.
- **Truth Grounding:** Enable Brave Search API fallback when local knowledge is insufficient.
- **Discussion Fallback:** Allow the LLM to answer from general knowledge when no documents match (default: enabled).
- **Chunking:** Adjust document fragment size and overlap (default: 1000 chars, 200 overlap).
- **Offline / Air-gapped Mode:** Run fully without internet — all models resolved from a local directory, web search disabled automatically.

See [Configuration Guide](QUICKREF.md#configuration) for details.

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

For security concerns, please also review our [Security Policy](SECURITY.md).

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

## 🧪 Development

### Running Tests
```bash
# Run all tests
make test

# With coverage
make test-cov

# Format and lint
make format
make lint
```

### Code Quality
This project uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **pytest** for testing
- **pre-commit** hooks for automated checks

See [Development Guide](DEVELOPMENT.md) for more details.

## 🤝 Contributing

We welcome contributions! Please see:
- [Contributing Guide](CONTRIBUTING.md) for guidelines
- [Development Guide](DEVELOPMENT.md) for setup instructions
- [Security Policy](SECURITY.md) for security best practices

## 📄 License

MIT License - See [LICENSE](LICENSE) file.
