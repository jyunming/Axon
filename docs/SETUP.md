# Setup Guide — Axon

This guide walks through setting up the application from scratch, including detailed steps for configuring every LLM and embedding model option.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install the Package](#2-install-the-package)
3. [Install and Start Ollama](#3-install-and-start-ollama)
4. [LLM Setup](#4-llm-setup)
5. [Embedding Model Setup](#5-embedding-model-setup)
6. [Optional Features (Code Corpus & Vision)](#6-optional-features-code-corpus--vision)
7. [Configure config.yaml](#7-configure-configyaml)
8. [Configure .env](#8-configure-env)
9. [Verify the Full Setup](#9-verify-the-full-setup)
10. [MCP Server Setup](#10-mcp-server-setup)
11. [VS Code Extension (GitHub Copilot Integration)](#11-vs-code-extension-github-copilot-integration)
    - [11b. REPL Security — Shell Passthrough](#11b-repl-security--shell-passthrough)
12. [Upgrading Axon](#12-upgrading-axon)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

Before starting, ensure you have:

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.10+ | `python --version` |
| **pip** | Latest | `pip install --upgrade pip` |
| **git** | Any | For cloning the repo |
| **Disk space** | 10–20 GB free | For models; depends on which you pull |
| **RAM** | 16 GB minimum | 32 GB recommended |

> **GPU (optional):** A CUDA-compatible GPU significantly speeds up inference. Ollama automatically uses the GPU if available. CPU-only works but is slower.
> **Windows:** Use [Windows Terminal](https://aka.ms/terminal) for best results. Add `$env:PYTHONUTF8=1` to your PowerShell profile (or run it before starting Axon) to avoid encoding errors when reading documents that contain non-ASCII characters:
> ```powershell
> # Add to $PROFILE to make permanent
> $env:PYTHONUTF8 = "1"
> ```

Clone the repository if you haven't already:

```bash
git clone https://github.com/jyunming/Axon.git
cd Axon
```

---

## 2. Install the Package

Create and activate a Python virtual environment:

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

Install the package. Choose the option that matches your needs:

```bash
# Minimal install (sentence-transformers + LanceDB + Ollama LLM)
pip install -e .
# With development tools (tests, linting, type checking)
pip install -e ".[dev]"
# With FastEmbed embedding support
pip install -e ".[fastembed]"
# With Qdrant vector store
pip install -e ".[qdrant]"
# Everything (all optional features + dev tools)
pip install -e ".[all]"
```

### Optional Feature Extras

| Extra | What it enables | Install |
|---|---|---|
| `graphrag` | GraphRAG community detection using the Leiden algorithm (better than the default networkx Louvain fallback) | `pip install -e ".[graphrag]"` |
| `gliner` | GLiNER fast NER backend for entity extraction — skips the LLM call during ingest (`graph_rag_ner_backend: gliner` in config) | `pip install -e ".[gliner]"` |
| `llmlingua` | LLMLingua-2 token compression for GraphRAG community reports before map-reduce (`graph_rag_report_compress: true` in config) | `pip install -e ".[llmlingua]"` |
| `loaders` | EPUB, RTF, and `.msg` (Outlook) file loaders | `pip install -e ".[loaders]"` |

> **`graphrag` extra and Python 3.13+:** The `[graphrag]` extra uses `leidenalg` + `igraph`, which ship pre-built wheels for Python 3.13 on all platforms.
> The older `graspologic` package (v0.3.x) is **not compatible** with Python 3.13 or NumPy 2.x — do not install it on Python 3.13.
> The code default is `graph_rag_community_backend: louvain` (networkx Louvain, no extra install required). For Python 3.13 we recommend pinning to `leidenalg` explicitly for better performance:
>
> ```yaml
> rag:
>   graph_rag_community_backend: leidenalg  # recommended for Python 3.13
> ```

Verify the install:

```bash
axon --help
```

You should see the CLI help output. If you get a `command not found` error, ensure your virtual environment is activated.

---

## 3. Install and Start Ollama

Ollama is required for the LLM and optionally for embeddings. **This is the most commonly missed step.**

### 3.1 Install Ollama

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**

Download from [ollama.com/download](https://ollama.com/download) and run the installer. Ollama runs as a background application.

**Windows:**

Download the installer from [ollama.com/download](https://ollama.com/download). After installation, Ollama starts automatically in the system tray.

### 3.2 Start the Ollama Server

On Linux (or if Ollama isn't running automatically):

```bash
ollama serve
```

Leave this terminal open, or run it in the background:

```bash
ollama serve &
```

On macOS/Windows, Ollama starts automatically when you open the app. You can verify in the system tray.

### 3.3 Verify Ollama is Running

```bash
curl http://localhost:11434/api/tags
```

Expected response (list of pulled models — may be empty on first run):

```json
{"models":[]}
```

If you get `connection refused`, Ollama is not running. Start it with `ollama serve`.

---

## 4. LLM Setup

Choose one LLM tier based on your hardware. The LLM is used to synthesize answers from retrieved document chunks.

### Tier 1 — Minimal (Low RAM / CPU only)

**Model:** Phi-3 Mini 3.8B
**Disk:** ~2.3 GB
**RAM needed:** 4–6 GB
**Speed (CPU):** ~15 tokens/sec

```bash
ollama pull phi3:mini
```

Update `config.yaml`:
```yaml
llm:
  provider: ollama
  model: phi3:mini
```

**Best for:** Development, testing, or machines with 8–16 GB RAM.

---

### Tier 2 — Balanced (Recommended)

**Model:** Llama 3.1 8B
**Disk:** ~4.7 GB
**RAM needed:** 6–8 GB
**Speed (GPU RTX 3060):** ~25 tokens/sec

```bash
ollama pull llama3.1:8b
```

Update `config.yaml`:
```yaml
llm:
  provider: ollama
  model: llama3.1:8b
```

**Best for:** Most use cases. Strong instruction following, 128K context window.

---

### Tier 3 — Quality / Multilingual

**Model:** Qwen2.5 7B
**Disk:** ~4.5 GB
**RAM needed:** 6–8 GB
**Speed (GPU RTX 3060):** ~30 tokens/sec

```bash
ollama pull qwen2.5:7b
```

Update `config.yaml`:
```yaml
llm:
  provider: ollama
  model: qwen2.5:7b
```

**Best for:** Technical documents, multilingual content, strong coding tasks.

---

### Tier 4 — Maximum Quality (GPU required)

**Model:** Llama 3.1 70B
**Disk:** ~40 GB
**VRAM needed:** 40–48 GB (or CPU with 64 GB RAM, very slow)
**Speed (GPU A100):** ~30 tokens/sec

```bash
ollama pull llama3.1:70b
```

Update `config.yaml`:
```yaml
llm:
  provider: ollama
  model: llama3.1:70b
```

**Best for:** Maximum quality answers when hardware allows.

---

### Verify the LLM

After pulling your chosen model, verify it works:

```bash
ollama run llama3.1:8b "Say hello in one sentence."
```

You should see the model respond. Press `Ctrl+D` to exit.

---

## 5. Embedding Model Setup

Embeddings convert text into vectors for semantic search. Choose one provider.

> **Important:** Once you ingest documents with a specific embedding model, you must keep using that model. Changing the embedding model requires re-ingesting all documents (the vector dimensions will not match).

---

### Option A — sentence-transformers (Default, No Ollama Required)

This is the default option. The model downloads automatically on first use.

**Install (if not already installed with the package):**
```bash
pip install sentence-transformers
```

**No manual model download needed** — it downloads on first run. To pre-download:

```python
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

**Config:**
```yaml
embedding:
  provider: sentence_transformers
  model: all-MiniLM-L6-v2
```

**Available models:**

| Model | Dims | Context | Quality | Speed |
|-------|------|---------|---------|-------|
| `all-MiniLM-L6-v2` | 384 | 256 tokens | Good | Very fast |
| `all-mpnet-base-v2` | 768 | 512 tokens | Better | Slower |
| `BAAI/bge-small-en-v1.5` | 384 | 512 tokens | Good | Fast |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 tokens | Best | Slowest |

**Verify:**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode("test sentence")
print(f"Embedding dimension: {len(embedding)}")  # Should print 384
```

---

### Option B — Ollama Embeddings (Best Quality, Requires Ollama)

Uses Ollama to serve embedding models locally. Better quality than `all-MiniLM-L6-v2` for retrieval.

**Pull an embedding model:**

```bash
# Recommended: nomic-embed-text (137M params, 768-dim, 8192 token context)
ollama pull nomic-embed-text
# Alternative: larger, stronger (335M params, 1024-dim)
ollama pull mxbai-embed-large
```

**Config:**
```yaml
embedding:
  provider: ollama
  model: nomic-embed-text
```

**Available Ollama embedding models:**

| Model | Dims | Context | Size | Quality |
|-------|------|---------|------|---------|
| `nomic-embed-text` | 768 | 8192 tokens | ~550 MB | Excellent |
| `mxbai-embed-large` | 1024 | 512 tokens | ~700 MB | Excellent |
| `all-minilm` | 384 | 256 tokens | ~80 MB | Good |

**Verify:**
```bash
curl http://localhost:11434/api/embeddings \
  -d '{"model": "nomic-embed-text", "prompt": "test sentence"}'
```

Expected response: `{"embedding": [0.123, -0.456, ...]}`

---

### Option C — FastEmbed (Optimized Batch Processing)

FastEmbed uses ONNX-optimized models for fast CPU inference. Models auto-download on first use.

**Install:**
```bash
pip install fastembed
# Or if using the package extras:
pip install -e ".[fastembed]"
```

**Config:**
```yaml
embedding:
  provider: fastembed
  model: BAAI/bge-small-en-v1.5
```

**Available FastEmbed models:**

| Model | Dims | Context | Quality | Notes |
|-------|------|---------|---------|-------|
| `BAAI/bge-small-en-v1.5` | 384 | 512 tokens | Good | Default; fastest |
| `BAAI/bge-base-en-v1.5` | 768 | 512 tokens | Better | Balanced |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 tokens | Best (English) | Slower |
| `BAAI/bge-m3` | 1024 | 8192 tokens | Best (multilingual) | Long-context; recommended for prose-heavy corpora |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 tokens | Good | Lightweight |

**Using BGE-M3 (recommended for long documents and multilingual content):**
```yaml
embedding:
  provider: fastembed
  model: BAAI/bge-m3
```

> **Dimension note:** BGE-M3 produces 1024-dim vectors. If you switch to BGE-M3 after
> ingesting data with a 384-dim model (e.g. `bge-small`), ChromaDB will raise a dimension
> mismatch error. Start a fresh project or run `axon --reset` before switching.

**Verify BGE-M3:**
```python
from fastembed import TextEmbedding
model = TextEmbedding("BAAI/bge-m3")
embeddings = list(model.embed(["test sentence"]))
print(f"Embedding dimension: {len(embeddings[0])}")  # Should print 1024
```

**Verify bge-small (default):**
```python
from fastembed import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = list(model.embed(["test sentence"]))
print(f"Embedding dimension: {len(embeddings[0])}")  # Should print 384
```

---

## 6. Optional Features (Code Corpus & Vision)

Both features are opt-in — skip this section on your first setup.

### 6.1 Code Corpus (Ingesting Source Code)

No extra install needed. Point `/ingest` at a code directory and Axon auto-detects code files, splitting them with syntax-aware chunking (Python AST for `.py`, regex boundaries for everything else).

**Supported languages:** Python, Go, Rust, TypeScript, JavaScript, Java, C/C++, C#, Ruby, Swift, Kotlin, PHP, and more.

To also build a structural graph (file → function/class edges, import links):

```yaml
# config.yaml
rag:
  code_graph: true        # build File/Symbol nodes with CONTAINS and IMPORTS edges
  code_graph_bridge: true # also link prose chunks that mention code symbols
```

> Note: `graph_rag` is automatically disabled for code corpora — use `code_graph` instead.

### 6.2 Vision / Images

To ingest image files, pull the LLaVA vision model first (requires ~5 GB disk and ~8 GB RAM/VRAM):

```bash
ollama pull llava
```

No config change needed. When Axon encounters a supported image file (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.pgm`) during ingest, it sends it to LLaVA for a text caption and indexes that caption.

---

## 7. Configure config.yaml

Axon auto-creates `~/.config/axon/config.yaml` on first run with sensible defaults.

**Quickest option — interactive wizard:**

```bash
axon --setup
```

The wizard walks you through provider, model, embedding, and RAG flag selection with
validated inputs and sensible defaults. No manual YAML editing required. You can also
run it at any time from inside the REPL with `/config wizard`.

To edit the file directly instead:

```bash
# Linux / macOS
$EDITOR ~/.config/axon/config.yaml
# Windows (PowerShell)
notepad $env:USERPROFILE\.config\axon\config.yaml
```

> **Data storage:** Your knowledge base always uses the AxonStore structure at `~/.axon/AxonStore/<username>/` by default. To move it to a shared drive or different disk, set `store.base` in `config.yaml` or the `AXON_STORE_BASE` env var — see [AXON_STORE.md](AXON_STORE.md).

Here are complete configurations for each tier:

### Minimal Setup (CPU, low memory)
```yaml
embedding:
  provider: sentence_transformers
  model: all-MiniLM-L6-v2
llm:
  provider: ollama
  model: phi3:mini
  base_url: http://localhost:11434
  temperature: 0.7
  max_tokens: 2048
vector_store:
  provider: lancedb   # path derived automatically from store.base
rag:
  top_k: 10
  similarity_threshold: 0.3
  hybrid_search: true
chunk:
  size: 1000
  overlap: 200
rerank:
  enabled: false
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Balanced Setup (Recommended)
```yaml
embedding:
  provider: sentence_transformers
  model: all-MiniLM-L6-v2
llm:
  provider: ollama
  model: llama3.1:8b
  base_url: http://localhost:11434
  temperature: 0.7
  max_tokens: 2048
vector_store:
  provider: lancedb   # path derived automatically from store.base
rag:
  top_k: 10
  similarity_threshold: 0.3
  hybrid_search: true
chunk:
  size: 1000
  overlap: 200
rerank:
  enabled: false
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

> For all available `rag:` flags (GraphRAG, RAPTOR, contextual retrieval, query router, etc.) see [ADMIN_REFERENCE.md § 6](ADMIN_REFERENCE.md).

### Quality Setup (GPU recommended)
```yaml
embedding:
  provider: ollama
  model: nomic-embed-text
  # Ollama endpoint is shared with the LLM — set base_url under llm: below,
  # or set the OLLAMA_HOST env var to override globally.
llm:
  provider: ollama
  model: qwen2.5:7b
  base_url: http://localhost:11434
  temperature: 0.7
  max_tokens: 2048
vector_store:
  provider: qdrant   # path derived automatically from store.base
rag:
  top_k: 10
  similarity_threshold: 0.4
  hybrid_search: true
chunk:
  size: 1000
  overlap: 200
rerank:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

### TurboQuantDB Setup

TurboQuantDB (`tqdb` on PyPI) is the **default vector store** — fastest ingest, smallest disk, no external service needed. It is bundled with `axon-rag` and requires no separate install step.

**Manual install (if needed):**
```bash
pip install tqdb
```

**config.yaml:**
```yaml
vector_store:
  provider: turboquantdb
  tqdb_bits: 4                  # quantization bits: 2 | 4 | 8 (default: 4)
  tqdb_fast_mode: false         # false = higher recall; true = faster queries
  tqdb_rerank: true             # true = internal ANN rerank pass (improves recall)
  # Optional tuning (omit to use TQDB defaults):
  # tqdb_rerank_precision: null # null | "f16" | "f32" — exact reranking uses more disk
  # tqdb_n_refinements: 5       # HNSW refinement passes; higher = better recall, slower build
  # tqdb_alpha: 1.2             # HNSW pruning aggressiveness
```

**Preset configurations** (benchmarked on 700–5 500 chunks, dim 384–768):

| Preset | bits | fast_mode | rerank | Recall@10 | p50 latency | Disk |
|--------|------|-----------|--------|-----------|-------------|------|
| b=4 Balanced *(default)* | 4 | false | true | 0.89 | 2.5 ms | 5.2 MB |
| b=8 High Quality | 8 | false | true | **0.99** | 3.4 ms | 9.2 MB |
| b=8 Fast Query | 8 | true | true | **0.99** | 1.8 ms | 9.2 MB |
| b=2 Minimal Disk | 2 | false | true | 0.67 | 2.1 ms | 3.2 MB |

> **b=4 Balanced** is the default — best recall/disk/RAM tradeoff for most workloads. Use b=8 when recall is critical, b=2 when disk is the constraint.

---

## 8. Configure .env

Copy the example file and set your values:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Ollama server location (default: localhost)
OLLAMA_HOST=http://localhost:11434
# Default LLM model name (must match what you pulled)
OLLAMA_MODEL=llama3.1
# Default embedding model for Ollama provider (if using ollama embeddings)
OLLAMA_EMBED_MODEL=nomic-embed-text
# API server settings
# Bind to localhost by default for local setups. For Docker/Compose deployments, set AXON_HOST=0.0.0.0 so the API is reachable from the host/other containers (only do this behind proper network/auth controls).
AXON_HOST=127.0.0.1
AXON_PORT=8000
# Vector store storage (Chroma-only — ignored for LanceDB and other providers)
# Where ChromaDB stores its data when `vector_store.provider: chroma`
CHROMA_DATA_PATH=./chroma_data
# Where BM25 index is stored
BM25_INDEX_PATH=./bm25_index
# Custom root directory for named projects (optional)
# Defaults to ~/.axon/projects — useful for shared drives or multiple workspaces
# AXON_PROJECTS_ROOT=/mnt/nas/axon-projects
# Log verbosity: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
# Security: restrict file ingestion to this directory
# RAG_INGEST_BASE=/home/user/documents
# Streamlit UI port
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

> **Security note:** Set `RAG_INGEST_BASE` to restrict which directories the `/ingest` endpoint can access. This prevents path traversal attacks.

---

## 9. Verify the Full Setup

### Step 1: Start the API server

```bash
axon-api
# Or with make:
make run-api
```

You should see:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Check health

```bash
curl http://localhost:8000/health
```

Expected:
```json
{"status": "ok", "project": "default"}
```

### Step 3: Ingest a test document

Create a test file:

```bash
# Linux / macOS
echo "The capital of France is Paris. Paris is known for the Eiffel Tower." > ./test_doc.txt
# Windows (PowerShell)
"The capital of France is Paris. Paris is known for the Eiffel Tower." | Out-File -FilePath .\test_doc.txt -Encoding utf8
```

Ingest it (replace the path with the absolute path to the file you just created):
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/absolute/path/to/test_doc.txt"}'
```

Expected:
```json
{"message": "Ingestion started for /tmp/test_doc.txt", "status": "processing"}
```

### Step 4: Run a test query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

Expected response includes the answer "Paris" synthesized from the ingested document.

### Step 5: (Optional) Start the Web UI

In a separate terminal:
```bash
axon-ui
# Or:
make run-ui
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 10. MCP Server Setup

`axon-mcp` is a standard MCP stdio server. It works with **any MCP-compatible AI tool** — not just VS Code. `axon-api` must be running before any MCP client connects.

### Supported clients

| Tool | Config file | Notes |
|---|---|---|
| **VS Code** (Copilot agent mode) | `.vscode/mcp.json` | Also needs `.vscode/settings.json` |
| **Claude Code** | `~/.claude/settings.json` or `.claude/settings.json` | Or use `claude mcp add` CLI command |
| **OpenAI Codex CLI** | `~/.codex/config.toml` | TOML format — differs from the JSON-based clients |
| **Google Gemini CLI** | `~/.gemini/settings.json` | Same `mcpServers` JSON shape as Claude Code |
| **Cursor** | `.cursor/mcp.json` | Same `mcpServers` JSON shape as Claude Code |
| **Claude Desktop** | `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows) | Same `mcpServers` JSON shape as Claude Code |
| **OpenAI Codex Desktop** | Settings → MCP Servers | Add `axon-mcp` as a new stdio connector |
| **Any other MCP stdio host** | varies | Use `command: axon-mcp` |

The two env vars are the same for every client:

| Variable | Default | Purpose |
|---|---|---|
| `RAG_API_BASE` | `http://localhost:8000` | Where `axon-api` is running |
| `RAG_API_KEY` | *(empty)* | API key — leave blank unless you enabled auth |

---

### VS Code (Copilot agent mode)

**1.** Create `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "axon": {
      "type": "stdio",
      "command": "axon-mcp",
      "args": [],
      "env": {
        "RAG_API_BASE": "http://localhost:8000",
        "RAG_API_KEY": ""
      }
    }
  }
}
```

**2.** Create `.vscode/settings.json`:

```json
{ "chat.mcp.access": "all" }
```

**3.** Start `axon-api`, then reload VS Code (`Ctrl+Shift+P` → "Reload Window"). Axon tools appear in the agent mode hammer menu.

**Platform notes — if `axon-mcp` is not found:**

| Platform | Use this command value |
|---|---|
| Windows (no PATH) | `C:\Users\<you>\AppData\Local\Programs\Python\Python313\Scripts\axon-mcp.exe` |
| Linux / macOS (venv) | `/home/<you>/Axon/venv/bin/axon-mcp` |
| WSL | Use `"command": "/home/<you>/Axon/venv/bin/python"` with `"args": ["-m", "axon.mcp_server"]` |

**Shared team setup:** run `axon-api` on a shared server, then set `RAG_API_BASE` to `http://<server-ip>:8000` in each user's config. The owner ingests; everyone else queries.

---

### Claude Code

```bash
# One-liner (adds axon to your global Claude Code MCP config)
claude mcp add axon axon-mcp --env RAG_API_BASE=http://localhost:8000
```

Or edit `~/.claude/settings.json` directly:

```json
{
  "mcpServers": {
    "axon": {
      "command": "axon-mcp",
      "env": { "RAG_API_BASE": "http://localhost:8000" }
    }
  }
}
```

For project-scoped config, add the same block to `.claude/settings.json` in your project root.

**Verify:** `claude mcp list` — `axon` should appear.

---

### Cursor

Create `.cursor/mcp.json` in your project root:

```json
{
  "mcpServers": {
    "axon": {
      "command": "axon-mcp",
      "env": { "RAG_API_BASE": "http://localhost:8000" }
    }
  }
}
```

Reload Cursor. Axon tools appear in the agent/composer panel.

---

### OpenAI Codex CLI

Codex uses TOML format. Add a block to `~/.codex/config.toml`:

```toml
[mcp_servers.axon]
command = "axon-mcp"
args = []
[mcp_servers.axon.env]
RAG_API_BASE = "http://localhost:8000"
```

Codex launches the server automatically when a session starts. If `axon-mcp` is not on PATH, replace `"axon-mcp"` with the full binary path (see platform notes under VS Code above).

**Verify:** in Codex, type `/tools` — `axon_*` tools should appear in the list.

---

### Google Gemini CLI

Add an `mcpServers` block to `~/.gemini/settings.json` (create the file if it does not exist):

```json
{
  "mcpServers": {
    "axon": {
      "command": "axon-mcp",
      "env": {
        "RAG_API_BASE": "http://localhost:8000"
      }
    }
  }
}
```

Start `axon-api`, then start Gemini CLI. Axon tools are available immediately in the session.

**Verify:** ask Gemini CLI to "list available tools" — `axon_*` entries should appear.

---

### Claude Desktop

Edit (or create) `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, or `%APPDATA%\Claude\claude_desktop_config.json` on Windows:

```json
{
  "mcpServers": {
    "axon": {
      "command": "axon-mcp",
      "env": {
        "RAG_API_BASE": "http://localhost:8000"
      }
    }
  }
}
```

Restart Claude Desktop. Axon tools appear in the tool picker automatically.

---

### OpenAI Codex Desktop

Open Codex Desktop → Settings → MCP Servers → Add new → choose **stdio** transport:

- **Command:** `axon-mcp`
- **Environment:** `RAG_API_BASE=http://localhost:8000` (add `RAG_API_KEY=<your-key>` if you set one in config.yaml)

Save and restart. All Axon tools will be listed in the agent tool picker.

---

### Any other MCP stdio host

Any tool that accepts `command` + `env` works with the same pattern. If `axon-mcp` is not on PATH, use the full path to the binary (see platform notes above) or fall back to:

```
command: python
args: ["-m", "axon.mcp_server"]
```

---

### Verify the server starts

```bash
printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}\n' | axon-mcp
```

Expected: `{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",...}}`

---

### Available MCP tools

For the full list of all 30 tools with parameter tables, see [MCP_TOOLS.md](MCP_TOOLS.md).

> **Tip:** use `search_knowledge` (not `query_knowledge`) in agent mode — the agent's own LLM synthesises the answer from raw chunks, so no Ollama is required.

---

## 11. VS Code Extension (GitHub Copilot Integration)

The Axon VS Code extension (`axon-copilot`) gives GitHub Copilot direct access to your knowledge base as **language model tools** — available in Copilot Chat, inline chat, and agent mode. It also provides VS Code commands for project management, ingestion, and sharing.

> **This is the recommended VS Code integration.** The VS Code extension runs entirely inside VS Code using the VS Code Language Model API. It is the primary interface for most VS Code users — tools appear automatically in Copilot without any extra configuration file.
>
> The MCP server (Section 10) is an alternative for Copilot agent mode with explicit `.vscode/mcp.json` wiring, or for non-VS Code MCP hosts. Both require `axon-api` to be running. Advanced users can run both; for most users the extension alone is sufficient.

### Prerequisites

- **VS Code** 1.93 or later
- **GitHub Copilot** extension installed and signed in (free tier works; Copilot Chat required for tool use)
- **Axon API** running (`axon-api`) — the extension connects to it at `http://localhost:8000` by default

### 1. Install the VSIX

**Option A — Install from release (recommended)**

Download `axon-copilot-<version>.vsix` from the [GitHub Releases](https://github.com/jyunming/Axon/releases) page, or install directly after `pip install axon-rag` using:

```bash
axon-ext
```

To install manually from a VSIX file:

```
1. Open the Extensions panel (Ctrl+Shift+X)
2. Click the "..." (More Actions) button at the top-right of the panel
3. Select "Install from VSIX..."
4. Navigate to axon-copilot-<version>.vsix and click Install
5. Reload VS Code when prompted (Ctrl+Shift+P → "Reload Window")
```

**Option B — Build from source**

```bash
cd integrations/vscode-axon
npm install
npm run package   # produces axon-copilot-<version>.vsix
```

Then install the generated `.vsix` as above.

### 2. How the extension finds Python

The extension needs to know which Python executable to use to start `axon-api` (when `autoStart` is enabled). It tries the following in order — **no configuration is usually needed**:

| Priority | Where it looks | When this applies |
|---|---|---|
| 1 | `axon.pythonPath` VS Code setting | You explicitly set a path — always wins |
| 2 | `~/.axon/.python_path` | Written automatically the first time you run `axon` in a terminal; covers pip/venv installs |
| 3 | pipx venv (`~/.local/pipx/venvs/axon/`) | You installed with `pipx install axon` |
| 4 | Workspace `.venv` / `venv` / `env` | A virtual environment exists in your open VS Code folder |
| 5 | System `python3` / `python` | Global install fallback; shows a warning if Axon is not found on this Python |

**The simplest path:**

- **Installed via pip into a venv?** Run `axon` once from that terminal → `~/.axon/.python_path` is written → done. No VS Code config needed.
- **Installed via pipx?** Nothing to do — the extension finds the pipx venv automatically.
- **Custom install location?** Set `axon.pythonPath` explicitly in VS Code Settings (Ctrl+,).

If auto-detection fails, VS Code shows a notification with a link to open the `axon.pythonPath` setting directly.

### 3. Configure extension settings

Open VS Code Settings (Ctrl+,) and search for `axon`:

| Setting | Default | Description |
|---|---|---|
| `axon.apiBase` | `http://localhost:8000` | URL of your running `axon-api` server |
| `axon.apiKey` | *(empty)* | API key if `RAG_API_KEY` is set on the server |
| `axon.topK` | `5` | Default number of chunks returned per query |
| `axon.autoStart` | `true` | Auto-start `axon-api` on extension activate (Linux/macOS only) |
| `axon.pythonPath` | *(auto-detect)* | Explicit Python path override — leave blank for auto-detection (see above) |
| `axon.useCopilotLlm` | `false` | Switch the extension's active LLM provider to Copilot for all inference (query answering, RAPTOR summarization, etc.), not just a helper |
| `axon.ingestBase` | *(empty)* | Restrict ingestion to a specific directory. Empty = filesystem root on macOS/Linux, current drive root on Windows auto-start. Cross-drive ingest on Windows requires setting this explicitly or starting `axon-api` manually with a broader `RAG_INGEST_BASE`. |
| `axon.storeBase` | *(empty)* | Base path for AxonStore multi-user mode |
| `axon.graphSynthesis` | `true` | When showing the knowledge/code graph, also call `POST /query` to synthesise a text answer. Disable if your LLM is slow or unavailable — the graph still renders without it. |

Or edit `settings.json` directly:

```json
{
  "axon.apiBase": "http://localhost:8000",
  "axon.topK": 10,
  "axon.autoStart": true
}
```

### 4. Start the Axon API

The API must be running before using any Copilot tools. On Linux/macOS with `autoStart: true` the extension starts it automatically. On Windows, start it manually:

```bash
axon-api
```

### 5. Verify in Copilot Chat

Open Copilot Chat (Ctrl+Shift+I or the chat icon in the Activity Bar). Type:

```
@axon what files have I ingested?
```

Copilot will call `list_knowledge` or `list_projects` automatically. You can also ask in plain language:

```
@axon search for information about neural networks
```

### Available tools (32 total)

| Tool | What it does |
|---|---|
| `search_knowledge` | Raw chunk retrieval — best for discovery, letting Copilot synthesise the answer. If `threshold` filters out all results, automatically falls back to top-N candidates with a note. |
| `query_knowledge` | Retrieval + answer via the configured LLM provider |
| `ingest_text` | Ingest a text snippet directly |
| `ingest_texts` | Ingest multiple text snippets in one call |
| `ingest_url` | Fetch and ingest a web page |
| `ingest_path` | Ingest a local file or directory (async; returns `job_id`) |
| `get_job_status` | Poll an ingest job by `job_id` — call after `ingest_path` before searching |
| `list_projects` | List available projects |
| `switch_project` | Switch active project |
| `create_project` | Create a new project |
| `delete_project` | Delete a project and all its data |
| `delete_documents` | Remove specific documents by ID |
| `list_knowledge` | List all ingested files with chunk counts |
| `clear_knowledge` | Wipe all data from the current project |
| `update_settings` | Adjust RAG settings (top_k, rerank, hyde, etc.) |
| `get_current_settings` | Read the current active Axon configuration (retrieval flags, RAG mode, LLM provider) |
| `share_project` | Generate a share key for a project |
| `redeem_share` | Mount a project shared by another user |
| `revoke_share` | Revoke an active share |
| `list_shares` | List outgoing and incoming project shares |
| `init_store` | Initialise AxonStore multi-user mode at a given base directory |
| `get_store_status` | Check whether the AxonStore is initialised and return its metadata |
| `ingest_image` | Describe an image via Copilot vision model and ingest the description. Accepts optional `alt_text` param to provide a description directly (enables headless/offline use without Copilot vision). |
| `refresh_ingest` | Re-ingest files whose content has changed since last ingest |
| `get_stale_docs` | Find documents not re-ingested within N days |
| `graph_status` | Show entity count, community summary count, and graph readiness |
| `show_graph` | Open the Axon Graph Panel for a query — shows answer, citations, and 3D entity/code graph side by side |
| `graph_finalize` | Rebuild community summaries and finalize the knowledge graph for global-mode queries |
| `graph_data` | Return raw graph payload (nodes + links) for the active project |
| `list_sessions` | List active REPL/API sessions |
| `get_session` | Get details for a specific session |
| `get_active_leases` | List active write-lease counts per project |

### Available VS Code commands

Access via Ctrl+Shift+P:

| Command | Description |
|---|---|
| `Axon: Switch Project` | Change active project |
| `Axon: Create Project` | Create a new project |
| `Axon: Ingest File` | Browse and ingest a single file |
| `Axon: Ingest Workspace` | Ingest the entire VS Code workspace |
| `Axon: Ingest Folder` | Browse and ingest a folder |
| `Axon: Start Server` | Start `axon-api` manually |
| `Axon: Stop Server` | Stop the managed `axon-api` process |
| `Axon: Init Store` | Change the store base path (e.g. to a shared drive) |
| `Axon: Share Project` | Generate a share key for a project |
| `Axon: Redeem Share` | Join a project shared by another user |
| `Axon: Revoke Share` | Revoke an active share |
| `Axon: List Shares` | View all active shares |
| `Axon: Show Graph for Query…` | Open the Graph Panel — prompts for a query, then shows answer + citations + 3D graph |
| `Axon: Show Graph for Selection` | Open the Graph Panel using the current editor selection as the query |

### Graph Panel

The Graph Panel opens a **split webview** directly inside VS Code — no external browser required.

```
┌──────────────────────┬──────────────────────────────────────┐
│  Q: <your query>     │  [ Knowledge Graph ]  [ Code Graph ] │
│  ────────────────    │                                      │
│  LLM-synthesised     │         ●─────◆─────●               │
│  answer with         │        /   3D force   \              │
│  inline citations    │       ▼    graph       ▼             │
│  ────────────────    │      ●                  ●            │
│  [1] file.py:42  ▸   │                                      │
│  [2] module.py   ▸   │  click node/citation → open file    │
└──────────────────────┴──────────────────────────────────────┘
```

**Knowledge Graph tab** — entity–relation graph extracted from **any document** (PDF, DOCX, Markdown, HTML, etc.) during ingest. Nodes are named entities (people, concepts, components); edges are extracted relations. Requires `graph_rag: true` in `config.yaml` (disabled in the shipped config by default — enable it once your corpus is ready).

**Code Graph tab** — structural file/class/function graph. Nodes are files, classes, and functions; edges are `IMPORTS`, `CONTAINS`, and `MENTIONED_IN` relationships (at query time, prose chunks mentioning code symbols are bridged via `MENTIONED_IN` edges). Requires `code_graph: true` in `config.yaml` (opt-in, off by default):

```yaml
# config.yaml
rag:
  code_graph: true        # build File + Symbol nodes with CONTAINS/IMPORTS edges
  code_graph_bridge: true # also link prose chunks that mention code symbols
```

> **CLI flags also available:** `--code-graph` / `--no-code-graph` toggle `code_graph` at runtime. `--code-graph-bridge` / `--no-code-graph-bridge` toggles the bridge. `--graph-rag` / `--no-graph-rag` controls the knowledge graph. Config file and CLI flags are both supported.

**Ingest before opening the panel:**

```bash
# Knowledge graph (prose documents)
axon --ingest ./docs/ --graph-rag
# Code graph (source code) — set code_graph: true in config.yaml first
axon --ingest ./src/
# Both at once — set code_graph: true in config.yaml, then:
axon --ingest ./project/ --graph-rag
```

**Open the panel:**

```
Ctrl+Shift+P → Axon: Show Graph for Query…
Ctrl+Shift+P → Axon: Show Graph for Selection   (select text first)
Copilot Chat:
  @axon show me the graph for how retrieval works
  @axon visualise the authentication module
```

Tabs that have no data are automatically disabled with a tooltip explaining which flag to enable. Clicking any citation or graph node opens the source file at the exact line.

### Typical workflow

```
1. axon-api starts (auto or manual)
2. Copilot Chat: "Ingest my documents at /path/to/docs"
   → extension calls ingest_path → returns job_id
3. Copilot Chat: "Check if ingest is done" (or wait a moment)
   → extension calls get_job_status(job_id) → "completed"
4. Copilot Chat: "What are the main topics in those docs?"
   → extension calls search_knowledge → Copilot synthesises answer
```

### `@axon` — Conversational Shortcut

The Axon extension also registers a `@axon` chat participant. It is a **conversational shortcut** over the same `axon_*` toolset — not a separate integration path.

```
@axon What are the main API changes in the last sprint?
```

`@axon` posts your prompt to `/query` on the backend, the same path as `query_knowledge`. The result is identical. Use it when you prefer a conversational style; use tools directly when you want explicit tool invocations (e.g. in agent mode or pipelines). There is no behavioral difference.

### Troubleshooting the extension

**Copilot says it cannot find Axon tools**
- Ensure the VSIX is installed and VS Code has been reloaded after install
- Check the extension is enabled (Extensions panel → search "Axon Copilot")
- Confirm `axon-api` is running: `curl http://localhost:8000/health`

**Tools appear but requests fail with connection errors**
- Verify `axon.apiBase` matches where `axon-api` is listening
- On Windows, `axon-api` may bind to `127.0.0.1` — make sure `axon.apiBase` uses `http://localhost:8000` (not `http://0.0.0.0:8000`)

**`autoStart` does not work (Linux/macOS)**
- Check `axon.pythonPath` is set correctly, or run `axon` once from the terminal to write `~/.axon/.python_path`
- The extension reads that file to discover Python automatically

**Image ingest fails with "model does not support images"**
- Switch your Copilot model to one with vision capability (GPT-4o or Claude 3.x Sonnet/Opus)
- In Copilot Chat, click the model selector and choose a multimodal model

---

## 11b. REPL Security — Shell Passthrough

The REPL supports running shell commands directly with the `!` prefix (e.g. `! ls docs/`). Access is controlled by the `repl.shell_passthrough` config field:

| Policy | Behaviour |
|--------|-----------|
| `local_only` | **(default)** Commands are blocked when the active project is a read-only mount or a remote scope; allowed for local and default projects. |
| `always` | Shell passthrough is always permitted regardless of project scope. |
| `off` | Shell passthrough is disabled entirely. |

**Set via `config.yaml`:**
```yaml
repl:
  shell_passthrough: local_only   # local_only | always | off
```

This option is currently configurable only via `config.yaml`; environment variable overrides are not supported.

> **Security note:** `always` permits unrestricted shell access. Use `off` in shared or multi-tenant deployments. The default `local_only` blocks shell commands when the active project is a read-only share mount.

---

## 12. Upgrading Axon

**From an editable install (`pip install -e .`):**

```bash
cd Axon          # the folder you cloned into
git pull         # fetch the latest code
pip install -e . # re-install to pick up any new dependencies
```

> `git pull` updates the source files. `pip install -e .` is needed if `pyproject.toml` changed (new or removed dependencies). It is safe to run both every time.

**Check for breaking changes:** read `CHANGELOG.md` (if present) or the GitHub release notes before upgrading. In particular, if the embedding model default changed you will need to delete your vector store and re-ingest (see [TROUBLESHOOTING.md — ChromaDB InvalidDimensionException](TROUBLESHOOTING.md)).

**Verify the upgrade:**
```bash
axon --version
```

---

## 13. Troubleshooting

For common errors and step-by-step fixes, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

---

## Quick Reference

| What | Command |
|------|---------|
| Start Ollama | `ollama serve` |
| Pull LLM | `ollama pull llama3.1:8b` |
| Pull embeddings (Ollama) | `ollama pull nomic-embed-text` |
| Pull vision model | `ollama pull llava` |
| List pulled models | `ollama list` |
| Start API | `axon-api` or `make run-api` |
| Start UI | `axon-ui` or `make run-ui` |
| Health check | `curl http://localhost:8000/health` |
| Ingest a file | `axon --ingest ./path/to/file` |
| Run a query | `axon "your question"` |

---

*For model comparisons and hardware recommendations, see [MODEL_GUIDE.md](MODEL_GUIDE.md).*
