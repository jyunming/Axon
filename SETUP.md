# Setup Guide — Axon

This guide walks through setting up the application from scratch, including detailed steps for configuring every LLM and embedding model option.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Install the Package](#2-install-the-package)
3. [Install and Start Ollama](#3-install-and-start-ollama)
4. [LLM Setup](#4-llm-setup)
5. [Embedding Model Setup](#5-embedding-model-setup)
6. [Vision / Multimodal Setup (Optional)](#6-vision--multimodal-setup-optional)
7. [Configure config.yaml](#7-configure-configyaml)
8. [Configure .env](#8-configure-env)
9. [Verify the Full Setup](#9-verify-the-full-setup)
10. [MCP Setup (Copilot Agent Mode)](#10-mcp-setup-copilot-agent-mode)
11. [VS Code Extension (GitHub Copilot Integration)](#11-vs-code-extension-github-copilot-integration)
12. [Troubleshooting](#12-troubleshooting)

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
# Minimal install (sentence-transformers + ChromaDB + Ollama LLM)
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
> The default `config.yaml` ships with `graph_rag_community_backend: leidenalg`, which skips `graspologic` entirely and uses the documented Python 3.13 path directly.
> To restore the legacy preference order (graspologic → leidenalg → networkx Louvain) set `graph_rag_community_backend: auto`.

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
  base_url: http://localhost:11434
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

| Model | Dims | Context | Quality |
|-------|------|---------|---------|
| `BAAI/bge-small-en-v1.5` | 384 | 512 tokens | Good |
| `BAAI/bge-base-en-v1.5` | 768 | 512 tokens | Better |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 tokens | Best |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 tokens | Good |

**Verify:**
```python
from fastembed import TextEmbedding
model = TextEmbedding("BAAI/bge-small-en-v1.5")
embeddings = list(model.embed(["test sentence"]))
print(f"Embedding dimension: {len(embeddings[0])}")  # Should print 384
```

---

## 6. Vision / Multimodal Setup (Optional)

Only needed if you want to ingest image files. The system uses a Vision-Language Model (VLM) to auto-caption images before indexing.

**Supported image formats:** `.bmp`, `.png`, `.tif`, `.tiff`, `.pgm`

All image formats are normalised to PNG via Pillow before being sent to the VLM.

```bash
ollama pull llava
```

> Note: LLaVA requires ~5 GB disk and ~8 GB RAM/VRAM.

No additional config changes are needed — the system automatically uses `llava` for image captioning when it encounters any supported image file during ingestion.

---

## 7. Configure config.yaml

Copy the template and customize it:

```bash
# config.yaml already exists in the repo root — edit it directly
# Or copy it as a personal override
cp config.yaml config.local.yaml  # pass --config config.local.yaml to the CLI
```

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
  provider: chroma
  path: ./chroma_data

bm25:
  path: ./bm25_index

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
  provider: chroma
  path: ./chroma_data

bm25:
  path: ./bm25_index

rag:
  top_k: 10
  similarity_threshold: 0.3
  hybrid_search: true

  # Graph-augmented retrieval expansion (inspired by GraphRAG; not the full
  # Microsoft GraphRAG method — no community detection or global search).
  # Extracts entities and optionally relation triples from each chunk at ingest
  # time; at query time, entity-matched and 1-hop-related chunks are injected as
  # extra context slots beyond the normal top_k.
  # Requires a capable LLM at ingest; adds per-chunk latency.
  # Limits: shallow graph, heuristic scoring, no entity canonicalisation.
  # graph_rag: false
  # graph_rag_budget: 3       # extra slots beyond top_k (0 = no guarantee)
  # graph_rag_relations: true # enable 1-hop relation traversal

chunk:
  size: 1000
  overlap: 200

rerank:
  enabled: false
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
```

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
  provider: qdrant
  path: ./qdrant_data

bm25:
  path: ./bm25_index

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

# Where ChromaDB stores its data
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
echo "The capital of France is Paris. Paris is known for the Eiffel Tower." > /tmp/test_doc.txt
```

Ingest it:
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/tmp/test_doc.txt"}'
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

## 10. MCP Setup (Copilot Agent Mode)

The MCP (Model Context Protocol) server lets GitHub Copilot in **agent mode**
call Axon tools directly — ingesting documents, searching knowledge, switching
projects — all without leaving VS Code.

### How it works

```
VS Code (Copilot agent mode)
  → spawns axon-mcp process (stdio, one per VS Code instance)
  → axon-mcp proxies calls to the Axon REST API (http://localhost:8000)
  → Axon handles retrieval; Copilot's LLM synthesises answers
```

The `axon-mcp` entry point is installed automatically when you
`pip install -e .`. The REST API (`axon-api`) must be running before Copilot
can use any MCP tools.

### 1. Create `.vscode/mcp.json` in your workspace

This file is gitignored — create it locally:

```jsonc
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

> **Windows users:** if `axon-mcp` is not found, use the full path to the
> entry point script installed in your Python environment's `Scripts/` folder,
> e.g. `C:\Users\<you>\AppData\Local\Programs\Python\Python313\Scripts\axon-mcp.exe`.

> **WSL users:** VS Code WSL Remote spawns the process inside WSL. Use the
> WSL-style path to the Windows Python executable, e.g.
> `/mnt/c/Users/<you>/AppData/Local/Programs/Python/Python313/python.exe`,
> with `args: ["-m", "axon.mcp_server"]`, instead of `axon-mcp`.

> **Linux (venv) users:** VS Code spawns `axon-mcp` without activating your
> virtualenv, so `"command": "axon-mcp"` will not be found. Use the full path
> to the binary inside your venv instead:
>
> **Option A — Private venv (per user)**
> Each user has their own venv. Use the absolute path to that venv's binary:
> ```jsonc
> {
>   "servers": {
>     "axon": {
>       "type": "stdio",
>       "command": "/home/<you>/Axon/venv/bin/axon-mcp",
>       "args": [],
>       "env": {
>         "RAG_API_BASE": "http://localhost:8000",
>         "RAG_API_KEY": ""
>       }
>     }
>   }
> }
> ```
> Replace `/home/<you>/Axon/venv` with the actual path to your venv.
> Alternatively use Python directly:
> `"command": "/home/<you>/Axon/venv/bin/python", "args": ["-m", "axon.mcp_server"]`
>
> **Option B — Shared venv (multi-user, one install)**
> An admin creates a venv in a shared location and sets read+execute permissions
> for all users. Each user's `.vscode/mcp.json` then points to the same binary.
>
> Admin sets it up once:
> ```bash
> python -m venv /opt/axon/venv
> /opt/axon/venv/bin/pip install -e /path/to/Axon
> chmod -R 755 /opt/axon/venv   # owner writes; everyone else reads + executes
> ```
>
> Each user's `.vscode/mcp.json`:
> ```jsonc
> {
>   "servers": {
>     "axon": {
>       "type": "stdio",
>       "command": "/opt/axon/venv/bin/axon-mcp",
>       "args": [],
>       "env": {
>         "RAG_API_BASE": "http://localhost:8000",
>         "RAG_API_KEY": ""
>       }
>     }
>   }
> }
> ```
> Each user still spawns their own lightweight `axon-mcp` proxy process;
> only the binary is shared. Only the admin should run `pip install` to update.

### 2. Enable MCP in VS Code workspace settings

Create `.vscode/settings.json` (also gitignored):

```json
{
  "chat.mcp.access": "all"
}
```

### 3. Start the Axon API

```bash
axon-api
# or: make run-api
```

### 4. Reload VS Code

Ctrl+Shift+P → **"Reload Window"**. The Axon server will appear in the
Copilot agent tools list (hammer icon in the chat input).

### Verify the MCP server starts

```bash
printf '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1"}}}\n' \
  | axon-mcp
```

Expected response:
```json
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{...},"serverInfo":{"name":"axon","version":"..."}}}
```

### Shared team knowledge base

For a team where one person owns ingestion and others query:

1. Run `axon-api` on a machine everyone can reach (e.g. Docker, internal server).
2. Each user creates their own `.vscode/mcp.json` pointing `RAG_API_BASE` at
   the shared server:
   ```jsonc
   {
     "servers": {
       "axon": {
         "type": "stdio",
         "command": "axon-mcp",
         "args": [],
         "env": {
           "RAG_API_BASE": "http://<server-ip>:8000",
           "RAG_API_KEY": ""
         }
       }
     }
   }
   ```
3. Each user's VS Code spawns its own lightweight `axon-mcp` proxy; the shared
   API handles all retrieval. The owner ingests; everyone else queries.

### Available MCP tools (23 total)

| Tool | Purpose |
|---|---|
| `ingest_text` | Ingest a single document |
| `ingest_texts` | Batch ingest (preferred) |
| `ingest_url` | Fetch a URL and ingest its content |
| `ingest_path` | Ingest a local file or directory (async) |
| `get_job_status` | Poll an async ingest job |
| `search_knowledge` | Raw chunk retrieval (Copilot synthesises) |
| `query_knowledge` | Retrieval + answer via local Ollama LLM |
| `list_knowledge` | List indexed sources with chunk counts |
| `switch_project` | Change active project namespace |
| `delete_documents` | Remove documents by ID |
| `list_projects` | List all project namespaces |
| `get_stale_docs` | Find docs not re-ingested within N days |
| `create_project` | Create a new project namespace |
| `delete_project` | Delete a project and all its data |
| `clear_knowledge` | Clear all vectors and BM25 index |
| `get_current_settings` | Return current RAG configuration |
| `update_settings` | Update RAG flags (top_k, rerank, hyde, etc.) |
| `list_sessions` | List saved conversation sessions |
| `get_session` | Retrieve a saved conversation session |
| `share_project` | Generate a share key for a project |
| `redeem_share` | Redeem a share key to mount a shared project |
| `list_shares` | List active outbound shares |
| `init_store` | Initialise AxonStore multi-user mode |

> **Recommended:** use `search_knowledge` (not `query_knowledge`) in agent mode.
> Copilot's LLM synthesises the answer from raw chunks — no Ollama required,
> no concurrency limit.

---

## 11. VS Code Extension (GitHub Copilot Integration)

The Axon VS Code extension (`axon-copilot`) gives GitHub Copilot direct access to your knowledge base as **language model tools** — available in Copilot Chat, inline chat, and agent mode. It also provides VS Code commands for project management, ingestion, and sharing.

> **Difference from MCP:** The VS Code extension runs entirely inside VS Code using the VS Code Language Model API. The MCP server (Section 10) uses the Model Context Protocol and works with any MCP-compatible host. Both require `axon-api` to be running. Choose based on your workflow — both can coexist.

### Prerequisites

- **VS Code** 1.93 or later
- **GitHub Copilot** extension installed and signed in (free tier works; Copilot Chat required for tool use)
- **Axon API** running (`axon-api`) — the extension connects to it at `http://localhost:8000` by default

### 1. Install the VSIX

**Option A — Install from release (recommended)**

Download `axon-copilot-1.0.0.vsix` from the [GitHub Releases](https://github.com/jyunming/Axon/releases) page or find it in `integrations/vscode-axon/` in the repo.

Open VS Code and install:

```
1. Open the Extensions panel (Ctrl+Shift+X)
2. Click the "..." (More Actions) button at the top-right of the panel
3. Select "Install from VSIX..."
4. Navigate to axon-copilot-1.0.0.vsix and click Install
5. Reload VS Code when prompted (Ctrl+Shift+P → "Reload Window")
```

**Option B — Build from source**

```bash
cd integrations/vscode-axon
npm install
npm run package   # produces axon-copilot-1.0.0.vsix
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
| `axon.useCopilotLlm` | `false` | Use Copilot's LLM (GPT-4o / Claude) for RAPTOR/GraphRAG instead of Ollama |
| `axon.ingestBase` | *(empty)* | Restrict ingestion paths to a specific directory |
| `axon.storeBase` | *(empty)* | Base path for AxonStore multi-user mode |

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

Open Copilot Chat (Ctrl+Shift+I or the chat icon in the Activity Bar). Type a message:

```
@workspace What does my knowledge base contain?
```

Copilot will automatically call `axon_getCollection` or `axon_listProjects` to answer. You can also use tools explicitly:

```
Search my Axon knowledge base for information about neural networks.
```

### Available tools (17 total)

| Tool | What it does |
|---|---|
| `axon_searchKnowledge` | Raw chunk retrieval — best for discovery, letting Copilot synthesise the answer |
| `axon_queryKnowledge` | Retrieval + answer via local LLM (requires Ollama) |
| `axon_ingestText` | Ingest a text snippet directly |
| `axon_ingestUrl` | Fetch and ingest a web page |
| `axon_ingestPath` | Ingest a local file or directory (async; returns `job_id`) |
| `axon_getIngestStatus` | Poll an ingest job by `job_id` — call after `axon_ingestPath` before searching |
| `axon_listProjects` | List available project namespaces |
| `axon_switchProject` | Switch active project |
| `axon_createProject` | Create a new project namespace |
| `axon_deleteProject` | Delete a project and all its data |
| `axon_deleteDocuments` | Remove specific documents by ID |
| `axon_getCollection` | List all ingested files with chunk counts |
| `axon_clearCollection` | Wipe all data from the current project |
| `axon_updateSettings` | Adjust RAG settings (top_k, rerank, hyde, etc.) |
| `axon_listShares` | List active project shares (AxonStore mode) |
| `axon_initStore` | Initialise AxonStore multi-user mode |
| `axon_ingestImage` | Describe an image via Copilot vision model and ingest the description |

### Available VS Code commands

Access via Ctrl+Shift+P:

| Command | Description |
|---|---|
| `Axon: Switch Project` | Change active project namespace |
| `Axon: Create Project` | Create a new project |
| `Axon: Ingest File` | Browse and ingest a single file |
| `Axon: Ingest Workspace` | Ingest the entire VS Code workspace |
| `Axon: Ingest Folder` | Browse and ingest a folder |
| `Axon: Start Server` | Start `axon-api` manually |
| `Axon: Stop Server` | Stop the managed `axon-api` process |
| `Axon: Init Store` | Initialise AxonStore multi-user mode |
| `Axon: Share Project` | Generate a share key for a project |
| `Axon: Redeem Share` | Join a project shared by another user |
| `Axon: Revoke Share` | Revoke an active share |
| `Axon: List Shares` | View all active shares |

### Typical workflow

```
1. axon-api starts (auto or manual)
2. Copilot Chat: "Ingest my documents at /path/to/docs"
   → extension calls axon_ingestPath → returns job_id
3. Copilot Chat: "Check if ingest is done" (or wait a moment)
   → extension calls axon_getIngestStatus(job_id) → "completed"
4. Copilot Chat: "What are the main topics in those docs?"
   → extension calls axon_searchKnowledge → Copilot synthesises answer
```

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

## 12. Troubleshooting

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
*For SOTA gap analysis and roadmap, see [SOTA_ANALYSIS.md](SOTA_ANALYSIS.md).*
