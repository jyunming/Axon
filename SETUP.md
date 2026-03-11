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
10. [Troubleshooting](#10-troubleshooting)

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
git clone https://github.com/jyunming/studio_brain_open.git
cd studio_brain_open
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
  similarity_threshold: 0.5
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
  similarity_threshold: 0.5
  hybrid_search: true

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
  model: nomic-embed-text  # Ollama uses the OLLAMA_HOST env var, not base_url

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
{"status": "healthy", "brain_ready": true}
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

## 10. Troubleshooting

### Ollama not running

**Symptom:** `ConnectionRefusedError` or `httpx.ConnectError` when starting the API.

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it:
ollama serve
```

---

### Model not found

**Symptom:** `model 'llama3.1' not found` error.

```bash
# List pulled models
ollama list

# Pull the model
ollama pull llama3.1:8b

# Make sure config.yaml model name matches exactly
# e.g., use "llama3.1:8b" not "llama3.1"
```

---

### Port already in use

**Symptom:** `Address already in use` on port 8000 or 8501.

```bash
# Find what's using the port
lsof -i :8000       # Linux / macOS
netstat -ano | findstr :8000  # Windows

# Change the port in .env
AXON_PORT=8001
```

---

### ChromaDB errors / corrupted index

**Symptom:** `chromadb.errors.InvalidCollectionException` or similar.

```bash
# Clear and rebuild the index (you will need to re-ingest documents)
rm -rf chroma_data/
rm -rf bm25_index/
```

---

### Embedding shape mismatch

**Symptom:** Error about vector dimension mismatch after changing the embedding model.

This happens when you change embedding providers/models after already ingesting documents. The existing vectors in ChromaDB have different dimensions than what the new model produces.

**Fix:** Clear the vector store and re-ingest:
```bash
rm -rf chroma_data/
rm -rf bm25_index/
# Then re-ingest your documents
axon --ingest ./your_documents/
```

---

### sentence-transformers model download fails

**Symptom:** Timeout or SSL error when downloading `all-MiniLM-L6-v2`.

```bash
# Pre-download with explicit cache dir
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Downloaded successfully')
"

# If behind a proxy, set environment variables:
export HTTPS_PROXY=http://your-proxy:port
export HTTP_PROXY=http://your-proxy:port
```

---

### FastEmbed import error

**Symptom:** `ModuleNotFoundError: No module named 'fastembed'`

```bash
pip install fastembed
# Or reinstall with extras:
pip install -e ".[fastembed]"
```

---

### Memory error / OOM when running LLM

**Symptom:** Process killed, OOM error, or system becomes unresponsive.

Options:
1. Use a smaller LLM (e.g., `phi3:mini` instead of `llama3.1:8b`)
2. Reduce `max_tokens` in `config.yaml`
3. Enable GPU if available (Ollama detects it automatically)
4. Close other applications to free RAM

---

### Docker setup

If using Docker Compose, Ollama runs as a separate service. The `docker-compose.yml` handles networking automatically:

```bash
docker-compose up --build
```

Check that all services are healthy:
```bash
docker-compose ps
```

If the API service can't reach Ollama, check the `OLLAMA_HOST` environment variable in `docker-compose.yml` — it should reference the Ollama service name, not `localhost`.

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
