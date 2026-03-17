# Quick Reference Guide

## Common Commands

### Installation
```bash
# Basic installation
pip install -e .

# With development tools
pip install -e ".[dev]"

# With optional features
pip install -e ".[qdrant,fastembed]"
pip install -e ".[all]"
```

### Development
```bash
# Format code
make format
black src/ tests/

# Lint code
make lint
ruff check src/ tests/

# Type check
make type-check
mypy src/axon/

# Run all checks
make all
```

### Testing
```bash
# Run all tests
make test
pytest

# With coverage
make test-cov
pytest --cov=axon --cov-report=html

# Specific test file
pytest tests/test_loaders.py

# Specific test
pytest tests/test_loaders.py::TestTextLoader::test_load_text_file

# With markers
pytest -m "not slow"
pytest -v -s  # Verbose with output
```

### Running Services
```bash
# API Server
make run-api
axon-api

# Streamlit UI
make run-ui
axon-ui

# CLI — interactive REPL (default when no args)
axon

# CLI — single-shot query
axon "What is RAG?"

# CLI — stream response token-by-token
axon --stream "Summarise my documents"

# CLI — ingest a directory
axon --ingest ./documents/

# CLI — list all ingested documents
axon --list

# CLI — switch model at runtime (auto-pulls Ollama model if missing)
axon --model gemma:2b "Your question"

# CLI — use a cloud provider
axon --provider gemini --model gemini-1.5-flash "Your question"
axon --provider openai --model gpt-4o "Your question"

# CLI — pull a model explicitly
axon --pull gemma:2b

# CLI — see all providers and locally available models
axon --list-models

# CLI — advanced RAG flags (can be combined)
axon --cite "Summarise my documents"        # inline source citations
axon --decompose "Complex multi-part query" # split into sub-questions
axon --compress "Your question"             # compress retrieved context
axon --raptor --ingest ./docs/              # hierarchical indexing on ingest
axon --graph-rag "Your question"            # entity-graph retrieval expansion

# CLI — project management
axon --project myproject "Your question"    # use a named project
axon --project-new myproject                # create project + ingest
axon --project-list                         # list all projects
axon --project-delete myproject             # delete a project
```

**@file / @folder context (REPL only):**

Attach file or directory contents inline to any query — Axon reads and embeds them before answering:
```
You: Explain this code @./src/axon/main.py
You: What changed in @./src/axon/
You: Compare @report.pdf with @notes.docx
```
Supported: `.txt`, `.md`, `.py`, `.json`, `.csv`, `.html`, `.docx`, `.pdf`, images (`.png`, `.bmp`, `.tif`, `.pgm`)

**REPL slash commands (interactive mode):**

| Command | Purpose |
|---------|---------|
| `/help [cmd]` | Show all commands or detailed help (try: `/help model`, `/help embed`, `/help ingest`, `/help rag`, `/help sessions`) |
| `/list` | List all ingested documents with chunk counts |
| `/ingest <path\|glob>` | Ingest a file or directory (supports glob patterns) |
| `/model [provider/model]` | Switch LLM provider and model on the fly; bare `/model <name>` auto-detects provider (`gemini-*`→gemini, `gpt-*`→openai, else→ollama); auto-pulls from Ollama if needed |
| `/embed [provider/model]` | Switch embedding provider and model |
| `/pull <name>` | Pull an Ollama model with progress indicator |
| `/vllm-url [url]` | Show or set the vLLM server base URL at runtime (e.g. `http://localhost:8000/v1`) |
| `/search` | Toggle Brave web search fallback (truth_grounding) |
| `/discuss` | Toggle discussion_fallback mode (allow general knowledge answers when no documents match) |
| `/rag [option]` | Show or modify RAG settings — try `/rag` with: `topk <n>`, `threshold <0-1>`, `hybrid`, `rerank`, `rerank-model <model>`, `hyde`, `multi`, `step-back`, `decompose`, `compress`, `cite`, `raptor`, `graph-rag` |
| `/project [list\|new\|switch\|delete\|folder]` | Manage named projects with isolated knowledge bases |
| `/keys [set provider]` | Show API key status for all providers; `/keys set <provider>` saves a key interactively |
| `/compact` | Summarize entire chat history via LLM to free context window space |
| `/context` | Display token usage bar, model info, RAG settings, chat history, and last retrieved sources |
| `/sessions` | List recent saved sessions (up to 20 most recent) |
| `/resume <id>` | Load a previous session by its timestamp ID |
| `/retry` | Re-send the last query (useful after switching model or RAG settings) |
| `/clear` | Clear current chat history (does not delete saved session) |
| `/quit`, `/exit` | Exit the REPL |

### Docker
```bash
# Build image
make docker-build
docker build -t axon:latest .

# Run with docker-compose
make docker-run
docker-compose up -d

# View logs
make docker-logs
docker-compose logs -f

# Stop services
make docker-down
docker-compose down
```

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/new-feature

# Stage changes
git add .

# Commit (pre-commit hooks run automatically)
git commit -m "feat: add new feature"

# Push
git push origin feature/new-feature

# Update from main
git fetch origin
git rebase origin/main
```

## Configuration

### Environment Variables
```bash
# Copy template
cp .env.example .env

# Edit configuration
nano .env
```

**Key Variables:**
```bash
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=gemma
AXON_HOST=127.0.0.1
AXON_PORT=8000
LOG_LEVEL=INFO
```

### Config File
Edit `config.yaml`:
```yaml
embedding:
  provider: sentence_transformers  # or ollama, fastembed
  model: all-MiniLM-L6-v2

llm:
  provider: ollama
  model: gemma
  temperature: 0.7

rag:
  top_k: 10
  hybrid_search: true

  # GraphRAG-style indexing and retrieval. Implements: hierarchical community detection
  # (Leiden via graspologic, or Louvain fallback), LLM community reports, map-reduce global
  # search, token-budgeted local search, entity/relation graphs, optional claim/covariate
  # extraction. Approximate where noted (fallback hierarchy, unified candidate ranking).
  #
  # What it does:
  #   - Ingest: LLM extracts named entities (with descriptions) and
  #             SUBJECT|RELATION|OBJECT triples (with descriptions) per chunk.
  #             After ingest, Louvain community detection clusters the entity graph
  #             (requires: pip install networkx). LLM generates a summary paragraph
  #             per community cluster.
  #   - Query:  graph_rag_mode controls retrieval strategy:
  #     local   — entity descriptions + relation descriptions + community snippet
  #               prepended before document excerpts (default)
  #     global  — top community summaries ranked by embedding similarity injected
  #               as primary context (good for corpus-wide questions)
  #     hybrid  — community reports + document excerpts combined
  #
  # Known limits:
  #   - Shallow graph: no canonical entity resolution, no alias handling,
  #     no relation normalisation, no multi-hop reasoning beyond 1 hop.
  #   - Heuristic scoring: not learned or calibrated.
  #   - Extraction quality depends on the configured LLM.
  #   - Community detection is async by default; first query after ingest may
  #     run before communities are ready (graph_rag_community_async: false to block).
  #
  # Requires an LLM at ingest time. Adds per-chunk latency.
  graph_rag: false
  graph_rag_budget: 3              # extra entity-linked slots beyond top_k (0 = no guarantee)
  graph_rag_relations: true        # extract relation triples for 1-hop traversal
  # graph_rag_community: false     # run Louvain community detection (needs networkx)
  # graph_rag_community_async: true
  # graph_rag_community_top_k: 5
  # graph_rag_mode: local          # local | global | hybrid
```

### Offline / Air-gapped Mode
Pre-fetch models on an internet machine, then copy them to the confined workspace:
```bash
python scripts/prefetch_models.py --dir C:/models
```

Enable in `config.yaml`:
```yaml
offline:
  enabled: true
  local_models_dir: C:/models   # absolute path
```

Effects when enabled:
- Bare model names (`all-MiniLM-L6-v2`, `bge-reranker-base`) auto-resolved to `<local_models_dir>/<name>/`
- `TRANSFORMERS_OFFLINE=1`, `HF_HUB_OFFLINE=1` set before any model loads
- Web search permanently disabled (`/search` toggle is blocked)

For Ollama models (gemma3, gpt-oss, etc.) copy `~/.ollama/models/` to the same path on the confined machine. See `scripts/README.md`.

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Query Knowledge Base
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "filters": {}
  }'
```

### Search Documents
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Python programming",
    "top_k": 5
  }'
```

### Add Text
```bash
curl -X POST http://localhost:8000/add_text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Important information to remember",
    "metadata": {"source": "user", "topic": "notes"}
  }'
```

### Ingest Files (async — returns job_id immediately)
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/documents"}'
# Response: {"message": "Ingestion started", "job_id": "abc123", "status": "processing"}
```

### Poll Ingest Status
```bash
curl http://localhost:8000/ingest/status/abc123
# Response: {"job_id": "abc123", "status": "completed", "documents_ingested": 12}
```

### List Tracked Documents
```bash
curl http://localhost:8000/tracked-docs
# Response: {"sources": [{"source_id": "/path/file.txt", "chunks": 5, "ingested_at": "..."}]}
```

### Refresh (Re-check Changed Files)
```bash
curl -X POST http://localhost:8000/ingest/refresh
# Re-ingests any files whose content has changed since last ingest
```

### Projects
```bash
# List all projects
curl http://localhost:8000/projects

# Switch project (pass project param on any request)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "project": "my-project"}'
```

### AxonStore (Multi-User Sharing)
```bash
# Initialise store for a user
curl -X POST http://localhost:8000/store/init \
  -H "Content-Type: application/json" \
  -d '{"base_path": "/data/axon-store"}'

# Generate a share key
curl -X POST http://localhost:8000/share/generate \
  -H "Content-Type: application/json" \
  -d '{"project": "my-project", "grantee": "<os-username>", "write_access": false}'

# Redeem a share key
curl -X POST http://localhost:8000/share/redeem \
  -H "Content-Type: application/json" \
  -d '{"share_string": "axon-share-..."}'

# List active shares
curl http://localhost:8000/share/list

# Revoke a share
curl -X POST http://localhost:8000/share/revoke \
  -H "Content-Type: application/json" \
  -d '{"key_id": "sk_a1b2c3d4"}'
```

### OpenAPI / Swagger UI
The full interactive API reference is available at:
```
http://localhost:8000/docs
```

## Troubleshooting

For common errors and fixes, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Code Examples

### Basic RAG Query
```python
from axon.main import AxonBrain

brain = AxonBrain()
response = brain.query("What is the main topic?")
print(response)
```

### Custom Configuration
```python
from axon.main import AxonBrain, AxonConfig

config = AxonConfig(
    embedding_provider="ollama",
    llm_model="llama3.1:70b",
    top_k=5,
    hybrid_search=True
)
brain = AxonBrain(config)
```

### Ingest Documents
```python
import asyncio
from axon.main import AxonBrain, AxonConfig

config = AxonConfig(vector_store_path="./chroma", bm25_path="./bm25")
brain = AxonBrain(config)

# Ingest a directory (async)
asyncio.run(brain.load_directory("./my_documents"))

# Or ingest a list of document dicts directly (sync)
brain.ingest([
    {"id": "doc1", "text": "Your document text here.", "metadata": {"source": "example.txt"}}
])
```

### Direct API Usage
```python
import httpx

response = httpx.post(
    "http://localhost:8000/query",
    json={"query": "What is RAG?"}
)
print(response.json()["response"])
```

## Testing Patterns

### Unit Test
```python
def test_feature():
    """Test description."""
    # Arrange
    input_data = setup_data()

    # Act
    result = function(input_data)

    # Assert
    assert result == expected
```

### Integration Test
```python
@pytest.mark.integration
def test_api_endpoint():
    """Test API endpoint."""
    from fastapi.testclient import TestClient
    from axon.api import app

    client = TestClient(app)
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 200
```

### Async Test
```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

## Performance Tips

### Batch Processing
```python
# Instead of one-by-one
for doc in documents:
    brain.ingest([doc])  # Slow

# Use batch
brain.ingest(documents)  # Fast
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_function(query: str):
    # Cached for repeated queries
    return result
```

### Optimize Settings
```yaml
# config.yaml
rag:
  top_k: 5           # Fewer results = faster
  hybrid_search: false  # Vector only = faster

chunk:
  size: 500          # Smaller chunks = more granular
  overlap: 50        # Less overlap = fewer chunks
```

## Useful Links

- **Repository:** https://github.com/jyunming/Axon
- **Issues:** https://github.com/jyunming/Axon/issues
- **Discussions:** https://github.com/jyunming/Axon/discussions
- **Ollama:** https://ollama.ai/
- **ChromaDB:** https://www.trychroma.com/
- **FastAPI:** https://fastapi.tiangolo.com/
- **Streamlit:** https://streamlit.io/

## Keyboard Shortcuts (Streamlit UI)

- `Ctrl + K` - Focus chat input
- `Ctrl + L` - Clear chat
- `R` - Rerun app
- `C` - Clear cache

## Environment Setup Scripts

### Linux/Mac
```bash
#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pre-commit install
cp .env.example .env
echo "Setup complete! Edit .env and run: make test"
```

### Windows
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -e .[dev]
pre-commit install
copy .env.example .env
echo "Setup complete! Edit .env and run: make test"
```

## License

MIT License - See [LICENSE](LICENSE) file.

---

**Last Updated:** 2026-03-13
**Version:** 1.0.0
