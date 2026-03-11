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
axon --graph-rag "Your question"            # entity-graph retrieval

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
| `/rag [option]` | Show or modify RAG settings — try `/rag` with: `topk <n>`, `threshold <0-1>`, `hybrid`, `rerank`, `rerank-model <model>`, `hyde`, `multi` |
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

# Update from master
git fetch origin
git rebase origin/master
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

### Ingest Files
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "path": "/path/to/documents"
  }'
```

## Troubleshooting

### Common Issues

**Issue: Import errors**
```bash
# Solution
pip install -e .
# Or reinstall
pip uninstall axon
pip install -e .
```

**Issue: Tests fail**
```bash
# Solution
pip install -e ".[dev]"
pytest -v  # Verbose mode to see details
```

**Issue: Ollama connection failed**
```bash
# Check Ollama is running
ollama list

# Start Ollama service
# On Linux/Mac: ollama serve
# On Windows: Start Ollama application

# Test connection
curl http://localhost:11434/api/tags
```

**Issue: ChromaDB errors**
```bash
# Clear and rebuild
rm -rf chroma_data/
rm -rf bm25_index/
# Re-ingest documents
```

**Issue: Port already in use**
```bash
# Find process using port
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Change port
export AXON_PORT=8001
axon-api
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
axon-api

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Code Examples

### Basic RAG Query
```python
from axon.main import OpenStudioBrain

brain = OpenStudioBrain()
response = brain.query("What is the main topic?")
print(response)
```

### Custom Configuration
```python
from axon.main import OpenStudioBrain, OpenStudioConfig

config = OpenStudioConfig(
    embedding_provider="ollama",
    llm_model="llama3.1:70b",
    top_k=5,
    hybrid_search=True
)
brain = OpenStudioBrain(config)
```

### Ingest Documents
```python
import asyncio
from axon.main import OpenStudioBrain

async def ingest_docs():
    brain = OpenStudioBrain()
    await brain.load_directory("./my_documents")

asyncio.run(ingest_docs())
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

- **Repository:** https://github.com/jyunming/studio_brain_open
- **Issues:** https://github.com/jyunming/studio_brain_open/issues
- **Discussions:** https://github.com/jyunming/studio_brain_open/discussions
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

**Last Updated:** 2026-03-08
**Version:** 2.0.0
