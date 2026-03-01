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
mypy src/rag_brain/

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
pytest --cov=rag_brain --cov-report=html

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
rag-brain-api

# Streamlit UI
make run-ui
rag-brain-ui

# CLI
rag-brain "What is RAG?"
rag-brain --ingest ./documents/
rag-brain --query "test" --stream
```

### Docker
```bash
# Build image
make docker-build
docker build -t local-rag-brain:latest .

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
OLLAMA_MODEL=llama3.1
RAG_BRAIN_HOST=0.0.0.0
RAG_BRAIN_PORT=8000
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
  model: llama3.1
  temperature: 0.7

rag:
  top_k: 10
  hybrid_search: true
```

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
pip uninstall local-rag-brain
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
export RAG_BRAIN_PORT=8001
rag-brain-api
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
rag-brain-api

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Code Examples

### Basic RAG Query
```python
from rag_brain.main import OpenStudioBrain

brain = OpenStudioBrain()
response = brain.query("What is the main topic?")
print(response)
```

### Custom Configuration
```python
from rag_brain.main import OpenStudioBrain, OpenStudioConfig

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
from rag_brain.main import OpenStudioBrain

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
    from rag_brain.api import app

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

- **Repository:** https://github.com/yourusername/studio_brain_open
- **Issues:** https://github.com/yourusername/studio_brain_open/issues
- **Discussions:** https://github.com/yourusername/studio_brain_open/discussions
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

**Last Updated:** 2026-02-28
**Version:** 2.0.0
