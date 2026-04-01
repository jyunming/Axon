# Development Guide

## Quick Start for Developers

### 1. Setup Environment

See [SETUP.md](SETUP.md) for platform-specific installation steps. Quick start (installs all features, dev tools, and evaluation suites):

```bash
git clone https://github.com/jyunming/Axon.git && cd Axon
pip install -e ".[all,dev,eval]"
```

**Note on Offline / Local Support:**
All core and optional dependencies (including `networkx`, `igraph`, `leidenalg`, and `graspologic`) are **local-first**. They perform all computations (graph clustering, embeddings, matrix math) on your local hardware and do not require internet access at runtime. For a fully air-gapped environment, ensure you also use a local LLM provider like **Ollama**.

---

### 2. Common Development Tasks

**Run Tests:**
```bash
python -m pytest tests/ -v --no-cov          # standard run (no coverage file)
python -m pytest tests/ --cov=axon --cov-report=html  # with HTML coverage report
python -m pytest tests/test_main.py -v --no-cov       # single file
python -m pytest -k test_name --no-cov                # match by name
```

> VS Code extension e2e tests (`tests/e2e/test_vscode_extension_*.py`) require a live VS Code instance. Exclude them with `-m "not extension"` if running headlessly. The pre-commit hook does this automatically.

**Code Quality:**
```bash
make format        # Auto-format code
make lint          # Check code style
make type-check    # Run type checking
make all           # Run all checks
```

**Run Services:**
```bash
make run-api       # Start FastAPI server
make run-ui        # Start Streamlit UI
```

**Docker:**
```bash
make docker-build  # Build image
make docker-run    # Start with docker-compose
make docker-logs   # View logs
```

### 3. Project Structure

```
Axon/
├── src/axon/          # Main package
│   ├── main.py             # Core RAG engine
│   ├── api.py              # FastAPI app factory and route registration
│   ├── api_routes/         # Route handlers (54 endpoints across 8 files)
│   ├── webapp.py           # Streamlit UI
│   ├── loaders.py          # Document loaders
│   ├── retrievers.py       # Search implementations
│   ├── splitters.py        # Text chunking
│   ├── tools.py            # Agent tool definitions
│   ├── projects.py         # Multi-user project management
│   ├── shares.py           # HMAC share key generation and redemption
│   └── mcp_server.py       # MCP stdio server for Copilot agent mode
├── tests/                  # Test suite
│   ├── conftest.py         # Shared fixtures (overrides tmp_path for Windows compat)
│   ├── test_api.py         # FastAPI endpoint tests
│   ├── test_config.py      # Configuration tests
│   ├── test_loaders.py     # Loader tests
│   ├── test_main.py        # Core RAG pipeline tests
│   ├── test_retrievers.py  # BM25 and RRF tests
│   ├── test_splitters.py   # Splitter tests
│   ├── test_streaming.py   # Streaming response tests
│   └── test_tools.py       # Agent tool definition tests
├── examples/               # Example scripts
├── docs/                   # All guides and references (SETUP, API_REFERENCE, etc.)
├── .github/workflows/      # CI/CD pipelines
├── pyproject.toml         # Package metadata (primary source of truth for version; also update package.json + __init__.py on bumps)
├── Makefile               # Development commands
└── README.md              # Root overview + links to docs/
```

> **Version sync:** `pyproject.toml` is the primary source of truth for the Python package version. A version bump requires updating all six locations in sync:
> - `pyproject.toml` → `version = "X.Y.Z"`
> - `setup.py` → `version="X.Y.Z"`
> - `src/__init__.py` → `__version__ = "X.Y.Z"`
> - `src/axon/__init__.py` → `__version__ = "X.Y.Z"`
> - `src/axon/api.py` → FastAPI `version=` field
> - `integrations/vscode-axon/package.json` → `"version": "X.Y.Z"` (VSIX artefact)

### 4. Development Workflow

1. **Create a feature branch from `main`:**
   > ⚠️ **Never commit directly to `main`.** Always branch first.
   ```bash
   git checkout main && git pull
   git checkout -b feature/your-feature
   ```

2. **Make changes and test:**
   ```bash
   # Edit code
   make format      # Format code
   make test        # Run tests
   ```

3. **Commit changes:**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   # Pre-commit hooks will run automatically
   ```

4. **Push and create PR targeting `main`:**
   ```bash
   git push origin feature/your-feature
   # Create pull request on GitHub targeting main
   ```

### 5. Testing Guidelines

**Write tests for:**
- All new features
- Bug fixes
- Edge cases

**Test structure:**
```python
def test_feature_name():
    """Test description."""
    # Arrange
    input_data = setup_test_data()
    # Act
    result = function_under_test(input_data)
    # Assert
    assert result == expected_output
```

**Run specific test categories:**
```bash
pytest tests/test_loaders.py              # Specific file
pytest -k "test_load"                     # Tests matching pattern
pytest -m "not slow"                      # Exclude slow tests
pytest --cov=axon --cov-report=html  # With coverage
```

### 6. Code Style

**Follow these conventions:**
- Use Black for formatting (max line length: 100)
- Follow PEP 8 style guide
- Add type hints to function signatures
- Write docstrings for public APIs
- Keep functions focused and small

**Example:**
```python
from typing import List, Dict, Any
def search_documents(
    query: str,
    top_k: int = 10,
    filters: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """Search for documents matching the query.
    Args:
        query: Search query string.
        top_k: Maximum number of results to return.
        filters: Optional metadata filters.
    Returns:
        List of documents with scores and metadata.
    Example:
        >>> results = search_documents("Python tutorial", top_k=5)
        >>> len(results)
        5
    """
    # Implementation here
    pass
```

### 7. Debugging Tips

**Enable debug logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Use debugger:**
```python
import pdb; pdb.set_trace()  # Insert breakpoint
```

**Check API responses:**
```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'
```

### 8. Performance Profiling

**Profile code:**
```python
import cProfile
import pstats
profiler = cProfile.Profile()
profiler.enable()
# Code to profile
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(10)
```

**Memory profiling:**
```bash
pip install memory_profiler
python -m memory_profiler your_script.py
```

### 9. Common Issues

**Issue: Import errors**
```bash
# Solution: Install in editable mode
pip install -e .
```

**Issue: Tests fail**
```bash
# Solution: Check dependencies
pip install -e ".[dev]"
pytest --verbose
```

**Issue: Pre-commit hooks fail**
```bash
# Solution: Format and fix
make format
make lint
```

### 10. Release Process

1. Update version in **all** version files (must stay in sync):
   - `pyproject.toml` ← **only file to change** for the Python package version
   - `integrations/vscode-axon/package.json` ← bump manually to match, then `npm run package` to rebuild VSIX
   - All other Python files (`api.py`, `llm.py`, `__init__.py`) read the version dynamically via `importlib.metadata`
2. Run all tests: `make ci`
3. Create git tag: `git tag v0.1.0`
4. Push tag: `git push --tags`
5. GitHub Actions will build and publish

## Resources

- [Contributing Guide](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)

## Getting Help

- 🐛 **Bug Reports:** [GitHub Issues](https://github.com/jyunming/Axon/issues)
- 💡 **Feature Requests:** [GitHub Discussions](https://github.com/jyunming/Axon/discussions)
- 💬 **Questions:** [Discord/Slack Community](#)

Happy coding! 🚀
