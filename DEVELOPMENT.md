# Development Guide

## Quick Start for Developers

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/jyunming/Axon.git
cd Axon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with dev dependencies
make install-dev
# Or: pip install -e ".[dev]"
```

### 2. Common Development Tasks

**Run Tests:**
```bash
make test          # Run all tests
make test-cov      # Run with coverage report
pytest -k test_name  # Run specific test
```

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
│   ├── api.py              # REST API endpoints
│   ├── webapp.py           # Streamlit UI
│   ├── loaders.py          # Document loaders
│   ├── retrievers.py       # Search implementations
│   ├── splitters.py        # Text chunking
│   └── tools.py            # Agent tool definitions
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
├── .github/workflows/      # CI/CD pipelines
├── config.yaml             # Configuration template
├── pyproject.toml         # Package metadata
├── Makefile               # Development commands
└── README.md              # User documentation
```

### 4. Development Workflow

1. **Create a feature branch from `master`:**
   > ⚠️ **Never commit directly to `master`.** Always branch first.
   ```bash
   git checkout master && git pull
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

4. **Push and create PR targeting `master`:**
   ```bash
   git push origin feature/your-feature
   # Create pull request on GitHub targeting master
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

1. Update version in `setup.py` and `pyproject.toml`
2. Run all tests: `make ci`
3. Create git tag: `git tag v2.0.0`
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
