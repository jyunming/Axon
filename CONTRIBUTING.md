# Contributing to Axon

Thank you for your interest in contributing to Axon! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites
- Python 3.10 or higher
- Git
- Ollama (for local LLM support)

### Setting Up Your Development Environment

See [SETUP.md](SETUP.md) for platform-specific installation steps. Quick start:

```bash
git clone https://github.com/jyunming/Axon.git && cd Axon
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

Run the full test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=axon --cov-report=html
```

Run specific tests:
```bash
pytest tests/test_splitters.py
pytest -k test_basic_split
```

### Code Quality

We use several tools to maintain code quality:

**Format code with Black:**
```bash
black src/ tests/
```

**Lint with Ruff:**
```bash
ruff check src/ tests/
ruff check --fix src/ tests/  # Auto-fix issues
```

**Type checking with MyPy:**
```bash
mypy src/axon/
```

**Pre-commit checks:**
All checks run automatically on commit if you installed pre-commit hooks:
```bash
pre-commit run --all-files  # Run manually
```

## Project Structure

```
Axon/
├── src/
│   └── axon/          # Main package
│       ├── main.py         # Core RAG engine
│       ├── api.py          # FastAPI endpoints
│       ├── webapp.py       # Streamlit UI
│       ├── loaders.py      # Document loaders
│       ├── retrievers.py   # BM25 and fusion
│       ├── splitters.py    # Text chunking
│       └── tools.py        # Agent tool definitions
├── tests/                  # Test suite
├── examples/               # Example scripts
├── pyproject.toml         # Package metadata and tool config
├── requirements.txt        # Dependencies
└── setup.py               # Package setup (legacy)
```

## Coding Standards

### Python Style
- Follow PEP 8
- Maximum line length: 100 characters
- Use type hints where appropriate
- Write docstrings for public functions and classes

### Documentation
- Use Google-style docstrings
- Include examples in docstrings where helpful
- Update README.md if adding major features
- **Keep docs in sync:** whenever a function, API endpoint, or config option is added or changed, update the relevant docs (README, SETUP.md, QUICKREF.md) in the same PR.

### Example Docstring:
```python
def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Search for documents matching the query.

    Args:
        query: The search query string.
        top_k: Maximum number of results to return.

    Returns:
        List of documents with scores and metadata.

    Example:
        >>> retriever = BM25Retriever()
        >>> results = retriever.search("Python programming", top_k=5)
    """
```

## Testing Guidelines

### Writing Tests
- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Test edge cases and error conditions

### Test Organization
- **Unit tests**: Test individual functions/classes
- **Integration tests**: Test component interactions
- **Mark slow tests**: Use `@pytest.mark.slow` for tests >1s

### Example Test:
```python
def test_add_and_search():
    """Test adding documents and searching."""
    retriever = BM25Retriever()
    docs = [{"id": "1", "text": "test", "metadata": {}}]
    retriever.add_documents(docs)
    results = retriever.search("test")
    assert len(results) > 0
```

## Pull Request Process

1. **Create a feature branch from `main`**
   > ⚠️ **Never commit directly to `main`.** All changes go through PRs.
   ```bash
   git checkout main && git pull
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

3. **Ensure all tests pass**
   ```bash
   pytest
   black src/ tests/
   ruff check src/ tests/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description"
   ```

5. **Push to your fork and open a PR targeting `main`**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear description
   - Reference any related issues
   - Ensure CI passes

### PR Guidelines
- Keep PRs focused on a single feature/fix
- Include tests for new functionality
- Update documentation as needed
- Respond to review feedback promptly

## Reporting Issues

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces
- Relevant configuration

## Feature Requests

We welcome feature requests! Please:
- Check if the feature already exists
- Describe the use case clearly
- Explain why it would be valuable
- Consider implementing it yourself

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## Questions?

- Open an issue for bugs/features
- Start a discussion for questions
- Check existing issues/PRs first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to Axon! 🧠
