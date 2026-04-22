# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/axon/`. Key areas: `api_routes/` for FastAPI handlers, `graph_backends/` and `dynamic_graph/` for graph retrieval, `code_graph.py` and `code_retrieval.py` for code-aware RAG, and `repl.py` / `cli.py` / `mcp_server.py` for user-facing entry points. Tests live in `tests/`; VS Code extension end-to-end coverage is under `tests/e2e/`. The bundled extension is in `integrations/vscode-axon/`. Top-level docs are in `docs/`.

## Build, Test, and Development Commands
- `pip install -e ".[dev]"`: install Axon with local dev tools.
- `make format`: run `black` and `ruff --fix` on `src/` and `tests/`.
- `make lint`: run `ruff check` and `black --check`.
- `make type-check`: run `mypy src/axon/ --ignore-missing-imports`.
- `make test`: run the Python test suite.
- `pytest tests/test_api.py -v --no-cov`: run a focused test file.
- `pytest -q tests/e2e`: run end-to-end coverage; requires Node and the compiled VS Code extension.
- `axon`, `axon-api`, `axon-ui`, `axon-mcp`: run the REPL, API server, Streamlit UI, and MCP server.

## Coding Style & Naming Conventions
Use Python 3.10+ and 4-space indentation. Black and Ruff are the enforced style tools; line length is `100`. Prefer type hints on public functions and keep module names `snake_case`. Use `PascalCase` for classes, `snake_case` for functions/tests, and descriptive test names such as `test_ingest_blocks_windows_system_path`.

## Testing Guidelines
Pytest is the main framework. Put new tests in `tests/` as `test_*.py`; keep one behavior per test where practical. Add regression tests for bug fixes. For surface changes, cover the narrowest layer first (unit/API/REPL) before broader e2e tests. Avoid writing to real user config or store paths; use temp directories and explicit config overrides.

## Commit & Pull Request Guidelines
Recent history favors short, imperative subjects with optional scopes, for example `fix(cli): add --non-interactive flag` or `api(query): harden dry-run`. Keep commits focused. PRs should include: the behavior change, test coverage, linked issues when applicable, and screenshots/GIFs for UI or VS Code panel changes. Update docs when commands, config, APIs, or user-visible flows change.

## Security & Configuration Tips
Do not assume network access or cloud credentials. Treat offline/local-first behavior as a core contract. When testing ingestion or storage flows, prefer isolated paths over `~/.axon` or live `config.yaml`.
