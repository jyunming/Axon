# Contributing to Axon

Thanks for your interest in Axon. This guide covers the development setup, tests and
hooks, how the code is laid out, evaluating answer quality, and the release process. User
documentation lives in [`docs/`](docs/README.md).

## Development setup

You need Python 3.10+, Git, and — for anything that calls an LLM — [Ollama](https://ollama.com)
or a cloud key.

```bash
git clone https://github.com/jyunming/Axon.git && cd Axon
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"          # tests, black 23.12.1, ruff, mypy, pre-commit, optional backends
pip install -e ".[all,dev]"      # also every optional runtime dependency
pre-commit install
```

`axon`, `axon-api`, `axon-mcp` and `axon-ext` now run from your checkout. `make help` lists
the Makefile targets (`make run-api`, `make test`, `make lint`, `make docker-run`, …).

**Version.** `src/axon/Cargo.toml` (`[package].version`) is the single source of truth.
`pyproject.toml` declares `dynamic = ["version"]`, and Python code reads the installed
version through `importlib.metadata`.

## Tests

```bash
python -m pytest tests/ -v --no-cov                 # full suite
python -m pytest tests/test_splitters.py -v --no-cov
python -m pytest -k test_basic_split --no-cov
python -m pytest tests/ --cov=axon --cov-report=html
python -m pytest -m "not slow" --no-cov
```

`--no-cov` matters: `pyproject.toml` turns coverage on by default, and a coverage run
rewrites coverage files. The VS Code end-to-end tests
(`tests/e2e/test_vscode_extension_*.py`) need a live VS Code; deselect them with
`-m "not extension"` when running headless. Markers (`e2e`, `slow`, `integration`, `eval`,
`demo`, `perf`, `stress`, …) are declared in `pyproject.toml`.

Tests that build an `AxonBrain` use a `MagicMock` brain: set attributes directly
(`brain._ingested_hashes = set()`) rather than patching properties. The Rust bridge hashes
documents with SHA-256 (64 hex characters).

## Pre-commit hooks

`pre-commit install` wires these up (see `.pre-commit-config.yaml`):

| Hook | Runs on |
|---|---|
| `check-yaml`, `check-json`, `check-toml`, `check-added-large-files` (1 MB), `detect-private-key` | Every commit |
| `black` 23.12.1 and `ruff --fix` | Python files |
| Trailing whitespace, merge-conflict markers, `py_compile` | Text / source files |
| `pytest-scoped` | Commits touching `src/axon/**/*.py`, `tests/**/*.py`, `pyproject.toml`, `src/axon/Cargo.toml`, `requirements*.txt` or `config.yaml` |
| `eval-smoke` | Same trigger: `tests/test_eval_smoke.py -m eval` |

The pytest hook (`scripts/precommit_pytest_scoped.py`) maps each staged file to a small set
of test files by path prefix — `axon/security/*` → the sealed-share tests,
`axon/api_routes/*` → the API tests, and so on. Documentation, HTML, SVG and script-only
commits skip it entirely. A change to a widely imported module (`cli.py`, `config.py`)
selects a broader subset and can take minutes. The mapping is heuristic; CI runs the full
suite on every push, and `python -m pytest tests/ --no-cov` runs it locally.

Format before committing: `python -m black <files> && ruff check --fix <files>`. If a hook
fails, fix the cause and make a new commit rather than amending.

**`pre-commit install` refuses with `core.hooksPath` set.** Some managed environments set
a global hooks path: `git config --unset core.hooksPath` (add `--global` if it is global),
then `pre-commit install`.

## Code quality

```bash
black src/ tests/                 # format (line length 100)
ruff check --fix src/ tests/      # lint
mypy src/axon/                    # types
pre-commit run --all-files
```

Follow PEP 8, type public signatures, write Google-style docstrings for public functions
and classes, and keep functions focused.

## Project layout

```
src/axon/
  main.py                 AxonBrain — the engine
  config.py               AxonConfig — every setting, YAML loading and validation
  query_router.py         the query pipeline and per-query routing
  retrievers.py, vector_store.py, embeddings.py, rerank.py, splitters.py, loaders.py
  graph_rag.py, graph_backends/, dynamic_graph/, code_graph.py, code_retrieval.py
  projects.py             projects, scopes and maintenance state
  shares.py, share_validity.py, security/     sharing and sealed projects
  api.py, api_routes/     the REST server (69 routes) and the web GUI mount
  gui/                    the web GUI served at /gui/
  cli.py, repl.py         the axon command and its REPL
  mcp_server.py           the MCP server (18 tools)
  surface_contract.py     which capability is on which surface
  axon_rust_lib.rs        Rust extension: BM25, score fusion, hashing, msgpack I/O
integrations/vscode-axon/ the VS Code extension (20 LM tools, 19 commands)
tests/                    the test suite
docs/                     user guides (GETTING_STARTED, REFERENCE, SHARING, TROUBLESHOOTING),
                          CAPABILITIES.md (internal reuse map) and architecture/ design notes
scripts/                  release, evaluation, model pre-fetch and QA helpers
```

## Changing behaviour

- **Check [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md) first.** It maps the reusable
  functions and classes per subsystem so new work extends what exists instead of
  duplicating it. Update it when you add a reusable capability or invalidate an entry.
- **New settings are `AxonConfig` fields.** Reading a setting with
  `getattr(cfg, "name", default)` bypasses YAML loading, validation and the config routes.
- **New user-facing capabilities register in `src/axon/surface_contract.py`.** Human
  surfaces (REST, CLI, REPL) come first; agent surfaces (MCP, VS Code LM tools) are opt-in
  and deliberately small — destructive, credential and store-administration operations stay
  human-only. `tests/test_surface_parity_contract.py` keeps the registry, the MCP tool list
  and the VS Code manifest in step.
- **Docs change in the same PR.** User-visible commands, flags, settings, routes and tools
  go in [`docs/REFERENCE.md`](docs/REFERENCE.md); first-run behaviour in
  [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md); sharing in
  [`docs/SHARING.md`](docs/SHARING.md); error fixes in
  [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md). Recount surface numbers rather than
  incrementing them.

## Evaluating answer quality

Two layers.

**Smoke tests** (`tests/test_eval_smoke.py`) compute precision@k, context relevance,
faithfulness and recall@k against a tiny mocked corpus with keyword matching — no model, no
network. They run in the pre-commit hook and in CI:

```bash
python -m pytest -m eval tests/test_eval_smoke.py -v --no-cov
```

**End-to-end scoring with RAGAS** (`scripts/evaluate.py`) sends real questions through
Axon and scores the answers with a local Ollama judge — no OpenAI key needed:

```bash
pip install ragas langchain-community datasets
ollama pull llama3.1:8b && ollama pull nomic-embed-text      # judge model and embeddings
python scripts/evaluate.py --testset examples/eval_testset.jsonl --config config.yaml
```

| Flag | Default | |
|---|---|---|
| `--testset` | required | JSONL file |
| `--config` | `config.yaml` | Axon config to evaluate |
| `--model` | from the config | Judge / answer model override |
| `--output` | `eval_report_<timestamp>.md` | Markdown report |

A testset has one JSON object per line — `{"question": "...", "ground_truth": "..."}`; a
starter set is in `examples/eval_testset.jsonl`. Use 20–50 questions from real use, with
specific ground truths and some multi-document questions.

| Metric | Low score means | Try |
|---|---|---|
| `faithfulness` | Claims not supported by the context | Citations, compression, a stronger model |
| `context_recall` | Needed information wasn't retrieved | Higher `top_k`, hybrid search, HyDE |
| `context_precision` | Retrieved chunks are noisy | Reranking, a higher threshold, lower `top_k` |
| `answer_relevancy` | Answers miss the question | Step-back or decomposition, prompt review |

`tests/test_deepeval_integration.py` checks the DeepEval integration (the `eval` extra);
it runs without a judge by default and against a real one when `DEEPEVAL_JUDGE_MODEL` and
a key are set.

## Pull requests

1. Branch from `main` — never commit to `main` directly:
   `git checkout main && git pull && git checkout -b feature/your-change`.
2. Make the change with tests and docs.
3. Run the relevant tests and `pre-commit run --files <changed files>`.
4. Push and open a PR against `main` with a clear description and linked issues; keep it
   focused (large PRs are hard to review — aim for well under 60 changed files).
5. Make CI pass and address review comments.

## Releases

Maintainers only. Publishing to PyPI is triggered by a version tag, not by merging.

```bash
python scripts/bump_version.py X.Y.Z        # Cargo.toml, VS Code package.json, index.html, VSIX, Cargo.lock
python scripts/audit_packaging.py --expected-version X.Y.Z
# commit, PR, merge, then:
git tag vX.Y.Z && git push origin vX.Y.Z
```

`bump_version.py` also rebuilds the VS Code extension and bundles it under
`src/axon/extensions/` (`--skip-vsix` / `--skip-cargo-lock` to skip steps). Bump only for a
functional change. PyPI releases are immutable and `README.md` is the PyPI page — check it
renders (absolute image URLs) before tagging.

## Reporting issues and requesting features

Open an [issue](https://github.com/jyunming/Axon/issues) with your Python version, OS, the
steps to reproduce, expected and actual behaviour, the error or stack trace, and the
relevant config (`axon --doctor` output helps). For features, describe the use case and why
it matters; check existing issues first. Security problems go through
[SECURITY.md](SECURITY.md), not public issues.

## Code of conduct

Be respectful and inclusive, welcome newcomers, keep feedback constructive, and assume good
intentions.

## License

By contributing you agree that your contributions are licensed under the MIT License.
