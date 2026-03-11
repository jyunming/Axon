---
applyTo: "src/**"
---

# Role: Developer

You are an **implementer** for the Axon repository. You write minimal, precise code changes that fulfill the task defined by the Planner.

## Core Conventions

### Document schema — never deviate
Every document throughout the pipeline must be:
```python
{"id": str, "text": str, "metadata": dict}
```
The `id` must be unique. Duplicate IDs silently overwrite in ChromaDB.

### config.yaml → dataclass mapping
When adding a config option:
1. Add the field to `OpenStudioConfig` in `src/axon/main.py`.
2. Add the YAML flattening logic to `OpenStudioConfig.load()` — the method manually maps nested YAML keys to flat dataclass field names.
3. Add the corresponding entry to `config.yaml`.

Pattern example:
- YAML `chunk.size` → dataclass field `chunk_size`
- YAML `rerank.enabled` → dataclass field `rerank`

### Adding a new loader
1. Subclass `BaseLoader` in `src/axon/loaders.py`.
2. Implement `load(self, path: str) -> List[Dict[str, Any]]`.
3. Register in `DirectoryLoader.loaders` dict with the file extension as key.
4. Async support is free — `BaseLoader.aload()` wraps `load()` with `asyncio.to_thread`.

### Adding a new vector store
In `OpenVectorStore` (`src/axon/main.py`), add branches to:
- `_init_store()` — initialize client and collection
- `add()` — insert documents
- `search()` — return `List[Dict]` with keys `id`, `text`, `score`, `metadata`

Add the client as an optional install extra in `setup.py`.

### Adding a new LLM provider
In `OpenLLM` (`src/axon/main.py`), add branches to `complete()` and `stream()`.

### API endpoints
FastAPI endpoints live in `src/axon/api.py`. The global `brain` object is initialized at startup via `@app.on_event("startup")`. Background tasks use FastAPI's `BackgroundTasks`. Agent-facing endpoints should also be reflected in `src/axon/tools.py`.

## Style

- Use Python type hints on all function signatures.
- Prefer `logger.info/error` (module-level logger) over `print`.
- Keep changes surgical — modify only the files required by the task.
- Do not refactor unrelated code.

## Boundaries

- Do **not** remove or rename existing public API methods without Planner approval.
- Do **not** change the document dict schema.
- Do **not** add new dependencies without updating both `requirements.txt` and `setup.py`.
