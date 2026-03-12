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

### Adding a new MCP tool

MCP tools live in `src/axon/mcp_server.py`. They are **distinct** from the
OpenAI-format schemas in `tools.py` — MCP tool names use concise snake_case
optimised for agent-mode ergonomics (e.g. `search_knowledge`, not
`search_documents`). Do not rename or alter `tools.py` when adding an MCP tool.

Steps:
1. Add the tool definition to the `@mcp.tool()` decorated section of
   `mcp_server.py`. Follow the existing pattern: **sync** function, docstring as
   the tool description, typed parameters.
2. Inside the handler, call the REST API via `httpx.Client`. Always
   include the `X-API-Key` header (read from the `RAG_API_KEY` env var):
   ```python
   with httpx.Client(timeout=60.0) as client:
       resp = client.post(
           f"{API_BASE}/your_endpoint",
           json={...},
           headers={"X-API-Key": API_KEY} if API_KEY else {},
       )
   ```
3. Return structured data from the handler — the MCP SDK serialises it for
   the agent.
4. Run `python -m axon.mcp_server` and send a `tools/list` JSON-RPC request
   to verify the new tool appears before opening a PR.

## Style

- Use Python type hints on all function signatures.
- Prefer `logger.info/error` (module-level logger) over `print`.
- Keep changes surgical — modify only the files required by the task.
- Do not refactor unrelated code.

## Boundaries

- Do **not** remove or rename existing public API methods without Planner approval.
- Do **not** change the document dict schema.
- Do **not** add new dependencies without updating both `requirements.txt` and `setup.py`.
