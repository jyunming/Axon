# Troubleshooting Guide

Common issues and fixes for Axon.

---

## Ollama: Not Running When You Query

**Symptom:** You run `axon` and the prompt appears, but the first query hangs or returns an error like `httpx.ConnectError` or `Connection refused`.

**Cause:** Ollama is not running. Axon starts successfully without Ollama, but fails the moment it tries to call the LLM (on the first query, not at startup).

**Fix:**
1. Open the Ollama app from your Applications / Start menu, **or** run `ollama serve` in a separate terminal.
2. Verify it is up: `curl http://localhost:11434` — you should see `Ollama is running`.
3. Confirm your model is pulled: `ollama list` — the model in your `config.yaml` must appear here.

**Tip:** If you want Ollama to start automatically on login, enable it in your system settings (macOS: System Settings → General → Login Items; Windows: Task Manager → Startup tab).

---

## ChromaDB: `InvalidDimensionException`

**Error:**
```
chromadb.errors.InvalidDimensionException: Embedding dimension 384 does not match collection dimensionality 768
```

**Cause:** You switched the embedding model (e.g., from `nomic-embed-text` at 768d to `all-MiniLM-L6-v2` at 384d) but the existing ChromaDB collection was created with the old dimension.

**Fix:**

> ⚠️ **Data loss warning:** The steps below delete all indexed documents. Back up `~/.axon/` (or wherever `AXON_STORE_BASE` points) before proceeding if you want to preserve your data.

```bash
# Non-Docker install — delete the local data directories
rm -rf ~/.axon/AxonStore/$(whoami)/default/chroma_data ~/.axon/AxonStore/$(whoami)/default/bm25_index
# Then restart axon — it will re-create empty indexes
# Docker install
docker compose stop
rm -rf ./chroma_data ./bm25_index
docker compose up -d
```

Re-ingest your files after restarting.

**Prevention:** Before switching embedding models, always back up your data directory. After switching, you must re-ingest everything — there is no migration path between different embedding dimensions.

---

## ChromaDB: `Could not connect to tenant default_tenant`

**Error:**
```
ValueError: Could not connect to tenant default_tenant. Are you sure it exists?
```

**Cause:** The ChromaDB data directory was partially deleted while containers were still running, corrupting the SQLite metadata.

**Fix:** Fully stop Docker, delete, and restart:
```bash
docker compose down
rm -rf ./chroma_data
docker compose up -d
```

---

## Ollama: LLM Forgets Conversation History

**Symptom:** The assistant doesn't remember what you discussed earlier in the same session, even though sessions are enabled.

**Cause:** Ollama defaults to a 2048-token context window (`num_ctx`). When RAG context documents fill this window, conversation history gets silently truncated.

**Fix:** This is now handled automatically — `num_ctx` is set to 8192 in all Ollama calls. If you still experience issues with very long conversations, you can increase it further in `llm.py` (`OpenLLM` class, search for `num_ctx`).

---

## SentenceTransformers: `unexpected keyword argument 'convert_to_list'`

**Error:**
```
TypeError: SentenceTransformer.encode() got an unexpected keyword argument 'convert_to_list'
```

**Cause:** Some versions of `sentence-transformers` don't support the `convert_to_list` parameter.

**Fix:** Already patched — the code now uses `.tolist()` on the numpy array instead.

---

## Docker: File Changes Not Reflected in Web UI

**Symptom:** You edited source files but the Streamlit UI still runs old code.

**Cause:** Docker Desktop on Windows sometimes fails to propagate filesystem events through volume mounts, so Streamlit's auto-reloader doesn't trigger.

**Fix:**
```bash
docker compose restart axon-ui
```

---

## Brave Search: `truth_grounding` Enabled But No Web Results

**Symptom:** Web search toggle is on but no 🌐 sources appear.

**Cause:** Missing or invalid Brave API key.

**Fix:**
1. Get a free API key at [https://brave.com/search/api/](https://brave.com/search/api/)
2. Either set `BRAVE_API_KEY` in your `.env` file, or enter it in the Web UI sidebar under "🌐 Web Search"

---

## General: `.env` File

Create a `.env` file in the project root for API keys:
```env
GEMINI_API_KEY=your-gemini-key
BRAVE_API_KEY=your-brave-key
OLLAMA_CLOUD_KEY=your-ollama-cloud-key
OLLAMA_CLOUD_URL=https://your-endpoint
```

The `.env` file is optional — Docker Compose won't fail if it's missing.

---

## Answer Quality Variability with Advanced RAG Features

**Symptom:** Features like `multi_query`, inline citations, GraphRAG entity extraction, or RAPTOR summarisation produce inconsistent or degraded results.

**Cause:** These features rely on the LLM following structured instructions. Smaller models (e.g., `llama3.2:1b`, `gemma:2b`) may ignore instructions, produce malformed output, or refuse to answer when context is complex. This is not a bug — it reflects the capability limits of the model.

**Guidance:**
- Use a capable model (7B+ parameters) for best results with advanced RAG features.
- If `multi_query` degrades answer quality, disable it: `/rag multi` or set `multi_query: false` in `config.yaml`.
- If citations are missing or the model refuses, try a larger model or disable citation mode: `/rag cite`.
- GraphRAG entity extraction requires an LLM that can follow extraction instructions reliably. If the entity graph remains empty after ingestion, check the server logs for a zero-entity warning and consider switching to a larger model.

---

## About the GraphRAG Feature

Axon's `graph_rag` option implements a **GraphRAG-style pipeline** with the following capabilities:

- Hierarchical community detection: Leiden algorithm via `graspologic` when available, Louvain fallback otherwise.
- LLM-generated community reports and summaries per community cluster.
- Map-reduce global search: community reports are chunked, mapped in parallel by LLM, then reduced into a single ranked answer.
- Token-budgeted local search over community hierarchy: entity descriptions, relation descriptions, community snippets, and raw text units assembled within a configurable token budget.
- Entity and relation graphs with strength tracking and support-count accumulation.
- Optional claim/covariate extraction (`graph_rag_claims: true`).
- Optional entity and relation description canonicalization (`graph_rag_canonicalize`, `graph_rag_canonicalize_relations`).

What remains approximate compared to the Microsoft GraphRAG reference implementation:
- The Louvain fallback produces a synthetic hierarchy via multi-resolution clustering (not true Leiden).
- Candidate ranking is unified (degree + embedding similarity) rather than full DRIFT search.
- No query-time claim filtering.

Known limits:
- Extraction quality depends entirely on the configured LLM. A weak or small model will produce a noisy or empty graph.
- Entity matching uses exact match for single tokens and token-Jaccard for multi-token phrases. Aliases, acronyms, and spelling variants are not resolved without canonicalization enabled.

---

## GraphRAG Adds No Extra Results

**Symptom:** GraphRAG is enabled and ingestion succeeded (entities were extracted), but query results never include any entity-linked documents beyond the normal top_k.

**Cause:** One of the following:
- `graph_rag_budget` is set to `0`, which disables the guaranteed expansion slots.
- Entity extraction worked but entity matching at query time finds no overlap (e.g. the query uses different terminology than the indexed entities).
- The entity graph is empty — see the section below.

**Fix:**
- Confirm `graph_rag_budget > 0` in `config.yaml` (default is `3`):
  ```yaml
  rag:
    graph_rag: true
    graph_rag_budget: 3
  ```
- Check server logs for `GraphRAG: entity extraction returned 0 entities` — if present, see the section below.
- Try the REPL: `/rag graph-rag` to verify the flag is on at runtime.

---

## GraphRAG: Entity Graph Empty After Ingestion

**Symptom:** GraphRAG is enabled but retrieval does not expand with entity-connected documents. Logs show: `GraphRAG: entity extraction returned 0 entities across all chunks.`

**Cause:** The LLM failed to extract any entities. Common reasons:
- Model is too small or not instruction-tuned.
- The model returned bullets or lists despite the "no bullets" instruction (now stripped automatically), but returned empty output entirely.
- LLM request timed out during entity extraction.

**Fix:** Switch to a larger or more capable model. A 7B+ instruction-tuned model (e.g., `llama3.1:8b`, `mistral:7b`) reliably extracts entities.

---

## Provider Auto-Detection for vLLM / HuggingFace-style Model Names

**Symptom:** You set `llm_model: meta-llama/Llama-3.1-8B-Instruct` but the model is served via vLLM, not Ollama.

**Cause:** Axon infers `llm_provider` from the model name when the provider is not explicitly set. HuggingFace-style names (e.g., `org/model-name`) are inferred as `ollama` by default, since Ollama also accepts many such names.

**Fix:** Explicitly set the provider in `config.yaml`:
```yaml
llm:
  provider: vllm
  model: meta-llama/Llama-3.1-8B-Instruct
  base_url: http://localhost:8000/v1
```

---

## VS Code Extension: Tools Not Appearing in Copilot Chat

**Symptom:** After installing the VSIX, no `axon_*` tools appear in Copilot Chat.

**Cause:** Extension not loaded or VS Code not reloaded after install.

**Fix:**
1. Open Extensions panel (Ctrl+Shift+X) and confirm "Axon Copilot" shows as **enabled**.
2. Reload VS Code: Ctrl+Shift+P → "Reload Window".
3. Open Copilot Chat (Ctrl+Shift+I) — tools are registered on activation.

---

## VS Code Extension: Requests Fail with Connection Error

**Symptom:** Copilot tools return `Failed to fetch` or `ECONNREFUSED`.

**Fix:**
1. Confirm `axon-api` is running: `curl http://localhost:8000/health`
2. Check `axon.apiBase` in VS Code settings matches the server address exactly (default: `http://localhost:8000`).
3. On Windows, ensure the API is bound to `localhost`, not `0.0.0.0` — both should work from the same machine, but double-check `AXON_HOST` in `.env`.

---

## VS Code Extension: `axon_ingestPath` Stuck in "processing"

**Symptom:** After calling `axon_ingestPath`, the status never reaches `completed`.

**Cause:** The ingest job is async. Poll `axon_getIngestStatus(job_id)` until it returns `completed` or `failed`. Large directories (many files) may take minutes.

**Fix:** Ask Copilot to check the status: *"Check if my ingest job `<job_id>` is done"* — or wait and retry. If permanently stuck, check `axon-api` logs for errors.

---

## VS Code Extension: `autoStart` Does Not Start the Server

**Symptom:** The extension shows as active but `axon-api` is not running and `autoStart` is `true`.

**Cause:** Python executable not found. The extension discovers Python via (in order):
1. `axon.pythonPath` setting
2. `~/.axon/.python_path` (written by the `axon` CLI on first run)
3. `pipx` installation
4. Workspace virtual environment
5. System `python3` / `python`

**Fix:** Run `axon` once from the terminal (the CLI writes its Python path to `~/.axon/.python_path`), or set `axon.pythonPath` explicitly in VS Code settings.

---

## `top_k` and Raw Retrieval Count

**Symptom:** The API or internal retrieval returns more chunks than the configured `top_k` value.

**Cause:** When `hybrid_search` or `rerank` is enabled, Axon internally fetches `top_k × 3` candidates to allow for score merging and re-ranking. The final result passed to the LLM is capped at `top_k` after all processing. Internal retrieval methods (used in debugging or qualification scripts) may show the pre-cap candidate set.

**Guidance:** `top_k` controls how many chunks the LLM receives as context. The pre-cap overfetch is intentional and improves hybrid/reranked result quality.

---

## API: HTTP 429 Too Many Requests

**Error:**
```
Error: HTTP 429 Too Many Requests
{"detail": "Rate limit exceeded. Retry after 60s."}
```

**Cause:** Per-IP rate limiting is enforced on `/share/*`, `/ingest/*`, and `/security/*`
endpoints. The default limit is 10 requests per 60-second window. Bulk operations that
send many requests in a short period will trigger this limit.

**Fix:**
- Wait 60 seconds before retrying.
- For bulk text ingest, use `POST /add_texts` (batched JSON) to send multiple documents in one
  request. For file uploads, use `POST /ingest/upload` which accepts multiple files per request.
- To raise the default limits for trusted environments, increase them in `config.yaml`:
  ```yaml
  api:
    rate_limit_requests: 50      # requests per window (default: 10)
    rate_limit_window_seconds: 60
  ```

---

## API: HTTP 413 Request Entity Too Large

**Error:**
```
Error: HTTP 413 Request Entity Too Large
{"detail": "Upload exceeds maximum size of 500 MiB"}
```

**Cause:** The uploaded file or batch exceeds `max_upload_bytes` (default 500 MiB) or
`max_files_per_request` (default 1000 files). Both limits were added to harden the ingest
API against oversized payloads.

**Fix:**
- Split large batches into smaller ones below the 500 MiB threshold.
- To raise the limits for trusted environments, adjust them in `config.yaml`:
  ```yaml
  api:
    max_upload_bytes: 1073741824   # 1 GiB
    max_files_per_request: 5000
  ```

---

## API: HTTP 422 Unprocessable Entity on Long Queries

**Error:**
```
Error: HTTP 422 Unprocessable Entity
{"detail": [{"loc": ["body", "query"], "msg": "ensure this value has at most 4096 characters"}]}
```

**Cause:** The query text exceeds the `max_length` field validation added to the request
schema. This prevents excessively long strings from being passed through the pipeline.

**Fix:**
- Shorten the query to under 4096 characters, or rephrase it as a more focused question.
- For programmatic use cases that genuinely need longer input, increase the field limit
  in `config.yaml`:
  ```yaml
  api:
    max_query_length: 8192   # characters (default: 4096)
  ```

---

## `pip install axon[graphrag]` fails with `gensim` build error

**Error:**
```
error: metadata-generation-failed
...AttributeError: 'dict' object has no attribute '__NUMPY_SETUP__'
gensim
```

**Cause:** `graspologic` 0.3.x on PyPI depends on `gensim` 3.8.x, which cannot build against NumPy 2.x or Python 3.13. This is a known upstream incompatibility.

**Fix:** The `[graphrag]` extra no longer includes `graspologic`. It uses `leidenalg` + `igraph` instead, which ship pre-built wheels for all platforms and Python 3.13:

```bash
pip install -e ".[graphrag]"
# installs: networkx, leidenalg, igraph
```

The default is now `graph_rag_community_backend: louvain` (safe on all platforms). To upgrade to Leiden resolution-sweeping:

```yaml
rag:
  graph_rag_community_backend: louvain    # default — networkx only, no extra deps
  # graph_rag_community_backend: leidenalg  # recommended when igraph/leidenalg are installed
  # graph_rag_community_backend: auto       # graspologic → leidenalg → louvain fallback chain
  #                                          # (unsafe on Python 3.13 — graspologic import hangs)
```

If you have `graspologic` installed from a Python ≤ 3.12 / NumPy 1.x environment and want Axon to use it, set `graph_rag_community_backend: auto`.

---

## `pip install graspologic` fails on Python 3.13 / NumPy 2.x

**Cause:** `graspologic` 0.3.x depends on `gensim` 3.8.x, which fails to build on Python 3.13 or with NumPy 2.x due to an `AttributeError` involving `__NUMPY_SETUP__`. Additionally, `graspologic` requires `networkx < 3.0`, which conflicts with other modern dependencies in the Axon stack.

**Fix (Manual Patching):**
If you must use `graspologic` on Python 3.13:
1. Install dependencies manually: `pip install graspologic-native umap-learn gensim>=4.0`.
2. Install `graspologic` without dependencies: `pip install graspologic==0.3.1 --no-deps`.
3. Force modern NetworkX: `pip install "networkx>=3.0"`.
4. Patch the `graspologic` source to support NetworkX 3.x:
   ```python
   # Run this snippet to replace deprecated 'OrderedGraph' references
   import pathlib
   import site
   sp = pathlib.Path(site.getsitepackages()[0]) / "graspologic"
   for p in sp.rglob("*.py"):
       content = p.read_text(encoding="utf-8")
       new = content.replace("nx.OrderedGraph", "nx.Graph").replace("nx.OrderedDiGraph", "nx.DiGraph")
       if new != content:
           p.write_text(new, encoding="utf-8")
   ```
This resolves the `AttributeError: module 'networkx' has no attribute 'OrderedGraph'` error.

---

## `pre-commit install` fails with `core.hooksPath` error

**Error:**
```
[ERROR] Cowardly refusing to install hooks with `core.hooksPath` set.
hint: `git config --unset-all core.hooksPath`
```

**Cause:** Your Git configuration has an explicit `core.hooksPath` set (common in some managed environments or CI setups), which prevents `pre-commit` from installing its own hooks into `.git/hooks`.

**Fix:**
Unset the global or local hooks path, install, and then re-set if necessary:
```bash
git config --unset core.hooksPath
pre-commit install
```

---

## First ingest is very slow with RAPTOR + GraphRAG enabled

**Symptom:** Ingest of a 10–50 document corpus takes several minutes instead of seconds.

**Cause:** RAPTOR and GraphRAG are disabled in the shipped `config.yaml` but enabled in the code dataclass defaults — if you are hitting this, you have explicitly enabled them. RAPTOR makes ~1 LLM call per 5 chunks
(summary generation). GraphRAG makes ~1–3 LLM calls per chunk (entity extraction, optionally
relation extraction). For 100 chunks that is 100–300 LLM calls before any query can be answered.

**Mitigations (choose one or combine):**

1. **Use the light extraction tier** — no LLM calls for entity extraction, ~0 ms per chunk:
   ```yaml
   rag:
     graph_rag_depth: light
   ```

2. **Budget relation extraction** — the default ships with `graph_rag_relation_budget: 30`, which caps
   relation extraction to the 30 most entity-dense chunks per batch. Reduce further if needed:
   ```yaml
   rag:
     graph_rag_relation_budget: 15      # strict budget (0 = unlimited)
     graph_rag_min_entities_for_relations: 5  # also skip sparse chunks
   ```

3. **Prune singleton entities** — `graph_rag_entity_min_frequency: 2` (default) excludes entities that
   appear in only one chunk before community detection, reducing graph size and community count:
   ```yaml
   rag:
     graph_rag_entity_min_frequency: 3  # stricter: entities must appear in >= 3 chunks
   ```

4. **Limit RAPTOR to small sources** — skip RAPTOR for sources smaller than N MB:
   ```yaml
   rag:
     raptor_min_source_size_mb: 2.0
   ```

5. **Increase RAPTOR group size** — more chunks per summary = fewer summaries = fewer LLM calls:
   ```yaml
   rag:
     raptor_chunk_group_size: 10   # default is 5
   ```

6. **Disable both for bulk ingest**, then re-enable for daily use (dedup skips unchanged chunks):
   ```yaml
   rag:
     raptor: false
     graph_rag: false
   ```

---

## `ImportError: No module named 'gliner'`

**Cause:** `graph_rag_ner_backend: gliner` is set in `config.yaml` but the `gliner` package is not installed.

**Fix:**
```bash
pip install axon[gliner]
```

Or revert to the default LLM backend:
```yaml
rag:
  graph_rag_ner_backend: llm   # default — no extra install needed
```

---

## `ImportError: No module named 'transformers'` when using REBEL

**Cause:** `graph_rag_relation_backend: rebel` is set but the `transformers` package is not installed.

**Fix:**
```bash
pip install axon[rebel]
```

Or revert to the default LLM backend:
```yaml
rag:
  graph_rag_relation_backend: llm   # default
```

---

## Community generation hangs / takes very long on first global query

**Cause:** In lazy mode (`graph_rag_community_lazy: true`), community summaries are generated on
the first global query. If many communities exist and no pre-filter is applied, the LLM is called
once per community — this can be 50–200+ calls on large corpora.

**Mitigations:**

1. **Limit community summaries at query time** (pre-filters before LLM calls):
   ```yaml
   rag:
     graph_rag_global_top_communities: 10   # default is 20
   ```

2. **Reduce community depth** (fewer clusters = fewer summaries):
   ```yaml
   rag:
     graph_rag_community_levels: 1
   ```

3. **Pre-generate summaries at ingest time** instead of lazily:
   ```yaml
   rag:
     graph_rag_community_lazy: false   # generate immediately after ingest
   ```

4. **Use the `/finalize` command** (REPL) or `POST /graph/finalize` (API) after batch ingest
   to trigger community detection and summary generation before the first query.

---

## vLLM: `Connection refused` or `404 Not Found`

**Error:**
```
httpx.ConnectError: [Errno 111] Connection refused
```
or
```
openai.NotFoundError: 404 The model `mistral-7b-instruct` was not found
```

**Cause:** Either vLLM is not running, or the model name in `config.yaml` doesn't match what vLLM is serving.

**Fix:**

1. Confirm your vLLM server is running and healthy:
   ```bash
   curl http://localhost:8000/v1/models
   ```
   The response lists all served models.

2. Set `vllm_base_url` to match your vLLM server address and copy the exact model name from the `/v1/models` response:
   ```yaml
   llm:
     provider: vllm
     model: mistral-7b-instruct-v0.2   # must match exactly
     vllm_base_url: http://localhost:8000/v1
   ```

3. Change the URL at runtime from the REPL:
   ```
   axon> /vllm-url http://your-server:8000/v1
   ```

---

## Gemini: `API key not valid` or `RESOURCE_EXHAUSTED`

**Error:**
```
google.api_core.exceptions.PermissionDenied: 403 API key not valid.
```
or
```
google.api_core.exceptions.ResourceExhausted: 429 Quota exceeded
```

**Cause 1 — Invalid key:** The key is wrong, expired, or the Gemini API is not enabled for the project.

**Fix:**
```bash
export GEMINI_API_KEY=AIza...
# or in config.yaml:
# llm:
#   gemini_api_key: AIza...
```
Enable the Generative Language API in Google Cloud Console for the project that owns the key.

**Cause 2 — Quota exhausted:** You hit the free-tier rate limit (typically 15 requests/minute on the free plan).

**Fix:** Wait 60 seconds and retry, or upgrade to a paid Gemini API plan. Alternatively switch to a local model temporarily:
```
axon> /model llama3.1:8b
```

**Cause 3 — Gemma model + system prompt:** Gemma models (e.g. `gemma-3-27b-it`) don't support `system_instruction` in the Gemini SDK. Axon automatically falls back to prepending the system prompt to the first user message — no action needed, but if you see unexpected output check the model name is recognised as a Gemma variant.

## Share mount: putting a project under OneDrive / Dropbox / Google Drive

Axon's share-mount model assumes the owner's project directory sits on a **coherent filesystem** (local disk, or on-prem SMB3 from a Windows-native grantee). Consumer cloud-sync tools do **not** qualify:

- **SQLite WAL corrupts on cloud sync.** SQLite's own maintainers categorically forbid WAL mode on filesystems where advisory locks or shared-memory mappings cannot be replicated coherently (see https://sqlite.org/useovernet.html and https://sqlite.org/wal.html). OneDrive / Dropbox / Google Drive all fit this description.
- **Axon mitigations in the default install:** the governance audit DB (`.governance.db`) and the Dynamic Graph backend (`.dynamic_graph.db`) now use journal mode `DELETE` instead of `WAL`, so there are no `-wal`/`-shm` sidecars for sync clients to re-order. Grantees on a share mount never open the owner's `.dynamic_graph.db`; they read a read-only JSON snapshot (`.dynamic_graph.snapshot.json`) that the owner exports on every ingest.
- **What's still risky for plaintext sharing:** cloud-sync is not a supported live-storage layer for plaintext vector store binaries. Sync clients can delay, reorder, or partially publish binary index updates, making the index temporarily inconsistent during sync. **Sealed sharing** (AES-256-GCM encrypted) is the correct solution for cloud sync: only the encrypted bytes are on the cloud drive, and the grantee decrypts into a local ephemeral cache at query time. See [SHARING.md](SHARING.md#sealed-sharing-onedrive--dropbox--google-drive) for the setup walkthrough.

For the full supported / unsupported filesystem matrix and the per-backend recommendation table, see [SHARING.md](SHARING.md#filesystem-compatibility-matrix).
