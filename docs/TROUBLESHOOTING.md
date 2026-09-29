# Troubleshooting

Common problems and their fixes. Sealed-sharing errors have their own table in
[Sharing](SHARING.md#troubleshooting-sealed-sharing); every setting named here is described
in the [Reference](REFERENCE.md).

> **First, run `axon --doctor`.** It checks the Python version, that the store is
> writable (both required), and — as advisories — whether Ollama answers, whether the
> configured model is pulled, a `local` LLM endpoint, the recommended extras and newer
> releases. Each problem comes with a one-line next step. `axon --config-validate` checks
> `config.yaml` for unknown keys, bad values and risky combinations.

**Contents:** [Install and start-up](#install-and-start-up) ·
[LLM providers](#llm-providers) · [Ingest and retrieval](#ingest-and-retrieval) ·
[GraphRAG and RAPTOR](#graphrag-and-raptor) · [Vector stores](#vector-stores) ·
[REST API](#rest-api) · [VS Code and MCP](#vs-code-and-mcp) ·
[Sharing and sync folders](#sharing-and-sync-folders)

---

## Install and start-up

### `pip install axon-rag[...]` fails with "no matches found"

Your shell treats the square brackets as a glob. Quote the requirement:
`pip install "axon-rag[starter]"`.

### `axon: command not found`

The virtual environment you installed into isn't active. Activate it
(`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows), or call the full
path (`.venv/bin/axon`).

### The setup wizard didn't appear

Plain `axon` runs the wizard only when there is no config file *and* no projects yet. Any
command that loads the config (`axon --ingest …`, `axon --doctor`, a question) creates
`~/.config/axon/config.yaml`, after which the wizard runs only on request: `axon --setup`,
or `/config wizard` in the REPL.

### Encoding errors on Windows

Documents with non-ASCII text can fail to load under a legacy code page. Set
`PYTHONUTF8=1` before starting Axon (in PowerShell: `$env:PYTHONUTF8 = "1"`, or add it to
your profile). Windows Terminal renders the REPL best.

### `axon-api` won't start: port in use, or "already serving this store"

`Could not bind 0.0.0.0:8420 — the port is already in use` means another process holds the
port: `axon-api --port 9000` (or `AXON_PORT=9000`), and point clients at the new port. If
the message says another `axon-api` is already serving the store, you probably have a
server running already — use it, or stop it. `AXON_ALLOW_MULTIPLE_SERVERS=1` overrides the
guard, but two servers writing one store is how index files get damaged
([below](#turboquantdb-queries-crash-the-process-or-errno-22-invalid-argument-on-ingest)).

### `axon-api` or the VS Code extension is slow to start

Through 0.4.5 the default embedder (`sentence_transformers`) imported PyTorch at every
start — 15–20 seconds. Since 0.4.6 the default is `fastembed` (ONNX, same vectors for
`all-MiniLM-L6-v2`, no re-ingest needed) and start-up takes about two seconds. If an older
config still says `sentence_transformers`, switch it:

```yaml
embedding:
  provider: fastembed
  model: sentence-transformers/all-MiniLM-L6-v2
```

The first run after install also downloads the embedding model (about 90 MB). If start-up
is still slow, antivirus scanning the virtual environment on first touch is the usual
cause — exclude its `site-packages` from real-time scanning.

### `--non-interactive` still opens the REPL

In this release `--non-interactive` only skips the first-run wizard and the start-up
animation; `axon --ingest DIR` still opens the REPL afterwards. In scripts, pass a
question, use a one-shot flag such as `--list`, or redirect stdin (`< /dev/null`).

---

## LLM providers

### Ollama isn't running when you ask

**Symptom:** the REPL opens, but the first question hangs or fails with
`httpx.ConnectError` / `Connection refused`. Axon starts without Ollama and only needs it
when it calls the model.

**Fix:** open the Ollama app or run `ollama serve`; check with
`curl http://localhost:11434` (`Ollama is running`) and `ollama list`. On another host or
port, set `OLLAMA_HOST` or `llm.base_url`.

### The model isn't pulled

With `provider: ollama`, asking a question or opening the REPL pulls a missing model
automatically when Ollama is reachable (`Model '…' not found locally — pulling from
Ollama...`; `llama3.1:8b` is about 4.7 GB). To do it yourself: `ollama pull llama3.1:8b`,
`axon --pull llama3.1:8b` or `/pull` in the REPL.

### Answers are empty, or the model forgets the conversation

- **Reasoning models** (Gemma 4, GPT-OSS, DeepSeek-R1 derivatives) spend tokens on hidden
  reasoning before answering. `llm.max_tokens` defaults to 8192; raise it before
  suspecting retrieval.
- **Ollama context:** Axon asks Ollama for an 8192-token context window on every call, so
  history is not silently truncated at Ollama's 2048 default. Very long sessions can still
  overflow — `/compact` summarises the history.
- **Small models** (1–3B) often ignore instructions, which degrades multi-query,
  citations, GraphRAG extraction and RAPTOR summaries. Use a 7B+ instruction-tuned model,
  or turn those features off (`/rag multi`, `/rag cite`).

### The wrong provider is used

A bare `--model` or `/model` name picks the provider: `gemini-*` → Gemini,
`gpt-*` / `o1-*` / `o3-*` / `o4-*` → OpenAI, anything else → Ollama — even when you also
passed `--provider`. HuggingFace-style names such as `meta-llama/Llama-3.1-8B-Instruct`
therefore go to Ollama. Prefix the provider (`--model vllm/meta-llama/Llama-3.1-8B-Instruct`)
or set it in `config.yaml`:

```yaml
llm:
  provider: vllm
  model: meta-llama/Llama-3.1-8B-Instruct
  vllm_base_url: http://localhost:8000/v1
```

For `grok` and `copilot`, set `llm.provider` in `config.yaml` (the prefix form doesn't
accept them).

### vLLM: `Connection refused` or `404 … model … was not found`

Check the vLLM server itself: `curl http://localhost:8000/v1/models` lists what it serves.
Copy the exact model id into `llm.model`, make `llm.vllm_base_url` match the server, or
change it live with `/vllm-url http://host:8000/v1`.

### `local` provider: empty answers, timeouts, "unreachable"

For llama.cpp's `llama-server`, LM Studio, TGI, LocalAI or any other OpenAI-compatible
server. Axon never starts or loads models for you; check the endpoint with
`axon --doctor` (the *Local LLM endpoint* line) or `/local-url ping`. *Reachable but no
models* is a real state — a router-mode `llama-server` answers before a model is resident;
load one and retry.

- **`APITimeoutError` on `step_back` / `query_decompose`:** these make several sequential
  calls. `llm.timeout` is 300 s for `local`; raise it or use a faster model. It is a
  per-read bound — every streamed chunk resets it — so cap `llm.max_tokens` for a hard
  limit.
- **Ingest hangs with `graph_rag: true`:** `graph_rag_depth: standard` makes an LLM call
  per chunk, which can take minutes each on a slow model. Use `graph_rag_depth: light` (no
  LLM calls) and keep `graph_rag_community: false`. `axon --config-validate` warns about
  this combination.
- **Port clash:** `local_base_url` defaults to `:8080`, `axon-api` to `:8420`. Move
  whichever side conflicts (`axon-api --port`, `/local-url`).

### Gemini: `API key not valid` or `RESOURCE_EXHAUSTED`

A `403 API key not valid` means a wrong or expired key, or the Generative Language API
isn't enabled for the key's Google Cloud project — set `GEMINI_API_KEY` or
`llm.gemini_api_key`. `429 … Quota exceeded` is the rate limit; wait, upgrade the plan, or
switch model for a while (`/model llama3.1:8b`). Gemma models don't accept a system
instruction; Axon folds the system prompt into the first user message automatically.

### Cloud keys aren't picked up

Keys come from `config.yaml`, the environment, a `.env` file in the working directory, or
`~/.axon/.env` (where the REPL's `/keys set <provider>` saves them). `/keys` shows which
are set. `api.key` in `config.yaml` is only a legacy alias for the OpenAI key.

---

## Ingest and retrieval

### `axon --dry-run` prints 0 chunks

In this release the `--dry-run` CLI flag also replaces query embeddings with zero vectors,
so every dense score is 0 and the default `similarity_threshold` (0.3) filters everything.
Add `--threshold 0` to see the ranking (it will be driven by BM25), or check retrieval
without an LLM through `axon-api`: `POST /search`, or `POST /query` with `"dry_run": true`,
which embed normally. `axon --list` confirms what was ingested.

### A question finds nothing although the document is there

- Check `axon --list` and that you are in the right project (`/project list`).
- `similarity_threshold` (0.3) is compared against the dense cosine score; very short or
  oddly worded questions can fall below it. Try `--threshold 0.2` or `/rag threshold 0.2`.
- Changed the embedding model since ingesting? Old vectors don't match — re-ingest.
- The query router applies a profile per question and can turn your transformations off
  for simple lookups; `rag.query_router: "off"` uses your settings exactly.

### Re-ingesting deleted text does nothing

**Symptom:** you deleted a document and ingested the same text again; the call succeeds
but the text never shows up, and the log says `Dedup: skipped N already-seen chunk(s)`.

Through 0.4.6, deleting through `POST /delete` (and so `delete_documents`) removed the
chunks but not their hashes. Current releases clear them. For text deleted on an older
release, either delete it again after upgrading (if any chunk remains), ingest once with
`rag.dedup_on_ingest: false` (or `axon --ingest … --no-dedup`), or `axon --clear --yes` and
re-ingest. Chunks ingested with `contextual_retrieval: true` on 0.4.6 or earlier need the
dedup-off route even after upgrading.

### `top_k` looks ignored in raw results

With hybrid search or reranking Axon fetches about three times `top_k` candidates to fuse
and rerank; the LLM still receives `top_k`. Diagnostic and raw views can show the larger
candidate set.

### `403 … outside the allowed ingest directory`

`POST /ingest` (and `ingest_knowledge(path=...)`) only read under `RAG_INGEST_BASE`, which
defaults to the directory `axon-api` was started in. Start the server with
`RAG_INGEST_BASE=/path/to/docs` (VS Code: the `axon.ingestBase` setting), or upload the
files with `POST /ingest/upload`.

### Web search is on but no web results appear

Web fallback needs a Brave key (`BRAVE_API_KEY`, `web_search.brave_api_key`, or
`/keys set brave` in the REPL) and `web_search.enabled: true`. Without CRAG-Lite it fires
only when local retrieval returns nothing; with `rag.crag_lite: true` it also fires on
low-confidence results. Brave failures fall back quietly — look for `BraveSearch error` in
the server log. Too many web answers? Lower `crag_lite_confidence_threshold` (e.g. to 0.2)
or turn CRAG-Lite off. `429` from Brave means the monthly quota is used up.

---

## GraphRAG and RAPTOR

### The first ingest is very slow

RAPTOR and GraphRAG are off in the shipped `config.yaml`; if ingest is slow you turned
them on (or built `AxonConfig()` in Python, where they default on). RAPTOR makes about one
LLM call per five chunks; GraphRAG `standard` about one to three per chunk. Options:

```yaml
rag:
  graph_rag_depth: light                 # regex extraction, no LLM calls
  graph_rag_relation_budget: 15          # relation extraction for the 15 densest chunks per batch
  graph_rag_min_entities_for_relations: 5
  graph_rag_entity_min_frequency: 3      # prune rare entities before community detection
  raptor_min_source_size_mb: 2.0         # RAPTOR only for sources above 2 MB
  raptor_chunk_group_size: 10            # fewer, larger summaries
```

Or turn both off for the bulk ingest and on again afterwards — dedup skips unchanged
chunks.

### The entity graph is empty, or GraphRAG adds nothing

- The log says `GraphRAG: entity extraction returned 0 entities` → the model didn't follow
  the extraction instructions. Use a 7B+ instruction-tuned model, or `graph_rag_depth:
  light`.
- `graph_rag_budget: 0` removes the guaranteed graph slots; the default is 3.
- The question uses different words from the extracted entities.
- The graph is built at ingest: documents ingested before you turned `graph_rag` on have
  no entities — re-ingest them (with `--no-dedup`).
- Check with `axon --graph-status` / `/graph status`, and that the flag is on (`/rag`).

### Community summaries hang on the first global question

With `graph_rag_community_lazy: true` summaries are written on the first `global` / `hybrid`
query — one LLM call per community, which can be hundreds. Cap them with
`graph_rag_global_top_communities: 10`, reduce `graph_rag_community_levels`, set
`graph_rag_community_lazy: false` to build them during ingest, or run
`axon --graph-finalize` (`/graph finalize`, `POST /graph/finalize`) after a batch ingest.

### `pip install "axon-rag[graphrag]"` and graspologic

The `graphrag` extra installs `networkx`, `leidenalg` and `igraph`, which have wheels for
Python 3.13. It does not install `graspologic`, whose 0.3.x releases need `gensim` 3.8 and
fail to build on Python 3.13 / NumPy 2 (`'dict' object has no attribute
'__NUMPY_SETUP__'`). Use `graph_rag_community_backend: louvain` (default) or `leidenalg`;
`auto` tries graspologic first and can hang on Python 3.13.

### `No module named 'gliner'` / `'transformers'`

`graph_rag_ner_backend: gliner` needs `pip install "axon-rag[gliner]"`;
`graph_rag_relation_backend: rebel` needs `pip install "axon-rag[rebel]"`. Or set both back
to `llm`.

---

## Vector stores

### TurboQuantDB: queries crash the process, or `[Errno 22] Invalid argument` on ingest

**Symptoms**, on a store that used to work:

```
pyo3_runtime.PanicException: ...            # kills the process outright
{"detail": "[Errno 22] Invalid argument"}   # POST /add_text, POST /ingest
OSError: [WinError 1224] The requested operation cannot be performed
                         on a file with a user-mapped section open
```

**Cause:** a bug in TurboQuantDB before **0.8.5**
([tqdb#102](https://github.com/jyunming/TurboQuantDB/issues/102)): `close()` didn't release
the memory map, so a later resize of the codes file failed (Windows refuses to grow or
truncate a mapped file), leaving `live_codes.bin` truncated and later reads panicking in
Rust. PyO3's `PanicException` isn't an `Exception`, so the process dies instead of
degrading. Two processes on one store — say an old `axon-api` on port 8000 next to a new
one on 8420 — make it far likelier.

**Fix:**

```bash
pip install -U "tqdb>=0.9.1"
python -c "import tqdb; print(tqdb.__version__)"
```

Current Axon requires that floor. **Every machine sharing a store must be on the same tqdb
minor:** 0.9 reads what 0.8 wrote, but 0.8 cannot read what 0.9 wrote (it fails with the
same end-of-file error).

**If a store still won't open**, upgrading alone doesn't repair it. Rebuild it from the
chunk text Axon keeps in `bm25_index/` — no source files are re-read, no LLM calls run, the
graph is untouched, and the old `vector_store_data/` is renamed aside (restored
automatically if the rebuild fails):

```bash
axon --rebuild-vector-store --rebuild-dry-run    # what would be re-embedded
axon --rebuild-vector-store
axon --project myproj --rebuild-vector-store     # another project
```

Until then Axon keeps running: retrieval returns nothing and says why, and ingest is
refused rather than starting a fresh store over unreadable files. The rebuild reports
chunks that share an id (indexed once; the rest stay keyword-searchable) and `os error
1224`, which means another process — usually a running `axon-api` — holds the store; stop
it first.

**Prevention:** one server per store.

### Chroma: `InvalidDimensionException`

```
chromadb.errors.InvalidDimensionException: Embedding dimension 384 does not match collection dimensionality 768
```

The embedding model changed after the collection was created. Switch back, or start over
for that project: back up the store, then `axon --clear --yes` (or create a new project)
and re-ingest. There is no migration between embedding dimensions. With Docker, stop the
containers before deleting data directories — deleting under a running Chroma produces
`Could not connect to tenant default_tenant`; `docker compose down`, remove the data,
`docker compose up -d`.

---

## REST API

### `401 Invalid or missing X-API-Key header`

The server has `RAG_API_KEY` set. Send `X-API-Key: <key>`; for MCP set `RAG_API_KEY` in the
client's env block, for VS Code the `axon.apiKey` setting.

### `409` project mismatch

A `project` field is an assertion: the server is serving another project. Switch first
(`POST /project/switch`) or drop the field. The `detail` names the active project.

### `429 Too Many Requests`

Per-IP limits per 60 seconds: `/ingest_url` 20, `/ingest/upload` 30, `/share/generate`
and `/share/redeem` 10, `/security/bootstrap` and `/security/change-passphrase` 10, plus a
failed-attempt limit on `/security/unlock`. They are fixed. Batch instead: `POST /add_texts`
for many texts, several files per `/ingest/upload`, or `POST /ingest` for a directory on
the server.

### `413` / `422` on uploads and long inputs

`413` — a file above `api.max_upload_bytes` (500 MiB); `422` — more files than
`api.max_files_per_request` (1000). Both are `api:` settings in `config.yaml`. A `query`
longer than 8192 characters is also a `422` and is not configurable — ask a shorter
question and put the long text into the knowledge base.

### `504 Query timed out`

`/query` gives up after `AXON_QUERY_TIMEOUT` seconds (120), or the request's `timeout`.
Turn off HyDE / reranking / decomposition for that request, or raise the limit.

### `503` with `X-Axon-Mount-Sync-Pending: true`

The active project is a mounted share whose owner re-ingested and whose files are still
syncing. Retry after the sync client catches up.

---

## VS Code and MCP

### Copilot doesn't show Axon's tools

Confirm the extension (*Axon Copilot*) is enabled, run *Reload Window*, and open Copilot
Chat — the tools register on activation. For MCP hosts, check the server is configured
(`claude mcp list`, `/tools` in Codex) and that `axon-api` is running:
`curl http://localhost:8420/health`.

### Tools fail with `Failed to fetch` / `ECONNREFUSED`

`axon-api` isn't running or isn't where the client looks. Check
`curl http://localhost:8420/health`; set `axon.apiBase` (VS Code) or `RAG_API_BASE` (MCP)
to the server's URL — `http://localhost:8420`, not `http://0.0.0.0:8420`.

### `autoStart` doesn't start the server

The extension couldn't find a Python with Axon installed. Run `axon` once from the
environment you installed into (it writes `~/.axon/.python_path`), or set
`axon.pythonPath`. Auto-start is for Linux and macOS; on Windows start `axon-api` yourself.

### A path ingest never finishes

`ingest_knowledge(path=...)` and `refresh` are asynchronous: poll `get_job_status` with the
returned `job_id` until `completed` or `failed`. Large directories take minutes; if it never
moves, read the `axon-api` log.

### Image ingest fails with "model does not support images"

`ingest_image` needs a Copilot model with vision (GPT-4o, Claude Sonnet / Opus). Pick one in
Copilot Chat's model selector, or pass `alt_text` to supply the description yourself.

### An agent can't clear, delete or seal

By design. Destructive, credential and store-administration operations are human-only
since 0.5.0; the [Reference](REFERENCE.md#92-what-agents-can-and-cannot-do) lists where a
person does each one.

---

## Sharing and sync folders

### `/project mounts/…` says "Unknown sub-command"

The REPL needs `switch`: `/project switch mounts/owner_research`. On the CLI it is
`axon --project mounts/owner_research`.

### Sealing or a sealed share fails with "Store … is locked"

The master key is unlocked per process, and each `axon --…` command is its own process.
Unlock and seal / share in one REPL session (`/store unlock <passphrase>`, then
`/project seal …`, `/share generate …`), or unlock a running `axon-api` with
`POST /security/unlock`. See [Sharing](SHARING.md#owner-1).

### A project in OneDrive / Dropbox / Google Drive

Consumer sync clients are not a safe live store for plaintext projects: SQLite forbids its
WAL mode on filesystems that can't replicate locks and shared memory
([sqlite.org/useovernet.html](https://sqlite.org/useovernet.html)), and sync clients delay,
reorder or half-publish binary index updates. Axon's `dynamic_graph` backend avoids WAL and
grantees read a JSON snapshot instead of the database, but the vector-store files remain
exposed. Use **sealed** sharing for any cloud-synced folder — only ciphertext is synced and
the grantee decrypts into a local cache: [setup](SHARING.md#sealed-sharing-onedrive--dropbox--google-drive),
[filesystem matrix](SHARING.md#filesystem-compatibility-matrix). `axon --config-validate`
warns when the store sits on a sync or network path.

### A shared project disappeared, or says revoked / expired / unverifiable

See [How share validity is decided](SHARING.md#how-share-validity-is-decided):
`unverifiable` means the owner's record can't be read yet (offline, sync incomplete) and
fixes itself; `revoked` and `expired` need a new share from the owner.
