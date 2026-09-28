# Axon MCP Tools Reference

Axon exposes a Model Context Protocol (MCP) server (`axon-mcp`) with **18 tools**.

> **Which integration should I use?**
> - **`@axon` chat participant** — install the VS Code extension (VSIX). Gives you a conversational `@axon` inside Copilot Chat. No `.vscode/mcp.json` needed.
> - **MCP tools** (this reference) — configure `axon-mcp` in your editor or agent CLI. Gives programmatic tool access to Claude Code, Codex CLI, Gemini CLI, Cursor, and VS Code Copilot agent mode.

`axon-api` must be running before any MCP client connects. See [SETUP.md § 10](SETUP.md#10-mcp-server-setup) for connection instructions per client.

The VS Code extension's Copilot LM tools are the same 18 tools with the same names and
parameters, plus two client-side tools: `show_graph` (opens the graph panel) and
`ingest_image` (describes an image with a Copilot vision model, then ingests the text).

### What agents can and cannot do (0.5.0)

The tool set covers what an agent needs to *use* a knowledge base: ask, search, ingest,
inspect, delete individual documents, pick or create a project, read and tune config, read
and write the graph, and share projects. Operations that are destructive, handle
credentials, or administer the store are **human-only** — they stay on the REST API, CLI
and REPL (and, where noted, VS Code commands), not on MCP:

| Human-only operation | Where a human does it |
|---|---|
| Wipe a project's knowledge base | REPL `/clear`, `axon --clear --yes`, `POST /clear`, VS Code command *Axon: Clear Knowledge Base* |
| Delete a project | REPL `/project delete <name>`, `axon --project-delete <name>`, `POST /project/delete/{name}` |
| Hard-revoke a sealed share (rotate the project key) | REPL `/share revoke <ssk_id> --project <name> --rotate`, `axon --share-rotate`, `POST /share/revoke {"rotate": true}` |
| Sealed store: bootstrap / unlock / lock / change passphrase / keyring mode / wipe cache | REPL `/store ...`, `axon --store-*`, `POST /security/*` |
| Seal, pack, unpack a project | REPL `/project seal|pack|unpack`, `axon --project-seal|--project-pack|--project-unpack`, `POST /project/seal|pack|unpack` |
| Init the store, store status / whoami | REPL `/store init|whoami`, `axon --store-init|--store-whoami`, `POST /store/init`, `GET /store/status|whoami` |
| Refresh a mounted share | REPL `/mount-refresh`, `axon --mount-refresh`, `POST /mount/refresh` |
| Stale-doc listing | REPL `/stale`, `axon --list-stale`, `GET /collection/stale` |
| Chat sessions | REPL `/sessions`, `axon --session-list`, `GET /sessions` |
| Graph status / finalize / conflicts / full dump | REPL `/graph status|finalize|conflicts`, `axon --graph-status|--graph-finalize|--graph-conflicts`, `GET /graph/status`, `POST /graph/finalize`, `GET /graph/conflicts`, `GET /graph/data` |
| Write-lease diagnostics | `GET /registry/leases` |

`src/axon/surface_contract.py` records the human route for every capability kept off the
agent surfaces; `tests/test_surface_parity_contract.py` keeps this tool list, the registry
and the VS Code manifest in lockstep.

### `project` is an assertion

Tools that accept `project` treat it as an **assertion**, not a switch: if the server is
serving a different project the call fails with **409** and nothing happens. To work in
another project, call `switch_project` first.

---

## Retrieval (2)

### `query_knowledge`

Full RAG query — retrieval + generation in one call. Use `search_knowledge` if you need to inspect chunks first.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | The question to answer |
| `top_k` | int | `null` | Chunks to retrieve (≥ 1); null uses the configured default |
| `filters` | dict | `null` | Optional metadata filters for retrieval |
| `project` | string | `null` | Expected active project (assertion — 409 on mismatch) |

**Returns:** `{"response": "...", "provenance": {"answer_source": "...", "retrieved_count": N}, "settings": {...}, "sources": [...], "citations": [...]}`

The `sources` array (slim view of every retrieved chunk made available to the LLM, indexed 0..N-1) and `citations` array (structured spans parsed from the response, one per `[N]` / `[Document N]` marker, with character offsets) are returned by default.

### `search_knowledge`

Retrieve raw document chunks with scores. Best for multi-step reasoning where you want to inspect chunks before synthesising an answer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | The search query |
| `top_k` | int | `5` | Number of chunks to return (must be ≥ 1) |
| `filters` | dict | `null` | Optional metadata filters, e.g. `{"source": "https://..."}` |
| `project` | string | `null` | Expected active project (assertion — 409 on mismatch) |

**Returns:** `[{"text": "...", "score": 0.87, "metadata": {...}}]`

---

## Ingest (2)

### `ingest_knowledge`

Add knowledge to the active project. Give **exactly one** source — zero or several raise an
error before any request is sent.

| Source parameter | Type | Route | Behaviour |
|---|---|---|---|
| `text` | string | `POST /add_text` | One document, synchronous. Set `metadata.source` so the collection can be audited. |
| `docs` | `[{text, doc_id?, metadata?}]` | `POST /add_texts` | Many documents in one batched embedding call — prefer this over repeated `text` calls. |
| `url` | string | `POST /ingest_url` | Fetch an HTTP/HTTPS page; HTML is stripped. Private/internal addresses are blocked server-side. Synchronous. |
| `path` | string | `POST /ingest` | A file or directory on the machine running `axon-api`, within `RAG_INGEST_BASE`. **Async** — returns a `job_id`. |
| `refresh` | bool | `POST /ingest/refresh` | `true` re-ingests previously indexed files whose content changed on disk. **Async** — returns a `job_id`. |

| Other parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metadata` | dict | `null` | Metadata for `text` / `url` ingest |
| `doc_id` | string | `null` | Stable ID for `text` ingest (`delete_documents` accepts it) |
| `project` | string | `null` | Expected active project (assertion — 409 on mismatch). Sent with every source, including `path` and `refresh`. |

Duplicate content (same SHA-256) is skipped with status `skipped`.

**Returns:** the route's response — e.g. `{"status": "success", "doc_id": "..."}` for `text`,
`{"job_id": "...", "status": "processing"}` for `path` / `refresh`.

> Replaces `ingest_text`, `ingest_texts`, `ingest_url`, `ingest_path` and `refresh_ingest`.
> The old `refresh_ingest(project=...)` silently switched the server's active project; the
> `project` parameter is now an assertion like everywhere else.

### `get_job_status`

Poll an async ingest job started by `ingest_knowledge(path=...)` or `ingest_knowledge(refresh=True)`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `job_id` | string | required | Job ID returned by `ingest_knowledge` |

**Returns:** `{"job_id": "...", "status": "processing|completed|failed", "started_at": "...", "completed_at": "...", "error": null, ...}`

---

## Collection (2)

### `list_knowledge`

List indexed sources with chunk counts for the active project. No parameters.

**Returns:** `{"total_files": N, "total_chunks": N, "files": [{"source": "file.md", "chunks": 12}]}`

### `delete_documents`

Remove chunks or whole documents from the index. Also clears their dedup records, so the same text can be ingested again afterwards.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `doc_ids` | `[string]` | required | Chunk IDs, or document IDs (the `doc_id` given to `ingest_knowledge`); a document ID deletes all its chunks |

**Returns:** `{"status": "success", "deleted": N, "doc_ids": [<chunk ids deleted>], "not_found": [...]}`

---

## Projects (3)

### `list_projects`

List all local projects and mounted shares. No parameters.

**Returns:** `{"projects": [{"name": "...", "graph_backend": "graphrag"|"dynamic_graph"|"none", ...}], ...}`

### `switch_project`

Switch the active project. **Warning:** mutates global server state — every later call, from any client, runs against the new project.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project_name` | string | required | Name of the project to activate (e.g. `mounts/alice_research` for a redeemed share) |

**Returns:** `{"status": "success", "active_project": "..."}`

### `create_project`

Create a new named project with an isolated knowledge base.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | required | Project name (max 5 slash-separated segments) |
| `description` | string | `""` | Optional human-readable description |
| `graph_backend` | string | `"graphrag"` | `graphrag`, `dynamic_graph`, or `none`. Immutable once set. Use `dynamic_graph` for a project agents will write facts into with `update_fact`. |

**Returns:** `{"status": "success", "project": "..."}`

---

## Configuration (2)

### `get_config`

Return the active configuration. Secrets are masked as `***` by the server, so this tool can never read a credential back out. Use it to discover the exact field names `set_config` accepts.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `validate` | bool | `false` | Also check `config.yaml` for errors, unknown keys and risky combinations (e.g. GraphRAG with a slow local LLM) |

**Returns:** the config as JSON (sensitive fields masked). With `validate=true`: `{"config": {...}, "validation": {"valid": bool, "issue_count": N, "issues": [...]}}`.

> Replaces `get_current_settings` and `validate_config`.

### `set_config`

Set one or more configuration fields in a single call (`POST /config/set` batch form).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `settings` | dict | required | `{key: value}`. A key is a dot-notation alias (`chunk.strategy`, `llm.model`, `rag.top_k`) **or** any `AxonConfig` field name (`graph_rag_depth`, `chunk_size`, `hybrid_search`) |
| `persist` | bool | `false` | Also write the changes to `config.yaml` so they survive a restart |

Every key is resolved before anything is applied: one unknown key rejects the whole batch
(400, listing every unknown key) and nothing changes. Changing `llm_provider` / `llm_model` /
`embedding_*` / `rerank` reinitialises the affected component once; if that fails (e.g. a
provider whose extra isn't installed) every key is rolled back, nothing is saved, and the
call fails with the reason. Switching the embedding
model invalidates existing vectors — re-ingest afterwards.

**Returns:** `{"status": "success", "applied": [{"key", "flat_key", "old_value", "new_value"}], "persisted": bool}` with secrets masked.

> Replaces `update_settings`, `update_config` and the single-key `set_config(key, value)`.
> (REST `POST /config/set` still accepts the single-key body `{key, value, persist}`.)

---

## Graph (2)

### `graph_retrieve`

Run the active graph backend's `retrieve()` directly and return graph contexts only — no LLM call. Surfaces point-in-time historical queries (`point_in_time`) and per-query federation weight overrides (`federation_weights`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | required | The query string |
| `top_k` | int \| null | `null` | Maximum graph contexts to return (server resolves to 10 when null; range 1-200) |
| `point_in_time` | string | `null` | ISO-8601 timestamp; return facts valid at that instant. Honoured only by bi-temporal backends (`dynamic_graph`); ignored elsewhere |
| `federation_weights` | dict[str, float] | `null` | Per-query RRF weights for the federated backend. Keys: `graphrag`, `dynamic_graph`. Ignored by other backends |
| `project` | string | `null` | Expected active project (assertion — 409 on mismatch) |

**Returns:** `{"backend": "...", "contexts": [{"context_id", "context_type", "text", "score", "rank", "valid_at", "invalid_at", "matched_entity_names", "hop_count", ...}, ...]}`

### `update_fact`

Assert or correct one fact `(subject, relation, object)` in the active project's graph (`POST /graph/facts`). History is bi-temporal: a replaced fact is superseded, not deleted, so `graph_retrieve(point_in_time=...)` still sees it. Only `dynamic_graph` and `federated` projects store facts; `graphrag` and `none` answer `status: "not_applicable"`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `subject` | string | required | Subject entity (1-200 chars) |
| `relation` | string | required | Relation, e.g. `WORKS_FOR` or `works for` (normalised to upper snake case) |
| `object` | string | required | Object entity (1-200 chars) |
| `description` | string | `""` | Optional note stored with the fact |
| `confidence` | float | `1.0` | 0.0-1.0 |
| `replace` | bool \| null | `null` | `true` = make this the only current fact for (subject, relation); `false` = add alongside; `null` = replace for exclusive relations (`IS_CEO_OF`, `MARRIED_TO`, ...), add otherwise |
| `project` | string | `null` | Expected active project (assertion — 409 on mismatch) |

**Returns:** `{"status": "created"|"superseded"|"unchanged"|"conflicted"|"not_applicable", "backend_id": "...", "fact_id": "...", "superseded_ids": [...], "conflicted_ids": [...], "detail": "..."}`

---

## Sharing (5)

Sharing stays agent-callable. Shares are read-only; the store must be initialised (`axon --store-init`) first.

### `share_project`

Generate a share key allowing another user to read one of your projects. Send the returned `share_string` to the recipient out-of-band. Sealed projects automatically get a `SEALED2:` envelope (the sealed store must be unlocked by the user first, else 409).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | string | required | Project to share (must exist) |
| `grantee` | string | required | OS username of the recipient |
| `ttl_days` | int | `null` | Optional positive number of days until expiry; `null` means no expiry. Honoured for plaintext (`sk_`) and sealed (`ssk_`) shares. `ttl_days <= 0` is rejected. |

**Returns:** `{"share_string": "...", "key_id": "...", "expires_at": "..."}` (`expires_at` present when `ttl_days` was set).

### `redeem_share`

Mount a shared project using a share string (read-only). After redemption the project appears as `mounts/{owner}_{project}`; `switch_project` to it to query. Sealed envelopes are detected automatically and their key is stored in the OS keyring.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `share_string` | string | required | Full share string from `share_project` |

**Returns:** `{"owner": "...", "project": "...", "mount_name": "..."}`

### `list_shares`

List outgoing shares (projects this user has shared, with revocation status) and incoming shares (projects shared with this user, with mount names). No parameters.

**Returns:** `{"sharing": [...], "shared": [...]}`

### `revoke_share`

Soft-revoke a previously generated share key. Plaintext shares (`sk_`): access ends on the grantee's next project-list or switch. Sealed shares (`ssk_`): the key wrap is deleted so the share can't be redeemed any more; a grantee who already redeemed keeps the key they cached.

Hard revoke — rotating the project's key and re-encrypting it, which invalidates **every** share — is human-only (see the table above). This tool has no `rotate` parameter.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key_id` | string | required | Key ID from `list_shares` |
| `project` | string | `null` | Project name — **required** for sealed (`ssk_`) shares, ignored for plaintext ones |

**Returns:** `{"key_id": "...", "grantee": "...", "project": "...", ...}`

### `extend_share`

Renew a share key's expiry, or clear it (`ttl_days=null`). **Plaintext shares (`sk_`) only** — sealed (`ssk_`) expiry lives in an Ed25519-signed sidecar that can't be re-signed in place; mint a fresh sealed share and revoke the old one instead.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `key_id` | string | required | Key ID of the share to extend (from `list_shares`) |
| `ttl_days` | int | `null` | New time-to-live in days from now; `null` clears the expiry entirely |

**Returns:** `{"key_id": "...", "expires_at": "..."}`

---

## Usage Notes

- `ingest_knowledge(path=...)` and `ingest_knowledge(refresh=True)` are async — poll `get_job_status` until `status` is `completed` or `failed`. `text`, `docs` and `url` ingest are synchronous.
- Mounted shares (via `redeem_share`) are always **read-only**. Ingest, delete and `update_fact` against a mount return 403.
- `set_config` defaults to `persist=false` — changes apply to the running server only. Pass `persist=true` to write `config.yaml`.
- Every request carries `X-Axon-Surface: mcp` for attribution; set `RAG_API_KEY` if the API requires a key.
- A failed call raises a tool error whose message includes the HTTP status and the server's `detail` — e.g. the unknown config keys, the project the server is actually serving on a 409, or "project is required" for a sealed revoke.

### Removed in 0.5.0

56 tools at the start of 0.5.0 → 18. Consolidated: `ingest_text`, `ingest_texts`, `ingest_url`, `ingest_path`, `refresh_ingest` → `ingest_knowledge`; `get_current_settings`, `validate_config` → `get_config`; `update_settings`, `update_config` → `set_config`. Removed (human-only, see the table above): `clear_knowledge`, `delete_project`, `get_stale_docs`, `get_active_leases`, `query_stream`, `list_sessions`, `get_session`, `get_store_status`, `init_store`, `security_status`, `security_bootstrap`, `security_unlock`, `security_lock`, `security_change_passphrase`, `suggest_passphrase`, `set_keyring_mode`, `wipe_sealed_cache`, `seal_project`, `pack_project`, `unpack_project`, `mount_refresh`, `graph_status`, `graph_finalize`, `graph_data`, `graph_backend_status`, `graph_conflicts`, and the five `governance_*` tools (removed with the governance console).
