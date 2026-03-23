# Axon MCP Tools Reference

Axon exposes a Model Context Protocol (MCP) server with 24 registered tools.

> **Which integration should I use?**
> - **VS Code users** (most users): install the Axon VS Code extension — the `axon_*` toolset appears in Copilot automatically. No `.vscode/mcp.json` required.
> - **Copilot agent mode / non-VS Code hosts** (advanced): configure MCP as described in [SETUP.md § 10](SETUP.md).
> - **Claude Code**: use MCP (this reference).

When Axon is connected as an MCP server the tools below are available to the AI assistant.

See [SETUP.md](SETUP.md) for connection instructions.

---

## Ingestion (5)

| Tool | Description |
|------|-------------|
| `ingest_text` | Ingest a single text string into the active project |
| `ingest_texts` | Batch ingest a list of text strings in one call |
| `ingest_url` | Fetch and ingest content from a remote URL |
| `ingest_path` | Walk and ingest a local file or directory (supports glob patterns) |
| `get_job_status` | Poll the status of an async ingest job by job_id |

---

## Search & Query (2)

| Tool | Description |
|------|-------------|
| `search_knowledge` | Semantic / hybrid search — returns raw document chunks with scores |
| `query_knowledge` | Full RAG query — returns a synthesised answer with optional citations |

---

## Knowledge Base Management (5)

| Tool | Description |
|------|-------------|
| `list_knowledge` | List indexed sources with chunk counts for the active project |
| `delete_documents` | Remove documents from the index by document ID list (`doc_ids`) |
| `clear_knowledge` | Wipe the active project's vector store and BM25 index entirely |
| `get_stale_docs` | List documents that have not been refreshed in N days (default 30) |
| `get_active_leases` | List active read/write leases held via AxonStore |

---

## Project Management (4)

| Tool | Description |
|------|-------------|
| `list_projects` | List all local projects and mounted shares |
| `switch_project` | Switch the active project by name |
| `create_project` | Create a new named project with an isolated knowledge base |
| `delete_project` | Delete a project and all its stored data permanently |

---

## Settings (2)

| Tool | Description |
|------|-------------|
| `get_current_settings` | Return active RAG flags, model config, and runtime settings |
| `update_settings` | Toggle HyDE, rerank, GraphRAG, citations, and other RAG flags at runtime |

---

## Sessions (2)

| Tool | Description |
|------|-------------|
| `list_sessions` | List saved conversation sessions (up to 20 most recent) |
| `get_session` | Retrieve a full session transcript by timestamp ID |

---

## AxonStore & Sharing (4)

| Tool | Description |
|------|-------------|
| `init_store` | Initialise AxonStore at a shared filesystem base path |
| `share_project` | Generate a read-only share key for a project and grantee |
| `redeem_share` | Mount a shared project using a share string (read-only) |
| `list_shares` | List outgoing and incoming shares, including revoked status |

---

## Usage Notes

- All tools operate on the **active project**. Most ingest, search, and query tools accept an
  optional `project` parameter validated against the brain's active project (returns an error on
  mismatch). Tools that do **not** accept `project`: `ingest_path`, `list_sessions`,
  `get_session`, `list_shares`. Use `switch_project` to change the active project before
  querying or ingesting into a different one.
- Ingest tools (`ingest_path`, `ingest_url`) are async by default and return a `job_id`.
  Poll `get_job_status` until `status == "completed"`.
- `clear_knowledge` is irreversible — it wipes all vectors and BM25 state for the project.
- Mounted shares (via `redeem_share`) are always **read-only**. Ingest calls against a mount
  return an error.
- `update_settings` changes are scoped to the current server session and are not persisted to
  `config.yaml`.
