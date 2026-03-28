# Axon Skills Reference

This document lists every tool available to LLM agents via the **MCP server** and the
**VS Code LM tools** (GitHub Copilot agent mode).  Read it when you need to know which
tool to call and what each one does.

---

## Installing Axon as a skill in Claude Code

Once you have the Axon MCP server running (see [SETUP.md § 10](SETUP.md#10-mcp-server-setup)),
all 30 Axon tools are available to Claude Code automatically — no additional install step required.

**Step 1 — Start the Axon API:**

```bash
axon-api          # starts at http://127.0.0.1:8000 by default
```

**Step 2 — Add the MCP server to Claude Code:**

```bash
# One-time registration (run from any directory):
claude mcp add axon-mcp -- axon-mcp
```

Or add it manually to `~/.claude/claude.json` under `mcpServers`:

```json
{
  "mcpServers": {
    "axon": {
      "command": "axon-mcp",
      "args": []
    }
  }
}
```

**Step 3 — Use Axon tools in any Claude Code session:**

```
# In Claude Code chat:
Search my knowledge base for "dependency injection patterns"
Ingest the docs at /path/to/repo/docs
Show me the graph for "authentication flow"
```

Claude will automatically call `search_knowledge`, `ingest_path`, or `show_graph`
from the tool list below — no slash commands needed.

**Optional — load this file as persistent context:**

Add this to your project's `CLAUDE.md` so Claude always knows which Axon tools are available:

```markdown
# Axon Knowledge Tools
See docs/SKILLS.md for the full Axon tool reference.
```

---

## How to read this document

Tool names are **identical** on both the MCP server and the VS Code LM tools — use the same name everywhere.

Each skill entry follows this pattern:

```
tool_name
  What it does — one sentence.
  Use when: ...
  Do NOT use when: ...
  Key parameters: ...
```

---

## 1  Retrieval — asking questions and searching

### `query_knowledge`
Ask a question and get a synthesised answer from the knowledge base.
- **Use when** the user asks a question and wants a direct answer.
- **Do not use** when you need to inspect raw chunks before answering — use `search_knowledge` instead.
- **Key params:** `query` (required), `top_k` (optional, default 5), `project` (optional).

### `search_knowledge`
Retrieve raw document chunks without calling the LLM.
- **Use when** you want to see which specific passages exist, or before multi-step reasoning.
- **Do not use** when the user just wants a plain answer — `query_knowledge` is cheaper.
- **Key params:** `query` (required), `top_k` (default 5), `filters` (optional metadata dict), `project` (optional).

---

## 2  Graph — visualise knowledge relationships

> The graph panel is **off by default** in VS Code. Enable `axon.showGraphOnQuery` in
> settings to auto-open it on every search/query, or call `show_graph` explicitly.

### `show_graph` (VS Code only)
Open the graph panel for a specific query, with hit-node highlighting.
- **Use when** the user asks to "show the graph", "visualise", or "map" knowledge for a topic.
- **Key params:** `query` (required).

### `graph_status`
Return entity count, edge count, community count, and whether a community rebuild is in progress.
- **Use when** the user asks how many entities are in the graph, or whether GraphRAG is ready.

### `graph_finalize`
Trigger an explicit Louvain community detection rebuild.
- **Use when** the user asks to "rebuild the graph", "finalize communities", or after a large batch ingest.
- **Do not use** casually — this is a slow background operation.

### `graph_data`
Return the full entity/relation knowledge-graph as JSON (nodes and links).
- **Use when** the user wants to inspect, export, or build a custom visualisation of the entity graph.
- Returns empty arrays when no graph has been built yet.

### Browser visualization (REST only — no tool)
`POST /query/visualize` and `POST /search/visualize` return a self-contained HTML page
with the answer, source citations, and highlighted graph nodes.  From VS Code, use the
**"↗ Open in browser"** button in the graph panel tab bar — it calls `/query/visualize`
automatically with the current query.

---

## 3  Ingestion — adding knowledge

### `ingest_text`
Ingest a single text string directly.
- **Use when** you have a short passage, a code snippet, or any text that is not a file or URL.
- **Always set** `metadata.source` so the document can be audited later.
- **Key params:** `text` (required), `metadata` (recommended: `{"source": "..."}`) , `project` (optional).

### `ingest_texts`
Ingest multiple text documents in one call (single embedding batch — faster than looping `ingest_text`).
- **Use when** ingesting more than one text document at once.
- **Key params:** `docs` — list of `{"text": "...", "metadata": {...}}` dicts, `project` (optional).

### `ingest_url`
Fetch a URL and ingest its content.
- **Use when** the user provides a web URL to add to the knowledge base.
- **Key params:** `url` (required), `metadata` (optional), `project` (optional).

### `ingest_path`
Ingest a local file or directory from the server's filesystem.
- **Use when** the user points to a local path (file or folder).
- **Key params:** `path` (required — must be within the allowed ingest base directory).

### `ingest_image` (VS Code only)
Ingest an image file (PNG, JPG, etc.) using multimodal extraction.
- **Use when** the user wants to add a diagram, screenshot, or photo to the knowledge base.

### `get_job_status`
Check whether an async ingest job has completed.
- **Use when** you called an ingest tool and need to confirm it finished before querying.
- **Key params:** `job_id` (required — returned by the ingest call).

### `refresh_ingest`
Re-ingest sources that have changed on disk since they were last indexed.
- **Use when** the user says files have been updated and the knowledge base is stale.
- Returns a `job_id` — poll with `get_job_status` until completed.

### `get_stale_docs`
List documents not re-ingested within N days.
- **Use when** the user asks which knowledge is outdated or needs refreshing.
- **Key params:** `days` (default 7).

---

## 4  Knowledge base management

### `list_knowledge`
List all indexed sources in the active project with chunk counts.
- **Use when** the user asks "what's in the knowledge base?" or before ingesting to check for duplicates.

### `delete_documents`
Remove specific documents by their IDs.
- **Use when** the user explicitly asks to delete specific content.
- **Key params:** `doc_ids` — list of document ID strings.

### `clear_knowledge`
Delete ALL documents from the active project.
- **Use when** the user explicitly asks to wipe the entire knowledge base.
- **Do not use** without explicit user confirmation — this is irreversible.

---

## 5  Projects — namespacing knowledge

> Projects are independent namespaces. Each has its own vector store, BM25 index,
> and entity graph.  The active project applies to all queries and ingests unless
> you pass `project` explicitly.

### `list_projects`
List all available projects.
- **Use when** the user asks what projects exist, or before switching.

### `switch_project`
Make a different project active on the server.
- **Use when** the user says "switch to project X" or before querying/ingesting a named project.
- **Warning:** mutates global server state. Do not call concurrently.
- **Key params:** `project_name` (required).

### `create_project`
Create a new project directory.
- **Use when** the user asks to create a new knowledge namespace.
- **Key params:** `name` (required — 1–5 slash-separated alphanumeric segments, e.g. `"research/papers/2024"`), `description` (optional).

### `delete_project`
Delete a project and all its data.
- **Use when** the user explicitly asks to remove a project.
- **Do not use** if the project has active shares — revoke them first.
- **Key params:** `name` (required).

---

## 6  Settings — runtime configuration

### `get_current_settings`
Return the current active configuration (sensitive fields masked).
- **Use when** the user asks what model, provider, or RAG settings are active.

### `update_settings`
Change runtime configuration without restarting the server.
- **Use when** the user wants to switch LLM provider, change model, enable/disable RAG features, etc.
- **Key params (any subset):** `llm_provider`, `llm_model`, `embedding_provider`, `embedding_model`,
  `top_k`, `similarity_threshold`, `hybrid_search`, `rerank`, `raptor`, `graph_rag`,
  `graph_rag_ner_backend`, `graph_rag_community`, `graph_rag_depth`, `graph_rag_budget`,
  `graph_rag_relations`, `graph_rag_relation_budget`, `code_graph_bridge`,
  `persist` (bool — write change to config.yaml).

---

## 7  Sharing — AxonStore multi-user

### `init_store`
Initialise the AxonStore directory and register the current user.
- **Use when** setting up AxonStore for the first time or on a new machine.
- **Key params:** `base_path` (required), `persist` (optional — write to config.yaml).

### `share_project`
Generate a share string for another user to mount your project read-only.
- **Key params:** `project_name` (required), `grantee_username` (required).

### `redeem_share`
Mount a shared project using a share string from another user.
- **Key params:** `share_string` (required — the string produced by `share_project`).

### `list_shares`
List all outgoing shares (projects you have shared) and incoming mounts.
- **Use when** the user asks who has access to their projects, or what shared projects they have mounted.

### `revoke_share`
Revoke a previously generated share key. The grantee's mount becomes broken immediately.
- **Key params:** `key_id` (required — from `list_shares` / `list_shares` output).

---

## 8  Sessions

### `list_sessions`
List saved chat sessions for the active project.

### `get_session`
Retrieve a specific session by ID.
- **Key params:** `session_id` (required).

---

## 9  Operator tools

### `get_active_leases`
Return active write-lease counts for all projects tracked by the server.
- **Use when** checking whether it is safe to put a project into maintenance mode.
- Wait for `active_leases` to reach 0 before taking a project offline.

---

## Decision tree

```
User wants an answer to a question
  └─ query_knowledge / query_knowledge

User wants to see raw matching passages
  └─ search_knowledge / search_knowledge

User wants to visualise the knowledge graph
  ├─ show_graph (VS Code webview)
  │   └─ "Open in browser" button → /query/visualize (standalone HTML)
  ├─ graph_data / graph_data  (raw JSON export)
  └─ graph_status / graph_status  (status check)

User wants to add content
  ├─ one text  → ingest_text / ingest_text
  ├─ many texts → ingest_texts / ingest_texts
  ├─ URL   → ingest_url / ingest_url
  ├─ file  → ingest_path / ingest_path (async → poll get_job_status / get_job_status)
  └─ image → ingest_image

User asks what's in the knowledge base
  └─ list_knowledge / list_knowledge

User wants to refresh stale content
  ├─ refresh_ingest / refresh_ingest  (re-ingest changed files)
  └─ get_stale_docs / get_stale_docs   (list outdated docs)

User wants to change project
  └─ switch_project / switch_project

User wants to change model or RAG settings
  ├─ update_settings / update_settings
  └─ get_current_settings / get_current_settings  (inspect before changing)

User wants graph status / rebuild
  ├─ graph_status / graph_status
  └─ graph_finalize / graph_finalize

User wants to share / revoke project access
  ├─ share_project / share_project
  ├─ redeem_share / redeem_share
  ├─ revoke_share
  └─ list_shares / list_shares

Operator: check if project can be taken offline
  └─ get_active_leases / get_active_leases
```
