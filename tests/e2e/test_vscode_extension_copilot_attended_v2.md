# VS Code Extension — Agent Test Design Guide

This guide enables an AI agent or human tester to systematically verify all 39 LM tools,
the VS Code command palette commands, and the `@axon` chat participant registered by the
Axon VS Code extension.

---

## Overview

The extension registers tools via `vscode.lm.registerTool()` on activation. Each tool is a
TypeScript class in `integrations/vscode-axon/src/tools/`. Tools call the Axon REST API
(default `http://127.0.0.1:8000`) and return a `LanguageModelToolResult` containing a text
string.

Two execution modes are available:

**Automated — Node.js subprocess runner** (no real VS Code required):
```
python -m pytest tests/e2e/test_vscode_extension_tools_e2e.py -v -m e2e
```
The runner (`tests/e2e/vscode_helpers.py`) stubs the `vscode` module, activates the
extension, invokes one tool, and returns JSON with these fields:
- `toolResult` — concatenated text from all `LanguageModelTextPart`s
- `toolError` — string if the tool threw or was not registered
- `registeredTools` — sorted list of all registered tool names
- `registeredCommands` — sorted list of all registered command names
- `registeredParticipants` — list of chat participant IDs
- `outputLines` — lines written to the output channel
- `panelCount` — number of webview panels opened
- `lastPanelHtml` — HTML of the most recently opened panel
- `clipboardWrites` — strings written to the clipboard

**Attended — real VS Code with GitHub Copilot**:
Open the workspace in VS Code with the extension installed. Type prompts in GitHub Copilot
Chat and observe tool invocations in the Axon output channel.

---

## Test Environment Setup

1. Start the Axon API server: `axon-api` (listens on `http://127.0.0.1:8000` by default)
2. Build the extension: `cd integrations/vscode-axon && npm run compile`
3. Run automated suite: `python -m pytest tests/e2e/test_vscode_extension_tools_e2e.py -v --no-cov -m e2e`

VS Code settings used by the extension (configurable via `axon.*`):
- `axon.apiBase` — REST API base URL (default `http://127.0.0.1:8000`)
- `axon.apiKey` — optional Bearer token
- `axon.autoStart` — whether to spawn the Python server on activation
- `axon.showGraphOnQuery` — whether to open the graph panel after search/query

---

## Tool Test Cases

### Category 1: Query Tools

#### T01 — `search_knowledge`

- **Class:** `AxonSearchTool` (`tools/query.ts`)
- **API endpoint:** `POST /search` (via `searchAxon` helper, which uses `httpPost` internally)
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `query` | string | yes | `"authentication flow"` |
  | `top_k` | number | no | `3` |
  | `threshold` | number | no | `0.7` |
  | `filters` | object | no | `{"source": "notes.md"}` |
  | `project` | string | no | `"engineering"` |

- **Example input:** `{"query": "authentication flow", "top_k": 3}`
- **Expected API call:** `POST /search` with body `{"query": "authentication flow", "top_k": 3}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` contains one or more chunks formatted as `[ID: <id>] Source: <source>\n<text>`
  - Each chunk has a non-empty `text` and a `metadata.source` value
- **Edge cases:**
  - Empty `query` string → API returns empty results; `toolResult` equals `"No results found."`
  - `threshold` set high (e.g. `0.99`) with no matching docs → falls back to top-N without threshold; `toolResult` contains `"*No results met the threshold"` note
  - `top_k: 0` → returns `"No results found."`
  - API unreachable → `toolResult` starts with `"Error during Axon search:"`
- **Automated coverage:** Yes — `test_tool_invocations[search_knowledge...]`

---

#### T02 — `query_knowledge`

- **Class:** `AxonQueryTool` (`tools/query.ts`)
- **API endpoint:** `POST /query`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `query` | string | yes | `"How does auth work?"` |
  | `top_k` | number | no | `5` |
  | `filters` | object | no | `{"project": "default"}` |
  | `project` | string | no | `"research"` |

- **Example input:** `{"query": "How does auth work?"}`
- **Expected API call:** `POST /query` with body `{"query": "How does auth work?", "discuss": false, "raptor": false, "graph_rag": false}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` equals `data.response` — a synthesised LLM answer string
  - No raw JSON blob in result
- **Edge cases:**
  - LLM returns empty `response` field → `toolResult` equals `"No answer generated."`
  - API returns non-200 → `toolResult` starts with `"Axon API Error (4xx):"`
  - `project` param forwarded in body when supplied
- **Automated coverage:** Yes — `test_tool_invocations[query_knowledge...]`

---

### Category 2: Ingest Tools

#### T03 — `ingest_text`

- **Class:** `AxonIngestTextTool` (`tools/ingest.ts`)
- **API endpoint:** `POST /add_text`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `text` | string | yes | `"Axon uses BM25 for retrieval."` |
  | `source` | string | no | `"meeting-notes"` (defaults to `"agent_input"`) |
  | `project` | string | no | `"engineering"` |

- **Example input:** `{"text": "Axon uses BM25 for retrieval.", "source": "meeting-notes"}`
- **Expected API call:** `POST /add_text` with body `{"text": "...", "metadata": {"source": "meeting-notes"}}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` matches `"Success: ok, ID: <doc_id>"`
- **Edge cases:**
  - Empty `text` → API may return 422; `toolResult` starts with `"Axon Ingest Error (422):"`
  - `project` appended to request body when provided
  - `source` defaults to `"agent_input"` when omitted
- **Automated coverage:** Yes — `test_tool_invocations[ingest_text...]`

---

#### T04 — `ingest_texts`

- **Class:** `AxonIngestTextsTool` (`tools/ingest.ts`)
- **API endpoint:** `POST /add_texts`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `docs` | array of `{text, metadata?}` | yes | `[{"text": "Doc 1"}, {"text": "Doc 2"}]` |
  | `project` | string | no | `"research"` |

- **Example input:** `{"docs": [{"text": "First doc"}, {"text": "Second doc"}]}`
- **Expected API call:** `POST /add_texts` with body `{"docs": [...]}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` matches `"Ingested <N> documents."` where N equals `docs.length`
- **Edge cases:**
  - Empty `docs` array → `toolResult` contains `"Ingested 0 documents."`
  - `docs` omitted entirely → tool may return a parse error or `"Ingested ? documents."`
- **Automated coverage:** Yes — `test_ingest_texts_tool`

---

#### T05 — `ingest_url`

- **Class:** `AxonIngestUrlTool` (`tools/ingest.ts`)
- **API endpoint:** `POST /ingest_url`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `url` | string | yes | `"https://example.com/docs"` |
  | `project` | string | no | `"engineering"` |

- **Example input:** `{"url": "https://example.com/docs"}`
- **Expected API call:** `POST /ingest_url` with body `{"url": "https://example.com/docs"}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` matches `"Status: ok, URL: https://example.com/docs"`
- **Edge cases:**
  - Malformed URL → API returns 422; `toolResult` starts with `"Axon URL Ingest Error"`
  - Unreachable URL → backend may return 4xx or error status
- **Automated coverage:** Yes — `test_tool_invocations[ingest_url...]`

---

#### T06 — `ingest_path`

- **Class:** `AxonIngestPathTool` (`tools/ingest.ts`)
- **API endpoint:** `POST /ingest`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `path` | string | yes | `"/home/user/docs"` |

- **Example input:** `{"path": "/home/user/docs"}`
- **Expected API call:** `POST /ingest` with body `{"path": "/home/user/docs"}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` matches `"Status: ok, Message: Ingestion started, JobID: <job_id>"`
- **Edge cases:**
  - Path does not exist → API returns 422; `toolResult` starts with `"Axon Path Ingest Error"`
  - Large directory → job starts async; caller should poll with `get_job_status`
- **Automated coverage:** Yes — `test_tool_invocations[ingest_path...]`

---

#### T07 — `get_job_status`

- **Class:** `AxonGetIngestStatusTool` (`tools/ingest.ts`)
- **API endpoint:** `GET /ingest/status/{job_id}`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `job_id` | string | yes | `"mock-job-001"` |

- **Example input:** `{"job_id": "mock-job-001"}`
- **Expected API call:** `GET /ingest/status/mock-job-001`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` equals `"Ingestion complete. Status: completed. You can now search the ingested documents."` when status is `"completed"`
  - `toolResult` contains `"Ingestion failed. Error: ..."` when status is `"failed"`
  - `toolResult` contains `"Ingestion still in progress"` when status is `"running"` or `"pending"`
- **Edge cases:**
  - Unknown `job_id` → API returns 404; `toolResult` equals `"Job not found: <job_id>"`
  - `job_id` with special characters → URL-encoded via `encodeURIComponent`
- **Automated coverage:** Yes — `test_tool_invocations[get_job_status...]`

---

#### T08 — `refresh_ingest`

- **Class:** `AxonRefreshIngestTool` (`tools/ingest.ts`)
- **API endpoint:** `POST /ingest/refresh`
- **Input schema:** None (no required fields)
- **Example input:** `{}`
- **Expected API call:** `POST /ingest/refresh` with empty body
- **Success criteria:**
  - `toolError` is null
  - Async path: `toolResult` contains `"Refresh started (job_id: ...)"` and instructions to poll
  - Sync path: `toolResult` contains counts: `"X re-ingested, Y unchanged, Z missing, W errors"`
- **Edge cases:**
  - No documents ingested yet → `0 re-ingested, 0 unchanged`
  - Server returns `job_id` → tool returns async message; follow up with `get_job_status`
- **Automated coverage:** Yes — `test_tool_invocations[refresh_ingest...]`

---

#### T09 — `get_stale_docs`

- **Class:** `AxonGetStaleDocsTool` (`tools/ingest.ts`)
- **API endpoint:** `GET /collection/stale?days=<N>`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `days` | number | no | `7` (default) |

- **Example input:** `{"days": 14}`
- **Expected API call:** `GET /collection/stale?days=14`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` is a JSON string of `{stale_docs: [...]}` where each entry has `project`, `doc_id`, `age_days`
- **Edge cases:**
  - `days` omitted → defaults to `7`
  - No stale docs → `stale_docs` is an empty array
- **Automated coverage:** Yes — `test_tool_invocations[get_stale_docs...]` (uses `extra={"_inputResponse": "7"}`)

---

#### T10 — `clear_knowledge`

- **Class:** `AxonClearKnowledgeTool` (`tools/ingest.ts`)
- **API endpoint:** `POST /clear`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `POST /clear` with empty body
- **Success criteria:**
  - `toolError` is null
  - `toolResult` equals `"Knowledge base cleared for current project."`
- **Edge cases:**
  - Already-empty knowledge base → still returns success
  - API error → `toolResult` starts with `"Clear error:"`
- **Automated coverage:** Yes — `test_tool_invocations[clear_knowledge...]`

---

#### T11 — `ingest_image`

- **Class:** `AxonIngestImageTool` (`tools/ingest.ts`)
- **API endpoint:** `POST /add_text` (after Copilot vision generates description)
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `imagePath` | string | yes | `"/workspace/diagrams/arch.png"` |
  | `project` | string | no | `"engineering"` |
  | `alt_text` | string | no | `"Architecture diagram showing..."` |

- **Example input (headless):** `{"imagePath": "/tmp/test.png", "alt_text": "A diagram of the Axon data pipeline."}`
- **Example input (vision):** `{"imagePath": "/tmp/test.png"}` (requires Copilot model with `supportsImageToText`)
- **Expected API call:** `POST /add_text` with `text` = Copilot-generated or `alt_text` description; `metadata.type = "image"`, `metadata.original_path = imagePath`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` contains `"Image ingested successfully. doc_id: ..."` and description length
- **Edge cases:**
  - Unsupported extension (`.gif`) → `toolResult` contains `"Unsupported image format".gif""` before any API call
  - No Copilot model available and no `alt_text` → `toolResult` equals `"No Copilot language model available."`
  - Copilot returns empty description → `toolResult` equals `"Copilot returned an empty description for the image."`
  - `alt_text` provided → Copilot vision skipped entirely; `metadata.ingested_via = "alt_text"`
- **CI note:** Skip in CI unless `alt_text` is provided — Copilot vision requires authenticated IDE session
- **Automated coverage:** Yes — `test_ingest_image_tool` (uses `_copilotModels` stub)

---

### Category 3: Project Management Tools

#### T12 — `list_projects`

- **Class:** `AxonListProjectsTool` (`tools/projects.ts`)
- **API endpoint:** `GET /projects`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `GET /projects`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` equals `"Projects: default, engineering, research, team/subproject"` (comma-separated names from `data.projects[].name`)
  - Returns `"Projects: None"` when the list is empty
- **Edge cases:**
  - API returns empty `projects` array → `"Projects: None"`
  - Shared mounts not shown (they appear via `switch_project` command, not this tool)
- **Automated coverage:** Yes — `test_tool_invocations[list_projects...]`

---

#### T13 — `switch_project`

- **Class:** `AxonSwitchProjectTool` (`tools/projects.ts`)
- **API endpoint:** `POST /project/switch`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `name` | string | yes | `"engineering"` |

- **Example input:** `{"name": "engineering"}`
- **Expected API call:** `POST /project/switch` with body `{"name": "engineering"}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` equals `"Switched to project: engineering"`
- **Edge cases:**
  - Non-existent project → API returns 404; `toolResult` starts with `"Axon API Error (404):"`
  - Project name with slashes (e.g. `"team/sub"`) → valid per `_MAX_DEPTH=5`
- **Automated coverage:** Yes — `test_tool_invocations[switch_project...]`

---

#### T14 — `create_project`

- **Class:** `AxonCreateProjectTool` (`tools/projects.ts`)
- **API endpoint:** `POST /project/new`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `name` | string | yes | `"new-project"` |
  | `description` | string | no | `"My new project"` (defaults to `""`) |

- **Example input:** `{"name": "new-project", "description": "Testing"}`
- **Expected API call:** `POST /project/new` with body `{"name": "new-project", "description": "Testing"}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` matches `"Status: ok, Project: new-project"`
- **Edge cases:**
  - Name with more than 5 slash segments → API returns 422
  - Duplicate name → API may return 409 or 400
  - `description` omitted → defaults to `""`
- **Automated coverage:** Yes — `test_tool_invocations[create_project...]`

---

#### T15 — `delete_project`

- **Class:** `AxonDeleteProjectTool` (`tools/projects.ts`)
- **API endpoint:** `POST /project/delete/{name}`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `name` | string | yes | `"old-project"` |

- **Example input:** `{"name": "old-project"}`
- **Expected API call:** `POST /project/delete/old-project`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` matches `"Status: ok, Message: Project deleted"`
- **Edge cases:**
  - Delete the active project → behaviour is backend-defined; may succeed or return 400
  - Delete `"default"` project → backend may reject
  - Non-existent name → API returns 404
- **Automated coverage:** Yes — `test_tool_invocations[delete_project...]`

---

#### T16 — `delete_documents`

- **Class:** `AxonDeleteDocumentsTool` (`tools/projects.ts`)
- **API endpoint:** `POST /delete`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `docIds` | string[] | yes | `["doc-001", "doc-002"]` |

- **Example input:** `{"docIds": ["doc-001", "doc-002"]}`
- **Expected API call:** `POST /delete` with body `{"doc_ids": ["doc-001", "doc-002"]}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` matches `"Status: ok, Deleted: 2"`
- **Edge cases:**
  - Empty `docIds` array → `"Status: ok, Deleted: 0"`
  - Unknown doc IDs → API returns count of successfully deleted docs (may be 0)
- **Automated coverage:** Yes — `test_tool_invocations[delete_documents...]`

---

#### T17 — `list_knowledge`

- **Class:** `AxonListKnowledgeTool` (`tools/projects.ts`)
- **API endpoint:** `GET /collection`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `GET /collection`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` contains `"Total Files: 2\nTotal Chunks: 10\n\nFiles:\n..."` with per-file chunk counts
- **Edge cases:**
  - Empty knowledge base → `"Total Files: 0\nTotal Chunks: 0\n\nFiles:\n"`
- **Automated coverage:** Yes — `test_tool_invocations[list_knowledge...]`

---

#### T18 — `update_settings`

- **Class:** `AxonUpdateSettingsTool` (`tools/projects.ts`)
- **API endpoint:** `POST /config/update`
- **Input schema:** Any key-value pairs matching AxonConfig fields (e.g. `top_k`, `rerank`)
- **Example input:** `{"top_k": 10, "rerank": true}`
- **Expected API call:** `POST /config/update` with body `{"top_k": 10, "rerank": true}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` equals `"Status: ok, Settings Applied."`
- **Edge cases:**
  - Unknown key → API may return 422 or silently ignore; `toolResult` may indicate error
  - Setting `embedding.base_url` → not a valid AxonConfig field; use `ollama_base_url` instead
- **Automated coverage:** Yes — `test_tool_invocations[update_settings...]`

---

#### T19 — `get_current_settings`

- **Class:** `AxonGetCurrentSettingsTool` (`tools/projects.ts`)
- **API endpoint:** `GET /config`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `GET /config`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` starts with `"Current Axon settings:\n"` followed by pretty-printed JSON
  - JSON includes keys like `llm_provider`, `llm_model`, `top_k`, `rerank`, `hybrid_search`
- **Edge cases:**
  - API unreachable → `toolResult` starts with `"Could not reach Axon API:"`
- **Automated coverage:** Yes — `test_tool_invocations[get_current_settings...]`

---

#### T20 — `list_sessions`

- **Class:** `AxonListSessionsTool` (`tools/projects.ts`)
- **API endpoint:** `GET /sessions`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `GET /sessions`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` is pretty-printed JSON containing `sessions` array
  - Each session has `session_id`, `project`, `active` fields
- **Edge cases:**
  - No active sessions → `sessions` is an empty array
- **Automated coverage:** Yes — `test_list_sessions_tool`

---

#### T21 — `get_session`

- **Class:** `AxonGetSessionTool` (`tools/projects.ts`)
- **API endpoint:** `GET /session/{session_id}`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `session_id` | string | yes | `"sess-001"` |

- **Example input:** `{"session_id": "sess-001"}`
- **Expected API call:** `GET /session/sess-001`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` is pretty-printed JSON with `session_id`, `project`, `active`, `messages` fields
- **Edge cases:**
  - Unknown `session_id` → API returns 404; `toolResult` starts with `"Session error:"`
  - `session_id` with special characters → URL-encoded via `encodeURIComponent`
- **Automated coverage:** Yes — `test_get_session_tool`

---

### Category 4: Sharing and AxonStore Tools

#### T22 — `share_project`

- **Class:** `AxonShareProjectTool` (`tools/shares.ts`)
- **API endpoint:** `POST /share/generate`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `project` | string | yes | `"research"` |
  | `grantee` | string | yes | `"bob"` |

- **Example input:** `{"project": "research", "grantee": "bob"}`
- **Expected API call:** `POST /share/generate` with body `{"project": "research", "grantee": "bob"}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` contains `"Share key generated.\nProject: research\nGrantee: bob\nAccess: read-only\nKey ID: sk_..."` and the `share_string` value
- **Edge cases:**
  - AxonStore not initialised → API returns error; `toolResult` starts with `"Share generation failed:"`
  - Non-existent project → API returns 404
- **Automated coverage:** Yes — `test_tool_invocations[share_project...]`

---

#### T23 — `redeem_share`

- **Class:** `AxonRedeemShareTool` (`tools/shares.ts`)
- **API endpoint:** `POST /share/redeem`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `share_string` | string | yes | `"axon-share-base64placeholder"` |

- **Example input:** `{"share_string": "axon-share-base64placeholder"}`
- **Expected API call:** `POST /share/redeem` with body `{"share_string": "axon-share-base64placeholder"}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` equals `"Share redeemed. Project \"alice/default\" mounted as \"mounts/alice_default\" (read-only)."`
- **Edge cases:**
  - Malformed share string → API returns 422
  - Already-redeemed share string → API may return 409
  - Revoked key → API returns 403
- **Automated coverage:** Yes — `test_tool_invocations[redeem_share...]`

---

#### T24 — `revoke_share`

- **Class:** `AxonRevokeShareTool` (`tools/shares.ts`)
- **API endpoint:** `POST /share/revoke`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `key_id` | string | yes | `"sk_test123abc"` |

- **Example input:** `{"key_id": "sk_test123abc"}`
- **Expected API call:** `POST /share/revoke` with body `{"key_id": "sk_test123abc"}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` equals `"Share sk_test123abc revoked. Grantee \"bob\" no longer has access to \"default\"."`
- **Edge cases:**
  - Already-revoked `key_id` → API may return 404 or idempotent success
  - Unknown `key_id` → API returns 404; `toolResult` starts with `"Revoke failed:"`
- **Automated coverage:** Yes — `test_tool_invocations[revoke_share...]`

---

#### T25 — `list_shares`

- **Class:** `AxonListSharesTool` (`tools/shares.ts`)
- **API endpoint:** `GET /share/list`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `GET /share/list`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` is pretty-printed JSON with `sharing` array (outbound) and `shared` array (inbound)
  - Each `sharing` entry has `key_id`, `project`, `grantee`, `revoked`
- **Edge cases:**
  - No shares → both arrays empty
  - AxonStore not initialised → API returns 500 or empty response
- **Automated coverage:** Yes — `test_tool_invocations[list_shares...]`

---

#### T26 — `init_store`

- **Class:** `AxonInitStoreTool` (`tools/shares.ts`)
- **API endpoint:** `POST /store/init`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `base_path` | string | yes | `"/data/axon-store"` |

- **Example input:** `{"base_path": "/data/axon-store"}`
- **Expected API call:** `POST /store/init` with body `{"base_path": "/data/axon-store"}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` equals `"AxonStore initialised at /tmp/AxonStore/testuser (user: testuser). Share tools are now available."`
- **Edge cases:**
  - Path not writable → API returns 500
  - Already-initialised store → API should succeed idempotently
- **Automated coverage:** Yes — `test_tool_invocations[init_store...]`

---

#### T27 — `get_store_status`

- **Class:** `AxonGetStoreStatusTool` (`tools/shares.ts`)
- **API endpoint:** `GET /store/status`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `GET /store/status`
- **Success criteria:**
  - `toolError` is null
  - If initialised: `toolResult` equals `"AxonStore ready — path: ..., version: ..., user: ..."`
  - If not initialised: `toolResult` equals `"AxonStore is not initialised. Call init_store first."`
- **Edge cases:**
  - Store not yet initialised → `data.initialized` is `false`; descriptive message returned
- **Automated coverage:** Partial — registered but not in parametrized list; covered implicitly by `run_tool` registration check

---

### Category 5: Graph and Visualization Tools

#### T28 — `graph_status`

- **Class:** `AxonGraphStatusTool` (`tools/graph.ts`)
- **API endpoint:** `GET /graph/status`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `GET /graph/status`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` is pretty-printed JSON with `entity_count`, `relation_count`, `community_summary_count`, `community_build_in_progress`
- **Edge cases:**
  - GraphRAG disabled in config → may still return counts (all zeros)
  - Build in progress → `community_build_in_progress` is `true`
- **Automated coverage:** Yes — `test_tool_invocations[graph_status...]`

---

#### T29 — `show_graph`

- **Class:** `AxonShowGraphTool` (`tools/graph.ts`)
- **API endpoint:** `GET /search/raw` + `GET /graph/data` (via `showGraphForQuery` panel helper)
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `query` | string | yes | `"how Axon connects docs"` |

- **Example input:** `{"query": "how Axon connects docs"}`
- **Expected side effect:** Opens a VS Code WebviewPanel; `panelCount` becomes 1
- **Success criteria:**
  - `toolError` is null
  - `res["panelCount"]` equals `1`
  - `toolResult` starts with `"Graph panel status:"`
- **Edge cases:**
  - No nodes in graph → panel opens but shows empty graph
  - GraphRAG disabled → panel still opens using vector search results
- **CI note:** Cannot be run in headless CI without a display. The Node.js stub creates a mock panel — this is sufficient to verify the code path.
- **Automated coverage:** Yes — `test_show_graph_tool`

---

#### T30 — `graph_finalize`

- **Class:** `AxonGraphFinalizeTool` (`tools/graph.ts`)
- **API endpoint:** `POST /graph/finalize`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `POST /graph/finalize` with empty body
- **Success criteria:**
  - `toolError` is null
  - `toolResult` equals `"Graph finalized. Community summaries: 7"`
- **Edge cases:**
  - No entities in graph → finalize succeeds with `community_summary_count: 0`
  - GraphRAG disabled → API may return 400
- **Automated coverage:** Yes — `test_tool_invocations[graph_finalize...]`

---

#### T31 — `graph_data`

- **Class:** `AxonGraphDataTool` (`tools/graph.ts`)
- **API endpoint:** `GET /graph/data`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `GET /graph/data`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` starts with `"Graph: <N> nodes, <M> edges.\n"` followed by JSON
  - `nodes` capped at 500; `links` capped at 1000; `truncated` field indicates if clipped
- **Edge cases:**
  - Empty graph → `"Graph: 0 nodes, 0 edges."`
  - Large graph → `nodes` truncated to 500; `toolResult` JSON contains `"truncated": true`
- **Automated coverage:** Yes — `test_graph_data_tool`

---

#### T32 — `get_active_leases`

- **Class:** `AxonGetActiveLeasesTool` (`tools/graph.ts`)
- **API endpoint:** `GET /registry/leases`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `GET /registry/leases`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` is pretty-printed JSON with a `leases` array
  - Each lease has `lease_id`, `project`, `holder`
- **Edge cases:**
  - No active leases → `leases` is an empty array
- **Automated coverage:** Yes — `test_get_active_leases_tool`

---

### Category 6: Config Tools

#### T33 — `axon_config_validate` (also registered as `axonConfigValidate`)

- **Class:** `AxonConfigValidateTool` (`tools/config.ts`)
- **API endpoint:** `GET /config/validate`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `GET /config/validate`
- **Success criteria:**
  - `toolError` is null
  - When no issues: `toolResult` equals `"Config validation passed. No issues found."`
  - When issues exist: `toolResult` starts with `"Config has errors (N error(s)):"` or `"Config has N notice(s):"` followed by lines formatted as `[LEVEL] section.field: message  Suggestion: ...`
- **Edge cases:**
  - Config file missing → API may return 500 or report a critical error
  - Camel-case alias `axonConfigValidate` must also be registered and produce identical results
- **Automated coverage:** Yes — `test_axon_config_validate_tool` (snake_case name); camel-case alias not explicitly tested

---

#### T34 — `axon_config_set` (also registered as `axonConfigSet`)

- **Class:** `AxonConfigSetTool` (`tools/config.ts`)
- **API endpoint:** `POST /config/set` (called once per key)
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `changes` | object (dot-notation keys) | yes | `{"chunk.strategy": "markdown", "rag.top_k": 15}` |
  | `persist` | boolean | no | `true` (default) |

- **Example input:** `{"changes": {"rag.top_k": 15, "chunk.strategy": "markdown"}}`
- **Expected API call:** `POST /config/set` with body `{"key": "rag.top_k", "value": 15, "persist": true}` (repeated per key)
- **Success criteria:**
  - `toolError` is null
  - `toolResult` lists each change as `"key = <new_value> (was <old_value>)"`
  - Sensitive keys (matching `/key|secret|password|token/i`) are redacted as `[redacted]`
- **Edge cases:**
  - Empty `changes` object → `toolResult` equals `"No changes provided."`
  - Invalid key → API returns 422; listed in error section of result
  - `persist: false` → changes are session-only, not written to `config.yaml`
  - Camel-case alias `axonConfigSet` must also be registered
- **Automated coverage:** Yes — `test_axon_config_set_tool` (snake_case name)

---

### Category 7: Security Tools

All security tools require the server to have `axon.security` configured (sealed-store
bootstrapped). Without it, `security_status` returns `initialized: false` and
`security_bootstrap` must be called first.

#### T35 — `security_status`

- **Class:** `AxonSecurityStatusTool` (`tools/security.ts`)
- **API endpoint:** `GET /security/status`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `GET /security/status`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` starts with `"Axon security status:\n"` followed by:
    - `initialized: true/false`
    - `unlocked: true/false`
    - `locked: true/false`
    - `sealed_hidden_count: N`
    - `public_key_fingerprint: <fingerprint>` (when present)
    - `cipher_suite: ChaCha20Poly1305` (when present)
- **Edge cases:**
  - Security not configured → `initialized: false`, `unlocked: false`
  - API unreachable → `toolResult` starts with `"Could not reach Axon API:"`
- **Automated coverage:** Yes — `test_security_status_tool`

---

#### T36 — `security_bootstrap`

- **Class:** `AxonSecurityBootstrapTool` (`tools/security.ts`)
- **API endpoint:** `POST /security/bootstrap`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `passphrase` | string | yes | `"my-strong-passphrase"` |

- **Example input:** `{"passphrase": "my-strong-passphrase"}`
- **Expected API call:** `POST /security/bootstrap` with body `{"passphrase": "my-strong-passphrase"}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` starts with `"Security bootstrapped.\n"` followed by status lines showing `initialized: true`, `unlocked: true`
- **Edge cases:**
  - Already-bootstrapped store → API returns 409; `toolResult` starts with `"Security bootstrap failed:"`
  - Weak passphrase → API may return 400 with validation message
- **Security note:** Passphrase is sent in plain JSON. Use HTTPS in production deployments.
- **Automated coverage:** Yes — `test_security_bootstrap_tool`

---

#### T37 — `security_unlock`

- **Class:** `AxonSecurityUnlockTool` (`tools/security.ts`)
- **API endpoint:** `POST /security/unlock`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `passphrase` | string | yes | `"my-strong-passphrase"` |

- **Example input:** `{"passphrase": "my-strong-passphrase"}`
- **Expected API call:** `POST /security/unlock` with body `{"passphrase": "my-strong-passphrase"}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` starts with `"Security unlocked.\n"` followed by status lines showing `unlocked: true`
- **Edge cases:**
  - Wrong passphrase → API returns 401; `toolResult` starts with `"Security unlock failed:"`
  - Already unlocked → API may return 200 idempotently or 400
  - Not yet bootstrapped → API returns 400
- **Automated coverage:** Yes — `test_security_unlock_tool`

---

#### T38 — `security_lock`

- **Class:** `AxonSecurityLockTool` (`tools/security.ts`)
- **API endpoint:** `POST /security/lock`
- **Input schema:** None
- **Example input:** `{}`
- **Expected API call:** `POST /security/lock` with empty body
- **Success criteria:**
  - `toolError` is null
  - `toolResult` starts with `"Security locked.\n"` followed by status lines showing `unlocked: false`, `locked: true`
- **Edge cases:**
  - Already locked → API returns 200 idempotently
  - Not bootstrapped → API returns 400
- **Automated coverage:** Yes — `test_security_lock_tool`

---

#### T39 — `security_change_passphrase`

- **Class:** `AxonSecurityChangePassphraseTool` (`tools/security.ts`)
- **API endpoint:** `POST /security/change-passphrase`
- **Input schema:**

  | Field | Type | Required | Example |
  |-------|------|----------|---------|
  | `old_passphrase` | string | yes | `"old-secret"` |
  | `new_passphrase` | string | yes | `"new-secret"` |

- **Example input:** `{"old_passphrase": "old-secret", "new_passphrase": "new-secret"}`
- **Expected API call:** `POST /security/change-passphrase` with body `{"old_passphrase": "old-secret", "new_passphrase": "new-secret"}`
- **Success criteria:**
  - `toolError` is null
  - `toolResult` starts with `"Security passphrase changed.\n"` followed by status lines
- **Edge cases:**
  - Wrong `old_passphrase` → API returns 401; `toolResult` starts with `"Passphrase change failed:"`
  - `new_passphrase` same as `old_passphrase` → API may allow or reject
  - Not bootstrapped → API returns 400
- **Automated coverage:** Yes — `test_security_change_passphrase_tool`

---

## Command Test Cases

VS Code commands are registered via `vscode.commands.registerCommand`. They are invoked
through the command palette (`Ctrl+Shift+P`) or programmatically. In the Node.js runner,
prefix the command name with `cmd:` in the `tool_name` argument.

### C01 — `axon.switchProject`

- **Handler:** `switchProject(apiBase)` (`tools/projects.ts`)
- **Behavior:** Fetches project list (GET /projects), shows QuickPick including shared mounts, then calls POST /project/switch with selected name
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.switchProject", {}, {"_quickPickResponse": "engineering"})
  ```
- **Success criteria:** Output channel contains `"Switched to project: engineering"`
- **Edge cases:** No projects → `showInformationMessage("No projects found.")`; API unreachable → error message shown

### C02 — `axon.createProject`

- **Handler:** `createNewProject(apiBase)` (`tools/projects.ts`)
- **Behavior:** Shows two InputBox prompts (name, description), calls POST /project/new, then auto-switches via POST /project/switch
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.createProject", {}, {"_inputResponses": ["my-proj", "My description"]})
  ```
- **Success criteria:** Output channel contains `"Created project: my-proj"`

### C03 — `axon.ingestFile`

- **Handler:** `ingestCurrentFile(apiBase)` (`tools/ingest.ts`)
- **Behavior:** Reads the active editor's text and file path, calls POST /add_texts
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.ingestFile", {}, {"_activeFile": "/tmp/sample.py", "_activeFileText": "print('hello')"})
  ```
- **Success criteria:** Output channel contains `"Ingested file:"`

### C04 — `axon.ingestWorkspace`

- **Handler:** `ingestWorkspaceFolder(apiBase)` (`tools/ingest.ts`)
- **Behavior:** Ingests the first workspace folder via POST /ingest
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.ingestWorkspace", {}, {"_workspaceFolders": ["/tmp/myproject"]})
  ```
- **Success criteria:** Output channel contains `"Ingest workspace:"`; info message shown with job_id

### C05 — `axon.ingestFolder`

- **Handler:** `ingestPickedFolder(apiBase)` (`tools/ingest.ts`)
- **Behavior:** Opens a file/folder picker dialog, calls POST /ingest with selected path
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.ingestFolder", {}, {"_openDialogResult": "/tmp/docs"})
  ```
- **Success criteria:** Output channel contains `"Ingest folder/file:"`

### C06 — `axon.startServer`

- **Handler:** `ensureServerRunning(apiBase, context)` (`client/server.ts`)
- **Behavior:** Checks if server is running; spawns Python process if not
- **Test recipe:** `run_tool(base_url, "cmd:axon.startServer", {})`
- **Success criteria:** No `toolError`; server health check attempted

### C07 — `axon.stopServer`

- **Handler:** `stopServer()` (`client/server.ts`)
- **Behavior:** Terminates the managed Python server process if running
- **Test recipe:** `run_tool(base_url, "cmd:axon.stopServer", {})`
- **Success criteria:** No `toolError`

### C08 — `axon.initStore`

- **Handler:** `initStore(apiBase)` (`tools/shares.ts`)
- **Behavior:** Reads `axon.storeBase` config or prompts for path, calls POST /store/init with `persist: true`
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.initStore", {}, {"storeBase": "/tmp/axon-data"})
  ```
- **Success criteria:** Info message contains `"AxonStore initialized"`

### C09 — `axon.shareProject`

- **Handler:** `shareProject(apiBase)` (`tools/shares.ts`)
- **Behavior:** Prompts for project name and grantee username, calls POST /share/generate, copies share_string to clipboard
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.shareProject", {}, {"_inputResponses": ["research", "bob"]})
  ```
- **Success criteria:** `clipboardWrites` contains the `share_string` value

### C10 — `axon.redeemShare`

- **Handler:** `redeemShare(apiBase)` (`tools/shares.ts`)
- **Behavior:** Prompts for share string, calls POST /share/redeem
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.redeemShare", {}, {"_inputResponse": "axon-share-base64placeholder"})
  ```
- **Success criteria:** Info message contains `"Mounted"`

### C11 — `axon.revokeShare`

- **Handler:** `revokeShare(apiBase)` (`tools/shares.ts`)
- **Behavior:** Fetches active shares, shows QuickPick, calls POST /share/revoke for the selected share
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.revokeShare", {})
  ```
- **Success criteria:** Info message contains `"Revoked access"`

### C12 — `axon.listShares`

- **Handler:** `listShares(apiBase)` (`tools/shares.ts`)
- **Behavior:** Calls GET /share/list, formats output to the Axon output channel
- **Test recipe:** `run_tool(base_url, "cmd:axon.listShares", {})`
- **Success criteria:** `outputLines` contains lines with `"Sharing with others:"` and `"Shared with me:"`

### C13 — `axon.refreshIngest`

- **Handler:** `refreshIngest(apiBase)` (`tools/ingest.ts`)
- **Behavior:** Calls POST /ingest/refresh, shows counts in output channel
- **Test recipe:** `run_tool(base_url, "cmd:axon.refreshIngest", {})`
- **Success criteria:** Output channel contains `"=== Axon Refresh ===" `

### C14 — `axon.listStaleDocs`

- **Handler:** `listStaleDocs(apiBase)` (`tools/ingest.ts`)
- **Behavior:** Prompts for day threshold (default 7), calls GET /collection/stale, shows results
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.listStaleDocs", {}, {"_inputResponse": "14"})
  ```
- **Success criteria:** Output channel contains `"=== Axon Stale Docs"`

### C15 — `axon.clearKnowledgeBase`

- **Handler:** `clearKnowledgeBase(apiBase)` (`tools/ingest.ts`)
- **Behavior:** Shows modal warning dialog, calls POST /clear only if user confirms
- **Test recipe (confirm):**
  ```
  run_tool(base_url, "cmd:axon.clearKnowledgeBase", {}, {"_confirmResponse": "Clear Knowledge Base"})
  ```
- **Success criteria:** Info message contains `"Knowledge base cleared"`
- **Edge cases:** Dismissing the dialog → no API call made

### C16 — `axon.showGraphStatus`

- **Handler:** `showGraphStatus(apiBase)` (`tools/graph.ts`)
- **Behavior:** Calls GET /graph/status, prints to output channel
- **Test recipe:** `run_tool(base_url, "cmd:axon.showGraphStatus", {})`
- **Success criteria:** Output channel contains `"=== Axon GraphRAG Status ==="`

### C17 — `axon.showGraphForQuery`

- **Handler:** inline lambda → `showGraphForQuery(context, query)` (`graph/panel.ts`)
- **Behavior:** Prompts for a query string, opens graph panel
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.showGraphForQuery", {}, {"_inputResponse": "data flow"})
  ```
- **Success criteria:** `panelCount` equals `1`

### C18 — `axon.showGraphForSelection`

- **Handler:** `showGraphForSelection(context)` (`graph/panel.ts`)
- **Behavior:** Uses the active editor selection as the query, opens graph panel
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.showGraphForSelection", {}, {"_selectedText": "vector search"})
  ```
- **Success criteria:** `panelCount` equals `1`

### C19 — `axon.showGovernancePanel`

- **Handler:** `showGovernancePanel(context)` (`governance/panel.ts`)
- **Behavior:** Opens the governance WebviewPanel
- **Test recipe:** `run_tool(base_url, "cmd:axon.showGovernancePanel", {})`
- **Success criteria:** `panelCount` equals `1`

### C20 — `axon.configSetup`

- **Handler:** `runConfigSetupWizard(apiBase, apiKey)` (`tools/config.ts`)
- **Behavior:** 5-step QuickPick/InputBox wizard; applies changes via POST /config/set
- **Test recipe:**
  ```
  run_tool(base_url, "cmd:axon.configSetup", {}, {
    "_quickPickResponse": "ollama",
    "_inputResponse": "llama3.1:8b"
  })
  ```
  Note: the stub returns `_quickPickResponse` for all QuickPick calls (steps 1, 3, 4, 5, and
  the confirmation). Step 2 (model name) is an InputBox and uses `_inputResponse`. Step 5
  uses `canPickMany: true` — the stub returns the single `_quickPickResponse` value, which
  must be an array for the toggle step; the runner wraps it automatically as `items[0]` (a
  single item array). To exercise all five steps use `_inputResponses` for the InputBox steps
  and keep `_quickPickResponse` as a plain string for the provider/embedding/strategy picks.
- **Success criteria:** Info message contains `"Axon config updated:"` or cancelled gracefully
- **Edge cases:**
  - Cancel at any step → no API calls made; all earlier steps discarded
  - Step 5 QuickPick (`canPickMany`) returns the first item only in the stub runner — attended
    tests should validate multi-select behaviour in real VS Code

---

## Chat Participant Test Cases

The `@axon` chat participant is registered as `axon.chat`. In the Node.js runner, use
`tool_name = "chat:axon.chat"` and `tool_input = {"prompt": "..."}`.

### P01 — Basic query

- **Input:** `{"prompt": "What is GraphRAG?"}`
- **Expected API call:** `POST /query` with body `{"query": "What is GraphRAG?", "discuss": false, "raptor": false, "graph_rag": false}`
- **Success criteria:**
  - `toolResult` equals `data.response` — a non-empty answer string
  - No raw JSON in response
- **Edge cases:** API returns empty `response` → participant outputs `"*No answer generated..."*`

### P02 — API unreachable

- **Input:** `{"prompt": "Can you answer me?"}` (with API at wrong port)
- **Success criteria:** `toolResult` starts with `"> **Axon error**: Could not reach the API"`

### P03 — Long prompt

- **Input:** `{"prompt": "Explain in detail how the ingest pipeline processes large PDF files and what chunking strategies are applied, referencing any relevant configuration options."}`
- **Success criteria:** `toolResult` is non-empty; no timeout or truncation error

### P04 — Non-English query (internationalisation smoke test)

- **Input:** `{"prompt": "¿Cómo funciona Axon?"}`
- **Success criteria:** `toolResult` is non-empty; no crash

---

## Automated Test Coverage Summary

| Tool name | Automated test | Test function |
|---|---|---|
| `search_knowledge` | Yes | `test_tool_invocations[search_knowledge...]` |
| `query_knowledge` | Yes | `test_tool_invocations[query_knowledge...]` |
| `ingest_text` | Yes | `test_tool_invocations[ingest_text...]` |
| `ingest_texts` | Yes | `test_ingest_texts_tool` |
| `ingest_url` | Yes | `test_tool_invocations[ingest_url...]` |
| `ingest_path` | Yes | `test_tool_invocations[ingest_path...]` |
| `get_job_status` | Yes | `test_tool_invocations[get_job_status...]` |
| `refresh_ingest` | Yes | `test_tool_invocations[refresh_ingest...]` |
| `get_stale_docs` | Yes | `test_tool_invocations[get_stale_docs...]` |
| `clear_knowledge` | Yes | `test_tool_invocations[clear_knowledge...]` |
| `ingest_image` | Yes | `test_ingest_image_tool` |
| `list_projects` | Yes | `test_tool_invocations[list_projects...]` |
| `switch_project` | Yes | `test_tool_invocations[switch_project...]` |
| `create_project` | Yes | `test_tool_invocations[create_project...]` |
| `delete_project` | Yes | `test_tool_invocations[delete_project...]` |
| `delete_documents` | Yes | `test_tool_invocations[delete_documents...]` |
| `list_knowledge` | Yes | `test_tool_invocations[list_knowledge...]` |
| `update_settings` | Yes | `test_tool_invocations[update_settings...]` |
| `get_current_settings` | Yes | `test_tool_invocations[get_current_settings...]` |
| `list_sessions` | Yes | `test_list_sessions_tool` |
| `get_session` | Yes | `test_get_session_tool` |
| `share_project` | Yes | `test_tool_invocations[share_project...]` |
| `redeem_share` | Yes | `test_tool_invocations[redeem_share...]` |
| `revoke_share` | Yes | `test_tool_invocations[revoke_share...]` |
| `list_shares` | Yes | `test_tool_invocations[list_shares...]` |
| `init_store` | Yes | `test_tool_invocations[init_store...]` |
| `get_store_status` | No — registration check only | manifest contract test |
| `graph_status` | Yes | `test_tool_invocations[graph_status...]` |
| `show_graph` | Yes | `test_show_graph_tool` |
| `graph_finalize` | Yes | `test_tool_invocations[graph_finalize...]` |
| `graph_data` | Yes | `test_graph_data_tool` |
| `get_active_leases` | Yes | `test_get_active_leases_tool` |
| `axon_config_validate` | Yes | `test_axon_config_validate_tool` |
| `axonConfigValidate` (alias) | No | — |
| `axon_config_set` | Yes | `test_axon_config_set_tool` |
| `axonConfigSet` (alias) | No | — |
| `security_status` | Yes | `test_security_status_tool` |
| `security_bootstrap` | Yes | `test_security_bootstrap_tool` |
| `security_unlock` | Yes | `test_security_unlock_tool` |
| `security_lock` | Yes | `test_security_lock_tool` |
| `security_change_passphrase` | Yes | `test_security_change_passphrase_tool` |

**Coverage gaps to address:**
- `get_store_status` — add a parametrized test entry hitting `GET /store/status`
- `axonConfigValidate` and `axonConfigSet` camelCase aliases — add two smoke tests verifying they are registered and produce the same results as their snake_case counterparts

---

## Known Limitations

1. **`show_graph` (AxonShowGraphTool)** — opens a VS Code WebviewPanel. The Node.js stub
   creates a mock panel object so the code path executes, but the actual rendered HTML cannot
   be interacted with outside a real VS Code window.

2. **`ingest_image` (AxonIngestImageTool)** — requires a Copilot model with
   `supportsImageToText`. In the automated runner the model is stubbed via
   `extra._copilotModels`. Attended tests should use a real PNG and observe the Copilot
   vision round-trip. Always supply `alt_text` when running in CI to skip the vision call.

3. **`axon.showGovernancePanel`** — the governance panel webview (`governance/panel.ts`) is
   not covered by the tool test parametrize list; verify via the command path (`cmd:` prefix)
   or in real VS Code.

4. **`axon.showGraphForSelection`** — reads `activeTextEditor.selection`; the stub always
   provides the `_selectedText` extra config value. Real selection behaviour requires a live
   editor.

5. **Security tools (T35–T39)** — all require the Axon server to have been bootstrapped with
   a passphrase. In automated tests, the `live_recorder_server` mock returns the expected
   200 response regardless; real attended tests must perform the bootstrap-unlock-lock cycle
   in order.

6. **`axon.configSetup` wizard (C20)** — the `showQuickPick` stub returns the first item in
   the list by default. Step 5 uses `canPickMany: true`; the stub cannot simulate a
   multi-selection. Use `_quickPickResponse` as a plain string (e.g. `"ollama"`) for provider
   steps; the final confirmation step returns the first item (`"Apply and save"`). Full
   multi-select validation requires an attended test in real VS Code.

7. **camelCase aliases** (`axonConfigValidate`, `axonConfigSet`) — registered alongside the
   snake_case names. Both must appear in `res["registeredTools"]`. No dedicated test asserts
   their functional equivalence; add tests to `test_vscode_extension_tools_e2e.py` if alias
   divergence becomes a risk.
