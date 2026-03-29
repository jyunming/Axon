# Axon Governance Console

Operator-facing dashboard and audit trail for Axon deployments.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Operator surfaces                                              │
│   VS Code panel (axon.showGovernancePanel)                      │
│   curl / HTTP client → /governance/*                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  governance.py          │
              │  AuditStore (SQLite WAL)│◄── emit() calls from
              │  CopilotSessionStore    │    ingest / graph /
              │  emit() background writes│   maintenance / shares
              └─────────────────────────┘
                           │
              ┌────────────▼─────────────────────────────────────┐
              │  SQLite (.governance.db in projects_root)         │
              │  Fallback: .governance.jsonl                      │
              └──────────────────────────────────────────────────┘
```

### Key design choices

| Concern | Decision |
|---------|----------|
| Storage | SQLite WAL — safe for concurrent HTTP workers; no new deps |
| Write latency | `emit()` spawns a daemon thread — callers never block |
| Retention | Default 90 days; pruned on first store open |
| Error safety | All audit writes swallow exceptions at DEBUG level |
| Fallback | JSONL file when SQLite unavailable (read-only FS, Docker) |

---

## API Reference — `/governance/*`

### `GET /governance/overview`

Aggregated operator status. No query parameters.

**Response 200**
```json
{
  "project": "default",
  "maintenance": {
    "maintenance_state": "normal",
    "active_leases": 0
  },
  "graph": {
    "entity_count": 1204,
    "relation_count": 3811,
    "community_count": 47
  },
  "stale_doc_count": 2,
  "active_ingest_jobs": 0,
  "active_leases": [],
  "copilot_sessions_active": 1,
  "project_count": 5
}
```

---

### `GET /governance/audit`

Query the audit log. All parameters are optional.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `project` | string | — | Filter by project name |
| `action` | string | — | Filter by action (see schema below) |
| `surface` | string | — | Filter by surface (`api`, `repl`, `vscode`, …) |
| `status` | string | — | Filter by status (`completed`, `failed`, `started`) |
| `since` | ISO-8601 string | — | Return events at or after this timestamp |
| `limit` | int (1–1000) | `50` | Max events to return |

**Response 200**
```json
{
  "events": [
    {
      "event_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "timestamp": "2026-03-21T10:00:00+00:00",
      "actor": "api",
      "surface": "api",
      "project": "research",
      "action": "ingest_completed",
      "target_type": "file",
      "target_id": "/home/user/docs",
      "status": "completed",
      "details": {"job_id": "a1b2c3", "documents_ingested": 42},
      "request_id": "550e8400-e29b-41d4-a716-446655440000"
    }
  ],
  "count": 1
}
```

#### Audit event schema

| Field | Type | Description |
|-------|------|-------------|
| `event_id` | UUID | Unique event identifier |
| `timestamp` | ISO-8601 | UTC event time |
| `actor` | string | Originating subsystem |
| `surface` | string | API surface |
| `project` | string | Active project at time of action |
| `action` | string | See valid actions below |
| `target_type` | string | Entity type acted on |
| `target_id` | string | Entity identifier |
| `status` | string | `completed`, `failed`, or `started` |
| `details` | object | Action-specific context |
| `request_id` | string | Forwarded `X-Request-ID` header |

#### Valid actions

| Action | Trigger |
|--------|---------|
| `ingest_started` | `POST /ingest` — background job begins |
| `ingest_completed` | Background job finishes successfully |
| `ingest_failed` | Background job throws an exception |
| `delete` | `POST /delete` |
| `graph_finalize` | `POST /graph/finalize` or `POST /governance/graph/rebuild` |
| `maintenance_changed` | `POST /project/maintenance` |
| `share_generated` | `POST /share/generate` |
| `share_redeemed` | `POST /share/redeem` |
| `share_revoked` | `POST /share/revoke` |
| `copilot_session_opened` | Copilot bridge session begins |
| `copilot_session_closed` | Copilot bridge session ends |
| `copilot_session_failed` | Copilot bridge session errors |

---

### `GET /governance/copilot/sessions`

Active and recent Copilot bridge sessions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | int | 20 | Max recent sessions |

**Response 200**
```json
{
  "active": [
    {
      "session_id": "sess-abc123",
      "request_id": "req-xyz",
      "project": "default",
      "opened_at": "2026-03-21T10:05:00+00:00",
      "closed_at": null,
      "is_active": true,
      "error": null
    }
  ],
  "recent": [...],
  "active_count": 1
}
```

---

### `GET /governance/projects`

All projects with their maintenance state and graph stats.

**Response 200**
```json
{
  "projects": [
    {
      "name": "research",
      "maintenance": {"maintenance_state": "normal", "active_leases": 0},
      "graph": {"entity_count": 200, "community_count": 8}
    }
  ],
  "count": 1
}
```

---

### `POST /governance/graph/rebuild`

Audited wrapper for graph community rebuild. Emits `graph_finalize` events before and after.

**Response 200**
```json
{"status": "ok", "community_summary_count": 47}
```

**Error 403** — project is in read-only or draining state.
**Error 500** — rebuild threw an unexpected exception.

---

### `POST /governance/project/maintenance`

Audited wrapper for maintenance state transitions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | — *(required)* | Project name |
| `state` | string | — *(required)* | `normal`, `readonly`, `draining`, or `offline` |

**Response 200**
```json
{
  "status": "ok",
  "project": "research",
  "maintenance_state": "draining",
  "active_leases": 2,
  "epoch": 5
}
```

---

### `POST /governance/copilot/session/{session_id}/expire`

Force-close a stuck Copilot bridge session.

**Response 200**
```json
{"status": "expired", "session_id": "sess-abc123"}
```

**Error 404** — session not found or already closed.

---

## Copilot SSE Lifecycle

```
VS Code Copilot             Axon backend              Governance audit
──────────────              ────────────              ────────────────
POST /copilot/agent ──────► handler opens ──────────► copilot_session_opened
                            SSE stream begins
  data: {type:"created"} ◄──
  data: {type:"text"} ◄────
  data: [DONE] ◄────────── handler closes ──────────► copilot_session_closed
                                 │ (on error)         copilot_session_failed
```

The `session_id` is the `agent_request_id` from the Copilot request body.

---

## Runbook: Graph Rebuild and Drain

1. Check current state:
   ```bash
   curl http://localhost:8000/governance/overview | jq .maintenance
   ```

2. Set project to draining (blocks new writes, waits for in-flight to finish):
   ```bash
   curl -X POST "http://localhost:8000/governance/project/maintenance?name=myproject&state=draining"
   ```

3. Wait for active leases to reach 0:
   ```bash
   watch -n 2 'curl -s http://localhost:8000/governance/overview | jq .maintenance.active_leases'
   ```

4. Trigger graph rebuild:
   ```bash
   curl -X POST http://localhost:8000/governance/graph/rebuild
   ```

5. Restore normal state:
   ```bash
   curl -X POST "http://localhost:8000/governance/project/maintenance?name=myproject&state=normal"
   ```

6. Verify in audit log:
   ```bash
   curl "http://localhost:8000/governance/audit?action=graph_finalize&limit=5"
   ```

---

## Runbook: Copilot Bridge Troubleshooting

### Symptom: VS Code shows no Copilot responses

1. Check active sessions:
   ```bash
   curl http://localhost:8000/governance/copilot/sessions | jq .active
   ```

2. If a session is stuck (opened but never closed), expire it:
   ```bash
   curl -X POST "http://localhost:8000/governance/copilot/session/SESS_ID/expire"
   ```

3. Check audit log for recent Copilot errors:
   ```bash
   curl "http://localhost:8000/governance/audit?action=copilot_session_failed&limit=10"
   ```

### Symptom: `/llm/copilot/tasks` returns empty repeatedly

The VS Code extension's Copilot LLM worker may have stopped polling.
Restart the extension (`Developer: Reload Window`) or check the Axon output channel.

---

## Maintenance State Reference

| State | Writes allowed | New leases | Notes |
|-------|---------------|------------|-------|
| `normal` | Yes | Yes | Default operational state |
| `readonly` | No | No | Reads only; good for backups |
| `draining` | No (new) | No | In-flight writes complete; use before offline |
| `offline` | No | No | Full lock; no operations |

Transition path for zero-downtime maintenance: `normal → draining → (wait) → readonly/offline`.
Transition back: `offline/readonly → normal`.

---

## VS Code: Show Governance Panel

Command Palette → **Axon: Show Governance Panel** (`axon.showGovernancePanel`)

The panel polls `/governance/overview` every 30 seconds and displays:
- Project name and maintenance state (colour-coded)
- Active write leases
- Graph entity and community counts
- Stale document count
- Active ingest jobs
- Active Copilot sessions
- Total project count

**Buttons:**
- **Rebuild Graph** — calls `POST /governance/graph/rebuild`
- **Set Maintenance** — quick-pick menu to transition project state
- **Refresh Now** — immediate poll
