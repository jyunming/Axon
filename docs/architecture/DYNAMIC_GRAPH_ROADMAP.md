# Dynamic Graph Backend — Roadmap

## Scope

### What this roadmap covers

Axon currently has one graph strategy: `GraphRagMixin`, wired directly into
`OpenStudioBrain` via inheritance. This roadmap introduces a **stable backend
interface** that decouples graph strategy from the core brain, and adds a second
backend — **Dynamic Graph** — for projects with evolving, timestamped knowledge.

### V1 constraints (locked)

| Constraint | Rule |
|---|---|
| Backend enum | `graphrag \| dynamic_graph \| none` — no other values in v1 |
| Immutability | `graph_backend` is set at project creation and cannot change |
| Offline | Both backends must work with zero network access |
| No external graph DB | Dynamic Graph uses local SQLite only |
| Code graph orthogonality | `code_graph.py` (AST) is a separate axis; not part of this enum |
| Mixed-backend scope | Federation at retrieval/answer layer only; no unified graph model |

### Explicitly out of scope for v1

- Fuzzy entity deduplication / alias resolution (exact canonical name only)
- External graph databases (Neo4j, Memgraph, etc.)
- `code_graph` as a backend enum value
- Real-time streaming graph updates
- Graph export / import across projects

---

## Decisions locked

### Backend contract

```python
class GraphBackend(Protocol):
    def ingest(self, chunks: list[Chunk]) -> IngestResult: ...
    def retrieve(self, query: str, cfg: RetrievalConfig) -> list[GraphContext]: ...
    def finalize(self, force: bool = False) -> FinalizationResult: ...
    def clear(self) -> None: ...
    def delete_documents(self, chunk_ids: list[str]) -> None: ...
    def status(self) -> dict: ...
    def graph_data(self, filters: GraphDataFilters | None = None) -> GraphPayload: ...
```

`render()` is **not** on the contract. `graph_render.py` calls `backend.graph_data()`.

### GraphContext fields

```python
@dataclass
class GraphContext:
    context_id: str
    context_type: str            # "entity" | "community" | "fact" | "episode"
    text: str
    score: float                 # normalized [0, 1] by the backend
    rank: int                    # position within this backend's ranked list
    backend_id: str              # "graphrag" | "dynamic_graph"
    source_id: str
    source_doc_id: str | None
    source_chunk_id: str | None
    metadata: dict
    valid_at: datetime | None    # None for GraphRAG
    invalid_at: datetime | None  # None for GraphRAG and current facts
    evidence_ids: list[str] | None
```

### Mixed-backend federation

Weighted RRF (Reciprocal Rank Fusion) over per-backend ranked lists.
Default: equal weights. Dedup by `source_chunk_id` when present,
otherwise by `(backend_id, context_id)`.

```
rrf_score(d) = Σ weight_i / (k + rank_of_d_in_backend_i)    k=60
```

### Dynamic Graph SQLite schema

```sql
CREATE TABLE episodes (
    id TEXT PRIMARY KEY,
    source_chunk_id TEXT,
    source_doc_id TEXT,
    project_id TEXT NOT NULL,
    ingested_at TEXT NOT NULL,     -- ISO-8601
    reference_time TEXT,           -- ISO-8601, nullable
    content TEXT,
    summary_json TEXT
);

CREATE TABLE entities (
    id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE facts (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES episodes(id),
    subject_entity_id TEXT NOT NULL REFERENCES entities(id),
    relation TEXT NOT NULL,
    object_entity_id TEXT REFERENCES entities(id),
    object_text TEXT,              -- when object is not an entity
    scope_key TEXT NOT NULL,       -- derived, not prompt-generated
    valid_at TEXT NOT NULL,
    invalid_at TEXT,               -- NULL = currently active
    status TEXT NOT NULL DEFAULT 'active',  -- active | superseded | conflicted
    confidence REAL DEFAULT 1.0,
    created_at TEXT NOT NULL
);

CREATE TABLE fact_evidence (
    fact_id TEXT NOT NULL REFERENCES facts(id),
    chunk_id TEXT NOT NULL,
    source_doc_id TEXT,
    PRIMARY KEY (fact_id, chunk_id)
);

-- Indexes
CREATE INDEX idx_facts_subject_relation_status ON facts(subject_entity_id, relation, status);
CREATE INDEX idx_facts_temporal ON facts(valid_at, invalid_at);
CREATE INDEX idx_facts_scope ON facts(scope_key, relation, status);
CREATE INDEX idx_entities_name ON entities(canonical_name);
```

WAL mode enabled on every connection open.

### Conflict resolution

- **Default policy**: append and preserve (most relations are multi-valued)
- **Exclusive relation families**: supersede prior active facts within the same scope
- **Same timestamp, incompatible values on exclusive relation**: mark both `conflicted`
- `scope_key` is derived in Axon code from relation metadata — not prompt-generated
  - Exclusive relations: `scope_key = f"{subject_entity_id}:{relation}:{object_ref}"`
  - Non-exclusive: `scope_key = f"{subject_entity_id}:{relation}"`

### Relation type registry

Built-in code defaults + optional per-project YAML override at
`<project_dir>/dynamic_graph_config.yaml`. SQLite is not the right
place for policy. Relation types include `cardinality` (one/many) and
`conflict_policy` (supersede/preserve/conflict).

### Entity canonical name policy

Exact match only in v1. No fuzzy deduplication. Explicit aliases may be added
as a future extension. False merges are worse than duplicates in a local graph.

### Legacy project migration

Lazy: projects without `graph_backend` in `meta.json` default to `"graphrag"`
on first read. No file rewrite until the next `ensure_project()` call.

### `none` backend semantics

- `retrieve()` → `[]`
- `finalize()` → `FinalizationResult(status="not_applicable")`
- `graph_data()` → empty `GraphPayload`
- `status()` → `{"backend": "none", "enabled": False}`

---

## Milestones

### M0 — Design Freeze *(internal, no version bump)*

> "All architectural decisions are agreed and written down before any code moves."

- ADR written: backend immutability, mixed-backend limits, offline constraint,
  Dynamic Graph vs GraphRAG distinction
- `src/axon/graph_backends/base.py` exists with type stubs only — no behavior
- `meta.json` schema extension documented (`graph_backend`, `created_with_version`)
- Conflict resolution policy frozen (relation registry design agreed)
- SQLite schema finalized and written to this doc
- Mixed-backend score normalization rule frozen (weighted RRF, equal default weights)
- `GraphContext` field list frozen
- `GraphDataFilters` field list frozen (`at_time`, `node_ids`, `relation_types`, `limit`)
- Legacy migration policy documented (lazy, `"graphrag"` default)

**Exit criteria:** no open design questions; every team member can describe the
backend boundary, conflict semantics, and federation behavior without looking
at code.

---

### M1 — GraphRAG Regression Harness *(internal, no version bump)*

> "Current GraphRAG behavior is black-box testable before the refactor starts."

- Fixture documents created in `tests/fixtures/graphrag_parity/` (6 scenarios:
  software guide, paper abstract, issue thread, stdlib docs, codebase, project doc)
- Canned LLM extraction responses defined for each fixture (pipe-delimited format
  matching `_extract_entities` and `_extract_relations` call signatures)
- Parity test suite in `tests/test_graphrag_parity.py`:
  - Ingest → entity extraction → relation extraction → community build → query →
    render payload, end-to-end for each fixture
  - LLM boundary mocked with canned responses; all other logic runs real
  - GLiNER/REBEL path tested under `@pytest.mark.slow` marker with real model
    weights; not in default CI gate
- Architecture enforcement test in `tests/test_architecture.py`:
  - Uses Python `ast` module (not grep) to parse `main.py`, `query_router.py`,
    `api_routes/graph.py`
  - Asserts zero direct attribute access to `_entity_graph`, `_relation_graph`,
    `_community_summaries`, `_claims_graph` outside the `GraphRagBackend` adapter
  - **Written here, expected to fail** — turns green when Phase 2 completes

**Exit criteria:** `test_graphrag_parity.py` passes against the unmodified codebase,
proving the current behavior is observable. Architecture test is committed and failing.

---

### M2 — Backend Boundary Refactor *(internal, no version bump)*

> "GraphRAG runs through the backend interface; nothing visible to users changes."

- `src/axon/graph_backends/__init__.py`
- `src/axon/graph_backends/base.py` — `GraphBackend` protocol, `GraphContext`,
  `GraphDataFilters`, `IngestResult`, `FinalizationResult`, `GraphPayload`
- `src/axon/graph_backends/factory.py` — `get_backend(project_meta) -> GraphBackend`
- `src/axon/graph_backends/graphrag_backend.py` — wraps `GraphRagMixin` methods;
  shims hold `_entity_graph` etc. internally during the landing period
- `src/axon/graph_backends/none_backend.py` — no-op implementation
- Graph operations in `main.py`, `query_router.py`, `api_routes/graph.py` routed
  through backend delegation — direct attribute access replaced
- Architecture test (`test_architecture.py`) turns green
- M1 parity suite (`test_graphrag_parity.py`) remains fully green — zero drift

**Exit criteria:** architecture test green, parity suite green, no direct
access of `_entity_graph` / `_relation_graph` / `_community_summaries` /
`_claims_graph` outside the adapter. No user-visible behavior change.

---

### v0.2 — Backend-Aware Projects *(Phase 3 — first package version bump)*

> "Every project declares its graph strategy at creation; the API reflects it."

- `graph_backend` field added to `meta.json` on project creation
- Valid values: `graphrag | dynamic_graph | none`
- Post-create mutation blocked in v1 with a clear error
- Lazy migration for legacy projects (default `"graphrag"` on first read)
- `/project/new`, `/project/list`, `/project/switch` return `graph_backend`
- `DynamicGraphBackend` stub exists — returns empty status safely, no 500s
- MCP tools `create_project`, `list_projects` updated with `graph_backend` field
- Docs updated: `ADMIN_REFERENCE.md`, `API_REFERENCE.md`, `MCP_TOOLS.md`

**Tests:**
- `meta.json` round-trip with and without `graph_backend`
- Immutability enforcement (post-create change attempt → error)
- Legacy project loading defaults to `"graphrag"`
- Project list and switch behavior for all three backend values
- API and MCP request/response schema coverage

**Exit criteria:** creating, listing, switching, and loading projects is
backend-aware and stable. Package version bumped to `0.2.0`.

---

### v0.3 — Dynamic Graph Storage *(Phase 4)*

> "Dynamic Graph projects persist evolving episodes and facts locally with no rebuild cost."

- `src/axon/dynamic_graph/models.py` — `Episode`, `Entity`, `Fact`, `FactEvidence`
- `src/axon/dynamic_graph/store.py` — SQLite store in WAL mode, full CRUD,
  temporal filtering, fact supersede/conflict logic
- `src/axon/dynamic_graph/` — `__init__.py`
- `src/axon/graph_backends/dynamic_graph_backend.py` — real implementation replacing
  the stub; wires to `store.py`
- Relation registry (`models.py`) with built-in defaults and YAML override path
- `scope_key` derivation from relation metadata (code-derived, not prompt-generated)
- Indexes on temporal and entity lookup paths
- Clear and delete lifecycle: `clear()` truncates all tables; `delete_documents()`
  invalidates facts whose only evidence is the given chunks

**Tests:**
- SQLite CRUD for all four tables
- `scope_key` derivation correctness for exclusive and non-exclusive relations
- Time-window filtering: `at_time` point-in-time and range queries
- Fact supersede: new episode overwrites exclusive fact → old `invalid_at` set
- Conflict: same timestamp, incompatible exclusive values → both `status=conflicted`
- Duplicate entity merge (exact canonical name match)
- Clear/delete lifecycle
- Restart persistence: data survives process restart
- Entity canonical name drift: same entity under 3 name variants produces 3 nodes
  (no fuzzy merge — expected and documented behavior)
- 10k sequential episode ingests confirm no full-dataset rebuild behavior

**Exit criteria:** Dynamic Graph storage is incremental, offline-capable, and
test-verified. Package version bumped to `0.3.0`.

---

### v0.4 — Dynamic Graph Queries *(Phase 5)*

> "Dynamic Graph projects answer current-state and historical questions; mixed-backend projects federate at retrieval level."

> **v0.3.2 partial delivery (2026-05):** the milestone's user-facing surfaces
> are now exposed via `POST /graph/retrieve` REST + `graph_retrieve` MCP +
> `/graph retrieve` REPL + `--graph-retrieve` CLI:
>
> - Point-in-time query (`RetrievalConfig.point_in_time`) — surfaced.
> - `/graph/finalize` returns `status: "not_applicable"` for Dynamic Graph — surfaced (capability flag).
> - Conflict inspection (`/graph/conflicts` + `graph_conflicts` MCP) — surfaced.
> - Per-query federated weight override (`RetrievalConfig.federation_weights`) — surfaced.
>
> The internal storage and retrieval primitives were already shipped in v0.3.0/v0.3.1 — v0.3.2 is the surface layer. The full v0.4.0 deliverable still requires plumbing `_graph_backend.retrieve()` into the main `/query` pipeline (today only `status()` and `graph_data()` are wired); the dedicated `POST /graph/retrieve` route exposes the backend protocol directly without touching `query_router.py`.

- `src/axon/dynamic_graph/ingest.py` — episode creation, entity extraction (AST
  path for code, LLM path for prose), fact extraction, duplicate merge
- `src/axon/dynamic_graph/retrieval.py` — current-state facts, point-in-time facts,
  latest episodes, entity-neighborhood expansion
- `DynamicGraphBackend.ingest()` and `retrieve()` fully implemented
- `/graph/finalize` returns `{"status": "not_applicable"}` for Dynamic Graph projects
- `graph_render.py` updated to consume `backend.graph_data()` instead of direct
  `brain._entity_graph` access; Dynamic Graph renders entity timeline view
- Mixed-backend federation: weighted RRF over per-backend ranked lists, dedup by
  `source_chunk_id`, equal weights by default
- Docs: Dynamic Graph usage guide, GraphRAG vs Dynamic Graph comparison,
  federation semantics, troubleshooting (missing timestamps, conflicting facts,
  unsupported finalize)

**Tests:**
- Ingest → query → history round-trip for prose and code fixtures
- Point-in-time query: "what was true at T?" returns correct facts
- Fact invalidation after supersede: query at T+1 excludes old fact
- Contradiction storm on one exclusive relation family: N alternating updates
  produce correct `conflicted` / `superseded` chain
- Mixed-backend federation: one GraphRAG project + one Dynamic Graph project,
  federated query returns results from both, deduplicated and ranked
- Renderer normalization: Dynamic Graph `graph_data()` payload accepted by
  `graph_render.py` without modification to the renderer
- `/graph/finalize` returns `not_applicable` for Dynamic Graph

**Exit criteria:** Dynamic Graph answers dynamic-state and historical questions.
Mixed-backend federation works at retrieval level. Package version bumped to `0.4.0`.

---

### v1.0 — Hardening *(Phase 6)*

> "Both backends are stress-tested, provenance-clean, documented, and ready for production."

- GraphRAG prompt text in `graph_rag.py` rewritten to reduce provenance risk
- Capability flags: unsupported operations return explicit errors, not silent no-ops
- Performance baselines measured and documented
- Full docs: user guide, troubleshooting, offline guide, benchmark notes
- `docs/TROUBLESHOOTING.md` and `docs/OFFLINE_GUIDE.md` updated for both backends

**Stress tests required to pass:**
- 50 concurrent read-only queries against a large GraphRAG project
- 20 concurrent Dynamic Graph queries + 5 concurrent ingests (one project)
- 10k sequential episode ingests: confirm no O(total dataset) rebuild
- Contradiction storm on one exclusive relation family (repeated alternating updates)
- Mixed-backend federated query under concurrent load
- Repeated project switching across many projects (backend lifecycle leak check)
- Offline boot and query for both backends with no network access
- Entity canonical name drift: 3 name variants for same entity → 3 nodes, documented

**Exit criteria:** all stress tests pass, both backends have documented limits and
performance baselines, provenance language in GraphRAG prompts reviewed.
Package version bumped to `1.0.0`.

---

## Sprint Timeline

Starting **2026-04-06**.

| Sprint | Dates | Milestone | Deliverable |
|---|---|---|---|
| S1 | Apr 06–12 | M0 | ADR, `base.py` stubs, schema, relation registry |
| S2 | Apr 13–19 | M1 | Fixture files, parity suite, architecture test (failing) |
| S3 | Apr 20–26 | M2a | Factory, `GraphRagBackend` adapter, shim landing |
| S4 | Apr 27–May 03 | M2b | Shim removal, arch test green, parity suite green |
| S5 | May 04–10 | **v0.2.0** | Backend-aware projects, API/MCP, docs |
| S6 | May 11–17 | v0.3a | SQLite schema, models, CRUD |
| S7 | May 18–24 | **v0.3.0** | Indexes, lifecycle, conflict/supersede, 10k test |
| S8 | May 25–31 | v0.4a | Episode ingest, entity/fact extraction |
| S9 | Jun 01–07 | v0.4b | Temporal query, contradiction handling |
| S10 | Jun 08–14 | **v0.4.0** | Federation, renderer, finalize no-op, docs |
| S11 | Jun 15–21 | v1.0a | Stress tests, benchmarks |
| S12 | Jun 22–28 | **v1.0.0** | Provenance cleanup, full docs, release |

---

## Performance Targets

| Metric | Target |
|---|---|
| Backend abstraction overhead (GraphRAG) | ≤ 10% P95 latency added to existing paths |
| Dynamic Graph ingest (steady state) | No O(total dataset) work per new episode |
| SQLite current-state lookup | P95 < 200 ms at 100k facts, excluding LLM |
| SQLite point-in-time lookup | P95 < 200 ms at 100k facts, excluding LLM |
| Mixed-backend retrieval overhead | P95 < 500 ms, excluding LLM |
| Memory growth under sustained Dynamic Graph ingest | Bounded by persisted SQLite, not in-memory cache |

---

## Test Strategy

| Layer | Scope |
|---|---|
| Unit | Metadata round-trip, backend factory, SQLite store, dedup, conflict handling, time filtering, scope_key derivation, relation registry |
| Integration | Project create/list/switch, GraphRAG parity (M1 suite), Dynamic Graph ingest/query, mixed-backend federation |
| API/MCP | Request/response schema coverage for all project and graph routes and tools |
| Architecture | AST-based enforcement of shim removal (no direct graph-state access outside adapter) |
| E2E | One GraphRAG project, one Dynamic Graph project, one mixed federated query scenario |
| Stress | All 8 cases listed in v1.0 exit criteria |
| Slow/optional | GLiNER/REBEL path under `@pytest.mark.slow` marker; not in default CI |
