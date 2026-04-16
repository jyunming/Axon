"""DynamicGraphBackend — SQLite-WAL temporal graph (v0.3).

Implements the ``GraphBackend`` Protocol using SQLite in WAL mode.

Schema mirrors Graphiti's bi-temporal model:
  episodes     → source_chunk_id, content, reference_time
  entities     → canonical_name, entity_type, description, first/last_seen_at
  facts        → subject, relation, object, valid_at, invalid_at, status, scope_key
  fact_evidence → fact_id, chunk_id, episode_id

Conflict resolution:
  - Append-and-preserve (default): scope_key = NULL; all facts kept
  - Exclusive override: scope_key = "subject:relation"; new fact supersedes old

LLM extraction uses the same pipe-delimited prompt format as GraphRagMixin so
prompt outputs are interchangeable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from axon.graph_backends.base import (
    FinalizationResult,
    GraphContext,
    GraphDataFilters,
    GraphPayload,
    IngestResult,
    RetrievalConfig,
)

if TYPE_CHECKING:
    pass

BACKEND_ID = "dynamic_graph"
logger = logging.getLogger("Axon")

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS episodes (
    episode_id   TEXT PRIMARY KEY,
    chunk_id     TEXT NOT NULL,
    content      TEXT NOT NULL,
    reference_time TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id    TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL UNIQUE,
    entity_type  TEXT NOT NULL DEFAULT 'UNKNOWN',
    description  TEXT NOT NULL DEFAULT '',
    first_seen_at TEXT NOT NULL,
    last_seen_at  TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS facts (
    fact_id      TEXT PRIMARY KEY,
    subject      TEXT NOT NULL,
    relation     TEXT NOT NULL,
    object       TEXT NOT NULL,
    valid_at     TEXT NOT NULL,
    invalid_at   TEXT,
    status       TEXT NOT NULL DEFAULT 'active',
    scope_key    TEXT,
    confidence   REAL NOT NULL DEFAULT 1.0,
    metadata     TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS fact_evidence (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id      TEXT NOT NULL,
    chunk_id     TEXT NOT NULL,
    episode_id   TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (fact_id) REFERENCES facts(fact_id)
);

CREATE INDEX IF NOT EXISTS idx_entities_name   ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_facts_subject   ON facts(subject, status);
CREATE INDEX IF NOT EXISTS idx_facts_object    ON facts(object, status);
CREATE INDEX IF NOT EXISTS idx_facts_scope     ON facts(scope_key) WHERE scope_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_evidence_fact   ON fact_evidence(fact_id);
CREATE INDEX IF NOT EXISTS idx_evidence_chunk  ON fact_evidence(chunk_id);
"""

# Relation types that are mutually exclusive (new supersedes old for same subject).
_EXCLUSIVE_RELATIONS: frozenset[str] = frozenset(
    {
        "IS_CEO_OF",
        "IS_CTO_OF",
        "IS_CFO_OF",
        "LEADS",
        "HEADQUARTERS_IN",
        "CURRENTLY_LIVES_IN",
        "MARRIED_TO",
        "CURRENT_VERSION",
    }
)

# ---------------------------------------------------------------------------
# Extraction prompts
# ---------------------------------------------------------------------------

_ENTITY_PROMPT_TMPL = (
    "Extract the key named entities from the following text.\n"
    "For each entity output one line:\n"
    "  ENTITY_NAME | ENTITY_TYPE | one-sentence description\n"
    "ENTITY_TYPE must be one of: PERSON, ORGANIZATION, GEO, EVENT, CONCEPT, PRODUCT\n"
    "No bullets, numbering, or extra text. If no entities, output nothing.\n\n{text}"
)

_FACT_PROMPT_TMPL = (
    "Extract key relationships from the following text as factual statements.\n"
    "For each relationship output one line:\n"
    "  SUBJECT | RELATION | OBJECT | one-sentence description | confidence 0-10\n"
    "SUBJECT and OBJECT must be named entities or noun phrases.\n"
    "RELATION should be a short verb phrase in UPPER_SNAKE_CASE (e.g. WORKS_FOR, FOUNDED_BY).\n"
    "No bullets, numbering, or extra text. If no relationships, output nothing.\n\n{text}"
)

_ENTITY_SYSTEM = "You are a named entity extraction specialist."
_FACT_SYSTEM = "You are a knowledge graph extraction specialist."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha8(text: str) -> str:
    """8-char hex digest of SHA-256 of text — used for deterministic IDs."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _norm(name: str) -> str:
    """Normalize entity name: strip and lowercase."""
    return name.strip().lower()


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class DynamicGraphBackend:
    """SQLite-WAL temporal graph backend satisfying ``GraphBackend`` Protocol.

    Args:
        brain: An ``AxonBrain`` instance.  Uses ``brain.config.bm25_path``
               for the database path, ``brain.llm`` for extraction LLM calls,
               and ``brain.config`` for retrieval configuration.
    """

    def __init__(self, brain: Any) -> None:
        self._brain = brain
        _base = Path(getattr(brain.config, "bm25_path", "."))
        self._db_path = _base / ".dynamic_graph.db"
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection = self._init_db()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_db(self) -> sqlite3.Connection:
        """Open (or create) the SQLite database and apply the schema."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        return conn

    def _execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cur.fetchall()

    def _executemany(self, sql: str, params_seq: list[tuple]) -> None:
        with self._lock:
            self._conn.executemany(sql, params_seq)
            self._conn.commit()

    def _write(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    def _llm_complete(self, prompt: str, system_prompt: str = "") -> str:
        """Call the brain's LLM; return empty string on error."""
        try:
            llm = getattr(self._brain, "llm", None)
            if llm is None:
                return ""
            kwargs: dict[str, Any] = {}
            if system_prompt:
                kwargs["system_prompt"] = system_prompt
            return llm.complete(prompt, **kwargs) or ""
        except Exception as exc:
            logger.debug("DynamicGraphBackend LLM call failed: %s", exc)
            return ""

    def _extract_entities(self, text: str) -> list[dict]:
        """Return list of {name, type, description} from text."""
        raw = self._llm_complete(
            _ENTITY_PROMPT_TMPL.format(text=text[:3000]),
            system_prompt=_ENTITY_SYSTEM,
        )
        entities: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                entities.append(
                    {"name": parts[0], "type": parts[1].upper(), "description": parts[2]}
                )
            elif len(parts) == 2:
                entities.append({"name": parts[0], "type": "UNKNOWN", "description": parts[1]})
            elif parts[0]:
                entities.append({"name": parts[0], "type": "UNKNOWN", "description": ""})
        return entities[:20]

    def _extract_facts(self, text: str) -> list[dict]:
        """Return list of {subject, relation, object, description, confidence}."""
        raw = self._llm_complete(
            _FACT_PROMPT_TMPL.format(text=text[:3000]),
            system_prompt=_FACT_SYSTEM,
        )
        facts: list[dict] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                conf = 1.0
                if len(parts) >= 5:
                    try:
                        conf = float(parts[4]) / 10.0
                    except ValueError:
                        pass
                facts.append(
                    {
                        "subject": parts[0],
                        "relation": parts[1].upper().replace(" ", "_"),
                        "object": parts[2],
                        "description": parts[3] if len(parts) >= 4 else "",
                        "confidence": max(0.0, min(1.0, conf)),
                    }
                )
        return facts[:20]

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def _upsert_entity(self, name: str, entity_type: str, description: str, now: str) -> str:
        """Insert or update an entity; return canonical_name."""
        canon = _norm(name)
        entity_id = _sha8(canon)
        with self._lock:
            existing = self._conn.execute(
                "SELECT entity_id FROM entities WHERE canonical_name = ?", (canon,)
            ).fetchone()
            if existing:
                self._conn.execute(
                    "UPDATE entities SET last_seen_at = ?, entity_type = CASE WHEN entity_type = 'UNKNOWN' THEN ? ELSE entity_type END WHERE canonical_name = ?",
                    (now, entity_type, canon),
                )
            else:
                self._conn.execute(
                    "INSERT INTO entities (entity_id, canonical_name, entity_type, description, first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (entity_id, canon, entity_type, description, now, now),
                )
            self._conn.commit()
        return canon

    def _upsert_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
        description: str,
        confidence: float,
        chunk_id: str,
        episode_id: str,
        now: str,
    ) -> str:
        """Insert a fact; supersede conflicting exclusive facts. Return fact_id."""
        subj_norm = _norm(subject)
        obj_norm = _norm(obj)
        rel_upper = relation.upper()

        # Determine scope_key for conflict resolution
        scope_key: str | None = None
        if rel_upper in _EXCLUSIVE_RELATIONS:
            scope_key = f"{subj_norm}:{rel_upper}"

        fact_id = _sha8(f"{subj_norm}|{rel_upper}|{obj_norm}|{now}")

        with self._lock:
            # Supersede existing active facts with the same scope_key (exclusive families)
            if scope_key is not None:
                self._conn.execute(
                    "UPDATE facts SET status = 'superseded', invalid_at = ? WHERE scope_key = ? AND status = 'active'",
                    (now, scope_key),
                )

            self._conn.execute(
                "INSERT OR IGNORE INTO facts (fact_id, subject, relation, object, valid_at, status, scope_key, confidence, metadata) VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)",
                (
                    fact_id,
                    subj_norm,
                    rel_upper,
                    obj_norm,
                    now,
                    scope_key,
                    confidence,
                    json.dumps({"description": description}),
                ),
            )
            self._conn.execute(
                "INSERT INTO fact_evidence (fact_id, chunk_id, episode_id) VALUES (?, ?, ?)",
                (fact_id, chunk_id, episode_id),
            )
            self._conn.commit()
        return fact_id

    # ------------------------------------------------------------------
    # GraphBackend protocol
    # ------------------------------------------------------------------

    def ingest(self, chunks: list[dict]) -> IngestResult:
        """Extract entities and facts from *chunks*; store in SQLite.

        Uses the brain's LLM to extract named entities and relational facts
        from each chunk's text.  Entities are upserted by canonical name;
        facts are inserted with temporal validity; exclusive facts supersede
        prior conflicting facts.
        """
        now = _now_iso()
        entities_added = 0
        relations_added = 0

        for chunk in chunks:
            text = chunk.get("text", chunk.get("page_content", ""))
            chunk_id = chunk.get("id", _sha8(text[:200]))
            if not text:
                continue

            episode_id = _sha8(f"ep|{chunk_id}|{now}")

            # Persist episode
            with self._lock:
                self._conn.execute(
                    "INSERT OR IGNORE INTO episodes (episode_id, chunk_id, content, reference_time, metadata) VALUES (?, ?, ?, ?, ?)",
                    (
                        episode_id,
                        chunk_id,
                        text[:10000],
                        now,
                        json.dumps(chunk.get("metadata", {})),
                    ),
                )
                self._conn.commit()

            # Entity extraction
            entities = self._extract_entities(text)
            for ent in entities:
                self._upsert_entity(ent["name"], ent["type"], ent["description"], now)
                entities_added += 1

            # Fact extraction
            facts = self._extract_facts(text)
            for fact in facts:
                self._upsert_fact(
                    subject=fact["subject"],
                    relation=fact["relation"],
                    obj=fact["object"],
                    description=fact["description"],
                    confidence=fact["confidence"],
                    chunk_id=chunk_id,
                    episode_id=episode_id,
                    now=now,
                )
                relations_added += 1

        return IngestResult(
            entities_added=entities_added,
            relations_added=relations_added,
            chunks_processed=len(chunks),
            backend_id=BACKEND_ID,
        )

    def retrieve(
        self,
        query: str,
        cfg: RetrievalConfig | None = None,
        existing_results: list[dict] | None = None,
    ) -> list[GraphContext]:
        """Return active facts related to *query* as :class:`GraphContext` objects.

        Steps:
        1. Extract query entities (via LLM or simple tokenization).
        2. Find active facts where subject or object matches a query entity.
        3. Return as GraphContext objects (each fact = one context).
        """
        top_k = (cfg.top_k if cfg else None) or 10

        # Step 1: extract query entities (lightweight — just tokenize query terms)
        query_terms = [_norm(w) for w in query.split() if len(w) >= 3]

        if not query_terms:
            return []

        # Step 2: find active facts matching query terms
        placeholders = ",".join("?" for _ in query_terms)
        rows = self._execute(
            f"SELECT fact_id, subject, relation, object, valid_at, confidence, metadata "
            f"FROM facts "
            f"WHERE status = 'active' AND (subject IN ({placeholders}) OR object IN ({placeholders})) "
            f"ORDER BY valid_at DESC LIMIT ?",
            (*query_terms, *query_terms, top_k),
        )

        # Step 3: convert to GraphContext
        _existing_ids = {r.get("id") for r in (existing_results or []) if r.get("id")}
        contexts: list[GraphContext] = []
        for rank, row in enumerate(rows):
            text = f"{row['subject']} {row['relation'].replace('_', ' ').lower()} {row['object']}"
            meta = json.loads(row["metadata"] or "{}")
            desc = meta.get("description", "")
            if desc:
                text = f"{text}: {desc}"

            ctx = GraphContext(
                context_id=row["fact_id"],
                context_type="fact",
                text=text,
                score=float(row["confidence"]),
                rank=rank,
                backend_id=BACKEND_ID,
                source_chunk_id="",
                metadata={"valid_at": row["valid_at"], **meta},
                valid_at=_parse_dt(row["valid_at"]),
                matched_entity_names=[row["subject"], row["object"]],
            )
            if ctx.context_id not in _existing_ids:
                contexts.append(ctx)

        return contexts

    def finalize(self, force: bool = False) -> FinalizationResult:
        """No-op — dynamic graph is episodic; no community detection step."""
        return FinalizationResult(backend_id=BACKEND_ID)

    def clear(self) -> None:
        """Delete all rows from all tables."""
        with self._lock:
            self._conn.executescript(
                "DELETE FROM fact_evidence; DELETE FROM facts; DELETE FROM entities; DELETE FROM episodes;"
            )
            self._conn.commit()

    def delete_documents(self, chunk_ids: list[str]) -> None:
        """Remove episodes, evidence rows, and orphaned facts for *chunk_ids*."""
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._lock:
            # Mark covered facts as superseded if all evidence is removed
            self._conn.execute(
                f"DELETE FROM fact_evidence WHERE chunk_id IN ({placeholders})",
                tuple(chunk_ids),
            )
            # Orphaned facts (no remaining evidence) → supersede
            self._conn.execute(
                "UPDATE facts SET status = 'superseded', invalid_at = ? "
                "WHERE fact_id NOT IN (SELECT DISTINCT fact_id FROM fact_evidence) AND status = 'active'",
                (_now_iso(),),
            )
            self._conn.execute(
                f"DELETE FROM episodes WHERE chunk_id IN ({placeholders})",
                tuple(chunk_ids),
            )
            self._conn.commit()

    def status(self) -> dict:
        """Return lightweight counts from all tables."""
        rows = self._execute(
            "SELECT "
            "(SELECT COUNT(*) FROM episodes) AS episodes, "
            "(SELECT COUNT(*) FROM entities) AS entities, "
            "(SELECT COUNT(*) FROM facts WHERE status = 'active') AS active_facts, "
            "(SELECT COUNT(*) FROM facts WHERE status = 'superseded') AS superseded_facts"
        )
        row = rows[0] if rows else {}
        return {
            "backend": BACKEND_ID,
            "episodes": row["episodes"] if row else 0,
            "entities": row["entities"] if row else 0,
            "active_facts": row["active_facts"] if row else 0,
            "superseded_facts": row["superseded_facts"] if row else 0,
        }

    def graph_data(self, filters: GraphDataFilters | None = None) -> GraphPayload:
        """Return current active facts as a nodes + links payload."""
        limit = (filters.limit if filters else None) or 500
        rows = self._execute(
            "SELECT fact_id, subject, relation, object, confidence "
            "FROM facts WHERE status = 'active' ORDER BY confidence DESC LIMIT ?",
            (limit,),
        )

        # Collect unique node names
        node_names: dict[str, dict] = {}
        links: list[dict] = []

        for row in rows:
            subj = row["subject"]
            obj = row["object"]

            for name in (subj, obj):
                if name not in node_names:
                    node_names[name] = {"id": name, "name": name, "label": name, "type": "entity"}

            links.append(
                {
                    "source": subj,
                    "target": obj,
                    "label": row["relation"].replace("_", " ").lower(),
                    "relation": row["relation"],
                    "weight": float(row["confidence"]),
                }
            )

        # Apply entity_type filter if requested
        if filters and filters.entity_types:
            allowed = set(filters.entity_types)
            entity_rows = self._execute("SELECT canonical_name, entity_type FROM entities")
            type_map = {r["canonical_name"]: r["entity_type"] for r in entity_rows}
            for name, node in node_names.items():
                node["type"] = type_map.get(name, "entity")
            node_names = {n: d for n, d in node_names.items() if d["type"] in allowed}

        return GraphPayload(nodes=list(node_names.values()), links=links)

    def close(self) -> None:
        """Close the SQLite connection."""
        try:
            self._conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None
