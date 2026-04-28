"""DynamicGraphBackend — SQLite temporal graph (v0.3).

Implements the ``GraphBackend`` Protocol using SQLite.

Schema mirrors Graphiti's bi-temporal model:
  episodes     → source_chunk_id, content, reference_time
  entities     → canonical_name, entity_type, description, first/last_seen_at
  facts        → subject, relation, object, valid_at, invalid_at, status, scope_key
  fact_evidence → fact_id, chunk_id, episode_id

Share-mount safety
------------------
- Journal mode is ``DELETE`` (not WAL). WAL relies on a shared-memory
  ``-shm`` segment that cannot be replicated coherently across machines
  over cloud-sync or SMB; see https://sqlite.org/wal.html.
- After every ingest, the owner exports a compact JSON snapshot to
  ``{bm25_path}/.dynamic_graph.snapshot.json``. Grantees (``mounts/<name>``)
  never open the owner's SQLite file; they load the snapshot into an
  in-memory SQLite so the existing retrieve() queries work unchanged.

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
import os
import shutil
import sqlite3
import threading
import time
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
CREATE INDEX IF NOT EXISTS idx_facts_temporal  ON facts(valid_at, invalid_at);
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


_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".r",
        ".sh",
        ".bash",
        ".ps1",
        ".lua",
        ".m",
        ".ml",
        ".ex",
        ".exs",
    }
)


def _is_code_chunk(chunk: dict) -> bool:
    """Return True when the chunk originates from a source-code file."""
    meta = chunk.get("metadata") or {}
    if meta.get("language"):
        return True
    if meta.get("source_type") == "code":
        return True
    src = str(meta.get("source", "") or meta.get("file_path", ""))
    if src:
        suffix = Path(src).suffix.lower()
        if suffix in _CODE_EXTENSIONS:
            return True
    return False


def _extract_python_entities_and_facts(text: str) -> tuple[list[dict], list[dict]]:
    """Parse Python source with ast and return (entities, facts) dicts."""
    import ast as _ast

    entities: list[dict] = []
    facts: list[dict] = []
    try:
        tree = _ast.parse(text)
    except SyntaxError:
        return [], []
    module_name = ""
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef):
            entities.append(
                {"name": node.name, "type": "CONCEPT", "description": f"class {node.name}"}
            )
            for base in node.bases:
                base_name = getattr(base, "id", None) or getattr(base, "attr", None)
                if base_name:
                    entities.append({"name": base_name, "type": "CONCEPT", "description": ""})
                    facts.append(
                        {
                            "subject": node.name,
                            "relation": "INHERITS",
                            "object": base_name,
                            "description": "",
                            "confidence": 1.0,
                        }
                    )
        elif isinstance(node, _ast.FunctionDef | _ast.AsyncFunctionDef):
            if node.col_offset == 0:
                entities.append(
                    {"name": node.name, "type": "CONCEPT", "description": f"function {node.name}"}
                )
        elif isinstance(node, _ast.Import | _ast.ImportFrom):
            if isinstance(node, _ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top and not module_name:
                    module_name = top
                entities.append({"name": top, "type": "PRODUCT", "description": f"module {top}"})
                facts.append(
                    {
                        "subject": "__module__",
                        "relation": "IMPORTS",
                        "object": top,
                        "description": "",
                        "confidence": 0.9,
                    }
                )
            elif isinstance(node, _ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    entities.append(
                        {"name": top, "type": "PRODUCT", "description": f"module {top}"}
                    )
                    facts.append(
                        {
                            "subject": "__module__",
                            "relation": "IMPORTS",
                            "object": top,
                            "description": "",
                            "confidence": 0.9,
                        }
                    )
    return entities[:20], facts[:20]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


SNAPSHOT_FILENAME = ".dynamic_graph.snapshot.json"
SNAPSHOT_VERSION = 1


class DynamicGraphBackend:
    """SQLite temporal graph backend satisfying ``GraphBackend`` Protocol.
    The owner writes to an on-disk SQLite database under ``bm25_path`` and
    exports a read-only JSON snapshot after each ingest.  Grantees of a
    shared project (``mounts/<mount_name>`` active project) never touch the
    owner's SQLite file — they load the snapshot into an in-memory SQLite
    so queries work unchanged.
    Args:
        brain: An ``AxonBrain`` instance.  Uses ``brain.config.bm25_path``
               for the database path, ``brain.llm`` for extraction LLM calls,
               ``brain._active_project`` to detect grantee mount state,
               and ``brain.config`` for retrieval configuration.
    """

    def __init__(self, brain: Any) -> None:
        self._brain = brain
        _base = Path(getattr(brain.config, "bm25_path", "."))
        self._snapshot_path = _base / SNAPSHOT_FILENAME
        self._write_lock = threading.Lock()
        active_project = getattr(brain, "_active_project", "") or ""
        self._is_mounted: bool = active_project.startswith("mounts/")
        # Owner-side DB lives under bm25_path by default, but is redirected to
        # a guaranteed-local path under ~/.axon/graphs/<id>/ when bm25_path is
        # itself on a cloud-sync / network filesystem. Even with DELETE journal
        # mode the owner's mid-write file state can be torn by a sync client
        # racing the writer; keeping the DB local-only side-steps the issue.
        # Snapshots are still emitted to the (synced) bm25_path for grantees.
        self._db_path, self._db_relocated = self._resolve_db_path(_base)
        if self._is_mounted:
            # Grantee: load the owner's JSON snapshot into an in-memory DB.
            # Never open the owner's .dynamic_graph.db directly — WAL/DELETE
            # sidecars and concurrent writes on a shared path would corrupt.
            self._conn = self._init_memory_db()
            self._load_snapshot()
        else:
            # Owner: real on-disk DB in DELETE journal mode (share-safe).
            self._maybe_migrate_legacy_db(_base)
            self._conn = self._init_db()
        self._cached_nx_graph: Any = None
        self._cached_nx_time: float = 0.0

    # ------------------------------------------------------------------
    # DB-path resolution (owner side; relocates off cloud-synced paths)
    # ------------------------------------------------------------------
    @staticmethod
    def _local_graphs_root() -> Path:
        """Return the always-local root for owner DBs (``~/.axon/graphs``)."""
        return Path.home() / ".axon" / "graphs"

    def _resolve_db_path(self, base: Path) -> tuple[Path, bool]:
        """Return ``(db_path, relocated)`` for the owner-side SQLite file.
        When *base* (== ``brain.config.bm25_path``) is on a cloud-sync /
        network / WSL-mount path, the DB is redirected to
        ``~/.axon/graphs/<project_id>/.dynamic_graph.db`` so the writer's
        mid-update file state is never observed by a sync client. ``base``
        is used as-is for the snapshot, which IS meant to be visible to
        grantees on a synced path.
        ``project_id`` is read from ``base.parent/meta.json`` if available,
        otherwise derived from a hash of ``base`` so it stays stable across
        process restarts on the same project.
        """
        from axon.paths import is_cloud_sync_or_mount_path

        default = base / ".dynamic_graph.db"
        if not is_cloud_sync_or_mount_path(base):
            return default, False
        # Read project_id from the project's meta.json (one level up from bm25_path).
        project_id = ""
        meta_path = base.parent / "meta.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            project_id = (meta.get("project_id") or "").strip()
        except Exception:
            pass
        if not project_id:
            project_id = _sha8(str(base.resolve() if base.exists() else base))
        local_dir = self._local_graphs_root() / project_id
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "Could not create local graphs dir %s (%s); falling back to %s",
                local_dir,
                exc,
                default,
            )
            return default, False
        return local_dir / ".dynamic_graph.db", True

    def _maybe_migrate_legacy_db(self, base: Path) -> None:
        """One-shot migration: copy a pre-existing DB at ``base`` to the new
        local path so existing projects keep their entities/facts."""
        if not self._db_relocated:
            return
        legacy = base / ".dynamic_graph.db"
        if not legacy.exists() or self._db_path.exists():
            return
        try:
            shutil.copy2(legacy, self._db_path)
            logger.info(
                "Migrated dynamic-graph DB off cloud-sync path: %s -> %s",
                legacy,
                self._db_path,
            )
        except Exception as exc:
            logger.warning(
                "Could not migrate legacy dynamic-graph DB %s -> %s: %s",
                legacy,
                self._db_path,
                exc,
            )

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_db(self) -> sqlite3.Connection:
        """Open (or create) the on-disk SQLite database and apply the schema."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # DELETE journal mode (was WAL): WAL relies on a shared-memory
        # wal-index file that cloud-sync/network filesystems cannot
        # replicate coherently. The per-instance write lock already
        # serialises writes, so WAL's concurrent-reader benefit is moot.
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        return conn

    def _init_memory_db(self) -> sqlite3.Connection:
        """Create an in-memory SQLite used by grantees to replay a snapshot."""
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
        return conn

    # ------------------------------------------------------------------
    # Snapshot export / import (share-mount safety)
    # ------------------------------------------------------------------
    def _export_snapshot(self) -> None:
        """Write a read-only JSON snapshot for grantees of a shared project.
        Called at the end of :meth:`ingest` on the owner side only.  The
        snapshot is the only graph artefact that lives on a potentially-
        synced path — the full SQLite database stays local (write-path
        stays single-writer) so WAL-on-sync corruption is impossible.
        Write is atomic: temp file + ``os.replace``.
        """
        if self._is_mounted:
            return
        try:
            entities = [
                dict(r)
                for r in self._execute(
                    "SELECT entity_id, canonical_name, entity_type, description, "
                    "first_seen_at, last_seen_at FROM entities"
                )
            ]
            facts = [
                dict(r)
                for r in self._execute(
                    "SELECT fact_id, subject, relation, object, valid_at, invalid_at, "
                    "status, scope_key, confidence, metadata FROM facts WHERE status = 'active'"
                )
            ]
            payload = {
                "snapshot_version": SNAPSHOT_VERSION,
                "generated_at": _now_iso(),
                "entities": entities,
                "facts": facts,
            }
            tmp = self._snapshot_path.with_suffix(".json.tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            # On Windows + cloud-sync paths, os.replace can fail with
            # PermissionError when the live target is briefly held by a
            # sync client / file indexer. Fall back to copy+unlink and
            # always clean up the .tmp on the failure path so we don't
            # leave junk for grantees to see.
            try:
                os.replace(tmp, self._snapshot_path)
            except OSError as primary_exc:
                try:
                    shutil.copy2(tmp, self._snapshot_path)
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                except OSError:
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
                    raise primary_exc
        except Exception as exc:
            logger.debug("DynamicGraph snapshot export failed: %s", exc)

    def _load_snapshot(self) -> None:
        """Populate the in-memory DB (grantee side) from the owner's snapshot.
        Missing or unreadable snapshot is not an error: the grantee simply
        sees an empty graph and retrieve() returns nothing, which is the
        sensible default when the owner has never ingested.
        """
        if not self._snapshot_path.exists():
            return
        try:
            data = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "DynamicGraph snapshot at %s could not be read: %s",
                self._snapshot_path,
                exc,
            )
            return
        if not isinstance(data, dict):
            return
        entities = data.get("entities") or []
        facts = data.get("facts") or []
        with self._write_lock:
            try:
                for ent in entities:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO entities "
                        "(entity_id, canonical_name, entity_type, description, "
                        "first_seen_at, last_seen_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            ent.get("entity_id", ""),
                            ent.get("canonical_name", ""),
                            ent.get("entity_type", "UNKNOWN"),
                            ent.get("description", ""),
                            ent.get("first_seen_at", _now_iso()),
                            ent.get("last_seen_at", _now_iso()),
                        ),
                    )
                for f in facts:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO facts "
                        "(fact_id, subject, relation, object, valid_at, invalid_at, "
                        "status, scope_key, confidence, metadata) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            f.get("fact_id", ""),
                            f.get("subject", ""),
                            f.get("relation", ""),
                            f.get("object", ""),
                            f.get("valid_at", _now_iso()),
                            f.get("invalid_at"),
                            f.get("status", "active"),
                            f.get("scope_key"),
                            float(f.get("confidence", 1.0)),
                            f.get("metadata", "{}"),
                        ),
                    )
                self._conn.commit()
            except Exception as exc:
                logger.warning("DynamicGraph snapshot replay failed: %s", exc)

    def _execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        # SELECT queries are thread-safe in WAL mode without explicit locking
        # when using check_same_thread=False.
        cur = self._conn.execute(sql, params)
        return cur.fetchall()

    def _executemany(self, sql: str, params_seq: list[tuple]) -> None:
        with self._write_lock:
            self._conn.executemany(sql, params_seq)
            self._conn.commit()

    def _write(self, sql: str, params: tuple = ()) -> None:
        with self._write_lock:
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
        with self._write_lock:
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
        with self._write_lock:
            # Supersede or conflict existing active facts with the same scope_key.
            # If the existing active fact has the same timestamp (±1 s), both are
            # conflicted (same-time contradictory assertions); otherwise supersede.
            if scope_key is not None:
                existing_rows = self._conn.execute(
                    "SELECT fact_id, valid_at FROM facts WHERE scope_key = ? AND status = 'active'",
                    (scope_key,),
                ).fetchall()
                for erow in existing_rows:
                    try:
                        existing_dt = datetime.fromisoformat(erow["valid_at"])
                        new_dt = datetime.fromisoformat(now)
                        delta = abs((new_dt - existing_dt).total_seconds())
                    except Exception:
                        delta = 999.0
                    new_status = "conflicted" if delta <= 1.0 else "superseded"
                    self._conn.execute(
                        "UPDATE facts SET status = ?, invalid_at = ? WHERE fact_id = ?",
                        (new_status, now, erow["fact_id"]),
                    )
                # If any existing facts were conflicted, mark the new one conflicted too.
                new_fact_status = (
                    "conflicted"
                    if any(
                        abs(
                            (
                                datetime.fromisoformat(now)
                                - datetime.fromisoformat(erow["valid_at"])
                            ).total_seconds()
                        )
                        <= 1.0
                        for erow in existing_rows
                        if erow["valid_at"]
                    )
                    else "active"
                )
            _insert_status = new_fact_status if scope_key is not None else "active"
            self._conn.execute(
                "INSERT OR IGNORE INTO facts (fact_id, subject, relation, object, valid_at, status, scope_key, confidence, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fact_id,
                    subj_norm,
                    rel_upper,
                    obj_norm,
                    now,
                    _insert_status,
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
            with self._write_lock:
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
            # Entity extraction — AST path for code, LLM path for prose
            if _is_code_chunk(chunk):
                entities, facts_raw = _extract_python_entities_and_facts(text)
                if not entities:
                    entities = self._extract_entities(text)
                    facts_raw = self._extract_facts(text)
            else:
                entities = self._extract_entities(text)
                facts_raw = self._extract_facts(text)
            for ent in entities:
                self._upsert_entity(ent["name"], ent["type"], ent["description"], now)
                entities_added += 1
            # Fact extraction
            facts = facts_raw
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
        result = IngestResult(
            entities_added=entities_added,
            relations_added=relations_added,
            chunks_processed=len(chunks),
            backend_id=BACKEND_ID,
        )
        # Export a grantee-readable snapshot so mounted readers stay in sync
        # without opening this backend's SQLite file directly.
        self._export_snapshot()
        return result

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
        3. Perform multi-hop BFS traversal to find linked facts (Epic 1/4).
        4. Return as GraphContext objects (each fact = one context).
        """
        top_k = (cfg.top_k if cfg else None) or 10
        # Step 1: extract query entities (lightweight — just tokenize query terms)
        query_terms = [_norm(w) for w in query.split() if len(w) >= 3]
        if not query_terms:
            return []
        # Step 2: find facts matching query terms.
        # When point_in_time is set, return facts valid at that instant
        # (valid_at <= pit AND (invalid_at IS NULL OR invalid_at > pit));
        # otherwise return currently active facts only.
        placeholders = ",".join("?" for _ in query_terms)
        pit = getattr(cfg, "point_in_time", None) if cfg else None
        if not isinstance(pit, datetime):
            pit = None
        if pit is not None:
            pit_str = pit.isoformat() if hasattr(pit, "isoformat") else str(pit)
            rows = self._execute(
                f"SELECT fact_id, subject, relation, object, valid_at, confidence, metadata "
                f"FROM facts "
                f"WHERE valid_at <= ? AND (invalid_at IS NULL OR invalid_at > ?) "
                f"  AND status != 'superseded' "
                f"  AND (subject IN ({placeholders}) OR object IN ({placeholders})) "
                f"ORDER BY confidence DESC, valid_at DESC LIMIT ?",
                (pit_str, pit_str, *query_terms, *query_terms, top_k),
            )
        else:
            rows = self._execute(
                f"SELECT fact_id, subject, relation, object, valid_at, confidence, metadata "
                f"FROM facts "
                f"WHERE status = 'active' AND (subject IN ({placeholders}) OR object IN ({placeholders})) "
                f"ORDER BY confidence DESC, valid_at DESC LIMIT ?",
                (*query_terms, *query_terms, top_k),
            )
        # Step 3: Perform multi-hop BFS if requested (Epic 1/4)
        max_hops = 1
        if cfg and hasattr(cfg, "graph_rag_max_hops"):
            max_hops = int(cfg.graph_rag_max_hops)
        else:
            max_hops = int(os.getenv("AXON_GRAPH_RAG_MAX_HOPS", "1"))
        hop_decay = 0.7
        if cfg and hasattr(cfg, "graph_rag_hop_decay"):
            hop_decay = float(cfg.graph_rag_hop_decay)
        else:
            hop_decay = float(os.getenv("AXON_GRAPH_RAG_HOP_DECAY", "0.7"))
        # Map to track best score/path for each fact
        fact_map: dict[str, dict] = {}
        for rank, row in enumerate(rows):
            fact_id = row["fact_id"]
            if fact_id not in fact_map:
                fact_map[fact_id] = {
                    "row": row,
                    "score": float(row["confidence"]),
                    "rank": rank,
                    "hop": 0,
                    "path": [],
                    "matched": {row["subject"], row["object"]},
                }
        if max_hops > 0:
            current_fringe = set(query_terms)
            visited_nodes = set(query_terms)
            # node -> path from seed to that node
            node_paths: dict[str, list[tuple[str, str, str]]] = {n: [] for n in query_terms}
            # node -> hop count
            node_hops: dict[str, int] = dict.fromkeys(query_terms, 0)
            for _hop in range(1, max_hops + 1):
                if not current_fringe:
                    break
                # Score for facts DISCOVERED at this hop level.
                # A fact is discovered at hop H if it contains a node reached at hop H-1.
                # Wait, if node X is at hop 0 (seed), facts containing X are hop 0.
                # If node Y is reached from X, Y is hop 1. Facts containing Y are hop 1.
                # So fact_hop = node_hop.
                linked_rows = []
                fringe_list = list(current_fringe)
                # SQLITE_MAX_VARIABLE_NUMBER safe limit (Epic 1/4 Phase 2.2)
                CHUNK_SIZE = 500
                for i in range(0, len(fringe_list), CHUNK_SIZE):
                    chunk = fringe_list[i : i + CHUNK_SIZE]
                    placeholders = ",".join("?" for _ in chunk)
                    linked_rows.extend(
                        self._execute(
                            f"SELECT fact_id, subject, relation, object, valid_at, confidence, metadata "
                            f"FROM facts "
                            f"WHERE status = 'active' AND (subject IN ({placeholders}) OR object IN ({placeholders}))",
                            (*chunk, *chunk),
                        )
                    )
                next_fringe = set()
                for row in linked_rows:
                    fact_id = row["fact_id"]
                    s_orig, r, o_orig = row["subject"], row["relation"], row["object"]
                    subj, obj = s_orig.lower(), o_orig.lower()
                    # The fact is reached via source_node which is in current_fringe
                    source_node = subj if subj in current_fringe else obj
                    fact_hop = node_hops[source_node]
                    # Update fact score/path
                    score = float(row["confidence"]) * (hop_decay**fact_hop)
                    if fact_id not in fact_map or score > fact_map[fact_id]["score"]:
                        fact_map[fact_id] = {
                            "row": row,
                            "score": score,
                            "rank": 999,
                            "hop": fact_hop,
                            "path": node_paths[source_node],
                            "matched": {s_orig, o_orig},
                        }
                    # Find new nodes reached via this fact
                    target_node = obj if subj == source_node else subj
                    if target_node not in visited_nodes:
                        visited_nodes.add(target_node)
                        next_fringe.add(target_node)
                        node_hops[target_node] = fact_hop + 1
                        node_paths[target_node] = node_paths[source_node] + [(s_orig, r, o_orig)]
                current_fringe = next_fringe
        _existing_ids = {r.get("id") for r in (existing_results or []) if r.get("id")}
        sorted_facts = sorted(
            fact_map.values(), key=lambda x: (x["score"], x["row"]["valid_at"]), reverse=True
        )[:top_k]
        # Step 4: convert to GraphContext
        contexts: list[GraphContext] = []
        for rank, item in enumerate(sorted_facts):
            row = item["row"]
            if row["fact_id"] in _existing_ids:
                continue
            text = f"{row['subject']} {row['relation'].replace('_', ' ').lower()} {row['object']}"
            meta = json.loads(row["metadata"] or "{}")
            desc = meta.get("description", "")
            if desc:
                text = f"{text}: {desc}"
            ctx = GraphContext(
                context_id=row["fact_id"],
                context_type="fact",
                text=text,
                score=item["score"],
                rank=rank,
                backend_id=BACKEND_ID,
                source_chunk_id="",
                metadata={"valid_at": row["valid_at"], **meta},
                valid_at=_parse_dt(row["valid_at"]),
                matched_entity_names=list(item["matched"]),
                hop_count=item["hop"],
                path=item["path"],
            )
            contexts.append(ctx)
        return contexts

    def _build_nx_graph_from_db(self):
        """Build a NetworkX graph from all active facts in the database (used in tests)."""
        now = time.time()
        ttl = float(os.getenv("AXON_GRAPH_CACHE_TTL", "300"))
        if self._cached_nx_graph and (now - self._cached_nx_time) < ttl:
            return self._cached_nx_graph
        import networkx as nx

        rows = self._execute(
            "SELECT subject, object, confidence FROM facts WHERE status = 'active'"
        )
        G = nx.Graph()
        for row in rows:
            u, v = row["subject"], row["object"]
            try:
                conf = float(row["confidence"])
            except (ValueError, TypeError):
                logger.warning(f"Malformed confidence value in fact DB: {row['confidence']!r}")
                conf = 1.0
            if G.has_edge(u, v):
                G[u][v]["weight"] += conf
            else:
                G.add_edge(u, v, weight=conf)
        # Post-process for distance
        for _u, _v, d in G.edges(data=True):
            w = d.get("weight", 1.0)
            d["distance"] = 1.0 / (w + 1e-6)
        self._cached_nx_graph = G
        self._cached_nx_time = now
        return G

    def finalize(self, force: bool = False) -> FinalizationResult:
        """No-op — dynamic graph is episodic; no community detection step."""
        return FinalizationResult(backend_id=BACKEND_ID)

    def clear(self) -> None:
        """Delete all rows from all tables."""
        with self._write_lock:
            self._conn.executescript(
                "DELETE FROM fact_evidence; DELETE FROM facts; DELETE FROM entities; DELETE FROM episodes;"
            )
            self._conn.commit()

    def delete_documents(self, chunk_ids: list[str]) -> None:
        """Remove episodes, evidence rows, and orphaned facts for *chunk_ids*."""
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._write_lock:
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
            "(SELECT COUNT(*) FROM facts WHERE status = 'superseded') AS superseded_facts, "
            "(SELECT COUNT(*) FROM facts WHERE status = 'conflicted') AS conflicted_facts"
        )
        if not rows:
            return {
                "backend": BACKEND_ID,
                "episodes": 0,
                "entities": 0,
                "active_facts": 0,
                "superseded_facts": 0,
                "conflicted_facts": 0,
            }
        row = rows[0]
        return {
            "backend": BACKEND_ID,
            "episodes": row["episodes"],
            "entities": row["entities"],
            "active_facts": row["active_facts"],
            "superseded_facts": row["superseded_facts"],
            "conflicted_facts": row["conflicted_facts"],
        }

    def graph_data(self, filters: GraphDataFilters | None = None) -> GraphPayload:
        """Return current active facts as a renderer-enriched nodes + links payload."""
        limit = (filters.limit if filters else None) or 500
        rows = self._execute(
            "SELECT fact_id, subject, relation, object, confidence, valid_at, invalid_at "
            "FROM facts WHERE status = 'active' ORDER BY confidence DESC LIMIT ?",
            (limit,),
        )
        # One query to build entity type + description lookup (O(n_entities), not O(n_facts))
        entity_rows = self._execute("SELECT canonical_name, entity_type, description FROM entities")
        entity_meta: dict[str, tuple[str, str]] = {
            r["canonical_name"]: (r["entity_type"], r["description"]) for r in entity_rows
        }
        # Lazy import of color map from graph_render (avoids hard dependency)
        try:
            from axon.graph_render import _VIZ_TYPE_COLORS as _colors
        except Exception:
            _colors: dict[str, str] = {}  # type: ignore[assignment]
        # Collect unique node names with visualization metadata
        node_names: dict[str, dict] = {}
        links: list[dict] = []
        for row in rows:
            subj = row["subject"]
            obj = row["object"]
            conf = float(row["confidence"])
            valid_at = row["valid_at"]
            invalid_at = row["invalid_at"]
            for name in (subj, obj):
                if name not in node_names:
                    etype, desc = entity_meta.get(name, ("UNKNOWN", ""))
                    node_names[name] = {
                        "id": name,
                        "name": name,
                        "label": name[:24],
                        "type": etype,
                        "color": _colors.get(etype, "#94a3b8"),
                        "val": 4,
                        "tooltip": f"<b>{name}</b><br/>{desc[:220]}",
                    }
            relation = row["relation"]
            label = relation.replace("_", " ").lower()
            if valid_at:
                label = f"{label} ({valid_at[:10]})"
            links.append(
                {
                    "source": subj,
                    "target": obj,
                    "label": label,
                    "relation": relation,
                    "value": conf,
                    "width": 1.0 + conf,
                    "weight": conf,
                    "valid_at": valid_at,
                    "invalid_at": invalid_at,
                }
            )
        # Apply entity_type filter if requested
        if filters and filters.entity_types:
            allowed = set(filters.entity_types)
            for name, node in node_names.items():
                if node["type"] == "entity":
                    etype, _ = entity_meta.get(name, ("UNKNOWN", ""))
                    node["type"] = etype
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
