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
import html
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from axon.graph_backends.base import (
    FactUpdateResult,
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

# Valid relation after normalisation (``.upper()`` + spaces -> underscores, the
# same normalisation LLM extraction applies): UPPER_SNAKE_CASE, <= 64 chars.
_RELATION_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")

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


def _fact_id(subj_norm: str, rel_upper: str, obj_norm: str, now: str) -> str:
    """Deterministic fact id — shared by extraction and explicit writes."""
    return _sha8(f"{subj_norm}|{rel_upper}|{obj_norm}|{now}")


def _normalize_relation(relation: str) -> str:
    """Normalise a relation the way LLM extraction does (UPPER_SNAKE_CASE)."""
    return (relation or "").strip().upper().replace(" ", "_")


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

    BACKEND_ID = BACKEND_ID

    def __init__(self, brain: Any) -> None:
        self._brain = brain
        _base = Path(getattr(brain.config, "bm25_path", "."))
        self._snapshot_path = _base / SNAPSHOT_FILENAME
        # Re-entrant: upsert_fact() holds the lock across its whole
        # check-then-write while calling _upsert_entity()/_upsert_fact(),
        # which take it themselves.
        self._write_lock = threading.RLock()
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
        self._migrate(conn)
        return conn

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """One-shot, idempotent data migrations keyed on ``PRAGMA user_version``.

        v1 — explicit retraction. Before it, ``delete_documents()`` marked a
        fact that lost all its evidence ``superseded``, indistinguishable from
        one replaced by a newer assertion. Facts retracted that way are the
        ``superseded`` facts with no evidence rows left; mark them
        ``retracted`` so point-in-time queries keep hiding them, as before.
        """
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version < 1:
            conn.execute(
                "UPDATE facts SET status = 'retracted' WHERE status = 'superseded' "
                "AND NOT EXISTS (SELECT 1 FROM fact_evidence ev WHERE ev.fact_id = facts.fact_id)"
            )
            conn.execute("PRAGMA user_version = 1")
            conn.commit()

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

        Audit P1: validate ``snapshot_version`` and skip non-dict rows. A
        future-version or attacker-shaped snapshot must not be silently
        replayed against the (possibly stricter) v1 schema — load nothing
        rather than partially insert junk that retrieve() then trips on.
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
        # Refuse snapshots written by a newer Axon that may carry fields the
        # v1 SQLite schema cannot represent. Missing/non-int version is
        # treated as v1 (legacy snapshots predating the field).
        snap_ver = data.get("snapshot_version", 1)
        if not isinstance(snap_ver, int) or snap_ver > SNAPSHOT_VERSION:
            logger.warning(
                "DynamicGraph snapshot at %s has unsupported snapshot_version=%r "
                "(this build supports up to %d); refusing to load.",
                self._snapshot_path,
                snap_ver,
                SNAPSHOT_VERSION,
            )
            return
        entities = data.get("entities") or []
        facts = data.get("facts") or []
        if not isinstance(entities, list) or not isinstance(facts, list):
            logger.warning(
                "DynamicGraph snapshot at %s has non-list entities/facts; " "refusing to load.",
                self._snapshot_path,
            )
            return
        with self._write_lock:
            try:
                for ent in entities:
                    if not isinstance(ent, dict):
                        continue
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
                    if not isinstance(f, dict):
                        continue
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
        *,
        explicit: bool = False,
        metadata: dict | None = None,
    ) -> str:
        """Insert a fact; supersede conflicting exclusive facts. Return fact_id.

        ``explicit=True`` is the agent/user write path (:meth:`upsert_fact`):
        the caller has already decided which facts to supersede, so the
        scope_key supersede/±1 s conflict rule below is skipped entirely and
        the fact is inserted as ``active``. Extraction (``explicit=False``)
        behaves exactly as before. ``metadata`` replaces the default
        ``{"description": description}`` payload.
        """
        subj_norm = _norm(subject)
        obj_norm = _norm(obj)
        rel_upper = relation.upper()
        # Determine scope_key for conflict resolution
        scope_key: str | None = None
        if rel_upper in _EXCLUSIVE_RELATIONS:
            scope_key = f"{subj_norm}:{rel_upper}"
        fact_id = _fact_id(subj_norm, rel_upper, obj_norm, now)
        with self._write_lock:
            new_fact_status = "active"
            # Supersede or conflict existing active facts with the same scope_key.
            # If the existing active fact has the same timestamp (±1 s), both are
            # conflicted (same-time contradictory assertions); otherwise supersede.
            if scope_key is not None and not explicit:
                # Include both 'active' and prior 'conflicted' facts for the same scope —
                # a new assertion always supersedes or re-conflicts existing ones.
                existing_rows = self._conn.execute(
                    "SELECT fact_id, valid_at FROM facts WHERE scope_key = ? AND status IN ('active', 'conflicted')",
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
            _insert_status = new_fact_status
            # Conflicted facts get invalid_at=now so temporal queries exclude them
            # (invalid_at IS NULL means "still valid"; conflicted is logically invalid).
            _invalid_at = now if _insert_status == "conflicted" else None
            self._conn.execute(
                "INSERT OR IGNORE INTO facts (fact_id, subject, relation, object, valid_at, invalid_at, status, scope_key, confidence, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    fact_id,
                    subj_norm,
                    rel_upper,
                    obj_norm,
                    now,
                    _invalid_at,
                    _insert_status,
                    scope_key,
                    confidence,
                    json.dumps(metadata if metadata is not None else {"description": description}),
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

    def _ensure_agent_evidence(self, fact_id: str, content: str, provenance: str, now: str) -> bool:
        """Attach the ``agent:<fact_id>`` sentinel evidence (+ its episode) to
        an existing fact unless it already has it. Caller holds the write lock
        and commits. Returns True when rows were inserted — repeat calls are
        no-ops, so neither evidence nor episodes are duplicated."""
        chunk_id = f"agent:{fact_id}"
        if self._conn.execute(
            "SELECT 1 FROM fact_evidence WHERE fact_id = ? AND chunk_id = ? LIMIT 1",
            (fact_id, chunk_id),
        ).fetchone():
            return False
        episode_id = _sha8(f"ep|{chunk_id}|{now}")
        self._conn.execute(
            "INSERT OR IGNORE INTO episodes (episode_id, chunk_id, content, "
            "reference_time, metadata) VALUES (?, ?, ?, ?, ?)",
            (episode_id, chunk_id, content, now, json.dumps({"provenance": provenance})),
        )
        self._conn.execute(
            "INSERT INTO fact_evidence (fact_id, chunk_id, episode_id) VALUES (?, ?, ?)",
            (fact_id, chunk_id, episode_id),
        )
        return True

    def upsert_fact(
        self,
        subject: str,
        relation: str,
        obj: str,
        *,
        description: str = "",
        confidence: float = 1.0,
        replace: bool | None = None,
        provenance: str = "agent",
    ) -> FactUpdateResult:
        """Record an explicit (agent/user-asserted) fact.

        * ``relation`` is normalised like extraction does (upper-case,
          spaces -> underscores) and must match ``^[A-Z][A-Z0-9_]{0,63}$``;
          ``subject``/``object`` must be non-empty after stripping, and
          ``confidence`` must be within 0.0-1.0 — otherwise ``ValueError``.
        * ``replace=None`` means "replace" for exclusive relations
          (``_EXCLUSIVE_RELATIONS``, e.g. ``IS_CEO_OF``) and "add" otherwise.
        * Replace mode: afterwards the asserted fact is the only current fact
          for (subject, relation). Every other ``active``/``conflicted`` fact
          with that subject and relation is superseded — matched on
          (subject, relation), not ``scope_key``, because extracted
          non-exclusive facts carry ``scope_key = NULL``. If the identical
          fact is already active it is kept (no new row) and only the others
          are superseded (status ``"superseded"``); if nothing else needed
          superseding the result is ``"unchanged"``.
        * Add mode: the fact is appended; existing facts are untouched. An
          identical active fact makes this a no-op (``"unchanged"``).
        * Explicit writes never trip the ±1 s same-timestamp conflict rule
          that extraction uses, so two quick updates supersede cleanly.
        * The fact gets a sentinel ``fact_evidence`` row (chunk id
          ``agent:<fact_id>``) plus an episode row recording the provenance,
          so ``delete_documents()`` of unrelated chunks — which supersedes
          evidence-less facts — never wipes it. Deleting that sentinel chunk
          id retracts the fact.
        * Raises ``PermissionError`` on a mounted share (grantees read a
          snapshot; only the owner can write).
        """
        if self._is_mounted:
            raise PermissionError(
                "This graph belongs to a mounted share and is read-only here; "
                "only the project owner can update facts."
            )
        subj = (subject or "").strip()
        obj_s = (obj or "").strip()
        if not subj:
            raise ValueError("subject must not be empty")
        if not obj_s:
            raise ValueError("object must not be empty")
        rel = _normalize_relation(relation)
        if not _RELATION_RE.match(rel):
            raise ValueError(
                f"relation {relation!r} is invalid: after normalisation it must match "
                "^[A-Z][A-Z0-9_]{0,63}$ (letters, digits, underscores; starts with a letter)"
            )
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            raise ValueError(f"confidence must be a number, got {confidence!r}") from None
        if not 0.0 <= conf <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence!r}")
        do_replace = (rel in _EXCLUSIVE_RELATIONS) if replace is None else bool(replace)
        desc = (description or "").strip()
        subj_norm = _norm(subj)
        obj_norm = _norm(obj_s)

        with self._write_lock:
            rows = self._conn.execute(
                "SELECT fact_id, object, status FROM facts "
                "WHERE subject = ? AND relation = ? AND status IN ('active', 'conflicted')",
                (subj_norm, rel),
            ).fetchall()
            keep = next(
                (r["fact_id"] for r in rows if r["status"] == "active" and r["object"] == obj_norm),
                None,
            )
            stale = [r["fact_id"] for r in rows if r["fact_id"] != keep] if do_replace else []
            now = _now_iso()
            content = f"{subj} {rel} {obj_s}" + (f": {desc}" if desc else "")
            if keep is not None:
                # An agent confirming an existing (possibly extracted) fact
                # takes ownership of it: give it the sentinel evidence so
                # deleting its source document no longer retracts it.
                added = self._ensure_agent_evidence(keep, content, provenance, now)
                if not stale:
                    if added:
                        self._conn.commit()
                    return FactUpdateResult(
                        status="unchanged",
                        backend_id=BACKEND_ID,
                        fact_id=keep,
                        detail="identical fact is already active",
                    )
            fact_id = keep or _fact_id(subj_norm, rel, obj_norm, now)
            chunk_id = f"agent:{fact_id}"
            episode_id = _sha8(f"ep|{chunk_id}|{now}")
            if keep is None:
                self._upsert_entity(subj, "UNKNOWN", "", now)
                self._upsert_entity(obj_s, "UNKNOWN", "", now)
                self._conn.execute(
                    "INSERT OR IGNORE INTO episodes (episode_id, chunk_id, content, "
                    "reference_time, metadata) VALUES (?, ?, ?, ?, ?)",
                    (episode_id, chunk_id, content, now, json.dumps({"provenance": provenance})),
                )
            # Supersede before inserting so the new row is never touched; the
            # commit inside _upsert_fact() (or the one below) makes the
            # supersede + insert land together.
            for fid in stale:
                self._conn.execute(
                    "UPDATE facts SET status = 'superseded', invalid_at = ? WHERE fact_id = ?",
                    (now, fid),
                )
            if keep is None:
                self._upsert_fact(
                    subj,
                    rel,
                    obj_s,
                    desc,
                    conf,
                    chunk_id,
                    episode_id,
                    now,
                    explicit=True,
                    metadata={"description": desc, "provenance": provenance},
                )
            else:
                self._conn.commit()
        self._cached_nx_graph = None
        self._export_snapshot()
        return FactUpdateResult(
            status="superseded" if stale else "created",
            backend_id=BACKEND_ID,
            fact_id=fact_id,
            superseded_ids=stale,
            detail=(
                f"replaced {len(stale)} earlier fact(s) for ({subj_norm}, {rel})" if stale else ""
            ),
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
        #
        # Superseded (and conflicted) facts ARE part of history: they stay
        # visible inside their [valid_at, invalid_at) window, even after their
        # source document is deleted. Only facts delete_documents() marked
        # 'retracted' (a current fact whose every source document was deleted)
        # are kept out of time-travel queries. The
        # multi-hop expansion below applies the same predicate so it cannot
        # leak facts from another point in time.
        placeholders = ",".join("?" for _ in query_terms)
        pit = getattr(cfg, "point_in_time", None) if cfg else None
        if not isinstance(pit, datetime):
            pit = None
        fact_filter: str
        filter_params: tuple
        if pit is not None:
            pit_str = pit.isoformat() if hasattr(pit, "isoformat") else str(pit)
            fact_filter = (
                "valid_at <= ? AND (invalid_at IS NULL OR invalid_at > ?) "
                "AND status != 'retracted'"
            )
            filter_params = (pit_str, pit_str)
        else:
            fact_filter = "status = 'active'"
            filter_params = ()
        rows = self._execute(
            f"SELECT fact_id, subject, relation, object, valid_at, confidence, metadata "
            f"FROM facts "
            f"WHERE {fact_filter} "
            f"  AND (subject IN ({placeholders}) OR object IN ({placeholders})) "
            f"ORDER BY confidence DESC, valid_at DESC LIMIT ?",
            (*filter_params, *query_terms, *query_terms, top_k),
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
                            f"WHERE {fact_filter} "
                            f"  AND (subject IN ({placeholders}) OR object IN ({placeholders}))",
                            (*filter_params, *chunk, *chunk),
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
        return FinalizationResult(
            backend_id=BACKEND_ID,
            status="not_applicable",
            detail="dynamic_graph has no community-detection step",
        )

    def list_conflicts(self, limit: int = 100) -> list[dict]:
        """Return facts whose ``status='conflicted'`` was set by ``_upsert_fact``.

        Conflicts arise when two exclusive-relation facts with the same
        ``scope_key`` and overlapping ``valid_at`` (±1 s) are ingested. Both
        are kept and surfaced here so the UI / agent can prompt the user to
        resolve the contradiction.
        """
        rows = self._execute(
            "SELECT fact_id, subject, relation, object, valid_at, invalid_at, "
            "       scope_key, confidence, metadata "
            "FROM facts WHERE status = 'conflicted' "
            "ORDER BY valid_at DESC LIMIT ?",
            (int(limit),),
        )
        out: list[dict] = []
        for r in rows:
            try:
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
            except Exception:
                meta = {}
            out.append(
                {
                    "fact_id": r["fact_id"],
                    "subject": r["subject"],
                    "relation": r["relation"],
                    "object": r["object"],
                    "valid_at": r["valid_at"],
                    "invalid_at": r["invalid_at"],
                    "scope_key": r["scope_key"],
                    "confidence": float(r["confidence"]),
                    "metadata": meta,
                }
            )
        return out

    def clear(self, *, persist: bool = False) -> None:
        """Delete all rows from all tables.

        *persist* is unused here — SQL DELETE + commit is inherently
        persistent, there is no separate in-memory-only reset for this
        backend. Accepted only to satisfy the widened GraphBackend Protocol.
        """
        with self._write_lock:
            self._conn.executescript(
                "DELETE FROM fact_evidence; DELETE FROM facts; DELETE FROM entities; DELETE FROM episodes;"
            )
            self._conn.commit()
        self._cached_nx_graph = None
        self._cached_nx_time = 0.0

    def delete_documents(self, chunk_ids: list[str]) -> None:
        """Remove episodes and evidence rows for *chunk_ids*; retract facts
        left without any evidence.

        A *current* fact (``active`` or ``conflicted``) whose last evidence
        row is removed here becomes ``retracted``: every source that asserted
        it is gone, so it is no longer current and is hidden from
        point-in-time history too. ``invalid_at`` is set to now unless it is
        already set (a conflicted fact keeps its boundary).

        A ``superseded`` fact is left as it is even when its last source goes:
        it was already replaced by a newer assertion, and what the graph
        believed during its validity window is history — deleting a stale
        source document in ordinary cleanup must not erase it from
        point-in-time queries. Facts that keep evidence from other chunks (or
        an agent sentinel ``agent:<fact_id>``) are untouched. Retracted facts
        are never re-activated; asserting the same triple again creates a new
        fact.
        """
        if not chunk_ids:
            return
        ids = tuple(dict.fromkeys(chunk_ids))
        now = _now_iso()
        with self._write_lock:
            affected: set[str] = set()
            # Chunked to stay under SQLITE_MAX_VARIABLE_NUMBER.
            for i in range(0, len(ids), 500):
                part = ids[i : i + 500]
                ph = ",".join("?" for _ in part)
                affected.update(
                    r["fact_id"]
                    for r in self._conn.execute(
                        f"SELECT DISTINCT fact_id FROM fact_evidence WHERE chunk_id IN ({ph})",
                        part,
                    )
                )
                self._conn.execute(f"DELETE FROM fact_evidence WHERE chunk_id IN ({ph})", part)
                self._conn.execute(f"DELETE FROM episodes WHERE chunk_id IN ({ph})", part)
            aff = list(affected)
            for i in range(0, len(aff), 500):
                part_f = tuple(aff[i : i + 500])
                ph = ",".join("?" for _ in part_f)
                self._conn.execute(
                    "UPDATE facts SET status = 'retracted', invalid_at = COALESCE(invalid_at, ?) "
                    f"WHERE fact_id IN ({ph}) AND status IN ('active', 'conflicted') "
                    "AND NOT EXISTS (SELECT 1 FROM fact_evidence ev WHERE ev.fact_id = facts.fact_id)",
                    (now, *part_f),
                )
            self._conn.commit()
        if affected:
            self._cached_nx_graph = None
            self._export_snapshot()

    def status(self) -> dict:
        """Return lightweight counts from all tables."""
        rows = self._execute(
            "SELECT "
            "(SELECT COUNT(*) FROM episodes) AS episodes, "
            "(SELECT COUNT(*) FROM entities) AS entities, "
            "(SELECT COUNT(*) FROM facts WHERE status = 'active') AS active_facts, "
            "(SELECT COUNT(*) FROM facts WHERE status = 'superseded') AS superseded_facts, "
            "(SELECT COUNT(*) FROM facts WHERE status = 'conflicted') AS conflicted_facts, "
            "(SELECT COUNT(*) FROM facts WHERE status = 'retracted') AS retracted_facts"
        )
        if not rows:
            return {
                "backend": BACKEND_ID,
                "episodes": 0,
                "entities": 0,
                "active_facts": 0,
                "superseded_facts": 0,
                "conflicted_facts": 0,
                "retracted_facts": 0,
            }
        row = rows[0]
        return {
            "backend": BACKEND_ID,
            "episodes": row["episodes"],
            "entities": row["entities"],
            "active_facts": row["active_facts"],
            "superseded_facts": row["superseded_facts"],
            "conflicted_facts": row["conflicted_facts"],
            "retracted_facts": row["retracted_facts"],
        }

    def has_entities(self) -> bool:
        """Always False: dynamic_graph tracks its own SQLite entities/facts
        tables, not GraphRagMixin's brain._entity_graph — this predicate
        gates GraphRAG-mixin-specific local/global search in query_router.py,
        which dynamic_graph never drives (it has its own retrieve() path).
        """
        return False

    def has_community_summaries(self) -> bool:
        """dynamic_graph has no community-detection step (see finalize())."""
        return False

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
                        "tooltip": f"<b>{html.escape(name)}</b><br/>{html.escape(desc[:220])}",
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
