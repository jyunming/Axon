"""
axon/governance.py — Governance audit store for the Operator Console.

Provides:
- AuditEvent dataclass: UUID, ISO-8601 timestamp, actor, surface, project,
  action, target_type, target_id, status, details (dict), request_id
- AuditStore: SQLite (WAL) backend with JSONL fallback; no new dependencies
- CopilotSessionStore: in-memory active/recent Copilot bridge session tracker
- emit(): fire-and-forget audit write (background thread, never blocks callers)

Design constraints
------------------
- SQLite WAL mode for concurrent access safety
- No new runtime dependencies (stdlib sqlite3, uuid, datetime only)
- Audit writes are background-threaded: callers are never blocked
- Fallback to JSONL if SQLite is unavailable (e.g., read-only filesystem)
- Default retention: 90 days (pruned on first open)
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Action vocabulary
# ---------------------------------------------------------------------------

VALID_ACTIONS: frozenset[str] = frozenset(
    {
        "ingest_started",
        "ingest_completed",
        "ingest_failed",
        "delete",
        "graph_finalize",
        "maintenance_changed",
        "share_generated",
        "share_redeemed",
        "share_revoked",
        "copilot_session_opened",
        "copilot_session_closed",
        "copilot_session_failed",
    }
)

_DEFAULT_DB_NAME = ".governance.db"
_FALLBACK_JSONL_NAME = ".governance.jsonl"
_DEFAULT_RETENTION_DAYS = 90


# ---------------------------------------------------------------------------
# AuditEvent
# ---------------------------------------------------------------------------


@dataclass
class AuditEvent:
    """Single governance audit record."""

    action: str
    target_type: str
    target_id: str
    project: str
    actor: str = "api"
    surface: str = "api"
    status: str = "completed"
    details: dict = field(default_factory=dict)
    request_id: str = ""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# AuditStore
# ---------------------------------------------------------------------------


class AuditStore:
    """SQLite-backed audit event store with JSONL fallback.

    Thread-safe: a per-instance lock serialises all writes so concurrent
    ``append()`` calls from different threads never corrupt the database.
    WAL journal mode allows concurrent readers alongside the writer.
    """

    def __init__(self, db_path: str | Path, retention_days: int = _DEFAULT_RETENTION_DAYS) -> None:
        self._db_path = Path(db_path)
        self._retention_days = retention_days
        self._lock = threading.Lock()
        self._use_jsonl = False
        self._jsonl_path = self._db_path.parent / _FALLBACK_JSONL_NAME
        try:
            self._init_db()
            self.prune(self._retention_days)
        except Exception as exc:
            logger.warning(
                "Governance SQLite unavailable (%s); falling back to JSONL at %s",
                exc,
                self._jsonl_path,
            )
            self._use_jsonl = True

    # ------------------------------------------------------------------
    # Internal — SQLite
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id    TEXT PRIMARY KEY,
                    timestamp   TEXT NOT NULL,
                    actor       TEXT NOT NULL,
                    surface     TEXT NOT NULL,
                    project     TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    target_type TEXT NOT NULL,
                    target_id   TEXT NOT NULL,
                    status      TEXT NOT NULL,
                    details     TEXT NOT NULL,
                    request_id  TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ae_project   ON audit_events(project)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ae_action    ON audit_events(action)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ae_timestamp ON audit_events(timestamp)")

    # ------------------------------------------------------------------
    # Internal — JSONL fallback
    # ------------------------------------------------------------------

    def _append_jsonl(self, event: AuditEvent) -> None:
        row = json.dumps(asdict(event), ensure_ascii=False)
        with self._jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(row + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, event: AuditEvent) -> None:
        """Persist an audit event (thread-safe, swallows errors silently)."""
        with self._lock:
            if self._use_jsonl:
                try:
                    self._append_jsonl(event)
                except Exception as exc:
                    logger.debug("Governance JSONL append failed: %s", exc)
                return
            try:
                with self._connect() as conn:
                    conn.execute(
                        "INSERT OR IGNORE INTO audit_events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            event.event_id,
                            event.timestamp,
                            event.actor,
                            event.surface,
                            event.project,
                            event.action,
                            event.target_type,
                            event.target_id,
                            event.status,
                            json.dumps(event.details, ensure_ascii=False),
                            event.request_id,
                        ),
                    )
            except Exception as exc:
                logger.debug("Governance DB append failed: %s; trying JSONL", exc)
                try:
                    self._append_jsonl(event)
                except Exception:
                    pass

    def query(
        self,
        project: str | None = None,
        action: str | None = None,
        surface: str | None = None,
        status: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Return audit events matching the given filters, newest first."""
        if self._use_jsonl:
            return self._query_jsonl(
                project=project,
                action=action,
                surface=surface,
                status=status,
                since=since,
                limit=limit,
            )
        clauses: list[str] = []
        params: list[Any] = []
        if project:
            clauses.append("project = ?")
            params.append(project)
        if action:
            clauses.append("action = ?")
            params.append(action)
        if surface:
            clauses.append("surface = ?")
            params.append(surface)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(max(1, min(limit, 1000)))
        sql = f"SELECT * FROM audit_events {where} ORDER BY timestamp DESC LIMIT ?"
        with self._lock:
            try:
                with self._connect() as conn:
                    rows = conn.execute(sql, params).fetchall()
                return [
                    AuditEvent(
                        event_id=r["event_id"],
                        timestamp=r["timestamp"],
                        actor=r["actor"],
                        surface=r["surface"],
                        project=r["project"],
                        action=r["action"],
                        target_type=r["target_type"],
                        target_id=r["target_id"],
                        status=r["status"],
                        details=json.loads(r["details"]),
                        request_id=r["request_id"],
                    )
                    for r in rows
                ]
            except Exception as exc:
                logger.debug("Governance DB query failed: %s", exc)
                return []

    def _query_jsonl(
        self,
        *,
        project: str | None = None,
        action: str | None = None,
        surface: str | None = None,
        status: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        events: list[AuditEvent] = []
        try:
            if not self._jsonl_path.exists():
                return []
            with self._jsonl_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if project and d.get("project") != project:
                        continue
                    if action and d.get("action") != action:
                        continue
                    if surface and d.get("surface") != surface:
                        continue
                    if status and d.get("status") != status:
                        continue
                    if since and d.get("timestamp", "") < since:
                        continue
                    events.append(AuditEvent(**d))
        except Exception as exc:
            logger.debug("Governance JSONL query failed: %s", exc)
        return list(reversed(events))[:limit]

    def prune(self, days: int = _DEFAULT_RETENTION_DAYS) -> int:
        """Delete events older than *days* days. Returns count deleted."""
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        if self._use_jsonl:
            return 0  # JSONL pruning not supported
        with self._lock:
            try:
                with self._connect() as conn:
                    cur = conn.execute("DELETE FROM audit_events WHERE timestamp < ?", (cutoff,))
                    return cur.rowcount
            except Exception as exc:
                logger.debug("Governance DB prune failed: %s", exc)
                return 0


# ---------------------------------------------------------------------------
# CopilotSession
# ---------------------------------------------------------------------------


@dataclass
class CopilotSession:
    """Represents one Copilot bridge session (active or recently closed)."""

    session_id: str
    request_id: str
    project: str
    opened_at: str
    closed_at: str | None = None
    error: str | None = None

    @property
    def is_active(self) -> bool:
        return self.closed_at is None


class CopilotSessionStore:
    """In-memory tracker for active and recent Copilot bridge sessions.

    Keeps at most *max_recent* sessions (oldest closed ones evicted first).
    """

    def __init__(self, max_recent: int = 50) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, CopilotSession] = {}
        self._max_recent = max_recent

    def _evict(self) -> None:
        """Evict oldest sessions when over capacity (closed first, then active)."""
        excess = len(self._sessions) - self._max_recent
        if excess <= 0:
            return
        # First pass: remove oldest closed sessions
        closed = sorted(
            (s for s in self._sessions.values() if s.closed_at is not None),
            key=lambda s: s.closed_at or "",
        )
        for s in closed[:excess]:
            self._sessions.pop(s.session_id, None)
        # Second pass: if still over cap, remove oldest active sessions too
        excess = len(self._sessions) - self._max_recent
        if excess > 0:
            active = sorted(
                (s for s in self._sessions.values() if s.closed_at is None),
                key=lambda s: s.opened_at,
            )
            for s in active[:excess]:
                self._sessions.pop(s.session_id, None)

    def open(self, session_id: str, request_id: str, project: str) -> None:
        """Register a new Copilot session."""
        with self._lock:
            self._sessions[session_id] = CopilotSession(
                session_id=session_id,
                request_id=request_id,
                project=project,
                opened_at=datetime.now(timezone.utc).isoformat(),
            )
            self._evict()

    def close(self, session_id: str, *, error: str | None = None) -> None:
        """Mark a session as closed (normal or error)."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess and sess.closed_at is None:
                sess.closed_at = datetime.now(timezone.utc).isoformat()
                sess.error = error
            self._evict()

    def expire(self, session_id: str) -> bool:
        """Force-close a stuck session. Returns True if found and active."""
        with self._lock:
            sess = self._sessions.get(session_id)
            if sess and sess.closed_at is None:
                sess.closed_at = datetime.now(timezone.utc).isoformat()
                sess.error = "force-expired by operator"
                return True
            return False

    def list_active(self) -> list[CopilotSession]:
        """Return sessions that have not yet been closed."""
        with self._lock:
            return [s for s in self._sessions.values() if s.closed_at is None]

    def list_recent(self, limit: int = 20) -> list[CopilotSession]:
        """Return most-recently-opened sessions (active + closed)."""
        with self._lock:
            sessions = sorted(self._sessions.values(), key=lambda s: s.opened_at, reverse=True)
            return sessions[:limit]


# ---------------------------------------------------------------------------
# Process-wide singletons
# ---------------------------------------------------------------------------

_store: AuditStore | None = None
_session_store = CopilotSessionStore()
_store_lock = threading.Lock()


def get_store() -> AuditStore:
    """Return the process-wide AuditStore, lazily initialised."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                try:
                    from axon.projects import PROJECTS_ROOT

                    root = Path(PROJECTS_ROOT)
                except Exception:
                    import os

                    root = Path(os.path.expanduser("~/.axon"))
                db_path = root / _DEFAULT_DB_NAME
                _store = AuditStore(db_path)
    return _store


def get_session_store() -> CopilotSessionStore:
    """Return the process-wide CopilotSessionStore singleton."""
    return _session_store


def emit(
    action: str,
    target_type: str,
    target_id: str,
    *,
    project: str,
    actor: str = "api",
    surface: str = "api",
    status: str = "completed",
    details: dict | None = None,
    request_id: str = "",
) -> None:
    """Fire-and-forget audit event write.

    Spawns a daemon thread so the calling code path is never blocked.
    Errors are logged at DEBUG level and never propagate.

    Args:
        action:      One of :data:`VALID_ACTIONS`.
        target_type: Entity type being acted on (e.g. ``"file"``, ``"project"``).
        target_id:   Identifier of the target (path, project name, share key ID…).
        project:     Active project name at the time of the action.
        actor:       Originating subsystem (``"api"``, ``"repl"``, ``"copilot"``…).
        surface:     API surface (``"api"``, ``"repl"``, ``"vscode"``…).
        status:      ``"completed"``, ``"failed"``, or ``"started"``.
        details:     Arbitrary extra context (serialised to JSON).
        request_id:  Forwarded ``X-Request-ID`` header value when available.
    """
    try:
        event = AuditEvent(
            action=action,
            target_type=target_type,
            target_id=target_id,
            project=project,
            actor=actor,
            surface=surface,
            status=status,
            details=details or {},
            request_id=request_id,
        )
        t = threading.Thread(target=get_store().append, args=(event,), daemon=True)
        t.start()
    except Exception as exc:
        logger.debug("emit() failed: %s", exc)
