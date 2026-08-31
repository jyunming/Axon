"""Single-instance detection + client-side routing for the ``axon`` CLI.

When an ``axon-api`` server is already running, store-mutating CLI commands
(``--ingest``, ``--project-new``, ``--project-delete``, ``--project`` switch)
are routed through that server's HTTP API instead of constructing a second
local :class:`~axon.main.AxonBrain` on the same project store. Two independent
processes opening the same TurboQuantDB store race on native file handles and
crash; and every extra CLI process re-loads the embedding model and attaches to
the shared rotating log file. Routing through the one running server avoids
both — there is a single owner of the store and the log.

Discovery is a cheap ``GET /health/ready`` probe. Transport is stdlib
``urllib`` (sync, no new dependency) — this mirrors what :mod:`axon.mcp_server`
already does over ``httpx`` for the MCP surface.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

# How long to wait for the readiness probe. Short: if no server is up, the
# connection is refused almost immediately, so this only bounds a hung server.
_PROBE_TIMEOUT_S = 1.5
# Per-request timeout for routed operations other than the (polled) ingest.
_OP_TIMEOUT_S = 60.0


def resolve_api_base(config: Any) -> str:
    """Return the base URL of the axon-api server a client should talk to.

    Precedence: ``AXON_API_BASE`` / ``RAG_API_BASE`` env vars, then the
    ``api_host``/``api_port`` config fields (default ``127.0.0.1:8420``).
    """
    env = os.getenv("AXON_API_BASE") or os.getenv("RAG_API_BASE")
    if env:
        return env.rstrip("/")
    host = getattr(config, "api_host", None) or "127.0.0.1"
    port = getattr(config, "api_port", None) or 8420
    return f"http://{host}:{port}"


def _headers(config: Any) -> dict[str, str]:
    h = {"Content-Type": "application/json", "X-Axon-Surface": "cli"}
    key = os.getenv("RAG_API_KEY") or getattr(config, "api_key", "") or ""
    if key:
        h["X-API-Key"] = key
    return h


def _request(
    method: str,
    url: str,
    headers: dict[str, str],
    body: dict | None = None,
    timeout: float = _OP_TIMEOUT_S,
) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        # Surface the server's JSON {"detail": ...} as a clean message instead
        # of a raw traceback (e.g. a 403 for a path outside RAG_INGEST_BASE).
        detail = None
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail")
        except Exception:
            pass
        raise ServerRequestError(exc.code, detail or exc.reason) from None
    parsed: dict = json.loads(raw) if raw else {}
    return parsed


class ServerRequestError(RuntimeError):
    """A routed operation failed on the server. Carries the HTTP status code."""

    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"server returned {status}: {detail}")


def _norm_path(p: str | None) -> str | None:
    if not p:
        return None
    try:
        return os.path.normcase(os.path.realpath(os.path.expanduser(p)))
    except Exception:
        return None


def detect_server(config: Any, *, timeout: float = _PROBE_TIMEOUT_S) -> dict | None:
    """Probe ``GET /health/ready``; return its JSON (``{status, project}``) if a
    server is up, ready, AND serving the same store as this CLI. Else ``None``.

    Returns ``None`` on connection refused (no server), timeout, non-200, or a
    **store mismatch** — the caller then falls back to a local brain. The store
    check is essential: routing writes through the server's active store, so
    ingesting into a server that serves a *different* ``projects_root`` (a test
    fixture's temp store, or a second install) would silently target the wrong
    corpus. We only auto-route when the two stores are provably the same.
    """
    try:
        base = resolve_api_base(config)
        req = urllib.request.Request(f"{base}/health/ready", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            payload: dict = json.loads(resp.read().decode("utf-8") or "{}")
    except Exception:
        # Best-effort probe: ANY failure (connection refused, timeout, a
        # malformed api_host/api_port, a mocked config in tests, bad JSON)
        # must fall back to a local brain, never break the CLI.
        return None

    # Verify the server serves the same store before routing anything to it.
    my_root = _norm_path(getattr(config, "projects_root", None))
    if isinstance(my_root, str) and my_root:
        try:
            # /config is NOT on the API's auth-bypass list, so a secured
            # deployment (RAG_API_KEY set) 401s an unauthenticated probe. Send
            # the same headers (incl. X-API-Key) the routed ops use, or
            # single-instance routing silently never engages.
            cfg_req = urllib.request.Request(
                f"{base}/config", method="GET", headers=_headers(config)
            )
            with urllib.request.urlopen(cfg_req, timeout=timeout) as resp:
                server_cfg = json.loads(resp.read().decode("utf-8") or "{}")
        except Exception:
            # Can't verify the store — fail safe to local rather than risk
            # writing to the wrong corpus.
            return None
        if _norm_path(server_cfg.get("projects_root")) != my_root:
            return None

    payload["_api_base"] = base
    return payload


# --- routed operations ---------------------------------------------------


def remote_project_new(
    base: str, name: str, headers: dict[str, str], graph_backend: str | None = None
) -> Any:
    body: dict = {"name": name}
    if graph_backend is not None:
        body["graph_backend"] = graph_backend
    return _request("POST", f"{base}/project/new", headers, body)


def remote_project_switch(base: str, name: str, headers: dict[str, str]) -> Any:
    return _request("POST", f"{base}/project/switch", headers, {"project_name": name})


def remote_project_delete(base: str, name: str, headers: dict[str, str]) -> Any:
    return _request("POST", f"{base}/project/delete/{name}", headers, {})


def remote_project_pack(
    base: str, name: str, headers: dict[str, str], *, out_path: str | None = None
) -> Any:
    body: dict = {"project_name": name}
    if out_path is not None:
        body["out_path"] = out_path
    return _request("POST", f"{base}/project/pack", headers, body)


def remote_project_unpack(
    base: str,
    zip_path: str,
    headers: dict[str, str],
    *,
    as_name: str | None = None,
    force: bool = False,
) -> Any:
    abs_path = os.path.abspath(zip_path)
    body: dict = {"zip_path": abs_path, "force": force}
    if as_name is not None:
        body["as_name"] = as_name
    return _request("POST", f"{base}/project/unpack", headers, body)


def remote_ingest(
    base: str,
    path: str,
    headers: dict[str, str],
    *,
    on_progress: Callable[[dict], None] | None = None,
    poll_interval: float = 2.0,
    max_wait_s: float = 3600.0,
) -> dict:
    """Ingest a file/dir through the running server and block until it finishes.

    POSTs the absolute path to ``/ingest`` (the server reads it locally — same
    machine), then polls ``/ingest/status/{job_id}`` until ``completed`` or
    ``failed``. Returns the final job dict. Raises ``RuntimeError`` on a failed
    job or if the job doesn't finish within ``max_wait_s``.
    """
    abs_path = os.path.abspath(path)
    started = _request("POST", f"{base}/ingest", headers, {"path": abs_path})
    job_id = started.get("job_id")
    if not job_id:
        # Some deployments ingest synchronously; treat a body without a job_id
        # as already-complete.
        return started
    waited = 0.0
    last_phase = None
    while True:
        status = _request("GET", f"{base}/ingest/status/{job_id}", headers, timeout=_OP_TIMEOUT_S)
        state = status.get("status")
        if on_progress and status.get("phase") != last_phase:
            last_phase = status.get("phase")
            on_progress(status)
        if state == "completed":
            return status
        if state == "failed":
            raise RuntimeError(status.get("error") or "ingest failed on server")
        time.sleep(poll_interval)
        waited += poll_interval
        if waited >= max_wait_s:
            raise RuntimeError(
                f"ingest job {job_id} did not finish within {int(max_wait_s)}s "
                f"(last phase: {last_phase})"
            )


# --- server store lock (singleton guard) ---------------------------------
# A per-store lockfile records which axon-api process currently serves a
# store. A second axon-api on the SAME store (e.g. :8420 and :9100 both
# pointing at one AxonStore) means two full brains writing the same
# TurboQuantDB files — the exact concurrent-access hazard the CLI routing
# avoids. On startup axon-api checks this lock and refuses (unless
# AXON_ALLOW_MULTIPLE_SERVERS is set) so there is one server per store.
_LOCK_NAME = ".axon-api.lock"


def _store_lock_path(config: Any) -> pathlib.Path | None:
    root = getattr(config, "projects_root", None)
    if not root:
        return None
    try:
        return pathlib.Path(root) / _LOCK_NAME
    except Exception:
        return None


def find_live_server_for_store(config: Any, *, timeout: float = _PROBE_TIMEOUT_S) -> dict | None:
    """If another axon-api is already serving this store, return its lock info
    (``{host, port, pid}``); else ``None``.

    Reads the per-store lockfile and verifies the recorded server is actually
    responding on ``/health/ready`` — a stale lock (server died without
    cleanup) is treated as absent so a restart is never blocked.
    """
    p = _store_lock_path(config)
    if p is None or not p.exists():
        return None
    try:
        info: dict = json.loads(p.read_text(encoding="utf-8"))
        # A lockfile with no port is malformed; treat it as no lock at all
        # rather than letting int(None) raise into the bare except below.
        raw_port = info.get("port")
        if raw_port is None:
            return None
        port = int(raw_port)
    except Exception:
        return None
    host = info.get("host") or "127.0.0.1"
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    req = urllib.request.Request(f"http://{probe_host}:{port}/health/ready", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return info  # 2xx → alive
    except urllib.error.HTTPError:
        # The server responded with a non-2xx (e.g. 503 while its brain is still
        # initializing, or 401 if secured). It is ALIVE — not a stale lock — so
        # a second server must still refuse. Only a connection-level failure
        # below means the recorded process is actually gone.
        return info
    except Exception:
        return None


def write_store_lock(config: Any, host: str, port: int) -> None:
    """Register this process as the server for the store (best-effort)."""
    p = _store_lock_path(config)
    if p is None:
        return
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"host": host, "port": int(port), "pid": os.getpid()}),
            encoding="utf-8",
        )
    except OSError:
        pass


def release_store_lock(config: Any) -> None:
    """Remove the store lock if it belongs to this process (best-effort)."""
    p = _store_lock_path(config)
    if p is None or not p.exists():
        return
    try:
        if json.loads(p.read_text(encoding="utf-8")).get("pid") == os.getpid():
            p.unlink()
    except Exception:
        pass
