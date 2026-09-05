"""Single-instance reuse: a ``RemoteBrain`` proxy over a running ``axon-api``.

When an ``axon-api`` server is already serving *the same store*, the CLI REPL
(:mod:`axon.cli`) should NOT construct a
second in-process :class:`~axon.main.AxonBrain` on that store — two processes
racing on the TurboQuantDB files crash, and every extra brain re-loads the
embedding model and attaches to the shared rotating log. Instead they call
:func:`get_brain`, which returns a :class:`RemoteBrain` that exposes the subset
of the ``AxonBrain`` interface that surface uses, backed by the server's
HTTP API (via :mod:`axon.server_client` helpers + stdlib ``urllib``).

Design notes
------------
* ``.config`` is the **local** :class:`~axon.config.AxonConfig` (identical,
  read-only). ``.llm`` / ``.embedding`` / ``.reranker`` are the **local**
  lightweight clients, constructed **lazily** on first access — ``OpenEmbedding``
  and ``OpenReranker`` eagerly load models in their constructors, so eager
  construction here would defeat the whole point (avoid re-loading the embedding
  model). A surface that only queries never touches ``.embedding`` and so never
  loads a model.
* Store-touching operations (``query``, ``switch_project``, ``ingest``,
  ``clear``, ``list_documents`` …) are routed through the one running server.
* Unsupported members are **loud**: a public member we don't proxy raises
  :class:`NotImplementedError`. Private (``_``-prefixed) and dunder members are
  left to raise ``AttributeError`` instead, because the REPL probes many private
  attributes defensively via ``getattr(brain, "_x", default)`` — raising
  ``NotImplementedError`` there would turn graceful fallbacks into crashes. The
  handful of private methods the REPL calls *directly* (e.g.
  ``_build_system_prompt``) are defined explicitly below to raise
  ``NotImplementedError``.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from axon import server_client as _sc
from axon.config import AxonConfig

logger = logging.getLogger("Axon")

# Streaming query may run for a while (retrieval + generation); give it room.
_STREAM_TIMEOUT_S = 300.0

# AxonConfig field name -> QueryRequest (api_schemas) field name. The REPL/web
# toggle RAG behaviour on the *local* config; to make those toggles take effect
# on the remote query we forward them as per-request overrides on every call.
_OVERRIDE_TO_QUERYREQ: dict[str, str] = {
    "top_k": "top_k",
    "similarity_threshold": "threshold",
    "hybrid_search": "hybrid",
    "rerank": "rerank",
    "hyde": "hyde",
    "multi_query": "multi_query",
    "step_back": "step_back",
    "query_decompose": "decompose",
    "compress_context": "compress",
    "discussion_fallback": "discuss",
    "llm_temperature": "temperature",
}


class RemoteBrain:
    """HTTP-backed proxy for the ``AxonBrain`` members the UI + REPL depend on.

    Construct via :func:`get_brain` — it only builds a ``RemoteBrain`` when a
    live same-store server has been detected. ``server_info`` is the dict from
    :func:`axon.server_client.detect_server` (carries ``project`` and
    ``_api_base``).
    """

    def __init__(self, config: AxonConfig, server_info: dict | None = None):
        info = server_info or {}
        self.config = config
        self._server_info = info
        self._api_base = info.get("_api_base") or _sc.resolve_api_base(config)
        # Track the active project; seed from the server's current project.
        self._active_project = info.get("project") or "default"
        # Lazy local lightweight clients (see module docstring).
        self._llm: Any = None
        self._embedding: Any = None
        self._reranker: Any = None
        self._warned_chat_history_dropped = False

    # ------------------------------------------------------------------ #
    # Local lightweight clients — lazily constructed, settable (the UI and
    # REPL hot-swap them, e.g. ``brain.llm = OpenLLM(config)`` on /model).
    # ------------------------------------------------------------------ #
    @property
    def llm(self):
        if self._llm is None:
            from axon.llm import OpenLLM

            self._llm = OpenLLM(self.config)
        return self._llm

    @llm.setter
    def llm(self, value):
        self._llm = value

    @property
    def embedding(self):
        if self._embedding is None:
            from axon.embeddings import OpenEmbedding

            self._embedding = OpenEmbedding(self.config)
        return self._embedding

    @embedding.setter
    def embedding(self, value):
        self._embedding = value

    @property
    def reranker(self):
        if self._reranker is None:
            from axon.rerank import OpenReranker

            self._reranker = OpenReranker(self.config)
        return self._reranker

    @reranker.setter
    def reranker(self, value):
        self._reranker = value

    # ------------------------------------------------------------------ #
    # HTTP plumbing (thin wrappers so tests can monkeypatch one surface).
    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        return _sc._headers(self.config)

    def _request(
        self, method: str, path: str, body: dict | None = None, timeout: float | None = None
    ):
        """Perform a non-streaming JSON request against the server."""
        url = f"{self._api_base}{path}"
        kwargs: dict[str, Any] = {}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return _sc._request(method, url, self._headers(), body, **kwargs)

    def _post_stream(self, path: str, body: dict) -> Iterator[str]:
        """POST *body* and yield decoded SSE ``data:`` payload strings.

        The server frames tokens as ``data: <chunk>\\n\\n`` (chunk may itself
        contain newlines) and control messages as ``data: {json}\\n\\n``. We
        read incrementally, split on the ``\\n\\n`` event boundary, and yield
        the text after the ``data:`` prefix so tokens stream as they arrive.
        """
        url = f"{self._api_base}{path}"
        data = json.dumps(body).encode("utf-8")
        headers = dict(self._headers())
        headers["Accept"] = "text/event-stream"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            resp = urllib.request.urlopen(req, timeout=_STREAM_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            detail = None
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("detail")
            except Exception:
                pass
            raise _sc.ServerRequestError(exc.code, detail or exc.reason) from None
        try:
            buffer = ""
            has_read1 = hasattr(resp, "read1")
            while True:
                block = resp.read1(4096) if has_read1 else resp.read(4096)
                if not block:
                    break
                buffer += block.decode("utf-8", errors="replace")
                while "\n\n" in buffer:
                    event, buffer = buffer.split("\n\n", 1)
                    payload = _parse_sse_event(event)
                    if payload is not None:
                        yield payload
            if buffer.strip():
                payload = _parse_sse_event(buffer)
                if payload is not None:
                    yield payload
        finally:
            try:
                resp.close()
            except Exception:
                pass

    def _build_query_body(self, query: str, filters: dict | None, overrides: dict | None) -> dict:
        cfg = self.config
        body: dict[str, Any] = {
            "query": query,
            "project": self._active_project,
            "top_k": getattr(cfg, "top_k", None),
            "threshold": getattr(cfg, "similarity_threshold", None),
            "hybrid": getattr(cfg, "hybrid_search", None),
            "rerank": getattr(cfg, "rerank", None),
            "hyde": getattr(cfg, "hyde", None),
            "multi_query": getattr(cfg, "multi_query", None),
            "step_back": getattr(cfg, "step_back", None),
            "decompose": getattr(cfg, "query_decompose", None),
            "compress": getattr(cfg, "compress_context", None),
            "discuss": getattr(cfg, "discussion_fallback", None),
            "temperature": getattr(cfg, "llm_temperature", None),
            "include_citations": True,
        }
        if filters:
            body["filters"] = filters
        if overrides:
            for k, v in overrides.items():
                mapped = _OVERRIDE_TO_QUERYREQ.get(k)
                if mapped is not None:
                    body[mapped] = v
        # Drop unset (None) fields so the server keeps its own defaults for them.
        return {k: v for k, v in body.items() if v is not None}

    # ------------------------------------------------------------------ #
    # Project namespace
    # ------------------------------------------------------------------ #
    def switch_project(self, name: str) -> None:
        self._request("POST", "/project/switch", {"project_name": name})
        self._active_project = name

    # ------------------------------------------------------------------ #
    # Query / retrieval
    # ------------------------------------------------------------------ #
    def _warn_chat_history_dropped(self) -> None:
        # NOTE: the server's QueryRequest has no chat_history field — remote
        # queries are single-turn. chat_history is accepted for signature
        # parity but not forwarded (see module docstring / report). Warned
        # once per instance, not per query, so an ongoing multi-turn REPL/
        # webapp session (chat_history is non-empty on every turn after the
        # first) doesn't get a warning on every single message.
        if not self._warned_chat_history_dropped:
            self._warned_chat_history_dropped = True
            logger.warning(
                "RemoteBrain: chat_history is not forwarded — queries against a "
                "reused running server are single-turn (no conversational "
                "context). Pass --local to run in-process with full history support."
            )

    def query(self, query, filters=None, chat_history=None, overrides=None) -> str:
        if chat_history:
            self._warn_chat_history_dropped()
        body = self._build_query_body(query, filters, overrides)
        resp = self._request("POST", "/query", body)
        return str((resp or {}).get("response", ""))

    def query_stream(self, query, filters=None, chat_history=None, overrides=None):
        if chat_history:
            self._warn_chat_history_dropped()
        body = self._build_query_body(query, filters, overrides)
        for payload in self._post_stream("/query/stream", body):
            obj: Any = None
            try:
                obj = json.loads(payload)
            except Exception:
                obj = None
            if isinstance(obj, dict) and "type" in obj:
                kind = obj.get("type")
                if kind == "sources":
                    yield {"type": "sources", "sources": obj.get("sources", [])}
                elif kind == "error":
                    # Surface as a plain-text chunk (mirrors the local "[ERROR]"
                    # convention) so callers that do ``full_response += chunk``
                    # don't blow up on a dict.
                    yield f"[ERROR] {obj.get('content', '')}"
                else:
                    yield payload
            else:
                # Regular token text (may be JSON-parseable like "5"/"true"):
                # yield the raw payload so numbers/words stream verbatim.
                yield payload

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest(self, documents, progress_callback=None) -> None:
        """Ingest already-loaded document dicts through the server's /add_texts.

        ``AxonBrain.ingest`` receives a list of ``{id, text, metadata}`` dicts
        (the loader has already run), so the path-based ``/ingest`` endpoint does
        not fit; ``/add_texts`` is the batch doc-dict ingest and performs the
        same store mutation through the single server.
        """
        if not documents:
            return
        if progress_callback is not None:
            try:
                progress_callback("loading")
            except Exception:
                pass
        docs = [
            {
                "text": d.get("text", ""),
                "doc_id": d.get("id"),
                "metadata": d.get("metadata", {}) or {},
            }
            for d in documents
        ]
        self._request("POST", "/add_texts", {"docs": docs, "project": self._active_project})
        if progress_callback is not None:
            try:
                progress_callback("completed")
            except Exception:
                pass

    async def load_directory(self, directory: str) -> None:
        """Ingest a directory via the path-based /ingest endpoint (+poll)."""
        import asyncio

        await asyncio.to_thread(_sc.remote_ingest, self._api_base, directory, self._headers())

    # ------------------------------------------------------------------ #
    # Mutation: clear
    # ------------------------------------------------------------------ #
    def clear(self) -> dict:
        """Clear the active project's vector store, BM25 index, hash store, and
        entity graph via the server's ``/clear``.

        Unlike ``AxonBrain.clear()`` (which calls ``_assert_write_allowed()``
        then ``collection_ops.clear_active_project()`` itself), write-access
        is enforced server-side by the route — there is no local
        ``vector_store``/``bm25``/graph state in this process to touch.
        """
        return self._request("POST", "/clear", {}) or {}

    # ------------------------------------------------------------------ #
    # Read-only introspection
    # ------------------------------------------------------------------ #
    def list_documents(self) -> list[dict[str, Any]]:
        resp = self._request("GET", "/collection")
        files: list[dict[str, Any]] = (resp or {}).get("files", [])
        return files

    def get_doc_versions(self) -> dict:
        resp = self._request("GET", "/tracked-docs")
        docs: dict = (resp or {}).get("docs", {})
        return docs

    # ------------------------------------------------------------------ #
    # Graph / mount operations that have a clean server endpoint
    # ------------------------------------------------------------------ #
    def finalize_graph(self, force: bool = False) -> None:
        self._request("POST", "/graph/finalize", {})

    def refresh_mount(self) -> bool:
        resp = self._request("POST", "/mount/refresh", {})
        return bool((resp or {}).get("refreshed", False))

    # ------------------------------------------------------------------ #
    # Explicit "loud" gaps — members the REPL calls directly that cannot be
    # sensibly proxied. Defined explicitly (rather than via __getattr__) so
    # they raise NotImplementedError even though they are ``_``-prefixed.
    # ------------------------------------------------------------------ #
    @property
    def vector_store(self):
        raise self._unsupported("vector_store")

    def should_recommend_project(self, *a, **k):
        raise self._unsupported("should_recommend_project")

    def export_graph_html(self, *a, **k):
        raise self._unsupported("export_graph_html")

    def wipe_sealed_cache(self, *a, **k):
        raise self._unsupported("wipe_sealed_cache")

    def _build_system_prompt(self, *a, **k):
        raise self._unsupported("_build_system_prompt")

    def _resolve_model_path(self, *a, **k):
        raise self._unsupported("_resolve_model_path")

    def _assert_write_allowed(self, *a, **k):
        # The REPL's local /clear path calls this directly before touching
        # the (non-existent, here) local vector_store; the REPL's remote
        # branch calls .clear() instead, which the server enforces write
        # access for itself. Raise loudly for any other direct caller so the
        # surface reports the gap rather than a bare AttributeError.
        raise self._unsupported("_assert_write_allowed")

    def _is_mounted_share(self, *a, **k):
        raise self._unsupported("_is_mounted_share")

    # ------------------------------------------------------------------ #
    def _unsupported(self, name: str) -> NotImplementedError:
        return NotImplementedError(
            f"RemoteBrain does not support .{name} — run with a local brain "
            f"(pass --local to the CLI, or set allow_remote=False)."
        )

    def __getattr__(self, name: str):
        # __getattr__ only fires for names not found normally.
        # * dunder  -> AttributeError so Python protocols behave.
        # * private -> AttributeError so the REPL's pervasive
        #   ``getattr(brain, "_x", default)`` introspection gets its default
        #   instead of exploding. (Private methods that must be loud are
        #   defined explicitly above.)
        # * public  -> NotImplementedError so a real, un-proxied gap is loud.
        if name.startswith("_"):
            raise AttributeError(name)
        raise NotImplementedError(
            f"RemoteBrain does not support .{name} — run with a local brain "
            f"(pass --local to the CLI, or set allow_remote=False)."
        )


def _parse_sse_event(event: str) -> str | None:
    """Extract the payload from one SSE event block, or None for non-data lines."""
    event = event.lstrip("\n")
    if not event:
        return None
    if event.startswith("data: "):
        return event[6:]
    if event.startswith("data:"):
        return event[5:]
    # Comment (":" prefix), "event:" lines, etc. — ignore.
    return None


def get_brain(config: AxonConfig, *, allow_remote: bool = True):
    """Return a brain for *config*.

    When ``allow_remote`` and an ``axon-api`` server is already serving the same
    store, return a :class:`RemoteBrain` routed to it (no second in-process
    brain). Otherwise construct a local :class:`~axon.main.AxonBrain`.
    """
    if allow_remote:
        try:
            server_info = _sc.detect_server(config)
        except Exception:
            # detect_server is already best-effort, but never let discovery
            # break brain construction.
            server_info = None
        if server_info is not None:
            logger.info(
                "Reusing running axon-api at %s (project=%s) — not building a second brain.",
                server_info.get("_api_base"),
                server_info.get("project"),
            )
            return RemoteBrain(config, server_info)
    from axon.main import AxonBrain

    return AxonBrain(config)
