"""Unit tests for single-instance reuse via RemoteBrain / get_brain.

All network + server interaction is mocked — these tests never touch a real
axon-api server or the network. They verify:
  * get_brain returns a RemoteBrain when a same-store server is detected,
    and a local AxonBrain otherwise (and honours allow_remote=False);
  * RemoteBrain routes switch_project / query / query_stream / ingest /
    list_documents / get_doc_versions through the right endpoints;
  * query_stream parses SSE (sources dict first, then text tokens, error->text);
  * unsupported public members + directly-called private methods raise
    NotImplementedError, while private introspection falls back gracefully.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from axon import remote_brain
from axon.remote_brain import RemoteBrain, get_brain


def _fake_config():
    return SimpleNamespace(
        top_k=5,
        similarity_threshold=0.25,
        hybrid_search=True,
        rerank=False,
        hyde=False,
        multi_query=False,
        step_back=False,
        query_decompose=False,
        compress_context=False,
        discussion_fallback=False,
        llm_temperature=0.3,
        api_host="127.0.0.1",
        api_port=8420,
        api_key="",
        projects_root="/store/physics_kg",
    )


def _make_brain(record):
    """Build a RemoteBrain whose HTTP layer is replaced by a recorder.

    *record* is a list; each request appends (method, path, body). The recorder
    returns a canned response keyed on path.
    """
    brain = RemoteBrain(_fake_config(), {"project": "physics_kg", "_api_base": "http://x:8420"})

    def fake_request(method, path, body=None, timeout=None):
        record.append((method, path, body))
        if path == "/query":
            return {"response": "the answer", "sources": [{"id": "d1"}]}
        if path == "/collection":
            return {"total_files": 1, "total_chunks": 3, "files": [{"source": "a.md", "chunks": 3}]}
        if path == "/tracked-docs":
            return {"docs": {"a.md": {"version": 1}}}
        if path == "/mount/refresh":
            return {"status": "success", "refreshed": True}
        return {"status": "success"}

    brain._request = fake_request
    return brain


# ---------------------------------------------------------------------------
# get_brain factory
# ---------------------------------------------------------------------------
def test_get_brain_returns_remote_when_server_detected(monkeypatch):
    monkeypatch.setattr(
        remote_brain._sc,
        "detect_server",
        lambda cfg: {"status": "ok", "project": "physics_kg", "_api_base": "http://127.0.0.1:8420"},
    )

    # AxonBrain must NOT be constructed on the remote path.
    import axon.main

    def _boom(*a, **k):
        raise AssertionError("AxonBrain should not be constructed when a server is reused")

    monkeypatch.setattr(axon.main, "AxonBrain", _boom)

    brain = get_brain(_fake_config())
    assert isinstance(brain, RemoteBrain)
    assert brain._active_project == "physics_kg"
    assert brain._api_base == "http://127.0.0.1:8420"


def test_get_brain_returns_local_when_no_server(monkeypatch):
    monkeypatch.setattr(remote_brain._sc, "detect_server", lambda cfg: None)

    import axon.main

    sentinel = object()

    class _Fake:
        def __init__(self, config):
            self.config = config
            self.tag = sentinel

    monkeypatch.setattr(axon.main, "AxonBrain", _Fake)

    brain = get_brain(_fake_config())
    assert not isinstance(brain, RemoteBrain)
    assert brain.tag is sentinel


def test_get_brain_allow_remote_false_skips_detection(monkeypatch):
    called = {"n": 0}

    def _detect(cfg):
        called["n"] += 1
        return {"project": "physics_kg", "_api_base": "http://x"}

    monkeypatch.setattr(remote_brain._sc, "detect_server", _detect)

    import axon.main

    class _Fake:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(axon.main, "AxonBrain", _Fake)

    brain = get_brain(_fake_config(), allow_remote=False)
    assert isinstance(brain, _Fake)
    assert called["n"] == 0  # detection skipped entirely


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
def test_query_routes_to_query_endpoint():
    record = []
    brain = _make_brain(record)
    out = brain.query("what is DUV?")
    assert out == "the answer"
    method, path, body = record[0]
    assert (method, path) == ("POST", "/query")
    assert body["query"] == "what is DUV?"
    assert body["project"] == "physics_kg"
    # Local config RAG flags forwarded as per-request overrides.
    assert body["hybrid"] is True
    assert body["top_k"] == 5


def test_query_chat_history_is_dropped_but_accepted():
    record = []
    brain = _make_brain(record)
    # Must not raise even though the server has no chat_history field.
    out = brain.query("q", chat_history=[{"role": "user", "content": "prev"}])
    assert out == "the answer"
    _, _, body = record[0]
    assert "chat_history" not in body


def test_switch_project_routes_and_updates_active():
    record = []
    brain = _make_brain(record)
    brain.switch_project("optics")
    method, path, body = record[0]
    assert (method, path) == ("POST", "/project/switch")
    assert body == {"project_name": "optics"}
    assert brain._active_project == "optics"
    # Subsequent query asserts the new project.
    brain.query("q2")
    assert record[1][2]["project"] == "optics"


def test_ingest_routes_to_add_texts_with_mapped_docs():
    record = []
    brain = _make_brain(record)
    brain.ingest([{"id": "doc1", "text": "hello", "metadata": {"source": "a.md"}}])
    method, path, body = record[0]
    assert (method, path) == ("POST", "/add_texts")
    assert body["project"] == "physics_kg"
    assert body["docs"] == [{"text": "hello", "doc_id": "doc1", "metadata": {"source": "a.md"}}]


def test_ingest_empty_is_noop():
    record = []
    brain = _make_brain(record)
    brain.ingest([])
    assert record == []


def test_list_documents_routes_to_collection():
    record = []
    brain = _make_brain(record)
    docs = brain.list_documents()
    assert record[0][:2] == ("GET", "/collection")
    assert docs == [{"source": "a.md", "chunks": 3}]


def test_get_doc_versions_routes_to_tracked_docs():
    record = []
    brain = _make_brain(record)
    versions = brain.get_doc_versions()
    assert record[0][:2] == ("GET", "/tracked-docs")
    assert versions == {"a.md": {"version": 1}}


def test_refresh_mount_returns_bool():
    record = []
    brain = _make_brain(record)
    assert brain.refresh_mount() is True
    assert record[0][:2] == ("POST", "/mount/refresh")


def test_finalize_graph_routes():
    record = []
    brain = _make_brain(record)
    assert brain.finalize_graph(True) is None
    assert record[0][:2] == ("POST", "/graph/finalize")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------
def test_query_stream_interprets_payloads(monkeypatch):
    brain = RemoteBrain(_fake_config(), {"project": "physics_kg", "_api_base": "http://x"})

    payloads = [
        json.dumps({"type": "sources", "sources": [{"id": "d1"}]}),
        "Hello ",
        "world",
        "5",  # a token that happens to be JSON-parseable -> stays a string
    ]
    monkeypatch.setattr(brain, "_post_stream", lambda path, body: iter(payloads))

    chunks = list(brain.query_stream("q"))
    assert chunks[0] == {"type": "sources", "sources": [{"id": "d1"}]}
    assert chunks[1:] == ["Hello ", "world", "5"]


def test_query_stream_error_payload_becomes_text(monkeypatch):
    brain = RemoteBrain(_fake_config(), {"project": "physics_kg", "_api_base": "http://x"})
    payloads = [json.dumps({"type": "error", "content": "boom"})]
    monkeypatch.setattr(brain, "_post_stream", lambda path, body: iter(payloads))
    chunks = list(brain.query_stream("q"))
    assert chunks == ["[ERROR] boom"]


class _FakeSSEResponse:
    """Minimal stand-in for the urllib streaming response object."""

    def __init__(self, data: bytes):
        self._data = data
        self._sent = False

    def read1(self, n):  # noqa: D401 - test stub
        if self._sent:
            return b""
        self._sent = True
        return self._data

    def close(self):
        pass


def test_post_stream_parses_sse_over_urlopen(monkeypatch):
    brain = RemoteBrain(_fake_config(), {"project": "physics_kg", "_api_base": "http://x"})
    sse = b'data: {"type": "sources", "sources": []}\n\n' b"data: Hello \n\n" b"data: world\n\n"
    monkeypatch.setattr(
        remote_brain.urllib.request, "urlopen", lambda req, timeout=None: _FakeSSEResponse(sse)
    )
    out = list(brain.query_stream("q"))
    assert out[0] == {"type": "sources", "sources": []}
    assert out[1:] == ["Hello ", "world"]


# ---------------------------------------------------------------------------
# Loud gaps vs graceful introspection
# ---------------------------------------------------------------------------
def test_unsupported_public_member_raises():
    brain = RemoteBrain(_fake_config(), {"project": "p", "_api_base": "http://x"})
    with pytest.raises(NotImplementedError):
        _ = brain.some_unproxied_public_member


def test_vector_store_raises():
    brain = RemoteBrain(_fake_config(), {"project": "p", "_api_base": "http://x"})
    with pytest.raises(NotImplementedError):
        _ = brain.vector_store


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.should_recommend_project(),
        lambda b: b.export_graph_html(),
        lambda b: b.wipe_sealed_cache(),
        lambda b: b._build_system_prompt(False),
        lambda b: b._resolve_model_path("x"),
        lambda b: b._assert_write_allowed("clear"),
        lambda b: b._is_mounted_share(),
    ],
)
def test_directly_called_gaps_raise(call):
    brain = RemoteBrain(_fake_config(), {"project": "p", "_api_base": "http://x"})
    with pytest.raises(NotImplementedError):
        call(brain)


def test_private_introspection_falls_back_gracefully():
    brain = RemoteBrain(_fake_config(), {"project": "p", "_api_base": "http://x"})
    # The REPL probes many private attrs defensively; these must yield the
    # default rather than raising, so getattr-with-default keeps working.
    assert getattr(brain, "_entity_graph", {}) == {}
    assert getattr(brain, "_graph_backend", None) is None
    assert getattr(brain, "_active_project_kind", "default") == "default"
    assert getattr(brain, "_community_summaries", {}) == {}


# ---------------------------------------------------------------------------
# Local lightweight clients
# ---------------------------------------------------------------------------
def test_config_is_the_local_config_object():
    cfg = _fake_config()
    brain = RemoteBrain(cfg, {"project": "p", "_api_base": "http://x"})
    assert brain.config is cfg


def test_llm_is_lazy_and_local_and_settable():
    brain = RemoteBrain(_fake_config(), {"project": "p", "_api_base": "http://x"})
    assert brain._llm is None  # not constructed until accessed
    from axon.llm import OpenLLM

    llm = brain.llm
    assert isinstance(llm, OpenLLM)
    assert brain.llm is llm  # cached
    replacement = OpenLLM(brain.config)
    brain.llm = replacement
    assert brain.llm is replacement


def test_embedding_and_reranker_not_constructed_without_access():
    # Constructing OpenEmbedding/OpenReranker loads models; ensure a plain
    # RemoteBrain never triggers that on its own.
    brain = RemoteBrain(_fake_config(), {"project": "p", "_api_base": "http://x"})
    assert brain._embedding is None
    assert brain._reranker is None
