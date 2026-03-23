import hashlib
import math
import re
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

import axon.api as api_module
import axon.main as main_module
import axon.projects as projects_module
from axon.main import AxonBrain, AxonConfig


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_./:-]+", text.lower())


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class FakeEmbedding:
    def __init__(self, config: AxonConfig):
        self.config = config
        self.provider = config.embedding_provider
        self.dimension = 32

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _tokenize(text):
            bucket = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16) % self.dimension
            vector[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm:
            vector = [v / norm for v in vector]
        return vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_text(query)


class FakeCollection:
    def __init__(self, vector_store: "FakeVectorStore"):
        self.vector_store = vector_store

    def add(self, ids, documents, embeddings, metadatas=None):
        self.vector_store.add(ids, documents, embeddings, metadatas)

    def query(self, query_embeddings, n_results=10, where=None):
        results = self.vector_store.search(query_embeddings[0], top_k=n_results, filter_dict=where)
        return {
            "ids": [[doc["id"] for doc in results]],
            "documents": [[doc["text"] for doc in results]],
            "metadatas": [[doc.get("metadata", {}) for doc in results]],
            "distances": [[1.0 - doc.get("score", 0.0) for doc in results]],
        }

    def get(self, ids=None, include=None):
        docs = (
            self.vector_store.get_by_ids(ids)
            if ids is not None
            else list(self.vector_store._storage.values())
        )
        return {
            "ids": [doc["id"] for doc in docs],
            "documents": [doc["text"] for doc in docs],
            "metadatas": [doc.get("metadata", {}) for doc in docs],
        }

    def delete(self, ids=None):
        self.vector_store.delete_by_ids(ids or [])

    def count(self):
        return len(self.vector_store._storage)


class FakeChromaClient:
    def __init__(self, vector_store: "FakeVectorStore"):
        self.vector_store = vector_store

    def delete_collection(self, _name: str):
        self.vector_store._storage.clear()

    def create_collection(self, name: str, metadata=None):
        return self.vector_store.collection


class FakeVectorStore:
    _stores: dict[str, dict[str, dict]] = {}

    def __init__(self, config: AxonConfig):
        self.config = config
        self.provider = "chroma"
        self.path = str(Path(config.vector_store_path).resolve())
        Path(self.path).mkdir(parents=True, exist_ok=True)
        self._storage = self._stores.setdefault(self.path, {})
        self.collection = FakeCollection(self)
        self.client = FakeChromaClient(self)

    @staticmethod
    def _matches_filters(metadata: dict, filter_dict: dict | None) -> bool:
        if not filter_dict:
            return True
        if "$and" in filter_dict:
            return all(
                FakeVectorStore._matches_filters(metadata, item) for item in filter_dict["$and"]
            )
        for key, value in filter_dict.items():
            if isinstance(value, dict) and "$eq" in value:
                if metadata.get(key) != value["$eq"]:
                    return False
            elif metadata.get(key) != value:
                return False
        return True

    def add(self, ids, texts, embeddings, metadatas=None):
        metadatas = metadatas or [{} for _ in ids]
        for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, metadatas):
            self._storage[doc_id] = {
                "id": doc_id,
                "text": text,
                "embedding": embedding,
                "metadata": metadata or {},
            }

    def search(self, query_embedding, top_k=10, filter_dict=None):
        matches = []
        for doc in self._storage.values():
            if not self._matches_filters(doc.get("metadata", {}), filter_dict):
                continue
            score = _cosine_similarity(query_embedding, doc["embedding"])
            matches.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": score,
                    "metadata": dict(doc.get("metadata", {})),
                }
            )
        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:top_k]

    def get_by_ids(self, ids):
        docs = []
        for doc_id in ids:
            if doc_id not in self._storage:
                continue
            doc = self._storage[doc_id]
            docs.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": 1.0,
                    "metadata": dict(doc.get("metadata", {})),
                }
            )
        return docs

    def delete_by_ids(self, ids):
        for doc_id in ids:
            self._storage.pop(doc_id, None)

    def list_documents(self):
        grouped: dict[str, dict] = {}
        for doc in self._storage.values():
            source = doc.get("metadata", {}).get("source", "unknown")
            entry = grouped.setdefault(source, {"source": source, "chunks": 0, "doc_ids": []})
            entry["chunks"] += 1
            entry["doc_ids"].append(doc["id"])
        return sorted(grouped.values(), key=lambda item: item["source"])

    def close(self):
        return None


class FakeReranker:
    def __init__(self, config: AxonConfig):
        self.config = config

    def rerank(self, query: str, documents: list[dict]) -> list[dict]:
        query_tokens = set(_tokenize(query))
        rescored = []
        for doc in documents:
            lexical = len(query_tokens.intersection(_tokenize(doc.get("text", ""))))
            rescored.append({**doc, "rerank_score": lexical + doc.get("score", 0.0)})
        rescored.sort(key=lambda item: item["rerank_score"], reverse=True)
        return rescored


class FakeLLM:
    def __init__(self, config: AxonConfig):
        self.config = config

    @staticmethod
    def _extract_blocks(system_prompt: str) -> list[tuple[str, str]]:
        marker = "**Relevant context from documents:**"
        if marker not in system_prompt:
            return []
        context = system_prompt.split(marker, 1)[1].strip()
        blocks = []
        current_label = None
        current_lines: list[str] = []
        for line in context.splitlines():
            if (
                line.startswith("[Document ")
                or line.startswith("[Web Result ")
                or line.startswith("**GraphRAG ")
                or line.startswith("**Knowledge Graph ")
            ):
                if current_label is not None:
                    blocks.append((current_label, "\n".join(current_lines).strip()))
                current_label = line.strip()
                current_lines = []
            else:
                current_lines.append(line)
        if current_label is not None:
            blocks.append((current_label, "\n".join(current_lines).strip()))
        return blocks

    def _best_answer(self, prompt: str, system_prompt: str | None) -> str:
        blocks = self._extract_blocks(system_prompt or "")
        if not blocks:
            return f"No grounded context found for: {prompt}"
        query_tokens = set(_tokenize(prompt))
        best_label, best_text = blocks[0]
        best_score = -1
        for label, text in blocks:
            candidate = text or label
            score = len(query_tokens.intersection(_tokenize(candidate)))
            if score > best_score:
                best_score = score
                best_label, best_text = label, text
        sentence = (
            re.split(r"(?<=[.!?])\s+|\n+", best_text.strip())[0].strip()
            if best_text.strip()
            else best_label
        )
        return f"{sentence} {best_label}".strip()

    def complete(self, prompt: str, system_prompt: str = None, chat_history=None) -> str:
        return self._best_answer(prompt, system_prompt)

    def stream(self, prompt: str, system_prompt: str = None, chat_history=None):
        answer = self._best_answer(prompt, system_prompt)
        for token in answer.split():
            yield token + " "


# NOTE: tmp_path is intentionally NOT overridden here.
# pytest's built-in tmp_path fixture (uses tempfile.gettempdir()) is used directly.
# A previous custom implementation that wrote to tests/e2e/.test_tmp caused Windows
# PermissionError failures because locked handles from prior test runs prevent both
# directory creation and ruff/black scans of the project tree.


@pytest.fixture(autouse=True)
def reset_api_state(monkeypatch, tmp_path):
    api_module.brain = None
    api_module._source_hashes.clear()
    api_module._jobs.clear()
    FakeVectorStore._stores.clear()

    projects_root = tmp_path / "projects_root"
    active_file = tmp_path / ".active_project"
    monkeypatch.setattr(projects_module, "PROJECTS_ROOT", projects_root)
    monkeypatch.setattr(projects_module, "_ACTIVE_FILE", active_file)
    monkeypatch.setenv("RAG_INGEST_BASE", str(tmp_path))

    yield

    if api_module.brain is not None:
        try:
            api_module.brain.close()
        except Exception:
            pass
    api_module.brain = None
    api_module._source_hashes.clear()
    api_module._jobs.clear()
    FakeVectorStore._stores.clear()


@pytest.fixture
def api_client():
    client = TestClient(api_module.app, raise_server_exceptions=False)
    yield client
    client.close()


@pytest.fixture
def make_brain(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "OpenEmbedding", FakeEmbedding)
    monkeypatch.setattr(main_module, "OpenVectorStore", FakeVectorStore)
    monkeypatch.setattr(main_module, "OpenLLM", FakeLLM)
    monkeypatch.setattr(main_module, "OpenReranker", FakeReranker)

    created: list[AxonBrain] = []

    def _make(**overrides) -> AxonBrain:
        projects_root = tmp_path / "projects_root"
        default_root = projects_root / "default"
        projects_root.mkdir(parents=True, exist_ok=True)
        config_kwargs = {
            "embedding_provider": "sentence_transformers",
            "embedding_model": "fake-minilm",
            "llm_provider": "ollama",
            "llm_model": "fake-llm",
            "vector_store": "chroma",
            "vector_store_path": str(default_root / "chroma_data"),
            "bm25_path": str(default_root / "bm25_index"),
            "projects_root": str(projects_root),
            "top_k": 5,
            "similarity_threshold": 0.0,
            "hybrid_search": True,
            "hybrid_mode": "weighted",
            "rerank": False,
            "raptor": False,
            "graph_rag": False,
            "graph_rag_relations": False,
            "graph_rag_community": False,
            "parent_chunk_size": 0,
            "discussion_fallback": False,
            "truth_grounding": False,
            "query_cache": False,
            "smart_ingest": True,
            "chunk_strategy": "semantic",
            "chunk_size": 220,
            "chunk_overlap": 40,
            "max_workers": 2,
        }
        config_kwargs.update(overrides)
        config = AxonConfig(**config_kwargs)
        brain = AxonBrain(config)
        created.append(brain)
        api_module.brain = brain
        return brain

    yield _make

    for brain in created:
        try:
            brain.close()
        except Exception:
            pass


@pytest.fixture
def sample_docs_dir(tmp_path) -> Path:
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "overview.txt").write_text(
        "Axon uses BM25 and vector search to retrieve grounded context for questions.",
        encoding="utf-8",
    )
    (docs_dir / "notes.md").write_text(
        "# Notes\n\nGraphRAG links entities and relationships for broader corpus reasoning.",
        encoding="utf-8",
    )
    return docs_dir


@pytest.fixture
def live_api_server():  # noqa: F811 - keep existing fixture name
    uvicorn = pytest.importorskip("uvicorn")
    servers: list[tuple[object, threading.Thread]] = []

    def _start() -> str:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
        sock.close()

        config = uvicorn.Config(
            api_module.app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
            lifespan="off",
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        base_url = f"http://127.0.0.1:{port}"
        last_error: Exception | None = None
        for _ in range(100):
            try:
                response = httpx.get(f"{base_url}/health", timeout=0.25)
                if response.status_code in (200, 503):
                    servers.append((server, thread))
                    return base_url
            except Exception as exc:  # pragma: no cover - startup race
                last_error = exc
                time.sleep(0.05)
            else:
                time.sleep(0.05)

        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError(f"Timed out starting live API server: {last_error}")

    yield _start

    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# VS Code extension e2e fixtures (WS5 — surface parity qualification)
# ---------------------------------------------------------------------------


def _load_vscode_helpers():
    """Load vscode_helpers module from the same directory without relying on sys.path."""
    import importlib.util

    helpers_path = Path(__file__).parent / "vscode_helpers.py"
    spec = importlib.util.spec_from_file_location("vscode_helpers", helpers_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def extension_root_path():
    """Path to the VS Code extension directory."""
    root = Path(__file__).parent.parent.parent / "integrations" / "vscode-axon"
    if not root.exists():
        pytest.skip("VS Code extension directory not found at integrations/vscode-axon/")
    return root


@pytest.fixture(scope="session")
def extension_js_path(extension_root_path):
    """Path to the compiled extension JS. Skips if not built or Node.js absent."""
    import shutil

    if not shutil.which("node"):
        pytest.skip("Node.js not available — install Node.js to run VS Code extension tests")
    js = extension_root_path / "out" / "extension.js"
    if not js.exists():
        pytest.skip("Extension not built — run: cd integrations/vscode-axon && npm run compile")
    return js


@pytest.fixture(scope="session")
def runner_js_path():
    """Write the Node.js runner template to a temporary file for the session."""
    import os
    import tempfile

    helpers = _load_vscode_helpers()
    fd, path = tempfile.mkstemp(suffix="_axon_runner.js", prefix="pytest_")
    try:
        os.write(fd, helpers._RUNNER_JS.encode("utf-8"))
    finally:
        os.close(fd)
    yield Path(path)
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def run_tool(runner_js_path, extension_js_path, extension_root_path):
    """Callable that runs a single VS Code LM tool via the Node.js runner."""
    helpers = _load_vscode_helpers()

    def _run(
        base_url: str,
        tool_name: str | None,
        tool_input: dict | None = None,
        extra_config: dict | None = None,
    ) -> dict:
        return helpers.run_extension_tool(
            runner_js_path=runner_js_path,
            extension_js=extension_js_path,
            extension_root=extension_root_path,
            api_base=base_url,
            tool_name=tool_name or "",
            tool_input=tool_input or {},
            extra_config=extra_config,
        )

    return _run


@pytest.fixture
def live_recorder_server():
    """Start a lightweight HTTP recording server that returns mock API responses.

    Yields ``(base_url, recorded)`` where ``recorded`` accumulates request dicts
    with keys ``method``, ``path``, and ``body``.
    """
    import json
    import socket
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import urlparse

    helpers = _load_vscode_helpers()
    recorded: list[dict] = []

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, _format, *_args):
            pass  # suppress server log noise in test output

        def do_GET(self):
            self._handle("GET", {})

        def do_POST(self):
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length)) if length else {}
            self._handle("POST", body)

        def _handle(self, method: str, body: dict):
            parsed = urlparse(self.path)
            path = parsed.path
            headers = {k.lower(): v for k, v in self.headers.items()}
            recorded.append({"method": method, "path": path, "body": body, "headers": headers})
            resp = helpers.mock_api_response(path, method, body)
            payload = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()

    server = HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    base_url = f"http://127.0.0.1:{port}"
    yield base_url, recorded

    server.shutdown()
    t.join(timeout=5)
