from __future__ import annotations

"""Tests for CodeGraphMixin in axon.code_graph."""
import json

from axon.config import AxonConfig


def _make_brain(tmp_path):
    """Return a minimal object with CodeGraphMixin capabilities."""
    from axon.code_graph import CodeGraphMixin

    cfg = AxonConfig()
    # Override derived paths so the brain reads/writes in tmp_path, not the real store
    cfg.bm25_path = str(tmp_path)
    cfg.vector_store_path = str(tmp_path)

    class FakeBrain(CodeGraphMixin):
        def __init__(self):
            self.config = cfg
            self._code_graph = {"nodes": {}, "edges": []}

    return FakeBrain()


class TestCodeGraphLoad:
    def test_load_returns_empty_when_no_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        result = brain._load_code_graph()
        assert result == {"nodes": {}, "edges": []}

    def test_load_returns_data_from_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        data = {"nodes": {"a": {"node_id": "a"}}, "edges": [{"source": "a"}]}
        (tmp_path / ".code_graph.json").write_text(json.dumps(data), encoding="utf-8")
        result = brain._load_code_graph()
        assert result["nodes"]["a"]["node_id"] == "a"

    def test_load_corrupt_file_returns_empty(self, tmp_path):
        brain = _make_brain(tmp_path)
        (tmp_path / ".code_graph.json").write_text("not json{{{", encoding="utf-8")
        result = brain._load_code_graph()
        assert result == {"nodes": {}, "edges": []}

    def test_load_missing_keys_returns_empty(self, tmp_path):
        brain = _make_brain(tmp_path)
        (tmp_path / ".code_graph.json").write_text(json.dumps({"other": 1}), encoding="utf-8")
        result = brain._load_code_graph()
        assert result == {"nodes": {}, "edges": []}


class TestCodeGraphSave:
    def test_save_writes_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._code_graph = {"nodes": {"a": {"node_id": "a"}}, "edges": []}
        brain._save_code_graph()
        path = tmp_path / ".code_graph.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "nodes" in data

    def test_save_creates_parent_dirs(self, tmp_path):
        brain = _make_brain(tmp_path)
        nested = tmp_path / "sub" / "dir"
        brain.config.bm25_path = str(nested)
        brain._code_graph = {"nodes": {}, "edges": []}
        brain._save_code_graph()
        assert (nested / ".code_graph.json").exists()


class TestBuildCodeGraph:
    def test_non_code_chunks_skipped(self, tmp_path):
        brain = _make_brain(tmp_path)
        chunks = [{"id": "c1", "text": "hello", "metadata": {"source_class": "text"}}]
        brain._build_code_graph_from_chunks(chunks)
        assert brain._code_graph["nodes"] == {}

    def test_code_chunk_creates_file_node(self, tmp_path):
        brain = _make_brain(tmp_path)
        chunks = [
            {
                "id": "c1",
                "text": "def foo(): pass",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/foo.py",
                    "language": "python",
                    "symbol_type": "block",
                    "symbol_name": "",
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        assert "/src/foo.py" in brain._code_graph["nodes"]
        node = brain._code_graph["nodes"]["/src/foo.py"]
        assert node["node_type"] == "file"
        assert node["language"] == "python"

    def test_code_chunk_creates_symbol_node(self, tmp_path):
        brain = _make_brain(tmp_path)
        chunks = [
            {
                "id": "c1",
                "text": "def foo(): pass",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/foo.py",
                    "language": "python",
                    "symbol_type": "function",
                    "symbol_name": "foo",
                    "signature": "def foo():",
                    "start_line": 1,
                    "end_line": 1,
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        sym_id = "/src/foo.py::foo"
        assert sym_id in brain._code_graph["nodes"]
        assert brain._code_graph["nodes"][sym_id]["node_type"] == "function"

    def test_contains_edge_created(self, tmp_path):
        brain = _make_brain(tmp_path)
        chunks = [
            {
                "id": "c1",
                "text": "def bar(): pass",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/bar.py",
                    "language": "python",
                    "symbol_type": "function",
                    "symbol_name": "bar",
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        edges = brain._code_graph["edges"]
        contains_edges = [e for e in edges if e["edge_type"] == "CONTAINS"]
        assert len(contains_edges) == 1
        assert contains_edges[0]["source"] == "/src/bar.py"

    def test_imports_edge_from_string(self, tmp_path):
        brain = _make_brain(tmp_path)
        # First add target file
        brain._code_graph["nodes"]["/src/utils.py"] = {
            "node_id": "/src/utils.py",
            "node_type": "file",
            "file_path": "/src/utils.py",
        }
        chunks = [
            {
                "id": "c1",
                "text": "import utils",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/main.py",
                    "language": "python",
                    "symbol_type": "block",
                    "symbol_name": "",
                    "imports": "from src.utils import helper",
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        edges = brain._code_graph["edges"]
        import_edges = [e for e in edges if e["edge_type"] == "IMPORTS"]
        assert len(import_edges) == 1

    def test_imports_from_list(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._code_graph["nodes"]["/src/utils.py"] = {
            "node_id": "/src/utils.py",
            "node_type": "file",
            "file_path": "/src/utils.py",
        }
        chunks = [
            {
                "id": "c2",
                "text": "",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/main2.py",
                    "language": "python",
                    "symbol_type": "block",
                    "symbol_name": "",
                    "imports": ["from src.utils import x"],
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        edges = brain._code_graph["edges"]
        assert any(e["edge_type"] == "IMPORTS" for e in edges)

    def test_duplicate_chunks_not_added(self, tmp_path):
        brain = _make_brain(tmp_path)
        chunk = {
            "id": "c1",
            "text": "def dup(): pass",
            "metadata": {
                "source_class": "code",
                "file_path": "/src/dup.py",
                "language": "python",
                "symbol_type": "function",
                "symbol_name": "dup",
            },
        }
        brain._build_code_graph_from_chunks([chunk, chunk])
        sym_node = brain._code_graph["nodes"]["/src/dup.py::dup"]
        assert sym_node["chunk_ids"].count("c1") == 1


class TestResolveImport:
    def test_from_import_resolves(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._code_graph["nodes"]["/src/axon/splitters.py"] = {
            "node_type": "file",
            "file_path": "/src/axon/splitters.py",
        }
        result = brain._resolve_import_to_file("from axon.splitters import Chunker")
        assert result == "/src/axon/splitters.py"

    def test_import_resolves(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._code_graph["nodes"]["/src/axon/splitters.py"] = {
            "node_type": "file",
            "file_path": "/src/axon/splitters.py",
        }
        result = brain._resolve_import_to_file("import axon.splitters")
        assert result == "/src/axon/splitters.py"

    def test_no_match_returns_none(self, tmp_path):
        brain = _make_brain(tmp_path)
        result = brain._resolve_import_to_file("from nonexistent import x")
        assert result is None

    def test_bad_stmt_returns_none(self, tmp_path):
        brain = _make_brain(tmp_path)
        result = brain._resolve_import_to_file("not an import statement")
        assert result is None


"""Tests for CodeGraphMixin in axon.code_graph."""


def _make_brain(tmp_path):
    """Return a minimal object with CodeGraphMixin capabilities."""
    from axon.code_graph import CodeGraphMixin

    cfg = AxonConfig()
    # Override derived paths so the brain reads/writes in tmp_path, not the real store
    cfg.bm25_path = str(tmp_path)
    cfg.vector_store_path = str(tmp_path)

    class FakeBrain(CodeGraphMixin):
        def __init__(self):
            self.config = cfg
            self._code_graph = {"nodes": {}, "edges": []}

    return FakeBrain()


class TestCodeGraphLoadV2:
    def test_load_returns_empty_when_no_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        result = brain._load_code_graph()
        assert result == {"nodes": {}, "edges": []}

    def test_load_returns_data_from_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        data = {"nodes": {"a": {"node_id": "a"}}, "edges": [{"source": "a"}]}
        (tmp_path / ".code_graph.json").write_text(json.dumps(data), encoding="utf-8")
        result = brain._load_code_graph()
        assert result["nodes"]["a"]["node_id"] == "a"

    def test_load_corrupt_file_returns_empty(self, tmp_path):
        brain = _make_brain(tmp_path)
        (tmp_path / ".code_graph.json").write_text("not json{{{", encoding="utf-8")
        result = brain._load_code_graph()
        assert result == {"nodes": {}, "edges": []}

    def test_load_missing_keys_returns_empty(self, tmp_path):
        brain = _make_brain(tmp_path)
        (tmp_path / ".code_graph.json").write_text(json.dumps({"other": 1}), encoding="utf-8")
        result = brain._load_code_graph()
        assert result == {"nodes": {}, "edges": []}


class TestCodeGraphSaveV2:
    def test_save_writes_file(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._code_graph = {"nodes": {"a": {"node_id": "a"}}, "edges": []}
        brain._save_code_graph()
        path = tmp_path / ".code_graph.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert "nodes" in data

    def test_save_creates_parent_dirs(self, tmp_path):
        brain = _make_brain(tmp_path)
        nested = tmp_path / "sub" / "dir"
        brain.config.bm25_path = str(nested)
        brain._code_graph = {"nodes": {}, "edges": []}
        brain._save_code_graph()
        assert (nested / ".code_graph.json").exists()


class TestBuildCodeGraphV2:
    def test_non_code_chunks_skipped(self, tmp_path):
        brain = _make_brain(tmp_path)
        chunks = [{"id": "c1", "text": "hello", "metadata": {"source_class": "text"}}]
        brain._build_code_graph_from_chunks(chunks)
        assert brain._code_graph["nodes"] == {}

    def test_code_chunk_creates_file_node(self, tmp_path):
        brain = _make_brain(tmp_path)
        chunks = [
            {
                "id": "c1",
                "text": "def foo(): pass",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/foo.py",
                    "language": "python",
                    "symbol_type": "block",
                    "symbol_name": "",
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        assert "/src/foo.py" in brain._code_graph["nodes"]
        node = brain._code_graph["nodes"]["/src/foo.py"]
        assert node["node_type"] == "file"
        assert node["language"] == "python"

    def test_code_chunk_creates_symbol_node(self, tmp_path):
        brain = _make_brain(tmp_path)
        chunks = [
            {
                "id": "c1",
                "text": "def foo(): pass",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/foo.py",
                    "language": "python",
                    "symbol_type": "function",
                    "symbol_name": "foo",
                    "signature": "def foo():",
                    "start_line": 1,
                    "end_line": 1,
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        sym_id = "/src/foo.py::foo"
        assert sym_id in brain._code_graph["nodes"]
        assert brain._code_graph["nodes"][sym_id]["node_type"] == "function"

    def test_contains_edge_created(self, tmp_path):
        brain = _make_brain(tmp_path)
        chunks = [
            {
                "id": "c1",
                "text": "def bar(): pass",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/bar.py",
                    "language": "python",
                    "symbol_type": "function",
                    "symbol_name": "bar",
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        edges = brain._code_graph["edges"]
        contains_edges = [e for e in edges if e["edge_type"] == "CONTAINS"]
        assert len(contains_edges) == 1
        assert contains_edges[0]["source"] == "/src/bar.py"

    def test_imports_edge_from_string(self, tmp_path):
        brain = _make_brain(tmp_path)
        # First add target file
        brain._code_graph["nodes"]["/src/utils.py"] = {
            "node_id": "/src/utils.py",
            "node_type": "file",
            "file_path": "/src/utils.py",
        }
        chunks = [
            {
                "id": "c1",
                "text": "import utils",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/main.py",
                    "language": "python",
                    "symbol_type": "block",
                    "symbol_name": "",
                    "imports": "from src.utils import helper",
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        edges = brain._code_graph["edges"]
        import_edges = [e for e in edges if e["edge_type"] == "IMPORTS"]
        assert len(import_edges) == 1

    def test_imports_from_list(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._code_graph["nodes"]["/src/utils.py"] = {
            "node_id": "/src/utils.py",
            "node_type": "file",
            "file_path": "/src/utils.py",
        }
        chunks = [
            {
                "id": "c2",
                "text": "",
                "metadata": {
                    "source_class": "code",
                    "file_path": "/src/main2.py",
                    "language": "python",
                    "symbol_type": "block",
                    "symbol_name": "",
                    "imports": ["from src.utils import x"],
                },
            }
        ]
        brain._build_code_graph_from_chunks(chunks)
        edges = brain._code_graph["edges"]
        assert any(e["edge_type"] == "IMPORTS" for e in edges)

    def test_duplicate_chunks_not_added(self, tmp_path):
        brain = _make_brain(tmp_path)
        chunk = {
            "id": "c1",
            "text": "def dup(): pass",
            "metadata": {
                "source_class": "code",
                "file_path": "/src/dup.py",
                "language": "python",
                "symbol_type": "function",
                "symbol_name": "dup",
            },
        }
        brain._build_code_graph_from_chunks([chunk, chunk])
        sym_node = brain._code_graph["nodes"]["/src/dup.py::dup"]
        assert sym_node["chunk_ids"].count("c1") == 1


class TestResolveImportV2:
    def test_from_import_resolves(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._code_graph["nodes"]["/src/axon/splitters.py"] = {
            "node_type": "file",
            "file_path": "/src/axon/splitters.py",
        }
        result = brain._resolve_import_to_file("from axon.splitters import Chunker")
        assert result == "/src/axon/splitters.py"

    def test_import_resolves(self, tmp_path):
        brain = _make_brain(tmp_path)
        brain._code_graph["nodes"]["/src/axon/splitters.py"] = {
            "node_type": "file",
            "file_path": "/src/axon/splitters.py",
        }
        result = brain._resolve_import_to_file("import axon.splitters")
        assert result == "/src/axon/splitters.py"

    def test_no_match_returns_none(self, tmp_path):
        brain = _make_brain(tmp_path)
        result = brain._resolve_import_to_file("from nonexistent import x")
        assert result is None

    def test_bad_stmt_returns_none(self, tmp_path):
        brain = _make_brain(tmp_path)
        result = brain._resolve_import_to_file("not an import statement")
        assert result is None
