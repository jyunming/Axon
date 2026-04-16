from __future__ import annotations

"""Tests for axon.code_retrieval to reach ≥90% coverage."""

from unittest.mock import MagicMock

import pytest

from axon.code_retrieval import (
    CodeRetrievalDiagnostics,
    CodeRetrievalMixin,
    _build_code_bm25_queries,
    _classify_retrieval_failure,
)
from axon.code_retrieval import (
    _looks_like_code_query as _is_code_query,
)

# ---------------------------------------------------------------------------
# _is_code_query (lines 80-92)
# ---------------------------------------------------------------------------


class TestIsCodeQuery:
    def test_camel_case_is_code(self):
        assert _is_code_query("QueryRouter") is True

    def test_snake_case_is_code(self):
        assert _is_code_query("split_python_ast") is True

    def test_code_extension_is_code(self):
        assert _is_code_query("main.py") is True
        assert _is_code_query("index.ts") is True

    def test_symbol_keyword(self):
        assert _is_code_query("def build_graph") is True

    def test_plain_prose_not_code(self):
        assert _is_code_query("what is the weather today") is False

    def test_short_tokens_not_code(self):
        assert _is_code_query("hi") is False


# ---------------------------------------------------------------------------
# _build_code_bm25_queries (lines 104-142)
# ---------------------------------------------------------------------------


class TestBuildCodeBm25Queries:
    def test_camel_case_expansion(self):
        result = _build_code_bm25_queries("CodeAwareSplitter", frozenset(["CodeAwareSplitter"]))
        assert any("Code" in r or "code" in r.lower() for r in result)

    def test_snake_case_expansion(self):
        # The token must differ from query (query is added to seen set on init)
        result = _build_code_bm25_queries("search", frozenset(["split_python_ast"]))
        assert any("split" in r for r in result)

    def test_dotted_module_path(self):
        result = _build_code_bm25_queries("search", frozenset(["axon.loaders"]))
        assert any("loaders" in r for r in result)

    def test_short_tokens_skipped(self):
        result = _build_code_bm25_queries("ab", frozenset(["ab"]))
        assert result == []

    def test_raw_token_included(self):
        result = _build_code_bm25_queries("SomeClass", frozenset(["SomeClass"]))
        joined = " ".join(result)
        assert "some" in joined.lower() or "someclass" in joined.lower()

    def test_empty_tokens(self):
        result = _build_code_bm25_queries("query", frozenset())
        assert result == []


# ---------------------------------------------------------------------------
# _classify_retrieval_failure (lines 293-353)
# ---------------------------------------------------------------------------


class TestClassifyRetrievalFailure:
    def test_empty_results_returns_empty(self):
        assert _classify_retrieval_failure([], frozenset(["foo"])) == []

    def test_exact_symbol_missed(self):
        results = [{"metadata": {"symbol_name": "unrelated_fn"}, "text": "hello"}]
        labels = _classify_retrieval_failure(results, frozenset(["my_target_symbol"]))
        assert "exact_symbol_missed" in labels

    def test_no_symbol_miss_when_matched(self):
        results = [{"metadata": {"symbol_name": "build_graph"}, "text": "build graph code"}]
        labels = _classify_retrieval_failure(results, frozenset(["build_graph"]))
        assert "exact_symbol_missed" not in labels

    def test_right_file_wrong_block(self):
        results = [
            {
                "metadata": {
                    "source": "axon/main.py",
                    "symbol_name": "some_other_fn",
                },
                "text": "code",
            }
        ]
        labels = _classify_retrieval_failure(results, frozenset(["main"]), expected_symbol="main")
        assert "right_file_wrong_block" in labels

    def test_too_many_broad_chunks(self):
        results = [
            {"metadata": {}, "text": "broad chunk"},
            {"metadata": {}, "text": "another broad"},
        ]
        labels = _classify_retrieval_failure(results, frozenset(["something"]))
        assert "too_many_broad_chunks" in labels

    def test_fallback_chunk_involved(self):
        results = [{"metadata": {"is_fallback": True}, "text": "fallback"}]
        labels = _classify_retrieval_failure(results, frozenset(["query"]))
        assert "fallback_chunk_involved" in labels


# ---------------------------------------------------------------------------
# CodeRetrievalMixin._build_code_doc_bridge (line 365)
# ---------------------------------------------------------------------------


def _make_mixin():
    """Create a minimal CodeRetrievalMixin instance."""

    class FakeBrain(CodeRetrievalMixin, CodeRetrievalDiagnostics):
        def __init__(self):
            self._code_graph = {"nodes": {}, "edges": []}
            self.config = MagicMock()
            self.config.code_graph = True
            self.config.code_max_chunks_per_file = 3
            self.config.graph_rag_budget = 3
            self.config.top_k = 10
            self.config.symbol_index_engine = "python"
            self.config.rust_fallback_enabled = True
            self.vector_store = MagicMock()
            self.vector_store.get_by_ids.return_value = []
            self.bm25 = None

    return FakeBrain()


class TestBuildCodeDocBridge:
    def test_empty_nodes_returns_immediately(self):
        brain = _make_mixin()
        brain._build_code_doc_bridge([{"text": "some prose", "id": "p1", "metadata": {}}])
        # no error — early return when nodes is empty

    def test_adds_mentioned_in_edges(self):
        brain = _make_mixin()
        brain._code_graph = {
            "nodes": {
                "axon/main.py::build_graph": {
                    "name": "build_graph",
                    "chunk_ids": ["c1"],
                    "node_type": "function",
                }
            },
            "edges": [],
        }
        prose = [{"text": "build_graph is used here", "id": "prose1", "metadata": {}}]
        brain._build_code_doc_bridge(prose)
        edges = brain._code_graph["edges"]
        assert any(e["edge_type"] == "MENTIONED_IN" for e in edges)


# ---------------------------------------------------------------------------
# CodeRetrievalMixin._symbol_channel_search (lines 553-615)
# ---------------------------------------------------------------------------


class TestSymbolChannelSearch:
    def test_returns_empty_when_no_bm25(self):
        brain = _make_mixin()
        brain.bm25 = None
        result = brain._symbol_channel_search(frozenset(["foo"]), top_k=5)
        assert result == []

    def test_returns_empty_when_no_tokens(self):
        brain = _make_mixin()
        result = brain._symbol_channel_search(frozenset(), top_k=5)
        assert result == []

    def test_returns_empty_when_no_long_tokens(self):
        brain = _make_mixin()
        result = brain._symbol_channel_search(frozenset(["ab"]), top_k=5)
        assert result == []

    def test_exact_symbol_match(self):
        brain = _make_mixin()
        corpus = [
            {
                "id": "chunk1",
                "text": "def build_graph(): pass",
                "metadata": {"symbol_name": "build_graph", "qualified_name": ""},
            }
        ]
        mock_bm25 = MagicMock()
        mock_bm25.corpus = corpus
        del mock_bm25._retrievers  # ensure hasattr() returns False for single-corpus path
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(frozenset(["build_graph"]), top_k=5)
        assert len(result) == 1
        assert result[0]["score"] == 1.0
        assert result[0]["metadata"]["channel"] == "symbol_name"

    def test_partial_symbol_match(self):
        brain = _make_mixin()
        corpus = [
            {
                "id": "chunk2",
                "text": "def build_something(): pass",
                "metadata": {"symbol_name": "build_something", "qualified_name": ""},
            }
        ]
        mock_bm25 = MagicMock()
        mock_bm25.corpus = corpus
        del mock_bm25._retrievers
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(frozenset(["build"]), top_k=5)
        assert len(result) == 1
        assert result[0]["score"] == pytest.approx(0.6)

    def test_qualified_name_exact_match(self):
        brain = _make_mixin()
        corpus = [
            {
                "id": "chunk3",
                "text": "class Foo: pass",
                "metadata": {
                    "symbol_name": "",
                    "qualified_name": "axon.main.foomethod",
                },
            }
        ]
        mock_bm25 = MagicMock()
        mock_bm25.corpus = corpus
        del mock_bm25._retrievers
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(frozenset(["axon.main.foomethod"]), top_k=5)
        assert len(result) == 1
        assert result[0]["metadata"]["channel"] == "qualified_name"

    def test_multi_retriever_fanout(self):
        brain = _make_mixin()
        corpus = [
            {
                "id": "chunk4",
                "text": "def run_query(): pass",
                "metadata": {"symbol_name": "run_query", "qualified_name": ""},
            }
        ]
        sub_retriever = MagicMock()
        sub_retriever.corpus = corpus
        mock_bm25 = MagicMock()
        mock_bm25._retrievers = [sub_retriever]
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(frozenset(["run_query"]), top_k=5)
        assert len(result) == 1

    def test_rust_symbol_channel_is_used_when_enabled(self):
        brain = _make_mixin()
        brain.config.symbol_index_engine = "rust"
        corpus = [
            {
                "id": "chunk-rust",
                "text": "def run_query(): pass",
                "metadata": {"symbol_name": "run_query", "qualified_name": ""},
            }
        ]
        mock_bm25 = MagicMock()
        mock_bm25.corpus = corpus
        del mock_bm25._retrievers
        brain.bm25 = mock_bm25

        class _Bridge:
            def can_symbol_search(self):
                return True

            def symbol_channel_search(self, corpora, query_tokens, top_k, filters):
                assert len(corpora) == 1
                assert "run_query" in query_tokens
                return [{"index": 0, "score": 1.0, "channel": "symbol_name"}]

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("axon.rust_bridge.get_rust_bridge", lambda: _Bridge())
            result = brain._symbol_channel_search(frozenset(["run_query"]), top_k=5)

        assert len(result) == 1
        assert result[0]["id"] == "chunk-rust"
        assert result[0]["metadata"]["channel"] == "symbol_name"

    def test_filter_excludes_non_matching(self):
        brain = _make_mixin()
        corpus = [
            {
                "id": "chunk5",
                "text": "def filtered(): pass",
                "metadata": {
                    "symbol_name": "filtered",
                    "qualified_name": "",
                    "project": "other",
                },
            }
        ]
        mock_bm25 = MagicMock()
        mock_bm25.corpus = corpus
        del mock_bm25._retrievers
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(
            frozenset(["filtered"]), top_k=5, filters={"project": "mine"}
        )
        assert result == []

    def test_no_corpus_attr_returns_empty(self):
        brain = _make_mixin()
        mock_bm25 = MagicMock(spec=[])  # no corpus, no _retrievers
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(frozenset(["something"]), top_k=5)
        assert result == []


# ---------------------------------------------------------------------------
# CodeRetrievalMixin._expand_with_code_graph (lines 617-703)
# ---------------------------------------------------------------------------


class TestExpandWithCodeGraph:
    def test_empty_nodes_returns_unchanged(self):
        brain = _make_mixin()
        results = [{"id": "r1", "text": "hi", "score": 0.9, "metadata": {}}]
        out, names = brain._expand_with_code_graph("hello", results)
        assert out == results
        assert names == []

    def test_matched_node_adds_chunk_ids(self):
        brain = _make_mixin()
        brain._code_graph = {
            "nodes": {
                "axon/main.py": {
                    "name": "main",
                    "chunk_ids": ["extra_chunk"],
                    "node_type": "file",
                }
            },
            "edges": [],
        }
        brain.vector_store.get_by_ids.return_value = [
            {"id": "extra_chunk", "text": "extra content", "metadata": {}}
        ]
        cfg = MagicMock()
        cfg.graph_rag_budget = 5
        cfg.top_k = 10
        results = [{"id": "r1", "text": "some code in main", "score": 0.8, "metadata": {}}]
        out, names = brain._expand_with_code_graph("main", results, cfg=cfg)
        assert any(r["id"] == "extra_chunk" for r in out)

    def test_no_matched_nodes_returns_unchanged(self):
        brain = _make_mixin()
        brain._code_graph = {
            "nodes": {"some_file.py": {"name": "some_file", "chunk_ids": [], "node_type": "file"}},
            "edges": [],
        }
        results = [{"id": "r1", "text": "hello world", "score": 0.5, "metadata": {}}]
        out, names = brain._expand_with_code_graph("zzznomatch", results)
        assert out == results

    def test_vector_store_exception_is_ignored(self):
        brain = _make_mixin()
        brain._code_graph = {
            "nodes": {
                "axon/query.py": {
                    "name": "query",
                    "chunk_ids": ["qchunk"],
                    "node_type": "file",
                }
            },
            "edges": [],
        }
        brain.vector_store.get_by_ids.side_effect = RuntimeError("db error")
        results = [{"id": "r1", "text": "query function", "score": 0.7, "metadata": {}}]
        out, names = brain._expand_with_code_graph("query", results)
        # Should not raise; results unchanged
        assert out[0]["id"] == "r1"

    def test_outgoing_edges_traversed(self):
        brain = _make_mixin()
        brain._code_graph = {
            "nodes": {
                "file.py": {"name": "file", "chunk_ids": [], "node_type": "file"},
                "file.py::myfunc": {
                    "name": "myfunc",
                    "chunk_ids": ["func_chunk"],
                    "node_type": "function",
                },
            },
            "edges": [{"source": "file.py", "target": "file.py::myfunc", "edge_type": "CONTAINS"}],
        }
        brain.vector_store.get_by_ids.return_value = [
            {"id": "func_chunk", "text": "def myfunc(): pass", "metadata": {}}
        ]
        results = [{"id": "r1", "text": "file myfunc", "score": 0.6, "metadata": {}}]
        cfg = MagicMock()
        cfg.graph_rag_budget = 5
        cfg.top_k = 10
        out, _ = brain._expand_with_code_graph("file", results, cfg=cfg)
        assert any(r.get("_code_graph_expanded") for r in out)


"""Tests for axon.code_retrieval to reach ≥90% coverage."""


# ---------------------------------------------------------------------------
# _is_code_query (lines 80-92)
# ---------------------------------------------------------------------------


class TestIsCodeQueryV2:
    def test_camel_case_is_code(self):
        assert _is_code_query("QueryRouter") is True

    def test_snake_case_is_code(self):
        assert _is_code_query("split_python_ast") is True

    def test_code_extension_is_code(self):
        assert _is_code_query("main.py") is True
        assert _is_code_query("index.ts") is True

    def test_symbol_keyword(self):
        assert _is_code_query("def build_graph") is True

    def test_plain_prose_not_code(self):
        assert _is_code_query("what is the weather today") is False

    def test_short_tokens_not_code(self):
        assert _is_code_query("hi") is False


# ---------------------------------------------------------------------------
# _build_code_bm25_queries (lines 104-142)
# ---------------------------------------------------------------------------


class TestBuildCodeBm25QueriesV2:
    def test_camel_case_expansion(self):
        result = _build_code_bm25_queries("CodeAwareSplitter", frozenset(["CodeAwareSplitter"]))
        assert any("Code" in r or "code" in r.lower() for r in result)

    def test_snake_case_expansion(self):
        # The token must differ from query (query is added to seen set on init)
        result = _build_code_bm25_queries("search", frozenset(["split_python_ast"]))
        assert any("split" in r for r in result)

    def test_dotted_module_path(self):
        result = _build_code_bm25_queries("search", frozenset(["axon.loaders"]))
        assert any("loaders" in r for r in result)

    def test_short_tokens_skipped(self):
        result = _build_code_bm25_queries("ab", frozenset(["ab"]))
        assert result == []

    def test_raw_token_included(self):
        result = _build_code_bm25_queries("SomeClass", frozenset(["SomeClass"]))
        joined = " ".join(result)
        assert "some" in joined.lower() or "someclass" in joined.lower()

    def test_empty_tokens(self):
        result = _build_code_bm25_queries("query", frozenset())
        assert result == []


# ---------------------------------------------------------------------------
# _classify_retrieval_failure (lines 293-353)
# ---------------------------------------------------------------------------


class TestClassifyRetrievalFailureV2:
    def test_empty_results_returns_empty(self):
        assert _classify_retrieval_failure([], frozenset(["foo"])) == []

    def test_exact_symbol_missed(self):
        results = [{"metadata": {"symbol_name": "unrelated_fn"}, "text": "hello"}]
        labels = _classify_retrieval_failure(results, frozenset(["my_target_symbol"]))
        assert "exact_symbol_missed" in labels

    def test_no_symbol_miss_when_matched(self):
        results = [{"metadata": {"symbol_name": "build_graph"}, "text": "build graph code"}]
        labels = _classify_retrieval_failure(results, frozenset(["build_graph"]))
        assert "exact_symbol_missed" not in labels

    def test_right_file_wrong_block(self):
        results = [
            {
                "metadata": {
                    "source": "axon/main.py",
                    "symbol_name": "some_other_fn",
                },
                "text": "code",
            }
        ]
        labels = _classify_retrieval_failure(results, frozenset(["main"]), expected_symbol="main")
        assert "right_file_wrong_block" in labels

    def test_too_many_broad_chunks(self):
        results = [
            {"metadata": {}, "text": "broad chunk"},
            {"metadata": {}, "text": "another broad"},
        ]
        labels = _classify_retrieval_failure(results, frozenset(["something"]))
        assert "too_many_broad_chunks" in labels

    def test_fallback_chunk_involved(self):
        results = [{"metadata": {"is_fallback": True}, "text": "fallback"}]
        labels = _classify_retrieval_failure(results, frozenset(["query"]))
        assert "fallback_chunk_involved" in labels


# ---------------------------------------------------------------------------
# CodeRetrievalMixin._build_code_doc_bridge (line 365)
# ---------------------------------------------------------------------------


def _make_mixin():
    """Create a minimal CodeRetrievalMixin instance."""

    class FakeBrain(CodeRetrievalMixin, CodeRetrievalDiagnostics):
        def __init__(self):
            self._code_graph = {"nodes": {}, "edges": []}
            self.config = MagicMock()
            self.config.code_graph = True
            self.config.code_max_chunks_per_file = 3
            self.config.graph_rag_budget = 3
            self.config.top_k = 10
            self.vector_store = MagicMock()
            self.vector_store.get_by_ids.return_value = []
            self.bm25 = None

    return FakeBrain()


class TestBuildCodeDocBridgeV2:
    def test_empty_nodes_returns_immediately(self):
        brain = _make_mixin()
        brain._build_code_doc_bridge([{"text": "some prose", "id": "p1", "metadata": {}}])
        # no error — early return when nodes is empty

    def test_adds_mentioned_in_edges(self):
        brain = _make_mixin()
        brain._code_graph = {
            "nodes": {
                "axon/main.py::build_graph": {
                    "name": "build_graph",
                    "chunk_ids": ["c1"],
                    "node_type": "function",
                }
            },
            "edges": [],
        }
        prose = [{"text": "build_graph is used here", "id": "prose1", "metadata": {}}]
        brain._build_code_doc_bridge(prose)
        edges = brain._code_graph["edges"]
        assert any(e["edge_type"] == "MENTIONED_IN" for e in edges)


# ---------------------------------------------------------------------------
# CodeRetrievalMixin._symbol_channel_search (lines 553-615)
# ---------------------------------------------------------------------------


class TestSymbolChannelSearchV2:
    def test_returns_empty_when_no_bm25(self):
        brain = _make_mixin()
        brain.bm25 = None
        result = brain._symbol_channel_search(frozenset(["foo"]), top_k=5)
        assert result == []

    def test_returns_empty_when_no_tokens(self):
        brain = _make_mixin()
        result = brain._symbol_channel_search(frozenset(), top_k=5)
        assert result == []

    def test_returns_empty_when_no_long_tokens(self):
        brain = _make_mixin()
        result = brain._symbol_channel_search(frozenset(["ab"]), top_k=5)
        assert result == []

    def test_exact_symbol_match(self):
        brain = _make_mixin()
        corpus = [
            {
                "id": "chunk1",
                "text": "def build_graph(): pass",
                "metadata": {"symbol_name": "build_graph", "qualified_name": ""},
            }
        ]
        mock_bm25 = MagicMock()
        mock_bm25.corpus = corpus
        del mock_bm25._retrievers  # ensure hasattr() returns False for single-corpus path
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(frozenset(["build_graph"]), top_k=5)
        assert len(result) == 1
        assert result[0]["score"] == 1.0
        assert result[0]["metadata"]["channel"] == "symbol_name"

    def test_partial_symbol_match(self):
        brain = _make_mixin()
        corpus = [
            {
                "id": "chunk2",
                "text": "def build_something(): pass",
                "metadata": {"symbol_name": "build_something", "qualified_name": ""},
            }
        ]
        mock_bm25 = MagicMock()
        mock_bm25.corpus = corpus
        del mock_bm25._retrievers
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(frozenset(["build"]), top_k=5)
        assert len(result) == 1
        assert result[0]["score"] == pytest.approx(0.6)

    def test_qualified_name_exact_match(self):
        brain = _make_mixin()
        corpus = [
            {
                "id": "chunk3",
                "text": "class Foo: pass",
                "metadata": {
                    "symbol_name": "",
                    "qualified_name": "axon.main.foomethod",
                },
            }
        ]
        mock_bm25 = MagicMock()
        mock_bm25.corpus = corpus
        del mock_bm25._retrievers
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(frozenset(["axon.main.foomethod"]), top_k=5)
        assert len(result) == 1
        assert result[0]["metadata"]["channel"] == "qualified_name"

    def test_multi_retriever_fanout(self):
        brain = _make_mixin()
        corpus = [
            {
                "id": "chunk4",
                "text": "def run_query(): pass",
                "metadata": {"symbol_name": "run_query", "qualified_name": ""},
            }
        ]
        sub_retriever = MagicMock()
        sub_retriever.corpus = corpus
        mock_bm25 = MagicMock()
        mock_bm25._retrievers = [sub_retriever]
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(frozenset(["run_query"]), top_k=5)
        assert len(result) == 1

    def test_filter_excludes_non_matching(self):
        brain = _make_mixin()
        corpus = [
            {
                "id": "chunk5",
                "text": "def filtered(): pass",
                "metadata": {
                    "symbol_name": "filtered",
                    "qualified_name": "",
                    "project": "other",
                },
            }
        ]
        mock_bm25 = MagicMock()
        mock_bm25.corpus = corpus
        del mock_bm25._retrievers
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(
            frozenset(["filtered"]), top_k=5, filters={"project": "mine"}
        )
        assert result == []

    def test_no_corpus_attr_returns_empty(self):
        brain = _make_mixin()
        mock_bm25 = MagicMock(spec=[])  # no corpus, no _retrievers
        brain.bm25 = mock_bm25
        result = brain._symbol_channel_search(frozenset(["something"]), top_k=5)
        assert result == []


# ---------------------------------------------------------------------------
# CodeRetrievalMixin._expand_with_code_graph (lines 617-703)
# ---------------------------------------------------------------------------


class TestExpandWithCodeGraphV2:
    def test_empty_nodes_returns_unchanged(self):
        brain = _make_mixin()
        results = [{"id": "r1", "text": "hi", "score": 0.9, "metadata": {}}]
        out, names = brain._expand_with_code_graph("hello", results)
        assert out == results
        assert names == []

    def test_matched_node_adds_chunk_ids(self):
        brain = _make_mixin()
        brain._code_graph = {
            "nodes": {
                "axon/main.py": {
                    "name": "main",
                    "chunk_ids": ["extra_chunk"],
                    "node_type": "file",
                }
            },
            "edges": [],
        }
        brain.vector_store.get_by_ids.return_value = [
            {"id": "extra_chunk", "text": "extra content", "metadata": {}}
        ]
        cfg = MagicMock()
        cfg.graph_rag_budget = 5
        cfg.top_k = 10
        results = [{"id": "r1", "text": "some code in main", "score": 0.8, "metadata": {}}]
        out, names = brain._expand_with_code_graph("main", results, cfg=cfg)
        assert any(r["id"] == "extra_chunk" for r in out)

    def test_no_matched_nodes_returns_unchanged(self):
        brain = _make_mixin()
        brain._code_graph = {
            "nodes": {"some_file.py": {"name": "some_file", "chunk_ids": [], "node_type": "file"}},
            "edges": [],
        }
        results = [{"id": "r1", "text": "hello world", "score": 0.5, "metadata": {}}]
        out, names = brain._expand_with_code_graph("zzznomatch", results)
        assert out == results

    def test_vector_store_exception_is_ignored(self):
        brain = _make_mixin()
        brain._code_graph = {
            "nodes": {
                "axon/query.py": {
                    "name": "query",
                    "chunk_ids": ["qchunk"],
                    "node_type": "file",
                }
            },
            "edges": [],
        }
        brain.vector_store.get_by_ids.side_effect = RuntimeError("db error")
        results = [{"id": "r1", "text": "query function", "score": 0.7, "metadata": {}}]
        out, names = brain._expand_with_code_graph("query", results)
        # Should not raise; results unchanged
        assert out[0]["id"] == "r1"

    def test_outgoing_edges_traversed(self):
        brain = _make_mixin()
        brain._code_graph = {
            "nodes": {
                "file.py": {"name": "file", "chunk_ids": [], "node_type": "file"},
                "file.py::myfunc": {
                    "name": "myfunc",
                    "chunk_ids": ["func_chunk"],
                    "node_type": "function",
                },
            },
            "edges": [{"source": "file.py", "target": "file.py::myfunc", "edge_type": "CONTAINS"}],
        }
        brain.vector_store.get_by_ids.return_value = [
            {"id": "func_chunk", "text": "def myfunc(): pass", "metadata": {}}
        ]
        results = [{"id": "r1", "text": "file myfunc", "score": 0.6, "metadata": {}}]
        cfg = MagicMock()
        cfg.graph_rag_budget = 5
        cfg.top_k = 10
        out, _ = brain._expand_with_code_graph("file", results, cfg=cfg)
        assert any(r.get("_code_graph_expanded") for r in out)
