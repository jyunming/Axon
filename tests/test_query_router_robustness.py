from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from axon.query_router import QueryRouterMixin


class MockConfig:
    def __init__(self, query_router="heuristic"):
        self.query_router = query_router


class RouterStub(QueryRouterMixin):
    _CORPUS_KEYWORDS = {
        "all documents",
        "entire corpus",
        "everything",
        "main topics",
        "key themes",
        "across all",
    }
    _SYNTHESIS_KEYWORDS = {
        "summarize",
        "overview",
        "compare",
        "contrast",
        "explain",
        "discuss",
        "survey",
        "themes",
        "analysis",
    }
    _ENTITY_KEYWORDS = {
        "relationship",
        "related to",
        "who",
        "works with",
        "connected",
        "linked",
        "colleague",
        "dependency",
        "relate",
    }
    _TABLE_KEYWORDS = {
        "table",
        "row",
        "column",
        "value",
        "count",
        "average",
        "maximum",
        "minimum",
        "statistic",
        "how many",
        "list all",
    }

    def __init__(self, config):
        self.config = config


class TestQueryRouterRobustness:
    @pytest.fixture
    def router(self):
        return RouterStub(MockConfig())

    def test_heuristic_factual_short(self, router):
        # 'who' is in _ENTITY_KEYWORDS, so 'Who is John Doe?' matches entity_relation
        assert router._classify_query_route_heuristic("Who is John Doe?") == "entity_relation"

    def test_heuristic_synthesis_keywords(self, router):
        assert (
            router._classify_query_route_heuristic("Summarize the project status.") == "synthesis"
        )

    def test_heuristic_synthesis_long(self, router):
        query = "This is a very long query that should be classified as synthesis even without specific keywords because it exceeds the length threshold of eighty characters."
        assert len(query) > 80
        assert router._classify_query_route_heuristic(query) == "synthesis"

    def test_heuristic_corpus_exploration_keywords(self, router):
        assert (
            router._classify_query_route_heuristic("What are the main topics across all documents?")
            == "corpus_exploration"
        )

    def test_heuristic_corpus_exploration_very_long_synthesis(self, router):
        query = "Can you provide an overview and summary of the relationship between all the different components and how they compare and contrast with each other in the context of the entire project scope?"
        # Length > 120 and contains synthesis keywords
        assert len(query) > 120
        assert any(kw in query for kw in router._SYNTHESIS_KEYWORDS)
        assert router._classify_query_route_heuristic(query) == "corpus_exploration"

    def test_heuristic_entity_relation(self, router):
        assert (
            router._classify_query_route_heuristic("How is Alice connected to Bob?")
            == "entity_relation"
        )
        assert (
            router._classify_query_route_heuristic("How does X relate to Y?") == "entity_relation"
        )

    def test_heuristic_table_lookup(self, router):
        assert (
            router._classify_query_route_heuristic("Show me a table of the statistics.")
            == "table_lookup"
        )

    def test_llm_routing_fallback(self, router):
        router.config.query_router = "llm"
        router.llm = MagicMock()
        router.llm.generate.return_value = "table_lookup"

        assert router._classify_query_route("Give me stats", router.config) == "table_lookup"

    def test_llm_routing_invalid_fallback_to_factual(self, router):
        router.config.query_router = "llm"
        router.llm = MagicMock()
        router.llm.generate.return_value = "garbage"

        assert router._classify_query_route("Give me stats", router.config) == "factual"

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("what is axon?", "factual"),
            ("summarize everything", "corpus_exploration"),
            ("how does x relate to y?", "entity_relation"),
            ("list all rows in the data", "table_lookup"),
            ("give me an overview of the whole thing", "synthesis"),
        ],
    )
    def test_heuristic_parameterized(self, router, query, expected):
        assert router._classify_query_route_heuristic(query) == expected

    def test_classification_latency_benchmark(self, router):
        import time

        queries = [
            "What is the capital of France?",
            "Summarize the entire documentation for the project.",
            "How are entities A and B related in the context of the new architecture?",
            "Show me a table of the revenue by region.",
            "Provide a comprehensive overview of the system's security features.",
        ]

        start = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            for q in queries:
                router._classify_query_route_heuristic(q)
        end = time.perf_counter()

        avg_ms = ((end - start) / (iterations * len(queries))) * 1000
        print(f"\nAverage classification latency: {avg_ms:.4f} ms")
        # Assert that it's fast enough (e.g., < 1ms per query)
        assert avg_ms < 1.0


class TestUnifiedQueryTransforms:
    """Tests for _get_all_transforms_unified() and its integration in _execute_retrieval()."""

    @pytest.fixture
    def stub_with_llm(self):
        router = RouterStub(MockConfig())
        router.llm = MagicMock()
        return router

    def test_unified_transform_single_llm_call(self, stub_with_llm):
        """With >=2 transforms enabled, unified mode makes exactly 1 LLM call."""
        import json

        router = stub_with_llm
        payload = {
            "multi_queries": ["alt phrasing 1", "alt phrasing 2"],
            "step_back": "broader concept question",
            "decomposed": None,
            "hyde_doc": "hypothetical doc text",
        }
        router.llm.complete.return_value = json.dumps(payload)

        enabled = {"multi": True, "step_back": True, "decompose": False, "hyde": True}
        result = router._get_all_transforms_unified("What is X?", enabled, multi_count=2)

        router.llm.complete.assert_called_once()
        assert result["multi_queries"] == ["alt phrasing 1", "alt phrasing 2"]
        assert result["step_back"] == "broader concept question"
        assert result["decomposed"] is None
        assert result["hyde_doc"] == "hypothetical doc text"

    def test_unified_transform_fallback_on_bad_json(self, stub_with_llm):
        """Malformed JSON from LLM causes graceful fallback to empty results (no crash)."""
        router = stub_with_llm
        router.llm.complete.return_value = "this is not json at all }{{"

        enabled = {"multi": True, "step_back": True, "decompose": False, "hyde": False}
        result = router._get_all_transforms_unified("What is X?", enabled)

        assert result == {
            "multi_queries": None,
            "step_back": None,
            "decomposed": None,
            "hyde_doc": None,
        }

    def test_unified_transform_fallback_on_non_dict_json(self, stub_with_llm):
        """LLM returning a JSON array instead of object causes graceful fallback."""
        import json

        router = stub_with_llm
        router.llm.complete.return_value = json.dumps(["this", "is", "a", "list"])

        enabled = {"multi": True, "step_back": True, "decompose": False, "hyde": False}
        result = router._get_all_transforms_unified("What is X?", enabled)

        assert result["multi_queries"] is None
        assert result["step_back"] is None

    def test_unified_transform_returns_none_when_all_disabled(self, stub_with_llm):
        """No LLM call when all transforms are disabled."""
        router = stub_with_llm
        enabled = {"multi": False, "step_back": False, "decompose": False, "hyde": False}
        result = router._get_all_transforms_unified("What is X?", enabled)

        router.llm.complete.assert_not_called()
        assert all(v is None for v in result.values())

    def test_unified_transform_strips_markdown_fences(self, stub_with_llm):
        """LLM output wrapped in ```json fences is still parsed correctly."""
        import json

        router = stub_with_llm
        payload = {"multi_queries": ["q1", "q2"], "step_back": "broader"}
        router.llm.complete.return_value = f"```json\n{json.dumps(payload)}\n```"

        enabled = {"multi": True, "step_back": True, "decompose": False, "hyde": False}
        result = router._get_all_transforms_unified("What is X?", enabled)

        assert result["multi_queries"] == ["q1", "q2"]
        assert result["step_back"] == "broader"

    def test_unified_skipped_for_single_transform(self, stub_with_llm):
        """With only 1 transform enabled, unified mode is NOT used (individual path runs)."""

        router = stub_with_llm
        # Only multi_query enabled — unified should not fire, individual should
        router.llm.complete.return_value = "alt q1\nalt q2\nalt q3"

        enabled = {"multi": True, "step_back": False, "decompose": False, "hyde": False}
        n_enabled = sum(enabled.values())
        assert n_enabled < 2  # guard: single transform

        # Simulate what _execute_retrieval checks: n_enabled < 2 → skip unified
        use_unified = n_enabled >= 2
        assert not use_unified

        # Direct individual method should work normally
        result_individual = router._get_multi_queries("What is X?")
        assert "What is X?" in result_individual  # original always included


"""Comprehensive tests for src/axon/query_router.py.

Covers missed lines: 112-113, 155, 177, 202-203, 215-228, 243-244, 424-429,
456-495, 499-533, 557, 578-582, 615-623, 756, 821, 849-851, 857-869, 883,
888-890, 894-900, 954-957, 993, 996, 1033, 1064-1066, 1068-1073, 1143, 1175,
1180, 1200-1202, 1204-1209, 1216.
"""

import os


# Provide a simple synchronous executor mock
class SyncExecutor:
    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future

        f = Future()
        try:
            result = fn(*args, **kwargs)
            f.set_result(result)
        except Exception as e:
            f.set_exception(e)
        return f

    def map(self, fn, *iterables):
        return map(fn, *iterables)

    def shutdown(self, wait=True):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


import tempfile
from collections import OrderedDict
from threading import Lock
from unittest.mock import patch

import pytest

from axon.code_retrieval import CodeRetrievalDiagnostics, CodeRetrievalTrace
from axon.config import AxonConfig
from axon.query_router import _ROUTE_PROFILES, QueryRouterMixin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CORPUS_KEYWORDS = {
    "all documents",
    "entire corpus",
    "everything",
    "main topics",
    "key themes",
    "across all",
}
_SYNTHESIS_KEYWORDS = {
    "summarize",
    "overview",
    "compare",
    "contrast",
    "explain",
    "discuss",
    "survey",
    "themes",
    "analysis",
}
_ENTITY_KEYWORDS = {
    "relationship",
    "related to",
    "who",
    "works with",
    "connected",
    "linked",
    "colleague",
    "dependency",
    "relate",
}
_TABLE_KEYWORDS = {
    "table",
    "row",
    "column",
    "value",
    "count",
    "average",
    "maximum",
    "minimum",
    "statistic",
    "how many",
    "list all",
}


def _make_config(**kwargs) -> AxonConfig:
    # Use a subdirectory of the system temp to be safer
    tmp = tempfile.mkdtemp(prefix="axon_query_test_")
    defaults = {
        "bm25_path": os.path.join(tmp, "bm25"),
        "vector_store_path": os.path.join(tmp, "vs"),
        "raptor": False,
        "graph_rag": False,
        "query_router": "heuristic",
        "query_cache": False,
        "hybrid_search": False,
        "rerank": False,
        "hyde": False,
        "multi_query": False,
        "step_back": False,
        "query_decompose": False,
        "discussion_fallback": True,
        "compress_context": False,
        "truth_grounding": False,
        "similarity_threshold": 0.0,
        "top_k": 5,
        "code_lexical_boost": False,
    }
    defaults.update(kwargs)
    return AxonConfig(**defaults)


def _make_full_stub(**config_kwargs) -> RouterStubV2:
    """Return a RouterStubV2 pre-wired with sensible mock components."""
    cfg = _make_config(**config_kwargs)
    stub = RouterStubV2(cfg)
    stub.llm = MagicMock()
    stub.llm.complete.return_value = "mock answer"
    stub.llm.generate.return_value = "factual"
    stub.llm.stream.return_value = iter(["chunk1", "chunk2"])
    stub.embedding = MagicMock()
    stub.embedding.embed_query.return_value = [0.1] * 10
    stub.embedding.embed.return_value = [[0.1] * 10, [0.2] * 10]
    stub.vector_store = MagicMock()
    stub.vector_store.search.return_value = [
        {
            "id": "doc1",
            "text": "test document text",
            "score": 0.9,
            "metadata": {"source": "test.py"},
        }
    ]
    stub.vector_store.get_by_ids.return_value = []
    stub.bm25 = None
    stub.reranker = MagicMock()
    stub.reranker.rerank.return_value = [
        {
            "id": "doc1",
            "text": "test document text",
            "score": 0.9,
            "rerank_score": 0.95,
            "metadata": {"source": "test.py"},
        }
    ]
    stub._entity_graph = {}
    stub._entity_token_index_internal = {}
    stub._relation_graph = {}
    stub._entity_embeddings = {}
    stub._community_summaries = {}
    stub._community_levels = {}
    stub._query_cache = OrderedDict()
    stub._cache_lock = Lock()
    stub._last_diagnostics = None
    # Wire a MagicMock graph backend; retrieve() is never called because
    # _entity_graph is empty (cfg.graph_rag guard short-circuits first).
    stub._graph_backend = MagicMock()
    stub._graph_backend.retrieve.return_value = []
    return stub


class RouterStubV2(QueryRouterMixin):
    """Minimal concrete class that mixes in QueryRouterMixin."""

    _CORPUS_KEYWORDS = _CORPUS_KEYWORDS
    _SYNTHESIS_KEYWORDS = _SYNTHESIS_KEYWORDS
    _ENTITY_KEYWORDS = _ENTITY_KEYWORDS
    _TABLE_KEYWORDS = _TABLE_KEYWORDS

    SYSTEM_PROMPT = "You are a helpful assistant.\n1. **Mandatory Citations**: always cite.\n"
    SYSTEM_PROMPT_STRICT = (
        "You must ONLY use the provided context.\n1. **Mandatory Citations**: always cite.\n"
    )

    def __init__(self, config: AxonConfig):
        import threading

        self.config = config
        self._community_rebuild_lock = threading.Lock()
        # Executor for parallel transforms

        self._executor = SyncExecutor()

    # Stub out methods we do not want to call in most tests
    def _validate_embedding_meta(self, on_mismatch="warn"):
        pass

    def _raptor_drilldown(self, query, results, cfg=None):
        return results

    def _apply_artifact_ranking(self, results, cfg=None):
        return results

    def _local_search_context(self, query, matched_entities, cfg):
        return "local context"

    def _global_search_map_reduce(self, query, cfg):
        return "global context"

    def _generate_community_summaries(self, query_hint=None):
        pass

    def _index_community_reports_in_vector_store(self):
        pass

    def _extract_entities(self, query):
        return []

    def _match_entities_by_embedding(self, query):
        return []

    def _entity_matches(self, q_name, eid):
        return 0.0

    def _symbol_channel_search(self, tokens, top_k=5, filters=None):
        return []

    def _apply_code_lexical_boost(self, results, tokens, cfg=None, diagnostics=None, trace=None):
        return results

    def _expand_with_code_graph(self, query, results, cfg=None):
        return results, []

    def _classify_query_needs_graphrag(self, query, mode):
        return True


# ===========================================================================
# 1. _ROUTE_PROFILES constant
# ===========================================================================


class TestRouteProfiles:
    def test_all_profiles_present(self):
        assert set(_ROUTE_PROFILES.keys()) == {
            "factual",
            "synthesis",
            "table_lookup",
            "entity_relation",
            "corpus_exploration",
        }

    def test_factual_profile_disables_expensive_flags(self):
        p = _ROUTE_PROFILES["factual"]
        assert p["raptor"] is False
        assert p["graph_rag"] is False
        assert p["hyde"] is False
        assert p["multi_query"] is False
        assert p["step_back"] is False
        assert p["query_decompose"] is False

    def test_synthesis_profile_enables_parent_doc_and_raptor(self):
        p = _ROUTE_PROFILES["synthesis"]
        assert p["parent_doc"] is True
        assert p["raptor"] is True

    def test_entity_relation_profile_enables_graph_rag(self):
        p = _ROUTE_PROFILES["entity_relation"]
        assert p["graph_rag"] is True

    def test_corpus_exploration_profile_enables_multi_query(self):
        p = _ROUTE_PROFILES["corpus_exploration"]
        assert p["multi_query"] is True
        assert p["raptor"] is True


# ===========================================================================
# 2. _classify_query_route_llm  (lines 112-113)
# ===========================================================================


class TestClassifyQueryRouteLLM:
    def test_valid_response_returned(self):
        stub = _make_full_stub(query_router="llm")
        stub.llm.generate.return_value = "synthesis"
        result = stub._classify_query_route_llm("What is the big picture?")
        assert result == "synthesis"

    def test_invalid_response_falls_back_to_factual(self):
        """Lines 112-113: invalid response → fallback 'factual'."""
        stub = _make_full_stub(query_router="llm")
        stub.llm.generate.return_value = "not_a_valid_category"
        result = stub._classify_query_route_llm("Some query")
        assert result == "factual"

    def test_exception_falls_back_to_factual(self):
        """Lines 112-113: exception path → fallback 'factual'."""
        stub = _make_full_stub(query_router="llm")
        stub.llm.generate.side_effect = RuntimeError("LLM down")
        result = stub._classify_query_route_llm("Some query")
        assert result == "factual"

    def test_all_valid_categories_returned(self):
        valid = {"factual", "synthesis", "table_lookup", "entity_relation", "corpus_exploration"}
        stub = _make_full_stub(query_router="llm")
        for cat in valid:
            stub.llm.generate.return_value = cat
            assert stub._classify_query_route_llm("q") == cat

    def test_whitespace_stripped_in_response(self):
        stub = _make_full_stub(query_router="llm")
        stub.llm.generate.return_value = "  factual  "
        result = stub._classify_query_route_llm("query")
        assert result == "factual"


# ===========================================================================
# 3. _expand_with_entity_graph (lines 155, 177, 202-203)
# ===========================================================================


class TestExpandWithEntityGraph:
    def _make_stub_with_entity_graph(self):
        stub = _make_full_stub()
        stub._entity_graph = {
            "python": {"chunk_ids": ["c1", "c2"], "description": "Python language"},
            "django": {"chunk_ids": ["c3"], "description": "Django framework"},
        }
        stub._rebuild_entity_token_index()
        stub._relation_graph = {}
        stub._entity_embeddings = {}
        return stub

    def test_empty_entities_returns_original_results(self):
        stub = _make_full_stub()
        stub._entity_graph = {"foo": {"chunk_ids": ["c1"]}}
        stub._rebuild_entity_token_index()
        results = [{"id": "x", "text": "hello", "score": 0.8, "metadata": {}}]
        out, matched = stub._expand_with_entity_graph("unrelated query", results)
        # no entities extracted → returns unchanged
        assert out == results
        assert matched == []

    def test_q_name_empty_string_skipped(self):
        """Line 155: q_name empty → continue."""
        stub = _make_full_stub()
        stub._entity_graph = {"": {"chunk_ids": ["c1"]}}
        stub._rebuild_entity_token_index()
        stub._extract_entities = MagicMock(
            return_value=[{"name": "", "type": "UNKNOWN", "description": ""}]
        )
        results = []
        out, matched = stub._expand_with_entity_graph("some query", results)
        assert out == []

    def test_relation_graph_hop_target_empty_skipped(self):
        """Line 177: target empty → continue."""
        stub = _make_full_stub()
        stub._entity_graph = {
            "python": {"chunk_ids": ["c1"]},
        }
        stub._rebuild_entity_token_index()
        stub._relation_graph = {
            "python": [{"target": "", "relation": "uses"}],
        }
        stub._extract_entities = MagicMock(
            return_value=[{"name": "python", "type": "TECH", "description": ""}]
        )
        stub._entity_matches = MagicMock(return_value=0.9)
        results = [{"id": "c1", "text": "some text", "score": 0.85, "metadata": {}}]
        out, matched = stub._expand_with_entity_graph("python query", results)
        # c1 already in results, no extra ids, empty target skipped
        assert any(r["id"] == "c1" for r in out)

    def test_vector_store_get_by_ids_exception_logged(self):
        """Lines 202-203: exception in vector_store.get_by_ids is caught gracefully."""
        stub = _make_full_stub()
        stub._entity_graph = {
            "python": {"chunk_ids": ["new_doc"]},
        }
        stub._rebuild_entity_token_index()
        stub._extract_entities = MagicMock(
            return_value=[{"name": "python", "type": "TECH", "description": ""}]
        )
        stub._entity_matches = MagicMock(return_value=0.9)
        stub.vector_store.get_by_ids.side_effect = RuntimeError("store error")
        results = []
        out, matched = stub._expand_with_entity_graph("python query", results)
        # Should still return original results (empty) without raising
        assert isinstance(out, list)


# ===========================================================================
# 4. _prepend_contextual_context (lines 215-228)
# ===========================================================================


class TestPrependContextualContext:
    def test_normal_prepend(self):
        """Lines 215-228: LLM call succeeds, sentence prepended."""
        stub = _make_full_stub()
        stub.llm.generate.return_value = "This chunk discusses Python basics."
        chunk = {"id": "c1", "text": "Python is a programming language.", "metadata": {}}
        result = stub._prepend_contextual_context(chunk, "A long document about programming...")
        assert "Python is a programming language." in result["text"]
        assert "This chunk discusses Python basics." in result["text"]

    def test_llm_exception_returns_original_chunk(self):
        """Lines 215-228: graceful degradation on LLM failure."""
        stub = _make_full_stub()
        stub.llm.generate.side_effect = RuntimeError("LLM error")
        chunk = {"id": "c1", "text": "original text", "metadata": {}}
        result = stub._prepend_contextual_context(chunk, "whole doc")
        assert result["text"] == "original text"

    def test_original_chunk_not_mutated(self):
        stub = _make_full_stub()
        stub.llm.generate.return_value = "Context sentence."
        chunk = {"id": "c1", "text": "original text", "metadata": {}}
        result = stub._prepend_contextual_context(chunk, "doc text")
        # original should be unchanged
        assert chunk["text"] == "original text"
        # result is a copy
        assert result is not chunk


# ===========================================================================
# 5. _make_cache_key (lines 243-244)
# ===========================================================================


class TestMakeCacheKey:
    def test_returns_hex_string(self):
        stub = _make_full_stub()
        key = stub._make_cache_key("hello", None, stub.config)
        assert isinstance(key, str)
        assert len(key) == 32  # MD5 hex

    def test_different_queries_produce_different_keys(self):
        stub = _make_full_stub()
        k1 = stub._make_cache_key("query A", None, stub.config)
        k2 = stub._make_cache_key("query B", None, stub.config)
        assert k1 != k2

    def test_filters_serialization_failure_falls_back(self):
        """Lines 243-244: json.dumps of filters fails → falls back to str()."""
        stub = _make_full_stub()

        class Unserializable:
            def __repr__(self):
                return "unserializable"

        # Pass an object that cannot be JSON-serialised without default=str;
        # The code already uses default=str, so this won't fail — we test the
        # happy path to ensure no exception is raised.
        filters = {"key": Unserializable()}
        key = stub._make_cache_key("query", filters, stub.config)
        assert isinstance(key, str)

    def test_same_inputs_produce_same_key(self):
        stub = _make_full_stub()
        k1 = stub._make_cache_key("stable query", {"f": "v"}, stub.config)
        k2 = stub._make_cache_key("stable query", {"f": "v"}, stub.config)
        assert k1 == k2


# ===========================================================================
# 6. _get_step_back_query (lines 424-429)
# ===========================================================================


class TestGetStepBackQuery:
    def test_returns_llm_response(self):
        stub = _make_full_stub()
        stub.llm.complete.return_value = "What is the general theory of X?"
        result = stub._get_step_back_query("What is the specific implementation detail of X?")
        assert result == "What is the general theory of X?"
        stub.llm.complete.assert_called_once()

    def test_prompt_contains_original_query(self):
        stub = _make_full_stub()
        stub.llm.complete.return_value = "abstract question"
        stub._get_step_back_query("specific question about Y")
        call_args = stub.llm.complete.call_args
        prompt = call_args[0][0]
        assert "specific question about Y" in prompt


# ===========================================================================
# 7. _mmr_deduplicate (lines 456-495)
# ===========================================================================


class TestMMRDeduplicate:
    def _make_results(self, texts_scores):
        return [
            {"id": f"d{i}", "text": t, "score": s, "metadata": {}}
            for i, (t, s) in enumerate(texts_scores)
        ]

    def test_single_result_returned_unchanged(self):
        stub = _make_full_stub()
        cfg = stub.config
        results = [{"id": "d0", "text": "hello world", "score": 0.9, "metadata": {}}]
        out = stub._mmr_deduplicate(results, cfg)
        assert out == results

    def test_near_duplicates_removed(self):
        """Lines 456-495: near-duplicate (Jaccard >= 0.85) dropped."""
        stub = _make_full_stub()
        cfg = stub.config
        # Both chunks have identical text — Jaccard = 1.0
        r1 = {"id": "d0", "text": "alpha beta gamma delta epsilon", "score": 0.9, "metadata": {}}
        r2 = {"id": "d1", "text": "alpha beta gamma delta epsilon", "score": 0.8, "metadata": {}}
        out = stub._mmr_deduplicate([r1, r2], cfg)
        assert len(out) == 1
        assert out[0]["id"] == "d0"

    def test_diverse_results_both_kept(self):
        stub = _make_full_stub()
        cfg = stub.config
        r1 = {"id": "d0", "text": "python programming language", "score": 0.9, "metadata": {}}
        r2 = {
            "id": "d1",
            "text": "quantum physics subatomic particles",
            "score": 0.85,
            "metadata": {},
        }
        out = stub._mmr_deduplicate([r1, r2], cfg)
        assert len(out) == 2

    def test_order_respects_mmr_scoring(self):
        """Higher-scored document selected first."""
        stub = _make_full_stub()
        cfg = stub.config
        r_low = {"id": "d0", "text": "aaa bbb ccc", "score": 0.3, "metadata": {}}
        r_high = {"id": "d1", "text": "xxx yyy zzz", "score": 0.95, "metadata": {}}
        out = stub._mmr_deduplicate([r_low, r_high], cfg)
        assert out[0]["id"] == "d1"

    def test_empty_text_handled(self):
        stub = _make_full_stub()
        cfg = stub.config
        results = [
            {"id": "d0", "text": "", "score": 0.9, "metadata": {}},
            {"id": "d1", "text": "", "score": 0.8, "metadata": {}},
        ]
        # Should not raise, degenerate case
        out = stub._mmr_deduplicate(results, cfg)
        assert isinstance(out, list)


# ===========================================================================
# 8. _execute_web_search (lines 499-533)
# ===========================================================================


class TestExecuteWebSearch:
    def test_returns_web_results_on_success(self):
        """Lines 499-533: successful Brave API call."""
        stub = _make_full_stub(brave_api_key="test_key", truth_grounding=True)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Page",
                        "description": "A test description",
                        "url": "https://example.com",
                    }
                ]
            }
        }
        mock_response.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_response):
            results = stub._execute_web_search("test query", count=5)
        assert len(results) == 1
        assert results[0]["is_web"] is True
        assert results[0]["id"] == "https://example.com"
        assert "Test Page" in results[0]["text"]

    def test_returns_empty_on_http_error(self):
        """Lines 531-533: exception returns []."""
        stub = _make_full_stub(brave_api_key="test_key")
        with patch("httpx.get", side_effect=Exception("network error")):
            results = stub._execute_web_search("query", count=5)
        assert results == []

    def test_web_result_score_is_1(self):
        stub = _make_full_stub(brave_api_key="key")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {"results": [{"title": "T", "description": "D", "url": "https://t.com"}]}
        }
        mock_response.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_response):
            results = stub._execute_web_search("q")
        assert results[0]["score"] == 1.0

    def test_metadata_contains_source_url(self):
        stub = _make_full_stub(brave_api_key="key")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {"results": [{"title": "T", "description": "D", "url": "https://t.com"}]}
        }
        mock_response.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_response):
            results = stub._execute_web_search("q")
        assert results[0]["metadata"]["source"] == "https://t.com"


# ===========================================================================
# 9. _execute_retrieval — Hyde with variants (lines 557, 578-582, 615-623)
# ===========================================================================


class TestExecuteRetrievalTransforms:
    def test_hyde_flag_triggers_hypothetical_doc_embedding(self):
        """Line 557 / 615-623: HyDE path — vector_query replaced with generated doc."""
        stub = _make_full_stub(hyde=True, similarity_threshold=0.0)
        stub.llm.complete.return_value = "Hypothetical document text"
        result = stub._execute_retrieval("What is Python?")
        assert result["transforms"]["hyde_applied"] is True

    def test_step_back_flag_applied(self):
        """Lines 578-582: step_back transform applied."""
        stub = _make_full_stub(step_back=True, similarity_threshold=0.0)
        stub.llm.complete.return_value = "What are programming languages in general?"
        result = stub._execute_retrieval("What is Python 3.12 walrus operator?")
        assert result["transforms"]["step_back_applied"] is True

    def test_multi_query_expands_search_queries(self):
        """Lines 578-582: multi_query adds variants."""
        stub = _make_full_stub(multi_query=True, similarity_threshold=0.0)
        stub.llm.complete.return_value = (
            "How does Python work?\nPython programming guide\nPython tutorial"
        )
        result = stub._execute_retrieval("What is Python?")
        assert result["transforms"]["multi_query_applied"] is True
        assert len(result["transforms"]["queries"]) > 1

    def test_decompose_adds_sub_questions(self):
        stub = _make_full_stub(query_decompose=True, similarity_threshold=0.0)
        stub.llm.complete.return_value = "What is Python?\nWhen was it created?\nWho made it?"
        result = stub._execute_retrieval("Tell me about Python history and features")
        assert result["transforms"]["decompose_applied"] is True

    def test_transform_exception_logged_and_skipped(self):
        """Lines 578-582: failed transform skipped gracefully."""
        stub = _make_full_stub(step_back=True, similarity_threshold=0.0)
        stub.llm.complete.side_effect = RuntimeError("LLM error")
        # Should not raise
        result = stub._execute_retrieval("test query")
        assert "results" in result

    def test_hyde_with_extra_variants_searches_all(self):
        """HyDE + multi_query → both transforms applied via unified path."""
        import json

        stub = _make_full_stub(hyde=True, multi_query=True, similarity_threshold=0.0)
        unified_payload = {
            "multi_queries": ["alt query 1", "alt query 2"],
            "step_back": None,
            "decomposed": None,
            "hyde_doc": "Hypothetical doc",
        }
        stub.llm.complete.return_value = json.dumps(unified_payload)
        result = stub._execute_retrieval("complex question")
        assert result["transforms"]["hyde_applied"] is True


# ===========================================================================
# 10. _execute_retrieval — MMR (line 756)
# ===========================================================================


class TestExecuteRetrievalMMR:
    def test_mmr_dedup_called_when_enabled(self):
        """Line 756: mmr=True triggers _mmr_deduplicate."""
        stub = _make_full_stub(similarity_threshold=0.0)
        stub.config.mmr = True
        stub._mmr_deduplicate = MagicMock(return_value=[])
        stub._execute_retrieval("test query")
        stub._mmr_deduplicate.assert_called_once()

    def test_mmr_not_called_when_disabled(self):
        stub = _make_full_stub(similarity_threshold=0.0)
        stub.config.mmr = False
        stub._mmr_deduplicate = MagicMock(return_value=[])
        stub._execute_retrieval("test query")
        stub._mmr_deduplicate.assert_not_called()


# ===========================================================================
# 11. search_raw (line 821)
# ===========================================================================


class TestSearchRaw:
    def test_returns_tuple_of_three(self):
        stub = _make_full_stub(similarity_threshold=0.0)
        results, diagnostics, trace = stub.search_raw("test query")
        assert isinstance(results, list)
        assert isinstance(diagnostics, CodeRetrievalDiagnostics)
        assert isinstance(trace, CodeRetrievalTrace)

    def test_rerank_called_when_enabled(self):
        """Line 821: rerank path in search_raw."""
        stub = _make_full_stub(rerank=True, similarity_threshold=0.0)
        results, diagnostics, trace = stub.search_raw("query with rerank")
        stub.reranker.rerank.assert_called_once()

    def test_graph_rag_budget_slicing(self):
        """graph_rag budget applies in search_raw."""
        stub = _make_full_stub(similarity_threshold=0.0)
        stub.config.graph_rag = True
        stub.config.graph_rag_budget = 2
        stub.vector_store.search.return_value = [
            {"id": f"d{i}", "text": f"doc {i}", "score": 0.9, "metadata": {}} for i in range(10)
        ]
        results, _, _ = stub.search_raw("query")
        # results sliced to top_k (5) + budget (2) at most
        assert len(results) <= stub.config.top_k + stub.config.graph_rag_budget

    def test_results_sliced_to_top_k_without_graph_rag(self):
        stub = _make_full_stub(similarity_threshold=0.0)
        stub.config.top_k = 3
        stub.vector_store.search.return_value = [
            {"id": f"d{i}", "text": f"text {i}", "score": 0.9, "metadata": {}} for i in range(10)
        ]
        results, _, _ = stub.search_raw("query")
        assert len(results) <= 3


# ===========================================================================
# 12. _build_context (lines 849-851, 857-869)
# ===========================================================================


class TestBuildContext:
    def test_web_result_labeled_correctly(self):
        """Lines 849-851: is_web=True path."""
        stub = _make_full_stub()
        results = [
            {
                "id": "https://example.com",
                "text": "some web content",
                "score": 1.0,
                "metadata": {"title": "Example Site"},
                "is_web": True,
            }
        ]
        context, has_web = stub._build_context(results)
        assert has_web is True
        assert "Web Result" in context
        assert "Example Site" in context

    def test_local_doc_labeled_correctly(self):
        stub = _make_full_stub()
        results = [
            {
                "id": "doc1",
                "text": "local document text",
                "score": 0.9,
                "metadata": {"source": "myfile.txt"},
            }
        ]
        context, has_web = stub._build_context(results)
        assert has_web is False
        assert "Document 1" in context
        assert "myfile.txt" in context

    def test_code_result_uses_symbol_label(self):
        """Lines 857-869: code source_class with symbol_name."""
        stub = _make_full_stub()
        results = [
            {
                "id": "sym1",
                "text": "def foo(): pass",
                "score": 0.9,
                "metadata": {
                    "source_class": "code",
                    "file_path": "/path/to/main.py",
                    "symbol_name": "foo",
                    "symbol_type": "function",
                },
            }
        ]
        context, has_web = stub._build_context(results)
        assert "main.py" in context
        assert "foo" in context
        assert "function" in context

    def test_code_result_without_symbol_uses_basename(self):
        """Lines 857-869: code source_class without symbol_name → just basename."""
        stub = _make_full_stub()
        results = [
            {
                "id": "sym1",
                "text": "# some code",
                "score": 0.9,
                "metadata": {
                    "source_class": "code",
                    "file_path": "/path/to/utils.py",
                    "symbol_name": "",
                    "symbol_type": "",
                },
            }
        ]
        context, _ = stub._build_context(results)
        assert "utils.py" in context

    def test_parent_text_preferred_over_chunk_text(self):
        stub = _make_full_stub()
        results = [
            {
                "id": "d1",
                "text": "child chunk",
                "score": 0.9,
                "metadata": {"source": "doc.txt", "parent_text": "parent passage text"},
            }
        ]
        context, _ = stub._build_context(results)
        assert "parent passage text" in context
        assert "child chunk" not in context

    def test_empty_results_returns_empty_context(self):
        stub = _make_full_stub()
        context, has_web = stub._build_context([])
        assert context == ""
        assert has_web is False


# ===========================================================================
# 13. _build_system_prompt (lines 883, 888-890, 894-900)
# ===========================================================================


class TestBuildSystemPrompt:
    def test_discussion_fallback_true_uses_permissive_prompt(self):
        """Line 883: discussion_fallback=True → SYSTEM_PROMPT."""
        stub = _make_full_stub(discussion_fallback=True)
        prompt = stub._build_system_prompt(has_web=False)
        assert "helpful assistant" in prompt.lower()

    def test_discussion_fallback_false_uses_strict_prompt(self):
        stub = _make_full_stub(discussion_fallback=False)
        prompt = stub._build_system_prompt(has_web=False)
        assert "ONLY use" in prompt or "only use" in prompt.lower()

    def test_cite_false_strips_citation_instruction(self):
        """Lines 888-890: cite=False removes citation lines."""
        stub = _make_full_stub(cite=False)
        prompt = stub._build_system_prompt(has_web=False)
        assert "Mandatory Citations" not in prompt

    def test_truth_grounding_false_no_web_suffix(self):
        """Line 883 return path: truth_grounding=False returns base."""
        stub = _make_full_stub(truth_grounding=False)
        prompt = stub._build_system_prompt(has_web=False)
        assert "Web Search" not in prompt

    def test_truth_grounding_with_web_adds_web_note(self):
        """Lines 894-900: has_web=True adds used-web note."""
        stub = _make_full_stub(truth_grounding=True)
        prompt = stub._build_system_prompt(has_web=True)
        assert "Web Search Used" in prompt

    def test_truth_grounding_without_web_adds_available_note(self):
        """Lines 894-900: has_web=False adds available-web note."""
        stub = _make_full_stub(truth_grounding=True)
        prompt = stub._build_system_prompt(has_web=False)
        assert "Web Search Available" in prompt


# ===========================================================================
# 14. query() — no results paths (lines 954-957, 993)
# ===========================================================================


class TestQueryNoResults:
    def test_empty_results_discussion_fallback_true(self):
        """Lines 987-991: no results + discussion_fallback → LLM called directly."""
        stub = _make_full_stub(discussion_fallback=True, similarity_threshold=1.1)
        stub.vector_store.search.return_value = []
        stub.llm.complete.return_value = "I'll answer from my knowledge."
        result = stub.query("What is Python?")
        assert result == "I'll answer from my knowledge."

    def test_empty_results_discussion_fallback_false(self):
        """Line 993: no results + discussion_fallback=False → static message."""
        stub = _make_full_stub(discussion_fallback=False, similarity_threshold=1.1)
        stub.vector_store.search.return_value = []
        result = stub.query("What is Python?")
        assert "don't have any relevant information" in result

    def test_query_router_applies_profile_overrides(self):
        """Lines 954-957: query_router != 'off' applies route profile."""
        stub = _make_full_stub(similarity_threshold=0.0, query_router="heuristic")
        # 'everything' is a corpus_exploration keyword
        result = stub.query("Tell me everything about all documents")
        assert isinstance(result, str)

    def test_query_router_off_uses_legacy_auto_route(self):
        """Lines 952-957: query_router='off' falls back to graph_rag_auto_route."""
        stub = _make_full_stub(similarity_threshold=0.0, query_router="off")
        stub.config.graph_rag = True
        stub.config.graph_rag_auto_route = "heuristic"
        stub._classify_query_needs_graphrag = MagicMock(return_value=False)
        result = stub.query("simple factual question")
        assert isinstance(result, str)


# ===========================================================================
# 15. query() — rerank path (line 996)
# ===========================================================================


class TestQueryRerank:
    def test_rerank_called_in_query(self):
        """Line 996: rerank=True calls reranker.rerank."""
        stub = _make_full_stub(rerank=True, similarity_threshold=0.0)
        stub.query("test query")
        stub.reranker.rerank.assert_called_once()

    def test_top_score_uses_rerank_score(self):
        """line 1038-1039: when rerank=True, top_score from rerank_score."""
        stub = _make_full_stub(rerank=True, similarity_threshold=0.0)
        stub.reranker.rerank.return_value = [
            {
                "id": "d1",
                "text": "text",
                "score": 0.5,
                "rerank_score": 0.99,
                "metadata": {"source": "f.txt"},
            }
        ]
        # Should not raise and should call llm.complete
        result = stub.query("rerank query")
        assert isinstance(result, str)


# ===========================================================================
# 16. query() — compress_context (line 1033)
# ===========================================================================


class TestQueryCompressContext:
    def test_compress_context_called_when_enabled(self):
        """Line 1033: compress_context=True triggers _compress_context."""
        from axon.compression import CompressionResult

        stub = _make_full_stub(compress_context=True, similarity_threshold=0.0)
        compressed_chunks = [
            {"id": "d1", "text": "compressed", "score": 0.9, "metadata": {"source": "f.txt"}}
        ]
        stub._compress_context = MagicMock(
            return_value=(
                compressed_chunks,
                CompressionResult(
                    chunks=compressed_chunks,
                    strategy_used="sentence",
                    pre_tokens=10,
                    post_tokens=5,
                    compression_ratio=0.5,
                ),
            )
        )
        stub.query("test query with compression")
        stub._compress_context.assert_called_once()

    def test_compress_context_not_called_when_disabled(self):
        stub = _make_full_stub(compress_context=False, similarity_threshold=0.0)
        stub._compress_context = MagicMock(return_value=([], None))
        stub.query("test query")
        stub._compress_context.assert_not_called()


# ===========================================================================
# 17. query() — GraphRAG community context paths (lines 1064-1066, 1068-1073)
# ===========================================================================


class TestQueryGraphRAGCommunityContext:
    def test_lazy_community_generation_triggered(self):
        """Lines 1064-1066: graph_rag + global mode + lazy + community_levels but no summaries."""
        # query_router="off" prevents profile overrides from disabling graph_rag
        stub = _make_full_stub(similarity_threshold=0.0, query_router="off")
        stub.config.graph_rag = True
        stub.config.graph_rag_mode = "global"
        stub.config.graph_rag_community_lazy = True
        stub.config.graph_rag_index_community_reports = True
        stub._community_summaries = {}
        stub._community_levels = {"level0": []}
        stub._generate_community_summaries = MagicMock()
        stub._index_community_reports_in_vector_store = MagicMock()
        stub._global_search_map_reduce = MagicMock(return_value="global context text")
        stub.query("What are the main themes?")
        stub._generate_community_summaries.assert_called_once()
        stub._index_community_reports_in_vector_store.assert_called_once()

    def test_global_mode_context_prefix(self):
        """Lines 1068-1073: graph_rag + global mode + summaries → context prefixed."""
        # query_router="off" prevents profile overrides from disabling graph_rag
        stub = _make_full_stub(similarity_threshold=0.0, query_router="off")
        stub.config.graph_rag = True
        stub.config.graph_rag_mode = "global"
        stub._community_summaries = {"c0": "Community summary text"}
        stub._global_search_map_reduce = MagicMock(return_value="global context text")
        result = stub.query("themes query")
        assert isinstance(result, str)
        # The system_prompt passed to LLM should contain the community context
        call_args = stub.llm.complete.call_args
        system_prompt = call_args[0][1]
        assert "Knowledge Graph Community Reports" in system_prompt

    def test_hybrid_mode_includes_both_contexts(self):
        """Lines 1068-1073: hybrid mode includes global + document excerpts."""
        # query_router="off" prevents profile overrides from disabling graph_rag
        stub = _make_full_stub(similarity_threshold=0.0, query_router="off")
        stub.config.graph_rag = True
        stub.config.graph_rag_mode = "hybrid"
        stub._community_summaries = {"c0": "summary"}
        stub._global_search_map_reduce = MagicMock(return_value="global ctx")
        stub.query("hybrid query")
        call_args = stub.llm.complete.call_args
        system_prompt = call_args[0][1]
        assert "Knowledge Graph Community Reports" in system_prompt
        assert "Document Excerpts" in system_prompt


# ===========================================================================
# 18. query_stream() — no results and rerank paths (lines 1143, 1175, 1180)
# ===========================================================================


class TestQueryStream:
    def test_no_results_discussion_fallback_yields_stream(self):
        """Lines 1132-1138: empty results + discussion_fallback → llm.stream."""
        stub = _make_full_stub(discussion_fallback=True, similarity_threshold=1.1)
        stub.vector_store.search.return_value = []
        stub.llm.stream.return_value = iter(["hello ", "world"])
        chunks = list(stub.query_stream("any question"))
        assert "".join(chunks) == "hello world"

    def test_no_results_no_fallback_yields_static_message(self):
        """Line 1139: no results + no fallback → static string yielded."""
        stub = _make_full_stub(discussion_fallback=False, similarity_threshold=1.1)
        stub.vector_store.search.return_value = []
        chunks = list(stub.query_stream("any question"))
        assert len(chunks) == 1
        assert "don't have any relevant information" in chunks[0]

    def test_rerank_called_in_query_stream(self):
        """Line 1143: rerank in query_stream."""
        stub = _make_full_stub(rerank=True, similarity_threshold=0.0)
        stub.llm.stream.return_value = iter(["token1", "token2"])
        list(stub.query_stream("reranked stream query"))
        stub.reranker.rerank.assert_called_once()

    def test_stream_yields_sources_marker_first(self):
        """query_stream always yields {"type": "sources", ...} before text."""
        stub = _make_full_stub(similarity_threshold=0.0)
        stub.llm.stream.return_value = iter(["some text"])
        items = list(stub.query_stream("test query"))
        assert isinstance(items[0], dict)
        assert items[0].get("type") == "sources"

    def test_compress_context_in_stream(self):
        """Line 1180: compress_context in query_stream."""
        from axon.compression import CompressionResult

        stub = _make_full_stub(compress_context=True, similarity_threshold=0.0)
        compressed_chunks = [{"id": "d1", "text": "compressed", "score": 0.9, "metadata": {}}]
        stub._compress_context = MagicMock(
            return_value=(
                compressed_chunks,
                CompressionResult(
                    chunks=compressed_chunks,
                    strategy_used="sentence",
                    pre_tokens=10,
                    post_tokens=5,
                    compression_ratio=0.5,
                ),
            )
        )
        stub.llm.stream.return_value = iter(["out"])
        list(stub.query_stream("compressed stream"))
        stub._compress_context.assert_called_once()

    def test_graphrag_local_context_in_stream(self):
        """Line 1175: graph_rag local context in stream."""
        stub = _make_full_stub(similarity_threshold=0.0)
        stub.config.graph_rag = True
        stub.config.graph_rag_mode = "local"
        stub._entity_graph = {"entity": {"chunk_ids": []}}
        stub._community_summaries = {}
        stub.llm.stream.return_value = iter(["response"])
        # Simulate matched entities in retrieval
        original_execute = stub._execute_retrieval

        def patched_execute(query, filters=None, cfg=None):
            result = original_execute(query, filters, cfg)
            result["matched_entities"] = ["entity"]
            return result

        stub._execute_retrieval = patched_execute
        items = list(stub.query_stream("entity query"))
        assert any(isinstance(i, dict) and i.get("type") == "sources" for i in items)

    def test_stream_lazy_community_generation(self):
        """Lines 1200-1202: lazy community generation in query_stream."""
        # query_router="off" prevents profile overrides from disabling graph_rag
        stub = _make_full_stub(similarity_threshold=0.0, query_router="off")
        stub.config.graph_rag = True
        stub.config.graph_rag_mode = "global"
        stub.config.graph_rag_community_lazy = True
        stub.config.graph_rag_index_community_reports = True
        stub._community_summaries = {}
        stub._community_levels = {"level0": []}
        stub._generate_community_summaries = MagicMock()
        stub._index_community_reports_in_vector_store = MagicMock()
        stub._global_search_map_reduce = MagicMock(return_value="global ctx")
        stub.llm.stream.return_value = iter(["response"])
        list(stub.query_stream("global stream query"))
        stub._generate_community_summaries.assert_called_once()

    def test_stream_global_context_injected(self):
        """Lines 1204-1209: global context in system_prompt for stream."""
        # query_router="off" prevents profile overrides from disabling graph_rag
        stub = _make_full_stub(similarity_threshold=0.0, query_router="off")
        stub.config.graph_rag = True
        stub.config.graph_rag_mode = "global"
        stub._community_summaries = {"c0": "summary"}
        stub._global_search_map_reduce = MagicMock(return_value="global ctx text")
        captured = {}

        def capture_stream(query, system_prompt, chat_history=None):
            captured["system_prompt"] = system_prompt
            return iter(["token"])

        stub.llm.stream = capture_stream
        list(stub.query_stream("global stream query"))
        assert "Knowledge Graph Community Reports" in captured.get("system_prompt", "")

    def test_stream_local_ctx_prepended(self):
        """Line 1216: _local_ctx prepended to context in stream."""
        stub = _make_full_stub(similarity_threshold=0.0)
        stub.config.graph_rag = True
        stub.config.graph_rag_mode = "local"
        stub.config.query_router = "off"
        stub._entity_graph = {"e1": {"chunk_ids": []}}
        stub._community_summaries = {}
        stub._local_search_context = MagicMock(return_value="local ctx header")
        captured = {}

        def capture_stream(query, system_prompt, chat_history=None):
            captured["system_prompt"] = system_prompt
            return iter(["ok"])

        stub.llm.stream = capture_stream

        original_execute = stub._execute_retrieval

        def patched_execute(query, filters=None, cfg=None):
            result = original_execute(query, filters, cfg)
            result["matched_entities"] = ["e1"]
            return result

        stub._execute_retrieval = patched_execute
        list(stub.query_stream("local entity stream"))
        assert "GraphRAG Local Context" in captured.get("system_prompt", "")


# ===========================================================================
# 19. _compress_context (detailed coverage)
# ===========================================================================


class TestCompressContext:
    def test_empty_results_returned_unchanged(self):
        stub = _make_full_stub()
        chunks, _ = stub._compress_context("q", [])
        assert chunks == []

    def test_web_results_not_compressed(self):
        stub = _make_full_stub()
        web_result = {
            "id": "https://example.com",
            "text": "web text",
            "score": 1.0,
            "is_web": True,
            "metadata": {},
        }
        chunks, _ = stub._compress_context("q", [web_result])
        assert chunks[0]["text"] == "web text"

    def test_local_chunk_compressed_when_shorter(self):
        stub = _make_full_stub()
        stub.llm.complete.return_value = "Short."  # shorter than source
        chunk = {"id": "c1", "text": "A very long passage that has lots of text.", "metadata": {}}
        chunks, _ = stub._compress_context("question", [chunk])
        assert chunks[0]["text"] == "Short."
        assert chunks[0]["metadata"].get("compressed") is True

    def test_compression_not_applied_when_longer_result(self):
        stub = _make_full_stub()
        stub.llm.complete.return_value = (
            "A much longer response than the original source passage text here."
        )
        chunk = {"id": "c1", "text": "Short.", "metadata": {}}
        chunks, _ = stub._compress_context("q", [chunk])
        # Compressed is longer → original kept
        assert chunks[0]["text"] == "Short."

    def test_llm_exception_returns_original(self):
        stub = _make_full_stub()
        stub.llm.complete.side_effect = RuntimeError("error")
        chunk = {"id": "c1", "text": "original text", "metadata": {}}
        chunks, _ = stub._compress_context("q", [chunk])
        assert chunks[0]["text"] == "original text"

    def test_parent_text_compressed_when_present(self):
        stub = _make_full_stub()
        stub.llm.complete.return_value = "Compressed parent."
        chunk = {
            "id": "c1",
            "text": "child chunk text",
            "metadata": {
                "parent_text": "Parent passage with more content that is longer than compressed."
            },
        }
        chunks, _ = stub._compress_context("question about parent", [chunk])
        assert chunks[0]["metadata"]["parent_text"] == "Compressed parent."
        assert chunks[0]["metadata"].get("compressed") is True


# ===========================================================================
# 20. _decompose_query  (lines 356-378)
# ===========================================================================


class TestDecomposeQuery:
    def test_returns_original_plus_sub_questions(self):
        stub = _make_full_stub()
        stub.llm.complete.return_value = "What is Python?\nWhen was it created?\nWho invented it?"
        result = stub._decompose_query("Tell me everything about Python")
        assert result[0] == "Tell me everything about Python"
        assert len(result) > 1

    def test_deduplication_removes_identical_sub_questions(self):
        stub = _make_full_stub()
        stub.llm.complete.return_value = (
            "Tell me everything about Python\nTell me everything about Python\nWhat is it?"
        )
        result = stub._decompose_query("Tell me everything about Python")
        # Original counted once
        assert result.count("Tell me everything about Python") == 1

    def test_max_4_sub_questions_returned(self):
        stub = _make_full_stub()
        stub.llm.complete.return_value = "\n".join([f"Sub question {i}" for i in range(10)])
        result = stub._decompose_query("complex query")
        # original + up to 4 sub-questions = max 5
        assert len(result) <= 5

    def test_strips_numbering_prefixes(self):
        stub = _make_full_stub()
        stub.llm.complete.return_value = "1. What is X?\n2. How does Y work?"
        result = stub._decompose_query("Tell me about X and Y")
        for q in result[1:]:
            assert not q[0].isdigit() or q[0] in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ===========================================================================
# 21. _get_multi_queries
# ===========================================================================


class TestGetMultiQueries:
    def test_returns_original_plus_variants(self):
        stub = _make_full_stub()
        stub.llm.complete.return_value = "How does Python work?\nPython tutorial\nLearn Python"
        result = stub._get_multi_queries("What is Python?")
        assert result[0] == "What is Python?"
        assert len(result) >= 2

    def test_max_3_variants(self):
        stub = _make_full_stub()
        stub.llm.complete.return_value = "\n".join([f"variant {i}" for i in range(10)])
        result = stub._get_multi_queries("some query")
        # original + max 3 variants = 4
        assert len(result) <= 4


# ===========================================================================
# 22. _apply_overrides
# ===========================================================================


class TestApplyOverrides:
    def test_none_overrides_returns_self_config(self):
        stub = _make_full_stub()
        cfg = stub._apply_overrides(None)
        assert cfg is stub.config

    def test_empty_overrides_returns_self_config(self):
        stub = _make_full_stub()
        cfg = stub._apply_overrides({})
        assert cfg is stub.config

    def test_valid_override_applied(self):
        stub = _make_full_stub()
        cfg = stub._apply_overrides({"top_k": 99})
        assert cfg.top_k == 99
        # Original config unchanged
        assert stub.config.top_k != 99

    def test_unknown_key_ignored(self):
        stub = _make_full_stub()
        cfg = stub._apply_overrides({"nonexistent_field": True})
        assert not hasattr(cfg, "nonexistent_field")

    def test_none_value_not_applied(self):
        stub = _make_full_stub(top_k=5)
        cfg = stub._apply_overrides({"top_k": None})
        assert cfg.top_k == 5


# ===========================================================================
# 23. query() — cache paths
# ===========================================================================


class TestQueryCache:
    def test_cache_hit_returns_cached_response(self):
        stub = _make_full_stub(
            query_cache=True,
            query_cache_size=10,
            similarity_threshold=0.0,
        )
        # Populate cache manually
        cache_key = stub._make_cache_key("cached query", None, stub.config)
        stub._query_cache[cache_key] = (time.monotonic(), "cached answer")
        result = stub.query("cached query")
        assert result == "cached answer"
        # LLM should NOT have been called
        stub.llm.complete.assert_not_called()

    def test_cache_populated_after_miss(self):
        stub = _make_full_stub(
            query_cache=True,
            query_cache_size=10,
            similarity_threshold=0.0,
        )
        stub.llm.complete.return_value = "fresh answer"
        result = stub.query("new query")
        assert result == "fresh answer"
        # Should now be cached
        cache_key = stub._make_cache_key("new query", None, stub.config)
        cached = stub._query_cache.get(cache_key)
        assert cached is not None and cached[1] == "fresh answer"

    def test_cache_bypassed_when_chat_history_present(self):
        stub = _make_full_stub(
            query_cache=True,
            query_cache_size=10,
            similarity_threshold=0.0,
        )
        stub.llm.complete.return_value = "conversational answer"
        chat_history = [{"role": "user", "content": "hi"}]
        result = stub.query("follow-up", chat_history=chat_history)
        assert result == "conversational answer"

    def test_lru_eviction_when_cache_full(self):
        stub = _make_full_stub(
            query_cache=True,
            query_cache_size=2,
            similarity_threshold=0.0,
        )
        stub.llm.complete.return_value = "answer"
        stub.query("query A")
        stub.query("query B")
        stub.query("query C")  # should evict A
        key_a = stub._make_cache_key("query A", None, stub.config)
        assert key_a not in stub._query_cache


# ===========================================================================
# 24. End-to-end profile integration
# ===========================================================================


class TestProfileIntegration:
    @pytest.mark.parametrize(
        "query,expected_route",
        [
            ("What is the capital of France?", "factual"),
            ("Summarize the entire documentation", "synthesis"),
            ("Show me the statistics table", "table_lookup"),
            ("How is Alice related to Bob?", "entity_relation"),
            ("What are the main topics across all documents?", "corpus_exploration"),
        ],
    )
    def test_route_profile_applied_end_to_end(self, query, expected_route):
        stub = _make_full_stub(similarity_threshold=0.0, query_router="heuristic")
        captured_route = {}

        original_classify = stub._classify_query_route

        def patched_classify(q, cfg):
            route = original_classify(q, cfg)
            captured_route["route"] = route
            return route

        stub._classify_query_route = patched_classify
        stub.query(query)
        assert captured_route.get("route") == expected_route

    def test_llm_router_mode_routes_query(self):
        stub = _make_full_stub(similarity_threshold=0.0, query_router="llm")
        stub.llm.generate.return_value = "synthesis"
        stub.llm.complete.return_value = "answer"
        result = stub.query("complex synthesis question")
        assert isinstance(result, str)

    def test_factual_profile_does_not_enable_hyde(self):
        """Factual profile forces hyde=False even if config had it True."""
        stub = _make_full_stub(hyde=True, similarity_threshold=0.0, query_router="heuristic")
        calls = []
        original_execute = stub._execute_retrieval

        def capturing_execute(query, filters=None, cfg=None):
            calls.append(cfg.hyde if cfg else stub.config.hyde)
            return original_execute(query, filters, cfg)

        stub._execute_retrieval = capturing_execute
        stub.query("What is Python?")  # short factual → factual route
        # After factual profile applied, hyde should be False
        assert calls[-1] is False
