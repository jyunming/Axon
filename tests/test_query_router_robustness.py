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

    @pytest.mark.parametrize("query,expected", [
        ("what is axon?", "factual"),
        ("summarize everything", "corpus_exploration"),
        ("how does x relate to y?", "entity_relation"),
        ("list all rows in the data", "table_lookup"),
        ("give me an overview of the whole thing", "synthesis"),
    ])
    def test_heuristic_parameterized(self, router, query, expected):
        assert router._classify_query_route_heuristic(query) == expected

    def test_classification_latency_benchmark(self, router):
        import time
        queries = [
            "What is the capital of France?",
            "Summarize the entire documentation for the project.",
            "How are entities A and B related in the context of the new architecture?",
            "Show me a table of the revenue by region.",
            "Provide a comprehensive overview of the system's security features."
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
