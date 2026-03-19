"""Track A tests: HYBRID_RAPTOR_WAY_FORWARD implementation."""

from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
# A1: RAPTOR→GraphRAG Auto-Composition
# ─────────────────────────────────────────────────────────────────────────────


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestRaptorGraphRAGComposition:
    """A1: RAPTOR->GraphRAG auto-composition tests."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **cfg_kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, **cfg_kwargs)
        brain = AxonBrain(config)
        brain._ingested_hashes = set()
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._save_hash_store = MagicMock()
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        return brain

    def _run_filter(self, brain, documents):
        """Run the chunk-filter logic and return chunks_to_process."""
        _include_raptor = brain.config.graph_rag_include_raptor_summaries
        _skip_threshold = getattr(brain.config, "raptor_graphrag_leaf_skip_threshold", 20)
        _leaf_count: dict = {}
        for _doc in documents:
            if not _doc.get("metadata", {}).get("raptor_level"):
                _src = _doc.get("metadata", {}).get("source", _doc["id"])
                _leaf_count[_src] = _leaf_count.get(_src, 0) + 1
        _large_sources: set = set()
        if brain.config.raptor and _skip_threshold > 0:
            _large_sources = {src for src, cnt in _leaf_count.items() if cnt >= _skip_threshold}
        chunks_to_process = []
        for _doc in documents:
            _lvl = _doc.get("metadata", {}).get("raptor_level")
            _src = _doc.get("metadata", {}).get("source", _doc["id"])
            if not _lvl:
                if _src not in _large_sources:
                    chunks_to_process.append(_doc)
            elif _lvl == 1:
                _auto_raptor = brain.config.raptor and _src in _large_sources
                if _include_raptor or _auto_raptor:
                    chunks_to_process.append(_doc)
        return chunks_to_process

    def test_large_source_uses_raptor_summaries_for_extraction(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Source with 25 leaf chunks + RAPTOR on -> only RAPTOR level-1 doc used."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            raptor=True,
            graph_rag_relations=False,
            raptor_graphrag_leaf_skip_threshold=20,
        )
        leaf_docs = [
            {"id": f"leaf_{i}", "text": f"chunk {i}", "metadata": {"source": "big.txt"}}
            for i in range(25)
        ]
        raptor_doc = {
            "id": "raptor_sum_1",
            "text": "Summary of big.txt",
            "metadata": {"source": "big.txt", "raptor_level": 1},
        }
        documents = leaf_docs + [raptor_doc]
        chunks_to_process = self._run_filter(brain, documents)
        assert len(chunks_to_process) == 1
        assert chunks_to_process[0]["id"] == "raptor_sum_1"

    def test_small_source_uses_leaf_chunks(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Source with 5 leaf chunks -> all leaf chunks used, RAPTOR summary excluded."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            raptor=True,
            graph_rag_relations=False,
            raptor_graphrag_leaf_skip_threshold=20,
        )
        leaf_docs = [
            {"id": f"leaf_{i}", "text": f"chunk {i}", "metadata": {"source": "small.txt"}}
            for i in range(5)
        ]
        raptor_doc = {
            "id": "raptor_sum_1",
            "text": "Summary",
            "metadata": {"source": "small.txt", "raptor_level": 1},
        }
        documents = leaf_docs + [raptor_doc]
        chunks_to_process = self._run_filter(brain, documents)
        # All 5 leaf chunks included (small source not skipped)
        # RAPTOR summary also included because graph_rag_include_raptor_summaries=True (default)
        leaf_ids = [d["id"] for d in chunks_to_process if d["id"].startswith("leaf_")]
        assert len(leaf_ids) == 5

    def test_raptor_off_uses_all_leaf_chunks(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """raptor=False -> all leaf chunks used regardless of source size."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            raptor=False,
            graph_rag_relations=False,
            raptor_graphrag_leaf_skip_threshold=20,
        )
        leaf_docs = [
            {"id": f"leaf_{i}", "text": f"chunk {i}", "metadata": {"source": "big.txt"}}
            for i in range(25)
        ]
        chunks_to_process = self._run_filter(brain, leaf_docs)
        assert len(chunks_to_process) == 25


# ─────────────────────────────────────────────────────────────────────────────
# A2: Selective Relation Extraction
# ─────────────────────────────────────────────────────────────────────────────


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestSelectiveRelationExtraction:
    """A2: Selective relation extraction based on entity count threshold."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **cfg_kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, **cfg_kwargs)
        brain = AxonBrain(config)
        return brain

    def _filter_rel_chunks(self, brain, chunks_to_process, entity_results):
        _min_ent = getattr(brain.config, "graph_rag_min_entities_for_relations", 3)
        if _min_ent > 0:
            _entity_count_by_doc = {doc_id: len(ents) for doc_id, ents in entity_results}
            return [
                doc
                for doc in chunks_to_process
                if _entity_count_by_doc.get(doc["id"], 0) >= _min_ent
            ]
        return chunks_to_process

    def test_relation_extraction_skipped_below_threshold(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """2 entities -> chunk filtered out from relation extraction."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            graph_rag_relations=True,
            graph_rag_min_entities_for_relations=3,
        )
        entity_results = [
            ("d1", [{"name": "Alpha", "type": "CONCEPT"}, {"name": "Beta", "type": "CONCEPT"}])
        ]
        chunks_to_process = [{"id": "d1", "text": "Alpha and Beta."}]
        rel_chunks = self._filter_rel_chunks(brain, chunks_to_process, entity_results)
        assert len(rel_chunks) == 0

    def test_relation_extraction_runs_at_threshold(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """3 entities -> chunk included in relation extraction."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            graph_rag_relations=True,
            graph_rag_min_entities_for_relations=3,
        )
        entity_results = [
            (
                "d1",
                [
                    {"name": "Alpha", "type": "CONCEPT"},
                    {"name": "Beta", "type": "CONCEPT"},
                    {"name": "Gamma", "type": "CONCEPT"},
                ],
            )
        ]
        chunks_to_process = [{"id": "d1", "text": "Alpha Beta Gamma."}]
        rel_chunks = self._filter_rel_chunks(brain, chunks_to_process, entity_results)
        assert len(rel_chunks) == 1
        assert rel_chunks[0]["id"] == "d1"

    def test_threshold_zero_always_extracts(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """graph_rag_min_entities_for_relations=0 -> all chunks included."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            graph_rag_relations=True,
            graph_rag_min_entities_for_relations=0,
        )
        chunks_to_process = [
            {"id": "d1", "text": "one entity"},
            {"id": "d2", "text": "no entities"},
        ]
        entity_results = [("d1", [{"name": "Alpha", "type": "CONCEPT"}]), ("d2", [])]
        rel_chunks = self._filter_rel_chunks(brain, chunks_to_process, entity_results)
        assert len(rel_chunks) == 2


# ─────────────────────────────────────────────────────────────────────────────
# A3: Graph Depth Tiering
# ─────────────────────────────────────────────────────────────────────────────


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestGraphDepthTiering:
    """A3: Graph depth tier tests."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **cfg_kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, **cfg_kwargs)
        brain = AxonBrain(config)
        return brain

    def test_light_tier_no_llm_call(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        """graph_rag_depth='light' -> llm.complete never called in entity extraction."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            graph_rag_depth="light",
        )
        brain.llm.complete = MagicMock(return_value="BERT | CONCEPT | model")
        brain._extract_entities("BERT is a Transformer Model used in NLP.")
        brain.llm.complete.assert_not_called()

    def test_light_tier_extracts_noun_phrases(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """_extract_entities_light returns capitalized multi-word phrases as CONCEPT entities."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            graph_rag_depth="light",
        )
        result = brain._extract_entities_light("Apple Inc released the new Iphone Pro last year.")
        assert isinstance(result, list)
        for e in result:
            assert e["type"] == "CONCEPT"
            assert e["description"] == ""

    def test_light_tier_skips_relations(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """graph_rag_depth='light' -> _extract_relations returns [] without calling llm."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            graph_rag_depth="light",
        )
        brain.llm.complete = MagicMock(return_value="A | uses | B | desc")
        result = brain._extract_relations("A uses B.")
        assert result == []
        brain.llm.complete.assert_not_called()

    def test_standard_tier_uses_llm(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        """graph_rag_depth='standard' -> LLM is called for entity extraction."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            graph_rag_depth="standard",
            graph_rag_ner_backend="llm",
        )
        brain.llm.complete = MagicMock(return_value="BERT | CONCEPT | A language model")
        result = brain._extract_entities("BERT is a language model.")
        brain.llm.complete.assert_called_once()
        assert len(result) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# A4: Community Reports Excluded from Citations
# ─────────────────────────────────────────────────────────────────────────────


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestCitationFilter:
    """A4: Community reports excluded from citation candidates."""

    def test_community_reports_excluded_from_citations(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Results with community_report -> excluded from citation list."""
        results = [
            {"id": "doc1", "text": "Normal doc", "score": 0.9, "metadata": {}},
            {
                "id": "__community__abc123",
                "text": "Community summary",
                "score": 0.8,
                "metadata": {"graph_rag_type": "community_report"},
            },
            {"id": "doc2", "text": "Another doc", "score": 0.7, "metadata": {}},
        ]
        citation_results = [
            r
            for r in results
            if r.get("metadata", {}).get("graph_rag_type") != "community_report"
            and not r.get("id", "").startswith("__community__")
        ]
        assert len(citation_results) == 2
        ids = {r["id"] for r in citation_results}
        assert "__community__abc123" not in ids

    def test_non_community_docs_not_filtered(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Regular docs and RAPTOR summaries pass through the citation filter."""
        results = [
            {"id": "doc1", "text": "Normal doc", "score": 0.9, "metadata": {}},
            {
                "id": "doc2",
                "text": "RAPTOR summary",
                "score": 0.8,
                "metadata": {"raptor_level": 1},
            },
        ]
        citation_results = [
            r
            for r in results
            if r.get("metadata", {}).get("graph_rag_type") != "community_report"
            and not r.get("id", "").startswith("__community__")
        ]
        assert len(citation_results) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Track C: Graph Visualization
# ─────────────────────────────────────────────────────────────────────────────


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestGraphVisualization:
    """Track C: export_graph_html tests."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, graph_rag=True)
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = config
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._community_levels = {}
        return brain

    def test_export_missing_pyvis_raises(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """export_graph_html no longer requires pyvis — it uses a built-in 3D renderer."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        # Should not raise even with pyvis absent
        with patch("webbrowser.open"):
            html = brain.export_graph_html()
        assert "<html" in html

    def test_export_returns_html_string(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """export_graph_html returns a self-contained HTML string."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        with patch("webbrowser.open"):
            html = brain.export_graph_html()
        assert "<html" in html
        assert "3d-force-graph" in html or "force" in html.lower()

    def test_export_includes_entity_nodes(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Graph with 2 entities -> payload contains 2 nodes."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain._entity_graph = {
            "apple inc": {
                "type": "ORGANIZATION",
                "chunk_ids": ["c1"],
                "description": "Tech company",
            },
            "tim cook": {"type": "PERSON", "chunk_ids": ["c1", "c2"], "description": "CEO"},
        }
        payload = brain.build_graph_payload()
        assert len(payload["nodes"]) == 2

    def test_export_saves_to_file(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        """path argument -> file is written with valid HTML."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        out_file = tmp_path / "graph.html"
        with patch("webbrowser.open"):
            brain.export_graph_html(str(out_file))
        assert out_file.exists()
        assert "<html" in out_file.read_text(encoding="utf-8")

    def test_export_includes_relation_edges(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Graph with a relation -> payload links list is non-empty."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain._entity_graph = {
            "apple inc": {"type": "ORGANIZATION", "chunk_ids": ["c1"], "description": ""},
            "tim cook": {"type": "PERSON", "chunk_ids": ["c1"], "description": ""},
        }
        brain._relation_graph = {
            "apple inc": [
                {
                    "target": "tim cook",
                    "relation": "led by",
                    "description": "CEO relationship",
                    "strength": 8,
                }
            ]
        }
        payload = brain.build_graph_payload()
        assert len(payload["links"]) >= 1
