from unittest.mock import MagicMock, patch

import pytest

from axon.main import AxonBrain, AxonConfig


class TestBrainLifecycle:
    @pytest.fixture
    def mock_dependencies(self):
        with patch("axon.main.OpenVectorStore"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.main.OpenEmbedding"
        ), patch("axon.main.OpenLLM"), patch("axon.main.OpenReranker"), patch(
            "axon.projects.ensure_project"
        ), patch(
            "axon.projects.ensure_user_namespace"
        ):
            yield

    def test_brain_init_defaults(self, mock_dependencies, tmp_path):
        config = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        # Ensure we don't try to load real files
        with patch.object(AxonBrain, "_load_hash_store", return_value=set()), patch.object(
            AxonBrain, "_load_doc_versions"
        ), patch.object(AxonBrain, "_load_entity_graph", return_value={}), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ):
            brain = AxonBrain(config)

            assert hasattr(brain, "vector_store")
            assert hasattr(brain, "bm25")
            assert hasattr(brain, "llm")
            assert hasattr(brain, "embedding")
            assert brain._entity_graph == {}

            brain.close()

    def test_brain_close_teardown(self, mock_dependencies, tmp_path):
        config = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        with patch.object(AxonBrain, "_load_hash_store", return_value=set()), patch.object(
            AxonBrain, "_load_doc_versions"
        ), patch.object(AxonBrain, "_load_entity_graph", return_value={}), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ):
            brain = AxonBrain(config)

            # Mocking the close methods of dependencies
            brain.vector_store.close = MagicMock()
            # bm25 might be None if import fails in real code, but here it's mocked
            if brain.bm25:
                brain.bm25.close = MagicMock()

            brain.close()

            brain.vector_store.close.assert_called_once()
            if brain.bm25:
                brain.bm25.close.assert_called_once()

    def test_brain_context_manager(self, mock_dependencies, tmp_path):
        config = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))

        with patch.object(AxonBrain, "_load_hash_store", return_value=set()), patch.object(
            AxonBrain, "_load_doc_versions"
        ), patch.object(AxonBrain, "_load_entity_graph", return_value={}), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_relation_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_levels", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_summaries", return_value={}
        ), patch.object(
            AxonBrain, "_load_entity_embeddings", return_value={}
        ), patch.object(
            AxonBrain, "_load_claims_graph", return_value={}
        ), patch.object(
            AxonBrain, "_load_community_hierarchy", return_value={}
        ):
            with AxonBrain(config) as brain:
                assert hasattr(brain, "vector_store")
                brain.vector_store.close = MagicMock()
                if brain.bm25:
                    brain.bm25.close = MagicMock()
                mock_close_vs = brain.vector_store.close
                mock_close_bm25 = brain.bm25.close if brain.bm25 else None

            mock_close_vs.assert_called_once()
            if mock_close_bm25:
                mock_close_bm25.assert_called_once()
