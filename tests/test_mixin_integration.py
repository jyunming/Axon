from unittest.mock import patch

import pytest

from axon.main import AxonBrain, AxonConfig


class TestMixinIntegration:
    @pytest.fixture
    def mock_brain(self, tmp_path):
        config = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag=True,
            graph_rag_relations=True,
            graph_rag_min_entities_for_relations=0,
        )
        # Mocking expensive parts
        with patch("axon.main.OpenVectorStore") as mock_vs, patch(
            "axon.retrievers.BM25Retriever"
        ), patch("axon.main.OpenEmbedding") as mock_embed, patch(
            "axon.main.OpenLLM"
        ) as mock_llm, patch(
            "axon.main.OpenReranker"
        ), patch(
            "axon.projects.ensure_project"
        ), patch(
            "axon.projects.ensure_user_project"
        ):
            # Mock vector store search to return something
            mock_vs.return_value.search.return_value = [
                {
                    "id": "doc1_p0_chunk_0",
                    "text": "Axon content",
                    "score": 0.9,
                    "metadata": {"source": "doc1"},
                }
            ]

            # Mock embedding to return dummy vectors
            mock_embed.return_value.embed.return_value = [[0.1] * 1536]
            mock_embed.return_value.embed_query.return_value = [0.1] * 1536

            def llm_complete_side_effect(prompt, system_prompt=None, **kwargs):
                prompt_l = prompt.lower()
                sys_l = (system_prompt or "").lower()

                if "extract the key named entities" in prompt_l or "entity extraction" in sys_l:
                    return "Axon | SOFTWARE | An open source RAG system\nLLM | CONCEPT | Large Language Model\nPython | LANGUAGE | Programming language"
                if "extract key relationships" in prompt_l or "knowledge graph extraction" in sys_l:
                    return "Axon | uses | LLM | Axon uses LLM for generation | 9"
                if "summarise the following passage" in prompt_l:
                    return "Summary of the passage."
                return "The answer is Axon."

            mock_llm.return_value.complete.side_effect = llm_complete_side_effect

            # Patch all load methods to avoid reading non-existent files
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
                yield brain
                brain.close()

    def test_ingest_populates_graph(self, mock_brain):
        docs = [{"id": "doc1", "text": "Axon is a software that uses LLM."}]
        mock_brain.ingest(docs)

        # Keys are lowercased and stripped
        assert "axon" in mock_brain._entity_graph
        assert "axon" in mock_brain._relation_graph
        assert mock_brain._relation_graph["axon"][0]["target"] == "llm"

    def test_query_routing_to_graph_rag(self, mock_brain):
        # 1. Ingest first to populate the graph
        docs = [{"id": "doc1", "text": "Axon is a software that uses LLM."}]
        mock_brain.ingest(docs)

        # Mock the classification to return entity_relation
        with patch.object(mock_brain, "_classify_query_route", return_value="entity_relation"):
            # Mock the actual local search context method
            with patch.object(
                mock_brain, "_local_search_context", return_value="Some graph context"
            ) as mock_graph_retrieval:
                mock_brain.query("Who is Axon?")
                # Should have been called because we have a populated graph and matched entities
                mock_graph_retrieval.assert_called()
