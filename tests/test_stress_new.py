import concurrent.futures
from unittest.mock import patch

import pytest

from axon.main import AxonBrain, AxonConfig


class TestStressNew:
    @pytest.fixture
    def mock_brain(self, tmp_path):
        config = AxonConfig(
            bm25_path=str(tmp_path),
            vector_store_path=str(tmp_path),
            graph_rag=True,
            graph_rag_relations=True,
            graph_rag_min_entities_for_relations=0,
            raptor=False,
            dedup_on_ingest=False,  # Disable dedup
            max_workers=10,
        )
        # Mocking expensive parts
        with patch("axon.main.OpenVectorStore"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.main.OpenEmbedding"
        ) as mock_embed, patch("axon.main.OpenLLM") as mock_llm, patch(
            "axon.main.OpenReranker"
        ), patch(
            "axon.projects.ensure_project"
        ), patch(
            "axon.projects.ensure_user_project"
        ):
            mock_embed.return_value.embed.return_value = [[0.1] * 1536]
            mock_embed.return_value.embed_query.return_value = [0.1] * 1536

            def llm_complete_side_effect(prompt, system_prompt=None, **kwargs):
                if "named entity extraction" in (system_prompt or "").lower():
                    return "EntityX | CONCEPT | desc"
                return "OK"

            mock_llm.return_value.complete.side_effect = llm_complete_side_effect

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

    def test_concurrent_duplicate_ingestion_race(self, mock_brain):
        """Ingest the same document ID concurrently and check for duplicate chunk_ids."""
        num_threads = 50
        # Same document ID for all calls
        doc = {"id": "duplicate_doc", "text": "Content with EntityX."}

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(mock_brain.ingest, [doc]) for _ in range(num_threads)]
            concurrent.futures.wait(futures)

        assert "entityx" in mock_brain._entity_graph

        count = len(mock_brain._entity_graph["entityx"]["chunk_ids"])
        print(f"DEBUG: entityx chunk_ids count={count}")

        # If thread-safe, count should be 1 because of 'if doc_id not in ...' check.
        # If there's a race condition, it will be > 1.
        assert count == 1
