from unittest.mock import MagicMock, patch

import pytest

from axon.graph_backends.graphrag_engine import GraphRagEngine
from axon.main import AxonBrain, AxonConfig


class TestBrainLifecycle:
    @pytest.fixture
    def mock_dependencies(self):
        with patch("axon.main.OpenVectorStore"), patch("axon.retrievers.BM25Retriever"), patch(
            "axon.main.OpenEmbedding"
        ), patch("axon.main.OpenLLM"), patch("axon.main.OpenReranker"), patch(
            "axon.projects.ensure_project"
        ), patch(
            "axon.projects.ensure_user_project"
        ):
            yield

    def test_brain_init_defaults(self, mock_dependencies, tmp_path):
        config = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        # Ensure we don't try to load real files
        with patch.object(AxonBrain, "_load_hash_store", return_value=set()), patch.object(
            AxonBrain, "_load_doc_versions"
        ), patch.object(GraphRagEngine, "_load_entity_graph", return_value={}), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_relation_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_levels", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_summaries", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_entity_embeddings", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_claims_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_hierarchy", return_value={}
        ):
            brain = AxonBrain(config)

            assert hasattr(brain, "vector_store")
            assert hasattr(brain, "bm25")
            assert hasattr(brain, "llm")
            assert hasattr(brain, "embedding")
            assert brain._graph_backend._engine._entity_graph == {}

            brain.close()

    def test_brain_close_teardown(self, mock_dependencies, tmp_path):
        config = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        with patch.object(AxonBrain, "_load_hash_store", return_value=set()), patch.object(
            AxonBrain, "_load_doc_versions"
        ), patch.object(GraphRagEngine, "_load_entity_graph", return_value={}), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_relation_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_levels", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_summaries", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_entity_embeddings", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_claims_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_hierarchy", return_value={}
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
        ), patch.object(GraphRagEngine, "_load_entity_graph", return_value={}), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_relation_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_levels", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_summaries", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_entity_embeddings", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_claims_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_hierarchy", return_value={}
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

    def test_close_wipes_sealed_cache_even_when_store_close_raises(
        self, mock_dependencies, tmp_path
    ):
        """Regression: a misbehaving store.close() must not prevent the sealed
        cache wipe from running. Pre-fix, an exception in the close loop
        terminated the loop early and skipped the ``_sealed_cache`` cleanup,
        leaving decrypted plaintext on disk.
        """
        config = AxonConfig(bm25_path=str(tmp_path), vector_store_path=str(tmp_path))
        with patch.object(AxonBrain, "_load_hash_store", return_value=set()), patch.object(
            AxonBrain, "_load_doc_versions"
        ), patch.object(GraphRagEngine, "_load_entity_graph", return_value={}), patch.object(
            AxonBrain, "_load_code_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_relation_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_levels", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_summaries", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_entity_embeddings", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_claims_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_hierarchy", return_value={}
        ):
            brain = AxonBrain(config)
            # Distinct objects so the close loop visits both arms.
            vs = MagicMock()
            vs.close = MagicMock(side_effect=OSError("file locked"))
            bm = MagicMock()
            bm.close = MagicMock()
            brain.vector_store = vs
            brain._own_vector_store = vs
            brain.bm25 = bm
            brain._own_bm25 = bm
            # Plant a fake sealed cache and intercept release_cache.
            fake_cache = MagicMock(path=str(tmp_path / "fake_cache"))
            brain._sealed_cache = fake_cache

            release_calls: list = []

            def _fake_release(cache):
                release_calls.append(cache)

            fake_mount_mod = MagicMock()
            fake_mount_mod.release_cache = _fake_release
            with patch.dict("sys.modules", {"axon.security.mount": fake_mount_mod}, clear=False):
                brain.close()

            # The vector store raised, but the rest of the cleanup must
            # still have happened: bm25 closed AND sealed cache released.
            bm.close.assert_called_once()
            assert release_calls == [fake_cache], (
                "Sealed cache release must run even when a prior store.close() raised; "
                f"got release_calls={release_calls!r}"
            )
            assert brain._sealed_cache is None

    def test_switch_project_rebinds_doc_versions_path(self, mock_dependencies, tmp_path):
        """Regression: switch_project must rebind ``_doc_versions_path`` and
        reload ``_doc_versions``. Pre-fix, the path stayed pinned to the
        project active at __init__, so ingests in the new project wrote
        their version metadata into the previous project's directory and
        ``get_doc_versions()`` returned stale cross-project data.
        """
        bm25_a = tmp_path / "proj_a" / "bm25"
        bm25_a.mkdir(parents=True, exist_ok=True)
        # Pre-stage a doc-versions file for project A so we can verify
        # it's the one initially loaded.
        (bm25_a / ".doc_versions.json").write_text(
            '{"src_a": {"chunk_count": 1}}', encoding="utf-8"
        )
        bm25_b = tmp_path / "proj_b" / "bm25"
        bm25_b.mkdir(parents=True, exist_ok=True)
        (bm25_b / ".doc_versions.json").write_text(
            '{"src_b": {"chunk_count": 7}}', encoding="utf-8"
        )
        config = AxonConfig(
            bm25_path=str(bm25_a),
            vector_store_path=str(tmp_path / "proj_a" / "vs"),
        )
        with patch.object(AxonBrain, "_load_hash_store", return_value=set()), patch.object(
            GraphRagEngine, "_load_entity_graph", return_value={}
        ), patch.object(AxonBrain, "_load_code_graph", return_value={}), patch.object(
            GraphRagEngine, "_load_relation_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_levels", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_summaries", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_entity_embeddings", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_claims_graph", return_value={}
        ), patch.object(
            GraphRagEngine, "_load_community_hierarchy", return_value={}
        ):
            brain = AxonBrain(config)
            # After construction, doc-versions came from project A.
            assert brain._doc_versions_path == str(bm25_a / ".doc_versions.json")
            assert "src_a" in brain._doc_versions

            proj_b_root = tmp_path / "proj_b_root"
            proj_b_root.mkdir(parents=True, exist_ok=True)
            (proj_b_root / "meta.json").write_text("{}", encoding="utf-8")
            with patch("axon.projects.project_dir", return_value=proj_b_root), patch(
                "axon.projects.project_vector_path",
                return_value=str(tmp_path / "proj_b" / "vs"),
            ), patch("axon.projects.project_bm25_path", return_value=str(bm25_b)), patch(
                "axon.projects.set_active_project"
            ), patch(
                "axon.projects.list_descendants", return_value=[]
            ), patch(
                "axon.projects.is_reserved_top_level_name", return_value=False
            ):
                brain.switch_project("proj_b")

            # After switch, both the path AND the loaded dict must reflect B.
            assert brain._doc_versions_path == str(bm25_b / ".doc_versions.json")
            assert "src_b" in brain._doc_versions
            assert "src_a" not in brain._doc_versions
            brain.close()
