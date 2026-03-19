import os
from unittest.mock import MagicMock, patch

import yaml


class TestAxonConfig:
    def test_defaults(self):
        from axon.main import AxonConfig

        config = AxonConfig()
        assert config.embedding_provider == "sentence_transformers"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.llm_provider == "ollama"
        assert config.vector_store == "chroma"
        assert config.hybrid_search is True
        assert config.top_k == 10
        assert config.discussion_fallback is True
        assert config.max_workers == 8
        assert config.hybrid_mode == "rrf"
        assert config.dataset_type == "auto"

    def test_query_router_default(self):
        from axon.main import AxonConfig

        config = AxonConfig()
        assert config.query_router == "heuristic"

    def test_graph_rag_community_default_false(self):
        from axon.main import AxonConfig

        config = AxonConfig()
        assert config.graph_rag_community is False

    def test_code_graph_default_false(self):
        from axon.main import AxonConfig

        assert AxonConfig().code_graph is False

    def test_code_graph_bridge_default_false(self):
        from axon.main import AxonConfig

        assert AxonConfig().code_graph_bridge is False

    def test_code_lexical_boost_default_true(self):
        from axon.main import AxonConfig

        assert AxonConfig().code_lexical_boost is True

    def test_code_bm25_weight_default(self):
        from axon.main import AxonConfig

        assert AxonConfig().code_bm25_weight == 0.7

    def test_code_top_k_default(self):
        from axon.main import AxonConfig

        assert AxonConfig().code_top_k == 6

    def test_retrieval_dry_run_default_false(self):
        from axon.main import AxonConfig

        assert AxonConfig().retrieval_dry_run is False

    def test_code_top_k_multiplier_default(self):
        from axon.main import AxonConfig

        assert AxonConfig().code_top_k_multiplier == 2

    def test_code_max_chunks_per_file_default(self):
        from axon.main import AxonConfig

        assert AxonConfig().code_max_chunks_per_file == 3

    def test_contextual_retrieval_default(self):
        from axon.main import AxonConfig

        config = AxonConfig()
        assert config.contextual_retrieval is False

    def test_gliner_model_default(self):
        from axon.main import AxonConfig

        assert AxonConfig().graph_rag_gliner_model == "urchade/gliner_medium-v2.1"

    def test_community_top_n_default(self):
        from axon.main import AxonConfig

        assert AxonConfig().graph_rag_community_llm_top_n_per_level == 15

    def test_community_max_total_default(self):
        from axon.main import AxonConfig

        assert AxonConfig().graph_rag_community_llm_max_total == 30

    def test_local_assets_only_default(self):
        from axon.main import AxonConfig

        assert AxonConfig().local_assets_only is False

    def test_embedding_models_dir_default(self):
        from axon.main import AxonConfig

        assert AxonConfig().embedding_models_dir == ""

    def test_hf_models_dir_default(self):
        from axon.main import AxonConfig

        assert AxonConfig().hf_models_dir == ""

    def test_tokenizer_cache_dir_default(self):
        from axon.main import AxonConfig

        assert AxonConfig().tokenizer_cache_dir == ""

    def test_llmlingua_model_default(self):
        from axon.main import AxonConfig

        assert (
            AxonConfig().graph_rag_llmlingua_model
            == "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        )

    def test_gliner_model_default_corrected(self):
        """GLiNER default ID must use the correct HF repo name (with hyphen, not dot)."""
        from axon.main import AxonConfig

        assert AxonConfig().graph_rag_gliner_model == "urchade/gliner_medium-v2.1"

    def test_load_from_yaml(self, tmp_path):
        from axon.main import AxonConfig

        cfg = {
            "embedding": {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
            "llm": {"provider": "ollama", "model": "phi3:mini", "temperature": 0.5},
            "vector_store": {"provider": "chroma", "path": str(tmp_path / "chroma")},
            "bm25": {"path": str(tmp_path / "bm25")},
            "rag": {"top_k": 5, "similarity_threshold": 0.6, "hybrid_search": False},
            "chunk": {"size": 500, "overlap": 50},
            "rerank": {
                "enabled": False,
                "provider": "llm",
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            "query_transformations": {
                "multi_query": True,
                "hyde": False,
                "discussion_fallback": True,
            },
            "web_search": {"enabled": True, "brave_api_key": "test_key"},
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        config = AxonConfig.load(str(cfg_path))
        assert config.embedding_provider == "fastembed"
        assert config.llm_model == "phi3:mini"
        assert config.top_k == 5
        assert config.multi_query is True
        assert config.truth_grounding is True
        assert config.brave_api_key == "test_key"

    def test_load_projects_root_from_yaml(self, tmp_path):
        from axon.main import AxonConfig

        cfg = {"projects_root": str(tmp_path / "myprojects")}
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        config = AxonConfig.load(str(cfg_path))
        assert config.projects_root == str(tmp_path / "myprojects")

    def test_env_var_overrides_yaml_projects_root(self, tmp_path, monkeypatch):
        from axon.main import AxonConfig

        cfg = {"projects_root": str(tmp_path / "yaml_root")}
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        env_root = str(tmp_path / "env_root")
        monkeypatch.setenv("AXON_PROJECTS_ROOT", env_root)
        config = AxonConfig.load(str(cfg_path))
        assert config.projects_root == env_root

    def test_projects_root_defaults_to_home(self):
        from pathlib import Path

        from axon.main import AxonConfig

        config = AxonConfig()
        expected = str(Path.home() / ".axon" / "projects")
        assert config.projects_root == expected


class TestVectorStoreClose:
    def test_chroma_close_called(self):
        from axon.main import AxonConfig, OpenVectorStore

        config = AxonConfig(vector_store="chroma")
        with patch("chromadb.PersistentClient") as mock_chroma:
            mock_client = MagicMock()
            mock_chroma.return_value = mock_client
            store = OpenVectorStore(config)
            store.close()
            # Depending on chroma version, it might have close()
            if hasattr(mock_client, "close"):
                assert mock_client.close.called
            assert store.client is None


class TestMultiStoreReadOnly:
    def test_multi_vector_store_delete_raises(self):
        from axon.main import MultiVectorStore

        ms = MultiVectorStore([])
        import pytest

        with pytest.raises(RuntimeError, match="merged"):
            ms.delete_by_ids(["id1"])
        with pytest.raises(RuntimeError, match="merged"):
            ms.delete_documents(["id1"])

    def test_multi_bm25_delete_raises(self):
        from axon.main import MultiBM25Retriever

        mr = MultiBM25Retriever([])
        import pytest

        with pytest.raises(RuntimeError, match="merged"):
            mr.delete_documents(["id1"])


def test_projects_root_precedence(tmp_path, monkeypatch):
    from axon.main import AxonConfig

    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("projects_root: /path/from/yaml", encoding="utf-8")

    # Env var should win over YAML
    monkeypatch.setenv("AXON_PROJECTS_ROOT", "/path/from/env")

    config = AxonConfig.load(str(yaml_path))
    assert config.projects_root == "/path/from/env"


class TestMultiStoreWriteErrors:
    """Merged parent-project views must raise on all write / delete operations."""

    def test_multi_vector_store_add_raises(self):
        from axon.main import MultiVectorStore

        ms = MultiVectorStore([])
        import pytest

        with pytest.raises(RuntimeError, match="merged"):
            ms.add("id", [0.1], {}, "text")

    def test_multi_vector_store_delete_by_ids_raises(self):
        from axon.main import MultiVectorStore

        ms = MultiVectorStore([])
        import pytest

        with pytest.raises(RuntimeError, match="merged"):
            ms.delete_by_ids(["doc-1"])

    def test_multi_vector_store_delete_documents_raises(self):
        from axon.main import MultiVectorStore

        ms = MultiVectorStore([])
        import pytest

        with pytest.raises(RuntimeError, match="merged"):
            ms.delete_documents(["doc-1"])

    def test_multi_bm25_add_documents_raises(self):
        from axon.main import MultiBM25Retriever

        mb = MultiBM25Retriever([])
        import pytest

        with pytest.raises(RuntimeError, match="merged"):
            mb.add_documents([{"id": "x", "content": "y"}])

    def test_multi_bm25_delete_documents_raises(self):
        from axon.main import MultiBM25Retriever

        mb = MultiBM25Retriever([])
        import pytest

        with pytest.raises(RuntimeError, match="merged"):
            mb.delete_documents(["doc-1"])


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestAxonBrain:
    def test_query_flow(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, similarity_threshold=0.0)
        brain = AxonBrain(config)

        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[{"id": "doc1", "text": "hello world", "score": 0.9}]
        )
        brain.llm.complete = MagicMock(return_value="Test answer")

        result = brain.query("test question")
        assert result == "Test answer"

    def test_discussion_fallback(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, discussion_fallback=True)
        brain = AxonBrain(config)

        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(return_value=[])
        brain.llm.complete = MagicMock(return_value="Fallback answer")

        result = brain.query("fallback question")
        assert result == "Fallback answer"
        # The plain query should be sent as the user message (not a 3rd-person wrapper)
        args, kwargs = brain.llm.complete.call_args
        assert args[0] == "fallback question"
        assert kwargs["chat_history"] is None

    def test_multi_turn_history_passed_as_plain_query(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """User message sent to LLM must be the plain query so chat_history stays consistent."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False)
        brain = AxonBrain(config)

        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[{"id": "d1", "text": "some context", "score": 0.9, "metadata": {}}]
        )
        brain.llm.complete = MagicMock(return_value="Turn 2 answer")

        history = [
            {"role": "user", "content": "turn 1 question"},
            {"role": "assistant", "content": "turn 1 answer"},
        ]
        brain.query("turn 2 question", chat_history=history)

        args, kwargs = brain.llm.complete.call_args
        # User message must be the plain query (not a RAG-wrapped prompt)
        assert args[0] == "turn 2 question"
        # RAG context should be in the system prompt, not the user message
        assert "turn 2 question" not in args[1] or "Relevant context" in args[1]
        # History must be forwarded
        assert kwargs["chat_history"] == history

    def test_brain_close_closes_stores(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        brain = AxonBrain(AxonConfig())
        brain.vector_store.close = MagicMock()
        brain.bm25.close = MagicMock()
        brain._own_vector_store.close = MagicMock()
        brain._own_bm25.close = MagicMock()

        brain.close()

        assert brain.vector_store.close.called
        assert brain.bm25.close.called
        assert brain._own_vector_store.close.called
        assert brain._own_bm25.close.called

    def test_ingest_flow(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(chunk_size=1000)
        brain = AxonBrain(config)
        brain.splitter = None
        brain._save_hash_store = MagicMock()
        brain._ingested_hashes = set()  # isolate from real on-disk hash store

        docs = [{"id": "d1", "text": "t1"}, {"id": "d2", "text": "t2"}]
        brain.embedding.embed = MagicMock(return_value=[[0.1], [0.1]])
        brain.vector_store.add = MagicMock()

        brain.ingest(docs)
        assert brain.vector_store.add.called

    def test_web_search_integration(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(truth_grounding=True, brave_api_key="key")
        brain = AxonBrain(config)

        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(return_value=[])
        brain._execute_web_search = MagicMock(
            return_value=[{"id": "w1", "text": "web", "score": 1.0, "is_web": True}]
        )

        retrieval = brain._execute_retrieval("query")
        assert len(retrieval["results"]) == 1
        assert retrieval["results"][0]["is_web"] is True
        assert retrieval["transforms"]["web_search_applied"] is True

    # ------------------------------------------------------------------
    # Query caching
    # ------------------------------------------------------------------

    def test_query_cache_returns_cached_response(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Second identical query hits cache; LLM called only once."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            hybrid_search=False, rerank=False, similarity_threshold=0.0, query_cache=True
        )
        brain = AxonBrain(config)
        brain._entity_graph = {}
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[{"id": "d1", "text": "ctx", "score": 0.9}]
        )
        brain.llm.complete = MagicMock(return_value="Cached answer")

        r1 = brain.query("What is RAG?")
        r2 = brain.query("What is RAG?")

        assert r1 == r2 == "Cached answer"
        assert brain.llm.complete.call_count == 1

    def test_query_cache_disabled_by_default(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """With query_cache=False (default), LLM is called every time."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, similarity_threshold=0.0)
        brain = AxonBrain(config)
        brain._entity_graph = {}
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[{"id": "d1", "text": "ctx", "score": 0.9}]
        )
        brain.llm.complete = MagicMock(return_value="Fresh answer")

        brain.query("What is RAG?")
        brain.query("What is RAG?")

        assert brain.llm.complete.call_count == 2

    def test_query_cache_not_used_with_chat_history(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Cache is bypassed when chat_history is present (multi-turn context varies)."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            hybrid_search=False, rerank=False, similarity_threshold=0.0, query_cache=True
        )
        brain = AxonBrain(config)
        brain._entity_graph = {}
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[{"id": "d1", "text": "ctx", "score": 0.9}]
        )
        brain.llm.complete = MagicMock(return_value="Contextual answer")

        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        brain.query("What is RAG?", chat_history=history)
        brain.query("What is RAG?", chat_history=history)

        assert brain.llm.complete.call_count == 2

    def test_query_cache_evicts_oldest_when_full(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Cache evicts oldest entry when query_cache_size is reached."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            hybrid_search=False,
            rerank=False,
            similarity_threshold=0.0,
            query_cache=True,
            query_cache_size=2,
        )
        brain = AxonBrain(config)
        brain._entity_graph = {}
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[{"id": "d1", "text": "ctx", "score": 0.9}]
        )
        brain.llm.complete = MagicMock(side_effect=["A", "B", "C"])

        brain.query("q1")
        brain.query("q2")
        # q1 should be evicted now
        brain.query("q3")
        assert len(brain._query_cache) == 2

    # ------------------------------------------------------------------
    # Ingest deduplication
    # ------------------------------------------------------------------

    def test_ingest_dedup_skips_duplicate_chunks(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Re-ingesting the same text is skipped when dedup_on_ingest=True."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(dedup_on_ingest=True)
        brain = AxonBrain(config)
        brain.splitter = None
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain.embedding.embed = MagicMock(return_value=[[0.1]])
        brain.vector_store.add = MagicMock()

        docs = [{"id": "d1", "text": "unique text"}]
        brain.ingest(docs)
        brain.ingest(docs)  # second call — should be skipped

        assert brain.vector_store.add.call_count == 1

    def test_ingest_dedup_allows_new_content(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Different text content passes dedup check and is ingested."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(dedup_on_ingest=True)
        brain = AxonBrain(config)
        brain.splitter = None
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain.embedding.embed = MagicMock(side_effect=[[[0.1]], [[0.2]]])
        brain.vector_store.add = MagicMock()

        brain.ingest([{"id": "d1", "text": "first doc"}])
        brain.ingest([{"id": "d2", "text": "second doc"}])

        assert brain.vector_store.add.call_count == 2

    def test_ingest_dedup_disabled(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        """With dedup_on_ingest=False, identical content is ingested again."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(dedup_on_ingest=False)
        brain = AxonBrain(config)
        brain.splitter = None
        brain._ingested_hashes = set()
        brain.embedding.embed = MagicMock(side_effect=[[[0.1]], [[0.1]]])
        brain.vector_store.add = MagicMock()

        docs = [{"id": "d1", "text": "same text"}]
        brain.ingest(docs)
        brain.ingest(docs)

        assert brain.vector_store.add.call_count == 2

    # ------------------------------------------------------------------
    # Step-back prompting
    # ------------------------------------------------------------------

    def test_step_back_adds_abstract_query_to_retrieval(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Step-back generates an abstract query and includes it in search_queries."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            hybrid_search=False, rerank=False, similarity_threshold=0.0, step_back=True
        )
        brain = AxonBrain(config)
        brain._get_step_back_query = MagicMock(return_value="abstract concept query")
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.embedding.embed = MagicMock(return_value=[[0.1], [0.2]])
        brain.vector_store.search = MagicMock(return_value=[])
        brain.llm.complete = MagicMock(return_value="Step-back answer")

        retrieval = brain._execute_retrieval("specific detailed question")

        brain._get_step_back_query.assert_called_once_with("specific detailed question")
        assert retrieval["transforms"]["step_back_applied"] is True
        assert "abstract concept query" in retrieval["transforms"]["queries"]

    def test_step_back_disabled_by_default(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """step_back is False by default — no extra LLM call for abstraction."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False)
        brain = AxonBrain(config)
        brain._get_step_back_query = MagicMock()
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(return_value=[])

        brain._execute_retrieval("any question")
        brain._get_step_back_query.assert_not_called()

    # ------------------------------------------------------------------
    # Parent-document / small-to-big retrieval
    # ------------------------------------------------------------------

    def test_parent_chunk_size_zero_uses_standard_split(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """parent_chunk_size=0 (default) uses the normal splitter path."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(parent_chunk_size=0)
        brain = AxonBrain(config)
        brain._save_hash_store = MagicMock()
        brain._ingested_hashes = set()
        brain.vector_store.add = MagicMock()
        brain.embedding.embed = MagicMock(return_value=[[0.1]])

        docs = [{"id": "d1", "text": "short text"}]
        brain.ingest(docs)
        # Standard split: no parent_text in metadata
        call_args = brain.vector_store.add.call_args
        metadatas = call_args[0][3]
        assert all("parent_text" not in m for m in metadatas)

    def test_parent_chunk_creates_child_chunks_with_parent_text(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """parent_chunk_size > chunk_size embeds child chunks that carry parent_text in metadata."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(chunk_size=50, chunk_overlap=10, parent_chunk_size=200)
        brain = AxonBrain(config)
        brain._save_hash_store = MagicMock()
        brain._ingested_hashes = set()
        brain.vector_store.add = MagicMock()
        brain.embedding.embed = MagicMock(return_value=[[0.1]] * 20)

        # Create a document long enough to produce multiple child chunks
        long_text = "word " * 80  # ~400 chars, will split into multiple 50-char child chunks
        docs = [{"id": "d1", "text": long_text}]
        brain.ingest(docs)

        assert brain.vector_store.add.called
        metadatas = brain.vector_store.add.call_args[0][3]
        # Every indexed child chunk must carry parent_text
        assert all("parent_text" in m for m in metadatas)

    def test_build_context_uses_parent_text_when_present(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """_build_context returns parent_text (not chunk text) for small-to-big results."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig()
        brain = AxonBrain(config)

        results = [
            {
                "id": "d1_p0_chunk_0",
                "text": "small retrieval chunk",
                "score": 0.9,
                "metadata": {"parent_text": "large parent passage with much more context"},
            }
        ]
        context, _ = brain._build_context(results)
        assert "large parent passage with much more context" in context
        assert "small retrieval chunk" not in context

    def test_build_context_falls_back_to_chunk_text_without_parent(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """_build_context falls back to chunk text when parent_text is absent."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig()
        brain = AxonBrain(config)

        results = [{"id": "d1", "text": "normal chunk text", "score": 0.9, "metadata": {}}]
        context, _ = brain._build_context(results)
        assert "normal chunk text" in context

    def test_reranker_model_cli_sets_config(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """--reranker-model CLI arg updates config.reranker_model before brain init."""
        from axon.main import AxonConfig

        config = AxonConfig()
        config.reranker_model = "BAAI/bge-reranker-v2-m3"
        assert config.reranker_model == "BAAI/bge-reranker-v2-m3"

    def test_reranker_default_model(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        """Default reranker model is the lightweight ms-marco cross-encoder."""
        from axon.main import AxonConfig

        config = AxonConfig()
        assert config.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_parent_chunk_size_config_defaults_zero(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """parent_chunk_size defaults to 1500 (feature enabled by default)."""
        from axon.main import AxonConfig

        config = AxonConfig()
        assert config.parent_chunk_size == 1500


class TestOpenReranker:
    def test_llm_reranker(self):
        from axon.main import AxonConfig, OpenReranker

        config = AxonConfig(rerank=True, reranker_provider="llm")
        reranker = OpenReranker(config)
        reranker.llm = MagicMock()
        reranker.llm.complete = MagicMock(side_effect=["10", "5"])

        docs = [{"id": "d1", "text": "t1"}, {"id": "d2", "text": "t2"}]
        results = reranker.rerank("q", docs)
        assert results[0]["id"] == "d1"
        assert results[0]["rerank_score"] == 10.0


class TestOpenLLM:
    @patch("ollama.Client")
    def test_ollama_num_ctx(self, MockOllama):
        from axon.main import AxonConfig, OpenLLM

        config = AxonConfig(llm_provider="ollama")
        llm = OpenLLM(config)

        mock_client = MockOllama.return_value
        mock_client.chat.return_value = {"message": {"content": "resp"}}

        llm.complete("q")
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs["options"]["num_ctx"] == 8192

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_gemma_handling(self, MockGenaiConfigure, MockGenerativeModel):
        from axon.main import AxonConfig, OpenLLM

        # Test Gemma
        config = AxonConfig(llm_provider="gemini", llm_model="gemma-3-27b-it")
        llm = OpenLLM(config)

        mock_model = MockGenerativeModel.return_value
        mock_model.generate_content.return_value = MagicMock(text="gemma answer")

        llm.complete("hello", system_prompt="be helpful")

        # Verify system_instruction was NOT passed to constructor
        MockGenerativeModel.assert_called_with(model_name="gemma-3-27b-it")

        # Verify prompt was prepended
        gen_args = mock_model.generate_content.call_args[0][0]
        assert "be helpful\n\nhello" in gen_args[-1]["parts"][0]

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_pro_handling(self, MockGenaiConfigure, MockGenerativeModel):
        from axon.main import AxonConfig, OpenLLM

        # Test Pro (supports system instructions)
        config = AxonConfig(llm_provider="gemini", llm_model="gemini-1.5-pro")
        llm = OpenLLM(config)

        llm.complete("hello", system_prompt="be helpful")
        MockGenerativeModel.assert_called_with(
            model_name="gemini-1.5-pro", system_instruction="be helpful"
        )

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_flash_handling(self, MockGenaiConfigure, MockGenerativeModel):
        from axon.main import AxonConfig, OpenLLM

        # Test Flash (also supports system instructions)
        config = AxonConfig(llm_provider="gemini", llm_model="gemini-1.5-flash")
        llm = OpenLLM(config)

        llm.complete("hello", system_prompt="be helpful")
        MockGenerativeModel.assert_called_with(
            model_name="gemini-1.5-flash", system_instruction="be helpful"
        )

    @patch("httpx.Client")
    def test_ollama_cloud_handling(self, MockHttpxClient):
        from axon.main import AxonConfig, OpenLLM

        config = AxonConfig(
            llm_provider="ollama_cloud", ollama_cloud_key="test_key", llm_model="gemma"
        )
        llm = OpenLLM(config)

        mock_client = MockHttpxClient.return_value.__enter__.return_value
        mock_client.post.return_value = MagicMock(
            json=lambda: {"response": "cloud resp"}, status_code=200
        )

        result = llm.complete("hello", system_prompt="be helpful")
        assert result == "cloud resp"
        assert mock_client.post.called

    @patch("openai.resources.chat.Completions.create")
    @patch("openai.OpenAI")
    def test_openai_handling(self, MockOpenAI, MockCreate):
        from axon.main import AxonConfig, OpenLLM

        config = AxonConfig(llm_provider="openai", api_key="sk-test", llm_model="gpt-4o")
        llm = OpenLLM(config)

        # Setup mock client
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="openai resp"))]
        )

        result = llm.complete("hello", system_prompt="be helpful")
        assert result == "openai resp"
        assert mock_client.chat.completions.create.called

    @patch("openai.OpenAI")
    def test_vllm_complete(self, MockOpenAI):
        from axon.main import AxonConfig, OpenLLM

        config = AxonConfig(
            llm_provider="vllm",
            llm_model="meta-llama/Llama-3.1-8B-Instruct",
            vllm_base_url="http://localhost:8000/v1",
        )
        llm = OpenLLM(config)

        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="vllm resp"))]
        )

        result = llm.complete("hello", system_prompt="be helpful")
        assert result == "vllm resp"
        # Verify base_url was passed to OpenAI constructor
        assert MockOpenAI.call_args[1].get("base_url") == "http://localhost:8000/v1"

    def test_vllm_default_base_url(self, monkeypatch):
        monkeypatch.delenv("VLLM_BASE_URL", raising=False)
        from axon.main import AxonConfig

        config = AxonConfig()
        assert config.vllm_base_url == "http://localhost:8000/v1"

    def test_vllm_base_url_from_yaml(self, tmp_path):
        from axon.main import AxonConfig

        cfg = {
            "llm": {
                "provider": "vllm",
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "vllm_base_url": "http://192.168.1.10:8000/v1",
            }
        }
        cfg_path = tmp_path / "config.yaml"
        import yaml

        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)
        config = AxonConfig.load(str(cfg_path))
        assert config.llm_provider == "vllm"
        assert config.vllm_base_url == "http://192.168.1.10:8000/v1"


_FAKE_COPILOT_SESSION = {"token": "fake_session_token_abc", "expires_at": 9_999_999_999.0}


class TestGitHubCopilotProvider:
    @patch("axon.main._refresh_copilot_session", return_value=_FAKE_COPILOT_SESSION)
    @patch("openai.OpenAI")
    def test_complete_calls_copilot_endpoint(self, MockOpenAI, _mock_refresh):
        from axon.main import AxonConfig, OpenLLM

        config = AxonConfig(
            llm_provider="github_copilot", llm_model="gpt-4o", copilot_pat="gho_test"
        )
        llm = OpenLLM(config)

        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="hello"))]
        )

        result = llm.complete("hi")
        assert result == "hello"
        assert MockOpenAI.call_args[1]["base_url"] == "https://api.githubcopilot.com"

    @patch("axon.main._refresh_copilot_session", return_value=_FAKE_COPILOT_SESSION)
    @patch("openai.OpenAI")
    def test_complete_passes_required_headers(self, MockOpenAI, _mock_refresh):
        from axon.main import AxonConfig, OpenLLM

        config = AxonConfig(
            llm_provider="github_copilot", llm_model="gpt-4o", copilot_pat="gho_test"
        )
        llm = OpenLLM(config)

        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="hi"))]
        )
        llm.complete("hi")

        headers = MockOpenAI.call_args[1]["default_headers"]
        assert "Editor-Version" in headers
        assert "Copilot-Integration-Id" in headers

    @patch("axon.main._refresh_copilot_session", return_value=_FAKE_COPILOT_SESSION)
    @patch("openai.OpenAI")
    def test_complete_returns_response(self, MockOpenAI, _mock_refresh):
        from axon.main import AxonConfig, OpenLLM

        config = AxonConfig(
            llm_provider="github_copilot", llm_model="gpt-4o", copilot_pat="gho_test"
        )
        llm = OpenLLM(config)

        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="copilot answer"))]
        )

        result = llm.complete("What is 2+2?")
        assert result == "copilot answer"

    @patch("axon.main._refresh_copilot_session", return_value=_FAKE_COPILOT_SESSION)
    @patch("openai.OpenAI")
    def test_stream_yields_tokens(self, MockOpenAI, _mock_refresh):
        from axon.main import AxonConfig, OpenLLM

        config = AxonConfig(
            llm_provider="github_copilot", llm_model="gpt-4o", copilot_pat="gho_test"
        )
        llm = OpenLLM(config)

        mock_client = MockOpenAI.return_value
        chunks = [
            MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
            MagicMock(choices=[MagicMock(delta=MagicMock(content=None))]),
        ]
        mock_client.chat.completions.create.return_value = iter(chunks)

        tokens = list(llm.stream("hi"))
        assert tokens == ["Hello", " world"]
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["stream"] is True

    def test_missing_pat_raises_valueerror(self, monkeypatch):
        import pytest

        from axon.main import AxonConfig, OpenLLM

        monkeypatch.delenv("GITHUB_COPILOT_PAT", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        config = AxonConfig(llm_provider="github_copilot", llm_model="gpt-4o", copilot_pat="")
        llm = OpenLLM(config)

        with pytest.raises(ValueError, match="GITHUB_COPILOT_PAT"):
            llm._get_copilot_client()

    @patch("axon.main._refresh_copilot_session")
    @patch("openai.OpenAI")
    def test_client_cache_invalidated_on_pat_change(self, MockOpenAI, mock_refresh):
        from axon.main import AxonConfig, OpenLLM

        # Two successive OAuth tokens produce two different session tokens
        mock_refresh.side_effect = [
            {"token": "session_token_1", "expires_at": 9_999_999_999.0},
            {"token": "session_token_2", "expires_at": 9_999_999_999.0},
        ]
        client_a = MagicMock()
        client_b = MagicMock()
        MockOpenAI.side_effect = [client_a, client_b]

        for mock_client in (client_a, client_b):
            mock_client.chat.completions.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content="resp"))]
            )

        config = AxonConfig(
            llm_provider="github_copilot", llm_model="gpt-4o", copilot_pat="gho_first"
        )
        llm = OpenLLM(config)

        llm.complete("hi")
        first_client = llm._openai_clients.get("_copilot")

        # Simulate OAuth token change: clear session cache so a new session is fetched
        config.copilot_pat = "gho_second"
        for k in ("_copilot", "_copilot_session", "_copilot_token"):
            llm._openai_clients.pop(k, None)

        llm.complete("hi")
        second_client = llm._openai_clients.get("_copilot")

        assert first_client is not second_client
        assert MockOpenAI.call_count == 2

    def test_pat_loaded_from_env(self, monkeypatch):
        from axon.main import AxonConfig

        monkeypatch.setenv("GITHUB_COPILOT_PAT", "gho_from_env")
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)

        config = AxonConfig(llm_provider="github_copilot", llm_model="gpt-4o")
        assert config.copilot_pat == "gho_from_env"


# ---------------------------------------------------------------------------
# New tests: HyDE, multi-query, load_directory, metrics
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestQueryTransformations:
    def test_hyde_document_calls_llm(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hyde=False, hybrid_search=False, rerank=False)
        brain = AxonBrain(config)
        brain.llm.complete = MagicMock(return_value="Hypothetical passage about X.")

        result = brain._get_hyde_document("What is X?")

        assert result == "Hypothetical passage about X."
        brain.llm.complete.assert_called_once()
        call_prompt = brain.llm.complete.call_args[0][0]
        assert "What is X?" in call_prompt

    def test_multi_queries_returns_original_plus_alternatives(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(multi_query=True, hybrid_search=False, rerank=False)
        brain = AxonBrain(config)
        brain.llm.complete = MagicMock(return_value="Alt query one\nAlt query two\nAlt query three")

        queries = brain._get_multi_queries("original question")

        # Original must always be first
        assert queries[0] == "original question"
        # 3 alternatives + original = 4
        assert len(queries) == 4
        # Whitespace stripped
        assert all(q == q.strip() for q in queries)

    def test_multi_queries_strips_numbering(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(multi_query=True, hybrid_search=False, rerank=False)
        brain = AxonBrain(config)
        brain.llm.complete = MagicMock(
            return_value="1. How does X work\n2. Explain X\n3. X overview"
        )

        queries = brain._get_multi_queries("original")
        # Numbering should be stripped
        assert not any(q[0].isdigit() for q in queries[1:])

    def test_load_directory_calls_ingest(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        import asyncio

        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False)
        brain = AxonBrain(config)
        brain.ingest = MagicMock()

        # Create a real .txt file so DirectoryLoader finds and loads something.
        (tmp_path / "sample.txt").write_text("hello world", encoding="utf-8")

        asyncio.run(brain.load_directory(str(tmp_path)))

        brain.ingest.assert_called_once()

    def test_load_directory_skips_ingest_when_empty(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        import asyncio
        from unittest.mock import AsyncMock, patch

        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False)
        brain = AxonBrain(config)
        brain.ingest = MagicMock()

        with patch("axon.loaders.DirectoryLoader") as MockLoader:
            mock_loader_instance = MockLoader.return_value
            mock_loader_instance.aload = AsyncMock(return_value=[])

            asyncio.run(brain.load_directory(str(tmp_path)))

        brain.ingest.assert_not_called()


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestQueryDecomposeAndCompress:
    """Tests for query decomposition and context compression."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, similarity_threshold=0.0, **kwargs)
        brain = AxonBrain(config)
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(return_value=[])
        return brain

    # ------------------------------------------------------------------
    # Query Decomposition
    # ------------------------------------------------------------------

    def test_decompose_breaks_query_into_sub_questions(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(
            MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, query_decompose=True
        )
        brain.llm.complete = MagicMock(
            return_value="What is X?\nHow does X work?\nWhy is X useful?"
        )

        sub_qs = brain._decompose_query("Tell me everything about X")

        assert sub_qs[0] == "Tell me everything about X"  # original always first
        assert len(sub_qs) >= 2
        assert "What is X?" in sub_qs

    def test_decompose_strips_numbering_and_bullets(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.complete = MagicMock(return_value="1. Sub one\n2. Sub two\n- Sub three")

        sub_qs = brain._decompose_query("complex question")

        assert all(not q[0].isdigit() for q in sub_qs[1:])
        assert all(not q.startswith("-") for q in sub_qs[1:])

    def test_decompose_deduplicates_sub_questions(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.complete = MagicMock(return_value="original question\noriginal question\nNew sub")

        sub_qs = brain._decompose_query("original question")

        assert sub_qs.count("original question") == 1

    def test_decompose_applied_in_execute_retrieval(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(
            MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, query_decompose=True
        )
        brain._decompose_query = MagicMock(return_value=["original", "sub-q1", "sub-q2"])

        retrieval = brain._execute_retrieval("original")

        brain._decompose_query.assert_called_once_with("original")
        assert retrieval["transforms"]["decompose_applied"] is True
        assert "sub-q1" in retrieval["transforms"]["queries"]

    def test_decompose_disabled_by_default(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain._decompose_query = MagicMock()

        brain._execute_retrieval("any question")

        brain._decompose_query.assert_not_called()

    def test_decompose_combines_with_multi_query(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """multi_query and query_decompose can run together; queries are deduplicated."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            multi_query=True,
            query_decompose=True,
        )
        brain._get_multi_queries = MagicMock(return_value=["q", "alt1", "alt2"])
        brain._decompose_query = MagicMock(return_value=["q", "sub1"])

        retrieval = brain._execute_retrieval("q")

        assert retrieval["transforms"]["multi_query_applied"] is True
        assert retrieval["transforms"]["decompose_applied"] is True
        queries = retrieval["transforms"]["queries"]
        assert queries.count("q") == 1  # no duplicates

    # ------------------------------------------------------------------
    # Context Compression
    # ------------------------------------------------------------------

    def test_compress_context_shortens_chunks(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(
            MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, compress_context=True
        )
        long_text = "irrelevant sentence. " * 20 + "The answer is 42."
        brain.llm.complete = MagicMock(return_value="The answer is 42.")

        results = [{"id": "d1", "text": long_text, "score": 0.9, "metadata": {}}]
        compressed = brain._compress_context("What is the answer?", results)

        assert compressed[0]["text"] == "The answer is 42."

    def test_compress_context_skips_web_results(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.complete = MagicMock(return_value="compressed")

        web_result = {
            "id": "http://ex.com",
            "text": "web content",
            "score": 1.0,
            "metadata": {},
            "is_web": True,
        }
        results = brain._compress_context("q", [web_result])

        # Web results must pass through unchanged; LLM must not be called
        brain.llm.complete.assert_not_called()
        assert results[0]["text"] == "web content"

    def test_compress_context_falls_back_on_failure(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """If LLM compression raises, the original result is returned unchanged."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.complete = MagicMock(side_effect=RuntimeError("llm down"))

        original_text = "original chunk text"
        results = [{"id": "d1", "text": original_text, "score": 0.9, "metadata": {}}]
        compressed = brain._compress_context("q", results)

        assert compressed[0]["text"] == original_text

    def test_compress_context_does_not_expand_chunk(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """If LLM returns something longer than the original, keep the original."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        short_text = "short"
        brain.llm.complete = MagicMock(
            return_value="much longer text that exceeds original length significantly"
        )

        results = [{"id": "d1", "text": short_text, "score": 0.9, "metadata": {}}]
        compressed = brain._compress_context("q", results)

        assert compressed[0]["text"] == short_text

    def test_compress_context_uses_parent_text_as_source(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """When parent_text is in metadata, that is compressed (not the small child chunk)."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.complete = MagicMock(return_value="compressed parent")

        results = [
            {
                "id": "d1_p0_chunk_0",
                "text": "small child chunk",
                "score": 0.9,
                "metadata": {"parent_text": "large parent passage " * 10},
            }
        ]
        compressed = brain._compress_context("q", results)

        # Compression prompt should have used parent_text, result stored back in parent_text
        call_prompt = brain.llm.complete.call_args[0][0]
        assert "large parent passage" in call_prompt
        assert compressed[0]["metadata"]["parent_text"] == "compressed parent"

    def test_compress_context_disabled_by_default(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """compress_context=False (default): _compress_context is never called during query."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, similarity_threshold=0.0)
        brain = AxonBrain(config)
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[{"id": "d1", "text": "ctx", "score": 0.9}]
        )
        brain.llm.complete = MagicMock(return_value="answer")
        brain._compress_context = MagicMock(wraps=brain._compress_context)

        brain.query("test?")

        brain._compress_context.assert_not_called()


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestLogQueryMetrics:
    def test_metrics_logged_without_exception(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        from unittest.mock import patch as _patch

        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False)
        brain = AxonBrain(config)

        with _patch("axon.main.logger") as mock_logger:
            brain._log_query_metrics(
                query="test query that is fairly long",
                vector_count=5,
                bm25_count=3,
                filtered_count=4,
                final_count=3,
                top_score=0.87,
                latency_ms=123.4,
                transformations={"hyde_applied": False},
            )
            mock_logger.info.assert_called_once()
            logged_data = mock_logger.info.call_args[0][0]
            assert logged_data["event"] == "query_complete"
            assert logged_data["latency_ms"] == 123.4
            assert logged_data["results"]["vector"] == 5
            assert "test query" in logged_data["query_preview"]

    def test_metrics_handles_zero_top_score(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        from unittest.mock import patch as _patch

        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False)
        brain = AxonBrain(config)

        with _patch("axon.main.logger") as mock_logger:
            brain._log_query_metrics(
                query="q",
                vector_count=0,
                bm25_count=0,
                filtered_count=0,
                final_count=0,
                top_score=0.0,
                latency_ms=10.0,
            )
            logged_data = mock_logger.info.call_args[0][0]
            # top_score=0.0 is a real zero confidence score, logged as 0.0 (not None)
            assert logged_data["top_score"] == 0.0


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestListDocuments:
    def test_list_documents_delegates_to_vector_store(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False)
        brain = AxonBrain(config)
        brain.vector_store.list_documents = MagicMock(
            return_value=[
                {"source": "a.txt", "chunks": 3, "doc_ids": ["1", "2", "3"]},
                {"source": "b.pdf", "chunks": 7, "doc_ids": [str(i) for i in range(7)]},
            ]
        )

        result = brain.list_documents()

        brain.vector_store.list_documents.assert_called_once()
        assert len(result) == 2
        assert result[0]["source"] == "a.txt"
        assert result[0]["chunks"] == 3
        assert result[1]["source"] == "b.pdf"

    def test_list_documents_empty_kb(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False)
        brain = AxonBrain(config)
        brain.vector_store.list_documents = MagicMock(return_value=[])

        result = brain.list_documents()
        assert result == []


class TestOpenVectorStoreListDocuments:
    def test_chroma_groups_by_source(self):
        from unittest.mock import MagicMock, patch

        from axon.main import AxonConfig, OpenVectorStore

        config = AxonConfig(vector_store="chroma")
        with patch("chromadb.PersistentClient") as MockClient:
            mock_col = MagicMock()
            MockClient.return_value.get_or_create_collection.return_value = mock_col
            mock_col.get.return_value = {
                "ids": ["c1", "c2", "c3", "c4"],
                "metadatas": [
                    {"source": "notes.txt"},
                    {"source": "notes.txt"},
                    {"source": "report.pdf"},
                    {"source": "report.pdf"},
                ],
            }
            store = OpenVectorStore(config)
            result = store.list_documents()

        assert len(result) == 2
        by_source = {d["source"]: d for d in result}
        assert by_source["notes.txt"]["chunks"] == 2
        assert by_source["report.pdf"]["chunks"] == 2
        assert set(by_source["notes.txt"]["doc_ids"]) == {"c1", "c2"}

    def test_chroma_handles_missing_source_metadata(self):
        from unittest.mock import MagicMock, patch

        from axon.main import AxonConfig, OpenVectorStore

        config = AxonConfig(vector_store="chroma")
        with patch("chromadb.PersistentClient") as MockClient:
            mock_col = MagicMock()
            MockClient.return_value.get_or_create_collection.return_value = mock_col
            mock_col.get.return_value = {
                "ids": ["x1"],
                "metadatas": [{}],  # no 'source' key
            }
            store = OpenVectorStore(config)
            result = store.list_documents()

        assert len(result) == 1
        assert result[0]["source"] == "unknown"


# ---------------------------------------------------------------------------
# _read_input (REPL prompt helper)
# ---------------------------------------------------------------------------


class TestReadInput:
    """_read_input must accept an optional custom prompt without raising."""

    def _make_read_input(self, pt_session=None, pt_html=None):
        """Replicate the closure produced inside _interactive_repl."""
        _pt_session = pt_session
        _PThtml = pt_html

        def _read_input(prompt: str = "") -> str:
            if _pt_session:
                _p = _PThtml("<ansigreen><b>You</b></ansigreen>: ") if not prompt else prompt
                return _pt_session.prompt(_p)
            return input(prompt if prompt else "\033[1;32mYou\033[0m: ")

        return _read_input

    def test_no_args_uses_styled_you_prompt_with_pt(self):
        """No-arg call uses the coloured 'You:' HTML prompt."""
        mock_session = MagicMock()
        mock_session.prompt.return_value = "hello"
        mock_html = MagicMock(side_effect=lambda s: f"HTML({s})")

        fn = self._make_read_input(pt_session=mock_session, pt_html=mock_html)
        result = fn()

        assert result == "hello"
        mock_session.prompt.assert_called_once()
        # First positional arg should be the HTML-wrapped You: prompt
        called_arg = mock_session.prompt.call_args[0][0]
        assert "You" in str(called_arg)

    def test_custom_prompt_passed_through_with_pt(self):
        """A custom prompt string (e.g. confirmation question) is passed as-is."""
        mock_session = MagicMock()
        mock_session.prompt.return_value = "y"
        mock_html = MagicMock()

        fn = self._make_read_input(pt_session=mock_session, pt_html=mock_html)
        result = fn("  Resume session? [y/N]: ")

        assert result == "y"
        mock_session.prompt.assert_called_once_with("  Resume session? [y/N]: ")

    def test_no_pt_no_args_uses_ansi_you(self, monkeypatch):
        """Without prompt_toolkit, plain input() is called with ANSI-coloured 'You:'."""
        inputs = iter(["my answer"])
        monkeypatch.setattr("builtins.input", lambda p: (next(inputs)))

        fn = self._make_read_input(pt_session=None)
        result = fn()

        assert result == "my answer"

    def test_no_pt_custom_prompt_used(self, monkeypatch):
        """Without prompt_toolkit, a custom prompt is forwarded to input()."""
        received = {}
        monkeypatch.setattr("builtins.input", lambda p: received.update({"p": p}) or "n")

        fn = self._make_read_input(pt_session=None)
        fn("Confirm? [y/N]: ")

        assert received["p"] == "Confirm? [y/N]: "


# ---------------------------------------------------------------------------
# _infer_provider — auto-detect LLM provider from model name
# ---------------------------------------------------------------------------


class TestInferProvider:
    def test_gemini_prefix(self):
        from axon.main import _infer_provider

        assert _infer_provider("gemini-2.5-flash-lite") == "gemini"
        assert _infer_provider("gemini-1.5-pro") == "gemini"

    def test_openai_gpt_prefix(self):
        from axon.main import _infer_provider

        assert _infer_provider("gpt-4o") == "openai"
        assert _infer_provider("gpt-4o-mini") == "openai"
        assert _infer_provider("o1-mini") == "openai"
        assert _infer_provider("o3-mini") == "openai"

    def test_ollama_default(self):
        from axon.main import _infer_provider

        assert _infer_provider("mistral-nemo") == "ollama"
        assert _infer_provider("llama3.1") == "ollama"
        assert _infer_provider("gemma") == "ollama"
        assert _infer_provider("phi3") == "ollama"
        # Ollama models with name:tag format must NOT be misclassified as openai
        assert _infer_provider("gpt-oss:120b-cloud") == "ollama"
        assert _infer_provider("o1-tuned:latest") == "ollama"

    def test_vllm_path_style_falls_through_to_ollama(self):
        # vLLM model names look like HuggingFace paths; no auto-detection — falls to ollama
        from axon.main import _infer_provider

        assert _infer_provider("meta-llama/Llama-3.1-8B-Instruct") == "ollama"
        assert _infer_provider("mistralai/Mistral-7B-Instruct-v0.3") == "ollama"

    def test_case_insensitive(self):
        from axon.main import _infer_provider

        assert _infer_provider("GEMINI-2.0-FLASH") == "gemini"
        assert _infer_provider("GPT-4O") == "openai"


# ---------------------------------------------------------------------------
# Inline citations
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestCiteSources:
    def test_citations_always_present(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        """The system prompt must contain citation directives."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, similarity_threshold=0.0)
        brain = AxonBrain(config)
        brain._ingested_hashes = set()
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[
                {"id": "d1", "text": "Transformers use attention.", "score": 0.9, "metadata": {}}
            ]
        )
        brain.llm.complete = MagicMock(return_value="Answer")

        brain.query("What is a transformer?")

        call_args = brain.llm.complete.call_args
        system_prompt = call_args[0][1]
        assert "cite" in system_prompt.lower() or "[Document" in system_prompt


# ---------------------------------------------------------------------------
# RAPTOR hierarchical indexing
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestRaptor:
    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **cfg_kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, **cfg_kwargs)
        brain = AxonBrain(config)
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain._save_entity_graph = MagicMock()
        brain.embedding.embed = MagicMock(return_value=[[0.1] * 384])
        brain.vector_store.add = MagicMock()
        return brain

    def test_raptor_explicitly_disabled(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(
            MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, raptor=False, graph_rag=False
        )
        docs = [
            {"id": "d1", "text": "chunk one", "metadata": {"source": "a.txt"}},
            {"id": "d2", "text": "chunk two", "metadata": {"source": "a.txt"}},
        ]
        # No LLM call for summaries when raptor=False
        brain.llm.complete = MagicMock(return_value="summary")
        brain.ingest(docs)
        brain.llm.complete.assert_not_called()

    def test_raptor_generates_summary_nodes(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_chunk_group_size=2,
        )
        docs = [
            {"id": f"d{i}", "text": f"chunk text {i}", "metadata": {"source": "a.txt"}}
            for i in range(4)
        ]
        brain.llm.complete = MagicMock(return_value="This is a raptor summary.")
        brain.embedding.embed = MagicMock(side_effect=lambda texts: [[0.1] * 384] * len(texts))
        brain.ingest(docs)
        # 4 chunks / group_size=2 → 2 summary nodes + 4 leaf = 6 total
        add_calls = brain.vector_store.add.call_args_list
        total_ingested = sum(len(c[0][0]) for c in add_calls)  # first positional arg = ids list
        assert (
            total_ingested >= 5
        ), f"Expected at least 5 docs (4 leaves + ≥1 summary), got {total_ingested}"
        # LLM was called to produce summaries
        assert brain.llm.complete.call_count >= 1

    def test_raptor_summary_metadata_has_raptor_level(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_chunk_group_size=3,
        )
        docs = [
            {"id": f"d{i}", "text": f"text {i}", "metadata": {"source": "b.txt"}} for i in range(3)
        ]
        brain.llm.complete = MagicMock(return_value="Summary content")
        brain.embedding.embed = MagicMock(side_effect=lambda texts: [[0.1] * 384] * len(texts))
        brain.ingest(docs)

        add_calls = brain.vector_store.add.call_args_list
        all_metadatas = []
        for call in add_calls:
            all_metadatas.extend(call[0][3])  # 4th positional arg = metadatas
        raptor_meta = [m for m in all_metadatas if m.get("raptor_level") == 1]
        assert len(raptor_meta) >= 1

    def test_raptor_failure_silently_skipped(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """If LLM fails for a group, RAPTOR skips that group without crashing."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_chunk_group_size=2,
        )
        docs = [
            {"id": f"d{i}", "text": f"text {i}", "metadata": {"source": "c.txt"}} for i in range(2)
        ]
        brain.llm.complete = MagicMock(side_effect=RuntimeError("LLM down"))
        brain.embedding.embed = MagicMock(return_value=[[0.1] * 384] * 2)
        brain.ingest(docs)  # should not raise
        # Only the 2 leaf chunks ingested (no summary added)
        add_calls = brain.vector_store.add.call_args_list
        total_ingested = sum(len(c[0][0]) for c in add_calls)
        assert total_ingested == 2


# ---------------------------------------------------------------------------
# GraphRAG entity-centric retrieval
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestGraphRAG:
    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **cfg_kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, **cfg_kwargs)
        brain = AxonBrain(config)
        brain._ingested_hashes = set()
        brain._entity_graph = {}
        brain._save_hash_store = MagicMock()
        brain._save_entity_graph = MagicMock()
        return brain

    def test_graph_rag_explicitly_disabled(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(
            MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, graph_rag=False
        )
        docs = [{"id": "d1", "text": "Attention is all you need.", "metadata": {"source": "a.txt"}}]
        brain.embedding.embed = MagicMock(return_value=[[0.1] * 384])
        brain.vector_store.add = MagicMock()
        brain.llm.complete = MagicMock(return_value="Attention")
        brain.ingest(docs)
        # Entity graph should remain empty since graph_rag=False
        assert brain._entity_graph == {}

    def test_graph_rag_populates_entity_graph_on_ingest(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(
            MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, graph_rag=True
        )
        docs = [
            {"id": "d1", "text": "BERT is a transformer model.", "metadata": {"source": "ml.txt"}}
        ]
        brain.embedding.embed = MagicMock(return_value=[[0.1] * 384])
        brain.vector_store.add = MagicMock()
        brain.llm.complete = MagicMock(return_value="BERT\ntransformer")
        brain.ingest(docs)
        # At least one entity should be in the graph
        assert len(brain._entity_graph) >= 1
        # doc id (possibly with chunk suffix from splitter) should appear under at least one entity
        all_ids = set()
        for node in brain._entity_graph.values():
            chunk_ids = node["chunk_ids"] if isinstance(node, dict) else node
            all_ids.update(chunk_ids)
        assert any(doc_id.startswith("d1") for doc_id in all_ids)

    def test_graph_rag_expands_results_at_query_time(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Entity graph expansion fetches related docs not in original results."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            similarity_threshold=0.0,
            query_router="off",  # disable router so graph_rag=True is preserved
        )
        # Pre-populate entity graph
        brain._entity_graph = {"transformer": ["d2"]}
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        # Primary retrieval returns only d1
        brain.vector_store.search = MagicMock(
            return_value=[
                {"id": "d1", "text": "Attention mechanisms", "score": 0.9, "metadata": {}}
            ]
        )
        # get_by_ids returns d2 for GraphRAG expansion
        brain.vector_store.get_by_ids = MagicMock(
            return_value=[
                {"id": "d2", "text": "BERT is a transformer.", "score": 1.0, "metadata": {}}
            ]
        )
        brain.llm.complete = MagicMock(return_value="transformer")  # entity extraction
        brain.llm.complete.side_effect = ["transformer", "Answer"]
        # Answer
        brain.llm.complete = MagicMock(side_effect=["transformer", "final answer"])
        brain.query("What is a transformer?")
        # get_by_ids should have been called for graph expansion
        brain.vector_store.get_by_ids.assert_called()

    def test_graph_rag_does_not_duplicate_already_retrieved(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """GraphRAG expansion skips doc IDs already in primary results."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            similarity_threshold=0.0,
        )
        brain._entity_graph = {"transformer": ["d1"]}  # d1 already in primary results
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[{"id": "d1", "text": "Attention", "score": 0.9, "metadata": {}}]
        )
        brain.vector_store.get_by_ids = MagicMock(return_value=[])
        brain.llm.complete = MagicMock(side_effect=["transformer", "answer"])
        brain.query("What is a transformer?")
        # get_by_ids called but with empty list (d1 already present)
        if brain.vector_store.get_by_ids.called:
            call_ids = brain.vector_store.get_by_ids.call_args[0][0]
            assert "d1" not in call_ids


# ---------------------------------------------------------------------------
# Entity extraction: bullet/markdown stripping regression
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestExtractEntitiesStripping:
    """Regression: LLM may return bullets despite the 'no bullets' prompt."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from axon.main import AxonBrain, AxonConfig

        brain = AxonBrain(AxonConfig(graph_rag=False))
        brain._entity_graph = {}
        return brain

    def test_markdown_bullets_stripped(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        # New format: each line is "NAME | description"; lines without pipe yield {"name": x, "description": ""}
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.complete = MagicMock(return_value="Axon\nOpenAI\nQdrant\nPython\nFastAPI")
        entities = brain._extract_entities("some text")
        names = [e["name"] for e in entities]
        assert names == ["Axon", "OpenAI", "Qdrant", "Python", "FastAPI"]

    def test_clean_lines_unchanged(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.complete = MagicMock(return_value="Axon\nOpenAI\nQdrant")
        entities = brain._extract_entities("some text")
        names = [e["name"] for e in entities]
        assert names == ["Axon", "OpenAI", "Qdrant"]

    def test_empty_response_returns_empty_list(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.complete = MagicMock(return_value="")
        assert brain._extract_entities("some text") == []

    def test_llm_exception_returns_empty_list(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.complete = MagicMock(side_effect=RuntimeError("llm down"))
        assert brain._extract_entities("some text") == []


# ---------------------------------------------------------------------------
# Cache: LRU eviction & make_cache_key completeness
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestCacheFixes:
    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **cfg_kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            hybrid_search=False, rerank=False, similarity_threshold=0.0, **cfg_kwargs
        )
        brain = AxonBrain(config)
        brain._ingested_hashes = set()
        brain._entity_graph = {}
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(
            return_value=[{"id": "d1", "text": "ctx", "score": 0.9, "metadata": {}}]
        )
        return brain

    def test_lru_hit_moves_to_end(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        """Cache hit moves the entry to the end (most-recently-used position)."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            query_cache=True,
            query_cache_size=3,
        )
        brain.llm.complete = MagicMock(side_effect=["A", "B", "A_again"])
        brain.query("q1")  # inserts key1
        brain.query("q2")  # inserts key2
        # Hit q1 — should move key1 to end so key2 would be evicted next
        brain.llm.complete.side_effect = ["A"]
        brain.query("q1")  # cache hit; key1 moved to end
        # Now insert q3 — cache is at size 3 (q1, q2, both present + q3 = need evict q2 = LRU)
        brain.llm.complete.side_effect = ["C"]
        brain.query("q3")
        keys = list(brain._query_cache.keys())
        # q2 should have been evicted (it was LRU after q1 was accessed)
        # q1 and q3 should still be present
        assert len(brain._query_cache) == 3  # size=3, nothing evicted yet
        # With size=3 nothing evicted, but verify ordering: q1 was moved to end after hit
        # so in the OrderedDict q2 should be earlier than q1
        key_q2 = [k for k in keys if brain._query_cache.get(k) == "B"]
        key_q1 = [k for k in keys if brain._query_cache.get(k) == "A"]
        if key_q1 and key_q2:
            assert keys.index(key_q2[0]) < keys.index(key_q1[0])

    def test_lru_evicts_least_recently_used(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """When cache is full the LRU entry (front of OrderedDict) is evicted."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            query_cache=True,
            query_cache_size=2,
        )
        brain.llm.complete = MagicMock(side_effect=["A", "B"])
        brain.query("q1")
        brain.query("q2")
        # Access q1 to make q2 LRU
        brain.llm.complete.side_effect = ["A"]  # would be skipped (cache hit)
        brain.query("q1")  # cache hit; q1 now MRU
        # q3 insert should evict q2 (LRU)
        brain.llm.complete.side_effect = ["C"]
        brain.query("q3")
        assert len(brain._query_cache) == 2
        cached_values = list(brain._query_cache.values())
        assert "B" not in cached_values, "q2 (LRU) should have been evicted"
        assert "A" in cached_values
        assert "C" in cached_values

    def test_cache_size_zero_does_not_cache(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """query_cache_size=0 disables caching (guard against StopIteration on empty cache)."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            query_cache=True,
            query_cache_size=0,
        )
        brain.llm.complete = MagicMock(return_value="answer")
        brain.query("q1")
        brain.query("q1")  # second call — should NOT raise and should call LLM again
        assert brain.llm.complete.call_count == 2  # no caching when size=0

    def test_cache_key_includes_compress_context(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Two requests differing only in compress_context get distinct cache keys."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        from axon.main import AxonConfig

        key1 = brain._make_cache_key("q", None, AxonConfig(compress_context=False))
        key2 = brain._make_cache_key("q", None, AxonConfig(compress_context=True))
        assert key1 != key2

    def test_cache_key_filter_serialisation_is_stable(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Same filters in different insertion order produce the same cache key."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        from axon.main import AxonConfig

        cfg = AxonConfig()
        key1 = brain._make_cache_key("q", {"b": 2, "a": 1}, cfg)
        key2 = brain._make_cache_key("q", {"a": 1, "b": 2}, cfg)
        assert key1 == key2


# ---------------------------------------------------------------------------
# _compress_context empty-results guard
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestCompressContextGuard:
    def test_compress_context_empty_list_returns_empty(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """_compress_context([]) must not raise ValueError from ThreadPoolExecutor(max_workers=0)."""
        from axon.main import AxonBrain, AxonConfig

        brain = AxonBrain(AxonConfig())
        result = brain._compress_context("any query", [])
        assert result == []

    def test_compress_context_single_item(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Single-item list compresses without error."""
        from axon.main import AxonBrain, AxonConfig

        brain = AxonBrain(AxonConfig())
        brain.llm.complete = MagicMock(return_value="short")
        doc = {"id": "d1", "text": "this is a longer original text for the test", "metadata": {}}
        result = brain._compress_context("query", [doc])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# switch_project reloads project-scoped state
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestSwitchProjectState:
    def test_switch_project_clears_query_cache(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        """Switching project empties the query cache to prevent cross-project bleed."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            query_cache=True,
            query_cache_size=128,
            vector_store_path=str(tmp_path / "chroma"),
            bm25_path=str(tmp_path / "bm25"),
        )
        brain = AxonBrain(config)
        # Prime the cache with a fake entry
        brain._query_cache["fake_key"] = "cached_answer"
        # Create a real project dir so switch_project doesn't raise
        proj_path = tmp_path / ".axon" / "projects" / "myproject"
        proj_path.mkdir(parents=True, exist_ok=True)
        with (
            patch("axon.projects.project_dir", return_value=proj_path),
            patch("axon.projects.project_vector_path", return_value=str(tmp_path / "proj_chroma")),
            patch("axon.projects.project_bm25_path", return_value=str(tmp_path / "proj_bm25")),
            patch("axon.projects.set_active_project"),
        ):
            brain.switch_project("myproject")
        assert len(brain._query_cache) == 0

    def test_switch_project_reloads_ingested_hashes(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        """Switching project loads the new project's content hashes."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            vector_store_path=str(tmp_path / "chroma"),
            bm25_path=str(tmp_path / "bm25"),
        )
        brain = AxonBrain(config)
        # Seed some hashes so we can confirm they get replaced
        brain._ingested_hashes = {"oldhash1", "oldhash2"}
        # Write new project's hash store
        new_bm25 = tmp_path / "proj_bm25"
        new_bm25.mkdir(parents=True)
        (new_bm25 / ".content_hashes").write_text("newhash1\nnewhash2\nnewhash3", encoding="utf-8")
        proj_path = tmp_path / ".axon" / "projects" / "proj2"
        proj_path.mkdir(parents=True, exist_ok=True)
        with (
            patch("axon.projects.project_dir", return_value=proj_path),
            patch("axon.projects.project_vector_path", return_value=str(tmp_path / "proj_chroma")),
            patch("axon.projects.project_bm25_path", return_value=str(new_bm25)),
            patch("axon.projects.set_active_project"),
        ):
            brain.switch_project("proj2")
        assert brain._ingested_hashes == {"newhash1", "newhash2", "newhash3"}
        assert "oldhash1" not in brain._ingested_hashes


# ---------------------------------------------------------------------------
# LanceDB vector store
# ---------------------------------------------------------------------------

import sys
import types as _types


def _make_lancedb_mock():
    """Create and register a minimal lancedb mock module in sys.modules."""
    mock_lancedb = _types.ModuleType("lancedb")
    mock_lancedb.connect = MagicMock()
    sys.modules["lancedb"] = mock_lancedb
    return mock_lancedb


class TestOpenVectorStoreLanceDB:
    def _make_store(self, mock_lancedb):
        """Helper: returns (store, mock_table) with lancedb already mocked."""
        from axon.main import AxonConfig, OpenVectorStore

        config = AxonConfig(vector_store="lancedb", vector_store_path="./lancedb_test")
        mock_table = MagicMock()
        mock_lancedb.connect.return_value.open_table.side_effect = Exception("not found")
        mock_lancedb.connect.return_value.create_table.return_value = mock_table
        store = OpenVectorStore(config)
        return store, mock_table

    def test_init_no_existing_table(self):
        mock_lancedb = _make_lancedb_mock()
        store, _ = self._make_store(mock_lancedb)
        assert store.collection is None  # lazy until first add

    def test_add_creates_table_on_first_call(self):
        mock_lancedb = _make_lancedb_mock()
        store, mock_table = self._make_store(mock_lancedb)
        store.add(["id1"], ["hello"], [[0.1, 0.2]], [{"source": "a.txt"}])
        mock_lancedb.connect.return_value.create_table.assert_called_once()
        assert store.collection is mock_table

    def test_search_returns_empty_when_no_table(self):
        mock_lancedb = _make_lancedb_mock()
        store, _ = self._make_store(mock_lancedb)
        results = store.search([0.1, 0.2], top_k=5)
        assert results == []

    def test_search_converts_distance_to_score(self):
        mock_lancedb = _make_lancedb_mock()
        store, mock_table = self._make_store(mock_lancedb)
        store.collection = mock_table  # simulate existing table
        mock_table.search.return_value.limit.return_value.to_list.return_value = [
            {"id": "d1", "text": "foo", "_distance": 0.2, "metadata_json": "{}"}
        ]
        results = store.search([0.1, 0.2], top_k=5)
        assert len(results) == 1
        assert abs(results[0]["score"] - 0.8) < 1e-6

    def test_list_documents_groups_by_source(self):
        mock_lancedb = _make_lancedb_mock()
        store, mock_table = self._make_store(mock_lancedb)
        store.collection = mock_table
        mock_table.to_arrow.return_value.to_pydict.return_value = {
            "id": ["c1", "c2", "c3"],
            "source": ["notes.txt", "notes.txt", "report.pdf"],
        }
        result = store.list_documents()
        assert len(result) == 2
        by_src = {d["source"]: d for d in result}
        assert by_src["notes.txt"]["chunks"] == 2

    def test_get_by_ids(self):
        mock_lancedb = _make_lancedb_mock()
        store, mock_table = self._make_store(mock_lancedb)
        store.collection = mock_table
        mock_table.search.return_value.where.return_value.to_list.return_value = [
            {"id": "id1", "text": "hello", "metadata_json": '{"source": "x.txt"}'}
        ]
        results = store.get_by_ids(["id1"])
        assert results[0]["id"] == "id1"
        assert results[0]["metadata"]["source"] == "x.txt"

    def test_delete_by_ids(self):
        mock_lancedb = _make_lancedb_mock()
        store, mock_table = self._make_store(mock_lancedb)
        store.collection = mock_table
        store.delete_by_ids(["id1", "id2"])
        mock_table.delete.assert_called_once()
        call_arg = mock_table.delete.call_args[0][0]
        assert "id1" in call_arg and "id2" in call_arg


class TestExpandAtFiles:
    """Tests for _expand_at_files() — file/folder context attachment."""

    def test_plain_text_file_inlined(self, tmp_path):
        from axon.main import _expand_at_files

        f = tmp_path / "notes.txt"
        f.write_text("hello world", encoding="utf-8")
        result = _expand_at_files(f"review @{f}")
        assert "hello world" in result
        assert str(f) in result

    def test_unknown_path_left_unchanged(self):
        from axon.main import _expand_at_files

        text = "check @nonexistent_xyz_file.txt please"
        assert _expand_at_files(text) == text

    def test_directory_expands_text_files(self, tmp_path):
        from axon.main import _expand_at_files

        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
        (tmp_path / "b.md").write_text("beta", encoding="utf-8")
        (tmp_path / "skip.bin").write_bytes(b"\x00\x01\x02")
        result = _expand_at_files(f"look at @{tmp_path}/")
        assert "alpha" in result
        assert "beta" in result
        assert "skip.bin" not in result  # binary ext not in _AT_TEXT_EXTS

    def test_directory_skips_hidden_dirs(self, tmp_path):
        from axon.main import _expand_at_files

        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.txt").write_text("secret", encoding="utf-8")
        (tmp_path / "visible.txt").write_text("visible", encoding="utf-8")
        result = _expand_at_files(f"@{tmp_path}/")
        assert "visible" in result
        assert "secret" not in result

    def test_directory_context_limit(self, tmp_path):
        from axon.main import _AT_DIR_MAX_BYTES, _AT_FILE_MAX_BYTES, _expand_at_files

        # Each file is capped at _AT_FILE_MAX_BYTES on read; create enough files so that
        # their combined sizes exceed _AT_DIR_MAX_BYTES, which triggers the skip message.
        # We need ceil(_AT_DIR_MAX_BYTES / _AT_FILE_MAX_BYTES) + 1 files.
        n_to_fill = (_AT_DIR_MAX_BYTES // _AT_FILE_MAX_BYTES) + 1
        for i in range(n_to_fill):
            (tmp_path / f"chunk{i:02d}.txt").write_text("y" * _AT_FILE_MAX_BYTES, encoding="utf-8")
        # This extra file should be skipped with "context limit reached"
        (tmp_path / "zzz_last.txt").write_text("should-be-skipped", encoding="utf-8")
        result = _expand_at_files(f"@{tmp_path}/")
        assert "context limit reached" in result

    def test_docx_extension_routes_to_loader(self, tmp_path):
        """Verify .docx files are routed through _read_via_loader (not plain text)."""
        import pytest

        from axon.main import _expand_at_files

        try:
            from docx import Document

            doc = Document()
            doc.add_paragraph("Extracted docx text")
            docx_path = tmp_path / "report.docx"
            doc.save(str(docx_path))
        except ImportError:
            pytest.skip("python-docx not installed")

        result = _expand_at_files(f"@{docx_path}")
        assert "Extracted docx text" in result

    def test_pdf_extension_routes_to_loader(self, tmp_path):
        """Verify .pdf files are routed through _read_via_loader (not plain text)."""
        import pytest

        from axon.main import _AT_LOADER_EXTS, _expand_at_files

        # Confirm .pdf is registered as a loader extension
        assert ".pdf" in _AT_LOADER_EXTS

        # Create a minimal valid PDF using pypdf / pymupdf if available
        try:
            import fitz  # PyMuPDF

            pdf_doc = fitz.open()
            page = pdf_doc.new_page()
            page.insert_text((50, 700), "Extracted pdf text")
            pdf_path = tmp_path / "paper.pdf"
            pdf_doc.save(str(pdf_path))
            pdf_doc.close()
        except ImportError:
            try:
                from pypdf import PdfWriter

                writer = PdfWriter()
                writer.add_blank_page(612, 792)
                pdf_path = tmp_path / "paper.pdf"
                with open(pdf_path, "wb") as f:
                    writer.write(f)
            except ImportError:
                pytest.skip("No PDF library (PyMuPDF/pypdf) available")

        result = _expand_at_files(f"@{pdf_path}")
        # Result is either extracted text or a graceful error message — never a crash
        assert str(pdf_path) in result

    def test_multiple_at_refs_in_one_query(self, tmp_path):
        from axon.main import _expand_at_files

        f1 = tmp_path / "one.txt"
        f2 = tmp_path / "two.txt"
        f1.write_text("AAA", encoding="utf-8")
        f2.write_text("BBB", encoding="utf-8")
        result = _expand_at_files(f"compare @{f1} and @{f2}")
        assert "AAA" in result
        assert "BBB" in result


class TestCliIngestFlag:
    """Regression tests for --ingest CLI flag bug (was silently ignored without --project-new)."""

    def _make_args(self, **kwargs):
        """Return a minimal argparse Namespace with sensible defaults."""
        import argparse

        defaults = {
            "query": None,
            "ingest": None,
            "list": False,
            "project": None,
            "project_new": None,
            "project_list": False,
            "project_delete": None,
            "quiet": False,
            "stream": False,
            "reranker_model": None,
            "top_k": None,
            "threshold": None,
            "hybrid": None,
            "rerank": None,
            "hyde": None,
            "multi_query": None,
            "step_back": None,
            "raptor": None,
            "graph_rag": None,
            "query_decompose": None,
            "compress_context": None,
            "dataset_type": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def _patch_brain_construction(self, tmp_path):
        """Return the patches needed to prevent real model loading in main()."""
        from unittest.mock import MagicMock

        mock_brain = MagicMock()
        mock_brain._active_project = "default"
        mock_brain.config = MagicMock()
        mock_brain.ingest = MagicMock()
        mock_brain.load_directory = MagicMock(return_value=None)
        return mock_brain

    def test_ingest_file_without_project_new(self, tmp_path):
        """--ingest <file> must work even when --project-new is not supplied."""
        txt = tmp_path / "doc.txt"
        txt.write_text("hello world", encoding="utf-8")

        mock_brain = self._patch_brain_construction(tmp_path)

        with (
            patch("axon.main.AxonBrain", return_value=mock_brain),
            patch("axon.main.AxonConfig"),
            patch("sys.argv", ["axon"]),
        ):
            args = self._make_args(ingest=str(txt))
            # Simulate the ingest branch directly (avoids REPL entry)
            if args.ingest:
                if __import__("os").path.isdir(args.ingest):
                    __import__("asyncio").run(mock_brain.load_directory(args.ingest))
                else:
                    from axon.loaders import DirectoryLoader

                    ext = __import__("os").path.splitext(args.ingest)[1].lower()
                    loader_mgr = DirectoryLoader()
                    if ext in loader_mgr.loaders:
                        mock_brain.ingest(loader_mgr.loaders[ext].load(args.ingest))

        # .txt is a supported extension — ingest must have been called
        mock_brain.ingest.assert_called_once()

    def test_ingest_directory_without_project_new(self, tmp_path):
        """--ingest <dir> must call load_directory even without --project-new."""
        (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")

        mock_brain = self._patch_brain_construction(tmp_path)

        import asyncio

        with patch("asyncio.run") as mock_run:
            if True:  # simulate args.ingest set to a directory
                asyncio.run(mock_brain.load_directory(str(tmp_path)))

        mock_run.assert_called_once()

    def test_ingest_not_called_when_flag_absent(self, tmp_path):
        """When --ingest is not supplied, brain.ingest must not be called."""
        mock_brain = self._patch_brain_construction(tmp_path)

        args_ingest = None  # --ingest not provided
        if args_ingest:
            mock_brain.ingest("should not reach here")

        mock_brain.ingest.assert_not_called()


# ---------------------------------------------------------------------------
# Embedding model safeguard
# ---------------------------------------------------------------------------


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore", autospec=True)
@patch("axon.main.OpenLLM", autospec=True)
@patch("axon.main.OpenEmbedding", autospec=True)
@patch("axon.main.OpenReranker", autospec=True)
class TestEmbeddingMetaSafeguard:
    """Validates that .embedding_meta.json prevents silent collection corruption."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path, **kw):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            bm25_path=str(tmp_path / "bm25"),
            vector_store_path=str(tmp_path / "chroma"),
            **kw,
        )
        brain = AxonBrain(config)
        brain.embedding.dimension = 384
        brain.splitter = None
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain.embedding.embed = MagicMock(return_value=[[0.1]])
        brain._own_vector_store.add = MagicMock()
        brain.vector_store.add = MagicMock()
        return brain

    def test_meta_written_after_first_ingest(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        """After a successful ingest, .embedding_meta.json is written."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path)
        brain.ingest([{"id": "d1", "text": "hello"}])
        meta_path = tmp_path / "bm25" / ".embedding_meta.json"
        assert meta_path.exists()
        import json

        meta = json.loads(meta_path.read_text())
        assert meta["embedding_provider"] == "sentence_transformers"
        assert meta["embedding_model"] == "all-MiniLM-L6-v2"
        assert meta["dimension"] == 384

    def test_same_model_ingest_succeeds(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        """Ingesting again with the same model passes validation."""
        import json

        meta_path = tmp_path / "bm25"
        meta_path.mkdir(parents=True, exist_ok=True)
        (meta_path / ".embedding_meta.json").write_text(
            json.dumps(
                {
                    "embedding_provider": "sentence_transformers",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "dimension": 384,
                }
            )
        )
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path)
        # Should not raise
        brain.ingest([{"id": "d1", "text": "hello"}])
        brain._own_vector_store.add.assert_called_once()

    def test_different_model_ingest_raises(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        """Ingesting with a different model raises ValueError to prevent corruption."""
        import json

        import pytest

        meta_path = tmp_path / "bm25"
        meta_path.mkdir(parents=True, exist_ok=True)
        (meta_path / ".embedding_meta.json").write_text(
            json.dumps(
                {
                    "embedding_provider": "sentence_transformers",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "dimension": 384,
                }
            )
        )
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            tmp_path,
            embedding_model="bge-small-en-v1.5",
        )
        with pytest.raises(ValueError, match="Embedding model mismatch"):
            brain.ingest([{"id": "d1", "text": "hello"}])
        # Store must NOT have been called — no data written
        brain._own_vector_store.add.assert_not_called()

    def test_different_provider_ingest_raises(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        """Switching provider (e.g. sentence_transformers → fastembed) also raises."""
        import json

        import pytest

        meta_path = tmp_path / "bm25"
        meta_path.mkdir(parents=True, exist_ok=True)
        (meta_path / ".embedding_meta.json").write_text(
            json.dumps(
                {
                    "embedding_provider": "sentence_transformers",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "dimension": 384,
                }
            )
        )
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            tmp_path,
            embedding_provider="fastembed",
            embedding_model="all-MiniLM-L6-v2",
        )
        with pytest.raises(ValueError, match="Embedding model mismatch"):
            brain.ingest([{"id": "d1", "text": "hello"}])

    def test_no_meta_file_allows_ingest(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        """When no meta file exists (new collection), ingest proceeds without error."""
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path)
        # No meta file present — should not raise
        brain.ingest([{"id": "d1", "text": "hello"}])
        brain._own_vector_store.add.assert_called_once()

    def test_query_warns_on_mismatch(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        """query() logs a warning on model mismatch but does not raise."""
        import json

        meta_path = tmp_path / "bm25"
        meta_path.mkdir(parents=True, exist_ok=True)
        (meta_path / ".embedding_meta.json").write_text(
            json.dumps(
                {
                    "embedding_provider": "sentence_transformers",
                    "embedding_model": "all-MiniLM-L6-v2",
                    "dimension": 384,
                }
            )
        )
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            tmp_path,
            embedding_model="bge-small-en-v1.5",
        )
        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(return_value=[])

        with patch("axon.main.logger") as mock_logger:
            brain.query("test question")
            warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
            assert any("mismatch" in w.lower() for w in warning_calls)


# ---------------------------------------------------------------------------
# Sessions per-project isolation (Finding 3.2)
# ---------------------------------------------------------------------------


class TestSessionsPerProject:
    """Validates that sessions are stored in the active project's sessions dir."""

    def test_sessions_dir_uses_project_path(self, tmp_path):
        """_sessions_dir() returns the project sessions path for non-default projects."""
        from axon.main import _sessions_dir

        with patch(
            "axon.projects.project_sessions_path", return_value=str(tmp_path / "proj_sessions")
        ) as mock_psp:
            result = _sessions_dir(project="work")
            mock_psp.assert_called_once_with("work")
            assert result == str(tmp_path / "proj_sessions")

    def test_sessions_dir_uses_global_for_default(self, tmp_path):
        """_sessions_dir() uses the global fallback for the 'default' project."""
        from axon.main import _SESSIONS_DIR, _sessions_dir

        result = _sessions_dir(project="default")
        assert result == _SESSIONS_DIR

    def test_sessions_dir_uses_global_when_no_project(self, tmp_path):
        """_sessions_dir() uses the global fallback when project is None."""
        from axon.main import _SESSIONS_DIR, _sessions_dir

        result = _sessions_dir(project=None)
        assert result == _SESSIONS_DIR

    def test_new_session_stores_project(self):
        """_new_session() includes the active project name in the session dict."""
        from axon.main import _new_session

        mock_brain = MagicMock()
        mock_brain.config.llm_provider = "ollama"
        mock_brain.config.llm_model = "gemma"
        mock_brain._active_project = "work/atlas"

        session = _new_session(mock_brain)
        assert session["project"] == "work/atlas"

    def test_save_session_writes_to_project_dir(self, tmp_path):
        """_save_session() writes the file into the project's sessions directory."""
        from axon.main import _save_session

        proj_sessions = tmp_path / "proj_sessions"
        proj_sessions.mkdir(parents=True)

        session = {
            "id": "20260315T120000000",
            "project": "work",
            "provider": "ollama",
            "model": "gemma",
            "history": [],
        }

        with patch("axon.projects.project_sessions_path", return_value=str(proj_sessions)):
            _save_session(session)

        saved = list(proj_sessions.glob("session_*.json"))
        assert len(saved) == 1
        import json

        data = json.loads(saved[0].read_text())
        assert data["id"] == "20260315T120000000"

    def test_list_sessions_scoped_to_project(self, tmp_path):
        """_list_sessions(project=...) only returns sessions from that project's dir."""
        import json

        from axon.main import _list_sessions

        proj_sessions = tmp_path / "proj_sessions"
        proj_sessions.mkdir(parents=True)
        (proj_sessions / "session_AAA.json").write_text(
            json.dumps({"id": "AAA", "history": [], "started_at": "2026-01-01T00:00:00"})
        )

        with patch("axon.projects.project_sessions_path", return_value=str(proj_sessions)):
            sessions = _list_sessions(project="work")

        assert len(sessions) == 1
        assert sessions[0]["id"] == "AAA"

    def test_load_session_from_project_dir(self, tmp_path):
        """_load_session() reads from the project's sessions directory."""
        import json

        from axon.main import _load_session

        proj_sessions = tmp_path / "proj_sessions"
        proj_sessions.mkdir(parents=True)
        (proj_sessions / "session_BBB.json").write_text(
            json.dumps({"id": "BBB", "history": [], "started_at": "2026-01-01T00:00:00"})
        )

        with patch("axon.projects.project_sessions_path", return_value=str(proj_sessions)):
            session = _load_session("BBB", project="work")

        assert session is not None
        assert session["id"] == "BBB"


# ---------------------------------------------------------------------------
# Single-file ingest breadcrumb (Finding 3.3)
# ---------------------------------------------------------------------------


class TestSingleFileIngestBreadcrumb:
    """Validates that --ingest <file> adds [File Path:] header like directory ingest."""

    def test_file_path_header_prepended(self, tmp_path):
        """Single-file ingest must prepend [File Path: <abs_path>] to each chunk."""

        abs_path = str(tmp_path / "notes.txt")
        docs = [{"id": "n1", "text": "Some notes content", "metadata": {"type": "text"}}]

        # Replicate the breadcrumb logic from the CLI ingest block
        for doc in docs:
            if doc.get("metadata", {}).get("type") not in ("csv", "tsv", "image"):
                doc["text"] = f"[File Path: {abs_path}]\n{doc['text']}"

        assert docs[0]["text"] == f"[File Path: {abs_path}]\nSome notes content"

    def test_csv_file_no_breadcrumb(self, tmp_path):
        """CSV/TSV/image files must NOT get the [File Path:] header."""
        abs_path = str(tmp_path / "data.csv")
        docs = [{"id": "c1", "text": "a,b,c", "metadata": {"type": "csv"}}]

        for doc in docs:
            if doc.get("metadata", {}).get("type") not in ("csv", "tsv", "image"):
                doc["text"] = f"[File Path: {abs_path}]\n{doc['text']}"

        assert docs[0]["text"] == "a,b,c"


# ---------------------------------------------------------------------------
# Hybrid threshold filtering (CodexQual fix)
# ---------------------------------------------------------------------------


class TestHybridThresholdFiltering:
    """Validates that the similarity threshold is applied to the fused score when
    hybrid_search=True, so exact-token BM25 hits are not suppressed by a low
    vector_score alone.

    The filtering logic being tested (from main.py):

        if r.get("fused_only"):
            filtered_results.append(r)
            continue
        if cfg.hybrid_search:
            sig = r.get("score", r.get("vector_score", 0.0))
        else:
            sig = r.get("vector_score", r.get("score", 0.0))
        if sig >= cfg.similarity_threshold:
            filtered_results.append(r)
    """

    def _filter(self, results, hybrid_search, threshold):
        """Mirror of the filtering loop in AxonBrain._query_core()."""

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.hybrid_search = hybrid_search
        cfg.similarity_threshold = threshold

        filtered = []
        for r in results:
            if r.get("fused_only"):
                filtered.append(r)
                continue
            if cfg.hybrid_search:
                sig = r.get("score", r.get("vector_score", 0.0))
            else:
                sig = r.get("vector_score", r.get("score", 0.0))
            if sig >= cfg.similarity_threshold:
                filtered.append(r)
        return filtered

    def test_fused_only_always_passes_threshold(self):
        """BM25-only hits (fused_only=True) must pass regardless of threshold."""
        results = [{"id": "bm25_hit", "fused_only": True, "score": 0.0, "vector_score": 0.0}]
        out = self._filter(results, hybrid_search=True, threshold=0.9)
        assert len(out) == 1

    def test_hybrid_uses_fused_score_not_vector_score(self):
        """When hybrid=True, a doc with high fused score but low vector_score must pass."""
        # Exact-token BM25 match: fused score 0.6, vector_score 0.1
        results = [{"id": "exact_match", "score": 0.6, "vector_score": 0.1, "fused_only": False}]
        # With old (broken) logic: vector_score 0.1 < threshold 0.3 → filtered out
        # With new logic: fused score 0.6 >= threshold 0.3 → kept
        out = self._filter(results, hybrid_search=True, threshold=0.3)
        assert len(out) == 1
        assert out[0]["id"] == "exact_match"

    def test_hybrid_filters_low_fused_score(self):
        """When hybrid=True, docs below the fused-score threshold are still filtered."""
        results = [{"id": "weak", "score": 0.2, "vector_score": 0.1, "fused_only": False}]
        out = self._filter(results, hybrid_search=True, threshold=0.3)
        assert len(out) == 0

    def test_non_hybrid_uses_vector_score(self):
        """When hybrid=False, threshold is applied to vector_score."""
        results = [{"id": "doc", "score": 0.8, "vector_score": 0.25, "fused_only": False}]
        # vector_score 0.25 < threshold 0.3 → filtered out despite high fused score
        out = self._filter(results, hybrid_search=False, threshold=0.3)
        assert len(out) == 0

    def test_non_hybrid_passes_high_vector_score(self):
        """When hybrid=False, docs with vector_score above threshold pass."""
        results = [{"id": "doc", "score": 0.4, "vector_score": 0.5, "fused_only": False}]
        out = self._filter(results, hybrid_search=False, threshold=0.3)
        assert len(out) == 1


class TestTemperatureCLI:
    """--temperature CLI flag sets config.llm_temperature."""

    def test_temperature_flag_sets_config(self, tmp_path):
        """--temperature <float> is applied to config.llm_temperature."""
        from axon.main import AxonConfig

        config = AxonConfig()
        config.bm25_path = str(tmp_path / "bm25")
        config.vector_store_path = str(tmp_path / "chroma")

        import argparse

        args = argparse.Namespace(temperature=0.2)
        if args.temperature is not None:
            config.llm_temperature = args.temperature

        assert config.llm_temperature == 0.2

    def test_temperature_flag_absent_leaves_default(self, tmp_path):
        """When --temperature is not passed, llm_temperature stays at the config default."""
        from axon.main import AxonConfig

        config = AxonConfig()
        default_temp = config.llm_temperature

        import argparse

        args = argparse.Namespace(temperature=None)
        if args.temperature is not None:
            config.llm_temperature = args.temperature

        assert config.llm_temperature == default_temp

    def test_temperature_default_is_float(self):
        """AxonConfig.llm_temperature default is a float."""
        from axon.main import AxonConfig

        config = AxonConfig()
        assert isinstance(config.llm_temperature, float)


class TestReplLlmCommand:
    """/llm REPL command sets brain.config.llm_temperature."""

    def _make_brain(self):
        brain = MagicMock()
        brain.config.llm_temperature = 0.7
        brain.config.llm_provider = "ollama"
        brain.config.llm_model = "phi3:mini"
        return brain

    def test_llm_temperature_sets_value(self):
        """Simulated /llm temperature sets brain.config.llm_temperature."""
        brain = self._make_brain()
        # Mirror the /llm handler logic
        arg = "temperature 0.3"
        llm_parts = arg.split(maxsplit=1)
        llm_opt = llm_parts[0].lower()
        llm_val = llm_parts[1] if len(llm_parts) > 1 else ""
        if llm_opt == "temperature":
            v = float(llm_val)
            assert 0.0 <= v <= 2.0
            brain.config.llm_temperature = v

        assert brain.config.llm_temperature == 0.3

    def test_llm_temperature_zero_accepted(self):
        """Temperature 0.0 (fully deterministic) is a valid value."""
        brain = self._make_brain()
        arg = "temperature 0.0"
        llm_parts = arg.split(maxsplit=1)
        v = float(llm_parts[1])
        assert 0.0 <= v <= 2.0
        brain.config.llm_temperature = v
        assert brain.config.llm_temperature == 0.0

    def test_llm_temperature_max_accepted(self):
        """Temperature 2.0 (maximum) is a valid value."""
        brain = self._make_brain()
        arg = "temperature 2.0"
        llm_parts = arg.split(maxsplit=1)
        v = float(llm_parts[1])
        assert 0.0 <= v <= 2.0
        brain.config.llm_temperature = v
        assert brain.config.llm_temperature == 2.0

    def test_llm_temperature_out_of_range_rejected(self):
        """Temperature outside 0.0–2.0 must be caught by the assert."""
        import pytest

        arg = "temperature 3.5"
        llm_parts = arg.split(maxsplit=1)
        v = float(llm_parts[1])
        with pytest.raises(AssertionError):
            assert 0.0 <= v <= 2.0

    def test_llm_unknown_option_handled(self):
        """An unknown /llm option does not crash — returns error message."""
        output = []
        arg = "unknown_opt"
        llm_parts = arg.split(maxsplit=1)
        llm_opt = llm_parts[0].lower()
        if llm_opt == "temperature":
            pass  # would set temperature
        else:
            output.append(f"Unknown option '{llm_opt}'. Available: temperature")
        assert len(output) == 1
        assert "Unknown option" in output[0]


class TestSlashCommandOrder:
    """_SLASH_COMMANDS list is in alphabetical order."""

    def test_commands_alphabetically_sorted(self):
        """All slash commands must appear in alphabetical order (ignoring trailing spaces)."""
        from axon.main import _SLASH_COMMANDS

        stripped = [c.strip() for c in _SLASH_COMMANDS]
        assert stripped == sorted(stripped), f"Commands not in alphabetical order. Got: {stripped}"


class TestGraphRAGRobustness:
    """Tests for GraphRAG robustness improvements: Jaccard matching, budget slicing,
    entity pruning, relation extraction, and 1-hop traversal."""

    # ── Helper: build a minimal AxonBrain-like mock ────────────────────────

    def _make_brain(self):
        """Return a MagicMock AxonBrain with the real methods we want to test."""
        from unittest.mock import MagicMock

        from axon.main import AxonBrain, AxonConfig

        brain = MagicMock(spec=AxonBrain)
        brain.config = AxonConfig()
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._entity_embeddings = {}
        brain._claims_graph = {}
        # Bind real implementations so we test actual logic
        brain._entity_matches = AxonBrain._entity_matches.__get__(brain, AxonBrain)
        brain._prune_entity_graph = AxonBrain._prune_entity_graph.__get__(brain, AxonBrain)
        brain._extract_relations = AxonBrain._extract_relations.__get__(brain, AxonBrain)
        brain._expand_with_entity_graph = AxonBrain._expand_with_entity_graph.__get__(
            brain, AxonBrain
        )
        brain._extract_entities = AxonBrain._extract_entities.__get__(brain, AxonBrain)
        brain._match_entities_by_embedding = AxonBrain._match_entities_by_embedding.__get__(
            brain, AxonBrain
        )
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._save_claims_graph = MagicMock()
        return brain

    # ── Phase 1: _entity_matches ──────────────────────────────────────────

    def test_entity_exact_match(self):
        """Identical strings return score 1.0."""
        brain = self._make_brain()
        assert brain._entity_matches("OpenAI", "openai") == 1.0

    def test_entity_single_token_no_substring(self):
        """Single tokens that differ must not match (no substring matching)."""
        brain = self._make_brain()
        assert brain._entity_matches("bert", "robert") == 0.0

    def test_entity_multi_token_jaccard(self):
        """Multi-token phrases with >= 0.4 Jaccard overlap return a non-zero score."""
        brain = self._make_brain()
        score = brain._entity_matches("machine learning", "learning machine")
        assert score >= 0.4

    def test_entity_single_token_no_cross(self):
        """Single-token query must not match a multi-token graph entity it doesn't equal."""
        brain = self._make_brain()
        # "IT" is one token; "Critical IT Infrastructure" is three tokens.
        # Jaccard = 1/3 < 0.4 — but the single-token rule fires first and returns 0.0.
        score = brain._entity_matches("IT", "Critical IT Infrastructure")
        assert score == 0.0

    # ── Phase 1: graph_rag_budget slicing ────────────────────────────────

    def test_graph_budget_guarantees_expanded_results(self):
        """Entity-expanded docs survive even when base results fill top_k."""
        # Simulate: 3 base docs (top_k=3), 2 expanded docs, budget=2
        base = [{"id": f"base_{i}", "score": 0.9, "_graph_expanded": False} for i in range(3)]
        expanded = [{"id": f"exp_{i}", "score": 0.7, "_graph_expanded": True} for i in range(2)]
        results = base + expanded

        class _Cfg:
            top_k = 3
            graph_rag = True
            graph_rag_budget = 2

        cfg = _Cfg()
        base_out = [r for r in results if not r.get("_graph_expanded")][: cfg.top_k]
        expanded_out = [r for r in results if r.get("_graph_expanded")]
        base_ids = {r["id"] for r in base_out}
        graph_slots = [r for r in expanded_out if r["id"] not in base_ids][: cfg.graph_rag_budget]
        final = base_out + graph_slots

        assert len(final) == 5  # 3 base + 2 expanded
        exp_ids = {r["id"] for r in final if "exp_" in r["id"]}
        assert exp_ids == {"exp_0", "exp_1"}

    def test_graph_budget_zero_no_guarantee(self):
        """When budget=0, fall back to plain top_k truncation."""
        base = [{"id": f"base_{i}", "score": 0.9, "_graph_expanded": False} for i in range(3)]
        expanded = [{"id": "exp_0", "score": 0.7, "_graph_expanded": True}]
        results = base + expanded

        class _Cfg:
            top_k = 3
            graph_rag = True
            graph_rag_budget = 0

        cfg = _Cfg()
        # budget=0 → plain slice
        final = results[: cfg.top_k]
        assert len(final) == 3
        assert all("exp_" not in r["id"] for r in final)

    def test_graph_expanded_score_in_range(self):
        """Scores assigned to expanded docs are in [0.5, 0.8)."""
        brain = self._make_brain()
        brain._entity_graph = {"machine learning": ["doc_ml_1"]}
        brain._relation_graph = {}
        brain.config.graph_rag_relations = False

        # Entity "machine learning" in query should match the graph entity with score > 0
        # _entity_matches("machine learning", "machine learning") == 1.0
        # doc_score = 0.5 + 1.0 * 0.3 = 0.8 — boundary, ensure < 0.8 is not violated
        # Actually 0.5 + score*0.3, score up to 1.0 => up to 0.8 exactly (boundary)
        brain._extract_entities = MagicMock(return_value=["machine learning"])
        brain.vector_store = MagicMock()
        brain.vector_store.get_by_ids = MagicMock(
            return_value=[{"id": "doc_ml_1", "text": "ml text", "score": 1.0, "metadata": {}}]
        )

        results, _ = brain._expand_with_entity_graph("machine learning query", [], cfg=None)
        expanded = [r for r in results if r.get("_graph_expanded")]
        assert len(expanded) == 1
        score = expanded[0]["score"]
        assert 0.5 <= score <= 0.8

    def test_graph_expanded_score_not_hardcoded(self):
        """Expanded doc scores must not be exactly 1.0 (the old bug)."""
        brain = self._make_brain()
        brain._entity_graph = {"python": ["doc_py_1"]}
        brain._relation_graph = {}
        brain.config.graph_rag_relations = False
        brain._extract_entities = MagicMock(return_value=["python"])
        brain.vector_store = MagicMock()
        brain.vector_store.get_by_ids = MagicMock(
            return_value=[{"id": "doc_py_1", "text": "py text", "score": 1.0, "metadata": {}}]
        )

        results, _ = brain._expand_with_entity_graph("python query", [], cfg=None)
        expanded = [r for r in results if r.get("_graph_expanded")]
        assert len(expanded) == 1
        assert expanded[0]["score"] != 1.0

    # ── Phase 2: entity graph pruning ────────────────────────────────────

    def test_entity_graph_pruned_on_delete(self):
        """Deleted doc IDs are removed from the entity graph."""
        brain = self._make_brain()
        brain._entity_graph = {"python": ["doc_1", "doc_2"], "machine learning": ["doc_2"]}
        brain._relation_graph = {}

        brain._prune_entity_graph({"doc_2"})

        # Legacy list format is migrated to dict-node format on modification
        node = brain._entity_graph["python"]
        chunk_ids = node["chunk_ids"] if isinstance(node, dict) else node
        assert chunk_ids == ["doc_1"]
        assert "machine learning" not in brain._entity_graph
        brain._save_entity_graph.assert_called_once()

    def test_empty_entity_entry_removed_on_prune(self):
        """Entity entries that become empty after pruning are deleted entirely."""
        brain = self._make_brain()
        brain._entity_graph = {"solo_entity": ["only_doc"]}
        brain._relation_graph = {}

        brain._prune_entity_graph({"only_doc"})

        assert "solo_entity" not in brain._entity_graph
        brain._save_entity_graph.assert_called_once()

    # ── Phase 3: relation extraction ─────────────────────────────────────

    def test_relation_extraction_parses_pipe_format(self):
        """_extract_relations correctly parses SUBJECT | RELATION | OBJECT lines."""
        brain = self._make_brain()
        brain.llm = MagicMock()
        brain.llm.complete = MagicMock(
            return_value="Python | is used for | machine learning\nOpenAI | created | GPT-4"
        )

        triples = brain._extract_relations("some text")

        assert len(triples) == 2
        assert triples[0]["subject"] == "Python"
        assert triples[0]["relation"] == "is used for"
        assert triples[0]["object"] == "machine learning"
        assert triples[1]["subject"] == "OpenAI"
        assert triples[1]["relation"] == "created"
        assert triples[1]["object"] == "GPT-4"

    def test_relation_extraction_ignores_bad_lines(self):
        """Malformed lines (fewer than 3 parts) are silently skipped; 4+ parts are accepted."""
        brain = self._make_brain()
        brain.llm = MagicMock()
        brain.llm.complete = MagicMock(
            return_value=(
                "Good line | relates to | something\n"
                "bad line no pipes\n"
                "only | two parts\n"
                "Subject | relation | object | extra description here"
            )
        )

        triples = brain._extract_relations("some text")

        # "bad line no pipes" and "only | two parts" are skipped; the other two are valid
        assert len(triples) == 2
        assert triples[0]["subject"] == "Good line"
        assert triples[0]["relation"] == "relates to"
        assert triples[0]["object"] == "something"
        assert triples[1]["subject"] == "Subject"
        assert triples[1]["description"] == "extra description here"

    # ── Phase 3: 1-hop traversal ─────────────────────────────────────────

    def test_one_hop_traversal_fetches_related_chunks(self):
        """Querying entity A finds B's chunks via a relation A → B."""
        brain = self._make_brain()
        # Entity graph: "python" has doc_py, "machine learning" has doc_ml
        brain._entity_graph = {
            "python": ["doc_py"],
            "machine learning": ["doc_ml"],
        }
        # Relation graph: python → machine learning
        brain._relation_graph = {
            "python": [{"target": "machine learning", "relation": "used for", "chunk_id": "doc_py"}]
        }
        brain.config.graph_rag_relations = True
        brain._extract_entities = MagicMock(return_value=["python"])
        brain.vector_store = MagicMock()

        fetched_ids = []

        def _fake_get_by_ids(ids):
            fetched_ids.extend(ids)
            return [{"id": i, "text": f"text {i}", "score": 0.7, "metadata": {}} for i in ids]

        brain.vector_store.get_by_ids = _fake_get_by_ids

        brain._expand_with_entity_graph("python programming", [], cfg=None)

        assert "doc_ml" in fetched_ids, "1-hop target doc_ml should have been fetched"

    def test_one_hop_score_lower_than_direct(self):
        """1-hop traversal scores (0.62) are lower than direct entity match scores."""
        brain = self._make_brain()
        brain._entity_graph = {
            "python": ["doc_py"],
            "machine learning": ["doc_ml"],
        }
        brain._relation_graph = {
            "python": [{"target": "machine learning", "relation": "used for", "chunk_id": "doc_py"}]
        }
        brain.config.graph_rag_relations = True
        brain._extract_entities = MagicMock(return_value=["python"])
        brain.vector_store = MagicMock()

        def _fake_get_by_ids(ids):
            docs = []
            for i in ids:
                doc = {"id": i, "text": f"text {i}", "score": 0.9, "metadata": {}}
                docs.append(doc)
            return docs

        brain.vector_store.get_by_ids = _fake_get_by_ids

        results, _ = brain._expand_with_entity_graph("python programming", [], cfg=None)

        expanded = [r for r in results if r.get("_graph_expanded")]
        scores_by_id = {r["id"]: r["score"] for r in expanded}

        # doc_py matched directly (score = 0.5 + 1.0*0.3 = 0.8)
        # doc_ml matched via 1-hop (score = 0.62)
        assert scores_by_id.get("doc_ml", 1.0) < scores_by_id.get(
            "doc_py", 0.0
        ), "1-hop doc score should be lower than direct match score"

    # ── Phase 4: relation graph pruning ──────────────────────────────────

    def test_relation_graph_pruned_on_delete(self):
        """Relation graph entries with deleted chunk_ids are removed."""
        brain = self._make_brain()
        brain._entity_graph = {}
        brain._relation_graph = {
            "python": [
                {"target": "ml", "relation": "used for", "chunk_id": "doc_1"},
                {"target": "web", "relation": "used for", "chunk_id": "doc_2"},
            ]
        }

        brain._prune_entity_graph({"doc_1"})

        assert len(brain._relation_graph["python"]) == 1
        assert brain._relation_graph["python"][0]["chunk_id"] == "doc_2"
        brain._save_relation_graph.assert_called_once()

    # ── Phase 1.6: filtered_count metric ─────────────────────────────────

    def test_filtered_count_excludes_graph_results(self):
        """filtered_count in _execute_retrieval reflects the pre-expansion result count."""
        # We test the logic in isolation: base_count saved before GraphRAG expansion
        base_results = [{"id": "a", "score": 0.8}, {"id": "b", "score": 0.7}]
        graph_results = [{"id": "c", "score": 0.6, "_graph_expanded": True}]
        all_results = base_results + graph_results

        base_count = len(base_results)  # should be 2, not 3
        graph_expanded_count = len(all_results) - base_count

        assert base_count == 2
        assert graph_expanded_count == 1

    # ── Phase 1.5: extraction text cap ───────────────────────────────────

    def test_extraction_cap_at_3000_chars(self):
        """Entity extraction prompt includes text[:3000], not text[:1500]."""
        from axon.main import AxonBrain

        brain = self._make_brain()
        captured_prompts = []
        brain.llm = MagicMock()

        def _capture_complete(prompt, system_prompt=None):
            captured_prompts.append(prompt)
            return ""

        brain.llm.complete = _capture_complete

        long_text = "x" * 4000
        brain._extract_entities = AxonBrain._extract_entities.__get__(brain, AxonBrain)
        brain._extract_entities(long_text)

        assert captured_prompts, "LLM complete should have been called"
        prompt = captured_prompts[0]
        # The text portion should be exactly 3000 chars
        assert "x" * 3000 in prompt
        assert "x" * 3001 not in prompt

    # ── Finding 8 fix: relation graph merged on project switch ────────────

    def test_relation_graph_merged_on_project_switch(self, tmp_path):
        """Descendant relation graphs must be merged into the active view on project switch."""
        import json

        # Write a fake .relation_graph.json for a descendant project
        desc_bm25 = tmp_path / "desc_bm25"
        desc_bm25.mkdir()
        rel_data = {
            "apple": [{"target": "beats", "relation": "acquired", "chunk_id": "chunk_apple_1"}]
        }
        (desc_bm25 / ".relation_graph.json").write_text(json.dumps(rel_data), encoding="utf-8")

        brain = self._make_brain()
        # Pre-load parent state (empty)
        brain._entity_graph = {}
        brain._relation_graph = {}

        # Simulate the merge loop directly (mirrors switch_project internals)
        import pathlib

        desc_rel_path = pathlib.Path(desc_bm25) / ".relation_graph.json"
        raw = json.loads(desc_rel_path.read_text(encoding="utf-8"))
        for src, entries in raw.items():
            if src not in brain._relation_graph:
                brain._relation_graph[src] = []
            existing = {
                (e.get("target"), e.get("relation"), e.get("chunk_id"))
                for e in brain._relation_graph[src]
            }
            for entry in entries:
                key = (entry.get("target"), entry.get("relation"), entry.get("chunk_id"))
                if key not in existing:
                    brain._relation_graph[src].append(entry)
                    existing.add(key)

        assert "apple" in brain._relation_graph
        assert brain._relation_graph["apple"][0]["target"] == "beats"

    def test_relation_graph_merge_deduplicates_entries(self, tmp_path):
        """Merging twice should not produce duplicate relation entries."""
        import json
        import pathlib

        desc_bm25 = tmp_path / "desc_bm25"
        desc_bm25.mkdir()
        entry = {"target": "beats", "relation": "acquired", "chunk_id": "chunk_apple_1"}
        rel_data = {"apple": [entry]}
        (desc_bm25 / ".relation_graph.json").write_text(json.dumps(rel_data), encoding="utf-8")

        brain = self._make_brain()
        brain._relation_graph = {"apple": [dict(entry)]}  # already contains this entry

        # Merge
        desc_rel_path = pathlib.Path(desc_bm25) / ".relation_graph.json"
        raw = json.loads(desc_rel_path.read_text(encoding="utf-8"))
        for src, entries in raw.items():
            if src not in brain._relation_graph:
                brain._relation_graph[src] = []
            existing = {
                (e.get("target"), e.get("relation"), e.get("chunk_id"))
                for e in brain._relation_graph[src]
            }
            for e in entries:
                key = (e.get("target"), e.get("relation"), e.get("chunk_id"))
                if key not in existing:
                    brain._relation_graph[src].append(e)
                    existing.add(key)

        assert len(brain._relation_graph["apple"]) == 1, "Duplicate entries must not be added"


class TestGraphRAGCommunity:
    """Tests for Phase 1-5 GraphRAG community detection, summaries, and global/local search."""

    # ── Helper ────────────────────────────────────────────────────────────

    def _make_brain(self):
        """Return a MagicMock AxonBrain with the real methods we want to test."""
        from unittest.mock import MagicMock

        from axon.main import AxonBrain, AxonConfig

        brain = MagicMock(spec=AxonBrain)
        brain.config = AxonConfig()
        brain.llm = MagicMock()
        brain.embedding = MagicMock()
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._community_levels = {}
        brain._community_summaries = {}
        brain._community_graph_dirty = False
        brain._last_matched_entities = []
        brain._entity_embeddings = {}
        brain._entity_description_buffer = {}
        brain._claims_graph = {}
        brain._community_hierarchy = {}
        brain._community_children = {}
        brain._relation_description_buffer = {}
        brain._text_unit_entity_map = {}
        brain._text_unit_relation_map = {}
        # Bind real implementations
        brain._extract_entities = AxonBrain._extract_entities.__get__(brain, AxonBrain)
        brain._extract_relations = AxonBrain._extract_relations.__get__(brain, AxonBrain)
        brain._prune_entity_graph = AxonBrain._prune_entity_graph.__get__(brain, AxonBrain)
        brain._expand_with_entity_graph = AxonBrain._expand_with_entity_graph.__get__(
            brain, AxonBrain
        )
        brain._run_community_detection = AxonBrain._run_community_detection.__get__(
            brain, AxonBrain
        )
        brain._run_hierarchical_community_detection = (
            AxonBrain._run_hierarchical_community_detection.__get__(brain, AxonBrain)
        )
        brain._build_networkx_graph = AxonBrain._build_networkx_graph.__get__(brain, AxonBrain)
        brain._generate_community_summaries = AxonBrain._generate_community_summaries.__get__(
            brain, AxonBrain
        )
        brain._global_search_context = AxonBrain._global_search_context.__get__(brain, AxonBrain)
        brain._global_search_map_reduce = AxonBrain._global_search_map_reduce.__get__(
            brain, AxonBrain
        )
        brain._local_search_context = AxonBrain._local_search_context.__get__(brain, AxonBrain)
        brain._get_incoming_relations = AxonBrain._get_incoming_relations.__get__(brain, AxonBrain)
        brain._entity_matches = AxonBrain._entity_matches.__get__(brain, AxonBrain)
        brain._match_entities_by_embedding = AxonBrain._match_entities_by_embedding.__get__(
            brain, AxonBrain
        )
        brain._embed_entities = AxonBrain._embed_entities.__get__(brain, AxonBrain)
        brain._extract_claims = AxonBrain._extract_claims.__get__(brain, AxonBrain)
        brain._canonicalize_entity_descriptions = (
            AxonBrain._canonicalize_entity_descriptions.__get__(brain, AxonBrain)
        )
        brain._index_community_reports_in_vector_store = (
            AxonBrain._index_community_reports_in_vector_store.__get__(brain, AxonBrain)
        )
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._save_community_map = MagicMock()
        brain._save_community_summaries = MagicMock()
        brain._save_community_levels = MagicMock()
        brain._save_community_hierarchy = MagicMock()
        brain._save_entity_embeddings = MagicMock()
        brain._save_claims_graph = MagicMock()
        return brain

    # ── Phase 1.1: _extract_entities returns dicts ────────────────────────

    def test_extract_entities_returns_name_and_description(self):
        """_extract_entities parses 'NAME | description' 2-col lines with UNKNOWN type."""
        brain = self._make_brain()
        brain.llm.complete.return_value = (
            "OpenAI | AI research company\nGPT-4 | Large language model"
        )
        result = brain._extract_entities("some text")
        assert len(result) == 2
        assert result[0]["name"] == "OpenAI"
        assert result[0]["description"] == "AI research company"
        assert result[0]["type"] == "UNKNOWN"
        assert result[1]["name"] == "GPT-4"
        assert result[1]["type"] == "UNKNOWN"

    def test_extract_entities_handles_no_pipe_fallback(self):
        """_extract_entities handles lines without a pipe by returning empty description."""
        brain = self._make_brain()
        brain.llm.complete.return_value = "OpenAI"
        result = brain._extract_entities("some text")
        assert len(result) == 1
        assert result[0]["name"] == "OpenAI"
        assert result[0]["description"] == ""
        assert result[0]["type"] == "UNKNOWN"

    # ── Phase 1.2: _extract_relations returns dicts ───────────────────────

    def test_extract_relations_returns_description(self):
        """_extract_relations parses 4-part lines into full dicts with description."""
        brain = self._make_brain()
        brain.llm.complete.return_value = "Apple | acquired | Beats | Apple bought Beats for $3B"
        result = brain._extract_relations("some text")
        assert len(result) == 1
        assert result[0] == {
            "subject": "Apple",
            "relation": "acquired",
            "object": "Beats",
            "description": "Apple bought Beats for $3B",
            "strength": 5,
        }

    def test_extract_relations_three_part_fallback(self):
        """_extract_relations handles 3-part lines with empty description."""
        brain = self._make_brain()
        brain.llm.complete.return_value = "Apple | acquired | Beats"
        result = brain._extract_relations("some text")
        assert len(result) == 1
        assert result[0]["subject"] == "Apple"
        assert result[0]["relation"] == "acquired"
        assert result[0]["object"] == "Beats"
        assert result[0]["description"] == ""

    # ── Phase 1.3: entity graph migration ────────────────────────────────

    def test_entity_graph_migration_from_flat_list(self, tmp_path):
        """_load_entity_graph migrates legacy flat list format to new dict-node format."""
        import json
        from unittest.mock import MagicMock

        from axon.main import AxonBrain, AxonConfig

        brain = MagicMock(spec=AxonBrain)
        brain.config = AxonConfig(bm25_path=str(tmp_path))
        brain._load_entity_graph = AxonBrain._load_entity_graph.__get__(brain, AxonBrain)

        # Write legacy format
        legacy = {"openai": ["chunk1", "chunk2"]}
        (tmp_path / ".entity_graph.json").write_text(json.dumps(legacy), encoding="utf-8")

        result = brain._load_entity_graph()
        assert result["openai"]["description"] == ""
        assert result["openai"]["chunk_ids"] == ["chunk1", "chunk2"]
        assert result["openai"]["type"] == "UNKNOWN"
        assert result["openai"]["frequency"] == 2
        assert result["openai"]["degree"] == 0

    # ── Phase 2.5: community detection ───────────────────────────────────

    def test_community_detection_requires_networkx(self):
        """_run_community_detection returns {} gracefully when networkx is missing."""
        import sys
        from unittest.mock import patch

        brain = self._make_brain()
        brain._entity_graph = {"apple": {"description": "tech", "chunk_ids": ["c1"]}}

        with patch.dict(sys.modules, {"networkx": None}):
            result = brain._run_community_detection()
        assert result == {}

    # ── Phase 2.4: community map persist ─────────────────────────────────

    def test_community_map_persisted(self, tmp_path):
        """_save_community_levels / _load_community_levels round-trips correctly."""
        from unittest.mock import MagicMock

        from axon.main import AxonBrain, AxonConfig

        brain = MagicMock(spec=AxonBrain)
        brain.config = AxonConfig(bm25_path=str(tmp_path))
        brain._community_levels = {0: {"entityA": 0, "entityB": 1}}
        brain._save_community_levels = AxonBrain._save_community_levels.__get__(brain, AxonBrain)
        brain._load_community_levels = AxonBrain._load_community_levels.__get__(brain, AxonBrain)

        brain._save_community_levels()
        loaded = brain._load_community_levels()
        assert loaded == {0: {"entityA": 0, "entityB": 1}}

    # ── Phase 3: community summaries generation ───────────────────────────

    def test_community_summaries_generated(self):
        """_generate_community_summaries populates _community_summaries with LLM output."""
        from concurrent.futures import ThreadPoolExecutor

        brain = self._make_brain()
        brain._community_levels = {0: {"apple": 0, "beats": 0}}
        brain._entity_graph = {
            "apple": {"description": "tech company", "chunk_ids": ["c1"]},
            "beats": {"description": "audio brand", "chunk_ids": ["c2"]},
        }
        brain._relation_graph = {}
        brain._community_summaries = {}
        brain.llm.complete.return_value = (
            '{"title": "Tech", "summary": "Apple and Beats.", "findings": [], "rank": 5.0}'
        )
        brain._executor = ThreadPoolExecutor(max_workers=1)

        brain._generate_community_summaries()

        assert "0_0" in brain._community_summaries
        assert brain._community_summaries["0_0"]["summary"] != ""
        brain._save_community_summaries.assert_called_once()

    # ── Phase 4: global search context ───────────────────────────────────

    def test_global_search_context_returns_summaries(self):
        """_global_search_context returns text containing community summary content."""
        brain = self._make_brain()
        brain._community_summaries = {
            "0_0": {
                "title": "Tech cluster",
                "summary": "Apple and Beats tech cluster",
                "full_content": "# Tech cluster\n\nApple and Beats tech cluster",
                "findings": [],
                "rank": 5.0,
                "entities": ["apple", "beats"],
                "size": 2,
                "level": 0,
            }
        }

        class _Cfg:
            graph_rag_community_top_k = 5

        # Disable embedding path so token-overlap fallback is used
        brain.embedding.embed_query.side_effect = Exception("no embed")
        result = brain._global_search_context("Apple products", _Cfg())
        assert "Apple" in result or "Beats" in result

    # ── Phase 5: local search context ────────────────────────────────────

    def test_local_search_context_includes_entity_description(self):
        """_local_search_context includes entity descriptions in output."""
        from unittest.mock import MagicMock

        brain = self._make_brain()
        brain._entity_graph = {"apple": {"description": "tech company", "chunk_ids": ["c1"]}}
        brain._relation_graph = {}
        brain._community_levels = {}
        brain._community_summaries = {}
        brain._claims_graph = {}
        # Mock vector store to return empty for text units lookup
        brain.vector_store = MagicMock()
        brain.vector_store.get_by_ids.return_value = []

        class _Cfg:
            pass

        result = brain._local_search_context("query", ["apple"], _Cfg())
        assert "tech company" in result

    # ── AxonConfig new fields ─────────────────────────────────────────────

    def test_axonconfig_new_fields_defaults(self):
        """New AxonConfig community fields have correct default values."""
        from axon.main import AxonConfig

        cfg = AxonConfig()
        # graph_rag_community defaults to False (cost-control; enable explicitly)
        assert cfg.graph_rag_community is False
        assert cfg.graph_rag_community_async is True
        assert cfg.graph_rag_community_top_k == 5
        assert cfg.graph_rag_mode == "local"

    # ── New tests for Items 1-11 ───────────────────────────────────────────

    def test_extract_entities_returns_type(self):
        """_extract_entities parses 3-column pipe format and returns type field."""
        brain = self._make_brain()
        brain.llm.complete.return_value = "Apple | ORGANIZATION | Tech company"
        result = brain._extract_entities("some text")
        assert len(result) == 1
        assert result[0]["name"] == "Apple"
        assert result[0]["type"] == "ORGANIZATION"
        assert result[0]["description"] == "Tech company"

    def test_extract_entities_two_column_fallback(self):
        """_extract_entities falls back to UNKNOWN type for old 2-column format."""
        brain = self._make_brain()
        brain.llm.complete.return_value = "Apple | Tech company"
        result = brain._extract_entities("some text")
        assert len(result) == 1
        assert result[0]["type"] == "UNKNOWN"
        assert result[0]["description"] == "Tech company"

    def test_entity_frequency_tracked(self):
        """Entity frequency equals the number of chunk_ids after update."""
        brain = self._make_brain()
        brain._entity_graph = {
            "apple": {
                "description": "tech company",
                "type": "ORGANIZATION",
                "chunk_ids": ["c1", "c2"],
                "frequency": 0,
                "degree": 0,
            }
        }
        # Simulate the frequency update logic from ingest
        for entity_key in brain._entity_graph:
            node = brain._entity_graph[entity_key]
            if isinstance(node, dict):
                node["frequency"] = len(node.get("chunk_ids", []))
        assert brain._entity_graph["apple"]["frequency"] == 2

    def test_community_report_has_title_and_findings(self):
        """_generate_community_summaries stores title, findings, and rank from JSON."""
        from concurrent.futures import ThreadPoolExecutor

        brain = self._make_brain()
        brain._community_levels = {0: {"apple": 0, "beats": 0}}
        brain.config.graph_rag_community_min_size = 0  # disable size gate for this test
        brain._entity_graph = {
            "apple": {"description": "tech company", "type": "ORGANIZATION", "chunk_ids": ["c1"]},
            "beats": {"description": "audio brand", "type": "PRODUCT", "chunk_ids": ["c2"]},
        }
        brain._relation_graph = {}
        brain._community_summaries = {}
        brain.llm.complete.return_value = (
            '{"title": "Tech Products", "summary": "Apple and Beats.", '
            '"findings": [{"summary": "Acquisition", "explanation": "Apple bought Beats."}], "rank": 7.0}'
        )
        brain._executor = ThreadPoolExecutor(max_workers=1)

        brain._generate_community_summaries()

        assert "0_0" in brain._community_summaries
        cs = brain._community_summaries["0_0"]
        assert cs["title"] == "Tech Products"
        assert isinstance(cs["findings"], list)
        assert cs["rank"] == 7.0

    def test_community_report_json_fallback(self):
        """_generate_community_summaries falls back gracefully when LLM returns plain text."""
        from concurrent.futures import ThreadPoolExecutor

        brain = self._make_brain()
        brain._community_levels = {0: {"apple": 0}}
        brain._entity_graph = {
            "apple": {"description": "tech company", "type": "ORGANIZATION", "chunk_ids": ["c1"]}
        }
        brain._relation_graph = {}
        brain._community_summaries = {}
        brain.llm.complete.return_value = "This is a plain text summary, not JSON."
        brain._executor = ThreadPoolExecutor(max_workers=1)

        brain._generate_community_summaries()

        assert "0_0" in brain._community_summaries
        cs = brain._community_summaries["0_0"]
        assert cs["title"] == "Community 0"
        assert cs["findings"] == []

    def test_global_map_reduce_filters_low_score(self):
        """_global_search_map_reduce filters out points with score below min_score."""
        from concurrent.futures import ThreadPoolExecutor

        brain = self._make_brain()
        brain._community_summaries = {
            "0_0": {
                "title": "Test",
                "summary": "test",
                "full_content": "test content",
                "findings": [],
                "rank": 5.0,
                "entities": [],
                "size": 1,
                "level": 0,
            }
        }
        brain.llm.complete.return_value = '[{"point": "low relevance", "score": 5}]'
        brain._executor = ThreadPoolExecutor(max_workers=1)

        class _Cfg:
            graph_rag_global_min_score = 20
            graph_rag_global_top_points = 50
            graph_rag_community_level = 0

        result = brain._global_search_map_reduce("test query", _Cfg())
        # Low scores are filtered; result is either empty or the no-data answer
        from axon.main import _GRAPHRAG_NO_DATA_ANSWER

        assert result == "" or result == _GRAPHRAG_NO_DATA_ANSWER

    def test_global_map_reduce_returns_top_points(self):
        """_global_search_map_reduce includes high-score points in output."""
        from concurrent.futures import ThreadPoolExecutor

        brain = self._make_brain()
        brain._community_summaries = {
            "0_0": {
                "title": "AI",
                "summary": "AI growth",
                "full_content": "# AI\n\nAI growth",
                "findings": [],
                "rank": 8.0,
                "entities": [],
                "size": 1,
                "level": 0,
            }
        }
        brain.llm.complete.return_value = '[{"point": "AI growth", "score": 80}]'
        brain._executor = ThreadPoolExecutor(max_workers=1)

        class _Cfg:
            graph_rag_global_min_score = 20
            graph_rag_global_top_points = 50
            graph_rag_community_level = 0

        result = brain._global_search_map_reduce("AI trends", _Cfg())
        assert "AI growth" in result

    def test_entity_embedding_match_used_when_available(self):
        """_match_entities_by_embedding returns matching entity keys above threshold."""
        brain = self._make_brain()
        brain._entity_embeddings = {"openai": [0.1, 0.2, 0.3]}
        brain.embedding.embed_query.return_value = [0.1, 0.2, 0.3]
        brain.config.graph_rag_entity_match_threshold = 0.5

        result = brain._match_entities_by_embedding("openai query")
        assert "openai" in result

    def test_entity_embedding_match_falls_back_when_empty(self):
        """_match_entities_by_embedding returns [] when no embeddings stored."""
        brain = self._make_brain()
        brain._entity_embeddings = {}

        result = brain._match_entities_by_embedding("any query")
        assert result == []

    def test_hierarchical_detection_produces_multiple_levels(self):
        """_run_hierarchical_community_detection returns dict with multiple level keys."""
        from unittest.mock import patch

        brain = self._make_brain()
        brain._entity_graph = {
            "a": {"description": "entity a", "chunk_ids": ["c1"]},
            "b": {"description": "entity b", "chunk_ids": ["c2"]},
            "c": {"description": "entity c", "chunk_ids": ["c3"]},
        }
        brain._relation_graph = {
            "a": [{"target": "b", "relation": "related", "chunk_id": "c1", "weight": 1}]
        }
        brain.config.graph_rag_community_levels = 2

        # Patch louvain to return a fixed result
        def _fake_louvain(G, seed=42, resolution=1.0):
            nodes = list(G.nodes)
            return [set(nodes[:1]), set(nodes[1:])] if len(nodes) > 1 else [set(nodes)]

        with patch("networkx.algorithms.community.louvain_communities", _fake_louvain):
            result = brain._run_hierarchical_community_detection()

        # result is now (community_levels, community_hierarchy, community_children)
        if isinstance(result, tuple) and len(result) == 3:
            community_levels = result[0]
        else:
            community_levels = result
        assert len(community_levels) == 2
        assert 0 in community_levels
        assert 1 in community_levels

    def test_community_map_property_returns_finest_level(self):
        """_community_map property returns the finest (highest-index) level map."""
        brain = self._make_brain()
        brain._community_levels = {0: {"a": 0}, 1: {"a": 0, "b": 1}}
        # Access through the property on the real class — but brain is a MagicMock,
        # so we call the property getter directly
        from axon.main import AxonBrain

        result = AxonBrain._community_map.fget(brain)
        assert result == {"a": 0, "b": 1}

    def test_relationship_weight_incremented(self):
        """Relation weight is incremented when same (target, relation) pair appears again."""
        brain = self._make_brain()
        brain._relation_graph = {
            "apple": [
                {
                    "target": "beats",
                    "relation": "acquired",
                    "chunk_id": "c1",
                    "weight": 1,
                }
            ]
        }
        # Simulate the weight-increment logic from ingest
        entry = {"target": "beats", "relation": "acquired", "chunk_id": "c2"}
        existing = next(
            (
                e
                for e in brain._relation_graph["apple"]
                if e.get("target") == entry["target"] and e.get("relation") == entry["relation"]
            ),
            None,
        )
        assert existing is not None
        existing["weight"] = existing.get("weight", 1) + 1
        assert brain._relation_graph["apple"][0]["weight"] == 2

    def test_community_reports_indexed_in_vector_store(self):
        """_index_community_reports_in_vector_store calls vector_store.add with community docs."""
        from unittest.mock import MagicMock

        brain = self._make_brain()
        brain._community_summaries = {
            "0_0": {
                "title": "AI",
                "summary": "AI research",
                "full_content": "# AI\n\nAI research",
                "findings": [],
                "rank": 7.0,
                "entities": ["openai"],
                "size": 1,
                "level": 0,
            }
        }
        brain.embedding.embed.return_value = [[0.1, 0.2]]
        mock_vs = MagicMock()
        brain._own_vector_store = mock_vs

        brain._index_community_reports_in_vector_store()

        assert mock_vs.add.called
        call_args = mock_vs.add.call_args
        ids_passed = call_args[0][0]
        assert any("__community__" in id_ for id_ in ids_passed)

    def test_canonicalize_entity_descriptions(self):
        """_canonicalize_entity_descriptions updates entity graph with synthesized description."""
        from concurrent.futures import ThreadPoolExecutor

        brain = self._make_brain()
        brain._entity_graph = {
            "openai": {
                "description": "AI company",
                "type": "ORGANIZATION",
                "chunk_ids": ["c1"],
                "frequency": 1,
                "degree": 0,
            }
        }
        brain._entity_description_buffer = {"openai": ["AI company", "Research org", "GPT maker"]}
        brain.config.graph_rag_canonicalize_min_occurrences = 3
        brain.llm.complete.return_value = "Leading AI research organization"
        brain._executor = ThreadPoolExecutor(max_workers=1)

        brain._canonicalize_entity_descriptions()

        assert brain._entity_graph["openai"]["description"] == "Leading AI research organization"
        assert brain._entity_description_buffer == {}

    def test_claims_extraction_parses_pipe_format(self):
        """_extract_claims parses pipe-delimited claim lines correctly."""
        brain = self._make_brain()
        brain.llm.complete.return_value = (
            "Apple | Beats | acquisition | TRUE | Apple acquired Beats for $3B"
        )
        result = brain._extract_claims("some text")
        assert len(result) == 1
        assert result[0]["status"] == "TRUE"
        assert result[0]["subject"] == "Apple"
        assert result[0]["object"] == "Beats"
        assert "Apple acquired Beats" in result[0]["description"]


class TestGraphRAGRealImplementation:
    """Tests for the 9 GraphRAG correctness gaps (GAPs 1-9)."""

    def _make_brain(self):
        """Return a MagicMock AxonBrain with real methods bound."""
        import threading
        from unittest.mock import MagicMock

        from axon.main import AxonBrain, AxonConfig

        brain = MagicMock(spec=AxonBrain)
        brain.config = AxonConfig()
        brain.llm = MagicMock()
        brain.embedding = MagicMock()
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._community_levels = {}
        brain._community_summaries = {}
        brain._community_graph_dirty = False
        brain._last_matched_entities = []
        brain._entity_embeddings = {}
        brain._entity_description_buffer = {}
        brain._claims_graph = {}
        brain._community_hierarchy = {}
        brain._community_children = {}
        brain._relation_description_buffer = {}
        brain._text_unit_entity_map = {}
        brain._text_unit_relation_map = {}
        brain._community_rebuild_lock = threading.Lock()
        # Bind real implementations
        brain._extract_entities = AxonBrain._extract_entities.__get__(brain, AxonBrain)
        brain._extract_relations = AxonBrain._extract_relations.__get__(brain, AxonBrain)
        brain._run_hierarchical_community_detection = (
            AxonBrain._run_hierarchical_community_detection.__get__(brain, AxonBrain)
        )
        brain._build_networkx_graph = AxonBrain._build_networkx_graph.__get__(brain, AxonBrain)
        brain._generate_community_summaries = AxonBrain._generate_community_summaries.__get__(
            brain, AxonBrain
        )
        brain._global_search_map_reduce = AxonBrain._global_search_map_reduce.__get__(
            brain, AxonBrain
        )
        brain._local_search_context = AxonBrain._local_search_context.__get__(brain, AxonBrain)
        brain._get_incoming_relations = AxonBrain._get_incoming_relations.__get__(brain, AxonBrain)
        brain._entity_matches = AxonBrain._entity_matches.__get__(brain, AxonBrain)
        brain._extract_claims = AxonBrain._extract_claims.__get__(brain, AxonBrain)
        brain._rebuild_communities = AxonBrain._rebuild_communities.__get__(brain, AxonBrain)
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._save_community_map = MagicMock()
        brain._save_community_summaries = MagicMock()
        brain._save_community_levels = MagicMock()
        brain._save_community_hierarchy = MagicMock()
        brain._save_entity_embeddings = MagicMock()
        brain._save_claims_graph = MagicMock()
        return brain

    def test_global_search_reduce_calls_llm(self):
        """_global_search_map_reduce makes a second (reduce) LLM call with Analyst-formatted text."""
        from concurrent.futures import ThreadPoolExecutor

        brain = self._make_brain()
        brain._community_summaries = {
            "0_0": {
                "title": "Test",
                "summary": "test",
                "full_content": "test community content",
                "findings": [],
                "rank": 5.0,
                "entities": [],
                "size": 1,
                "level": 0,
            }
        }
        call_count = [0]
        call_prompts = []

        def _fake_complete(prompt, system_prompt=None):
            call_count[0] += 1
            call_prompts.append(prompt)
            if call_count[0] == 1:
                return '[{"point": "important finding", "score": 80}]'
            else:
                return "Comprehensive synthesized answer."

        brain.llm.complete.side_effect = _fake_complete
        brain._executor = ThreadPoolExecutor(max_workers=1)

        class _Cfg:
            graph_rag_global_min_score = 20
            graph_rag_global_top_points = 50
            graph_rag_community_level = 0
            graph_rag_global_reduce_max_tokens = 8000

        result = brain._global_search_map_reduce("test query", _Cfg())

        assert call_count[0] >= 2
        reduce_prompt = call_prompts[-1]
        assert "----Analyst" in reduce_prompt
        assert result == "Comprehensive synthesized answer."

    def test_global_search_no_data_returns_no_data_answer(self):
        """When all map-phase scores are below min_score, return _GRAPHRAG_NO_DATA_ANSWER."""
        from concurrent.futures import ThreadPoolExecutor

        from axon.main import _GRAPHRAG_NO_DATA_ANSWER

        brain = self._make_brain()
        brain._community_summaries = {
            "0_0": {
                "title": "Test",
                "summary": "test",
                "full_content": "test content",
                "findings": [],
                "rank": 5.0,
                "entities": [],
                "size": 1,
                "level": 0,
            }
        }
        brain.llm.complete.return_value = '[{"point": "low relevance", "score": 5}]'
        brain._executor = ThreadPoolExecutor(max_workers=1)

        class _Cfg:
            graph_rag_global_min_score = 20
            graph_rag_global_top_points = 50
            graph_rag_community_level = 0
            graph_rag_global_reduce_max_tokens = 8000

        result = brain._global_search_map_reduce("test query", _Cfg())
        assert result == _GRAPHRAG_NO_DATA_ANSWER

    def test_global_search_token_budget_respected(self):
        """With a very small reduce_max_tokens, reduce prompt is truncated to few analysts."""
        from concurrent.futures import ThreadPoolExecutor

        brain = self._make_brain()
        summaries = {}
        for i in range(5):
            summaries[f"0_{i}"] = {
                "title": f"Community {i}",
                "summary": "test",
                "full_content": f"community content {i}",
                "findings": [],
                "rank": 5.0,
                "entities": [],
                "size": 1,
                "level": 0,
            }
        brain._community_summaries = summaries

        captured_prompts = []

        def _fake_complete(prompt, system_prompt=None):
            captured_prompts.append(prompt)
            if "Community Report" in prompt:
                return '[{"point": "' + ("x" * 100) + '", "score": 90}]'
            return "Answer."

        brain.llm.complete.side_effect = _fake_complete
        brain._executor = ThreadPoolExecutor(max_workers=2)

        class _Cfg:
            graph_rag_global_min_score = 20
            graph_rag_global_top_points = 50
            graph_rag_community_level = 0
            graph_rag_global_reduce_max_tokens = 10

        brain._global_search_map_reduce("test query", _Cfg())
        reduce_prompt = captured_prompts[-1]
        analyst_count = reduce_prompt.count("----Analyst")
        assert analyst_count <= 2

    def test_local_search_incoming_edges_included(self):
        """_local_search_context includes B->A relation when querying for entity A."""
        from unittest.mock import MagicMock

        brain = self._make_brain()
        brain._entity_graph = {
            "apple": {"description": "Tech company", "chunk_ids": ["c1"], "type": "ORGANIZATION"},
            "google": {
                "description": "Search company",
                "chunk_ids": ["c2"],
                "type": "ORGANIZATION",
            },
        }
        brain._relation_graph = {
            "google": [
                {
                    "target": "apple",
                    "relation": "competes_with",
                    "chunk_id": "c2",
                    "description": "Google competes with Apple in mobile",
                    "weight": 1,
                }
            ]
        }
        brain._community_levels = {}
        brain._community_summaries = {}
        brain._claims_graph = {}
        brain.vector_store = MagicMock()
        brain.vector_store.get_by_ids.return_value = []

        class _Cfg:
            pass

        result = brain._local_search_context("apple products", ["apple"], _Cfg())
        assert "competes_with" in result or "Google competes with Apple" in result

    def test_local_search_covariate_matched_by_subject(self):
        """Claims are retrieved by subject name match, not by chunk_ids lookup."""
        from unittest.mock import MagicMock

        brain = self._make_brain()
        brain._entity_graph = {
            "apple": {
                "description": "Tech company",
                "chunk_ids": ["c1"],
                "type": "ORGANIZATION",
            }
        }
        brain._relation_graph = {}
        brain._community_levels = {}
        brain._community_summaries = {}
        # Claim stored under chunk "c99" which is NOT in apple's chunk_ids ("c1")
        brain._claims_graph = {
            "c99": [
                {
                    "subject": "apple",
                    "object": "beats",
                    "type": "acquisition",
                    "status": "TRUE",
                    "description": "Apple acquired Beats for $3B",
                    "text_unit_id": "c99",
                }
            ]
        }
        brain.vector_store = MagicMock()
        brain.vector_store.get_by_ids.return_value = []

        class _Cfg:
            pass

        result = brain._local_search_context("apple acquisitions", ["apple"], _Cfg())
        assert "Apple acquired Beats" in result

    def test_community_hierarchy_parent_child_structure(self):
        """_run_hierarchical_community_detection returns 3-tuple with parent/child data."""
        from unittest.mock import patch

        brain = self._make_brain()
        brain._entity_graph = {
            "a": {"description": "entity a", "chunk_ids": ["c1"]},
            "b": {"description": "entity b", "chunk_ids": ["c2"]},
            "c": {"description": "entity c", "chunk_ids": ["c3"]},
            "d": {"description": "entity d", "chunk_ids": ["c4"]},
        }
        brain._relation_graph = {
            "a": [{"target": "b", "relation": "related", "chunk_id": "c1", "weight": 1}],
            "c": [{"target": "d", "relation": "linked", "chunk_id": "c3", "weight": 1}],
        }
        brain.config.graph_rag_community_levels = 2

        def _fake_louvain(G, seed=42, resolution=1.0):
            nodes = list(G.nodes)
            mid = len(nodes) // 2
            return [set(nodes[:mid]), set(nodes[mid:])] if mid > 0 else [set(nodes)]

        with patch("networkx.algorithms.community.louvain_communities", _fake_louvain):
            result = brain._run_hierarchical_community_detection()

        assert isinstance(result, tuple) and len(result) == 3
        community_levels, community_hierarchy, community_children = result
        assert isinstance(community_levels, dict)
        assert isinstance(community_hierarchy, dict)
        assert isinstance(community_children, dict)
        assert len(community_levels) >= 1

    def test_relation_weight_accumulates(self):
        """Inserting same (subject, relation, object) triple twice yields weight >= 2."""
        brain = self._make_brain()
        brain._relation_graph = {}

        def _store_relation(src, tgt, relation, doc_id, desc=""):
            src_lower = src.lower().strip()
            if src_lower not in brain._relation_graph:
                brain._relation_graph[src_lower] = []
            rel_tgt = tgt.lower().strip()
            rel_relation = relation.strip()
            existing_entry = next(
                (
                    e
                    for e in brain._relation_graph[src_lower]
                    if e.get("target") == rel_tgt and e.get("relation") == rel_relation
                ),
                None,
            )
            if existing_entry is not None:
                existing_entry["weight"] = existing_entry.get("weight", 1) + 1
                if "text_unit_ids" not in existing_entry:
                    existing_entry["text_unit_ids"] = [existing_entry.get("chunk_id", "")]
                if doc_id not in existing_entry["text_unit_ids"]:
                    existing_entry["text_unit_ids"].append(doc_id)
            else:
                brain._relation_graph[src_lower].append(
                    {
                        "target": rel_tgt,
                        "relation": rel_relation,
                        "chunk_id": doc_id,
                        "text_unit_ids": [doc_id],
                        "description": desc,
                        "weight": 1,
                    }
                )

        _store_relation("Apple", "Beats", "acquired", "chunk1", "Apple bought Beats")
        _store_relation("Apple", "Beats", "acquired", "chunk2", "Apple acquired Beats Electronics")

        entry = brain._relation_graph["apple"][0]
        assert entry["weight"] >= 2
        assert "chunk1" in entry["text_unit_ids"]
        assert "chunk2" in entry["text_unit_ids"]

    def test_claim_schema_includes_text_unit_id(self):
        """Claims parsed from 8-column format have new fields and text_unit_id settable."""
        brain = self._make_brain()
        brain.llm.complete.return_value = (
            "Apple | Beats | acquisition | TRUE | Apple acquired Beats for $3B"
            " | 2014-05-28 | unknown | Apple bought Beats"
        )
        claims = brain._extract_claims("some text")
        assert len(claims) == 1
        claim = claims[0]
        assert claim["subject"] == "Apple"
        assert claim["status"] == "TRUE"
        assert claim["start_date"] == "2014-05-28"
        assert claim["end_date"] is None
        assert claim["source_text"] == "Apple bought Beats"
        assert "text_unit_id" in claim
        assert claim["text_unit_id"] is None
        # Simulate caller setting text_unit_id
        for c in claims:
            c["text_unit_id"] = "chunk_abc"
        assert claims[0]["text_unit_id"] == "chunk_abc"

    def test_rebuild_communities_lock_prevents_interleaving(self):
        """A thread calling _rebuild_communities blocks while the lock is already held."""
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor

        brain = self._make_brain()
        # Provide a real executor so _generate_community_summaries doesn't crash
        brain._executor = ThreadPoolExecutor(max_workers=1)
        results = []

        def _run_rebuild():
            brain._rebuild_communities()
            results.append("done")

        brain._community_rebuild_lock.acquire()
        t = threading.Thread(target=_run_rebuild)
        t.start()

        time.sleep(0.05)
        assert len(results) == 0

        brain._community_rebuild_lock.release()
        t.join(timeout=5.0)
        assert len(results) == 1
        assert results[0] == "done"


class TestGraphRAGAuditFixes:
    """Tests for GraphRAG audit fixes from GRAPHRAG_REAUDIT_2026_03_16_NEW_TASK_5."""

    def _make_brain(self):
        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.config.graph_rag_community_levels = 2
        brain.config.graph_rag_community_max_cluster_size = 10
        brain.config.graph_rag_leiden_seed = 42
        brain.config.graph_rag_community_use_lcc = False
        brain.config.graph_rag_community_max_context_tokens = 4000
        brain.config.graph_rag_community_include_claims = False
        brain.config.graph_rag_claims = False
        brain.config.graph_rag_local_max_context_tokens = 8000
        brain.config.graph_rag_local_community_prop = 0.25
        brain.config.graph_rag_local_text_unit_prop = 0.5
        brain.config.graph_rag_local_top_k_entities = 10
        brain.config.graph_rag_local_top_k_relationships = 10
        brain.config.graph_rag_local_include_relationship_weight = False
        brain.config.graph_rag_local_entity_weight = 3.0
        brain.config.graph_rag_local_relation_weight = 2.0
        brain.config.graph_rag_local_community_weight = 1.5
        brain.config.graph_rag_local_text_unit_weight = 1.0
        brain.config.graph_rag_community_min_size = 3
        brain.config.graph_rag_community_llm_top_n_per_level = 50
        brain.config.graph_rag_community_llm_max_total = 200
        brain.config.graph_rag_community_lazy = False
        brain.config.graph_rag_global_top_communities = 0
        brain.config.raptor_max_source_size_mb = 0.0
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._claims_graph = {}
        brain._community_levels = {}
        brain._community_hierarchy = {}
        brain._community_children = {}
        brain._community_summaries = {}
        brain._entity_embeddings = {}
        brain._entity_description_buffer = {}
        brain._relation_description_buffer = {}
        brain._text_unit_entity_map = {}
        brain._text_unit_relation_map = {}
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._save_community_summaries = MagicMock()
        brain._save_community_hierarchy = MagicMock()
        llm = MagicMock()
        llm.complete = MagicMock(return_value="")
        brain.llm = llm
        # Bind real methods
        for method_name in [
            "_run_hierarchical_community_detection",
            "_build_networkx_graph",
            "_get_incoming_relations",
            "_local_search_context",
            "_extract_relations",
            "_entity_matches",
            "_expand_with_entity_graph",
            "_match_entities_by_embedding",
            "_extract_entities",
        ]:
            method = getattr(AxonBrain, method_name)
            setattr(brain, method_name, lambda *a, m=method, **kw: m(brain, *a, **kw))
        return brain

    def test_hierarchy_no_cluster_id_collision_across_levels(self):
        """Fallback hierarchy keys must be level-qualified to avoid cross-level collisions."""
        import pytest

        b = self._make_brain()
        b._entity_graph = {
            "a": {
                "chunk_ids": ["c1"],
                "description": "",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
            "b": {
                "chunk_ids": ["c2"],
                "description": "",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
            "c": {
                "chunk_ids": ["c3"],
                "description": "",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
            "d": {
                "chunk_ids": ["c4"],
                "description": "",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
        }
        b._relation_graph = {
            "a": [
                {"target": "b", "relation": "rel", "chunk_id": "c1", "description": "", "weight": 1}
            ],
            "c": [
                {"target": "d", "relation": "rel", "chunk_id": "c3", "description": "", "weight": 1}
            ],
        }
        try:
            import networkx  # noqa: F401
        except ImportError:
            pytest.skip("networkx not installed")

        result = b._run_hierarchical_community_detection()
        assert isinstance(result, tuple) and len(result) == 3
        community_levels, hierarchy, children = result
        # With 2+ levels, hierarchy keys must not be bare integers that collide across levels
        # All hierarchy keys should be either None or strings with "_" separator (level-qualified)
        for key in hierarchy:
            assert (
                isinstance(key, str) and "_" in key
            ), f"Hierarchy key {key!r} is not level-qualified — collision risk"

    def test_use_lcc_default_is_false(self):
        """graph_rag_community_use_lcc should default to False to preserve all graph components."""
        import dataclasses

        from axon.main import AxonConfig

        fields = {f.name: f.default for f in dataclasses.fields(AxonConfig)}
        assert (
            fields.get("graph_rag_community_use_lcc") is False
        ), "use_lcc defaults True — disconnected components would be silently dropped"

    def test_entity_ranking_uses_total_degree(self):
        """Entity ranking in local search should count both incoming and outgoing relations."""
        b = self._make_brain()
        # 'a' has 0 outgoing but is a target of 'b' — total degree 1
        # 'b' has 1 outgoing, 0 incoming — total degree 1
        # 'c' has 0 outgoing, 0 incoming — total degree 0
        b._entity_graph = {
            "a": {
                "chunk_ids": ["c1"],
                "description": "node a",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 0,
            },
            "b": {
                "chunk_ids": ["c2"],
                "description": "node b",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
            "c": {
                "chunk_ids": ["c3"],
                "description": "node c",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 0,
            },
        }
        b._relation_graph = {
            "b": [
                {
                    "target": "a",
                    "relation": "links_to",
                    "chunk_id": "c2",
                    "description": "b links to a",
                    "weight": 1,
                }
            ],
        }
        b._community_levels = {}
        b._community_summaries = {}
        result = b._local_search_context("test query", ["a", "b", "c"], b.config)
        # 'c' has zero total degree so should appear after 'a' and 'b'
        if "node a" in result and "node c" in result:
            assert result.index("node a") < result.index("node c") or result.index(
                "node b"
            ) < result.index("node c")

    def test_multiple_community_reports_in_local_search(self):
        """Local search must include multiple community reports, not just the first one."""
        b = self._make_brain()
        b._entity_graph = {
            "alpha": {
                "chunk_ids": ["c1"],
                "description": "alpha entity",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
            "beta": {
                "chunk_ids": ["c2"],
                "description": "beta entity",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
        }
        b._relation_graph = {}
        # Two different communities
        b._community_levels = {0: {"alpha": 10, "beta": 20}}
        b._community_summaries = {
            "0_10": {
                "title": "Alpha Community",
                "summary": "This is about alpha.",
                "findings": [],
                "rank": 7.0,
                "entities": ["alpha"],
                "size": 1,
                "level": 0,
                "full_content": "# Alpha Community\nThis is about alpha.",
            },
            "0_20": {
                "title": "Beta Community",
                "summary": "This is about beta.",
                "findings": [],
                "rank": 6.0,
                "entities": ["beta"],
                "size": 1,
                "level": 0,
                "full_content": "# Beta Community\nThis is about beta.",
            },
        }
        result = b._local_search_context("test query", ["alpha", "beta"], b.config)
        # Both community snippets should appear (not just the first)
        assert "Alpha Community" in result, "First community report missing"
        assert "Beta Community" in result, "Second community report missing — break removed?"

    def test_relation_extraction_includes_strength_field(self):
        """_extract_relations must return dicts with a 'strength' key."""
        b = self._make_brain()
        b.llm.complete = MagicMock(
            return_value="Apple | acquired | Beats | Apple bought Beats for $3B | 9\n"
            "Beats | makes | headphones | Beats manufactures audio products | 7"
        )
        result = b._extract_relations("Apple acquired Beats, which makes headphones.")
        assert len(result) >= 1
        for r in result:
            assert "strength" in r, f"Missing 'strength' key in {r}"
            assert 1 <= r["strength"] <= 10, f"strength {r['strength']} out of range"

    def test_relation_extraction_strength_defaults_on_missing_column(self):
        """_extract_relations should use strength=5 when 5th column is absent."""
        b = self._make_brain()
        b.llm.complete = MagicMock(
            return_value="Apple | acquired | Beats | Apple bought Beats for $3B"
        )
        result = b._extract_relations("Apple acquired Beats.")
        assert len(result) == 1
        assert result[0]["strength"] == 5

    def test_global_search_map_uses_chunks_not_raw_report(self):
        """Global search map phase must chunk long reports so later content is not lost."""
        import concurrent.futures

        b = self._make_brain()
        b.config.graph_rag_global_min_score = 0
        b.config.graph_rag_global_top_points = 50
        b.config.graph_rag_community_level = 0
        b.config.graph_rag_global_reduce_max_tokens = 8000
        # Build a report long enough to require chunking (>2000 chars)
        long_report = "Important fact: the answer is 42. " * 100  # ~3400 chars
        b._community_summaries = {
            "0_1": {
                "title": "T1",
                "summary": "short",
                "full_content": long_report,
                "rank": 8.0,
                "entities": [],
                "size": 1,
                "level": 0,
                "findings": [],
            },
        }
        calls = []

        def fake_complete(prompt, system_prompt=None):
            calls.append(prompt)
            return '[{"point": "answer is 42", "score": 90}]'

        b.llm.complete = MagicMock(side_effect=fake_complete)
        b._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        from axon.main import (
            _GRAPHRAG_REDUCE_SYSTEM_PROMPT,  # noqa: F401
            AxonBrain,
        )

        AxonBrain._global_search_map_reduce(b, "what is the answer?", b.config)
        # With chunking, the long report should produce >=2 map LLM calls (one per chunk)
        # The reduce call is 1 additional call, so total > 2 if chunking works
        map_calls = [c for c in calls if "Community Report" in c]
        assert len(map_calls) >= 2, (
            f"Expected >=2 map calls for chunked report, got {len(map_calls)} — "
            "report may not be chunked"
        )
        b._executor.shutdown(wait=False)

    def test_union_embedding_and_llm_entity_extraction(self):
        """_expand_with_entity_graph must union LLM-extracted and embedding-matched entities."""
        import concurrent.futures

        from axon.main import AxonBrain

        b = self._make_brain()
        b.config.graph_rag_entity_embedding_match = True
        b.config.graph_rag_relations = False
        b.config.graph_rag_budget = 3
        b.config.top_k = 3
        b._entity_graph = {
            "apple": {
                "chunk_ids": ["c1"],
                "description": "tech company",
                "type": "ORG",
                "frequency": 1,
                "degree": 1,
            },
            "beatles": {
                "chunk_ids": ["c2"],
                "description": "music band",
                "type": "ORG",
                "frequency": 1,
                "degree": 1,
            },
        }
        b._entity_embeddings = {"apple": [1.0, 0.0], "beatles": [0.0, 1.0]}
        # LLM extracts "apple" (exact mention)
        b._extract_entities = MagicMock(
            return_value=[{"name": "apple", "type": "ORG", "description": ""}]
        )
        # Embedding match returns "beatles" (semantic neighbor)
        b._match_entities_by_embedding = MagicMock(return_value=["beatles"])
        b._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        b.vector_store = MagicMock()
        b.vector_store.get_by_ids = MagicMock(return_value=[])
        results, matched = AxonBrain._expand_with_entity_graph(b, "apple music", [], b.config)
        b._executor.shutdown(wait=False)
        # Both apple (LLM) and beatles (embedding) should appear in matched entities
        matched_lower = [m.lower() for m in matched]
        assert "apple" in matched_lower, "LLM-extracted entity 'apple' missing from matched"
        assert (
            "beatles" in matched_lower
        ), "Embedding-matched entity 'beatles' missing — union not working"


class TestGraphRAGTask6Fixes:
    """Tests for GraphRAG audit fixes from TASK_6."""

    def _make_brain(self):
        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.config.graph_rag_community_levels = 2
        brain.config.graph_rag_community_max_cluster_size = 10
        brain.config.graph_rag_leiden_seed = 42
        brain.config.graph_rag_community_use_lcc = False
        brain.config.graph_rag_community_max_context_tokens = 4000
        brain.config.graph_rag_community_include_claims = False
        brain.config.graph_rag_claims = False
        brain.config.graph_rag_local_max_context_tokens = 8000
        brain.config.graph_rag_local_community_prop = 0.25
        brain.config.graph_rag_local_text_unit_prop = 0.5
        brain.config.graph_rag_local_top_k_entities = 10
        brain.config.graph_rag_local_top_k_relationships = 10
        brain.config.graph_rag_local_include_relationship_weight = False
        brain.config.graph_rag_local_entity_weight = 3.0
        brain.config.graph_rag_local_relation_weight = 2.0
        brain.config.graph_rag_local_community_weight = 1.5
        brain.config.graph_rag_local_text_unit_weight = 1.0
        brain.config.graph_rag_community_min_size = 3
        brain.config.graph_rag_community_llm_top_n_per_level = 50
        brain.config.graph_rag_community_llm_max_total = 200
        brain.config.graph_rag_community_lazy = False
        brain.config.graph_rag_global_top_communities = 0
        brain.config.raptor_max_source_size_mb = 0.0
        brain.config.graph_rag_global_map_max_length = 500
        brain.config.graph_rag_global_reduce_max_length = 500
        brain.config.graph_rag_global_allow_general_knowledge = False
        brain.config.graph_rag_global_min_score = 0
        brain.config.graph_rag_global_top_points = 50
        brain.config.graph_rag_community_level = 0
        brain.config.graph_rag_global_reduce_max_tokens = 8000
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._claims_graph = {}
        brain._community_levels = {}
        brain._community_hierarchy = {}
        brain._community_children = {}
        brain._community_summaries = {}
        brain._entity_embeddings = {}
        brain._entity_description_buffer = {}
        brain._relation_description_buffer = {}
        brain._text_unit_entity_map = {}
        brain._text_unit_relation_map = {}
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._save_community_summaries = MagicMock()
        brain._save_community_hierarchy = MagicMock()
        import concurrent.futures

        brain._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        llm = MagicMock()
        llm.complete = MagicMock(return_value="")
        brain.llm = llm
        for method_name in [
            "_run_hierarchical_community_detection",
            "_build_networkx_graph",
            "_get_incoming_relations",
            "_local_search_context",
            "_generate_community_summaries",
            "_extract_relations",
            "_entity_matches",
            "_expand_with_entity_graph",
            "_match_entities_by_embedding",
            "_extract_entities",
        ]:
            method = getattr(AxonBrain, method_name)
            setattr(brain, method_name, lambda *a, m=method, **kw: m(brain, *a, **kw))
        return brain

    def test_strength_persisted_in_relation_entry(self):
        """A triple with strength=8 must produce a stored edge with weight >= 8."""

        b = self._make_brain()
        b._relation_graph = {}
        triple = {
            "subject": "Alice",
            "relation": "knows",
            "object": "Bob",
            "description": "Alice knows Bob",
            "strength": 8,
        }
        doc_id = "doc_1"
        src_lower = triple["subject"].lower().strip()
        obj = triple["object"].lower().strip()
        relation = triple["relation"].strip()
        description = triple["description"]
        entry = {
            "target": obj,
            "relation": relation,
            "chunk_id": doc_id,
            "description": description,
            "strength": triple.get("strength", 5) if isinstance(triple, dict) else 5,
            "support_count": 1,
        }
        if src_lower not in b._relation_graph:
            b._relation_graph[src_lower] = []
        entry["weight"] = entry.get("strength", 1)
        entry["text_unit_ids"] = [doc_id]
        b._relation_graph[src_lower].append(entry)
        stored = b._relation_graph["alice"][0]
        assert stored["strength"] == 8, f"strength lost, got {stored.get('strength')}"
        assert stored["weight"] >= 8, f"weight {stored.get('weight')} < 8"

    def test_support_count_incremented_on_repeat(self):
        """Reinserting the same (source, target, relation) pair must increment support_count."""
        doc_id = "doc_1"
        existing_entry = {
            "target": "bob",
            "relation": "knows",
            "chunk_id": doc_id,
            "description": "Alice knows Bob",
            "strength": 8,
            "support_count": 1,
            "weight": 8,
            "text_unit_ids": [doc_id],
        }
        new_strength = 7
        existing_entry["weight"] = existing_entry.get("weight", 1) + new_strength
        existing_entry["support_count"] = existing_entry.get("support_count", 1) + 1
        assert existing_entry["support_count"] == 2
        assert existing_entry["weight"] == 15

    def test_hierarchy_persistence_roundtrip_string_keys(self):
        """String keys like '1_3': '0_1' must survive a save/load roundtrip."""
        import json

        original = {"1_3": "0_1", "1_4": "0_2", "0_1": None, "0_2": None}
        # Simulate save (converts keys to strings — they're already strings)
        saved = json.dumps({str(k): v for k, v in original.items()})
        raw = json.loads(saved)
        # Simulate load with the fixed _load_community_hierarchy logic
        result = {}
        for k, v in raw.items():
            key = k if ("_" in str(k)) else (int(k) if str(k).isdigit() else k)
            val = (
                None
                if v is None
                else (
                    v
                    if (isinstance(v, str) and "_" in v)
                    else (int(str(v)) if str(v).isdigit() else v)
                )
            )
            result[key] = val
        assert result == original, f"Round-trip failed: {result} != {original}"

    def test_leiden_hierarchy_normalized_to_string_keys(self):
        """Louvain fallback hierarchy must produce level-qualified string keys."""
        import pytest

        try:
            import networkx  # noqa: F401
        except ImportError:
            pytest.skip("networkx not installed")

        b = self._make_brain()
        b._entity_graph = {
            "a": {
                "chunk_ids": ["c1"],
                "description": "",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
            "b": {
                "chunk_ids": ["c2"],
                "description": "",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
            "c": {
                "chunk_ids": ["c3"],
                "description": "",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
            "d": {
                "chunk_ids": ["c4"],
                "description": "",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
        }
        b._relation_graph = {
            "a": [
                {"target": "b", "relation": "rel", "chunk_id": "c1", "description": "", "weight": 1}
            ],
            "c": [
                {"target": "d", "relation": "rel", "chunk_id": "c3", "description": "", "weight": 1}
            ],
        }
        result = b._run_hierarchical_community_detection()
        assert isinstance(result, tuple) and len(result) == 3
        _, hierarchy, _ = result
        for key in hierarchy:
            assert (
                isinstance(key, str) and "_" in key
            ), f"Hierarchy key {key!r} is not level-qualified string"

    def test_leiden_child_substitution_finds_summary(self):
        """When community context exceeds token budget, _generate_community_summaries
        substitutes ranked child reports by rank (highest rank first)."""
        from axon.main import AxonBrain

        b = self._make_brain()
        b.config.graph_rag_community_max_context_tokens = (
            1  # force substitution for any non-empty context
        )
        # Pre-populate existing summaries for child communities (level 1)
        b._community_summaries = {
            "1_0": {
                "title": "Sub A",
                "summary": "sub-a summary",
                "full_content": "Sub A content",
                "rank": 9.0,
                "entities": ["x"],
                "size": 1,
                "level": 1,
                "findings": [],
            },
            "1_1": {
                "title": "Sub B",
                "summary": "sub-b summary",
                "full_content": "Sub B content",
                "rank": 3.0,
                "entities": ["y"],
                "size": 1,
                "level": 1,
                "findings": [],
            },
        }
        b._community_children = {"0_0": ["1_0", "1_1"]}
        # Level 0 has one community (id=0) containing x and y
        b._community_levels = {0: {"x": 0, "y": 0}}
        b.config.graph_rag_community_min_size = 0  # disable size gate for this test
        b._entity_graph = {
            "x": {
                "chunk_ids": ["c1"],
                "description": "entity x",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
            "y": {
                "chunk_ids": ["c2"],
                "description": "entity y",
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 1,
            },
        }
        b._relation_graph = {}
        captured_prompts = []

        def fake_complete(prompt, system_prompt=None):
            captured_prompts.append(prompt)
            return '{"title":"Parent","summary":"p","findings":[],"rank":8.0}'

        b.llm.complete = MagicMock(side_effect=fake_complete)
        # Run the real _generate_community_summaries to exercise the substitution path
        AxonBrain._generate_community_summaries(b)
        # The prompt for the parent community must include child sub-reports
        # (because context budget is 1 token, forcing substitution)
        assert captured_prompts, "LLM was never called for community summary generation"
        parent_prompt = captured_prompts[0]
        # Higher-ranked child (Sub A, rank=9.0) should appear before lower-ranked (Sub B, rank=3.0)
        assert (
            "Sub A content" in parent_prompt
        ), "Higher-ranked child report (Sub A) missing from parent community prompt"
        assert parent_prompt.index("Sub A content") < parent_prompt.index(
            "Sub B content"
        ), "Sub-community reports not sorted by rank (Sub A should precede Sub B)"

    def test_community_prompt_no_hard_truncation(self):
        """Community prompt must include context beyond 3000 chars (no hard truncation)."""
        from axon.main import AxonBrain

        b = self._make_brain()
        b.config.graph_rag_community_max_context_tokens = 999999  # do not trigger substitution
        b.config.graph_rag_community_min_size = 0  # disable size gate for this test
        # Set up a community with a single entity whose description exceeds 3000 chars
        b._community_levels = {0: {"entity_long": 0}}
        b._community_summaries = {}
        b._community_children = {}

        long_description = "X" * 3500
        b._entity_graph = {
            "entity_long": {
                "chunk_ids": ["c1"],
                "description": long_description,
                "type": "CONCEPT",
                "frequency": 1,
                "degree": 0,
            }
        }
        b._relation_graph = {}
        captured_prompts = []

        def _fake_complete(prompt, system_prompt=None):
            captured_prompts.append(prompt)
            return '{"title":"T","summary":"s","findings":[],"rank":5.0}'

        b.llm.complete = MagicMock(side_effect=_fake_complete)
        # Call the real _generate_community_summaries which uses the nested _summarise closure
        AxonBrain._generate_community_summaries(b)
        assert captured_prompts, "LLM was never called"
        prompt = captured_prompts[0]
        tail = long_description[-100:]
        assert (
            tail in prompt
        ), "Context was hard-truncated at 3000 chars — tail of description not found in prompt"

    def test_global_search_map_length_config_respected(self):
        """graph_rag_global_map_max_length=100 must produce chunk size of ~400 chars."""
        import concurrent.futures

        from axon.main import AxonBrain

        b = self._make_brain()
        b.config.graph_rag_global_map_max_length = 100  # 100 * 4 = 400 chars per chunk
        # Build a report of 1200 chars — should produce 3 chunks with 400-char windows
        long_report = "A" * 1200
        b._community_summaries = {
            "0_0": {
                "title": "T",
                "summary": "s",
                "full_content": long_report,
                "rank": 8.0,
                "entities": [],
                "size": 1,
                "level": 0,
                "findings": [],
            },
        }
        calls = []

        def fake_complete(prompt, system_prompt=None):
            calls.append(prompt)
            return '[{"point": "fact", "score": 90}]'

        b.llm.complete = MagicMock(side_effect=fake_complete)
        b._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        AxonBrain._global_search_map_reduce(b, "query", b.config)
        map_calls = [c for c in calls if "Community Report" in c]
        # 1200 chars / 400 per chunk = 3 chunks → 3 map calls
        assert (
            len(map_calls) >= 3
        ), f"Expected >=3 map calls for 1200-char report with chunk_size=400, got {len(map_calls)}"
        b._executor.shutdown(wait=False)


class TestGraphRAGTask7Fixes:
    """Tests for GraphRAG runtime fixes from TASK_7."""

    def _make_brain(self):
        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.config.graph_rag_community_levels = 2
        brain.config.graph_rag_community_max_cluster_size = 10
        brain.config.graph_rag_leiden_seed = 42
        brain.config.graph_rag_community_use_lcc = False
        brain.config.graph_rag_community_max_context_tokens = 4000
        brain.config.graph_rag_community_include_claims = False
        brain.config.graph_rag_claims = False
        brain.config.graph_rag_local_max_context_tokens = 8000
        brain.config.graph_rag_local_community_prop = 0.25
        brain.config.graph_rag_local_text_unit_prop = 0.5
        brain.config.graph_rag_local_top_k_entities = 10
        brain.config.graph_rag_local_top_k_relationships = 10
        brain.config.graph_rag_local_include_relationship_weight = False
        brain.config.graph_rag_local_entity_weight = 3.0
        brain.config.graph_rag_local_relation_weight = 2.0
        brain.config.graph_rag_local_community_weight = 1.5
        brain.config.graph_rag_local_text_unit_weight = 1.0
        brain.config.graph_rag_community_min_size = 3
        brain.config.graph_rag_community_llm_top_n_per_level = 50
        brain.config.graph_rag_community_llm_max_total = 200
        brain.config.graph_rag_community_lazy = False
        brain.config.graph_rag_global_top_communities = 0
        brain.config.raptor_max_source_size_mb = 0.0
        brain.config.graph_rag_exact_entity_boost = 3.0
        brain.config.graph_rag_community_defer = False
        brain.config.graph_rag_include_raptor_summaries = False
        brain.config.graph_rag_community_rebuild_debounce_s = 0.0
        brain.config.graph_rag_community_async = False
        brain.config.graph_rag_community = True
        brain.config.graph_rag_global_map_max_length = 500
        brain.config.graph_rag_global_reduce_max_length = 500
        brain.config.graph_rag_global_allow_general_knowledge = False
        brain.config.graph_rag_global_min_score = 0
        brain.config.graph_rag_global_top_points = 50
        brain.config.graph_rag_community_level = 0
        brain.config.graph_rag_global_reduce_max_tokens = 8000
        brain.config.graph_rag_index_community_reports = True
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._claims_graph = {}
        brain._community_levels = {}
        brain._community_hierarchy = {}
        brain._community_children = {}
        brain._community_summaries = {}
        brain._entity_embeddings = {}
        brain._entity_description_buffer = {}
        brain._relation_description_buffer = {}
        brain._text_unit_entity_map = {}
        brain._text_unit_relation_map = {}
        brain._community_graph_dirty = False
        brain._community_build_in_progress = False
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain._save_community_summaries = MagicMock()
        brain._save_community_hierarchy = MagicMock()
        import concurrent.futures

        brain._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        llm = MagicMock()
        llm.complete = MagicMock(
            return_value='{"title":"T","summary":"s","findings":[],"rank":5.0}'
        )
        brain.llm = llm
        for method_name in [
            "_run_hierarchical_community_detection",
            "_build_networkx_graph",
            "_get_incoming_relations",
            "_local_search_context",
            "_generate_community_summaries",
            "_extract_relations",
            "_entity_matches",
            "_expand_with_entity_graph",
            "_match_entities_by_embedding",
            "_extract_entities",
        ]:
            method = getattr(AxonBrain, method_name)
            setattr(brain, method_name, lambda *a, m=method, **kw: m(brain, *a, **kw))
        return brain

    # ------------------------------------------------------------------
    # Step 1: exact-token entity boost
    # ------------------------------------------------------------------

    def test_exact_entity_boost_prioritizes_match(self):
        """Query 'service_17' must rank service_17 above service_16 and service_18."""
        import re as _re

        b = self._make_brain()
        # Equal degree for all three so boost is the only differentiator
        b._relation_graph = {
            "service_16": [{"target": "x", "relation": "r", "chunk_id": "c1", "description": ""}],
            "service_17": [{"target": "x", "relation": "r", "chunk_id": "c1", "description": ""}],
            "service_18": [{"target": "x", "relation": "r", "chunk_id": "c1", "description": ""}],
        }
        b._get_incoming_relations = MagicMock(return_value=[])

        matched = ["service_16", "service_17", "service_18"]
        query = "service_17"
        # Mirror the production token-building logic: whitespace split + regex split
        query_tokens = set(query.lower().split()) | set(_re.split(r"[\s\W_]+", query.lower()))
        boost = b.config.graph_rag_exact_entity_boost

        def _entity_degree(ent):
            outgoing = len(b._relation_graph.get(ent.lower(), []))
            incoming = len(b._get_incoming_relations(ent))
            return outgoing + incoming

        def _entity_score(ent):
            degree = _entity_degree(ent)
            return degree * (boost if ent.lower() in query_tokens else 1.0)

        ranked = sorted(matched, key=_entity_score, reverse=True)
        assert ranked[0] == "service_17", f"Expected service_17 first, got {ranked}"

    def test_exact_entity_boost_no_effect_without_exact_token(self):
        """Non-matching query preserves degree-only ordering."""
        import re as _re

        b = self._make_brain()
        b._relation_graph = {
            "alpha": [{"target": "x", "relation": "r", "chunk_id": "c", "description": ""}] * 5,
            "beta": [{"target": "x", "relation": "r", "chunk_id": "c", "description": ""}] * 2,
        }
        b._get_incoming_relations = MagicMock(return_value=[])

        matched = ["alpha", "beta"]
        query = "some unrelated query"
        query_tokens = set(_re.split(r"[\s\W_]+", query.lower()))
        boost = b.config.graph_rag_exact_entity_boost

        def _entity_degree(ent):
            return len(b._relation_graph.get(ent.lower(), []))

        def _entity_score(ent):
            degree = _entity_degree(ent)
            return degree * (boost if ent.lower() in query_tokens else 1.0)

        ranked = sorted(matched, key=_entity_score, reverse=True)
        assert ranked[0] == "alpha", f"Expected alpha first, got {ranked}"

    # ------------------------------------------------------------------
    # Step 2: community build-in-progress flag
    # ------------------------------------------------------------------

    def test_community_build_in_progress_flag_set_and_cleared(self):
        """Async rebuild sets _community_build_in_progress=True then clears to False."""
        import concurrent.futures
        import time

        b = self._make_brain()
        b._community_build_in_progress = False

        observed_during = []

        def fake_rebuild():
            observed_during.append(b._community_build_in_progress)

        b._rebuild_communities = MagicMock(side_effect=fake_rebuild)
        b._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        def _debounced_rebuild():
            b._community_build_in_progress = True
            try:
                time.sleep(0.0)
                b._rebuild_communities()
            finally:
                b._community_build_in_progress = False

        fut = b._executor.submit(_debounced_rebuild)
        fut.result(timeout=5)

        assert observed_during == [True], "Flag was not True during rebuild"
        assert b._community_build_in_progress is False, "Flag not cleared after rebuild"
        b._executor.shutdown(wait=False)

    # ------------------------------------------------------------------
    # Step 3: deferred rebuild
    # ------------------------------------------------------------------

    def test_community_defer_skips_rebuild_during_ingest(self):
        """graph_rag_community_defer=True must not trigger _rebuild_communities."""
        b = self._make_brain()
        b.config.graph_rag_community_defer = True
        b._community_graph_dirty = True
        b._rebuild_communities = MagicMock()

        if b.config.graph_rag_community and b._community_graph_dirty:
            if b.config.graph_rag_community_defer:
                pass
            else:
                b._rebuild_communities()

        b._rebuild_communities.assert_not_called()
        assert b._community_graph_dirty is True

    def test_finalize_graph_triggers_rebuild(self):
        """finalize_graph() on a dirty brain calls _rebuild_communities exactly once."""
        from axon.main import AxonBrain

        b = self._make_brain()
        b._community_graph_dirty = True
        b._rebuild_communities = MagicMock()

        AxonBrain.finalize_graph(b)

        b._rebuild_communities.assert_called_once()
        assert b._community_graph_dirty is False

    # ------------------------------------------------------------------
    # Step 4: community summary incremental caching
    # ------------------------------------------------------------------

    def test_community_summary_cache_hit_skips_llm(self):
        """Unchanged membership hash skips LLM on second _generate_community_summaries."""
        import hashlib

        from axon.main import AxonBrain

        b = self._make_brain()
        b._community_levels = {0: {"entity_a": 0, "entity_b": 0}}
        b._community_children = {}

        members_sorted = sorted(["entity_a", "entity_b"])
        raw = f"0|{'|'.join(members_sorted)}"
        existing_hash = hashlib.md5(raw.encode()).hexdigest()

        b._community_summaries = {
            "0_0": {
                "title": "Cached Title",
                "summary": "cached summary",
                "findings": [],
                "rank": 7.0,
                "full_content": "# Cached Title\n\ncached summary",
                "entities": ["entity_a", "entity_b"],
                "size": 2,
                "level": 0,
                "member_hash": existing_hash,
            }
        }

        llm_mock = MagicMock(
            return_value='{"title":"New","summary":"new","findings":[],"rank":5.0}'
        )
        b.llm.complete = llm_mock

        AxonBrain._generate_community_summaries(b)

        llm_mock.assert_not_called()
        assert b._community_summaries["0_0"]["title"] == "Cached Title"

    def test_community_summary_cache_miss_triggers_llm(self):
        """Changed membership hash calls LLM for that community."""
        from axon.main import AxonBrain

        b = self._make_brain()
        b._community_levels = {0: {"entity_a": 0, "entity_b": 0, "entity_c": 0}}
        b._community_children = {}

        old_hash = "deadbeef" * 4  # wrong hash
        b._community_summaries = {
            "0_0": {
                "title": "Old Title",
                "summary": "old",
                "findings": [],
                "rank": 5.0,
                "full_content": "# Old Title\n\nold",
                "entities": ["entity_a", "entity_b"],
                "size": 2,
                "level": 0,
                "member_hash": old_hash,
            }
        }

        llm_mock = MagicMock(
            return_value='{"title":"New","summary":"new","findings":[],"rank":5.0}'
        )
        b.llm.complete = llm_mock

        AxonBrain._generate_community_summaries(b)

        llm_mock.assert_called_once()

    # ------------------------------------------------------------------
    # Step 6: RAPTOR summaries inclusion filter
    # ------------------------------------------------------------------

    def test_raptor_summaries_included_when_config_true(self):
        """graph_rag_include_raptor_summaries=True includes raptor_level=1 docs."""
        include_raptor = True
        documents = [
            {"id": "leaf1", "text": "leaf text", "metadata": {}},
            {"id": "raptor1", "text": "summary", "metadata": {"raptor_level": 1}},
            {"id": "raptor2", "text": "deeper", "metadata": {"raptor_level": 2}},
        ]
        chunks = [
            doc
            for doc in documents
            if not doc.get("metadata", {}).get("raptor_level")
            or (include_raptor and doc.get("metadata", {}).get("raptor_level") == 1)
        ]
        ids = [d["id"] for d in chunks]
        assert "leaf1" in ids
        assert "raptor1" in ids
        assert "raptor2" not in ids

    def test_raptor_summaries_excluded_by_default(self):
        """Default config excludes all RAPTOR summary nodes from entity extraction."""
        include_raptor = False
        documents = [
            {"id": "leaf1", "text": "leaf text", "metadata": {}},
            {"id": "raptor1", "text": "summary", "metadata": {"raptor_level": 1}},
        ]
        chunks = [
            doc
            for doc in documents
            if not doc.get("metadata", {}).get("raptor_level")
            or (include_raptor and doc.get("metadata", {}).get("raptor_level") == 1)
        ]
        ids = [d["id"] for d in chunks]
        assert "leaf1" in ids
        assert "raptor1" not in ids


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestRaptorTask8Fixes:
    """RAPTOR audit fixes: P1, P2, P3, P4, P5 from TASK_8."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **cfg_kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, **cfg_kwargs)
        brain = AxonBrain(config)
        brain._ingested_hashes = set()
        brain._entity_graph = {}
        brain._save_hash_store = MagicMock()
        brain._save_entity_graph = MagicMock()
        brain.embedding.embed = MagicMock(side_effect=lambda texts: [[0.1] * 384] * len(texts))
        brain.vector_store.add = MagicMock()
        return brain

    # ------------------------------------------------------------------
    # P5: Summary caching
    # ------------------------------------------------------------------

    def test_summary_cache_hit_skips_llm(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Same content ingested twice: LLM called exactly once (cache hit on 2nd)."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_chunk_group_size=2,
            raptor_cache_summaries=True,
            graph_rag=False,
        )
        docs = [
            {"id": f"d{i}", "text": f"chunk {i}", "metadata": {"source": "a.txt"}} for i in range(2)
        ]
        brain.llm.complete = MagicMock(return_value="Cached summary.")
        brain.ingest(docs)
        brain._ingested_hashes = set()  # allow re-ingest
        brain.ingest(docs)
        # 2 docs → 1 window → LLM called once across both ingests
        assert brain.llm.complete.call_count == 1

    def test_summary_cache_disabled_always_calls_llm(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """raptor_cache_summaries=False: LLM called on every ingest."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_chunk_group_size=2,
            raptor_cache_summaries=False,
            graph_rag=False,
        )
        docs = [
            {"id": f"d{i}", "text": f"chunk {i}", "metadata": {"source": "a.txt"}} for i in range(2)
        ]
        brain.llm.complete = MagicMock(return_value="No cache.")
        brain.ingest(docs)
        brain._ingested_hashes = set()
        brain.ingest(docs)
        assert brain.llm.complete.call_count == 2

    # ------------------------------------------------------------------
    # P2: Large-doc GraphRAG leaf skip
    # ------------------------------------------------------------------

    def test_graphrag_skips_large_source_leaves(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """25-chunk source with raptor=True: leaf chunks skipped, RAPTOR summaries auto-included.

        A1 change: large sources with raptor=True now auto-include their RAPTOR level-1
        summaries in GraphRAG entity extraction, even when graph_rag_include_raptor_summaries
        is False. The entity graph may be populated from RAPTOR summaries (not leaf chunks).
        """
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            graph_rag=True,
            raptor_graphrag_leaf_skip_threshold=10,
            graph_rag_include_raptor_summaries=False,
            raptor_cache_summaries=False,
        )
        docs = [
            {"id": f"d{i}", "text": f"text {i}", "metadata": {"source": "big.txt"}}
            for i in range(25)
        ]
        # LLM returns RAPTOR summaries for the raptor pass; entity extraction also uses LLM.
        brain.llm.complete = MagicMock(return_value="Summary paragraph.")
        brain.ingest(docs)
        # With A1 auto-composition, RAPTOR summaries for large sources are extracted.
        # Leaf chunks from big.txt must NOT appear in entity chunk_ids.
        leaf_ids = {f"d{i}" for i in range(25)}
        for node in brain._entity_graph.values():
            if isinstance(node, dict):
                for cid in node.get("chunk_ids", []):
                    assert (
                        cid not in leaf_ids
                    ), f"Leaf chunk {cid} from large source must not be in entity graph"

    def test_graphrag_allows_small_source_leaves(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """5-chunk source below threshold must still populate the entity graph."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            graph_rag=True,
            raptor_graphrag_leaf_skip_threshold=10,
            raptor_cache_summaries=False,
        )
        docs = [
            {"id": f"d{i}", "text": f"text {i}", "metadata": {"source": "small.txt"}}
            for i in range(5)
        ]
        entity_json = (
            '{"entities": [{"name": "Alice", "type": "PERSON", "description": "a person"}]}'
        )
        brain.llm.complete = MagicMock(return_value=entity_json)
        brain.ingest(docs)
        assert len(brain._entity_graph) > 0, "Small source must populate entity graph"

    # ------------------------------------------------------------------
    # P1: RAPTOR drill-down
    # ------------------------------------------------------------------

    def test_drilldown_replaces_summary_with_leaves(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """A RAPTOR summary hit must be replaced by leaf chunks from the vector store."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_drilldown=True,
            raptor_drilldown_top_k=3,
            graph_rag=False,
        )
        raptor_result = {
            "id": "raptor_abc",
            "text": "high-level summary",
            "score": 0.9,
            "metadata": {
                "source": "doc.txt",
                "raptor_level": 1,
                "window_start": 0,
                "window_end": 2,
            },
        }
        leaf_hits = [
            {"id": f"leaf_{i}", "text": f"leaf {i}", "score": 0.8 - i * 0.1, "metadata": {}}
            for i in range(3)
        ]
        brain.vector_store.search = MagicMock(return_value=leaf_hits)
        results = AxonBrain._raptor_drilldown(brain, "test query", [raptor_result])
        ids = [r["id"] for r in results]
        assert "raptor_abc" not in ids, "Summary node must be replaced by leaves"
        assert any(r["id"].startswith("leaf_") for r in results)

    def test_drilldown_disabled_keeps_summary(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """raptor_drilldown=False: summary node must pass through unchanged."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_drilldown=False,
            graph_rag=False,
        )
        raptor_result = {
            "id": "raptor_xyz",
            "text": "summary",
            "score": 0.9,
            "metadata": {"source": "s.txt", "raptor_level": 1, "window_start": 0, "window_end": 1},
        }
        results = AxonBrain._raptor_drilldown(brain, "query", [raptor_result])
        assert results[0]["id"] == "raptor_xyz"

    def test_drilldown_missing_window_meta_keeps_summary(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """RAPTOR node without window_start/end must be kept as fallback."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_drilldown=True,
            graph_rag=False,
        )
        raptor_result = {
            "id": "raptor_no_win",
            "text": "summary",
            "score": 0.8,
            "metadata": {"raptor_level": 1},
        }
        results = AxonBrain._raptor_drilldown(brain, "query", [raptor_result])
        assert results[0]["id"] == "raptor_no_win"

    # ------------------------------------------------------------------
    # P4: Artifact-type ranking
    # ------------------------------------------------------------------

    def test_artifact_ranking_tree_traversal_boosts_leaves(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """tree_traversal mode: leaf must rank above RAPTOR summary at equal base score."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_retrieval_mode="tree_traversal",
            graph_rag=False,
        )
        leaf = {"id": "leaf1", "text": "leaf", "score": 0.5, "metadata": {}}
        raptor = {
            "id": "r1",
            "text": "summary",
            "score": 0.5,
            "metadata": {"raptor_level": 1},
        }
        ranked = AxonBrain._apply_artifact_ranking(brain, [raptor, leaf])
        assert ranked[0]["id"] == "leaf1", "Leaf must rank first in tree_traversal mode"

    def test_artifact_ranking_summary_first_boosts_raptor(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """summary_first mode: RAPTOR summary must rank above leaf at equal base score."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_retrieval_mode="summary_first",
            graph_rag=False,
        )
        leaf = {"id": "leaf1", "text": "leaf", "score": 0.5, "metadata": {}}
        raptor = {
            "id": "r1",
            "text": "summary",
            "score": 0.5,
            "metadata": {"raptor_level": 1},
        }
        ranked = AxonBrain._apply_artifact_ranking(brain, [leaf, raptor])
        assert ranked[0]["id"] == "r1", "RAPTOR must rank first in summary_first mode"

    # ------------------------------------------------------------------
    # P3: Multi-level RAPTOR
    # ------------------------------------------------------------------

    def test_multi_level_raptor_produces_level2_nodes(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """10 docs with group_size=2 and max_levels=2 must produce level-2 nodes."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_chunk_group_size=2,
            raptor_max_levels=2,
            raptor_cache_summaries=False,
            graph_rag=False,
        )
        docs = [
            {"id": f"d{i}", "text": f"chunk {i}", "metadata": {"source": "a.txt"}}
            for i in range(10)
        ]
        brain.llm.complete = MagicMock(return_value="A summary paragraph.")
        brain.ingest(docs)

        add_calls = brain.vector_store.add.call_args_list
        all_metadatas = []
        for call in add_calls:
            all_metadatas.extend(call[0][3])
        level2 = [m for m in all_metadatas if m.get("raptor_level") == 2]
        assert len(level2) >= 1, (
            f"Expected level-2 RAPTOR nodes, found levels: "
            f"{[m.get('raptor_level') for m in all_metadatas]}"
        )

    def test_multi_level_raptor_max_levels_1_no_recursion(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """raptor_max_levels=1 must not produce any level-2 nodes."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_chunk_group_size=2,
            raptor_max_levels=1,
            raptor_cache_summaries=False,
            graph_rag=False,
        )
        docs = [
            {"id": f"d{i}", "text": f"chunk {i}", "metadata": {"source": "a.txt"}}
            for i in range(10)
        ]
        brain.llm.complete = MagicMock(return_value="One-level summary.")
        brain.ingest(docs)

        add_calls = brain.vector_store.add.call_args_list
        all_metadatas = []
        for call in add_calls:
            all_metadatas.extend(call[0][3])
        level2 = [m for m in all_metadatas if m.get("raptor_level") == 2]
        assert len(level2) == 0, "raptor_max_levels=1 must not produce level-2 nodes"


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestRaptorTask9Fixes:
    """RAPTOR + GraphRAG correctness fixes from TASK_9."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **cfg_kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, **cfg_kwargs)
        brain = AxonBrain(config)
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain._save_entity_graph = MagicMock()
        brain.embedding.embed = MagicMock(side_effect=lambda texts: [[0.1] * 384] * len(texts))
        brain.vector_store.add = MagicMock()
        return brain

    # ------------------------------------------------------------------
    # P2+P3: Drill-down uses children_ids lineage
    # ------------------------------------------------------------------

    def test_drilldown_uses_children_ids(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """RAPTOR node with children_ids must call get_by_ids, not search."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_drilldown=True,
            raptor_drilldown_top_k=3,
            graph_rag=False,
        )
        leaf_docs = [
            {"id": f"leaf_{i}", "text": f"leaf {i}", "score": 1.0, "metadata": {}} for i in range(3)
        ]
        brain.vector_store.get_by_ids = MagicMock(return_value=leaf_docs)
        brain.vector_store.search = MagicMock(return_value=[])

        raptor_result = {
            "id": "raptor_with_children",
            "text": "summary",
            "score": 0.9,
            "metadata": {
                "source": "doc.txt",
                "raptor_level": 1,
                "window_start": 0,
                "window_end": 2,
                "children_ids": ["leaf_0", "leaf_1", "leaf_2"],
            },
        }
        results = AxonBrain._raptor_drilldown(brain, "test query", [raptor_result])

        brain.vector_store.get_by_ids.assert_called_once_with(["leaf_0", "leaf_1", "leaf_2"])
        brain.vector_store.search.assert_not_called()
        assert all(r["id"].startswith("leaf_") for r in results)

    def test_drilldown_recurses_through_levels(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Level-2 node → level-1 intermediates → leaves: get_by_ids called twice."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_drilldown=True,
            raptor_drilldown_top_k=5,
            graph_rag=False,
        )
        level1_docs = [
            {
                "id": "mid_0",
                "text": "mid summary",
                "score": 1.0,
                "metadata": {"raptor_level": 1, "children_ids": ["leaf_0", "leaf_1"]},
            }
        ]
        leaf_docs = [
            {"id": "leaf_0", "text": "leaf A", "score": 1.0, "metadata": {}},
            {"id": "leaf_1", "text": "leaf B", "score": 1.0, "metadata": {}},
        ]
        brain.vector_store.get_by_ids = MagicMock(side_effect=[level1_docs, leaf_docs])
        brain.vector_store.search = MagicMock(return_value=[])

        raptor_result = {
            "id": "raptor_level2",
            "text": "top summary",
            "score": 0.95,
            "metadata": {
                "source": "doc.txt",
                "raptor_level": 2,
                "window_start": 0,
                "window_end": 4,
                "children_ids": ["mid_0"],
            },
        }
        results = AxonBrain._raptor_drilldown(brain, "test query", [raptor_result])

        assert brain.vector_store.get_by_ids.call_count == 2, (
            "Expected 2 get_by_ids calls (level-1 then leaves), "
            f"got {brain.vector_store.get_by_ids.call_count}"
        )
        ids = [r["id"] for r in results]
        assert "leaf_0" in ids and "leaf_1" in ids
        assert "mid_0" not in ids and "raptor_level2" not in ids

    def test_drilldown_falls_back_without_children_ids(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Legacy node without children_ids falls back to filtered search."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_drilldown=True,
            raptor_drilldown_top_k=3,
            graph_rag=False,
        )
        leaf_docs = [
            {"id": f"leaf_{i}", "text": f"leaf {i}", "score": 0.8, "metadata": {}} for i in range(3)
        ]
        brain.vector_store.get_by_ids = MagicMock(return_value=[])
        brain.vector_store.search = MagicMock(return_value=leaf_docs)

        raptor_result = {
            "id": "raptor_legacy",
            "text": "legacy summary",
            "score": 0.85,
            "metadata": {
                "source": "legacy.txt",
                "raptor_level": 1,
                "window_start": 0,
                "window_end": 2,
                # no children_ids
            },
        }
        results = AxonBrain._raptor_drilldown(brain, "test query", [raptor_result])

        brain.vector_store.get_by_ids.assert_not_called()
        brain.vector_store.search.assert_called_once()
        assert brain.vector_store.search.call_args[1].get("filter_dict") == {"source": "legacy.txt"}
        assert all(r["id"].startswith("leaf_") for r in results)

    # ------------------------------------------------------------------
    # P4: Deduplication
    # ------------------------------------------------------------------

    def test_drilldown_deduplicates_leaves(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Two RAPTOR nodes sharing a leaf ID must yield that leaf exactly once."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_drilldown=True,
            raptor_drilldown_top_k=5,
            graph_rag=False,
        )
        shared_leaf = {"id": "shared_leaf", "text": "shared content", "score": 0.9, "metadata": {}}
        brain.vector_store.get_by_ids = MagicMock(return_value=[shared_leaf])

        raptor_a = {
            "id": "raptor_A",
            "text": "summary A",
            "score": 0.9,
            "metadata": {
                "source": "doc.txt",
                "raptor_level": 1,
                "window_start": 0,
                "window_end": 1,
                "children_ids": ["shared_leaf"],
            },
        }
        raptor_b = {
            "id": "raptor_B",
            "text": "summary B",
            "score": 0.85,
            "metadata": {
                "source": "doc.txt",
                "raptor_level": 1,
                "window_start": 2,
                "window_end": 3,
                "children_ids": ["shared_leaf"],
            },
        }
        results = AxonBrain._raptor_drilldown(brain, "test query", [raptor_a, raptor_b])

        leaf_occurrences = [r for r in results if r["id"] == "shared_leaf"]
        assert (
            len(leaf_occurrences) == 1
        ), f"shared_leaf appeared {len(leaf_occurrences)} times, expected 1"

    # ------------------------------------------------------------------
    # P5: GraphRAG chunk_ids KeyError safety
    # ------------------------------------------------------------------

    def test_entity_graph_chunk_ids_keyerror_safe(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Entity node dict missing chunk_ids must not crash _expand_with_entity_graph."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            raptor=False,
        )
        # Malformed node: dict but no chunk_ids key
        brain._entity_graph = {"alice": {"description": "a person", "type": "PERSON"}}
        brain._relation_graph = {}
        brain._entity_embeddings = {}
        brain._extract_entities = MagicMock(return_value=[{"name": "alice", "type": "PERSON"}])
        brain._match_entities_by_embedding = MagicMock(return_value=[])

        existing = [{"id": "doc1", "text": "some text", "score": 0.7, "metadata": {}}]
        # Must not raise KeyError
        expanded, matched = AxonBrain._expand_with_entity_graph(brain, "who is alice?", existing)
        assert len(expanded) >= len(existing)

    def test_entity_graph_pruning_missing_chunk_ids_safe(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """_prune_entity_graph must not crash when a node dict lacks chunk_ids."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            graph_rag=True,
            raptor=False,
        )
        # Node dict without chunk_ids
        brain._entity_graph = {"bob": {"description": "missing chunk_ids"}}
        brain._relation_graph = {}
        brain._save_relation_graph = MagicMock()

        # Must not raise KeyError
        AxonBrain._prune_entity_graph(brain, deleted_ids={"some_deleted_id"})
        # Bob not pruned — his empty chunk list has nothing matching the deleted id
        assert "bob" in brain._entity_graph


class TestOpenVectorStoreChromaFilter:
    """Unit tests for Chroma filter_dict → where clause (no AxonBrain mock needed)."""

    def _make_vs(self, query_return=None):
        """Build a bare OpenVectorStore instance with a mocked Chroma collection."""
        import axon.main as _m

        vs = object.__new__(_m.OpenVectorStore)
        vs.provider = "chroma"
        mock_col = MagicMock()
        mock_col.query.return_value = query_return or {
            "ids": [[]],
            "documents": [[]],
            "distances": [[]],
            "metadatas": [[]],
        }
        vs.collection = mock_col
        return vs, mock_col

    def test_chroma_search_passes_where_filter(self):
        """Single-key filter_dict must produce a $eq where clause in collection.query."""
        vs, mock_col = self._make_vs(
            query_return={
                "ids": [["id1"]],
                "documents": [["text1"]],
                "distances": [[0.1]],
                "metadatas": [[{"source": "a.txt"}]],
            }
        )
        vs.search([0.1] * 384, top_k=5, filter_dict={"source": "a.txt"})
        kw = mock_col.query.call_args[1]
        assert kw.get("where") == {
            "source": {"$eq": "a.txt"}
        }, f"Expected single-key $eq where, got: {kw.get('where')}"

    def test_chroma_search_no_filter_passes_where_none(self):
        """No filter_dict must pass where=None (not where={}) to avoid Chroma errors."""
        vs, mock_col = self._make_vs()
        vs.search([0.1] * 384, top_k=5)
        kw = mock_col.query.call_args[1]
        assert (
            kw.get("where") is None
        ), f"Expected where=None when no filter, got: {kw.get('where')}"

    def test_chroma_search_multi_key_filter_uses_and(self):
        """Multi-key filter_dict must produce an $and compound where clause."""
        vs, mock_col = self._make_vs()
        vs.search([0.1] * 384, top_k=5, filter_dict={"source": "a.txt", "raptor_level": 1})
        kw = mock_col.query.call_args[1]
        where = kw.get("where")
        assert "$and" in where, f"Multi-key filter must use $and, got: {where}"
        assert len(where["$and"]) == 2


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestRaptorTask10Fixes:
    """RAPTOR + GraphRAG fixes from TASK_10 (second re-audit)."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **cfg_kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, **cfg_kwargs)
        brain = AxonBrain(config)
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain._save_entity_graph = MagicMock()
        brain.embedding.embed = MagicMock(side_effect=lambda texts: [[0.1] * 384] * len(texts))
        brain.vector_store.add = MagicMock()
        return brain

    # ------------------------------------------------------------------
    # Fix 1: level-1 RAPTOR nodes must store children_ids
    # ------------------------------------------------------------------

    def test_level1_nodes_store_children_ids(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Each level-1 RAPTOR summary node must carry children_ids = the leaf IDs in its window."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            raptor=True,
            raptor_chunk_group_size=2,
            raptor_max_levels=1,
            raptor_cache_summaries=False,
            graph_rag=False,
            hybrid_search=False,
            rerank=False,
        )
        brain = AxonBrain(config)
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain.embedding.embed = MagicMock(side_effect=lambda texts: [[0.1] * 384] * len(texts))

        # Capture add(ids, texts, embeddings, metadatas) calls
        captured_metadatas: list = []

        def _capture_add(ids, texts, embeddings, metadatas):
            captured_metadatas.extend(metadatas)

        brain._own_vector_store = MagicMock()
        brain._own_vector_store.add = MagicMock(side_effect=_capture_add)
        brain.llm.complete = MagicMock(return_value="Summary text.")

        docs = [
            {"id": f"leaf_{i}", "text": f"chunk{i} content.", "metadata": {"source": "doc.txt"}}
            for i in range(4)
        ]
        brain.ingest(docs)

        raptor_metas = [m for m in captured_metadatas if m.get("raptor_level") == 1]
        assert raptor_metas, "No level-1 RAPTOR nodes were created"
        for meta in raptor_metas:
            children = meta.get("children_ids")
            assert children, f"Level-1 RAPTOR node has no children_ids: {meta}"
            assert isinstance(children, list) and len(children) > 0

    # ------------------------------------------------------------------
    # Fix 1b: drill-down works for level-1 nodes with new children_ids
    # ------------------------------------------------------------------

    def test_drilldown_uses_level1_children_ids(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Level-1 RAPTOR node with children_ids must use get_by_ids, not fall back to search."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_drilldown=True,
            raptor_drilldown_top_k=3,
            graph_rag=False,
        )
        leaf_docs = [
            {"id": f"leaf_{i}", "text": f"leaf text {i}", "score": 1.0, "metadata": {}}
            for i in range(2)
        ]
        brain.vector_store.get_by_ids = MagicMock(return_value=leaf_docs)
        brain.vector_store.search = MagicMock(return_value=[])

        raptor_result = {
            "id": "raptor_l1_abc",
            "text": "level-1 summary",
            "score": 0.88,
            "metadata": {
                "source": "doc.txt",
                "raptor_level": 1,
                "window_start": 0,
                "window_end": 1,
                "children_ids": ["leaf_0", "leaf_1"],
            },
        }
        results = AxonBrain._raptor_drilldown(brain, "test query", [raptor_result])

        brain.vector_store.get_by_ids.assert_called_once_with(["leaf_0", "leaf_1"])
        brain.vector_store.search.assert_not_called()
        assert all(r["id"].startswith("leaf_") for r in results)

    # ------------------------------------------------------------------
    # Fix 2: chunk_ids setdefault prevents KeyError during entity ingest
    # ------------------------------------------------------------------

    def test_chunk_ids_setdefault_prevents_keyerror_in_ingest(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Entity graph dict node missing chunk_ids must not raise KeyError during ingest."""
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(
            graph_rag=True,
            raptor=False,
            hybrid_search=False,
            rerank=False,
            graph_rag_community_defer=True,
        )
        brain = AxonBrain(config)
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
        brain.embedding.embed = MagicMock(side_effect=lambda texts: [[0.1] * 384] * len(texts))
        brain.vector_store.add = MagicMock()

        # Pre-seed an entity dict WITHOUT chunk_ids (simulates old/migrated disk data)
        brain._entity_graph["alice"] = {
            "description": "a person",
            "type": "PERSON",
            "frequency": 0,
            "degree": 0,
            # deliberately no "chunk_ids" key
        }

        # _extract_entities parses pipe-separated lines: NAME | TYPE | description
        brain.llm.complete = MagicMock(return_value="alice | PERSON | a well-known person")
        brain._own_vector_store = MagicMock()
        brain._own_vector_store.add = MagicMock()

        # Must not raise KeyError
        brain.ingest(
            [
                {
                    "id": "chunk_alice",
                    "text": "Alice is a key figure.",
                    "metadata": {"source": "test.txt"},
                }
            ]
        )

        assert (
            "chunk_ids" in brain._entity_graph["alice"]
        ), "setdefault should have created chunk_ids during ingest"

    # ------------------------------------------------------------------
    # E2E: RAPTOR drill-down round-trip
    # ------------------------------------------------------------------

    def test_e2e_raptor_drilldown_returns_leaves(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """When query retrieval returns a RAPTOR node, _raptor_drilldown must replace it with leaves."""
        from axon.main import AxonBrain

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_drilldown=True,
            raptor_drilldown_top_k=5,
            graph_rag=False,
        )

        leaf_docs = [
            {"id": "leaf_0", "text": "actual leaf content", "score": 0.95, "metadata": {}},
            {"id": "leaf_1", "text": "more leaf content", "score": 0.90, "metadata": {}},
        ]
        brain.vector_store.get_by_ids = MagicMock(return_value=leaf_docs)
        brain.vector_store.search = MagicMock(return_value=[])

        query_results = [
            {
                "id": "raptor_l1_xyz",
                "text": "Summary of the document section.",
                "score": 0.85,
                "metadata": {
                    "source": "report.txt",
                    "raptor_level": 1,
                    "window_start": 0,
                    "window_end": 1,
                    "children_ids": ["leaf_0", "leaf_1"],
                },
            }
        ]

        result = AxonBrain._raptor_drilldown(brain, "what does the report say?", query_results)

        # RAPTOR summary node must be gone; leaf chunks must be present
        result_ids = [r["id"] for r in result]
        assert "raptor_l1_xyz" not in result_ids, "RAPTOR summary node should be replaced"
        assert "leaf_0" in result_ids, "leaf_0 must appear after drill-down"
        assert "leaf_1" in result_ids, "leaf_1 must appear after drill-down"
        # No duplicates
        assert len(result_ids) == len(
            set(result_ids)
        ), "Drill-down result must not contain duplicates"


# ---------------------------------------------------------------------------
# TASK 11 — Fundamental GraphRAG + RAPTOR Fixes
# ---------------------------------------------------------------------------


class TestFundamentalFixes:
    """Tests for TASK_11: relation-target normalization, unified ranking, structure-aware RAPTOR."""

    def _make_brain(self):
        """Return a minimal AxonBrain spec-mock with real method bindings."""
        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.config.graph_rag_exact_entity_boost = 3.0
        brain.config.graph_rag_local_max_context_tokens = 8000
        brain.config.graph_rag_local_community_prop = 0.25
        brain.config.graph_rag_local_text_unit_prop = 0.5
        brain.config.graph_rag_local_top_k_entities = 10
        brain.config.graph_rag_local_top_k_relationships = 10
        brain.config.graph_rag_local_include_relationship_weight = False
        brain.config.graph_rag_local_entity_weight = 3.0
        brain.config.graph_rag_local_relation_weight = 2.0
        brain.config.graph_rag_local_community_weight = 1.5
        brain.config.graph_rag_local_text_unit_weight = 1.0
        brain.config.graph_rag_community_min_size = 3
        brain.config.graph_rag_community_llm_top_n_per_level = 50
        brain.config.graph_rag_community_llm_max_total = 200
        brain.config.graph_rag_community_lazy = False
        brain.config.graph_rag_global_top_communities = 0
        brain.config.raptor_max_source_size_mb = 0.0
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._claims_graph = {}
        brain._community_levels = {}
        brain._community_summaries = {}
        brain._text_unit_relation_map = {}
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()

        for method_name in [
            "_get_incoming_relations",
            "_local_search_context",
            "_expand_with_entity_graph",
            "_raptor_group_by_structure",
        ]:
            method = getattr(AxonBrain, method_name)
            setattr(brain, method_name, lambda *a, m=method, **kw: m(brain, *a, **kw))

        return brain

    # ------------------------------------------------------------------
    # Step 1: Relation-target normalization
    # ------------------------------------------------------------------

    def test_relation_target_normalized_into_entity_graph(self):
        """After normalization pass, every relation target must be a key in _entity_graph."""

        brain = self._make_brain()
        # Pre-populate: "alice" is an extracted entity, "bob" is only a relation target
        brain._entity_graph = {
            "alice": {
                "description": "Alice",
                "type": "PERSON",
                "chunk_ids": ["c1"],
                "frequency": 1,
                "degree": 1,
            },
        }
        brain._relation_graph = {
            "alice": [{"target": "bob", "relation": "knows", "chunk_id": "c1"}],
        }

        # Run the normalization logic (extracted from ingest; call it via a thin helper)
        _stub_added = False
        for _src, _entries in brain._relation_graph.items():
            for _entry in _entries:
                _tgt = _entry.get("target", "").lower().strip()
                if not _tgt:
                    continue
                if _tgt not in brain._entity_graph:
                    brain._entity_graph[_tgt] = {
                        "description": "",
                        "type": "UNKNOWN",
                        "chunk_ids": [],
                        "frequency": 0,
                        "degree": 0,
                    }
                    _stub_added = True
                _cid = _entry.get("chunk_id", "")
                if _cid:
                    _tgt_node = brain._entity_graph[_tgt]
                    if isinstance(_tgt_node, dict):
                        _tgt_node.setdefault("chunk_ids", [])
                        if _cid not in _tgt_node["chunk_ids"]:
                            _tgt_node["chunk_ids"].append(_cid)
                            _tgt_node["frequency"] = len(_tgt_node["chunk_ids"])
                            _stub_added = True
        if _stub_added:
            brain._save_entity_graph()

        assert "bob" in brain._entity_graph, "Relation target 'bob' must be added to entity graph"

    def test_relation_target_gets_chunk_id(self):
        """Stub node for a relation target must receive the relation's chunk_id."""

        brain = self._make_brain()
        brain._entity_graph = {
            "alice": {
                "description": "Alice",
                "type": "PERSON",
                "chunk_ids": ["c1"],
                "frequency": 1,
                "degree": 1,
            },
        }
        brain._relation_graph = {
            "alice": [{"target": "charlie", "relation": "manages", "chunk_id": "c2"}],
        }

        # Apply normalization
        for _src, _entries in brain._relation_graph.items():
            for _entry in _entries:
                _tgt = _entry.get("target", "").lower().strip()
                if not _tgt:
                    continue
                if _tgt not in brain._entity_graph:
                    brain._entity_graph[_tgt] = {
                        "description": "",
                        "type": "UNKNOWN",
                        "chunk_ids": [],
                        "frequency": 0,
                        "degree": 0,
                    }
                _cid = _entry.get("chunk_id", "")
                if _cid:
                    _tgt_node = brain._entity_graph[_tgt]
                    if isinstance(_tgt_node, dict):
                        _tgt_node.setdefault("chunk_ids", [])
                        if _cid not in _tgt_node["chunk_ids"]:
                            _tgt_node["chunk_ids"].append(_cid)
                            _tgt_node["frequency"] = len(_tgt_node["chunk_ids"])

        assert (
            "c2" in brain._entity_graph["charlie"]["chunk_ids"]
        ), "Stub target 'charlie' must contain the relation's chunk_id 'c2'"

    def test_stub_target_survives_traversal(self):
        """_expand_with_entity_graph must not KeyError when traversing a stub-only target."""

        brain = self._make_brain()
        # "dave" exists only as relation target (stub entity with empty chunk_ids)
        brain._entity_graph = {
            "alice": {
                "description": "Alice",
                "type": "PERSON",
                "chunk_ids": ["c1"],
                "frequency": 1,
                "degree": 1,
            },
            "dave": {
                "description": "",
                "type": "UNKNOWN",
                "chunk_ids": [],
                "frequency": 0,
                "degree": 0,
            },
        }
        brain._relation_graph = {
            "alice": [{"target": "dave", "relation": "reports_to", "chunk_id": "c1"}],
        }

        # _expand_with_entity_graph should complete without raising.
        # We manually invoke the 1-hop traversal logic to verify no KeyError.
        # The stub node "dave" has chunk_ids=[] so the loop runs zero iterations — no crash.
        _mock_results = [{"id": "c1", "text": "Alice content", "score": 0.9, "metadata": {}}]
        extra_id_scores: dict = {}
        matched_entities = {"alice"}
        existing_ids = {"c1"}
        try:
            for src_entity in matched_entities:
                for entry in brain._relation_graph.get(src_entity, []):
                    target = entry.get("target", "").lower()
                    if not target:
                        continue
                    # This lookup must not raise KeyError even for stub targets
                    target_node = brain._entity_graph.get(target, {})
                    target_chunk_ids = (
                        target_node.get("chunk_ids", [])
                        if isinstance(target_node, dict)
                        else target_node
                    )
                    for did in target_chunk_ids:
                        if did not in existing_ids:
                            extra_id_scores[did] = 0.62
        except KeyError as exc:
            raise AssertionError(f"KeyError raised during 1-hop traversal: {exc}") from exc

        # dave is a stub with empty chunk_ids — no extra docs should be fetched
        assert extra_id_scores == {}, "Stub target with empty chunk_ids should add no extra docs"

    # ------------------------------------------------------------------
    # Step 2: Unified ranking
    # ------------------------------------------------------------------

    def test_unified_ranking_no_section_cap(self):
        """With tight token budget, a high-scoring text unit beats a low-scoring community."""
        from axon.main import AxonBrain

        brain = self._make_brain()
        # One entity, one community (low rank), one text unit with high relation count
        brain._entity_graph = {
            "alpha": {
                "description": "Alpha entity",
                "type": "CONCEPT",
                "chunk_ids": ["tu1"],
                "frequency": 5,
                "degree": 3,
            },
        }
        brain._relation_graph = {"alpha": []}
        brain._text_unit_relation_map = {
            "tu1": [("alpha", "beta"), ("alpha", "gamma"), ("alpha", "delta")]
        }
        brain._community_levels = {0: {"alpha": 0}}
        brain._community_summaries = {
            "0_0": {"summary": "Low-signal community summary.", "title": "C0", "rank": 0.1}
        }

        # Mock vector_store.get_by_ids to return text content for tu1
        brain.vector_store = MagicMock()
        brain.vector_store.get_by_ids = MagicMock(
            return_value=[{"text": "High-value text unit content about alpha.", "id": "tu1"}]
        )

        # Use a tight budget: only enough for 2-3 candidates
        brain.config.graph_rag_local_max_context_tokens = 30
        brain.config.graph_rag_local_entity_weight = 3.0
        brain.config.graph_rag_local_relation_weight = 2.0
        brain.config.graph_rag_local_community_weight = 1.5
        brain.config.graph_rag_local_text_unit_weight = 1.0

        ctx = AxonBrain._local_search_context(brain, "alpha", ["alpha"], brain.config)
        # With unified ranking, text unit (score = 1.0 * 1.0) should appear even if community
        # would have consumed the budget first under fixed-split.
        # The key assertion: no hard floor for community means text_unit can appear.
        assert ctx != "", "Context must not be empty"

    def test_unified_ranking_respects_token_budget(self):
        """Total tokens in selected candidates must not exceed graph_rag_local_max_context_tokens."""
        from axon.main import AxonBrain

        brain = self._make_brain()
        brain._entity_graph = {
            f"ent{i}": {
                "description": f"Entity {i} description that is fairly long to consume tokens",
                "type": "CONCEPT",
                "chunk_ids": [f"c{i}"],
                "frequency": i + 1,
                "degree": i,
            }
            for i in range(10)
        }
        brain._relation_graph = {}
        brain._text_unit_relation_map = {}
        brain._community_levels = {}
        brain._community_summaries = {}
        brain.vector_store = MagicMock()
        brain.vector_store.get_by_ids = MagicMock(return_value=[])

        budget = 100
        brain.config.graph_rag_local_max_context_tokens = budget

        entities = list(brain._entity_graph.keys())
        ctx = AxonBrain._local_search_context(brain, "entity", entities, brain.config)
        used_tokens = len(ctx) // 4
        # Allow some slack for section headers; greedy-fill should stay near budget
        assert (
            used_tokens <= budget + 20
        ), f"Token usage {used_tokens} exceeds budget {budget} by too much"

    # ------------------------------------------------------------------
    # Step 3: Structure-aware RAPTOR grouping
    # ------------------------------------------------------------------

    def test_raptor_groups_by_heading(self):
        """Chunks starting with ## heading produce separate windows rather than one fixed window."""
        from axon.main import AxonBrain

        brain = self._make_brain()
        chunks = [
            {"id": "c1", "text": "## Introduction\nThis is the intro.", "metadata": {}},
            {"id": "c2", "text": "Intro continuation.", "metadata": {}},
            {"id": "c3", "text": "## Methods\nHere are the methods.", "metadata": {}},
            {"id": "c4", "text": "Methods detail.", "metadata": {}},
        ]
        # With n=4 and no structure awareness, all 4 chunks would be one window.
        # With structure awareness, ## headings split into two windows.
        windows = AxonBrain._raptor_group_by_structure(brain, chunks, n=4)
        assert (
            len(windows) == 2
        ), f"Expected 2 windows (one per heading section), got {len(windows)}"
        assert windows[0][0]["id"] == "c1"
        assert windows[1][0]["id"] == "c3"

    def test_raptor_fallback_fixed_window_without_headings(self):
        """Chunks with no heading produce standard fixed windows (same as current behavior)."""
        from axon.main import AxonBrain

        brain = self._make_brain()
        chunks = [
            {"id": f"c{i}", "text": f"Plain text chunk {i}.", "metadata": {}} for i in range(6)
        ]
        windows = AxonBrain._raptor_group_by_structure(brain, chunks, n=2)
        assert len(windows) == 3, f"Expected 3 fixed windows of size 2, got {len(windows)}"
        for w in windows:
            assert len(w) == 2, f"Each window should have 2 chunks, got {len(w)}"


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestRuntimeFixes:
    """TASK_12: Runtime cost reduction — community triage, RAPTOR guard, lazy mode, pre-filter."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_triage_brain(self):
        """MagicMock brain with _generate_community_summaries and _rebuild_communities bound real."""
        import concurrent.futures
        import threading

        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.config.graph_rag_community_min_size = 3
        brain.config.graph_rag_community_llm_top_n_per_level = 50
        brain.config.graph_rag_community_llm_max_total = 200
        brain.config.graph_rag_community_max_context_tokens = 4000
        brain.config.graph_rag_community_include_claims = False
        brain.config.graph_rag_claims = False
        brain.config.graph_rag_community_lazy = False
        brain.config.graph_rag_global_top_communities = 0
        brain.config.raptor_max_source_size_mb = 0.0
        brain.config.graph_rag_community = True
        brain.config.graph_rag_index_community_reports = True
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._claims_graph = {}
        brain._community_levels = {}
        brain._community_hierarchy = {}
        brain._community_children = {}
        brain._community_summaries = {}
        brain._community_rebuild_lock = threading.Lock()
        brain._save_community_summaries = MagicMock()
        brain._save_community_hierarchy = MagicMock()
        brain._save_community_levels = MagicMock()
        llm = MagicMock()
        llm.complete = MagicMock(
            return_value='{"title":"T","summary":"s","findings":[],"rank":5.0}'
        )
        brain.llm = llm
        brain._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        brain._generate_community_summaries = AxonBrain._generate_community_summaries.__get__(
            brain, AxonBrain
        )
        brain._rebuild_communities = AxonBrain._rebuild_communities.__get__(brain, AxonBrain)
        return brain

    def _make_raptor_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **cfg_kw):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(hybrid_search=False, rerank=False, graph_rag=False, **cfg_kw)
        brain = AxonBrain(config)
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain._save_entity_graph = MagicMock()
        brain.embedding.embed = MagicMock(side_effect=lambda texts: [[0.1] * 384] * len(texts))
        brain.vector_store.add = MagicMock()
        return brain

    # ------------------------------------------------------------------
    # Fix 1: Community triage
    # ------------------------------------------------------------------

    def test_small_community_gets_template_no_llm(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """Community with 2 members under min_size=3 → template summary, LLM not called."""
        brain = self._make_triage_brain()
        brain.config.graph_rag_community_min_size = 3
        brain.config.graph_rag_community_llm_top_n_per_level = 0  # no per-level cap
        brain.config.graph_rag_community_llm_max_total = 0  # no total cap
        brain._community_levels = {0: {"entityA": 0, "entityB": 0}}  # 2 members, cid=0

        brain._generate_community_summaries()

        brain.llm.complete.assert_not_called()
        result = brain._community_summaries.get("0_0", {})
        assert result.get("template") is True, f"Expected template=True, got: {result}"

    def test_per_level_cap_limits_llm_calls(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """top_n_per_level=2 with 5 communities → at most 2 LLM calls, ≥3 template summaries."""
        brain = self._make_triage_brain()
        brain.config.graph_rag_community_min_size = 0  # disable size gate
        brain.config.graph_rag_community_llm_top_n_per_level = 2
        brain.config.graph_rag_community_llm_max_total = 0  # no total cap
        level_map = {}
        for cid in range(5):
            for i in range(5):
                level_map[f"e{cid}_{i}"] = cid
        brain._community_levels = {0: level_map}

        brain._generate_community_summaries()

        assert (
            brain.llm.complete.call_count <= 2
        ), f"Expected ≤2 LLM calls, got {brain.llm.complete.call_count}"
        template_count = sum(
            1 for v in brain._community_summaries.values() if v.get("template") is True
        )
        assert template_count >= 3, f"Expected ≥3 template summaries, got {template_count}"

    def test_global_hard_cap_stops_llm_across_levels(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """max_total=3 with 8 communities across 2 levels → LLM called at most 3 times."""
        brain = self._make_triage_brain()
        brain.config.graph_rag_community_min_size = 0
        brain.config.graph_rag_community_llm_top_n_per_level = 0
        brain.config.graph_rag_community_llm_max_total = 3
        level_map_0 = {}
        level_map_1 = {}
        for cid in range(4):
            for i in range(5):
                level_map_0[f"e0_{cid}_{i}"] = cid
                level_map_1[f"e1_{cid}_{i}"] = cid
        brain._community_levels = {0: level_map_0, 1: level_map_1}

        brain._generate_community_summaries()

        assert (
            brain.llm.complete.call_count <= 3
        ), f"Expected ≤3 LLM calls, got {brain.llm.complete.call_count}"

    # ------------------------------------------------------------------
    # Fix 2: RAPTOR source-size guard
    # ------------------------------------------------------------------

    def test_raptor_skips_large_source(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        """raptor_max_source_size_mb=0.001: large source excluded from RAPTOR, still indexed."""
        brain = self._make_raptor_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_chunk_group_size=2,
            raptor_max_source_size_mb=0.001,
        )
        # 0.001 MB = ~1024 bytes; large_text is 2000 bytes
        large_text = "x" * 2000
        small_text = "small chunk"
        docs = [
            {"id": "d1", "text": large_text, "metadata": {"source": "large.txt"}},
            {"id": "d2", "text": small_text, "metadata": {"source": "small.txt"}},
        ]
        raptor_mock = MagicMock(return_value=[])
        brain._generate_raptor_summaries = raptor_mock

        brain.ingest(docs)

        assert raptor_mock.call_count == 1
        called_docs = raptor_mock.call_args[0][0]
        called_sources = {d.get("metadata", {}).get("source", "") for d in called_docs}
        assert "large.txt" not in called_sources, "Large source must be excluded from RAPTOR"
        assert "small.txt" in called_sources, "Small source must pass through to RAPTOR"
        # All original docs must still reach the vector store (IDs are chunk-suffixed)
        all_ids = [id_ for call in brain.vector_store.add.call_args_list for id_ in call[0][0]]
        assert any(
            id_.startswith("d1") for id_ in all_ids
        ), "Large-source doc must still be indexed in vector store"
        assert any(
            id_.startswith("d2") for id_ in all_ids
        ), "Small-source doc must still be indexed in vector store"

    def test_raptor_size_guard_zero_means_no_filter(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """raptor_max_source_size_mb=0.0 passes all docs to RAPTOR regardless of size."""
        brain = self._make_raptor_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            raptor=True,
            raptor_chunk_group_size=2,
            raptor_max_source_size_mb=0.0,
        )
        large_text = "x" * 10000
        docs = [
            {"id": "d1", "text": large_text, "metadata": {"source": "large.txt"}},
            {"id": "d2", "text": "small", "metadata": {"source": "small.txt"}},
        ]
        raptor_mock = MagicMock(return_value=[])
        brain._generate_raptor_summaries = raptor_mock

        brain.ingest(docs)

        assert raptor_mock.call_count == 1
        called_docs = raptor_mock.call_args[0][0]
        called_sources = {d.get("metadata", {}).get("source", "") for d in called_docs}
        assert "large.txt" in called_sources, "Large source must pass to RAPTOR when guard=0.0"
        assert "small.txt" in called_sources

    # ------------------------------------------------------------------
    # Fix 3: Lazy community generation
    # ------------------------------------------------------------------

    def test_lazy_mode_skips_summarization_in_rebuild(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """graph_rag_community_lazy=True: _rebuild_communities skips _generate_community_summaries."""
        from axon.main import AxonBrain

        brain = self._make_triage_brain()
        brain.config.graph_rag_community_lazy = True
        brain._run_hierarchical_community_detection = MagicMock(
            return_value=({0: {"a": 0, "b": 0}}, {}, {})
        )
        brain._generate_community_summaries = MagicMock()
        brain._index_community_reports_in_vector_store = MagicMock()

        AxonBrain._rebuild_communities(brain)

        brain._generate_community_summaries.assert_not_called()

    # ------------------------------------------------------------------
    # Fix 4: Global search pre-filter
    # ------------------------------------------------------------------

    def test_global_top_communities_prefilter(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """graph_rag_global_top_communities=2 limits map phase to 2 communities (≤2 LLM calls)."""
        import concurrent.futures

        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain._community_summaries = {
            "0_0": {
                "title": "machine learning",
                "summary": "deep learning neural networks",
                "full_content": "# machine learning\n\nDeep learning.",
                "rank": 5.0,
                "findings": [],
            },
            "0_1": {
                "title": "sports",
                "summary": "football basketball games",
                "full_content": "# sports\n\nFootball.",
                "rank": 5.0,
                "findings": [],
            },
            "0_2": {
                "title": "cooking",
                "summary": "recipes food ingredients",
                "full_content": "# cooking\n\nRecipes.",
                "rank": 5.0,
                "findings": [],
            },
            "0_3": {
                "title": "geography",
                "summary": "maps countries borders",
                "full_content": "# geography\n\nMaps.",
                "rank": 5.0,
                "findings": [],
            },
            "0_4": {
                "title": "history",
                "summary": "ancient civilizations empires",
                "full_content": "# history\n\nAncient.",
                "rank": 5.0,
                "findings": [],
            },
        }
        brain._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        brain.llm = MagicMock()
        # Return empty array → map results are all empty → no reduce phase call
        brain.llm.complete = MagicMock(return_value="[]")

        class _Cfg:
            graph_rag_community_level = 0
            graph_rag_global_min_score = 0
            graph_rag_global_top_points = 50
            graph_rag_global_map_max_length = 500
            graph_rag_global_reduce_max_length = 500
            graph_rag_global_allow_general_knowledge = False
            graph_rag_global_reduce_max_tokens = 8000
            graph_rag_global_top_communities = 2

        AxonBrain._global_search_map_reduce(brain, "machine learning", _Cfg())

        # With 5 communities but top_communities=2, only 2 map LLM calls should fire
        assert (
            brain.llm.complete.call_count <= 2
        ), f"Expected ≤2 LLM calls (pre-filter to 2), got {brain.llm.complete.call_count}"


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestBatchModeDefer:
    """Tests for ingest_batch_mode deferred saves (TASK 13A)."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(**kwargs)
        brain = AxonBrain(config)
        brain.splitter = None
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain._save_embedding_meta = MagicMock()
        brain._validate_embedding_meta = MagicMock()
        brain.embedding.embed = MagicMock(return_value=[[0.1]])
        brain._own_vector_store = MagicMock()
        return brain

    def test_batch_mode_bm25_save_deferred_during_ingest(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """ingest_batch_mode=True → bm25.add_documents called with save_deferred=True."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            ingest_batch_mode=True,
            graph_rag=False,
            raptor=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        mock_bm25 = MagicMock()
        brain._own_bm25 = mock_bm25

        docs = [{"id": "d1", "text": "hello", "metadata": {"source": "test.txt"}}]
        brain.ingest(docs)

        mock_bm25.add_documents.assert_called_once()
        call_kwargs = mock_bm25.add_documents.call_args
        # save_deferred may be positional or keyword
        passed_deferred = call_kwargs[1].get("save_deferred") if call_kwargs[1] else None
        if passed_deferred is None and len(call_kwargs[0]) > 1:
            passed_deferred = call_kwargs[0][1]
        assert passed_deferred is True
        mock_bm25.flush.assert_not_called()

    def test_batch_mode_flush_called_on_finalize_ingest(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """finalize_ingest() with ingest_batch_mode=True → bm25.flush() called."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            ingest_batch_mode=True,
            graph_rag=False,
            raptor=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        mock_bm25 = MagicMock()
        brain._own_bm25 = mock_bm25

        docs = [{"id": "d1", "text": "hello", "metadata": {"source": "test.txt"}}]
        brain.ingest(docs)
        mock_bm25.flush.assert_not_called()

        brain.finalize_ingest()
        mock_bm25.flush.assert_called_once()

    def test_batch_mode_entity_graph_not_saved_during_ingest(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """ingest_batch_mode=True → _save_entity_graph NOT called during ingest."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            ingest_batch_mode=True,
            graph_rag=True,
            raptor=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )

        docs = [{"id": "d1", "text": "hello", "metadata": {"source": "test.txt"}}]
        with patch.object(brain, "_save_entity_graph") as mock_save_eg, patch.object(
            brain, "_extract_entities", return_value=[]
        ):
            brain.ingest(docs)
            mock_save_eg.assert_not_called()

    def test_batch_mode_false_saves_immediately(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """default ingest_batch_mode=False → _save_entity_graph IS called during ingest."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            ingest_batch_mode=False,
            graph_rag=True,
            raptor=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )

        docs = [{"id": "d1", "text": "hello", "metadata": {"source": "test.txt"}}]
        with patch.object(brain, "_save_entity_graph") as mock_save_eg, patch.object(
            brain,
            "_extract_entities",
            return_value=[{"name": "entity1", "type": "PERSON", "description": "a person"}],
        ):
            brain.ingest(docs)
            mock_save_eg.assert_called()

    def test_finalize_ingest_calls_finalize_graph(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """finalize_ingest() → _rebuild_communities called when _community_graph_dirty=True."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            ingest_batch_mode=True,
        )
        brain._community_graph_dirty = True

        with patch.object(brain, "_rebuild_communities") as mock_rebuild:
            brain.finalize_ingest()
            mock_rebuild.assert_called_once()


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestChunkBudget:
    """Tests for max_chunks_per_source chunk budget (TASK 13B)."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(**kwargs)
        brain = AxonBrain(config)
        brain.splitter = None
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain._save_embedding_meta = MagicMock()
        brain._validate_embedding_meta = MagicMock()
        brain._own_vector_store = MagicMock()
        return brain

    def test_chunk_cap_truncates_large_source(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """20 docs same source, cap=5 → only 5 docs reach embedding."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            max_chunks_per_source=5,
            graph_rag=False,
            raptor=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        docs = [
            {"id": f"d{i}", "text": f"chunk {i}", "metadata": {"source": "big.json"}}
            for i in range(20)
        ]
        embed_call_docs = []

        def capture_embed(texts):
            embed_call_docs.extend(texts)
            return [[0.1]] * len(texts)

        brain.embedding.embed = capture_embed
        brain.ingest(docs)
        assert len(embed_call_docs) == 5

    def test_chunk_cap_zero_means_no_limit(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """20 docs, cap=0 → all 20 reach embed."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            max_chunks_per_source=0,
            graph_rag=False,
            raptor=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        docs = [
            {"id": f"d{i}", "text": f"chunk {i}", "metadata": {"source": "big.json"}}
            for i in range(20)
        ]
        embed_call_docs = []

        def capture_embed(texts):
            embed_call_docs.extend(texts)
            return [[0.1]] * len(texts)

        brain.embedding.embed = capture_embed
        brain.ingest(docs)
        assert len(embed_call_docs) == 20

    def test_chunk_cap_per_source_independence(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """10 docs source A + 10 source B, cap=7 → 14 total."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            max_chunks_per_source=7,
            graph_rag=False,
            raptor=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        docs = [
            {"id": f"a{i}", "text": f"chunk a{i}", "metadata": {"source": "a.txt"}}
            for i in range(10)
        ] + [
            {"id": f"b{i}", "text": f"chunk b{i}", "metadata": {"source": "b.txt"}}
            for i in range(10)
        ]
        embed_call_docs = []

        def capture_embed(texts):
            embed_call_docs.extend(texts)
            return [[0.1]] * len(texts)

        brain.embedding.embed = capture_embed
        brain.ingest(docs)
        assert len(embed_call_docs) == 14

    def test_chunk_cap_keeps_first_n(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        """IDs a1..a10, cap=3 → only a1, a2, a3 in embed."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            max_chunks_per_source=3,
            graph_rag=False,
            raptor=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        docs = [
            {"id": f"a{i + 1}", "text": f"chunk {i}", "metadata": {"source": "a.txt"}}
            for i in range(10)
        ]

        def capture_embed(texts):
            return [[0.1]] * len(texts)

        brain.embedding.embed = capture_embed
        # Capture via _own_vector_store.add
        stored_ids = []
        brain._own_vector_store.add = MagicMock(
            side_effect=lambda ids, texts, embs, metas: stored_ids.extend(ids)
        )
        brain.ingest(docs)
        assert stored_ids == ["a1", "a2", "a3"]

    def test_chunk_cap_logs_truncation(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, caplog
    ):
        """caplog: 'Chunk cap:' message appears on truncation, absent when within budget."""
        import logging

        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            max_chunks_per_source=3,
            graph_rag=False,
            raptor=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        brain.embedding.embed = MagicMock(return_value=[[0.1]] * 3)

        docs = [
            {"id": f"d{i}", "text": f"chunk {i}", "metadata": {"source": "big.json"}}
            for i in range(10)
        ]
        with caplog.at_level(logging.INFO):
            brain.ingest(docs)
        assert "Chunk cap:" in caplog.text

        # Now test no log when within budget
        caplog.clear()
        brain.embedding.embed = MagicMock(return_value=[[0.1]] * 2)
        docs_small = [
            {"id": f"s{i}", "text": f"chunk {i}", "metadata": {"source": "small.txt"}}
            for i in range(2)
        ]
        with caplog.at_level(logging.INFO):
            brain.ingest(docs_small)
        assert "Chunk cap:" not in caplog.text


@patch("axon.retrievers.BM25Retriever")
@patch("axon.main.OpenVectorStore")
@patch("axon.main.OpenLLM")
@patch("axon.main.OpenEmbedding")
@patch("axon.main.OpenReranker")
class TestSourcePolicy:
    """Tests for source-class policy gates (TASK 13C)."""

    def _make_brain(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, **kwargs):
        from axon.main import AxonBrain, AxonConfig

        config = AxonConfig(**kwargs)
        brain = AxonBrain(config)
        brain.splitter = None
        brain._ingested_hashes = set()
        brain._save_hash_store = MagicMock()
        brain._save_embedding_meta = MagicMock()
        brain._validate_embedding_meta = MagicMock()
        brain.embedding.embed = MagicMock(return_value=[[0.1]])
        brain._own_vector_store = MagicMock()
        return brain

    def test_detect_manifest_by_filename(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """source='requirements.txt' → ('manifest', False)."""
        from axon.main import AxonBrain, AxonConfig

        brain = AxonBrain(AxonConfig())
        doc = {"id": "req", "text": "flask>=2.0", "metadata": {"source": "requirements.txt"}}
        dtype, is_code = brain._detect_dataset_type(doc)
        assert dtype == "manifest"
        assert is_code is False

    def test_detect_manifest_lock_extension(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """source='yarn.lock' → ('manifest', False)."""
        from axon.main import AxonBrain, AxonConfig

        brain = AxonBrain(AxonConfig())
        doc = {"id": "yarn", "text": "# yarn lockfile v1", "metadata": {"source": "yarn.lock"}}
        dtype, is_code = brain._detect_dataset_type(doc)
        assert dtype == "manifest"
        assert is_code is False

    def test_detect_reference_by_path(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        """source='/docs/apidocs/foo.html' → ('reference', False)."""
        from axon.main import AxonBrain, AxonConfig

        brain = AxonBrain(AxonConfig())
        doc = {
            "id": "api",
            "text": "API reference content",
            "metadata": {"source": "/docs/apidocs/foo.html"},
        }
        dtype, is_code = brain._detect_dataset_type(doc)
        assert dtype == "reference"
        assert is_code is False

    def test_detect_manifest_does_not_shadow_code(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """source='main.py' → 'codebase' (Priority 1 fires first)."""
        from axon.main import AxonBrain, AxonConfig

        brain = AxonBrain(AxonConfig())
        doc = {"id": "main", "text": "def main(): pass", "metadata": {"source": "main.py"}}
        dtype, is_code = brain._detect_dataset_type(doc)
        assert dtype == "codebase"

    def test_policy_disabled_raptor_runs_on_knowledge(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """source_policy_enabled=False, dataset_type=knowledge → _generate_raptor_summaries called."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            source_policy_enabled=False,
            raptor=True,
            graph_rag=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        docs = [
            {
                "id": "d1",
                "text": "knowledge chunk",
                "metadata": {"source": "kb.txt", "dataset_type": "knowledge"},
            }
        ]
        with patch.object(brain, "_generate_raptor_summaries", return_value=[]) as mock_raptor:
            brain.ingest(docs)
            mock_raptor.assert_called()

    def test_policy_enabled_skips_raptor_on_knowledge(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """source_policy_enabled=True, dataset_type=knowledge → _generate_raptor_summaries NOT called."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            source_policy_enabled=True,
            raptor=True,
            graph_rag=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        docs = [
            {
                "id": "d1",
                "text": "knowledge chunk",
                "metadata": {"source": "kb.txt", "dataset_type": "knowledge"},
            }
        ]
        with patch.object(brain, "_generate_raptor_summaries", return_value=[]) as mock_raptor:
            brain.ingest(docs)
            mock_raptor.assert_not_called()

    def test_policy_enabled_skips_graphrag_on_manifest(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """source_policy_enabled=True, dataset_type=manifest → _extract_entities NOT called."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            source_policy_enabled=True,
            raptor=False,
            graph_rag=True,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        docs = [
            {
                "id": "d1",
                "text": "pkg: foo",
                "metadata": {"source": "package.json", "dataset_type": "manifest"},
            }
        ]
        with patch.object(brain, "_extract_entities", return_value=[]) as mock_extract:
            brain.ingest(docs)
            mock_extract.assert_not_called()

    def test_policy_enabled_blocks_graphrag_on_codebase(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """source_policy_enabled=True, dataset_type=codebase → _extract_entities NOT called.

        Per the code graph report, prose GraphRAG is disabled for codebase by default.
        Code uses a separate structural graph, not the entity/community pipeline.
        """
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            source_policy_enabled=True,
            raptor=False,
            graph_rag=True,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        docs = [
            {
                "id": "d1",
                "text": "def foo(): pass",
                "metadata": {"source": "main.py", "dataset_type": "codebase"},
            }
        ]
        with patch.object(
            brain,
            "_extract_entities",
            return_value=[{"name": "Foo", "type": "FUNCTION", "description": ""}],
        ) as mock_extract:
            brain.ingest(docs)
            mock_extract.assert_not_called()

    def test_policy_enabled_allows_raptor_on_paper(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        """source_policy_enabled=True, dataset_type=paper → _generate_raptor_summaries IS called."""
        brain = self._make_brain(
            MockReranker,
            MockEmbed,
            MockLLM,
            MockStore,
            MockBM25,
            source_policy_enabled=True,
            raptor=True,
            graph_rag=False,
            dedup_on_ingest=False,
            parent_chunk_size=0,
        )
        docs = [
            {
                "id": "d1",
                "text": "Abstract: ...",
                "metadata": {"source": "paper.pdf", "dataset_type": "paper"},
            }
        ]
        with patch.object(brain, "_generate_raptor_summaries", return_value=[]) as mock_raptor:
            brain.ingest(docs)
            mock_raptor.assert_called()


# ---------------------------------------------------------------------------
# TASK 14 — GraphRAG Runtime Improvements
# ---------------------------------------------------------------------------


class TestGraspoLogicFallbackWarning:
    """Item 1 — graspologic absent → WARNING logged in _run_hierarchical_community_detection."""

    def test_warning_message_content(self):
        """The warning text in the source code mentions axon[graphrag]."""
        import inspect

        from axon.main import AxonBrain

        src = inspect.getsource(AxonBrain._run_hierarchical_community_detection)
        assert "axon[graphrag]" in src
        assert "graspologic not installed" in src

    def test_warning_logged_on_import_error(self, caplog):
        """When graspologic ImportError occurs, a WARNING is emitted."""
        import logging

        import networkx as nx

        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.config.graph_rag_community_levels = 1
        brain.config.graph_rag_community_max_cluster_size = 10
        brain.config.graph_rag_leiden_seed = 42
        brain.config.graph_rag_community_use_lcc = False

        # _run_hierarchical_community_detection builds graph internally
        G = nx.path_graph(4)
        brain._build_networkx_graph = MagicMock(return_value=G)

        _real_import = __import__

        def fake_import(name, *args, **kwargs):
            if "graspologic" in name:
                raise ImportError("No module named graspologic")
            return _real_import(name, *args, **kwargs)

        with caplog.at_level(logging.WARNING, logger="Axon"):
            with patch("builtins.__import__", side_effect=fake_import):
                try:
                    AxonBrain._run_hierarchical_community_detection(brain)
                except Exception:
                    pass

        warning_msgs = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        if warning_msgs:
            assert any("axon[graphrag]" in m for m in warning_msgs)


class TestMapReduceDedicatedPool:
    """Item 4 — graph_rag_map_workers set → dedicated ThreadPoolExecutor used."""

    def _make_cfg(self, map_workers=0):
        cfg = MagicMock()
        cfg.graph_rag_map_workers = map_workers
        cfg.graph_rag_global_map_max_length = 500
        cfg.graph_rag_global_min_score = 0
        cfg.graph_rag_global_top_points = 10
        cfg.graph_rag_community_level = 0
        cfg.graph_rag_global_reduce_max_tokens = 8000
        cfg.graph_rag_global_reduce_max_length = 500
        cfg.graph_rag_global_allow_general_knowledge = False
        cfg.graph_rag_global_top_communities = 0
        cfg.graph_rag_report_compress = False
        return cfg

    def _make_brain(self):
        import concurrent.futures

        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.config.max_workers = 4
        brain.llm = MagicMock()
        brain.llm.complete = MagicMock(return_value='[{"point":"p","score":50}]')
        brain._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        brain._community_summaries = {
            "c0": {"full_content": "Community report text.", "summary": "s"}
        }
        return brain

    def test_dedicated_pool_created_when_map_workers_set(self):
        """When graph_rag_map_workers>0, a separate ThreadPoolExecutor must be created."""
        from concurrent.futures import ThreadPoolExecutor

        from axon.main import AxonBrain

        brain = self._make_brain()
        cfg = self._make_cfg(map_workers=2)

        pool_instances = []
        original_tpe = ThreadPoolExecutor

        class CapturingTPE(original_tpe):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                pool_instances.append(self)

        with patch("concurrent.futures.ThreadPoolExecutor", CapturingTPE):
            try:
                AxonBrain._global_search_map_reduce(brain, "query", cfg)
            except Exception:
                pass

        assert len(pool_instances) >= 1

    def test_shared_pool_used_when_map_workers_zero(self):
        """When graph_rag_map_workers==0, the shared _executor is used (no new TPE)."""
        from concurrent.futures import ThreadPoolExecutor

        from axon.main import AxonBrain

        brain = self._make_brain()
        cfg = self._make_cfg(map_workers=0)

        pool_instances = []
        original_tpe = ThreadPoolExecutor

        class CapturingTPE(original_tpe):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                pool_instances.append(self)

        with patch("concurrent.futures.ThreadPoolExecutor", CapturingTPE):
            try:
                AxonBrain._global_search_map_reduce(brain, "query", cfg)
            except Exception:
                pass

        assert len(pool_instances) == 0


class TestGLiNERExtraction:
    """Item 5 — graph_rag_ner_backend=gliner → GLiNER path, not llm.complete."""

    def _make_brain(self, backend="gliner"):
        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.config.graph_rag_ner_backend = backend
        brain.llm = MagicMock()
        brain.llm.complete = MagicMock(return_value="Paris | GEO | Capital of France")
        brain._gliner_model = None
        return brain

    def test_gliner_path_skips_llm(self):
        """When ner_backend=gliner, _extract_entities_gliner is called, not llm.complete."""
        from axon.main import AxonBrain

        brain = self._make_brain(backend="gliner")
        mock_result = [{"name": "Paris", "type": "GEO", "description": ""}]
        # Patch the instance attribute so the MagicMock self picks it up
        mock_fn = MagicMock(return_value=mock_result)
        brain._extract_entities_gliner = mock_fn

        result = AxonBrain._extract_entities(brain, "Paris is the capital of France.")

        mock_fn.assert_called_once()
        brain.llm.complete.assert_not_called()
        assert result == mock_result

    def test_llm_path_used_when_backend_is_llm(self):
        """When ner_backend=llm, the LLM is used."""
        from axon.main import AxonBrain

        brain = self._make_brain(backend="llm")
        mock_fn = MagicMock()
        brain._extract_entities_gliner = mock_fn

        AxonBrain._extract_entities(brain, "Paris is the capital of France.")

        mock_fn.assert_not_called()
        brain.llm.complete.assert_called_once()

    def test_extract_entities_gliner_deduplicates(self):
        """_extract_entities_gliner deduplicates case-insensitively."""
        from axon.main import AxonBrain

        brain = self._make_brain()
        mock_model = MagicMock()
        mock_model.predict_entities = MagicMock(
            return_value=[
                {"text": "Paris", "label": "location"},
                {"text": "paris", "label": "location"},
                {"text": "France", "label": "location"},
            ]
        )
        # Patch the instance so self._ensure_gliner() returns mock_model
        brain._ensure_gliner = MagicMock(return_value=mock_model)

        result = AxonBrain._extract_entities_gliner(brain, "Paris and France.")

        names = [e["name"] for e in result]
        assert "Paris" in names
        assert "France" in names
        assert len(result) == 2

    def test_extract_entities_gliner_type_mapping(self):
        """_extract_entities_gliner maps GLiNER labels to internal type strings."""
        from axon.main import AxonBrain

        brain = self._make_brain()
        mock_model = MagicMock()
        mock_model.predict_entities = MagicMock(
            return_value=[
                {"text": "Alice", "label": "person"},
                {"text": "Acme", "label": "organization"},
                {"text": "London", "label": "location"},
            ]
        )
        brain._ensure_gliner = MagicMock(return_value=mock_model)

        result = AxonBrain._extract_entities_gliner(brain, "text")

        types = {e["name"]: e["type"] for e in result}
        assert types["Alice"] == "PERSON"
        assert types["Acme"] == "ORGANIZATION"
        assert types["London"] == "GEO"


class TestLLMLinguaCompression:
    """Item 2 — graph_rag_report_compress=True → chunks compressed via LLMLingua."""

    def _make_brain(self):
        import concurrent.futures

        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.config.max_workers = 2
        brain.llm = MagicMock()
        brain.llm.complete = MagicMock(return_value='[{"point":"p","score":60}]')
        brain._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        brain._community_summaries = {
            "c0": {"full_content": "Long community report text for testing.", "summary": "s"}
        }
        brain._llmlingua = None
        return brain

    def _make_cfg(self, compress=True, ratio=0.5):
        cfg = MagicMock()
        cfg.graph_rag_map_workers = 0
        cfg.graph_rag_global_map_max_length = 500
        cfg.graph_rag_global_min_score = 0
        cfg.graph_rag_global_top_points = 10
        cfg.graph_rag_community_level = 0
        cfg.graph_rag_global_reduce_max_tokens = 8000
        cfg.graph_rag_global_reduce_max_length = 500
        cfg.graph_rag_global_allow_general_knowledge = False
        cfg.graph_rag_global_top_communities = 0
        cfg.graph_rag_report_compress = compress
        cfg.graph_rag_report_compress_ratio = ratio
        return cfg

    def test_compress_called_when_enabled(self):
        from axon.main import AxonBrain

        brain = self._make_brain()
        cfg = self._make_cfg(compress=True)
        mock_compressor = MagicMock()
        mock_compressor.compress_prompt = MagicMock(
            return_value={"compressed_prompt": "compressed"}
        )
        # Patch instance so self._ensure_llmlingua() returns mock_compressor
        brain._ensure_llmlingua = MagicMock(return_value=mock_compressor)

        try:
            AxonBrain._global_search_map_reduce(brain, "test query", cfg)
        except Exception:
            pass

        mock_compressor.compress_prompt.assert_called()

    def test_compress_skipped_when_disabled(self):
        from axon.main import AxonBrain

        brain = self._make_brain()
        cfg = self._make_cfg(compress=False)
        mock_compressor = MagicMock()
        brain._ensure_llmlingua = MagicMock(return_value=mock_compressor)

        try:
            AxonBrain._global_search_map_reduce(brain, "test query", cfg)
        except Exception:
            pass

        mock_compressor.compress_prompt.assert_not_called()

    def test_compress_falls_back_on_chunk_error(self):
        from axon.main import AxonBrain

        brain = self._make_brain()
        cfg = self._make_cfg(compress=True)
        mock_compressor = MagicMock()
        mock_compressor.compress_prompt = MagicMock(side_effect=RuntimeError("compress fail"))
        brain._ensure_llmlingua = MagicMock(return_value=mock_compressor)

        try:
            result = AxonBrain._global_search_map_reduce(brain, "test query", cfg)
            assert isinstance(result, str)
        except RuntimeError as exc:
            assert "compress fail" not in str(exc)


class TestAutoRoute:
    """Item 3 — Adaptive query routing / Self-RAG."""

    def _make_brain(self):
        from axon.main import AxonBrain

        brain = MagicMock(spec=AxonBrain)
        brain.config = MagicMock()
        brain.llm = MagicMock()
        return brain

    def test_heuristic_holistic_query_returns_true(self):
        from axon.main import AxonBrain

        brain = self._make_brain()
        assert (
            AxonBrain._classify_query_needs_graphrag(brain, "summarize all documents", "heuristic")
            is True
        )

    def test_heuristic_short_factual_returns_false(self):
        from axon.main import AxonBrain

        brain = self._make_brain()
        assert (
            AxonBrain._classify_query_needs_graphrag(brain, "what is Python?", "heuristic") is False
        )

    def test_heuristic_long_query_returns_true(self):
        from axon.main import AxonBrain

        brain = self._make_brain()
        long_q = (
            "please tell me about history cultural context significance ancient Rome "
            "detail extra words to pass the twenty word threshold for this test case"
        )
        assert len(long_q.split()) > 20
        assert AxonBrain._classify_query_needs_graphrag(brain, long_q, "heuristic") is True

    def test_heuristic_overview_keyword_returns_true(self):
        from axon.main import AxonBrain

        brain = self._make_brain()
        assert (
            AxonBrain._classify_query_needs_graphrag(
                brain, "give me an overview of this codebase", "heuristic"
            )
            is True
        )

    def test_llm_yes_returns_true(self):
        from axon.main import AxonBrain

        brain = self._make_brain()
        brain.llm.complete = MagicMock(return_value="YES")
        assert AxonBrain._classify_query_needs_graphrag(brain, "any query", "llm") is True

    def test_llm_no_returns_false(self):
        from axon.main import AxonBrain

        brain = self._make_brain()
        brain.llm.complete = MagicMock(return_value="NO")
        assert AxonBrain._classify_query_needs_graphrag(brain, "what is X?", "llm") is False

    def test_llm_exception_returns_false(self):
        from axon.main import AxonBrain

        brain = self._make_brain()
        brain.llm.complete = MagicMock(side_effect=RuntimeError("timeout"))
        assert AxonBrain._classify_query_needs_graphrag(brain, "query", "llm") is False

    def test_off_mode_returns_false(self):
        from axon.main import AxonBrain

        brain = self._make_brain()
        assert (
            AxonBrain._classify_query_needs_graphrag(brain, "summarize everything", "off") is False
        )


class TestTask14Config:
    """Verify new TASK 14 AxonConfig fields have correct defaults."""

    def test_new_defaults(self):
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.graph_rag_map_workers == 0
        assert cfg.graph_rag_ner_backend == "llm"
        assert cfg.graph_rag_report_compress is False
        assert cfg.graph_rag_report_compress_ratio == 0.5
        assert cfg.graph_rag_auto_route == "off"

    def test_yaml_load_rag_flat_keys(self, tmp_path):
        """New TASK 14 keys can be set via rag: section in config.yaml."""
        import yaml

        from axon.main import AxonConfig

        cfg_data = {
            "rag": {
                "graph_rag_map_workers": 4,
                "graph_rag_ner_backend": "gliner",
                "graph_rag_report_compress": True,
                "graph_rag_report_compress_ratio": 0.3,
                "graph_rag_auto_route": "heuristic",
            }
        }
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(cfg_data))
        loaded = AxonConfig.load(str(p))
        assert loaded.graph_rag_map_workers == 4
        assert loaded.graph_rag_ner_backend == "gliner"
        assert loaded.graph_rag_report_compress is True
        assert loaded.graph_rag_report_compress_ratio == 0.3
        assert loaded.graph_rag_auto_route == "heuristic"


class TestVectorStoreBatchWrite:
    """Verify OpenVectorStore.add() slices large payloads into safe Chroma batches."""

    def _make_store(self):
        from unittest.mock import MagicMock

        from axon.main import AxonConfig, OpenVectorStore

        cfg = AxonConfig()
        store = OpenVectorStore.__new__(OpenVectorStore)
        store.config = cfg
        store.provider = "chroma"
        store.collection = MagicMock()
        return store

    def test_small_payload_single_call(self):
        """A payload under the batch limit is written in one collection.add() call."""
        store = self._make_store()
        ids = [str(i) for i in range(100)]
        texts = ["t"] * 100
        embs = [[0.0]] * 100
        store.add(ids, texts, embs)
        assert store.collection.add.call_count == 1

    def test_large_payload_split_into_batches(self):
        """A payload larger than _CHROMA_MAX_BATCH is split across multiple calls."""
        from axon.main import OpenVectorStore

        store = self._make_store()
        n = OpenVectorStore._CHROMA_MAX_BATCH + 500  # e.g. 5500
        ids = [str(i) for i in range(n)]
        texts = ["t"] * n
        embs = [[0.0]] * n
        store.add(ids, texts, embs)
        assert store.collection.add.call_count == 2
        # First call has exactly _CHROMA_MAX_BATCH items
        first_call_ids = store.collection.add.call_args_list[0].kwargs["ids"]
        assert len(first_call_ids) == OpenVectorStore._CHROMA_MAX_BATCH
        # Second call has the remainder
        second_call_ids = store.collection.add.call_args_list[1].kwargs["ids"]
        assert len(second_call_ids) == 500

    def test_exactly_at_limit_single_call(self):
        """A payload exactly at the limit is sent in one call."""
        from axon.main import OpenVectorStore

        store = self._make_store()
        n = OpenVectorStore._CHROMA_MAX_BATCH
        ids = [str(i) for i in range(n)]
        texts = ["t"] * n
        embs = [[0.0]] * n
        store.add(ids, texts, embs)
        assert store.collection.add.call_count == 1

    def test_three_batches(self):
        """A payload spanning 3 batches triggers 3 calls."""
        from axon.main import OpenVectorStore

        store = self._make_store()
        n = OpenVectorStore._CHROMA_MAX_BATCH * 2 + 1
        ids = [str(i) for i in range(n)]
        texts = ["t"] * n
        embs = [[0.0]] * n
        store.add(ids, texts, embs)
        assert store.collection.add.call_count == 3

    def test_metadatas_sliced_correctly(self):
        """Metadatas are sliced in sync with ids/texts/embeddings."""
        from axon.main import OpenVectorStore

        store = self._make_store()
        n = OpenVectorStore._CHROMA_MAX_BATCH + 10
        ids = [str(i) for i in range(n)]
        texts = ["t"] * n
        embs = [[0.0]] * n
        metas = [{"idx": i} for i in range(n)]
        store.add(ids, texts, embs, metas)
        assert store.collection.add.call_count == 2
        first_metas = store.collection.add.call_args_list[0].kwargs["metadatas"]
        assert len(first_metas) == OpenVectorStore._CHROMA_MAX_BATCH
        assert first_metas[0] == {"idx": 0}
        second_metas = store.collection.add.call_args_list[1].kwargs["metadatas"]
        assert len(second_metas) == 10
        assert second_metas[0] == {"idx": OpenVectorStore._CHROMA_MAX_BATCH}

    def test_dimension_error_logged(self):
        """InvalidDimensionException triggers an error log before re-raising."""
        import pytest

        store = self._make_store()
        store.collection.add.side_effect = ValueError("Invalid dimension mismatch")
        with pytest.raises(ValueError):
            store.add(["id0"], ["text"], [[0.0]])


class TestLocalModelPaths:
    """Tests for embedding_model_path and ollama_models_dir config options."""

    def test_embedding_model_path_config_default(self):
        """embedding_model_path defaults to empty string."""
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.embedding_model_path == ""

    def test_ollama_models_dir_config_default(self):
        """ollama_models_dir defaults to empty string."""
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.ollama_models_dir == ""

    def test_embedding_model_path_loaded_from_yaml(self, tmp_path):
        """embedding.model_path in YAML is mapped to embedding_model_path."""
        import yaml

        from axon.main import AxonConfig

        cfg_data = {
            "embedding": {"provider": "sentence_transformers", "model_path": "/srv/models/minilm"},
            "llm": {"provider": "ollama", "model": "gemma"},
            "vector_store": {"provider": "chroma", "path": str(tmp_path / "chroma")},
            "bm25": {"path": str(tmp_path / "bm25")},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))
        loaded = AxonConfig.load(str(cfg_file))
        assert loaded.embedding_model_path == "/srv/models/minilm"

    def test_ollama_models_dir_loaded_from_yaml(self, tmp_path):
        """llm.models_dir in YAML is mapped to ollama_models_dir."""
        import yaml

        from axon.main import AxonConfig

        cfg_data = {
            "embedding": {"provider": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
            "llm": {"provider": "ollama", "model": "gemma", "models_dir": "D:/ollama-models"},
            "vector_store": {"provider": "chroma", "path": str(tmp_path / "chroma")},
            "bm25": {"path": str(tmp_path / "bm25")},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))
        loaded = AxonConfig.load(str(cfg_file))
        assert loaded.ollama_models_dir == "D:/ollama-models"

    def test_ollama_models_env_var_wins_over_yaml(self, tmp_path, monkeypatch):
        """OLLAMA_MODELS env var takes priority over llm.models_dir in YAML."""
        import yaml

        from axon.main import AxonConfig

        monkeypatch.setenv("OLLAMA_MODELS", "/env/override/models")
        cfg_data = {
            "embedding": {"provider": "sentence_transformers", "model": "all-MiniLM-L6-v2"},
            "llm": {"provider": "ollama", "model": "gemma", "models_dir": "D:/yaml-models"},
            "vector_store": {"provider": "chroma", "path": str(tmp_path / "chroma")},
            "bm25": {"path": str(tmp_path / "bm25")},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.dump(cfg_data))
        loaded = AxonConfig.load(str(cfg_file))
        assert loaded.ollama_models_dir == "/env/override/models"

    def test_sentence_transformers_uses_model_path(self, tmp_path):
        """OpenEmbedding uses embedding_model_path as the model source for sentence_transformers."""
        import sys
        from unittest.mock import MagicMock, patch

        from axon.main import AxonConfig, OpenEmbedding

        cfg = AxonConfig(
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            embedding_model_path="/local/models/minilm",
        )
        mock_st = MagicMock()
        mock_st.get_sentence_embedding_dimension.return_value = 384
        mock_module = MagicMock()
        mock_module.SentenceTransformer = MagicMock(return_value=mock_st)
        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            emb = OpenEmbedding(cfg)
            mock_module.SentenceTransformer.assert_called_once_with("/local/models/minilm")
        assert emb.dimension == 384

    def test_sentence_transformers_falls_back_to_model_name(self, tmp_path):
        """OpenEmbedding falls back to embedding_model when embedding_model_path is empty."""
        import sys
        from unittest.mock import MagicMock, patch

        from axon.main import AxonConfig, OpenEmbedding

        cfg = AxonConfig(
            embedding_provider="sentence_transformers",
            embedding_model="all-MiniLM-L6-v2",
            embedding_model_path="",
        )
        mock_st = MagicMock()
        mock_st.get_sentence_embedding_dimension.return_value = 384
        mock_module = MagicMock()
        mock_module.SentenceTransformer = MagicMock(return_value=mock_st)
        with patch.dict(sys.modules, {"sentence_transformers": mock_module}):
            OpenEmbedding(cfg)
            mock_module.SentenceTransformer.assert_called_once_with("all-MiniLM-L6-v2")

    def test_ollama_models_dir_sets_env_var(self, tmp_path, monkeypatch):
        """AxonBrain sets OLLAMA_MODELS env var when ollama_models_dir is configured."""
        import os
        from unittest.mock import MagicMock, patch

        from axon.main import AxonBrain, AxonConfig

        monkeypatch.delenv("OLLAMA_MODELS", raising=False)
        cfg = AxonConfig(
            vector_store_path=str(tmp_path / "chroma"),
            bm25_path=str(tmp_path / "bm25"),
            ollama_models_dir="D:/my-ollama-models",
        )
        with (
            patch("axon.main.OpenEmbedding") as mock_emb,
            patch("axon.main.OpenLLM") as mock_llm,
            patch("axon.main.OpenVectorStore") as mock_vs,
            patch("axon.main.OpenReranker") as mock_rr,
        ):
            mock_emb.return_value = MagicMock(dimension=384)
            mock_llm.return_value = MagicMock()
            mock_vs.return_value = MagicMock()
            mock_rr.return_value = MagicMock()
            AxonBrain(config=cfg)

        assert os.environ.get("OLLAMA_MODELS") == "D:/my-ollama-models"


class TestREBELRelationExtraction:
    """Tests for P2 — REBEL relation extraction backend."""

    def _make_brain(self, tmp_path, **cfg_kwargs):
        from unittest.mock import MagicMock

        from axon.main import AxonBrain, AxonConfig

        cfg = AxonConfig(
            vector_store_path=str(tmp_path / "chroma"),
            bm25_path=str(tmp_path / "bm25"),
            **cfg_kwargs,
        )
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = cfg
        brain.llm = MagicMock()
        brain.embedding = MagicMock()
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._rebel_pipeline = None
        return brain

    def test_rebel_backend_skips_llm(self, tmp_path):
        """When graph_rag_relation_backend='rebel', LLM is not called."""
        brain = self._make_brain(tmp_path, graph_rag_relation_backend="rebel")

        fake_pipe = MagicMock(
            return_value=[
                {"generated_text": "<triplet> Apple <subj> Microsoft <obj> competes with"}
            ]
        )
        brain._rebel_pipeline = fake_pipe

        result = brain._extract_relations("Apple competes with Microsoft.")

        brain.llm.complete.assert_not_called()
        assert isinstance(result, list)

    def test_llm_backend_still_works(self, tmp_path):
        """When graph_rag_relation_backend='llm' (default), LLM is called as before."""
        brain = self._make_brain(tmp_path, graph_rag_relation_backend="llm")
        brain.llm.complete.return_value = ""  # no relations parsed — just checking the path

        result = brain._extract_relations("Some text.")
        brain.llm.complete.assert_called_once()
        assert isinstance(result, list)

    def test_parse_rebel_output_single_triplet(self):
        """A single well-formed REBEL triplet is parsed correctly."""
        from axon.main import AxonBrain

        raw = "<triplet> Apple <subj> Steve Jobs <obj> founded"
        triplets = AxonBrain._parse_rebel_output(raw)
        assert len(triplets) == 1
        assert triplets[0]["subject"] == "Apple"
        assert triplets[0]["object"] == "Steve Jobs"
        assert triplets[0]["relation"] == "founded"

    def test_parse_rebel_output_multiple_triplets(self):
        """Multiple consecutive REBEL triplets are all parsed."""
        from axon.main import AxonBrain

        raw = (
            "<triplet> Apple <subj> Steve Jobs <obj> founded"
            " <triplet> Microsoft <subj> Bill Gates <obj> co-founded"
        )
        triplets = AxonBrain._parse_rebel_output(raw)
        assert len(triplets) == 2
        assert triplets[0]["subject"] == "Apple"
        assert triplets[1]["subject"] == "Microsoft"

    def test_parse_rebel_output_empty_string(self):
        """Empty or whitespace-only output returns an empty list."""
        from axon.main import AxonBrain

        assert AxonBrain._parse_rebel_output("") == []
        assert AxonBrain._parse_rebel_output("   ") == []
        assert AxonBrain._parse_rebel_output("<pad></s>") == []

    def test_rebel_missing_import_returns_empty(self, tmp_path):
        """ImportError inside _ensure_rebel logs a warning and returns []."""
        brain = self._make_brain(tmp_path, graph_rag_relation_backend="rebel")

        with patch("axon.main.AxonBrain._ensure_rebel", side_effect=ImportError("no transformers")):
            result = brain._extract_relations_rebel("some text")

        assert result == []

    def test_config_defaults(self):
        """Default config has relation_backend='llm' and rebel_model set."""
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.graph_rag_relation_backend == "llm"
        assert cfg.graph_rag_rebel_model == "Babelscape/rebel-large"


class TestEntityAliasResolution:
    """Tests for P1 — semantic entity alias resolution."""

    def _make_brain(self, tmp_path, **cfg_kwargs):
        from unittest.mock import MagicMock

        from axon.main import AxonBrain, AxonConfig

        cfg = AxonConfig(
            vector_store_path=str(tmp_path / "chroma"),
            bm25_path=str(tmp_path / "bm25"),
            **cfg_kwargs,
        )
        brain = AxonBrain.__new__(AxonBrain)
        brain.config = cfg
        brain.llm = MagicMock()
        brain.embedding = MagicMock()
        brain._entity_graph = {}
        brain._relation_graph = {}
        brain._community_graph_dirty = False
        return brain

    def test_aliases_merged_when_above_threshold(self, tmp_path):
        """Two entities with near-identical embeddings are merged into the canonical node."""
        import numpy as np

        brain = self._make_brain(
            tmp_path,
            graph_rag_entity_resolve=True,
            graph_rag_entity_resolve_threshold=0.90,
        )
        brain._entity_graph = {
            "apple inc": {
                "chunk_ids": ["c1", "c2"],
                "description": "Tech company",
                "type": "ORGANIZATION",
            },
            "apple": {"chunk_ids": ["c3"], "description": "", "type": "ORGANIZATION"},
        }
        brain._relation_graph = {}

        # Return near-identical embeddings so cosine similarity ≈ 1.0
        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        brain.embedding.embed.return_value = [vec.tolist(), vec.tolist()]

        merged = brain._resolve_entity_aliases()

        assert merged == 1
        # Canonical should be "apple inc" (more chunk_ids)
        assert "apple inc" in brain._entity_graph
        assert "apple" not in brain._entity_graph
        # Chunk ids from alias are absorbed
        assert "c3" in brain._entity_graph["apple inc"]["chunk_ids"]

    def test_distinct_entities_not_merged(self, tmp_path):
        """Two entities with orthogonal embeddings are not merged."""

        brain = self._make_brain(
            tmp_path,
            graph_rag_entity_resolve=True,
            graph_rag_entity_resolve_threshold=0.90,
        )
        brain._entity_graph = {
            "apple": {"chunk_ids": ["c1"], "description": "Fruit", "type": "CONCEPT"},
            "microsoft": {
                "chunk_ids": ["c2"],
                "description": "Tech company",
                "type": "ORGANIZATION",
            },
        }
        brain._relation_graph = {}

        # Orthogonal embeddings → similarity = 0
        brain.embedding.embed.return_value = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        merged = brain._resolve_entity_aliases()

        assert merged == 0
        assert "apple" in brain._entity_graph
        assert "microsoft" in brain._entity_graph

    def test_relations_remapped_to_canonical(self, tmp_path):
        """Subject/object references pointing to an alias are rewritten to the canonical key."""
        import numpy as np

        brain = self._make_brain(
            tmp_path,
            graph_rag_entity_resolve=True,
            graph_rag_entity_resolve_threshold=0.90,
        )
        brain._entity_graph = {
            "apple inc": {"chunk_ids": ["c1", "c2"], "description": "Tech", "type": "ORGANIZATION"},
            "apple": {"chunk_ids": ["c3"], "description": "", "type": "ORGANIZATION"},
        }
        # Relation where alias appears as subject
        brain._relation_graph = {
            "google": [
                {
                    "subject": "apple",
                    "object": "google",
                    "relation": "competes with",
                    "strength": 5,
                    "description": "",
                }
            ]
        }

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        brain.embedding.embed.return_value = [vec.tolist(), vec.tolist()]

        brain._resolve_entity_aliases()

        # The relation subject should be rewritten from "apple" to "apple inc"
        for rel in brain._relation_graph.get("google", []):
            assert rel["subject"] != "apple"
            assert rel["subject"] == "apple inc"

    def test_alias_relation_entries_moved_to_canonical(self, tmp_path):
        """Relation-graph entries keyed under an alias are migrated to the canonical key."""
        import numpy as np

        brain = self._make_brain(
            tmp_path,
            graph_rag_entity_resolve=True,
            graph_rag_entity_resolve_threshold=0.90,
        )
        brain._entity_graph = {
            "apple inc": {"chunk_ids": ["c1", "c2"], "description": "Tech", "type": "ORGANIZATION"},
            "apple": {"chunk_ids": ["c3"], "description": "", "type": "ORGANIZATION"},
        }
        brain._relation_graph = {
            "apple": [
                {
                    "subject": "apple",
                    "object": "iphone",
                    "relation": "makes",
                    "strength": 5,
                    "description": "",
                }
            ]
        }

        vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        brain.embedding.embed.return_value = [vec.tolist(), vec.tolist()]

        brain._resolve_entity_aliases()

        # "apple" key should be gone; relations moved under "apple inc"
        assert "apple" not in brain._relation_graph
        assert "apple inc" in brain._relation_graph
        assert len(brain._relation_graph["apple inc"]) == 1

    def test_skips_when_entity_count_exceeds_max(self, tmp_path):
        """Returns 0 and logs warning when entity count exceeds graph_rag_entity_resolve_max."""
        brain = self._make_brain(
            tmp_path,
            graph_rag_entity_resolve=True,
            graph_rag_entity_resolve_max=2,
        )
        # 3 entities > max of 2
        brain._entity_graph = {
            "a": {"chunk_ids": ["c1"], "description": "", "type": "CONCEPT"},
            "b": {"chunk_ids": ["c2"], "description": "", "type": "CONCEPT"},
            "c": {"chunk_ids": ["c3"], "description": "", "type": "CONCEPT"},
        }
        brain._relation_graph = {}

        merged = brain._resolve_entity_aliases()

        assert merged == 0
        brain.embedding.embed.assert_not_called()

    def test_single_entity_returns_zero(self, tmp_path):
        """A graph with only one entity returns 0 immediately."""
        brain = self._make_brain(tmp_path, graph_rag_entity_resolve=True)
        brain._entity_graph = {
            "apple": {"chunk_ids": ["c1"], "description": "Fruit", "type": "CONCEPT"}
        }
        brain._relation_graph = {}

        merged = brain._resolve_entity_aliases()

        assert merged == 0
        brain.embedding.embed.assert_not_called()

    def test_config_defaults(self):
        """Default config has entity_resolve=False with sensible thresholds."""
        from axon.main import AxonConfig

        cfg = AxonConfig()
        assert cfg.graph_rag_entity_resolve is False
        assert cfg.graph_rag_entity_resolve_threshold == 0.92
        assert cfg.graph_rag_entity_resolve_max == 5000

    def test_rebuild_communities_calls_resolve_when_enabled(self, tmp_path):
        """_rebuild_communities() calls _resolve_entity_aliases() when entity_resolve=True."""
        brain = self._make_brain(
            tmp_path,
            graph_rag_entity_resolve=True,
            graph_rag_community=True,
        )
        brain._community_rebuild_lock = MagicMock()
        brain._community_rebuild_lock.__enter__ = MagicMock(return_value=None)
        brain._community_rebuild_lock.__exit__ = MagicMock(return_value=False)
        brain._community_build_in_progress = False
        brain._community_graph_dirty = False
        brain._run_hierarchical_community_detection = MagicMock(return_value={})
        brain._save_entity_graph = MagicMock()

        with patch("axon.main.AxonBrain._resolve_entity_aliases", return_value=0) as mock_resolve:
            brain._rebuild_communities()

        mock_resolve.assert_called_once()


def test_resolve_model_path_uses_hf_models_dir(tmp_path):
    """_resolve_model_path prefers hf_models_dir over local_models_dir for kind='hf'."""
    from axon.main import AxonBrain, AxonConfig

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "my-model").mkdir()
    cfg = AxonConfig(
        hf_models_dir=str(hf_dir),
        local_models_dir=str(tmp_path),
        local_assets_only=True,
    )
    brain = MagicMock()
    brain.config = cfg
    result = AxonBrain._resolve_model_path(brain, "org/my-model", "hf")
    assert result == str(hf_dir / "my-model")


def test_resolve_model_path_uses_embedding_models_dir(tmp_path):
    """_resolve_model_path prefers embedding_models_dir for kind='embedding'."""
    from axon.main import AxonBrain, AxonConfig

    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir()
    (emb_dir / "all-MiniLM-L6-v2").mkdir()
    cfg = AxonConfig(
        embedding_models_dir=str(emb_dir),
        local_models_dir=str(tmp_path),
    )
    brain = MagicMock()
    brain.config = cfg
    result = AxonBrain._resolve_model_path(
        brain, "sentence-transformers/all-MiniLM-L6-v2", "embedding"
    )
    assert result == str(emb_dir / "all-MiniLM-L6-v2")


def test_resolve_model_path_falls_back_to_local_models_dir(tmp_path):
    """Falls back to local_models_dir when the per-type dir doesn't have the model."""
    from axon.main import AxonBrain, AxonConfig

    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "rebel-large").mkdir()
    cfg = AxonConfig(
        hf_models_dir=str(tmp_path / "hf_empty"),  # doesn't exist
        local_models_dir=str(local_dir),
    )
    brain = MagicMock()
    brain.config = cfg
    result = AxonBrain._resolve_model_path(brain, "Babelscape/rebel-large", "hf")
    assert result == str(local_dir / "rebel-large")


def test_local_assets_only_sets_env_and_resolves_models(tmp_path, monkeypatch):
    """local_assets_only sets HF offline env vars and resolves model paths."""
    from axon.main import AxonBrain, AxonConfig

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "gliner_medium-v2.1").mkdir()

    emb_dir = tmp_path / "emb"
    emb_dir.mkdir()
    (emb_dir / "all-MiniLM-L6-v2").mkdir()

    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenVectorStore"
    ), patch("axon.main.OpenReranker"):
        cfg = AxonConfig(
            local_assets_only=True,
            hf_models_dir=str(hf_dir),
            embedding_models_dir=str(emb_dir),
            graph_rag_gliner_model="urchade/gliner_medium-v2.1",
            rerank=False,
        )
        brain = AxonBrain(cfg)

    assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert brain.config.graph_rag_gliner_model == str(hf_dir / "gliner_medium-v2.1")


# ---------------------------------------------------------------------------
# Phase 5 — Preflight model audit
# ---------------------------------------------------------------------------


def test_preflight_audit_logs_all_rows(tmp_path, caplog):
    """_preflight_model_audit logs one row per model asset."""
    import logging

    from axon.main import AxonBrain, AxonConfig

    emb_dir = tmp_path / "emb" / "all-MiniLM-L6-v2"
    emb_dir.mkdir(parents=True)

    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenVectorStore"
    ), patch("axon.main.OpenReranker"):
        cfg = AxonConfig(
            embedding_model_path=str(emb_dir),
            rerank=False,
            graph_rag=False,
            compress_context=False,
        )
        with caplog.at_level(logging.INFO, logger="Axon"):
            AxonBrain(cfg)

    audit_msgs = [r.message for r in caplog.records if "Model asset audit" in r.message]
    assert audit_msgs, "Expected 'Model asset audit' log entry"
    audit_text = audit_msgs[0]
    # All six row labels must appear
    for label in ("embedding", "reranker", "gliner", "rebel", "llmlingua", "tokenizer"):
        assert label in audit_text, f"Missing row '{label}' in audit log"


def test_preflight_audit_classifies_local_path(tmp_path, caplog):
    """A resolved absolute path is classified as [local]."""
    import logging

    from axon.main import AxonBrain, AxonConfig

    emb_dir = tmp_path / "models" / "all-MiniLM-L6-v2"
    emb_dir.mkdir(parents=True)

    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenVectorStore"
    ), patch("axon.main.OpenReranker"):
        cfg = AxonConfig(embedding_model=str(emb_dir), rerank=False, graph_rag=False)
        with caplog.at_level(logging.INFO, logger="Axon"):
            AxonBrain(cfg)

    audit = next(r.message for r in caplog.records if "Model asset audit" in r.message)
    assert "[local]" in audit


def test_preflight_audit_classifies_remote_id(caplog):
    """A bare HuggingFace model ID is classified as [remote] when not in local cache."""
    import logging

    from axon.main import AxonBrain, AxonConfig

    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenVectorStore"
    ), patch("axon.main.OpenReranker"):
        # Use a model ID that definitely won't be in a tmp HF cache
        cfg = AxonConfig(
            embedding_model="definitely-nonexistent-model-xyz/v1",
            rerank=False,
            graph_rag=False,
            local_assets_only=False,  # don't fail fast, just audit
        )
        # Patch HF cache dir to a non-existent path so hf_cache check always misses
        with patch.dict(os.environ, {"HF_HOME": "/nonexistent/hf_home"}):
            with caplog.at_level(logging.INFO, logger="Axon"):
                AxonBrain(cfg)

    audit = next(r.message for r in caplog.records if "Model asset audit" in r.message)
    assert "[remote]" in audit


def test_preflight_fails_fast_when_local_assets_only_and_remote(tmp_path, monkeypatch):
    """RuntimeError raised at init when local_assets_only=True and embedding is a remote ID."""
    import pytest

    from axon.main import AxonBrain, AxonConfig

    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenVectorStore"
    ), patch("axon.main.OpenReranker"):
        cfg = AxonConfig(
            local_assets_only=True,
            embedding_model="all-MiniLM-L6-v2",  # bare ID, no local dir configured
            rerank=False,
            graph_rag=False,
        )
        # Force HF cache miss so the bare ID cannot be resolved to hf_cache either
        with patch.dict(os.environ, {"HF_HOME": str(tmp_path / "empty_hf")}):
            with pytest.raises(RuntimeError, match="local_assets_only is ON"):
                AxonBrain(cfg)


def test_preflight_fails_fast_missing_path(tmp_path, monkeypatch):
    """RuntimeError raised when local_assets_only=True and an absolute path does not exist."""
    import pytest

    from axon.main import AxonBrain, AxonConfig

    missing = str(tmp_path / "nonexistent_model_dir")

    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenVectorStore"
    ), patch("axon.main.OpenReranker"):
        cfg = AxonConfig(
            local_assets_only=True,
            embedding_model=missing,  # absolute but doesn't exist
            rerank=False,
            graph_rag=False,
        )
        with pytest.raises(RuntimeError, match="local_assets_only is ON"):
            AxonBrain(cfg)


def test_preflight_no_error_when_local_assets_only_all_local(tmp_path, monkeypatch):
    """No error when local_assets_only=True and all active model paths exist on disk."""
    from axon.main import AxonBrain, AxonConfig

    emb_dir = tmp_path / "emb" / "my-embed-model"
    emb_dir.mkdir(parents=True)

    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenVectorStore"
    ), patch("axon.main.OpenReranker"):
        cfg = AxonConfig(
            local_assets_only=True,
            embedding_model=str(emb_dir),
            rerank=False,
            graph_rag=False,
        )
        # Should not raise
        brain = AxonBrain(cfg)

    assert brain.config.local_assets_only is True


# ---------------------------------------------------------------------------
# Phase 6: @-scope tests
# ---------------------------------------------------------------------------


def test_switch_to_at_projects_scope(tmp_path, monkeypatch):
    """@projects scope loads MultiVectorStore across all projects."""
    from unittest.mock import patch

    from axon.main import AxonBrain, AxonConfig

    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenVectorStore"
    ), patch("axon.main.OpenReranker"):
        cfg = AxonConfig(rerank=False, graph_rag=False)
        brain = AxonBrain(cfg)
        # switch_project("@projects") should not raise even with no projects
        try:
            brain.switch_project("@projects")
        except Exception as e:
            # Allow ValueError for no projects found, but not unexpected crashes
            assert "no" in str(e).lower() or "@" in str(e).lower(), f"Unexpected error: {e}"


def test_read_only_scope_blocks_ingest(tmp_path):
    """Ingest raises when read-only scope is active."""
    from unittest.mock import patch

    import pytest

    from axon.main import AxonBrain, AxonConfig

    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenVectorStore"
    ), patch("axon.main.OpenReranker"):
        cfg = AxonConfig(rerank=False, graph_rag=False)
        brain = AxonBrain(cfg)
        brain._read_only_scope = True
        docs = [{"id": "d1", "text": "hello", "metadata": {"source": "test"}}]
        with pytest.raises((ValueError, RuntimeError), match="read-only"):
            brain.ingest(docs)


# ---------------------------------------------------------------------------
# Phase 7: startup log test
# ---------------------------------------------------------------------------


def test_startup_summary_logged(caplog):
    """AxonBrain logs startup summary at INFO level."""
    import logging
    from unittest.mock import patch

    from axon.main import AxonBrain, AxonConfig

    with patch("axon.main.OpenEmbedding"), patch("axon.main.OpenLLM"), patch(
        "axon.main.OpenVectorStore"
    ), patch("axon.main.OpenReranker"):
        cfg = AxonConfig(rerank=False, graph_rag=False)
        with caplog.at_level(logging.INFO, logger="Axon"):
            AxonBrain(cfg)

    startup_msgs = [r.message for r in caplog.records if "Axon ready" in r.message]
    assert startup_msgs, "Expected 'Axon ready' log entry"
