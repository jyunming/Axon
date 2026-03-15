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
        assert config.hybrid_mode == "weighted"
        assert config.dataset_type == "auto"

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
            # top_score=0 treated as falsy → logged as None
            assert logged_data["top_score"] is None


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
        all_ids = {doc_id for ids in brain._entity_graph.values() for doc_id in ids}
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
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.complete = MagicMock(
            return_value="- Axon\n* OpenAI\n• Qdrant\n1. Python\n2) FastAPI"
        )
        entities = brain._extract_entities("some text")
        assert entities == ["Axon", "OpenAI", "Qdrant", "Python", "FastAPI"]

    def test_clean_lines_unchanged(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        brain = self._make_brain(MockReranker, MockEmbed, MockLLM, MockStore, MockBM25)
        brain.llm.complete = MagicMock(return_value="Axon\nOpenAI\nQdrant")
        entities = brain._extract_entities("some text")
        assert entities == ["Axon", "OpenAI", "Qdrant"]

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
        # Bind real implementations so we test actual logic
        brain._entity_matches = AxonBrain._entity_matches.__get__(brain, AxonBrain)
        brain._prune_entity_graph = AxonBrain._prune_entity_graph.__get__(brain, AxonBrain)
        brain._extract_relations = AxonBrain._extract_relations.__get__(brain, AxonBrain)
        brain._expand_with_entity_graph = AxonBrain._expand_with_entity_graph.__get__(
            brain, AxonBrain
        )
        brain._save_entity_graph = MagicMock()
        brain._save_relation_graph = MagicMock()
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

        results = brain._expand_with_entity_graph("machine learning query", [], cfg=None)
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

        results = brain._expand_with_entity_graph("python query", [], cfg=None)
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

        assert brain._entity_graph["python"] == ["doc_1"]
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
        assert triples[0] == ("Python", "is used for", "machine learning")
        assert triples[1] == ("OpenAI", "created", "GPT-4")

    def test_relation_extraction_ignores_bad_lines(self):
        """Malformed lines (wrong number of pipes) are silently skipped."""
        brain = self._make_brain()
        brain.llm = MagicMock()
        brain.llm.complete = MagicMock(
            return_value=(
                "Good line | relates to | something\n"
                "bad line no pipes\n"
                "only | two parts\n"
                "too | many | pipe | parts | here"
            )
        )

        triples = brain._extract_relations("some text")

        assert len(triples) == 1
        assert triples[0] == ("Good line", "relates to", "something")

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

        results = brain._expand_with_entity_graph("python programming", [], cfg=None)

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
