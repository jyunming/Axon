import pytest
import os
import yaml
from unittest.mock import patch, MagicMock


class TestOpenStudioConfig:
    def test_defaults(self):
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig()
        assert config.embedding_provider == "sentence_transformers"
        assert config.llm_provider == "ollama"
        assert config.vector_store == "chroma"
        assert config.hybrid_search is True
        assert config.top_k == 10
        assert config.discussion_fallback is True

    def test_load_from_yaml(self, tmp_path):
        from rag_brain.main import OpenStudioConfig
        cfg = {
            "embedding": {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
            "llm": {"provider": "ollama", "model": "phi3:mini", "temperature": 0.5},
            "vector_store": {"provider": "chroma", "path": str(tmp_path / "chroma")},
            "bm25": {"path": str(tmp_path / "bm25")},
            "rag": {"top_k": 5, "similarity_threshold": 0.6, "hybrid_search": False},
            "chunk": {"size": 500, "overlap": 50},
            "rerank": {"enabled": False, "provider": "llm", "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"},
            "query_transformations": {"multi_query": True, "hyde": False, "discussion_fallback": True},
            "web_search": {"enabled": True, "brave_api_key": "test_key"}
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        config = OpenStudioConfig.load(str(cfg_path))
        assert config.embedding_provider == "fastembed"
        assert config.llm_model == "phi3:mini"
        assert config.top_k == 5
        assert config.multi_query is True
        assert config.truth_grounding is True
        assert config.brave_api_key == "test_key"


@patch("rag_brain.retrievers.BM25Retriever")
@patch("rag_brain.main.OpenVectorStore")
@patch("rag_brain.main.OpenLLM")
@patch("rag_brain.main.OpenEmbedding")
@patch("rag_brain.main.OpenReranker")
class TestOpenStudioBrain:
    def test_query_flow(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig
        config = OpenStudioConfig(hybrid_search=False, rerank=False, similarity_threshold=0.0)
        brain = OpenStudioBrain(config)

        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(return_value=[
            {"id": "doc1", "text": "hello world", "score": 0.9}
        ])
        brain.llm.complete = MagicMock(return_value="Test answer")

        result = brain.query("test question")
        assert result == "Test answer"

    def test_discussion_fallback(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig
        config = OpenStudioConfig(hybrid_search=False, rerank=False, discussion_fallback=True)
        brain = OpenStudioBrain(config)

        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(return_value=[])
        brain.llm.complete = MagicMock(return_value="Fallback answer")

        result = brain.query("fallback question")
        assert result == "Fallback answer"
        # The plain query should be sent as the user message (not a 3rd-person wrapper)
        args, kwargs = brain.llm.complete.call_args
        assert args[0] == "fallback question"
        assert kwargs['chat_history'] is None

    def test_multi_turn_history_passed_as_plain_query(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        """User message sent to LLM must be the plain query so chat_history stays consistent."""
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig
        config = OpenStudioConfig(hybrid_search=False, rerank=False)
        brain = OpenStudioBrain(config)

        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(return_value=[
            {"id": "d1", "text": "some context", "score": 0.9, "metadata": {}}
        ])
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
        assert kwargs['chat_history'] == history

    def test_ingest_flow(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig
        config = OpenStudioConfig(chunk_size=1000)
        brain = OpenStudioBrain(config)
        brain.splitter = None

        docs = [{"id": "d1", "text": "t1"}, {"id": "d2", "text": "t2"}]
        brain.embedding.embed = MagicMock(return_value=[[0.1], [0.1]])
        brain.vector_store.add = MagicMock()

        brain.ingest(docs)
        assert brain.vector_store.add.called

    def test_web_search_integration(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig
        config = OpenStudioConfig(truth_grounding=True, brave_api_key="key")
        brain = OpenStudioBrain(config)

        brain.embedding.embed_query = MagicMock(return_value=[0.1])
        brain.vector_store.search = MagicMock(return_value=[])
        brain._execute_web_search = MagicMock(return_value=[{"id": "w1", "text": "web", "score": 1.0, "is_web": True}])
        
        retrieval = brain._execute_retrieval("query")
        assert retrieval['web_count'] == 1
        assert retrieval['results'][0]['is_web'] is True


class TestOpenReranker:
    def test_llm_reranker(self):
        from rag_brain.main import OpenReranker, OpenStudioConfig
        config = OpenStudioConfig(rerank=True, reranker_provider="llm")
        reranker = OpenReranker(config)
        reranker.llm = MagicMock()
        reranker.llm.complete = MagicMock(side_effect=["10", "5"])

        docs = [{"id": "d1", "text": "t1"}, {"id": "d2", "text": "t2"}]
        results = reranker.rerank("q", docs)
        assert results[0]['id'] == "d1"
        assert results[0]['rerank_score'] == 10.0


class TestOpenLLM:
    @patch("ollama.Client")
    def test_ollama_num_ctx(self, MockOllama):
        from rag_brain.main import OpenLLM, OpenStudioConfig
        config = OpenStudioConfig(llm_provider="ollama")
        llm = OpenLLM(config)
        
        mock_client = MockOllama.return_value
        mock_client.chat.return_value = {"message": {"content": "resp"}}
        
        llm.complete("q")
        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs['options']['num_ctx'] == 8192

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_gemma_handling(self, MockGenaiConfigure, MockGenerativeModel):
        from rag_brain.main import OpenLLM, OpenStudioConfig
        # Test Gemma
        config = OpenStudioConfig(llm_provider="gemini", llm_model="gemma-3-27b-it")
        llm = OpenLLM(config)
        
        mock_model = MockGenerativeModel.return_value
        mock_model.generate_content.return_value = MagicMock(text="gemma answer")
        
        llm.complete("hello", system_prompt="be helpful")
        
        # Verify system_instruction was NOT passed to constructor
        MockGenerativeModel.assert_called_with(model_name="gemma-3-27b-it")
        
        # Verify prompt was prepended
        gen_args = mock_model.generate_content.call_args[0][0]
        assert "be helpful\n\nhello" in gen_args[-1]['parts'][0]

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_pro_handling(self, MockGenaiConfigure, MockGenerativeModel):
        from rag_brain.main import OpenLLM, OpenStudioConfig
        # Test Pro (supports system instructions)
        config = OpenStudioConfig(llm_provider="gemini", llm_model="gemini-1.5-pro")
        llm = OpenLLM(config)
        
        llm.complete("hello", system_prompt="be helpful")
        MockGenerativeModel.assert_called_with(model_name="gemini-1.5-pro", system_instruction="be helpful")

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_flash_handling(self, MockGenaiConfigure, MockGenerativeModel):
        from rag_brain.main import OpenLLM, OpenStudioConfig
        # Test Flash (also supports system instructions)
        config = OpenStudioConfig(llm_provider="gemini", llm_model="gemini-1.5-flash")
        llm = OpenLLM(config)
        
        llm.complete("hello", system_prompt="be helpful")
        MockGenerativeModel.assert_called_with(model_name="gemini-1.5-flash", system_instruction="be helpful")

    @patch("httpx.Client")
    def test_ollama_cloud_handling(self, MockHttpxClient):
        from rag_brain.main import OpenLLM, OpenStudioConfig
        config = OpenStudioConfig(llm_provider="ollama_cloud", ollama_cloud_key="test_key", llm_model="gemma")
        llm = OpenLLM(config)
        
        mock_client = MockHttpxClient.return_value.__enter__.return_value
        mock_client.post.return_value = MagicMock(json=lambda: {"response": "cloud resp"}, status_code=200)
        
        result = llm.complete("hello", system_prompt="be helpful")
        assert result == "cloud resp"
        assert mock_client.post.called

    @patch("openai.resources.chat.Completions.create")
    @patch("openai.OpenAI")
    def test_openai_handling(self, MockOpenAI, MockCreate):
        from rag_brain.main import OpenLLM, OpenStudioConfig
        config = OpenStudioConfig(llm_provider="openai", api_key="sk-test", llm_model="gpt-4o")
        llm = OpenLLM(config)

        # Setup mock client
        mock_client = MockOpenAI.return_value
        mock_client.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="openai resp"))])

        result = llm.complete("hello", system_prompt="be helpful")
        assert result == "openai resp"
        assert mock_client.chat.completions.create.called

    @patch("openai.OpenAI")
    def test_vllm_complete(self, MockOpenAI):
        from rag_brain.main import OpenLLM, OpenStudioConfig
        config = OpenStudioConfig(
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

    def test_vllm_default_base_url(self):
        from rag_brain.main import OpenStudioConfig
        config = OpenStudioConfig()
        assert config.vllm_base_url == "http://localhost:8000/v1"

    def test_vllm_base_url_from_yaml(self, tmp_path):
        from rag_brain.main import OpenStudioConfig
        cfg = {"llm": {"provider": "vllm", "model": "meta-llama/Llama-3.1-8B-Instruct",
                       "vllm_base_url": "http://192.168.1.10:8000/v1"}}
        cfg_path = tmp_path / "config.yaml"
        import yaml
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)
        config = OpenStudioConfig.load(str(cfg_path))
        assert config.llm_provider == "vllm"
        assert config.vllm_base_url == "http://192.168.1.10:8000/v1"


# ---------------------------------------------------------------------------
# New tests: HyDE, multi-query, load_directory, metrics
# ---------------------------------------------------------------------------

@patch("rag_brain.retrievers.BM25Retriever")
@patch("rag_brain.main.OpenVectorStore")
@patch("rag_brain.main.OpenLLM")
@patch("rag_brain.main.OpenEmbedding")
@patch("rag_brain.main.OpenReranker")
class TestQueryTransformations:
    def test_hyde_document_calls_llm(self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig

        config = OpenStudioConfig(hyde=False, hybrid_search=False, rerank=False)
        brain = OpenStudioBrain(config)
        brain.llm.complete = MagicMock(return_value="Hypothetical passage about X.")

        result = brain._get_hyde_document("What is X?")

        assert result == "Hypothetical passage about X."
        brain.llm.complete.assert_called_once()
        call_prompt = brain.llm.complete.call_args[0][0]
        assert "What is X?" in call_prompt

    def test_multi_queries_returns_original_plus_alternatives(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig

        config = OpenStudioConfig(multi_query=True, hybrid_search=False, rerank=False)
        brain = OpenStudioBrain(config)
        brain.llm.complete = MagicMock(
            return_value="Alt query one\nAlt query two\nAlt query three"
        )

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
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig

        config = OpenStudioConfig(multi_query=True, hybrid_search=False, rerank=False)
        brain = OpenStudioBrain(config)
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
        from unittest.mock import AsyncMock, patch
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig

        config = OpenStudioConfig(hybrid_search=False, rerank=False)
        brain = OpenStudioBrain(config)
        brain.ingest = MagicMock()

        docs = [{"id": "f1.txt", "text": "hello", "metadata": {}}]

        with patch("rag_brain.loaders.DirectoryLoader") as MockLoader:
            mock_loader_instance = MockLoader.return_value
            mock_loader_instance.aload = AsyncMock(return_value=docs)

            asyncio.run(brain.load_directory(str(tmp_path)))

        brain.ingest.assert_called_once_with(docs)

    def test_load_directory_skips_ingest_when_empty(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25, tmp_path
    ):
        import asyncio
        from unittest.mock import AsyncMock, patch
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig

        config = OpenStudioConfig(hybrid_search=False, rerank=False)
        brain = OpenStudioBrain(config)
        brain.ingest = MagicMock()

        with patch("rag_brain.loaders.DirectoryLoader") as MockLoader:
            mock_loader_instance = MockLoader.return_value
            mock_loader_instance.aload = AsyncMock(return_value=[])

            asyncio.run(brain.load_directory(str(tmp_path)))

        brain.ingest.assert_not_called()


@patch("rag_brain.retrievers.BM25Retriever")
@patch("rag_brain.main.OpenVectorStore")
@patch("rag_brain.main.OpenLLM")
@patch("rag_brain.main.OpenEmbedding")
@patch("rag_brain.main.OpenReranker")
class TestLogQueryMetrics:
    def test_metrics_logged_without_exception(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        import logging
        from unittest.mock import patch as _patch
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig

        config = OpenStudioConfig(hybrid_search=False, rerank=False)
        brain = OpenStudioBrain(config)

        with _patch("rag_brain.main.logger") as mock_logger:
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
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig

        config = OpenStudioConfig(hybrid_search=False, rerank=False)
        brain = OpenStudioBrain(config)

        with _patch("rag_brain.main.logger") as mock_logger:
            brain._log_query_metrics(
                query="q", vector_count=0, bm25_count=0,
                filtered_count=0, final_count=0, top_score=0.0, latency_ms=10.0
            )
            logged_data = mock_logger.info.call_args[0][0]
            # top_score=0 treated as falsy → logged as None
            assert logged_data["top_score"] is None


# ---------------------------------------------------------------------------
# list_documents
# ---------------------------------------------------------------------------

@patch("rag_brain.retrievers.BM25Retriever")
@patch("rag_brain.main.OpenVectorStore")
@patch("rag_brain.main.OpenLLM")
@patch("rag_brain.main.OpenEmbedding")
@patch("rag_brain.main.OpenReranker")
class TestListDocuments:
    def test_list_documents_delegates_to_vector_store(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig

        config = OpenStudioConfig(hybrid_search=False, rerank=False)
        brain = OpenStudioBrain(config)
        brain.vector_store.list_documents = MagicMock(return_value=[
            {"source": "a.txt", "chunks": 3, "doc_ids": ["1", "2", "3"]},
            {"source": "b.pdf", "chunks": 7, "doc_ids": [str(i) for i in range(7)]},
        ])

        result = brain.list_documents()

        brain.vector_store.list_documents.assert_called_once()
        assert len(result) == 2
        assert result[0]["source"] == "a.txt"
        assert result[0]["chunks"] == 3
        assert result[1]["source"] == "b.pdf"

    def test_list_documents_empty_kb(
        self, MockReranker, MockEmbed, MockLLM, MockStore, MockBM25
    ):
        from rag_brain.main import OpenStudioBrain, OpenStudioConfig

        config = OpenStudioConfig(hybrid_search=False, rerank=False)
        brain = OpenStudioBrain(config)
        brain.vector_store.list_documents = MagicMock(return_value=[])

        result = brain.list_documents()
        assert result == []


class TestOpenVectorStoreListDocuments:
    def test_chroma_groups_by_source(self):
        from rag_brain.main import OpenVectorStore, OpenStudioConfig
        from unittest.mock import MagicMock, patch

        config = OpenStudioConfig(vector_store="chroma")
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
        from rag_brain.main import OpenVectorStore, OpenStudioConfig
        from unittest.mock import MagicMock, patch

        config = OpenStudioConfig(vector_store="chroma")
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
                _p = _PThtml('<ansigreen><b>You</b></ansigreen>: ') if not prompt else prompt
                return _pt_session.prompt(_p)
            return input(prompt if prompt else '\033[1;32mYou\033[0m: ')

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
        from rag_brain.main import _infer_provider
        assert _infer_provider("gemini-2.5-flash-lite") == "gemini"
        assert _infer_provider("gemini-1.5-pro") == "gemini"

    def test_openai_gpt_prefix(self):
        from rag_brain.main import _infer_provider
        assert _infer_provider("gpt-4o") == "openai"
        assert _infer_provider("gpt-4o-mini") == "openai"
        assert _infer_provider("o1-mini") == "openai"
        assert _infer_provider("o3-mini") == "openai"

    def test_ollama_default(self):
        from rag_brain.main import _infer_provider
        assert _infer_provider("mistral-nemo") == "ollama"
        assert _infer_provider("llama3.1") == "ollama"
        assert _infer_provider("gemma") == "ollama"
        assert _infer_provider("phi3") == "ollama"
        # Ollama models with name:tag format must NOT be misclassified as openai
        assert _infer_provider("gpt-oss:120b-cloud") == "ollama"
        assert _infer_provider("o1-tuned:latest") == "ollama"

    def test_vllm_path_style_falls_through_to_ollama(self):
        # vLLM model names look like HuggingFace paths; no auto-detection — falls to ollama
        from rag_brain.main import _infer_provider
        assert _infer_provider("meta-llama/Llama-3.1-8B-Instruct") == "ollama"
        assert _infer_provider("mistralai/Mistral-7B-Instruct-v0.3") == "ollama"

    def test_case_insensitive(self):
        from rag_brain.main import _infer_provider
        assert _infer_provider("GEMINI-2.0-FLASH") == "gemini"
        assert _infer_provider("GPT-4O") == "openai"

