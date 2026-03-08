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
        assert config.discussion_fallback is False

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
        # Check that it was called with correctly formatted fallback prompt
        args, kwargs = brain.llm.complete.call_args
        assert "found no relevant documents" in args[0]
        assert kwargs['chat_history'] is None

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
