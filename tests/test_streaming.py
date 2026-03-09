"""
tests/test_streaming.py

Tests for OpenLLM.stream() and OpenStudioBrain.query_stream().
"""
import pytest
from unittest.mock import patch, MagicMock, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_brain(config):
    """Construct an OpenStudioBrain with all heavy dependencies mocked."""
    with (
        patch("rag_brain.main.OpenEmbedding"),
        patch("rag_brain.main.OpenLLM"),
        patch("rag_brain.main.OpenVectorStore"),
        patch("rag_brain.main.OpenReranker"),
        patch("rag_brain.retrievers.BM25Retriever"),
    ):
        from rag_brain.main import OpenStudioBrain
        return OpenStudioBrain(config)


# ---------------------------------------------------------------------------
# OpenLLM.stream() — all 4 providers
# ---------------------------------------------------------------------------

class TestOpenLLMStream:
    @patch("ollama.Client")
    def test_ollama_stream_yields_strings(self, MockOllama):
        from rag_brain.main import OpenLLM, OpenStudioConfig

        config = OpenStudioConfig(llm_provider="ollama")
        llm = OpenLLM(config)

        chunks = [{"message": {"content": t}} for t in ["Hello", " world", "!"]]
        MockOllama.return_value.chat.return_value = iter(chunks)

        result = list(llm.stream("hi"))
        assert result == ["Hello", " world", "!"]
        assert all(isinstance(c, str) for c in result)

    @patch("google.generativeai.GenerativeModel")
    @patch("google.generativeai.configure")
    def test_gemini_stream_yields_strings(self, _cfg, MockModel):
        from rag_brain.main import OpenLLM, OpenStudioConfig

        config = OpenStudioConfig(llm_provider="gemini", llm_model="gemini-2.0-flash")
        llm = OpenLLM(config)

        chunk_mocks = [MagicMock(text=t) for t in ["The", " answer", " is", " 42"]]
        MockModel.return_value.generate_content.return_value = iter(chunk_mocks)

        result = list(llm.stream("question"))
        assert result == ["The", " answer", " is", " 42"]
        assert all(isinstance(c, str) for c in result)

    @patch("httpx.Client")
    def test_ollama_cloud_stream_yields_strings(self, MockHttpx):
        import json as _json
        from rag_brain.main import OpenLLM, OpenStudioConfig

        config = OpenStudioConfig(llm_provider="ollama_cloud", ollama_cloud_key="k", llm_model="m")
        llm = OpenLLM(config)

        lines = [_json.dumps({"response": t}) for t in ["token1", " token2"]]
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = iter(lines)
        mock_resp.raise_for_status = MagicMock()

        ctx_manager = MagicMock()
        ctx_manager.__enter__ = MagicMock(return_value=mock_resp)
        ctx_manager.__exit__ = MagicMock(return_value=False)

        mock_client = MockHttpx.return_value.__enter__.return_value
        mock_client.stream.return_value = ctx_manager

        result = list(llm.stream("hi"))
        assert result == ["token1", " token2"]

    @patch("openai.OpenAI")
    def test_openai_stream_yields_strings(self, MockOpenAI):
        from rag_brain.main import OpenLLM, OpenStudioConfig

        config = OpenStudioConfig(llm_provider="openai", api_key="sk-test", llm_model="gpt-4o")
        llm = OpenLLM(config)

        def _make_chunk(text):
            c = MagicMock()
            c.choices[0].delta.content = text
            return c

        stream_chunks = [_make_chunk(t) for t in ["A", "B", "C"]]
        MockOpenAI.return_value.chat.completions.create.return_value = iter(stream_chunks)

        result = list(llm.stream("hello"))
        assert result == ["A", "B", "C"]

    @patch("openai.OpenAI")
    def test_vllm_stream_yields_strings(self, MockOpenAI):
        from rag_brain.main import OpenLLM, OpenStudioConfig

        config = OpenStudioConfig(
            llm_provider="vllm",
            llm_model="meta-llama/Llama-3.1-8B-Instruct",
            vllm_base_url="http://localhost:8000/v1",
        )
        llm = OpenLLM(config)

        def _make_chunk(text):
            c = MagicMock()
            c.choices[0].delta.content = text
            return c

        stream_chunks = [_make_chunk(t) for t in ["Hello", " from", " vLLM"]]
        MockOpenAI.return_value.chat.completions.create.return_value = iter(stream_chunks)

        result = list(llm.stream("hi"))
        assert result == ["Hello", " from", " vLLM"]
        # Verify base_url was passed to OpenAI constructor
        assert MockOpenAI.call_args[1].get("base_url") == "http://localhost:8000/v1"


# ---------------------------------------------------------------------------
# OpenStudioBrain.query_stream()
# ---------------------------------------------------------------------------

class TestQueryStream:
    def _make_brain_with_mocks(self):
        from rag_brain.main import OpenStudioConfig

        config = OpenStudioConfig(hybrid_search=False, rerank=False, similarity_threshold=0.0)
        with (
            patch("rag_brain.main.OpenEmbedding"),
            patch("rag_brain.main.OpenLLM"),
            patch("rag_brain.main.OpenVectorStore"),
            patch("rag_brain.main.OpenReranker"),
            patch("rag_brain.retrievers.BM25Retriever"),
        ):
            from rag_brain.main import OpenStudioBrain
            brain = OpenStudioBrain(config)
        return brain

    def test_yields_sources_then_tokens(self):
        brain = self._make_brain_with_mocks()

        docs = [{"id": "d1", "text": "hello", "score": 0.9, "metadata": {}}]
        brain._execute_retrieval = MagicMock(return_value={
            "results": docs,
            "vector_count": 1,
            "bm25_count": 0,
            "web_count": 0,
            "filtered_count": 1,
            "transforms": {},
        })
        brain.llm.stream = MagicMock(return_value=iter(["tok1", " tok2"]))

        chunks = list(brain.query_stream("my question"))

        source_chunk = next(c for c in chunks if isinstance(c, dict) and c.get("type") == "sources")
        assert source_chunk["sources"] == docs

        token_chunks = [c for c in chunks if isinstance(c, str)]
        assert "".join(token_chunks) == "tok1 tok2"

    def test_no_docs_no_fallback_yields_message(self):
        brain = self._make_brain_with_mocks()
        brain.config.discussion_fallback = False

        brain._execute_retrieval = MagicMock(return_value={
            "results": [],
            "vector_count": 0,
            "bm25_count": 0,
            "web_count": 0,
            "filtered_count": 0,
            "transforms": {},
        })

        chunks = list(brain.query_stream("unknown"))
        text = "".join(c for c in chunks if isinstance(c, str))
        assert len(text) > 0  # some fallback message

    def test_no_docs_with_fallback_streams_llm(self):
        brain = self._make_brain_with_mocks()
        brain.config.discussion_fallback = True

        brain._execute_retrieval = MagicMock(return_value={
            "results": [],
            "vector_count": 0,
            "bm25_count": 0,
            "web_count": 0,
            "filtered_count": 0,
            "transforms": {},
        })
        brain.llm.stream = MagicMock(return_value=iter(["fallback answer"]))

        chunks = list(brain.query_stream("orphan query"))
        assert "fallback answer" in "".join(c for c in chunks if isinstance(c, str))
        brain.llm.stream.assert_called_once()

    def test_completes_without_exception(self):
        brain = self._make_brain_with_mocks()

        brain._execute_retrieval = MagicMock(return_value={
            "results": [{"id": "x", "text": "ctx", "score": 1.0, "metadata": {}}],
            "vector_count": 1,
            "bm25_count": 0,
            "web_count": 0,
            "filtered_count": 1,
            "transforms": {},
        })
        brain.llm.stream = MagicMock(return_value=iter(["done"]))

        # Should not raise
        result = list(brain.query_stream("fine query"))
        assert any(isinstance(c, str) for c in result)
