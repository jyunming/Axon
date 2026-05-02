"""Tests for axon.integrations.langchain and axon.integrations.llama_index.

Library-required tests use ``pytest.importorskip`` so they auto-skip in
environments without the extra installed. The "not-installed" code path is
covered by patching the module's availability flag.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


def _fake_brain() -> SimpleNamespace:
    """Build a fake AxonBrain with a stubbed search_raw().

    ``search_raw`` returns ``(results, diagnostics, trace)``; the diagnostics
    and trace objects don't matter for the adapter — only ``results`` is
    consumed.
    """
    brain = MagicMock()
    sample = [
        {
            "id": "chunk-1",
            "text": "Axon is a local-first RAG.",
            "score": 0.91,
            "metadata": {"source": "README.md"},
        },
        {
            "id": "web-1",
            "text": "External web hit",
            "score": 0.42,
            "metadata": {"title": "Web result", "url": "https://example.com"},
            "is_web": True,
        },
    ]
    brain.search_raw = MagicMock(return_value=(sample, MagicMock(), MagicMock()))
    return brain


# ---------------------------------------------------------------------------
# LangChain adapter
# ---------------------------------------------------------------------------


class TestLangChainAdapter:
    def test_install_hint_raised_when_extra_missing(self):
        """When langchain_core is not installed, AxonRetriever raises a clear
        ImportError pointing at the [langchain] extra."""
        from axon.integrations import langchain as mod

        original = mod._LANGCHAIN_AVAILABLE
        try:
            mod._LANGCHAIN_AVAILABLE = False
            with pytest.raises(ImportError, match=r"axon-rag\[langchain\]"):
                mod._check_available()
        finally:
            mod._LANGCHAIN_AVAILABLE = original

    def test_retriever_returns_langchain_documents(self):
        pytest.importorskip("langchain_core")
        from langchain_core.documents import Document

        from axon.integrations.langchain import AxonRetriever

        brain = _fake_brain()
        retriever = AxonRetriever(brain=brain, top_k=5)
        docs = retriever.invoke("hello")
        assert len(docs) == 2
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].page_content == "Axon is a local-first RAG."
        assert docs[0].metadata["source"] == "README.md"
        assert docs[0].metadata["score"] == pytest.approx(0.91)
        assert docs[0].metadata["id"] == "chunk-1"
        # Web row gets a source_kind tag
        assert docs[1].metadata["source_kind"] == "web"

    def test_top_k_lands_in_overrides(self):
        pytest.importorskip("langchain_core")
        from axon.integrations.langchain import AxonRetriever

        brain = _fake_brain()
        retriever = AxonRetriever(brain=brain, top_k=7)
        retriever.invoke("hello")
        # top_k forwards as overrides["top_k"]
        kwargs = brain.search_raw.call_args.kwargs
        assert kwargs["overrides"]["top_k"] == 7

    def test_filters_pass_through(self):
        pytest.importorskip("langchain_core")
        from axon.integrations.langchain import AxonRetriever

        brain = _fake_brain()
        retriever = AxonRetriever(brain=brain, filters={"source_class": "code"})
        retriever.invoke("hello")
        kwargs = brain.search_raw.call_args.kwargs
        assert kwargs["filters"] == {"source_class": "code"}

    def test_with_overrides_does_not_mutate_original(self):
        pytest.importorskip("langchain_core")
        from axon.integrations.langchain import AxonRetriever

        brain = _fake_brain()
        base = AxonRetriever(brain=brain, overrides={"hybrid": True})
        derived = base.with_overrides({"rerank": True})
        # Base retains only hybrid; derived has both
        assert base.overrides == {"hybrid": True}
        assert derived.overrides == {"hybrid": True, "rerank": True}

    def test_rag_flag_overrides_pass_through(self):
        pytest.importorskip("langchain_core")
        from axon.integrations.langchain import AxonRetriever

        brain = _fake_brain()
        retriever = AxonRetriever(
            brain=brain, overrides={"hyde": True, "rerank": True, "graph_rag": False}
        )
        retriever.invoke("hello")
        ov = brain.search_raw.call_args.kwargs["overrides"]
        assert ov["hyde"] is True
        assert ov["rerank"] is True
        assert ov["graph_rag"] is False


# ---------------------------------------------------------------------------
# LlamaIndex adapter
# ---------------------------------------------------------------------------


class TestLlamaIndexAdapter:
    def test_install_hint_raised_when_extra_missing(self):
        from axon.integrations import llama_index as mod

        original = mod._LLAMA_INDEX_AVAILABLE
        try:
            mod._LLAMA_INDEX_AVAILABLE = False
            with pytest.raises(ImportError, match=r"axon-rag\[llama-index\]"):
                mod._check_available()
        finally:
            mod._LLAMA_INDEX_AVAILABLE = original

    def test_retriever_returns_node_with_score(self):
        pytest.importorskip("llama_index.core")
        from llama_index.core.schema import NodeWithScore

        from axon.integrations.llama_index import AxonLlamaRetriever

        brain = _fake_brain()
        retriever = AxonLlamaRetriever(brain=brain, top_k=5)
        nodes = retriever.retrieve("hello")
        assert len(nodes) == 2
        assert all(isinstance(n, NodeWithScore) for n in nodes)
        assert nodes[0].node.text == "Axon is a local-first RAG."
        assert nodes[0].score == pytest.approx(0.91)
        assert nodes[0].node.metadata["source"] == "README.md"

    def test_filters_and_overrides_pass_through(self):
        pytest.importorskip("llama_index.core")
        from axon.integrations.llama_index import AxonLlamaRetriever

        brain = _fake_brain()
        retriever = AxonLlamaRetriever(
            brain=brain,
            top_k=3,
            filters={"is_code": True},
            overrides={"hybrid": True},
        )
        retriever.retrieve("hello")
        kwargs = brain.search_raw.call_args.kwargs
        assert kwargs["filters"] == {"is_code": True}
        assert kwargs["overrides"]["top_k"] == 3
        assert kwargs["overrides"]["hybrid"] is True

    def test_handles_score_none(self):
        pytest.importorskip("llama_index.core")
        from axon.integrations.llama_index import AxonLlamaRetriever

        brain = MagicMock()
        brain.search_raw.return_value = (
            [{"id": "x", "text": "abc", "metadata": {}}],  # no score key
            MagicMock(),
            MagicMock(),
        )
        retriever = AxonLlamaRetriever(brain=brain)
        nodes = retriever.retrieve("hi")
        assert nodes[0].score is None
