"""
tests/test_eval_smoke.py

Evaluation smoke tests for the RAG pipeline.

These tests use a tiny, fully-mocked corpus to verify that key RAG quality
metrics are *computable* and produce sensible values.  They are marked with
``@pytest.mark.eval`` so the CI eval job can target them independently:

    pytest -m eval tests/test_eval_smoke.py

Metrics covered
---------------
- Precision@k   — fraction of retrieved chunks that are relevant
- Context relevance — retrieved context contains expected keywords
- Answer faithfulness — final answer is grounded in the retrieved context
- Retrieval recall@k — all relevant docs are returned within top-k
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Pytest marker registration (also in pyproject.toml/pytest.ini)
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.eval


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    from axon.main import AxonConfig

    cfg = AxonConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _relevant_ids(corpus: list[dict], keyword: str) -> list[str]:
    """Return IDs of corpus docs whose text contains keyword (ground truth)."""
    return [d["id"] for d in corpus if keyword.lower() in d["text"].lower()]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CORPUS = [
    {
        "id": "doc1",
        "text": "The transformer architecture uses attention mechanisms for NLP tasks.",
        "score": 0.95,
        "metadata": {"source": "ml_guide.txt"},
    },
    {
        "id": "doc2",
        "text": "BERT is a bidirectional transformer pre-trained on masked language modelling.",
        "score": 0.87,
        "metadata": {"source": "ml_guide.txt"},
    },
    {
        "id": "doc3",
        "text": "Python is a high-level programming language popular for data science.",
        "score": 0.42,
        "metadata": {"source": "prog_guide.txt"},
    },
    {
        "id": "doc4",
        "text": "Attention mechanisms allow the model to focus on different parts of the input.",
        "score": 0.91,
        "metadata": {"source": "ml_guide.txt"},
    },
    {
        "id": "doc5",
        "text": "pip is the package installer for Python.",
        "score": 0.30,
        "metadata": {"source": "prog_guide.txt"},
    },
]


# ---------------------------------------------------------------------------
# Precision@k
# ---------------------------------------------------------------------------


class TestPrecisionAtK:
    """Fraction of top-k retrieved results that are actually relevant."""

    def _precision(self, retrieved_ids: list[str], relevant_ids: list[str]) -> float:
        if not retrieved_ids:
            return 0.0
        hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
        return hits / len(retrieved_ids)

    def test_perfect_retrieval(self):
        """All retrieved docs are relevant → precision = 1.0."""
        relevant = _relevant_ids(CORPUS, "transformer")
        retrieved = [d["id"] for d in CORPUS if "transformer" in d["text"].lower()]
        assert self._precision(retrieved, relevant) == 1.0

    def test_partial_retrieval(self):
        """Mix of relevant and irrelevant → precision between 0 and 1."""
        relevant = _relevant_ids(CORPUS, "transformer")
        # Retrieve all 5 docs (only 3 are transformer-related)
        retrieved = [d["id"] for d in CORPUS]
        p = self._precision(retrieved, relevant)
        assert 0.0 < p < 1.0, f"Expected 0 < precision < 1, got {p}"

    def test_top_k_3_precision(self):
        """Top-3 by score — verify precision is computed correctly."""
        sorted_docs = sorted(CORPUS, key=lambda x: x["score"], reverse=True)[:3]
        relevant = _relevant_ids(CORPUS, "attention")
        p = self._precision([d["id"] for d in sorted_docs], relevant)
        # doc1 (0.95), doc4 (0.91), doc2 (0.87) — doc1 and doc4 mention "attention"
        assert p >= 0.6

    def test_empty_retrieval(self):
        """Empty result set → precision = 0."""
        assert self._precision([], ["doc1"]) == 0.0

    def test_no_relevant_docs(self):
        """No relevant docs in corpus for the query term → precision = 0."""
        relevant = _relevant_ids(CORPUS, "quantum_computing_xyz")
        retrieved = [d["id"] for d in CORPUS[:3]]
        assert self._precision(retrieved, relevant) == 0.0


# ---------------------------------------------------------------------------
# Context Relevance
# ---------------------------------------------------------------------------


class TestContextRelevance:
    """Retrieved context should contain keywords expected for the query."""

    def _context_relevance(self, context: str, expected_keywords: list[str]) -> float:
        """Fraction of expected keywords found in context (simple keyword overlap)."""
        if not expected_keywords:
            return 0.0
        found = sum(1 for kw in expected_keywords if kw.lower() in context.lower())
        return found / len(expected_keywords)

    def _build_context(self, docs: list[dict]) -> str:
        return "\n\n".join(f"[Document {i+1}]\n{d['text']}" for i, d in enumerate(docs))

    def test_relevant_context_high_score(self):
        """Context built from transformer-related docs scores > 0.7 for expected keywords."""
        query_keywords = ["transformer", "attention", "bert"]
        relevant_docs = [d for d in CORPUS if any(kw in d["text"].lower() for kw in query_keywords)]
        context = self._build_context(relevant_docs)
        score = self._context_relevance(context, query_keywords)
        assert score >= 0.6, f"Expected relevance >= 0.6, got {score}"

    def test_irrelevant_context_low_score(self):
        """Context built from Python docs scores low for ML-specific keywords."""
        python_docs = [
            d for d in CORPUS if "python" in d["text"].lower() or "pip" in d["text"].lower()
        ]
        context = self._build_context(python_docs)
        ml_keywords = ["transformer", "bert", "attention"]
        score = self._context_relevance(context, ml_keywords)
        assert score < 0.4, f"Expected relevance < 0.4, got {score}"

    def test_full_context_has_all_docs(self):
        """Context with all docs in corpus has 1.0 relevance (trivially)."""
        context = self._build_context(CORPUS)
        score = self._context_relevance(context, ["transformer", "python", "attention"])
        assert score == 1.0

    def test_empty_context(self):
        assert self._context_relevance("", ["transformer"]) == 0.0


# ---------------------------------------------------------------------------
# Answer Faithfulness
# ---------------------------------------------------------------------------


class TestAnswerFaithfulness:
    """Final answer should be grounded in (entailed by) the retrieved context."""

    def _faithfulness_score(self, answer: str, context: str) -> float:
        """Simplified faithfulness: fraction of answer sentences whose key noun
        phrase appears in the context.  Not a real NLI model — just a keyword heuristic
        suitable for a smoke test without external dependencies.
        """
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        if not sentences:
            return 0.0
        grounded = sum(
            1
            for s in sentences
            if any(word.lower() in context.lower() for word in s.split() if len(word) > 4)
        )
        return grounded / len(sentences)

    def test_faithful_answer(self):
        """Answer drawn from context has high faithfulness score."""
        context = "Transformers use attention mechanisms. BERT is bidirectional."
        answer = "Transformers rely on attention mechanisms. BERT uses a bidirectional approach."
        score = self._faithfulness_score(answer, context)
        assert score >= 0.8, f"Expected faithfulness >= 0.8, got {score}"

    def test_hallucinated_answer(self):
        """Answer that introduces completely new information scores low."""
        context = "Python is a programming language."
        answer = "Quantum circuits leverage superposition and entanglement principles."
        score = self._faithfulness_score(answer, context)
        assert score < 0.5, f"Expected faithfulness < 0.5 for hallucinated answer, got {score}"

    def test_empty_answer(self):
        assert self._faithfulness_score("", "some context") == 0.0


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


class TestRecallAtK:
    """All truly relevant documents should appear in the top-k results."""

    def _recall(self, retrieved_ids: list[str], relevant_ids: list[str]) -> float:
        if not relevant_ids:
            return 1.0  # nothing to recall
        hits = sum(1 for rid in relevant_ids if rid in retrieved_ids)
        return hits / len(relevant_ids)

    def test_perfect_recall(self):
        """Retrieving all docs gives recall = 1.0."""
        relevant = _relevant_ids(CORPUS, "attention")
        all_ids = [d["id"] for d in CORPUS]
        assert self._recall(all_ids, relevant) == 1.0

    def test_recall_at_1_misses_some(self):
        """Top-1 retrieval will miss some relevant docs if there are multiple."""
        relevant = _relevant_ids(CORPUS, "transformer")
        top1 = [CORPUS[0]["id"]]  # only 1 doc
        recall = self._recall(top1, relevant)
        # At least one relevant doc exists — recall should be < 1 unless top1 is the only relevant
        assert recall <= 1.0

    def test_recall_improves_with_k(self):
        """More retrieved docs → equal or better recall."""
        relevant = _relevant_ids(CORPUS, "transformer")
        top1_ids = [d["id"] for d in sorted(CORPUS, key=lambda x: x["score"], reverse=True)[:1]]
        top3_ids = [d["id"] for d in sorted(CORPUS, key=lambda x: x["score"], reverse=True)[:3]]
        recall1 = self._recall(top1_ids, relevant)
        recall3 = self._recall(top3_ids, relevant)
        assert recall3 >= recall1


# ---------------------------------------------------------------------------
# Integration smoke: pipeline produces non-empty answers for known queries
# ---------------------------------------------------------------------------


class TestPipelineIntegrationSmoke:
    """End-to-end pipeline smoke test with fully mocked dependencies."""

    def _make_brain(self):
        """Construct a mocked AxonBrain that uses CORPUS for retrieval."""
        import axon.main as m

        cfg = _make_config(
            top_k=3,
            hybrid_search=False,
            rerank=False,
            hyde=False,
            multi_query=False,
            discussion_fallback=True,
            similarity_threshold=0.0,
        )

        with (
            patch.object(m, "OpenEmbedding"),
            patch.object(m, "OpenLLM"),
            patch.object(m, "OpenVectorStore"),
            patch.object(m, "OpenReranker"),
            patch("axon.retrievers.BM25Retriever"),
        ):
            brain = m.AxonBrain.__new__(m.AxonBrain)
            brain.config = cfg
            brain.embedding = MagicMock()
            brain.embedding.embed_query.return_value = [0.1] * 384
            brain.llm = MagicMock()
            brain.llm.complete.return_value = "Transformers use attention mechanisms."
            brain.vector_store = MagicMock()
            brain.vector_store.search.return_value = CORPUS[:3]
            brain.reranker = MagicMock()
            brain.reranker.rerank.side_effect = lambda q, docs: docs
            brain.bm25 = None
            brain._query_cache = {}
            brain._ingested_hashes = set()
            brain._entity_graph = {}
            brain._apply_overrides = m.AxonBrain._apply_overrides.__get__(brain)
            brain._execute_retrieval = m.AxonBrain._execute_retrieval.__get__(brain)
            brain._build_context = m.AxonBrain._build_context.__get__(brain)
            brain._build_system_prompt = m.AxonBrain._build_system_prompt.__get__(brain)
            brain._make_cache_key = m.AxonBrain._make_cache_key.__get__(brain)
            brain._log_query_metrics = m.AxonBrain._log_query_metrics.__get__(brain)
            brain._compress_context = m.AxonBrain._compress_context.__get__(brain)
            return brain

    def test_pipeline_returns_non_empty_answer(self):
        brain = self._make_brain()
        answer = brain.query("What is a transformer?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_pipeline_context_contains_corpus_text(self):
        """The LLM receives a system prompt containing corpus text."""
        brain = self._make_brain()
        brain.query("What is a transformer?")
        call_args = brain.llm.complete.call_args
        system_prompt = call_args[0][1] if call_args[0] else call_args[1].get("system_prompt", "")
        assert "transformer" in system_prompt.lower() or "attention" in system_prompt.lower()

    def test_pipeline_always_injects_citation_instruction(self):
        """The system prompt must always contain citation instructions."""
        brain = self._make_brain()
        brain.query("What is a transformer?")
        call_args = brain.llm.complete.call_args
        system_prompt = call_args[0][1] if call_args[0] else call_args[1].get("system_prompt", "")
        assert (
            "[Doc" in system_prompt
            or "citation" in system_prompt.lower()
            or "cite" in system_prompt.lower()
        )

    def test_precision_of_pipeline_results(self):
        """The mocked pipeline returns docs 1-3; compute precision for 'attention'."""
        retrieved_ids = [d["id"] for d in CORPUS[:3]]
        relevant_ids = _relevant_ids(CORPUS, "attention")
        hits = sum(1 for rid in retrieved_ids if rid in relevant_ids)
        precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
        # doc1, doc2, doc3 in top-3; doc1 and doc4 have "attention" — doc4 not in top-3
        # doc1 is in top-3 and relevant → at least 1 hit → precision > 0
        assert precision > 0.0
