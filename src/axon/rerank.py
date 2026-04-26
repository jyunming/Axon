"""
src/axon/rerank.py

OpenReranker extracted from main.py for Phase 2 of the Axon refactor.
"""

import logging
from typing import Any

from axon.config import AxonConfig

logger = logging.getLogger("Axon")


class OpenReranker:
    """Document reranker supporting cross-encoder and pointwise LLM providers.
    When ``reranker_provider="cross-encoder"`` a sentence-transformers CrossEncoder
    scores each (query, document) pair.  When ``reranker_provider="llm"`` the
    active LLM rates each document on a 1–10 scale in parallel threads
    (pointwise LLM reranker, not listwise RankGPT).
    """

    def __init__(self, config: AxonConfig):
        self.config = config
        self.model = None
        self.llm = None
        if self.config.rerank:
            if self.config.reranker_provider == "cross-encoder":
                try:
                    from sentence_transformers import CrossEncoder

                    logger.info(f"Loading Reranker: {self.config.reranker_model}")
                    self.model = CrossEncoder(self.config.reranker_model)
                except ImportError:
                    logger.error("sentence-transformers not installed. Reranking disabled.")
                    self.config.rerank = False
            elif self.config.reranker_provider == "llm":
                from axon.llm import OpenLLM

                logger.info("Using LLM for Re-ranking (pointwise)")
                self.llm = OpenLLM(self.config)

    def rerank(self, query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Rerank a list of documents based on a query.
        """
        if not self.config.rerank or (not self.model and not self.llm) or not documents:
            return documents
        logger.info(f"Reranking {len(documents)} documents...")
        if self.config.reranker_provider == "llm" and self.llm:
            return self._llm_rerank(query, documents)
        # Cross-encoder pointwise scoring
        # Prepare pairs: (query, doc_text)
        pairs = [[query, doc["text"]] for doc in documents]
        # Get scores
        scores = self.model.predict(pairs)
        # Add scores to documents and sort
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)
        # Sort by rerank score descending
        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs

    def _llm_rerank(self, query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pointwise LLM scoring implementation (scores each document independently on 1–10 scale)."""
        system_prompt = "You are an expert relevance ranker. Rate the relevance of the document to the query on a scale from 1 to 10. Output ONLY the integer score."
        # Tolerant parser: pulls the first integer-or-float run out of
        # the model's response. The previous ``response.replace(".", "",
        # 1).isdigit()`` returned 0.0 for anything that wasn't a bare
        # number — so a perfectly valid ``"Score: 8"`` reply scored 0,
        # making the rerank output useless. (audit P2)
        import re as _re
        from concurrent.futures import ThreadPoolExecutor

        _SCORE_RE = _re.compile(r"-?\d+(?:\.\d+)?")

        def score_doc(doc):
            prompt = f"Query: {query}\n\nDocument: {doc['text']}\n\nScore (1-10):"
            try:
                response = self.llm.complete(prompt, system_prompt=system_prompt).strip()
                m = _SCORE_RE.search(response)
                if not m:
                    return 0.0
                try:
                    return max(0.0, min(10.0, float(m.group(0))))
                except ValueError:
                    return 0.0
            except Exception:
                return 0.0

        if not documents:
            return documents
        # Batch concurrent requests to reduce overall latency
        with ThreadPoolExecutor(max_workers=max(1, min(10, len(documents)))) as executor:
            scores = list(executor.map(score_doc, documents))
        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score
        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs
