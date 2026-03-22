"""Query routing, retrieval, context assembly and response generation (QueryRouterMixin)."""
from __future__ import annotations

import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from axon.config import AxonConfig

from axon.code_retrieval import (  # noqa: E402
    CodeRetrievalDiagnostics,
    CodeRetrievalTrace,
    _build_code_bm25_queries,
    _extract_code_query_tokens,
    _looks_like_code_query,
)

logger = logging.getLogger("Axon")

_ROUTE_PROFILES: dict = {
    "factual": {
        "raptor": False,
        "graph_rag": False,
        "parent_doc": False,
        "hyde": False,
        "multi_query": False,
        "step_back": False,
        "query_decompose": False,
    },
    "synthesis": {
        "parent_doc": True,
        "raptor": True,
        "graph_rag": False,
    },
    "table_lookup": {
        "graph_rag": False,
        "raptor": False,
        "parent_doc": False,
        "dataset_type": "knowledge",
    },
    "entity_relation": {
        "graph_rag": True,
        "graph_rag_community": False,
        "parent_doc": False,
        "raptor": False,
    },
    "corpus_exploration": {
        "raptor": True,
        "graph_rag": False,
        "parent_doc": True,
        "multi_query": True,
    },
}


# ---------------------------------------------------------------------------


class QueryRouterMixin:
    def _classify_query_route(self, query: str, cfg: AxonConfig) -> str:
        """Return one of: factual | synthesis | table_lookup | entity_relation | corpus_exploration."""
        if cfg.query_router == "llm":
            return self._classify_query_route_llm(query)
        return self._classify_query_route_heuristic(query)

    def _classify_query_route_heuristic(self, query: str) -> str:
        q = query.lower()

        def contains_keyword(text: str, keywords: set[str], full_text_match: bool = False) -> bool:
            for kw in keywords:
                # Use regex with word boundaries for both single and multi-word keywords
                pattern = rf"\b{re.escape(kw)}\b"
                if re.search(pattern, text):
                    return True
            return False

        # corpus_exploration
        if contains_keyword(q, self._CORPUS_KEYWORDS) or (
            len(query) > 120 and contains_keyword(q, self._SYNTHESIS_KEYWORDS)
        ):
            return "corpus_exploration"
        # entity_relation
        if contains_keyword(q, self._ENTITY_KEYWORDS):
            return "entity_relation"
        # table_lookup
        if contains_keyword(q, self._TABLE_KEYWORDS):
            return "table_lookup"
        # synthesis
        if contains_keyword(q, self._SYNTHESIS_KEYWORDS) or len(query) > 80:
            return "synthesis"
        return "factual"

    def _classify_query_route_llm(self, query: str) -> str:
        prompt = (
            "Classify this query into exactly one category. Respond with only the category name.\n\n"
            "Categories:\n"
            "- factual: short lookup, specific named fact or definition\n"
            "- synthesis: broad question needing information from multiple sections or documents\n"
            "- table_lookup: asks for numerical data, statistics, or structured rows/columns\n"
            "- entity_relation: asks about relationships, connections, or dependencies between entities\n"
            "- corpus_exploration: asks about themes, topics, or a high-level summary of the entire corpus\n\n"
            f"Query: {query}\n\nCategory:"
        )
        valid = {"factual", "synthesis", "table_lookup", "entity_relation", "corpus_exploration"}
        try:
            response = self.llm.generate(prompt, max_tokens=20).strip().lower()
            if response in valid:
                return response
        except Exception:
            pass
        return "factual"

    def _expand_with_entity_graph(
        self, query: str, results: list[dict], cfg=None
    ) -> tuple[list[dict], list[str]]:
        """Expand retrieval results using GraphRAG entity linkage.

        1. Extract entities from the query.
        2. Match query entities against the entity graph using Jaccard similarity.
        3. Perform 1-hop traversal via the relation graph (when enabled).
        4. Fetch any chunks not already in results, tag with _graph_expanded.
        5. Return the expanded list (top_k slicing is deferred to the caller).
        """
        # Item 5: Union LLM-extracted entities with embedding-based matches.
        # LLM extraction captures exact textual mentions; embedding matching adds semantic neighbors.
        query_entities = self._extract_entities(query)
        if (
            getattr(self.config, "graph_rag_entity_embedding_match", True)
            and self._entity_embeddings
        ):
            matched_keys = self._match_entities_by_embedding(query)
            seen_names = {e.get("name", "").lower() for e in query_entities}
            for k in matched_keys:
                if k.lower() not in seen_names:
                    query_entities.append({"name": k, "type": "UNKNOWN", "description": ""})
        if not query_entities:
            return results, []

        active_top_k = (cfg.top_k if cfg is not None else None) or self.config.top_k
        active_cfg = cfg if cfg is not None else self.config

        existing_ids = {r["id"] for r in results}
        # {doc_id: best_score} so we don't lower a score if the same ID matches again
        extra_id_scores: dict[str, float] = {}

        matched_entities: set[str] = set()

        for query_entity in query_entities:
            # Support both new dict-node format and legacy list format
            q_name = query_entity if isinstance(query_entity, str) else query_entity.get("name", "")
            if not q_name:
                continue
            for eid, node in self._entity_graph.items():
                score = self._entity_matches(q_name, eid)
                if score <= 0.0:
                    continue
                matched_entities.add(eid)
                # Scale matched score into [0.5, 0.8) range so it is clearly below
                # a direct vector-match score but still meaningfully ranked.
                doc_score = 0.5 + score * 0.3
                doc_ids = node.get("chunk_ids", [])
                for did in doc_ids:
                    if did not in existing_ids:
                        if extra_id_scores.get(did, 0.0) < doc_score:
                            extra_id_scores[did] = doc_score

        # 1-hop traversal via relation graph
        use_relations = getattr(active_cfg, "graph_rag_relations", True) and self._relation_graph
        if use_relations and matched_entities:
            for src_entity in matched_entities:
                for entry in self._relation_graph.get(src_entity, []):
                    target = entry.get("target", "").lower()
                    if not target:
                        continue
                    target_node = self._entity_graph.get(target, {})
                    target_chunk_ids = target_node.get("chunk_ids", [])
                    for did in target_chunk_ids:
                        if did not in existing_ids:
                            # 1-hop score: lower than direct match
                            hop_score = 0.62
                            if extra_id_scores.get(did, 0.0) < hop_score:
                                extra_id_scores[did] = hop_score

        if not extra_id_scores:
            return results, list(matched_entities)

        # Fetch the extra chunks from the vector store (capped to avoid huge fetches)
        extra_ids = list(extra_id_scores.keys())[:active_top_k]
        try:
            extra_results = self.vector_store.get_by_ids(extra_ids)
            if extra_results:
                logger.info(
                    f"   GraphRAG: expanded results by {len(extra_results)} entity-linked doc(s)"
                )
                for r in extra_results:
                    r["score"] = extra_id_scores.get(r["id"], 0.65)
                    r["_graph_expanded"] = True
                results = list(results) + extra_results
        except Exception as e:
            logger.debug(f"GraphRAG expansion failed: {e}")

        return results, list(matched_entities)

    def _doc_hash(self, doc: dict) -> str:
        """Return an MD5 hex digest of the document's text content."""
        import hashlib

        return hashlib.md5(doc.get("text", "").encode("utf-8", errors="replace")).hexdigest()

    def _prepend_contextual_context(self, chunk: dict, whole_doc_text: str) -> dict:
        """Prepend LLM-generated situating context to chunk text (Anthropic method)."""
        prompt = (
            "<document>\n{doc}\n</document>\n\n"
            "Here is the chunk we want to situate within the whole document:\n"
            "<chunk>\n{chunk}\n</chunk>\n\n"
            "Provide a concise sentence (≤30 words) that situates this chunk "
            "within the document. Output ONLY that sentence, no preamble."
        ).format(doc=whole_doc_text[:3000], chunk=chunk["text"][:800])
        try:
            ctx_sentence = self.llm.generate(prompt, max_tokens=60).strip()
            chunk = dict(chunk)
            chunk["text"] = ctx_sentence + "\n" + chunk["text"]
        except Exception:
            pass  # graceful degradation
        return chunk

    def _make_cache_key(self, query: str, filters, cfg) -> str:
        """Return a stable cache key for the given query + active config.

        All flags that change retrieval or generation are included so two
        requests that differ only by (e.g.) compress_context
        receive distinct cache entries.
        """
        import hashlib
        import json

        # Serialize filters safely regardless of type
        try:
            filters_key = json.dumps(filters or {}, sort_keys=True, default=str)
        except Exception:
            filters_key = str(filters)
        key_data = {
            "q": query,
            "f": filters_key,
            "top_k": cfg.top_k,
            "hybrid": cfg.hybrid_search,
            "hybrid_mode": cfg.hybrid_mode,
            "hybrid_weight": cfg.hybrid_weight,
            "rerank": cfg.rerank,
            "reranker_provider": cfg.reranker_provider,
            "reranker_model": cfg.reranker_model,
            "hyde": cfg.hyde,
            "multi_query": cfg.multi_query,
            "step_back": cfg.step_back,
            "query_decompose": cfg.query_decompose,
            "threshold": cfg.similarity_threshold,
            "discuss": cfg.discussion_fallback,
            "compress_context": cfg.compress_context,
            "cite": cfg.cite,
            "truth_grounding": cfg.truth_grounding,
            "graph_rag": cfg.graph_rag,
            "graph_rag_mode": getattr(cfg, "graph_rag_mode", "local"),
            "raptor": cfg.raptor,
            "raptor_chunk_group_size": cfg.raptor_chunk_group_size,
            "parent_chunk_size": cfg.parent_chunk_size,
            "sentence_window": getattr(cfg, "sentence_window", False),
            "sentence_window_size": getattr(cfg, "sentence_window_size", 3),
            "crag_lite": getattr(cfg, "crag_lite", False),
            "code_graph": getattr(cfg, "code_graph", False),
            "llm_provider": cfg.llm_provider,
            "llm_model": cfg.llm_model,
            "embedding_provider": cfg.embedding_provider,
            "embedding_model": cfg.embedding_model,
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

    def _log_query_metrics(
        self,
        query: str,
        vector_count: int,
        bm25_count: int,
        filtered_count: int,
        final_count: int,
        top_score: float,
        latency_ms: float,
        transformations: dict = None,
        cfg=None,
    ):
        """Log structured metrics for a query."""
        active = cfg if cfg is not None else self.config
        logger.info(
            {
                "event": "query_complete",
                "query_preview": query[:80],
                "latency_ms": round(latency_ms, 1),
                "results": {
                    "vector": vector_count,
                    "bm25": bm25_count,
                    "after_filter": filtered_count,
                    "final": final_count,
                },
                "top_score": round(top_score, 4) if top_score is not None else None,
                "hybrid": active.hybrid_search,
                "rerank": active.rerank,
                "transformations": transformations or {},
            }
        )

    # ── Dataset type detection ────────────────────────────────────────────────
    _SOURCE_POLICY: dict = {
        # (raptor_allowed, graph_rag_allowed)
        "paper": (True, True),
        "doc": (True, True),
        "knowledge": (False, False),
        "discussion": (False, True),
        "codebase": (False, False),  # prose GraphRAG off; code graph is separate
        "manifest": (False, False),
        "reference": (False, False),
    }
    _SOURCE_POLICY_DEFAULT = (True, True)

    _CODE_EXTENSIONS = {
        ".py",
        ".ts",
        ".js",
        ".go",
        ".rs",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".sh",
        ".bash",
        ".zsh",
        ".rb",
        ".pl",
        ".pm",
        ".jl",
        ".kt",
        ".swift",
        ".jsx",
        ".tsx",
        ".cs",
        ".php",
        ".scala",
    }
    _CODE_LINE_PATTERNS = re.compile(
        r"^(def |class |import |from |fn |func |//|#!|public |private |protected |package |using |namespace )"
    )
    _PAPER_SIGNALS = re.compile(
        r"\b(Abstract|Introduction|References|DOI|arXiv|Figure \d|Conclusion|Methodology)\b"
    )
    _DOC_SIGNALS = re.compile(
        r"(^#{1,3} |\*\*\w|\bChapter\b|\bNote:\b|\bStep \d|^\d+\.\s)", re.MULTILINE
    )

    def _decompose_query(self, query: str) -> list[str]:
        """Break a complex query into atomic sub-questions for independent retrieval.

        Returns the original query plus up to 4 sub-questions.
        """
        prompt = (
            "Break the following question into 2–4 simpler, atomic sub-questions that together "
            "cover all aspects of the original. Output each sub-question on a new line with no "
            "numbering or bullet prefix. If the question is already simple, output it unchanged.\n\n"
            "Question: " + query
        )
        response = self.llm.complete(
            prompt, system_prompt="You are an expert at decomposing complex questions."
        )
        sub_qs = [q.strip("- \t1234567890.)") for q in response.split("\n") if q.strip()]
        # Dedupe while preserving order; always keep original first
        seen = {query}
        unique = [query]
        for q in sub_qs[:4]:
            if q and q not in seen:
                seen.add(q)
                unique.append(q)
        return unique

    def _compress_context(
        self, query: str, results: list[dict], cfg=None
    ) -> tuple[list[dict], Any]:
        """Delegate to :class:`axon.compression.ContextCompressor` (Epic 3, Story 3.2).

        Selects strategy and token budget from *cfg* when provided; falls back to
        ``sentence`` compression with no budget for backward compatibility.

        Returns ``(compressed_chunks, CompressionResult)`` so the caller can
        record telemetry without re-computing token counts.
        """
        from axon.compression import CompressionResult, ContextCompressor

        if not results:
            return results, CompressionResult(
                chunks=results,
                strategy_used="none",
                pre_tokens=0,
                post_tokens=0,
                compression_ratio=1.0,
            )

        strategy = getattr(cfg, "compression_strategy", "sentence") if cfg else "sentence"
        token_budget = getattr(cfg, "compression_token_budget", 0) if cfg else 0
        llmlingua_model = getattr(cfg, "graph_rag_llmlingua_model", "") if cfg else ""

        compressor = ContextCompressor(llm=self.llm, llmlingua_model=llmlingua_model)
        result = compressor.compress(query, results, strategy=strategy, token_budget=token_budget)
        return result.chunks, result

    def _get_step_back_query(self, query: str) -> str:
        """Generate a more abstract, step-back version of the query for retrieval."""
        prompt = (
            "Given the following specific question, generate a more general, abstract version "
            "of it that would help retrieve relevant background knowledge. Output only the "
            "abstract question, nothing else.\n\nSpecific question: " + query
        )
        return self.llm.complete(
            prompt,
            system_prompt="You are an expert at abstracting questions to their core concepts.",
        )

    def _get_hyde_document(self, query: str) -> str:
        """Generate a Hypothetical Document Embedding (HyDE) passage."""
        prompt = f"Please write a hypothetical, detailed passage that directly answers the following question. Use informative and factual language.\n\nQuestion: {query}"
        return self.llm.complete(
            prompt, system_prompt="You are a helpful expert answering questions."
        )

    def _get_multi_queries(self, query: str) -> list[str]:
        """Generate alternative query phrasings for multi-query retrieval."""
        prompt = f"Generate 3 alternative phrasings of the following question to help with retrieving documents from a vector database. Output each phrasing on a new line and DO NOT output anything else.\n\nQuestion: {query}"
        response = self.llm.complete(prompt, system_prompt="You are an expert search engineer.")
        queries = [q.strip("- \t1234567890.") for q in response.split("\n") if q.strip()]
        return [query] + queries[:3]  # Always include original query

    def _mmr_deduplicate(self, results: list[dict], cfg) -> list[dict]:
        """Reorder and deduplicate results using Maximal Marginal Relevance (Jaccard similarity).

        Iteratively selects the next document maximising:
            lambda * relevance_score - (1-lambda) * max_jaccard_similarity_to_selected

        Near-duplicates (Jaccard >= 0.85) are dropped entirely.
        """
        if len(results) <= 1:
            return results

        lambda_mult = getattr(cfg, "mmr_lambda", 0.5)
        dup_threshold = 0.85

        def _tok(text: str) -> frozenset:
            return frozenset(re.sub(r"[^\w\s]", "", text.lower()).split())

        def _jac(a: frozenset, b: frozenset) -> float:
            union = a | b
            return len(a & b) / len(union) if union else 0.0

        token_sets = [_tok(r.get("text", "")) for r in results]
        selected_idx: list[int] = []
        remaining = list(range(len(results)))

        while remaining:
            if not selected_idx:
                best = max(remaining, key=lambda i: results[i].get("score", 0.0))
            else:
                best = max(
                    remaining,
                    key=lambda i: (
                        lambda_mult * results[i].get("score", 0.0)
                        - (1 - lambda_mult)
                        * max(_jac(token_sets[i], token_sets[j]) for j in selected_idx)
                    ),
                )
            remaining.remove(best)
            # Drop near-duplicates of already-selected docs
            if (
                selected_idx
                and max(_jac(token_sets[best], token_sets[j]) for j in selected_idx)
                >= dup_threshold
            ):
                continue
            selected_idx.append(best)

        return [results[i] for i in selected_idx]

    def _execute_web_search(self, query: str, count: int = 5) -> list[dict]:
        """Execute a web search using the Brave Search API and return results."""
        import httpx

        try:
            response = httpx.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self.config.brave_api_key,
                },
                params={"q": query, "count": count},
                timeout=10.0,
            )
            response.raise_for_status()
            data = response.json()

            web_results = []
            raw_results = data.get("web", {}).get("results", [])[:count]
            total = max(len(raw_results), 1)
            for i, item in enumerate(raw_results):
                snippet = item.get("description", "")
                title = item.get("title", "")
                url = item.get("url", "")
                web_results.append(
                    {
                        "id": url,
                        "text": f"{title}\n{snippet}",
                        "score": 1.0 - (i / total) * 0.5,
                        "metadata": {"source": url, "title": title},
                        "is_web": True,
                    }
                )
            logger.info(f"Brave Search returned {len(web_results)} results for: {query[:60]}")
            return web_results
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    def _check_mount_revocation(self) -> None:
        """Raise PermissionError if the active mounted share has been revoked since switch."""
        if getattr(self, "_active_project_kind", None) != "mounted":
            return
        desc = getattr(self, "_active_mount_descriptor", None)
        if not desc:
            return
        owner_user_dir = desc.get("owner_user_dir", "")
        key_id = desc.get("share_key_id", "")
        if not owner_user_dir or not key_id:
            return
        import json as _json
        from pathlib import Path as _Path

        manifest_path = _Path(owner_user_dir) / "shares" / ".share_manifest.json"
        try:
            manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return  # Can't reach owner manifest — leave mounted
        for record in manifest.get("issued", []):
            if record.get("key_id") == key_id and record.get("revoked"):
                raise PermissionError(
                    f"Share '{desc.get('project', '')}' has been revoked by the owner. "
                    "Run `/project switch default` to continue with your own projects."
                )

    def _execute_retrieval(self, query: str, filters: dict = None, cfg=None) -> dict:
        """Central retrieval execution logic supporting HyDE, Multi-Query, and Web Search (Parallelized)."""
        self._check_mount_revocation()
        if cfg is None:
            cfg = self.config
        transforms = {
            "hyde_applied": False,
            "multi_query_applied": False,
            "step_back_applied": False,
            "decompose_applied": False,
            "web_search_applied": False,
            "queries": [query],
        }

        # --- Phase 1: Parallel Query Transformation ---
        future_to_task = {}
        if cfg.multi_query:
            future_to_task[self._executor.submit(self._get_multi_queries, query)] = "multi"
        if cfg.step_back:
            future_to_task[self._executor.submit(self._get_step_back_query, query)] = "step_back"
        if cfg.query_decompose:
            future_to_task[self._executor.submit(self._decompose_query, query)] = "decompose"
        if cfg.hyde:
            future_to_task[self._executor.submit(self._get_hyde_document, query)] = "hyde"

        search_queries = [query]
        vector_query = query

        from concurrent.futures import as_completed

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                res = future.result()
                if task == "multi":
                    search_queries.extend([q for q in res if q not in search_queries])
                    transforms["multi_query_applied"] = True
                elif task == "step_back":
                    if res not in search_queries:
                        search_queries.append(res)
                    transforms["step_back_applied"] = True
                elif task == "decompose":
                    search_queries.extend([q for q in res if q not in search_queries])
                    transforms["decompose_applied"] = True
                elif task == "hyde":
                    vector_query = res
                    transforms["hyde_applied"] = True
            except Exception as e:
                logger.warning(f"Retrieval transformation '{task}' failed: {e}")

        transforms["queries"] = search_queries

        # --- Retrieval accountability objects ---
        diagnostics = CodeRetrievalDiagnostics()
        trace = CodeRetrievalTrace()
        trace.query_rewrite_variants = list(search_queries[1:])

        # --- Code mode detection ---
        _code_mode = cfg.code_lexical_boost and _looks_like_code_query(query)
        _code_query_tokens: frozenset = frozenset()
        _effective_top_k = cfg.top_k
        if _code_mode:
            _code_query_tokens = _extract_code_query_tokens(query)
            diagnostics.code_mode_triggered = True
            diagnostics.tokens_extracted = sorted(_code_query_tokens)
            if getattr(cfg, "code_top_k", 0) > 0:
                _effective_top_k = cfg.code_top_k

        # --- Phase 2: Retrieval ---
        fetch_k = cfg.top_k * 3 if (cfg.rerank or cfg.hybrid_search) else cfg.top_k
        if _code_mode:
            fetch_k = int(fetch_k * cfg.code_top_k_multiplier)

        all_vector_results = []
        all_bm25_results = []

        # Vector Search
        if cfg.hyde:
            # HyDE: embed the hypothetical document and search with it.
            # If multi-query/step-back/decompose also produced variants, search with
            # those as well so transforms compose rather than HyDE replacing them.
            all_vector_results.extend(
                self.vector_store.search(
                    self.embedding.embed_query(vector_query), top_k=fetch_k, filter_dict=filters
                )
            )
            if len(search_queries) > 1:
                # search_queries[0] is the original query; additional entries are transform variants
                for variant in search_queries[1:]:
                    all_vector_results.extend(
                        self.vector_store.search(
                            self.embedding.embed_query(variant),
                            top_k=fetch_k,
                            filter_dict=filters,
                        )
                    )
        else:
            # Batch embed all unique queries
            if len(search_queries) == 1:
                all_vector_results.extend(
                    self.vector_store.search(
                        self.embedding.embed_query(search_queries[0]),
                        top_k=fetch_k,
                        filter_dict=filters,
                    )
                )
            else:
                embeddings = self.embedding.embed(search_queries)
                for emb in embeddings:
                    all_vector_results.extend(
                        self.vector_store.search(emb, top_k=fetch_k, filter_dict=filters)
                    )

        # Dedupe vector store results based on ID
        dedup_vector = {}
        for r in all_vector_results:
            if r["id"] not in dedup_vector or r["score"] > dedup_vector[r["id"]]["score"]:
                dedup_vector[r["id"]] = r
        vector_results = list(dedup_vector.values())
        vector_count = len(vector_results)
        diagnostics.channels_activated.append("dense")
        trace.channel_raw_counts["dense"] = vector_count

        # Sentence-Window channel (Story 1.4 — additive dense path)
        # Retrieves by sentence granularity and expands each hit to a coherent
        # ±N-sentence context window.  Window results use chunk-level IDs so they
        # merge cleanly with dense results before BM25 fusion.
        _sw_vs = getattr(self, "_sw_vs", None)
        _sw_index = getattr(self, "_sw_index", None)
        if getattr(cfg, "sentence_window", False) and _sw_vs is not None and len(_sw_vs) > 0:
            _sw_window_size = getattr(cfg, "sentence_window_size", 3)
            _sw_query_emb = self.embedding.embed_query(search_queries[0])
            _sw_hits = _sw_vs.search(_sw_query_emb, top_k=fetch_k)
            diagnostics.sentence_window_hits = len(_sw_hits)
            if _sw_hits and _sw_index is not None:
                # Expand sentence hits to windows, deduplicate by chunk_id
                # (overlapping windows from the same chunk keep the highest score)
                _best_by_chunk: dict[str, dict] = {}
                for hit in _sw_hits:
                    sent_id = hit["id"]
                    chunk_id = hit["metadata"].get("chunk_id", sent_id)
                    score = hit["score"]
                    if chunk_id in _best_by_chunk and _best_by_chunk[chunk_id]["score"] >= score:
                        continue
                    window_text = _sw_index.get_window(sent_id, _sw_window_size)
                    if not window_text:
                        continue
                    _best_by_chunk[chunk_id] = {
                        "id": chunk_id,
                        "text": window_text,
                        "score": score,
                        "vector_score": score,
                        "metadata": {
                            "source": hit["metadata"].get("source", chunk_id),
                            "chunk_id": chunk_id,
                            "_sw_sentence_id": sent_id,
                            "_sw_expanded": True,
                        },
                    }
                sw_window_results = list(_best_by_chunk.values())
                if sw_window_results:
                    diagnostics.sentence_window_used = True
                    diagnostics.channels_activated.append("sentence_window")
                    trace.channel_raw_counts["sentence_window"] = len(sw_window_results)
                    # Merge into dense dedup dict — best score wins per chunk
                    for sw_r in sw_window_results:
                        cid = sw_r["id"]
                        if cid not in dedup_vector or sw_r["score"] > dedup_vector[cid]["score"]:
                            dedup_vector[cid] = sw_r
                    vector_results = list(dedup_vector.values())
                    vector_count = len(vector_results)
                    logger.debug(
                        "sentence_window: %d sentence hits → %d window results merged",
                        len(_sw_hits),
                        len(sw_window_results),
                    )

        # Hybrid Search (Keyword component)
        bm25_count = 0
        if cfg.hybrid_search and self.bm25:
            # Deterministic code query rewriting — extra BM25-only sub-queries
            bm25_queries = list(search_queries)
            if _code_mode and _code_query_tokens:
                code_variants = _build_code_bm25_queries(query, _code_query_tokens)
                for v in code_variants:
                    if v not in bm25_queries:
                        bm25_queries.append(v)
                if code_variants:
                    trace.query_rewrite_variants = [
                        v for v in code_variants if v not in search_queries
                    ]

            # Parallelize BM25 searches across all queries (original + code variants)
            def _bm25_search(q):
                return self.bm25.search(q, top_k=fetch_k)

            all_bm25_lists = list(self._executor.map(_bm25_search, bm25_queries))
            for b_list in all_bm25_lists:
                all_bm25_results.extend(b_list)

            dedup_bm25 = {}
            for r in all_bm25_results:
                if r["id"] not in dedup_bm25 or r["score"] > dedup_bm25[r["id"]]["score"]:
                    dedup_bm25[r["id"]] = r

            # --- Symbol channel injection (code mode only) ---
            if _code_mode and _code_query_tokens:
                sym_hits = self._symbol_channel_search(
                    _code_query_tokens, top_k=fetch_k, filters=filters
                )
                if sym_hits:
                    sym_channels = {
                        r.get("metadata", {}).get("channel", "symbol_name") for r in sym_hits
                    }
                    for ch in sym_channels:
                        if ch not in diagnostics.channels_activated:
                            diagnostics.channels_activated.append(ch)
                        trace.channel_raw_counts[ch] = sum(
                            1 for r in sym_hits if r.get("metadata", {}).get("channel") == ch
                        )
                    # Merge into BM25 dedup (best score wins)
                    for sr in sym_hits:
                        sid = sr["id"]
                        if sid not in dedup_bm25 or sr["score"] > dedup_bm25[sid]["score"]:
                            dedup_bm25[sid] = sr

            bm25_results = list(dedup_bm25.values())
            bm25_count = len(bm25_results)
            diagnostics.channels_activated.append("bm25")
            trace.channel_raw_counts["bm25"] = bm25_count

            # Code mode: override BM25 weight (effective only in weighted mode)
            _fusion_weight = (
                getattr(cfg, "code_bm25_weight", cfg.hybrid_weight)
                if _code_mode
                else cfg.hybrid_weight
            )

            if cfg.hybrid_mode == "rrf":
                from axon.retrievers import reciprocal_rank_fusion

                results = reciprocal_rank_fusion(vector_results, bm25_results)
            else:
                from axon.retrievers import weighted_score_fusion

                results = weighted_score_fusion(vector_results, bm25_results, weight=_fusion_weight)
        else:
            results = vector_results

        # Web Search Fallback (if enabled and local results are insufficient)
        filtered_results = []
        for r in results:
            # BM25-only hits (fused_only=True) have no meaningful vector_score;
            # skip threshold for them so lexical-exact matches always surface.
            if r.get("fused_only"):
                filtered_results.append(r)
                continue
            # When hybrid search is active, apply threshold to the fused score so that
            # docs with strong BM25 scores (e.g. exact-token matches like INC-44721) are
            # not silently suppressed by a low vector_score alone.
            if cfg.hybrid_search:
                sig = r.get("score", r.get("vector_score", 0.0))
            else:
                sig = r.get("vector_score", r.get("score", 0.0))
            if sig >= cfg.similarity_threshold:
                filtered_results.append(r)

        # CRAG-Lite correction policy (Epic 2, Stories 2.2–2.3)
        _crag_lite_enabled = getattr(cfg, "crag_lite", False)
        if _crag_lite_enabled:
            from axon.crag import assess_confidence, evaluate_correction_policy

            _crag_conf = assess_confidence(
                filtered_results=filtered_results,
                total_candidates=len(results),
                similarity_threshold=cfg.similarity_threshold,
            )
            _crag_decision = evaluate_correction_policy(
                confidence=_crag_conf,
                has_local_results=bool(filtered_results),
                truth_grounding_enabled=bool(cfg.truth_grounding),
                crag_lite_threshold=getattr(cfg, "crag_lite_confidence_threshold", 0.4),
            )
            # Story 2.3 — populate diagnostics
            diagnostics.crag_confidence = _crag_conf.score
            diagnostics.crag_verdict = _crag_conf.verdict
            diagnostics.crag_factors = dict(_crag_conf.factors)
            diagnostics.crag_fallback_triggered = _crag_decision.trigger_web_fallback
            diagnostics.crag_fallback_reason = _crag_decision.reason
            logger.debug(
                "CRAG-Lite: score=%.3f verdict=%s decision=%s",
                _crag_conf.score,
                _crag_conf.verdict,
                _crag_decision.reason,
            )
            if _crag_decision.trigger_web_fallback:
                transforms["web_search_applied"] = True
                web_results = self._execute_web_search(query, count=5)
                results = web_results
            elif _crag_decision.trust_local:
                if _crag_conf.verdict == "medium":
                    logger.warning(
                        "CRAG-Lite: medium confidence (%.3f) — trusting local results"
                        " but knowledge-base may be shallow for this query.",
                        _crag_conf.score,
                    )
                results = filtered_results
            else:
                results = filtered_results
        elif not filtered_results and cfg.truth_grounding:
            # Legacy hard-wired guard (active when crag_lite=False)
            transforms["web_search_applied"] = True
            web_results = self._execute_web_search(query, count=5)
            results = web_results
        else:
            results = filtered_results

        # MMR deduplication — reorder and remove near-duplicate chunks
        if getattr(cfg, "mmr", False) and results:
            results = self._mmr_deduplicate(results, cfg)

        # Code lexical boost pass (before graph expansion, after threshold filtering)
        if cfg.code_lexical_boost and results:
            code_fraction = sum(
                1 for r in results if r.get("metadata", {}).get("source_class") == "code"
            ) / max(len(results), 1)
            if code_fraction >= 0.3:
                boost_tokens = _code_query_tokens or _extract_code_query_tokens(query)
                results = self._apply_code_lexical_boost(
                    results, boost_tokens, cfg=cfg, diagnostics=diagnostics, trace=trace
                )

        # Save base count before GraphRAG expansion for accurate metrics
        base_count = len(results)

        # GraphRAG: expand results with entity-linked documents
        _matched_entities: list = []
        if cfg.graph_rag and self._entity_graph:
            results, _matched_entities = self._expand_with_entity_graph(query, results, cfg=cfg)

        # Code graph expansion (structural, independent of prose GraphRAG)
        if getattr(cfg, "code_graph", False) and self._code_graph.get("nodes"):
            results, _matched_code_syms = self._expand_with_code_graph(query, results, cfg=cfg)
            if _matched_code_syms:
                logger.debug("code_graph: matched symbols=%s", _matched_code_syms)

        # Finalize diagnostics
        diagnostics.result_count = len(results)
        diagnostics.fallback_chunks_in_results = sum(
            1 for r in results if r.get("metadata", {}).get("is_fallback")
        )

        return {
            "results": results,
            "vector_count": vector_count,
            "bm25_count": bm25_count,
            "filtered_count": base_count,
            "graph_expanded_count": len(results) - base_count,
            "matched_entities": _matched_entities,
            "transforms": transforms,
            "diagnostics": diagnostics,
            "trace": trace,
            "_effective_top_k": _effective_top_k,
        }

    @staticmethod
    def _merge_graph_slots(results: list[dict], top_k: int, budget: int) -> list[dict]:
        """Merge base and graph-expanded results, preserving graph budget.

        After reranking, graph-expanded chunks may score above base chunks.  The
        previous logic always appended them after base[:top_k], so they could be
        dropped when synthesis only consumed the first ``top_k`` items.

        This helper:
        1. Selects the top ``top_k`` results by score (may include graph-expanded).
        2. Guarantees that at least ``min(budget, n_expanded)`` graph-expanded
           chunks survive, appending any not already in the top-k window.

        Result size is bounded by ``top_k + budget``.
        """
        expanded = [r for r in results if r.get("_graph_expanded")]
        # Sort all results by score descending; take the best top_k
        merged = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)[:top_k]
        merged_ids = {r["id"] for r in merged}
        graph_in_merged = sum(1 for r in merged if r.get("_graph_expanded"))
        # Guarantee budget: append high-scoring graph slots not yet represented
        for r in expanded:
            if graph_in_merged >= budget:
                break
            if r["id"] not in merged_ids:
                merged.append(r)
                merged_ids.add(r["id"])
                graph_in_merged += 1
        return merged

    def search_raw(
        self,
        query: str,
        filters: dict = None,
        overrides: dict = None,
    ) -> tuple:
        """Run retrieval without calling the LLM.

        Returns ``(results, diagnostics, trace)`` — useful for benchmark harnesses,
        the ``/search/raw`` API endpoint, and the ``--dry-run`` CLI flag.
        The result list is already sliced to the effective top-k.
        """
        cfg = self._apply_overrides(overrides)
        retrieval = self._execute_retrieval(query, filters=filters, cfg=cfg)
        results = retrieval["results"]
        diagnostics = retrieval.get("diagnostics", CodeRetrievalDiagnostics())
        trace = retrieval.get("trace", CodeRetrievalTrace())

        if cfg.rerank:
            results = self.reranker.rerank(query, results)

        _top_k = retrieval.get("_effective_top_k", cfg.top_k)
        if cfg.graph_rag and cfg.graph_rag_budget > 0:
            results = self._merge_graph_slots(results, _top_k, cfg.graph_rag_budget)
        else:
            results = results[:_top_k]
        for r in results:
            r.pop("_graph_expanded", None)

        diagnostics.result_count = len(results)
        self._last_diagnostics = diagnostics
        return results, diagnostics, trace

    def _build_context(self, results: list[dict]) -> tuple:
        """Build context string from results, labelling web vs local sources distinctly.

        Returns:
            Tuple of (context_string, has_web_results).
        """
        parts = []
        has_web = False
        for i, r in enumerate(results):
            if r.get("is_web"):
                has_web = True
                title = r.get("metadata", {}).get("title", r["id"])
                parts.append(f"[Web Result {i+1} — {title} ({r['id']})]\n{r['text']}")
            else:
                # Small-to-big: prefer parent passage for richer LLM context
                context_text = r.get("metadata", {}).get("parent_text") or r["text"]
                meta = r.get("metadata", {})
                if meta.get("source_class") == "code":
                    basename = os.path.basename(
                        meta.get("file_path") or meta.get("source") or "unknown"
                    )
                    sym = meta.get("symbol_name", "")
                    sym_type = meta.get("symbol_type", "")
                    if sym:
                        label = (
                            f"{basename} :: {sym} ({sym_type})"
                            if sym_type
                            else f"{basename} :: {sym}"
                        )
                    else:
                        label = basename
                else:
                    label = meta.get("source", r["id"])
                parts.append(f"[Document {i+1} — {label}]\n{context_text}")
        return "\n\n".join(parts), has_web

    def _build_system_prompt(self, has_web: bool, cfg=None, no_context: bool = False) -> str:
        """Return the system prompt based on discussion_fallback and web search state.

        When discussion_fallback is False, uses a strict context-only prompt.
        When True, uses the permissive prompt.
        When cite is False, the citation instruction is removed from the prompt.
        When no_context is True, appends a disclaimer indicating general-knowledge fallback.
        """
        if cfg is None:
            cfg = self.config
        base = self.SYSTEM_PROMPT if cfg.discussion_fallback else self.SYSTEM_PROMPT_STRICT

        if not cfg.cite:
            # Strip citation instruction lines from the prompt
            import re as _re

            base = _re.sub(r"\d+\. \*\*Mandatory Citations\*\*:.*?\n", "", base)

        if no_context:
            return base + (
                "\n\n**Note:** No relevant documents were found in the knowledge base for this query. "
                "The following answer draws on general knowledge only and is not grounded in your documents."
            )
        if not cfg.truth_grounding:
            return base
        if has_web:
            return base + (
                "\n\n**Web Search Used**: Local documents did not contain sufficient information, "
                "so live Brave Search results have been added to your context (marked as '[Web Result]'). "
                "Use these web results to answer the question and always cite the source URL."
            )
        return base + (
            "\n\n**Web Search Available**: If the local documents above are insufficient, "
            "you have access to live Brave Search as a fallback tool. It was not needed for this query."
        )

    def _apply_overrides(self, overrides: dict | None) -> AxonConfig:
        """Return a config copy with per-request overrides applied (thread-safe)."""
        if not overrides:
            return self.config
        import copy

        cfg = copy.copy(self.config)
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def query(
        self,
        query: str,
        filters: dict = None,
        chat_history: list[dict[str, str]] = None,
        overrides: dict = None,
    ) -> str:
        """Retrieve relevant context and synthesise a natural-language answer.

        Args:
            query: The question or prompt to answer.
            filters: Optional metadata filters applied to vector search.
            chat_history: Previous conversation turns as ``[{"role": ..., "content": ...}]``.
                When non-empty, the query cache is bypassed.
            overrides: Per-request config overrides (e.g. ``{"top_k": 5, "rerank": True}``).
                Keys must match :class:`AxonConfig` field names.

        Returns:
            A synthesised answer string from the LLM.
        """
        # Warn (don't block) if the embedding model has changed — retrieval
        # results will be degraded but the user can still access existing data.
        self._validate_embedding_meta(on_mismatch="warn")

        t0 = time.time()
        cfg = self._apply_overrides(overrides)

        # --- System-level query router ---
        if cfg.query_router != "off":
            route = self._classify_query_route(query, cfg)
            profile_overrides = _ROUTE_PROFILES.get(route, {})
            for k, v in profile_overrides.items():
                if hasattr(cfg, k):
                    object.__setattr__(cfg, k, v)
            logger.debug("query_router: route=%s overrides=%s", route, profile_overrides)
        elif cfg.graph_rag_auto_route != "off" and cfg.graph_rag:
            # legacy binary classifier fallback
            _needs_grag = self._classify_query_needs_graphrag(query, cfg.graph_rag_auto_route)
            if not _needs_grag:
                cfg = self._apply_overrides({**(overrides or {}), "graph_rag": False})
                logger.debug("Auto-route: GraphRAG bypassed for query '%s...'", query[:60])

        # Cache lookup (only when chat_history is empty — cached responses don't track turns)
        cache_key = None
        if cfg.query_cache and not chat_history and cfg.query_cache_size >= 1:
            cache_key = self._make_cache_key(query, filters, cfg)
            with self._cache_lock:
                if cache_key in self._query_cache:
                    logger.info(f"Cache hit for query: {query[:60]}")
                    # Move to end so this entry is treated as most-recently-used (LRU)
                    self._query_cache.move_to_end(cache_key)
                    return self._query_cache[cache_key]

        retrieval = self._execute_retrieval(query, filters, cfg=cfg)
        results = retrieval["results"]
        self._last_diagnostics = retrieval.get("diagnostics", CodeRetrievalDiagnostics())

        if not results:
            self._log_query_metrics(
                query,
                retrieval["vector_count"],
                retrieval["bm25_count"],
                retrieval["filtered_count"],
                0,
                0.0,
                (time.time() - t0) * 1000,
                retrieval["transforms"],
                cfg=cfg,
            )

            if cfg.discussion_fallback:
                # Send plain query as user message so multi-turn history stays consistent
                self._last_provenance = {
                    "answer_source": "no_context_fallback",
                    "retrieved_count": 0,
                    "web_count": 0,
                }
                return self.llm.complete(
                    query,
                    self._build_system_prompt(False, cfg=cfg, no_context=True),
                    chat_history=chat_history,
                )

            self._last_provenance = {
                "answer_source": "no_results",
                "retrieved_count": 0,
                "web_count": 0,
            }
            return "I don't have any relevant information to answer that question."

        if cfg.rerank:
            results = self.reranker.rerank(query, results)

        _top_k = retrieval.get("_effective_top_k", cfg.top_k)
        if cfg.graph_rag and cfg.graph_rag_budget > 0:
            results = self._merge_graph_slots(results, _top_k, cfg.graph_rag_budget)
        else:
            results = results[:_top_k]
        for r in results:
            r.pop("_graph_expanded", None)

        # RAPTOR drill-down — replace summary hits with grounded leaf chunks
        if cfg.raptor and getattr(cfg, "raptor_drilldown", True):
            results = self._raptor_drilldown(query, results, cfg=cfg)

        # Artifact-type ranking pass
        if cfg.raptor or cfg.graph_rag:
            results = self._apply_artifact_ranking(results, cfg=cfg)

        graph_mode = getattr(cfg, "graph_rag_mode", "local")
        _matched_entities = retrieval.get("matched_entities", [])

        # GraphRAG local context header
        if (
            cfg.graph_rag
            and graph_mode in ("local", "hybrid")
            and _matched_entities
            and (self._entity_graph or self._community_summaries)
        ):
            _local_ctx = self._local_search_context(query, _matched_entities, cfg)
        else:
            _local_ctx = ""

        if cfg.compress_context:
            results, _comp = self._compress_context(query, results, cfg=cfg)
            # Story 3.3: write compression telemetry into diagnostics
            _d = self._last_diagnostics
            _d.compression_strategy = _comp.strategy_used
            _d.compression_pre_tokens = _comp.pre_tokens
            _d.compression_post_tokens = _comp.post_tokens
            _d.compression_ratio = _comp.compression_ratio
            _d.compression_fallback_reason = _comp.fallback_reason
        final_count = len(results)
        # Use rerank_score when reranking was active (reflects the actual ranking signal)
        if results:
            r0 = results[0]
            top_score = (
                r0.get("rerank_score", r0.get("score", 0)) if cfg.rerank else r0.get("score", 0)
            )
        else:
            top_score = 0

        # A4: Exclude community report synthetic docs from citation candidates.
        # They are internal artifacts and confuse users when cited as source documents.
        # Global community context is injected separately via _global_search_map_reduce.
        citation_results = [
            r
            for r in results
            if r.get("metadata", {}).get("graph_rag_type") != "community_report"
            and not r.get("id", "").startswith("__community__")
        ]
        context, has_web = self._build_context(citation_results)

        # GraphRAG global context injection
        # Lazy mode — generate summaries on first global query if not yet generated
        if (
            cfg.graph_rag
            and graph_mode in ("global", "hybrid")
            and not self._community_summaries
            and self._community_levels
            and getattr(cfg, "graph_rag_community_lazy", False)
        ):
            with self._community_rebuild_lock:
                if not self._community_summaries:
                    self._generate_community_summaries(query_hint=query)
                    if getattr(cfg, "graph_rag_index_community_reports", True):
                        self._index_community_reports_in_vector_store()
        if cfg.graph_rag and graph_mode in ("global", "hybrid") and self._community_summaries:
            _global_ctx = self._global_search_map_reduce(query, cfg)
            if _global_ctx:
                if graph_mode == "global":
                    context = f"**Knowledge Graph Community Reports:**\n{_global_ctx}"
                else:  # hybrid
                    context = (
                        f"**Knowledge Graph Community Reports:**\n{_global_ctx}\n\n"
                        f"**Document Excerpts:**\n{context}"
                    )

        # Prepend local GraphRAG header
        if _local_ctx:
            context = (
                f"**GraphRAG Local Context:**\n{_local_ctx}\n\n**Document Excerpts:**\n{context}"
            )

        # Inject RAG context into system prompt so the user message stays as the plain
        # question — this keeps multi-turn chat_history consistent across turns.
        system_prompt = (
            self._build_system_prompt(has_web, cfg=cfg)
            + f"\n\n**Relevant context from documents:**\n{context}"
        )
        _web_count = sum(1 for r in citation_results if r.get("is_web"))
        self._last_provenance = {
            "answer_source": "web_snippet_fallback" if has_web else "local_kb",
            "retrieved_count": len(citation_results),
            "web_count": _web_count,
        }
        response = self.llm.complete(query, system_prompt, chat_history=chat_history)
        self._log_query_metrics(
            query,
            retrieval["vector_count"],
            retrieval["bm25_count"],
            retrieval["filtered_count"],
            final_count,
            top_score,
            (time.time() - t0) * 1000,
            retrieval["transforms"],
            cfg=cfg,
        )

        if cache_key is not None:
            with self._cache_lock:
                # Evict least-recently-used entry when cache is at capacity
                if len(self._query_cache) >= cfg.query_cache_size and self._query_cache:
                    self._query_cache.popitem(last=False)  # pop LRU (front of OrderedDict)
                self._query_cache[cache_key] = response
                self._query_cache.move_to_end(cache_key)  # mark as most-recently-used

        return response

    def query_stream(
        self,
        query: str,
        filters: dict = None,
        chat_history: list[dict[str, str]] = None,
        overrides: dict = None,
    ):
        """Streaming variant of :meth:`query` — yields text chunks as they arrive.

        The first yielded item is always a ``{"type": "sources", "sources": [...]}``
        dict so callers can display source attribution before streaming begins.
        Subsequent items are plain string chunks from the LLM stream.
        """
        self._validate_embedding_meta(on_mismatch="warn")
        cfg = self._apply_overrides(overrides)
        retrieval = self._execute_retrieval(query, filters, cfg=cfg)
        results = retrieval["results"]
        self._last_diagnostics = retrieval.get("diagnostics", CodeRetrievalDiagnostics())

        if not results:
            if cfg.discussion_fallback:
                # Send plain query as user message so multi-turn history stays consistent
                self._last_provenance = {
                    "answer_source": "no_context_fallback",
                    "retrieved_count": 0,
                    "web_count": 0,
                }
                yield from self.llm.stream(
                    query,
                    self._build_system_prompt(False, cfg=cfg, no_context=True),
                    chat_history=chat_history,
                )
                return
            self._last_provenance = {
                "answer_source": "no_results",
                "retrieved_count": 0,
                "web_count": 0,
            }
            yield "I don't have any relevant information to answer that question."
            return

        if cfg.rerank:
            results = self.reranker.rerank(query, results)

        _top_k = retrieval.get("_effective_top_k", cfg.top_k)
        if cfg.graph_rag and cfg.graph_rag_budget > 0:
            results = self._merge_graph_slots(results, _top_k, cfg.graph_rag_budget)
        else:
            results = results[:_top_k]
        for r in results:
            r.pop("_graph_expanded", None)

        # RAPTOR drill-down — replace summary hits with grounded leaf chunks
        if cfg.raptor and getattr(cfg, "raptor_drilldown", True):
            results = self._raptor_drilldown(query, results, cfg=cfg)

        # Artifact-type ranking pass
        if cfg.raptor or cfg.graph_rag:
            results = self._apply_artifact_ranking(results, cfg=cfg)

        graph_mode = getattr(cfg, "graph_rag_mode", "local")
        _matched_entities = retrieval.get("matched_entities", [])

        # GraphRAG local context header
        if (
            cfg.graph_rag
            and graph_mode in ("local", "hybrid")
            and _matched_entities
            and (self._entity_graph or self._community_summaries)
        ):
            _local_ctx = self._local_search_context(query, _matched_entities, cfg)
        else:
            _local_ctx = ""

        if cfg.compress_context:
            results, _comp = self._compress_context(query, results, cfg=cfg)
            # Story 3.3: telemetry on the streaming path
            _sd = self._last_diagnostics
            _sd.compression_strategy = _comp.strategy_used
            _sd.compression_pre_tokens = _comp.pre_tokens
            _sd.compression_post_tokens = _comp.post_tokens
            _sd.compression_ratio = _comp.compression_ratio
            _sd.compression_fallback_reason = _comp.fallback_reason

        # A4: Exclude community report synthetic docs from citation candidates.
        citation_results = [
            r
            for r in results
            if r.get("metadata", {}).get("graph_rag_type") != "community_report"
            and not r.get("id", "").startswith("__community__")
        ]
        context, has_web = self._build_context(citation_results)

        # GraphRAG global context injection
        # Lazy mode — generate summaries on first global query if not yet generated
        if (
            cfg.graph_rag
            and graph_mode in ("global", "hybrid")
            and not self._community_summaries
            and self._community_levels
            and getattr(cfg, "graph_rag_community_lazy", False)
        ):
            with self._community_rebuild_lock:
                if not self._community_summaries:
                    self._generate_community_summaries(query_hint=query)
                    if getattr(cfg, "graph_rag_index_community_reports", True):
                        self._index_community_reports_in_vector_store()
        if cfg.graph_rag and graph_mode in ("global", "hybrid") and self._community_summaries:
            _global_ctx = self._global_search_map_reduce(query, cfg)
            if _global_ctx:
                if graph_mode == "global":
                    context = f"**Knowledge Graph Community Reports:**\n{_global_ctx}"
                else:  # hybrid
                    context = (
                        f"**Knowledge Graph Community Reports:**\n{_global_ctx}\n\n"
                        f"**Document Excerpts:**\n{context}"
                    )

        # Prepend local GraphRAG header
        if _local_ctx:
            context = (
                f"**GraphRAG Local Context:**\n{_local_ctx}\n\n**Document Excerpts:**\n{context}"
            )

        # Inject RAG context into system prompt so the user message stays as the plain
        # question — this keeps multi-turn chat_history consistent across turns.
        system_prompt = (
            self._build_system_prompt(has_web, cfg=cfg)
            + f"\n\n**Relevant context from documents:**\n{context}"
        )
        _web_count = sum(1 for r in citation_results if r.get("is_web"))
        self._last_provenance = {
            "answer_source": "web_snippet_fallback" if has_web else "local_kb",
            "retrieved_count": len(citation_results),
            "web_count": _web_count,
        }

        # Yield a marker object so UI can optionally reconstruct sources
        yield {"type": "sources", "sources": results}

        yield from self.llm.stream(query, system_prompt, chat_history=chat_history)

    async def load_directory(self, directory: str):
        from axon.loaders import DirectoryLoader

        loader = DirectoryLoader()
        logger.info(f"Scanning: {directory}")
        documents = await loader.aload(directory)
        if documents:
            self.ingest(documents)

    def list_documents(self) -> list[dict[str, Any]]:
        """Return all unique source files in the knowledge base with chunk counts.

        Returns:
            List of dicts sorted by source name, each with keys:
                - source (str): File name / source identifier.
                - chunks (int): Number of stored chunks for this source.
                - doc_ids (List[str]): All chunk IDs for this source.
        """
        return self.vector_store.list_documents()
