"""Code-aware retrieval: doc bridge, lexical boost, symbol search (CodeRetrievalMixin)."""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger("Axon")

_SYMBOL_KEYWORDS: frozenset[str] = frozenset(
    {
        "class",
        "method",
        "function",
        "def",
        "func",
        "import",
        "module",
        "package",
        "splitter",
        "loader",
        "retriever",
    }
)

_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".go", ".rs", ".ts", ".js", ".sh", ".rb", ".jl", ".cpp", ".c", ".java", ".kt"}
)


def _extract_code_query_tokens(query: str) -> frozenset[str]:
    """Extract identifier-like tokens from a query for lexical code matching.

    Returns a frozenset of lowercase token strings covering:
    - CamelCase identifiers and their split parts
    - snake_case parts
    - Basename references (loaders.py → "loaders")
    - Qualified names (foo.bar → "foo", "bar", "foo.bar")
    - All identifiers of length >= 4
    """
    tokens: set[str] = set()

    # Basename references: strip known code extensions
    for ext in _CODE_EXTENSIONS:
        for m in re.finditer(r"\b(\w+)" + re.escape(ext) + r"\b", query):
            tokens.add(m.group(1).lower())

    # Qualified names: foo.bar → {"foo", "bar", "foo.bar"}
    for m in re.finditer(r"\b([A-Za-z_]\w+)\.([A-Za-z_]\w+)\b", query):
        tokens.add(m.group(1).lower())
        tokens.add(m.group(2).lower())
        tokens.add((m.group(1) + "." + m.group(2)).lower())

    # All identifier-like tokens (length >= 4)
    for m in re.finditer(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b", query):
        word = m.group(0)
        tokens.add(word.lower())

        # CamelCase split: CodeAwareSplitter → ["Code", "Aware", "Splitter"]
        camel_parts = re.findall(r"[A-Z][a-z0-9]*|[a-z][a-z0-9]*", word)
        for part in camel_parts:
            if len(part) >= 3:
                tokens.add(part.lower())

        # snake_case split: _split_python_ast → ["split", "python", "ast"]
        if "_" in word:
            for part in word.split("_"):
                if len(part) >= 3:
                    tokens.add(part.lower())

    return frozenset(tokens)


def _looks_like_code_query(query: str) -> bool:
    """Return True if the query is likely asking about code identifiers or files."""
    # CamelCase identifier
    if re.search(r"[a-z][A-Z]", query):
        return True
    # snake_case identifier
    if re.search(r"[a-z]_[a-z]", query):
        return True
    # Filename with a known code extension
    for ext in _CODE_EXTENSIONS:
        if ext in query:
            return True
    # Symbol keyword
    query_lower = query.lower()
    for kw in _SYMBOL_KEYWORDS:
        if kw in query_lower.split():
            return True
    return False


def _build_code_bm25_queries(query: str, query_tokens: frozenset) -> list[str]:
    """Build deterministic BM25-only sub-queries from code identifier tokens.

    Expands CamelCase, snake_case, dotted module paths, and filename stems
    into short search strings that target BM25's lexical index.  These are
    *not* fed to vector search — they are added to the BM25 pool only.

    Returns a list of query strings (may be empty if no useful variants found).
    """
    variants: list[str] = []
    seen: set[str] = {query.lower()}

    for tok in sorted(query_tokens):  # deterministic order
        if len(tok) < 4:
            continue
        if tok in seen:
            continue

        # CamelCase → spaced words (e.g. "CodeAwareSplitter" → "Code Aware Splitter")
        camel_parts = re.findall(r"[A-Z][a-z0-9]*|[a-z][a-z0-9]*", tok)
        if len(camel_parts) > 1:
            spaced = " ".join(p for p in camel_parts if len(p) >= 2)
            if spaced.lower() not in seen:
                variants.append(spaced)
                seen.add(spaced.lower())

        # snake_case → spaced words (e.g. "split_python_ast" → "split python ast")
        if "_" in tok:
            parts = [p for p in tok.split("_") if len(p) >= 2]
            if len(parts) > 1:
                spaced = " ".join(parts)
                if spaced not in seen:
                    variants.append(spaced)
                    seen.add(spaced)

        # Dotted module path → base name (e.g. "axon.loaders" → "loaders")
        if "." in tok:
            base = tok.rsplit(".", 1)[-1]
            if len(base) >= 4 and base not in seen:
                variants.append(base)
                seen.add(base)

        # Raw token itself as a BM25 exact-ish search
        if tok not in seen:
            variants.append(tok)
            seen.add(tok)

    return variants


# ---------------------------------------------------------------------------
# Feature evidence surface registry
# ---------------------------------------------------------------------------
# Every retrieval feature must declare its evidence surface here so that:
#   1. Diagnostics remain coherent as the feature set grows.
#   2. Benchmark authors know which query slice exercises each feature.
#   3. Silent complexity growth is prevented — if a feature has no measurable
#      counter and no benchmark slice it should not exist.
#
# Schema per entry:
#   diagnostic_field   — field name on CodeRetrievalDiagnostics (str, or None for trace-only)
#   activation_counter — attribute on CodeRetrievalDiagnostics that counts activations
#                        (int field, or None if boolean flag suffices)
#   benchmark_slice    — human-readable description of the query type / corpus slice
#                        where this feature is expected to improve results
# ---------------------------------------------------------------------------

_RETRIEVAL_FEATURES: dict[str, dict] = {
    "code_mode_detection": {
        "diagnostic_field": "code_mode_triggered",
        "activation_counter": None,  # boolean: True/False per query
        "benchmark_slice": (
            "Queries containing CamelCase identifiers, snake_case names, "
            "file extensions (.py/.go/etc.), or symbol keywords (class/function/def)"
        ),
    },
    "lexical_boost": {
        "diagnostic_field": "boost_applied",
        "activation_counter": None,  # boolean: True when lex scores are non-zero
        "benchmark_slice": (
            "Code queries where exact symbol_name or qualified_name appears in the query; "
            "expected to rank exact-match chunks above broad module-level chunks"
        ),
    },
    "symbol_channel": {
        "diagnostic_field": "channels_activated",  # entry 'symbol' in list
        "activation_counter": None,
        "benchmark_slice": (
            "Queries with a known function/class/method name that exists verbatim in "
            "BM25 corpus metadata (symbol_name or qualified_name field)"
        ),
    },
    "code_bm25_rewrite": {
        "diagnostic_field": "channels_activated",  # entry 'bm25_variants' in list
        "activation_counter": None,
        "benchmark_slice": (
            "Code queries with multi-part identifiers (CamelCase, snake_case, dotted paths); "
            "rewrite expands to spaced tokens for improved BM25 recall"
        ),
    },
    "per_file_diversity": {
        "diagnostic_field": None,  # trace-only: diversity_cap_deferrals / deferred_chunk_ids
        "activation_counter": None,
        "benchmark_slice": (
            "Queries over a corpus with many chunks per file (e.g. large main.py); "
            "diversity cap prevents top-k from being monopolised by a single file"
        ),
    },
    "line_range_tightness": {
        "diagnostic_field": None,  # trace-only: per_result_score_breakdown 'tightness' key
        "activation_counter": None,
        "benchmark_slice": (
            "Deep implementation-location queries (e.g. 'how does X work') where the correct "
            "answer is a ≤30-line function chunk rather than a broad 80-line module chunk"
        ),
    },
    "fallback_chunk_telemetry": {
        "diagnostic_field": "fallback_chunks_in_results",
        "activation_counter": "fallback_chunks_in_results",
        "benchmark_slice": (
            "Queries where the correct chunk was produced by the splitter's fallback path "
            "(e.g. non-Python/non-supported language files); "
            "high fallback count indicates the code corpus needs better loader coverage"
        ),
    },
    "dry_run_mode": {
        "diagnostic_field": None,  # mode flag, not a per-query metric
        "activation_counter": None,
        "benchmark_slice": (
            "All queries in benchmark harness; dry-run surfaces diagnostics+trace without "
            "an LLM call, enabling fast iterative ranking evaluation"
        ),
    },
}


# ---------------------------------------------------------------------------
# Retrieval accountability types (Layer 1)
# ---------------------------------------------------------------------------


@dataclass
class CodeRetrievalDiagnostics:
    """Stable external contract for code retrieval observability.

    Versioned and serializable. Returned in API responses (include_diagnostics=True)
    and benchmark output. Schema version bumps when field semantics change.
    """

    diagnostics_version: str = "1.3"
    code_mode_triggered: bool = False
    tokens_extracted: list = field(default_factory=list)
    channels_activated: list = field(default_factory=list)
    result_count: int = 0
    boost_applied: bool = False
    fallback_chunks_in_results: int = 0
    # Sentence-window retrieval fields (Epic 1, Story 1.4)
    sentence_window_used: bool = False
    sentence_window_hits: int = 0  # unique sentence hits before window expansion
    # CRAG-Lite confidence fields (Epic 2, Story 2.3)
    crag_confidence: float = -1.0  # -1.0 = CRAG-Lite not run
    crag_verdict: str = ""  # "high" | "medium" | "low" | ""
    crag_factors: dict = field(default_factory=dict)  # per-signal contributions
    crag_fallback_triggered: bool = False
    crag_fallback_reason: str = ""
    # Compression telemetry (Epic 3, Story 3.3)
    compression_strategy: str = ""  # "none" | "sentence" | "llmlingua" | "" = not run
    compression_pre_tokens: int = 0
    compression_post_tokens: int = 0
    compression_ratio: float = 1.0  # post/pre; 1.0 = no reduction
    compression_fallback_reason: str = ""  # non-empty when fell back from requested strategy

    def to_dict(self) -> dict:
        return {
            "diagnostics_version": self.diagnostics_version,
            "code_mode_triggered": self.code_mode_triggered,
            "tokens_extracted": list(self.tokens_extracted),
            "channels_activated": list(self.channels_activated),
            "result_count": self.result_count,
            "boost_applied": self.boost_applied,
            "fallback_chunks_in_results": self.fallback_chunks_in_results,
            "sentence_window_used": self.sentence_window_used,
            "sentence_window_hits": self.sentence_window_hits,
            "crag_confidence": self.crag_confidence,
            "crag_verdict": self.crag_verdict,
            "crag_factors": dict(self.crag_factors),
            "crag_fallback_triggered": self.crag_fallback_triggered,
            "crag_fallback_reason": self.crag_fallback_reason,
            "compression_strategy": self.compression_strategy,
            "compression_pre_tokens": self.compression_pre_tokens,
            "compression_post_tokens": self.compression_post_tokens,
            "compression_ratio": self.compression_ratio,
            "compression_fallback_reason": self.compression_fallback_reason,
        }

    def to_json(self) -> str:
        import json as _j

        return _j.dumps(self.to_dict())


@dataclass
class CodeRetrievalTrace:
    """Internal debug trace — volatile, not returned in API by default.

    Contains per-result score breakdowns, channel raw counts, query variants,
    and diversity-cap deferral details. Use for offline tuning only.
    """

    per_result_score_breakdown: list = field(default_factory=list)
    channel_raw_counts: dict = field(default_factory=dict)
    query_rewrite_variants: list = field(default_factory=list)
    diversity_cap_deferrals: int = 0
    deferred_chunk_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "per_result_score_breakdown": list(self.per_result_score_breakdown),
            "channel_raw_counts": dict(self.channel_raw_counts),
            "query_rewrite_variants": list(self.query_rewrite_variants),
            "diversity_cap_deferrals": self.diversity_cap_deferrals,
            "deferred_chunk_ids": list(self.deferred_chunk_ids),
        }


def _classify_retrieval_failure(
    results: list[dict],
    query_tokens: frozenset,
    expected_symbol: str | None = None,
) -> list[str]:
    """Classify automatable retrieval failure modes for CI benchmarking.

    Returns a list of applicable labels. Labels that require ground truth
    (e.g. synthesis failures) are NOT included — those are benchmark-only.

    Labels:
    - ``exact_symbol_missed``: query had symbol tokens but none matched result symbol_names
    - ``right_file_wrong_block``: expected_symbol's file appeared but not the exact symbol chunk
    - ``too_many_broad_chunks``: >50% results lack code source_class or symbol_type
    - ``fallback_chunk_involved``: any result has is_fallback=True in metadata
    """
    if not results:
        return []

    labels: list[str] = []

    result_symbols = [(r.get("metadata", {}).get("symbol_name") or "").lower() for r in results]
    result_sources = [
        (
            r.get("metadata", {}).get("source") or r.get("metadata", {}).get("file_path") or ""
        ).lower()
        for r in results
    ]

    # exact_symbol_missed
    code_tokens = frozenset(t for t in query_tokens if len(t) >= 3)
    if code_tokens:
        any_symbol_hit = any(
            sym and any(tok in sym or sym in tok for tok in code_tokens) for sym in result_symbols
        )
        if not any_symbol_hit:
            labels.append("exact_symbol_missed")

    # right_file_wrong_block
    if expected_symbol:
        exp_lower = expected_symbol.lower()
        file_hit = any(exp_lower in src for src in result_sources)
        sym_hit = any(exp_lower in sym for sym in result_symbols if sym)
        if file_hit and not sym_hit:
            labels.append("right_file_wrong_block")

    # too_many_broad_chunks
    broad = sum(
        1
        for r in results
        if r.get("metadata", {}).get("source_class") != "code"
        or r.get("metadata", {}).get("symbol_type") is None
    )
    if broad / len(results) > 0.5:
        labels.append("too_many_broad_chunks")

    # fallback_chunk_involved
    if any(r.get("metadata", {}).get("is_fallback") for r in results):
        labels.append("fallback_chunk_involved")

    return labels


class CodeRetrievalMixin:
    def _build_code_doc_bridge(self, prose_chunks: list[dict]) -> None:
        """Phase 3: add MENTIONED_IN edges from code symbol nodes to prose chunks.

        Scans prose chunk text for occurrences of known code symbol/file names.
        Only matches names >= 4 chars to avoid false positives on short tokens.
        """
        nodes: dict = self._code_graph.get("nodes", {})
        if not nodes:
            return

        edges_list: list = self._code_graph.setdefault("edges", [])
        existing_edges: set = {(e["source"], e["target"], e["edge_type"]) for e in edges_list}

        # Build lookup: name → node_id  (symbols + file basenames/stems)
        sym_lookup: dict[str, str] = {}
        for node_id, node in nodes.items():
            name = node.get("name", "")
            if len(name) >= 4:
                sym_lookup[name] = node_id
            if node.get("node_type") == "file":
                stem = os.path.splitext(name)[0]
                if len(stem) >= 4:
                    sym_lookup[stem] = node_id

        for chunk in prose_chunks:
            text = chunk.get("text", "")
            chunk_id = chunk.get("id", "")
            if not text or not chunk_id:
                continue
            for sym_name, node_id in sym_lookup.items():
                if re.search(r"\b" + re.escape(sym_name) + r"\b", text):
                    ek = (node_id, chunk_id, "MENTIONED_IN")
                    if ek not in existing_edges:
                        edges_list.append(
                            {
                                "source": node_id,
                                "target": chunk_id,
                                "edge_type": "MENTIONED_IN",
                                "chunk_id": chunk_id,
                            }
                        )
                        existing_edges.add(ek)

    def _apply_code_lexical_boost(
        self,
        results: list[dict],
        query_tokens: frozenset,
        cfg=None,
        diagnostics: CodeRetrievalDiagnostics | None = None,
        trace: CodeRetrievalTrace | None = None,
    ) -> list[dict]:
        """Re-score code results by lexical identifier match, then enforce per-file diversity.

        Only results with ``metadata.source_class == "code"`` are re-scored; all
        others pass through unchanged.  If no query tokens match any result, the
        original scores are preserved (no degradation on non-identifier queries).
        Fills ``diagnostics`` and ``trace`` in-place when provided.
        """
        if cfg is None:
            cfg = self.config

        if not query_tokens:
            return results

        long_tokens = frozenset(t for t in query_tokens if len(t) >= 4)

        lex_scores: list[float] = []
        score_signals: list[dict] = []  # per-result signal breakdown for trace
        for r in results:
            meta = r.get("metadata", {})
            if meta.get("source_class") != "code":
                lex_scores.append(0.0)
                score_signals.append({})
                continue

            score = 0.0
            signals: dict = {}
            sym_name = (meta.get("symbol_name") or "").lower()
            sym_type = (meta.get("symbol_type") or "").lower()
            file_path = meta.get("file_path") or meta.get("source") or ""
            basename = os.path.splitext(os.path.basename(file_path))[0].lower()
            qualified = f"{basename}.{sym_name}" if sym_name and basename else ""
            text_lower = r.get("text", "").lower()

            # Exact symbol name match
            if sym_name and sym_name in query_tokens:
                score += 1.0
                signals["symbol_hit"] = "exact"
            # Partial symbol name match
            elif sym_name:
                for tok in long_tokens:
                    if tok in sym_name:
                        score += 0.5
                        signals["symbol_hit"] = "partial"
                        break

            # Basename match
            if basename and basename in query_tokens:
                score += 0.4
                signals["file_hit"] = True

            # Qualified name match
            if qualified and qualified in query_tokens:
                score += 1.0
                signals["qualified_hit"] = True

            # Token-in-text hits (capped)
            text_hits = sum(1 for tok in long_tokens if tok in text_lower)
            score += min(text_hits * 0.08, 0.32)
            if text_hits:
                signals["text_hits"] = text_hits

            # Multiplier for function/method results that matched anything
            if score > 0.0 and sym_type in {"function", "method"}:
                score *= 1.1
                signals["sym_type_boost"] = True

            # Line-range tightness tie-breaker: reward narrowly scoped chunks.
            # +0.05 for ≤30 lines, +0.02 for ≤80 lines (secondary signal only).
            start_line = meta.get("start_line")
            end_line = meta.get("end_line")
            if start_line is not None and end_line is not None:
                span = max(int(end_line) - int(start_line), 1)
                if span <= 30:
                    score += 0.05
                    signals["line_range_tight"] = span
                elif span <= 80:
                    score += 0.02

            lex_scores.append(score)
            score_signals.append(signals)

        max_lex = max(lex_scores) if lex_scores else 0.0
        if max_lex == 0.0:
            # No matches — skip re-scoring entirely
            return results

        if diagnostics is not None:
            diagnostics.boost_applied = True

        # Blend: normalize lex scores then mix with original score
        boosted: list[dict] = []
        for r, lex, signals in zip(results, lex_scores, score_signals):
            norm_lex = lex / max_lex
            orig = r.get("score", 0.0)
            r = dict(r)
            final = orig * 0.45 + norm_lex * 0.55
            r["score"] = final
            if trace is not None and signals:
                trace.per_result_score_breakdown.append(
                    {
                        "id": r.get("id", ""),
                        "orig_score": orig,
                        "final_score": final,
                        "lex_score": lex,
                        **signals,
                    }
                )
            boosted.append(r)

        # Per-file diversity cap
        cap = cfg.code_max_chunks_per_file
        seen_files: dict[str, int] = {}
        diverse: list[dict] = []
        deferred: list[dict] = []
        for r in sorted(boosted, key=lambda x: x["score"], reverse=True):
            fp = r.get("metadata", {}).get("file_path") or r.get("metadata", {}).get("source", "")
            count = seen_files.get(fp, 0)
            if count < cap:
                diverse.append(r)
                seen_files[fp] = count + 1
            else:
                deferred.append(r)

        if trace is not None:
            trace.diversity_cap_deferrals = len(deferred)
            trace.deferred_chunk_ids = [r.get("id", "") for r in deferred]

        return diverse + deferred

    def _symbol_channel_search(
        self,
        query_tokens: frozenset,
        top_k: int,
        filters: dict = None,
    ) -> list[dict]:
        """Dedicated symbol_name + qualified_name retrieval channel.

        Scans the in-memory BM25 corpus for chunks whose ``symbol_name`` or
        ``qualified_name`` metadata field exactly (or partially) matches any
        query token.  Returns scored result dicts with ``metadata.channel``
        set to ``"symbol_name"`` or ``"qualified_name"``.

        Handles both ``BM25Retriever`` (single corpus) and
        ``MultiBM25Retriever`` (fan-out across projects).
        """
        if not self.bm25 or not query_tokens:
            return []

        long_tokens = frozenset(t for t in query_tokens if len(t) >= 3)
        if not long_tokens:
            return []

        # Collect all corpora (handles single and multi-project fan-out)
        if hasattr(self.bm25, "_retrievers"):
            corpora = [r.corpus for r in self.bm25._retrievers if hasattr(r, "corpus")]
        elif hasattr(self.bm25, "corpus"):
            corpora = [self.bm25.corpus]
        else:
            return []

        hits: dict[str, dict] = {}  # id → best result
        for corpus in corpora:
            for doc in corpus:
                meta = doc.get("metadata", {})
                sym = (meta.get("symbol_name") or "").lower()
                qname = (meta.get("qualified_name") or "").lower()
                if not sym and not qname:
                    continue

                # Apply metadata filters if provided
                if filters:
                    match = all(meta.get(k) == v for k, v in filters.items())
                    if not match:
                        continue

                # Score: exact match = 1.0, partial = 0.6
                best_score = 0.0
                best_channel = "symbol_name"
                if sym:
                    if sym in long_tokens:
                        best_score = 1.0
                        best_channel = "symbol_name"
                    elif any(tok in sym for tok in long_tokens):
                        best_score = max(best_score, 0.6)
                        best_channel = "symbol_name"
                if qname:
                    if qname in long_tokens:
                        if 1.0 >= best_score:
                            best_score = 1.0
                            best_channel = "qualified_name"
                    elif any(tok in qname for tok in long_tokens):
                        if 0.6 > best_score:
                            best_score = 0.6
                            best_channel = "qualified_name"

                if best_score == 0.0:
                    continue

                doc_id = doc.get("id", "")
                if doc_id not in hits or best_score > hits[doc_id]["score"]:
                    result = dict(doc)
                    result["score"] = best_score
                    result["metadata"] = dict(meta)
                    result["metadata"]["channel"] = best_channel
                    hits[doc_id] = result

        ranked = sorted(hits.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    def _expand_with_code_graph(
        self, query: str, results: list[dict], cfg=None
    ) -> tuple[list[dict], list[str]]:
        """Expand retrieval results using structural code graph traversal.

        Matches query tokens against code node names, then follows CONTAINS,
        IMPORTS, and MENTIONED_IN edges to fetch related chunks.
        """
        nodes: dict = self._code_graph.get("nodes", {})
        edges_list: list = self._code_graph.get("edges", [])
        if not nodes:
            return results, []

        # Build edge index
        outgoing: dict[str, list[tuple[str, str]]] = {}
        incoming: dict[str, list[tuple[str, str]]] = {}
        for edge in edges_list:
            src, tgt, et = edge["source"], edge["target"], edge["edge_type"]
            outgoing.setdefault(src, []).append((tgt, et))
            incoming.setdefault(tgt, []).append((src, et))

        # Extract tokens from query (identifier-like words, >= 3 chars)
        query_tokens: set[str] = set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_.]{2,}\b", query))

        # Match against node names
        matched_node_ids: set[str] = set()
        matched_names: list[str] = []
        for node_id, node in nodes.items():
            name = node.get("name", "")
            if name in query_tokens or any(t in name for t in query_tokens if len(t) >= 4):
                matched_node_ids.add(node_id)
                if name not in matched_names:
                    matched_names.append(name)

        # Also match file_path from already-retrieved results
        for r in results:
            fp = r.get("metadata", {}).get("file_path", "")
            if fp and fp in nodes:
                matched_node_ids.add(fp)

        if not matched_node_ids:
            return results, []

        # Collect extra chunk_ids via 1-hop traversal
        already_ids: set[str] = {r["id"] for r in results}
        extra_chunk_ids: set[str] = set()

        for node_id in list(matched_node_ids):
            # Own chunk_ids
            for cid in nodes[node_id].get("chunk_ids", []):
                extra_chunk_ids.add(cid)
            # Outgoing: IMPORTS → target file chunks; CONTAINS → symbol chunks; MENTIONED_IN → prose chunks
            for tgt_id, et in outgoing.get(node_id, []):
                if et in ("IMPORTS", "CONTAINS", "MENTIONED_IN"):
                    tgt_node = nodes.get(tgt_id)
                    if tgt_node:
                        for cid in tgt_node.get("chunk_ids", []):
                            extra_chunk_ids.add(cid)
            # Incoming CONTAINS: if we matched a symbol, also get 2 chunks from its parent file
            for src_id, et in incoming.get(node_id, []):
                if et == "CONTAINS":
                    src_node = nodes.get(src_id)
                    if src_node:
                        for cid in src_node.get("chunk_ids", [])[:2]:
                            extra_chunk_ids.add(cid)

        extra_chunk_ids -= already_ids
        if not extra_chunk_ids:
            return results, matched_names

        budget = getattr(cfg, "graph_rag_budget", 3)
        fetch_ids = list(extra_chunk_ids)[: budget * 2]

        try:
            extra_docs = self.vector_store.get_by_ids(fetch_ids)
            for doc in extra_docs:
                if doc["id"] not in already_ids:
                    doc.setdefault("score", 0.5)
                    doc["_code_graph_expanded"] = True
                    results.append(doc)
                    already_ids.add(doc["id"])
                    if len(results) > getattr(cfg, "top_k", 10) + budget:
                        break
        except Exception:
            pass

        return results, matched_names
