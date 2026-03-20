"""
Core engine for Axon - Open Source RAG Interface.
"""

# Suppress TensorFlow/Keras noise before any imports that might trigger them
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("USE_TF", "0")  # tell transformers to skip TF backend
# Disable ChromaDB / PostHog telemetry (avoids atexit noise on Ctrl+C)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

import logging  # noqa: E402
import re  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

# Load environment variables — project .env first, then user-global ~/.axon/.env
load_dotenv()
_user_env = Path.home() / ".axon" / ".env"
if _user_env.exists():
    load_dotenv(_user_env)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Axon")

# ---------------------------------------------------------------------------
# Extracted submodule imports — Phase 2 refactor
# ---------------------------------------------------------------------------
from axon.config import _USER_CONFIG_PATH, AxonConfig  # noqa: E402,F401
from axon.embeddings import _KNOWN_DIMS, OpenEmbedding  # noqa: E402,F401
from axon.llm import (  # noqa: E402,F401
    _COPILOT_MODELS_FALLBACK,
    _COPILOT_OAUTH_CLIENT_ID,
    _COPILOT_SESSION_REFRESH_BUFFER,
    OpenLLM,
    _copilot_bridge_lock,
    _copilot_device_flow,
    _copilot_responses,
    _copilot_task_queue,
    _fetch_copilot_models,
    _get_copilot_session_token,
    _refresh_copilot_session,
)
from axon.rerank import OpenReranker  # noqa: E402,F401
from axon.vector_store import (  # noqa: E402,F401
    _MERGED_VIEW_WRITE_ERROR,
    MultiBM25Retriever,
    MultiVectorStore,
    OpenVectorStore,
)

# GraphRAG reduce-phase system prompt (GAP 1)
_GRAPHRAG_REDUCE_SYSTEM_PROMPT = (
    "You are a helpful assistant responding to questions about a dataset by synthesizing the "
    "perspectives from multiple data analysts.\n\n"
    "Generate a response of the target length and format that responds to the user's question, "
    "summarize all the reports from multiple analysts who focused on different parts of the "
    "dataset.\n\n"
    "Note that the analysts' reports provided below are ranked in the importance of addressing "
    "the user's question.\n\n"
    "If you don't know the answer or if the input reports do not contain sufficient information "
    "to provide an answer, just say so. Do not make anything up.\n\n"
    "The response should preserve the original meaning and use of modal verbs such as 'shall', "
    "'may' or 'will'.\n\n"
    "Points supported by data should list their data references as follows:\n"
    '"This is an example sentence supported by multiple data references [Data: Reports (report1), (report2)]."\n\n'
    "**Do not list more than 5 data references in a single reference**. Instead, list the top 5 "
    'most relevant references and add "+more" to indicate that there are more.\n\n'
    "Everything needs to be explained at length and in detail, with specific quotes and passages "
    "from the data providing evidence to back up each and every claim."
)

_GRAPHRAG_NO_DATA_ANSWER = (
    "I am sorry but I am unable to answer this question given the provided data."
)

# Route profiles for the multi-class query router (Option B).
# Each key maps to a set of AxonConfig field overrides applied to the per-request config copy.
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
# Code-query lexical boost — module-level utilities
# ---------------------------------------------------------------------------

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

    diagnostics_version: str = "1.0"
    code_mode_triggered: bool = False
    tokens_extracted: list = field(default_factory=list)
    channels_activated: list = field(default_factory=list)
    result_count: int = 0
    boost_applied: bool = False
    fallback_chunks_in_results: int = 0

    def to_dict(self) -> dict:
        return {
            "diagnostics_version": self.diagnostics_version,
            "code_mode_triggered": self.code_mode_triggered,
            "tokens_extracted": list(self.tokens_extracted),
            "channels_activated": list(self.channels_activated),
            "result_count": self.result_count,
            "boost_applied": self.boost_applied,
            "fallback_chunks_in_results": self.fallback_chunks_in_results,
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


# ---------------------------------------------------------------------------


class AxonBrain:
    """Core RAG engine that wires together embedding, vector store, BM25, reranker, and LLM.

    Instantiate with an :class:`AxonConfig` (or omit to load from the default
    config path).  Call :meth:`ingest` to add documents and :meth:`query` to
    retrieve and synthesise answers.  Use :meth:`switch_project` to change the
    active knowledge-base namespace at runtime.
    """

    SYSTEM_PROMPT = """You are the 'Axon', a highly capable and friendly AI assistant.
Your primary goal is to help the user by answering questions based on the provided context from their private documents.

**Guidelines:**
1. **Prioritize Context**: If relevant information is found in the provided context, use it to answer the question accurately.
2. **Mandatory Citations**: ALWAYS cite your sources. When using information from the context, cite it inline using the document label exactly as shown (e.g. [Document 1 (ID: ...)]). If using information from a Web Search result, cite it as [Web Result] and include the source URL. Place the citation immediately after the relevant sentence or fact.
3. **General Knowledge Fallback**: If no relevant information is found in the context, DO NOT strictly refuse to answer. Instead, use your broad internal knowledge to provide a helpful response.
4. **Be Transparent**: If you are using your general knowledge because no local documents matched the query, briefly mention it (e.g., 'I couldn't find specific details in your documents, but based on my general knowledge...').
5. **Agentic & Proactive**: Be helpful, concise, and encourage further discussion or ingestion of more data if needed.
6. **No emoji**: Do not use emoji in your responses. Plain text only.
"""

    SYSTEM_PROMPT_STRICT = """You are the 'Axon', a focused AI assistant that answers ONLY from the provided document context.

**Guidelines:**
1. **Context Only**: Answer exclusively from the provided context. Do NOT use general knowledge or information outside the documents.
2. **No Match — Say So**: If the context does not contain relevant information to answer the question, respond with: "I don't have relevant information in my documents to answer that."
3. **Mandatory Citations**: ALWAYS cite your sources. Reference the relevant document or section inline using the document label exactly as shown (e.g. [Document 1 (ID: ...)]). If information comes from a Web Search result, cite it as [Web Result] and include the source URL.
4. **No Speculation**: Do not infer, guess, or fill gaps with outside knowledge.
5. **No emoji**: Do not use emoji in your responses. Plain text only.
"""

    def _resolve_model_path(self, model_name: str, kind: str = "hf") -> str:
        """Resolve a HuggingFace model ID to a local path when offline_mode or local_assets_only is on.

        If the name is already an absolute path or starts with '.' it is returned
        unchanged.  Otherwise the short name and org--name variants are looked up
        across the configured model roots in priority order.

        kind:
            "embedding"  — checks embedding_models_dir before local_models_dir
            "hf"         — checks hf_models_dir before local_models_dir
        """
        if os.path.isabs(model_name) or model_name.startswith("."):
            return model_name

        # Build root list in priority order for this kind
        roots: list[str] = []
        if kind == "embedding" and self.config.embedding_models_dir:
            roots.append(self.config.embedding_models_dir)
        elif kind == "hf" and self.config.hf_models_dir:
            roots.append(self.config.hf_models_dir)
        if self.config.local_models_dir:
            roots.append(self.config.local_models_dir)

        if not roots:
            return model_name

        short_name = model_name.split("/")[-1]
        hf_name = model_name.replace("/", "--")

        for base in roots:
            for candidate_name in (short_name, hf_name):
                candidate = os.path.join(base, candidate_name)
                if os.path.isdir(candidate):
                    return candidate

        logger.warning(
            "Local model resolution: '%s' not found in %s (tried '%s' and '%s')",
            model_name,
            roots,
            short_name,
            hf_name,
        )
        return model_name

    def _preflight_model_audit(self) -> None:
        """Log the source classification for every model asset and fail fast when
        ``local_assets_only`` is enabled but a model still resolves to a remote ID.

        Source kinds:
          local_path         — absolute path that exists on disk
          local_path_missing — absolute path that does NOT exist (misconfigured)
          hf_cache           — bare HF model ID present in the local HF hub cache
          remote_id          — bare HF model ID with no local copy found
          n/a                — feature disabled, model will never be loaded
        """
        cfg = self.config

        def _hf_cache_dir() -> str:
            return os.path.join(
                os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                "hub",
            )

        def _classify(path: str) -> str:
            if not path:
                return "n/a"
            if os.path.isabs(path) or path.startswith("."):
                return "local_path" if os.path.isdir(path) else "local_path_missing"
            # Check HF disk cache: models--org--name directory
            hf_slug = "models--" + path.replace("/", "--")
            if os.path.isdir(os.path.join(_hf_cache_dir(), hf_slug)):
                return "hf_cache"
            return "remote_id"

        # Build the audit table — only include helper models when their feature is active
        _gliner_active = cfg.graph_rag and cfg.graph_rag_ner_backend == "gliner"
        _rebel_active = cfg.graph_rag and cfg.graph_rag_relation_backend == "rebel"
        _llmlingua_active = cfg.compress_context

        rows: list[tuple[str, str]] = [
            ("embedding", cfg.embedding_model),
            ("reranker", cfg.reranker_model if cfg.rerank else ""),
            ("gliner", cfg.graph_rag_gliner_model if _gliner_active else ""),
            ("rebel", cfg.graph_rag_rebel_model if _rebel_active else ""),
            ("llmlingua", cfg.graph_rag_llmlingua_model if _llmlingua_active else ""),
            (
                "tokenizer",
                cfg.tokenizer_cache_dir or os.getenv("TIKTOKEN_CACHE_DIR", ""),
            ),
        ]

        _KIND_LABEL = {
            "local_path": "[local]        ",
            "local_path_missing": "[MISSING]      ",
            "hf_cache": "[hf_cache]     ",
            "remote_id": "[remote]       ",
            "n/a": "[n/a]          ",
        }

        lines: list[str] = []
        problems: list[str] = []
        for name, path in rows:
            kind = _classify(path)
            label = _KIND_LABEL[kind]
            display = path if path else "(disabled)"
            lines.append(f"  {label} {name:<12} {display}")
            # Fail-fast candidates: active models that are not on local disk
            if cfg.local_assets_only and path and kind in ("remote_id", "local_path_missing"):
                problems.append(f"{name}: {display!r} ({kind})")

        logger.info("Model asset audit:\n%s", "\n".join(lines))

        if problems:
            raise RuntimeError(
                "local_assets_only is ON but the following model assets are not available locally:\n"
                + "".join(f"  - {p}\n" for p in problems)
                + "Set embedding_models_dir / hf_models_dir in config.yaml, "
                "or provide absolute paths to pre-downloaded checkpoints."
            )

    def __init__(self, config: AxonConfig | None = None):
        self.config = config or AxonConfig.load()

        # ── Local-assets-only: enforce local model files without disabling RAPTOR/GraphRAG ──
        if self.config.local_assets_only:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            if self.config.truth_grounding:
                logger.info("Local-assets-only: disabling web search (truth_grounding → OFF)")
                self.config.truth_grounding = False
            # Resolve all model IDs to local paths — RAPTOR + GraphRAG remain enabled
            self.config.embedding_model = self._resolve_model_path(
                self.config.embedding_model, "embedding"
            )
            self.config.reranker_model = self._resolve_model_path(self.config.reranker_model, "hf")
            self.config.graph_rag_gliner_model = self._resolve_model_path(
                self.config.graph_rag_gliner_model, "hf"
            )
            self.config.graph_rag_rebel_model = self._resolve_model_path(
                self.config.graph_rag_rebel_model, "hf"
            )
            self.config.graph_rag_llmlingua_model = self._resolve_model_path(
                self.config.graph_rag_llmlingua_model, "hf"
            )
            logger.info(
                "Local-assets-only ON  |  hf_models_dir: %s  |  embedding_models_dir: %s",
                self.config.hf_models_dir or "(not set)",
                self.config.embedding_models_dir or "(not set)",
            )

        # ── Offline mode: lock down all network access before any model loads ──
        if self.config.offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            if self.config.truth_grounding:
                logger.info("Offline mode: disabling web search (truth_grounding → OFF)")
                self.config.truth_grounding = False
            if self.config.raptor:
                logger.warning(
                    "Offline mode: RAPTOR requires LLM calls and is not supported offline. "
                    "Disabling raptor for this session."
                )
                self.config.raptor = False
            if self.config.graph_rag:
                logger.warning(
                    "Offline mode: GraphRAG requires LLM calls and is not supported offline. "
                    "Disabling graph_rag for this session."
                )
                self.config.graph_rag = False
            # Resolve bare HF model IDs to local paths
            self.config.embedding_model = self._resolve_model_path(
                self.config.embedding_model, "embedding"
            )
            self.config.reranker_model = self._resolve_model_path(self.config.reranker_model, "hf")
            logger.info(
                "Offline mode ON  |  models dir: %s",
                self.config.local_models_dir or "(not set)",
            )

        # Apply tiktoken cache directory if configured (Phase 4 tokenizer locality)
        if self.config.tokenizer_cache_dir and not os.getenv("TIKTOKEN_CACHE_DIR"):
            os.environ["TIKTOKEN_CACHE_DIR"] = self.config.tokenizer_cache_dir
            logger.info("Tiktoken cache dir: %s", self.config.tokenizer_cache_dir)

        # ── Phase 5: startup preflight audit ──
        # Logs the source classification for every model asset (local / hf_cache / remote).
        # Raises RuntimeError early when local_assets_only is ON but assets are missing.
        self._preflight_model_audit()

        # Apply Ollama model directory override before any Ollama client is constructed.
        # Sets OLLAMA_MODELS so the Ollama daemon resolves blobs from the given path.
        if self.config.ollama_models_dir and not os.getenv("OLLAMA_MODELS"):
            os.environ["OLLAMA_MODELS"] = self.config.ollama_models_dir
            logger.info("Ollama models dir: %s", self.config.ollama_models_dir)

        # Apply custom projects root from config before any project operations
        if self.config.projects_root:
            from axon import projects as _proj_mod

            _proj_mod.set_projects_root(self.config.projects_root)
            logger.info(f"Projects root: {_proj_mod.PROJECTS_ROOT}")

        # In AxonStore mode, ensure the user namespace directories exist
        if self.config.axon_store_mode:
            from axon.projects import ensure_user_namespace

            ensure_user_namespace(Path(self.config.projects_root))

        # Ensure the 'default' project directory exists
        from axon.projects import ensure_project

        ensure_project("default")

        logger.info("Initializing Axon...")
        self.embedding = OpenEmbedding(self.config)
        self.llm = OpenLLM(self.config)
        self.vector_store = OpenVectorStore(self.config)
        self.reranker = OpenReranker(self.config)

        # Stash original (config.yaml) paths so we can restore them for "default"
        self._base_vector_store_path: str = os.path.abspath(self.config.vector_store_path)
        self._base_bm25_path: str = os.path.abspath(self.config.bm25_path)
        self._active_project: str = "default"
        self._read_only_scope: bool = False
        self._mounted_share: bool = False  # legacy alias — use _active_project_kind
        self._active_project_kind: str = "default"  # "default" | "local" | "mounted" | "scope"
        self._active_mount_descriptor: dict | None = None

        try:
            from axon.retrievers import BM25Retriever

            self.bm25 = BM25Retriever(storage_path=self.config.bm25_path)
        except ImportError:
            self.bm25 = None

        # Write-path stores: always point to the active project's own data dir.
        # When a parent project is active, self.vector_store / self.bm25 become
        # Multi* fan-out wrappers (read-only views across all descendants), while
        # _own_vector_store / _own_bm25 keep pointing to this project's own dir
        # so ingest(), dedup, and GraphRAG always write to the right place.
        self._own_vector_store: OpenVectorStore = self.vector_store
        self._own_bm25 = self.bm25

        try:
            if self.config.chunk_strategy == "semantic":
                from axon.splitters import SemanticTextSplitter

                self.splitter = SemanticTextSplitter(
                    chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
                )
            elif self.config.chunk_strategy == "markdown":
                from axon.splitters import MarkdownSplitter

                self.splitter = MarkdownSplitter(
                    chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
                )
            elif self.config.chunk_strategy == "cosine_semantic":
                from axon.splitters import CosineSemanticSplitter

                self.splitter = CosineSemanticSplitter(
                    embed_fn=self.embedding.embed,
                    breakpoint_threshold=self.config.cosine_semantic_threshold,
                    max_chunk_size=self.config.cosine_semantic_max_size,
                )
            else:
                from axon.splitters import RecursiveCharacterTextSplitter

                self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
                )
        except ImportError:
            self.splitter = None

        # In-memory query result cache (keyed by hash of query + filters + active flags).
        # Uses OrderedDict for true LRU eviction: hits are moved to the end so the
        # least-recently-used entry is always at the front and evicted first.
        from collections import OrderedDict

        self._query_cache: OrderedDict[str, str] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._last_diagnostics: CodeRetrievalDiagnostics = CodeRetrievalDiagnostics()

        # Content hash store for ingest deduplication
        self._ingested_hashes: set = self._load_hash_store()

        # Doc versions store for smart re-ingest
        self._doc_versions: dict = {}
        self._doc_versions_path = os.path.join(self.config.bm25_path, ".doc_versions.json")
        self._load_doc_versions()

        # GraphRAG entity → doc_id mapping (entity name -> list of chunk IDs)
        self._entity_graph: dict[str, list[str]] = self._load_entity_graph()

        # Structural code graph: {nodes: {node_id: {...}}, edges: [{source, target, edge_type, chunk_id}]}
        self._code_graph: dict = self._load_code_graph()

        # GraphRAG relation graph: {source_entity_lower: [{target, relation, chunk_id, description}]}
        self._relation_graph: dict = self._load_relation_graph()

        # GraphRAG community detection state
        self._community_levels: dict = self._load_community_levels()
        self._community_summaries: dict = self._load_community_summaries()
        self._community_graph_dirty: bool = False
        self._community_build_in_progress: bool = False
        self._last_matched_entities: list = []

        # GraphRAG entity embeddings (for embedding-based entity matching at query time)
        self._entity_embeddings: dict = self._load_entity_embeddings()

        # GraphRAG entity description buffer (transient, not persisted)
        self._entity_description_buffer: dict = {}

        # P5: In-memory RAPTOR summary cache {cache_key: summary_text}
        self._raptor_summary_cache: dict[str, str] = {}

        # GraphRAG claims graph
        self._claims_graph: dict = self._load_claims_graph()

        # GAP 2: Hierarchical community parent/child structure
        self._community_hierarchy: dict[int, int] = self._load_community_hierarchy()
        self._community_children: dict[int, list[int]] = {}
        # Rebuild children from hierarchy on load
        for child_cid, parent_cid in self._community_hierarchy.items():
            if parent_cid is not None:
                if parent_cid not in self._community_children:
                    self._community_children[parent_cid] = []
                if child_cid not in self._community_children[parent_cid]:
                    self._community_children[parent_cid].append(child_cid)

        # GAP 3b: Relation description buffer (transient, not persisted)
        self._relation_description_buffer: dict = (
            {}
        )  # (subject_lower, object_lower) -> [descriptions]

        # GAP 6: Rebuild lock to prevent concurrent community rebuilds
        self._community_rebuild_lock: threading.Lock = threading.Lock()

        # GAP 9: TextUnit reverse maps (in-memory only, rebuilt during ingest)
        self._text_unit_entity_map: dict[str, list[str]] = {}  # chunk_id -> [entity_names]
        self._text_unit_relation_map: dict[
            str, list[tuple[str, str]]
        ] = {}  # chunk_id -> [(src, tgt)]

        # Shared executor for background/parallel tasks
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        self._log_startup_summary()

    def close(self):
        """Explicitly release all resources (connections, file handles)."""
        if hasattr(self, "_executor") and self._executor:
            self._executor.shutdown(wait=False)
        if hasattr(self, "vector_store") and self.vector_store:
            if hasattr(self.vector_store, "close"):
                self.vector_store.close()
        if hasattr(self, "bm25") and self.bm25:
            if hasattr(self.bm25, "close"):
                self.bm25.close()
        if hasattr(self, "_own_vector_store") and self._own_vector_store:
            if hasattr(self._own_vector_store, "close"):
                self._own_vector_store.close()
        if hasattr(self, "_own_bm25") and self._own_bm25:
            if hasattr(self._own_bm25, "close"):
                self._own_bm25.close()

    def _log_startup_summary(self) -> None:
        """Log a one-line startup summary: active project, namespace ID, scope type."""
        from axon.projects import get_project_namespace_id

        scope = "read-only merged" if getattr(self, "_read_only_scope", False) else "authoritative"
        ns_id = get_project_namespace_id(self._active_project)
        if ns_id is None:
            # Try reading from the actual project dir (for default which maps to config path)
            try:
                pdir = Path(self.config.bm25_path).parent
                meta_file = pdir / "meta.json"
                if meta_file.exists():
                    import json as _json

                    meta = _json.loads(meta_file.read_text())
                    ns_id = meta.get("project_namespace_id", "none")
            except Exception:
                ns_id = "none"
        logger.info(
            "Axon ready  |  project: %s  |  ns: %s  |  scope: %s",
            self._active_project,
            ns_id or "none",
            scope,
        )

    def should_recommend_project(self) -> bool:
        """Return True if we should recommend creating a dedicated project.

        True if the active project is 'default' AND no OTHER named projects exist yet.
        """
        if self._active_project != "default":
            return False
        from axon.projects import list_projects

        try:
            # list_projects includes 'default' if it was ensured.
            # We only recommend if NO OTHER projects exist.
            projects = list_projects()
            named_projects = [p for p in projects if p["name"] != "default"]
            return len(named_projects) == 0
        except Exception:
            return False

    # Reserved directory names excluded from @projects / @store scope merges
    _SCOPE_RESERVED_DIRS: frozenset = frozenset({"default", "mounts", ".shares"})

    def _switch_to_scope(self, scope: str) -> None:
        """Switch to a merged read-only scope (@projects, @mounts, or @store).

        Collects vector stores and BM25 indices from the relevant project dirs,
        wraps them in MultiVectorStore / MultiBM25Retriever, and marks the brain
        as read-only so ingest() raises a clear error.

        Args:
            scope: One of "@projects", "@mounts", or "@store".

        Raises:
            ValueError: If no projects are found for the requested scope.
        """
        import copy

        from axon.projects import PROJECTS_ROOT, project_bm25_path, project_vector_path
        from axon.retrievers import BM25Retriever as _BM25

        _valid_scopes = {"@projects", "@mounts", "@store"}
        if scope not in _valid_scopes:
            raise ValueError(
                f"Unknown scope '{scope}'. Valid scopes: {', '.join(sorted(_valid_scopes))}"
            )

        # ── Collect authoritative project dirs (exclude reserved names) ──────
        def _authoritative_project_dirs() -> list[Path]:
            dirs = []
            if not PROJECTS_ROOT.exists():
                return dirs
            for entry in sorted(PROJECTS_ROOT.iterdir()):
                if not entry.is_dir():
                    continue
                if entry.name in self._SCOPE_RESERVED_DIRS:
                    continue
                if not (entry / "meta.json").exists():
                    continue
                dirs.append(entry)
            return dirs

        # ── Collect mount dirs ────────────────────────────────────────────────
        def _mount_dirs() -> list[tuple[str, str]]:
            """Return (vector_path, bm25_path) pairs from active mount descriptors."""
            from axon.mounts import list_mount_descriptors

            paths = []
            for desc in list_mount_descriptors(PROJECTS_ROOT):
                target = desc.get("target_project_dir", "")
                if target and Path(target).exists():
                    paths.append(
                        (
                            str(Path(target) / "chroma_data"),
                            str(Path(target) / "bm25_index"),
                        )
                    )
            return paths

        # ── Build the list of (vector_path, bm25_path) pairs for the scope ───
        project_paths: list[tuple[str, str]] = []

        if scope in ("@projects", "@store"):
            for pdir in _authoritative_project_dirs():
                pname = pdir.name
                try:
                    vpath = project_vector_path(pname)
                    bpath = project_bm25_path(pname)
                    project_paths.append((vpath, bpath))
                except Exception as e:
                    logger.warning("Scope %s: skipping project '%s': %s", scope, pname, e)

        if scope == "@store":
            # Include the default project
            project_paths.insert(0, (self._base_vector_store_path, self._base_bm25_path))

        # @mounts and @store — descriptor-backed mounted projects
        if scope in ("@mounts", "@store"):
            for vpath, bpath in _mount_dirs():
                project_paths.append((vpath, bpath))

        if not project_paths:
            raise ValueError(f"No authoritative projects found under {scope} scope.")

        # ── Close existing stores and rebuild executor ────────────────────────
        self.close()
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # ── Build Multi* wrappers ─────────────────────────────────────────────
        all_vs = []
        all_bm25 = []
        for vpath, bpath in project_paths:
            cfg = copy.copy(self.config)
            cfg.vector_store_path = vpath
            cfg.bm25_path = bpath
            all_vs.append(OpenVectorStore(cfg))
            try:
                all_bm25.append(_BM25(storage_path=bpath))
            except Exception:
                pass

        self.vector_store = MultiVectorStore(all_vs)
        self.bm25 = MultiBM25Retriever(all_bm25) if all_bm25 else None
        # Scope stores are read-only; own store points to first real store for
        # diagnostic methods that inspect self._own_vector_store.
        self._own_vector_store = all_vs[0]
        self._own_bm25 = all_bm25[0] if all_bm25 else None

        # ── Reload project state (use base paths for GraphRAG state) ─────────
        from collections import OrderedDict

        with self._cache_lock:
            self._query_cache = OrderedDict()
        self._ingested_hashes = set()
        self._entity_graph = {}
        self._relation_graph = {}
        self._community_levels = {}
        self._community_summaries = {}
        self._entity_embeddings = {}
        self._entity_description_buffer = {}
        self._claims_graph = {}
        self._community_graph_dirty = False
        self._community_hierarchy = {}
        self._community_children = {}
        self._relation_description_buffer = {}
        self._text_unit_entity_map = {}
        self._text_unit_relation_map = {}
        self._raptor_summary_cache = {}

        self._active_project = scope
        self._read_only_scope = True
        self._active_project_kind = "scope"
        self._active_mount_descriptor = None
        self._mounted_share = False
        logger.info(
            "Switched to %s scope  |  %d store(s) merged  |  read-only",
            scope,
            len(all_vs),
        )

    def _is_mounted_share(self) -> bool:
        """Return True if the active project is a received share mount (always read-only).

        Uses descriptor-based kind state when available; falls back to legacy
        ``_mounted_share`` flag set by older code paths.
        """
        kind = getattr(self, "_active_project_kind", None)
        if kind is not None:
            return kind == "mounted"
        return getattr(self, "_mounted_share", False)

    def _assert_write_allowed(self, operation: str = "write") -> None:
        """Raise PermissionError if current project is read-only (scope, mounted share, or maintenance state)."""
        from axon.access import check_write_allowed

        check_write_allowed(
            operation,
            self._active_project,
            getattr(self, "_read_only_scope", False),
            self._is_mounted_share(),
        )

    def switch_project(self, name: str) -> None:
        """Switch the active project, reinitializing vector store and BM25.

        Embedding and LLM are kept (expensive to reload). The "default" sentinel
        restores the paths from config.yaml.

        When switching to a *parent* project (one that has sub-projects), the
        read path (self.vector_store / self.bm25) becomes a Multi* fan-out over
        the parent's own store plus all descendants. The write path
        (self._own_vector_store / self._own_bm25) always points only to the
        parent's own data directory.

        Args:
            name: Project name (slash-separated for sub-projects) or "default".

        Raises:
            ValueError: If the project does not exist (use /project new first).
        """
        _prev_project = self._active_project  # stash for epoch bump below

        from axon.projects import (
            list_descendants,
            project_bm25_path,
            project_dir,
            project_vector_path,
            set_active_project,
        )

        # ── @-scope handling: merged read-only views ──────────────────────────
        if name.startswith("@"):
            return self._switch_to_scope(name)

        # ── Descriptor-backed mounted project (mounts/<mount_name>) ───────────
        if name.startswith("mounts/"):
            mount_name = name[len("mounts/") :]
            from axon.mounts import load_mount_descriptor, validate_mount_descriptor

            user_dir = Path(self.config.projects_root)
            desc = load_mount_descriptor(user_dir, mount_name)
            if desc is None:
                raise ValueError(f"Mounted project '{name}' not found. Redeem the share first.")
            valid, reason = validate_mount_descriptor(desc)
            if not valid:
                raise ValueError(f"Mounted project '{name}' is not accessible: {reason}")
            target = Path(desc["target_project_dir"])
            self.config.vector_store_path = str(target / "chroma_data")
            self.config.bm25_path = str(target / "bm25_index")
        elif name == "default":
            self.config.vector_store_path = self._base_vector_store_path
            self.config.bm25_path = self._base_bm25_path
        else:
            root = project_dir(name)
            if not root.exists():
                raise ValueError(
                    f"Project '{name}' does not exist. Create it first with /project new {name}"
                )
            self.config.vector_store_path = project_vector_path(name)
            self.config.bm25_path = project_bm25_path(name)

        # Close existing stores before replacing them
        self.close()

        # Recreate the executor — close() shuts it down and it cannot be reused
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        # Own store: always the project's own data (used for writes / dedup / GraphRAG)
        own_vs = OpenVectorStore(self.config)
        try:
            from axon.retrievers import BM25Retriever

            own_bm25 = BM25Retriever(storage_path=self.config.bm25_path)
        except ImportError:
            own_bm25 = None

        self._own_vector_store = own_vs
        self._own_bm25 = own_bm25

        # If the project has sub-projects, build fan-out stores for reads
        # Mounted projects are leaf-only; skip descendant lookup for them.
        if name != "default" and not name.startswith("mounts/"):
            descendants = list_descendants(name)
        else:
            descendants = []

        if descendants:
            import copy

            from axon.retrievers import BM25Retriever as _BM25

            child_configs = []
            for desc in descendants:
                child_cfg = copy.copy(self.config)
                child_cfg.vector_store_path = project_vector_path(desc)
                child_cfg.bm25_path = project_bm25_path(desc)
                child_configs.append(child_cfg)

            all_vs = [own_vs] + [OpenVectorStore(cfg) for cfg in child_configs]
            all_bm25 = [own_bm25] if own_bm25 else []
            for cfg in child_configs:
                try:
                    all_bm25.append(_BM25(storage_path=cfg.bm25_path))
                except Exception:
                    pass

            self.vector_store = MultiVectorStore(all_vs)
            self.bm25 = MultiBM25Retriever(all_bm25) if all_bm25 else None
        else:
            # Leaf project or default: read == write
            self.vector_store = own_vs
            self.bm25 = own_bm25

        # Reload project-scoped state so dedup/GraphRAG use the new project's data
        # and cached answers from the previous project cannot bleed across.
        from collections import OrderedDict

        with self._cache_lock:
            self._query_cache = OrderedDict()
        self._ingested_hashes = self._load_hash_store()
        self._entity_graph = self._load_entity_graph()
        self._relation_graph = self._load_relation_graph()
        self._community_levels = self._load_community_levels()
        self._community_summaries = self._load_community_summaries()
        self._entity_embeddings = self._load_entity_embeddings()
        self._entity_description_buffer = {}
        self._claims_graph = self._load_claims_graph()
        self._community_graph_dirty = False
        # GAP 2: reload hierarchy
        self._community_hierarchy = self._load_community_hierarchy()
        self._community_children = {}
        for child_cid, parent_cid in self._community_hierarchy.items():
            if parent_cid is not None:
                if parent_cid not in self._community_children:
                    self._community_children[parent_cid] = []
                if child_cid not in self._community_children[parent_cid]:
                    self._community_children[parent_cid].append(child_cid)
        # GAP 3b, 9: reset transient maps
        self._relation_description_buffer = {}
        self._text_unit_entity_map = {}
        self._text_unit_relation_map = {}
        self._raptor_summary_cache = {}

        # Merge entity graphs and relation graphs from all descendant projects so
        # that GraphRAG expansion is coherent when querying from a parent project.
        if descendants:
            import pathlib

            for desc in descendants:
                desc_bm25_path = project_bm25_path(desc)
                desc_base = pathlib.Path(desc_bm25_path)

                # --- entity graph ---
                desc_graph_path = desc_base / ".entity_graph.json"
                if desc_graph_path.exists():
                    try:
                        import json as _json

                        raw = _json.loads(desc_graph_path.read_text(encoding="utf-8"))
                        if isinstance(raw, dict):
                            for entity, node in raw.items():
                                if not isinstance(entity, str):
                                    continue
                                if not isinstance(node, dict):
                                    continue
                                doc_ids = node.get("chunk_ids", [])
                                if not doc_ids:
                                    continue
                                existing = self._entity_graph.get(entity)
                                if existing is None:
                                    self._entity_graph[entity] = {
                                        "description": node.get("description", ""),
                                        "type": node.get("type", "UNKNOWN"),
                                        "chunk_ids": [d for d in doc_ids if isinstance(d, str)],
                                        "frequency": len(
                                            [d for d in doc_ids if isinstance(d, str)]
                                        ),
                                        "degree": node.get("degree", 0),
                                    }
                                elif isinstance(existing, dict):
                                    existing_ids = set(existing.get("chunk_ids", []))
                                    new_ids = [
                                        d
                                        for d in doc_ids
                                        if isinstance(d, str) and d not in existing_ids
                                    ]
                                    if new_ids:
                                        existing.setdefault("chunk_ids", []).extend(new_ids)
                                        existing["frequency"] = len(existing["chunk_ids"])
                    except Exception as e:
                        logger.warning(f"Could not merge entity graph for '{desc}': {e}")

                # --- relation graph ---
                desc_rel_path = desc_base / ".relation_graph.json"
                if desc_rel_path.exists():
                    try:
                        import json as _json

                        raw = _json.loads(desc_rel_path.read_text(encoding="utf-8"))
                        if isinstance(raw, dict):
                            for src, entries in raw.items():
                                if isinstance(src, str) and isinstance(entries, list):
                                    if src not in self._relation_graph:
                                        self._relation_graph[src] = []
                                    existing = {
                                        (e.get("target"), e.get("relation"), e.get("chunk_id"))
                                        for e in self._relation_graph[src]
                                    }
                                    for entry in entries:
                                        if isinstance(entry, dict):
                                            key = (
                                                entry.get("target"),
                                                entry.get("relation"),
                                                entry.get("chunk_id"),
                                            )
                                            if key not in existing:
                                                self._relation_graph[src].append(entry)
                                                existing.add(key)
                    except Exception as e:
                        logger.warning(f"Could not merge relation graph for '{desc}': {e}")

                # --- entity embeddings ---
                desc_emb_path = desc_base / ".entity_embeddings.json"
                if desc_emb_path.exists():
                    try:
                        raw = _json.loads(desc_emb_path.read_text(encoding="utf-8"))
                        if isinstance(raw, dict):
                            for entity_key, embedding in raw.items():
                                if (
                                    isinstance(entity_key, str)
                                    and entity_key not in self._entity_embeddings
                                ):
                                    self._entity_embeddings[entity_key] = embedding
                    except Exception as e:
                        logger.warning(f"Could not merge entity embeddings for '{desc}': {e}")

                # --- claims ---
                desc_claims_path = desc_base / ".claims_graph.json"
                if desc_claims_path.exists():
                    try:
                        raw = _json.loads(desc_claims_path.read_text(encoding="utf-8"))
                        if isinstance(raw, dict):
                            for chunk_id, claims in raw.items():
                                if isinstance(chunk_id, str) and isinstance(claims, list):
                                    if chunk_id not in self._claims_graph:
                                        self._claims_graph[chunk_id] = []
                                    existing_claim_keys = {
                                        (c.get("subject"), c.get("object"), c.get("type"))
                                        for c in self._claims_graph[chunk_id]
                                    }
                                    for claim in claims:
                                        if isinstance(claim, dict):
                                            key = (
                                                claim.get("subject"),
                                                claim.get("object"),
                                                claim.get("type"),
                                            )
                                            if key not in existing_claim_keys:
                                                self._claims_graph[chunk_id].append(claim)
                                                existing_claim_keys.add(key)
                    except Exception as e:
                        logger.warning(f"Could not merge claims for '{desc}': {e}")

                # --- community summaries (namespaced to avoid community-ID collision) ---
                desc_summ_path = desc_base / ".community_summaries.json"
                if desc_summ_path.exists():
                    try:
                        raw = _json.loads(desc_summ_path.read_text(encoding="utf-8"))
                        if isinstance(raw, dict):
                            for summ_key, summary in raw.items():
                                if isinstance(summ_key, str) and isinstance(summary, dict):
                                    namespaced_key = f"desc_{desc}_{summ_key}"
                                    if namespaced_key not in self._community_summaries:
                                        self._community_summaries[namespaced_key] = dict(summary)
                    except Exception as e:
                        logger.warning(f"Could not merge community summaries for '{desc}': {e}")

        self._active_project = name
        self._read_only_scope = False
        # Resolve kind from switch operation — descriptor-based, not string-pattern
        if name.startswith("mounts/"):
            self._active_project_kind = "mounted"
            # descriptor was already loaded above; store it for write-guard and UI
            mount_name = name[len("mounts/") :]
            from axon.mounts import load_mount_descriptor as _lmd

            self._active_mount_descriptor = _lmd(Path(self.config.projects_root), mount_name)
        elif name == "default":
            self._active_project_kind = "default"
            self._active_mount_descriptor = None
        elif name.startswith("@"):
            self._active_project_kind = "scope"
            self._active_mount_descriptor = None
        else:
            self._active_project_kind = "local"
            self._active_mount_descriptor = None
        self._mounted_share = self._active_project_kind == "mounted"
        set_active_project(name)
        logger.info(f"Switched to project '{name}'")

        # Phase 3: bump epoch on the old project to fence any stale in-flight writers.
        if _prev_project != "default" and not _prev_project.startswith("@"):
            from axon.runtime import get_registry as _get_registry

            _get_registry().bump_epoch(_prev_project)

    def _load_hash_store(self) -> set:
        """Load persisted content hashes for ingest deduplication."""
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".content_hashes"
        if path.exists():
            return set(path.read_text(encoding="utf-8").splitlines())
        return set()

    def _save_hash_store(self) -> None:
        """Persist content hashes to disk."""
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".content_hashes"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._ingested_hashes), encoding="utf-8")

    def _load_doc_versions(self) -> None:
        """Load doc versions from disk."""
        if os.path.exists(self._doc_versions_path):
            try:
                import json as _json

                with open(self._doc_versions_path, encoding="utf-8") as f:
                    self._doc_versions = _json.load(f)
            except Exception as e:
                logger.warning(f"Could not load doc versions: {e}")
                self._doc_versions = {}
        else:
            self._doc_versions = {}

    def _save_doc_versions(self) -> None:
        """Persist doc versions to disk."""
        try:
            import json as _json

            os.makedirs(os.path.dirname(self._doc_versions_path), exist_ok=True)
            with open(self._doc_versions_path, "w", encoding="utf-8") as f:
                _json.dump(self._doc_versions, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save doc versions: {e}")

    def get_doc_versions(self) -> dict:
        """Return all tracked document versions."""
        return dict(self._doc_versions)

    # ------------------------------------------------------------------
    # Embedding model metadata — guards against silent collection corruption
    # when the embedding model is changed after documents have been ingested.
    # ------------------------------------------------------------------

    @property
    def _embedding_meta_path(self) -> str:
        import pathlib

        return str(pathlib.Path(self.config.bm25_path) / ".embedding_meta.json")

    def _load_embedding_meta(self) -> dict | None:
        """Return the persisted embedding meta for this project, or None if absent."""
        import json
        import pathlib

        path = pathlib.Path(self._embedding_meta_path)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _save_embedding_meta(self) -> None:
        """Persist the current embedding provider/model to disk."""
        import json
        import pathlib
        from datetime import datetime, timezone

        try:
            dimension = int(self.embedding.dimension)
        except (TypeError, ValueError):
            dimension = 0

        meta = {
            "embedding_provider": self.config.embedding_provider,
            "embedding_model": self.config.embedding_model,
            "dimension": dimension,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        path = pathlib.Path(self._embedding_meta_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _validate_embedding_meta(self, *, on_mismatch: str = "raise") -> None:
        """Compare the current embedding config against the persisted collection meta.

        Args:
            on_mismatch: ``"raise"`` (default, used before ingest) raises
                ``ValueError`` to prevent silent collection corruption.
                ``"warn"`` logs a warning but allows the operation to continue
                (used at query time so existing data is still accessible).
        """
        meta = self._load_embedding_meta()
        if meta is None:
            return  # New collection — nothing to validate yet

        stored_provider = meta.get("embedding_provider", "")
        stored_model = meta.get("embedding_model", "")
        current_provider = self.config.embedding_provider
        current_model = self.config.embedding_model

        if stored_provider == current_provider and stored_model == current_model:
            return  # All good

        msg = (
            f"Embedding model mismatch: this project's collection was built with "
            f"'{stored_provider}/{stored_model}' "
            f"but the current config uses '{current_provider}/{current_model}'. "
            f"Mixing embedding models corrupts retrieval even when dimensions match. "
            f"To switch models: delete the project data and re-ingest all documents, "
            f"or revert embedding_provider/embedding_model in config.yaml."
        )
        if on_mismatch == "raise":
            raise ValueError(msg)
        else:
            logger.warning(msg)

    def _load_entity_graph(self) -> dict:
        """Load persisted entity→doc_id graph from disk.

        Shape: {entity_lower: {"description": str, "chunk_ids": list[str]}}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_graph.json"
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    return {}
                cleaned: dict = {}
                for key, value in raw.items():
                    if not isinstance(key, str):
                        continue
                    if isinstance(value, dict) and "chunk_ids" in value:
                        node = value
                        node.setdefault("type", "UNKNOWN")
                        node.setdefault("frequency", len(node.get("chunk_ids", [])))
                        node.setdefault("degree", 0)
                        cleaned[key] = node
                    # Otherwise skip malformed entries
                return cleaned
            except Exception:
                pass
        return {}

    def _save_entity_graph(self) -> None:
        """Persist entity graph to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._entity_graph), encoding="utf-8")

    def _load_code_graph(self) -> dict:
        """Load code graph from disk. Returns empty graph if not found."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".code_graph.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "nodes" in data and "edges" in data:
                    return data
            except Exception:
                pass
        return {"nodes": {}, "edges": []}

    def _save_code_graph(self) -> None:
        """Persist code graph to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".code_graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._code_graph), encoding="utf-8")

    def _load_relation_graph(self) -> dict:
        """Load persisted relation graph from disk.

        Shape: {source_entity_lower: [{target: str, relation: str, chunk_id: str}]}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".relation_graph.json"
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    return {}
                cleaned: dict = {}
                for key, value in raw.items():
                    if not isinstance(key, str) or not isinstance(value, list):
                        continue
                    cleaned[key] = [
                        entry
                        for entry in value
                        if isinstance(entry, dict)
                        and "target" in entry
                        and "relation" in entry
                        and "chunk_id" in entry
                    ]
                return cleaned
            except Exception:
                pass
        return {}

    def _save_relation_graph(self) -> None:
        """Persist relation graph to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".relation_graph.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._relation_graph), encoding="utf-8")

    def _load_community_levels(self) -> dict:
        """Load persisted hierarchical community levels from disk.

        Shape: {level_int: {entity_lower: community_id}}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_levels.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return {int(k): v for k, v in raw.items() if isinstance(v, dict)}
        except Exception:
            pass
        return {}

    def _save_community_levels(self) -> None:
        """Persist hierarchical community levels to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_levels.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({str(k): v for k, v in self._community_levels.items()}),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"Could not save community levels: {e}")

    def _load_community_hierarchy(self) -> dict:
        """Load persisted community hierarchy from disk.

        Shape: {cluster_id_int: parent_cluster_id_int_or_None}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_hierarchy.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    # JSON keys are strings; support both legacy int keys and
                    # level-qualified string keys like "1_3".
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
                    return result
        except Exception:
            pass
        return {}

    def _save_community_hierarchy(self) -> None:
        """Persist community hierarchy to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_hierarchy.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({str(k): v for k, v in self._community_hierarchy.items()}),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug(f"Could not save community hierarchy: {e}")

    def _load_community_summaries(self) -> dict:
        """Load persisted community summaries from disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_summaries.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            pass
        return {}

    def _save_community_summaries(self) -> None:
        """Persist community summaries to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_summaries.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._community_summaries), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Could not save community summaries: {e}")

    def _load_entity_embeddings(self) -> dict:
        """Load persisted entity embeddings from disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_embeddings.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            pass
        return {}

    def _save_entity_embeddings(self) -> None:
        """Persist entity embeddings to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_embeddings.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._entity_embeddings), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Could not save entity embeddings: {e}")

    def _load_claims_graph(self) -> dict:
        """Load persisted claims graph from disk.

        Shape: {chunk_id: [claim_dict, ...]}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".claims_graph.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            pass
        return {}

    def _save_claims_graph(self) -> None:
        """Persist claims graph to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".claims_graph.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._claims_graph), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Could not save claims graph: {e}")

    def _build_networkx_graph(self):
        """Build a NetworkX undirected graph from entity and relation data."""
        import networkx as nx

        _min_freq_raw = getattr(self.config, "graph_rag_entity_min_frequency", 1)
        _min_freq = int(_min_freq_raw) if isinstance(_min_freq_raw, int | float) else 1
        G = nx.Graph()
        for entity, node in self._entity_graph.items():
            if isinstance(node, dict) and node.get("frequency", 1) < _min_freq:
                continue
            desc = node.get("description", "") if isinstance(node, dict) else ""
            G.add_node(entity, description=desc)
        for src, entries in self._relation_graph.items():
            for entry in entries:
                tgt = entry.get("target", "")
                if src and tgt:
                    edge_weight = entry.get("weight", 1)
                    if G.has_edge(src, tgt):
                        G[src][tgt]["weight"] += edge_weight
                    else:
                        G.add_edge(
                            src,
                            tgt,
                            weight=edge_weight,
                            relation=entry.get("relation", ""),
                            description=entry.get("description", ""),
                        )
        return G

    def _run_community_detection(self) -> dict:
        """Run Louvain community detection. Returns {entity_lower: community_id}."""
        try:
            import networkx as nx  # noqa: F401 — ensures ImportError fires when networkx is absent
            import networkx.algorithms.community as nx_comm
        except ImportError:
            logger.warning(
                "GraphRAG community detection requires networkx. "
                "Install with: pip install networkx"
            )
            return {}
        G = self._build_networkx_graph()
        if len(G.nodes) < 2:
            return dict.fromkeys(G.nodes, 0)
        try:
            communities = nx_comm.louvain_communities(G, seed=42)
            mapping = {}
            for cid, members in enumerate(communities):
                for entity in members:
                    mapping[entity] = cid
            return mapping
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return {}

    def _run_hierarchical_community_detection(self) -> tuple:
        """Run hierarchical community detection.

        Returns:
            (community_levels, community_hierarchy, community_children)
            - community_levels: {level_int: {entity_lower: cluster_id}}
            - community_hierarchy: {cluster_id: parent_cluster_id}  (None for root)
            - community_children: {cluster_id: [child_cluster_ids]}
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning(
                "GraphRAG community detection requires networkx. "
                "Install with: pip install networkx"
            )
            return {}, {}, {}

        G = self._build_networkx_graph()
        if G.number_of_nodes() < 2:
            nodes = list(G.nodes())
            return {0: dict.fromkeys(nodes, 0)}, {0: None}, {0: []}

        n_levels = max(1, getattr(self.config, "graph_rag_community_levels", 2))
        max_cluster_size = getattr(self.config, "graph_rag_community_max_cluster_size", 10)
        seed = getattr(self.config, "graph_rag_leiden_seed", 42)
        use_lcc = getattr(self.config, "graph_rag_community_use_lcc", True)

        components = list(nx.connected_components(G))
        if use_lcc and components:
            try:
                lcc_nodes = max(components, key=len)
                dropped = G.number_of_nodes() - len(lcc_nodes)
                if dropped > 0:
                    logger.info(
                        f"GraphRAG: use_lcc=True — clustering {len(lcc_nodes)} nodes, "
                        f"dropping {dropped} nodes in {len(components) - 1} smaller components"
                    )
                G = G.subgraph(lcc_nodes).copy()
            except Exception:
                pass
        elif not use_lcc and len(components) > 1:
            logger.debug(
                f"GraphRAG: clustering all {len(components)} connected components "
                f"({G.number_of_nodes()} total nodes)"
            )

        _backend = getattr(self.config, "graph_rag_community_backend", "auto")

        try:
            # Try graspologic hierarchical Leiden — skipped when backend != "auto"
            if _backend != "auto":
                raise ImportError("backend override — skipping graspologic")
            from graspologic.partition import hierarchical_leiden

            partitions = hierarchical_leiden(G, max_cluster_size=max_cluster_size, random_seed=seed)
            community_levels: dict = {}
            community_hierarchy: dict = {}
            community_children: dict = {}

            for triple in partitions:
                level = triple.level
                entity = triple.node
                cluster = triple.cluster
                parent = getattr(triple, "parent_cluster", None)

                if level not in community_levels:
                    community_levels[level] = {}
                community_levels[level][entity] = cluster

                if cluster not in community_hierarchy:
                    community_hierarchy[cluster] = parent
                if parent is not None:
                    if parent not in community_children:
                        community_children[parent] = []
                    if cluster not in community_children[parent]:
                        community_children[parent].append(cluster)

            # Normalize Leiden hierarchy to level-qualified string keys to match Louvain format
            # and avoid cross-level integer collisions.
            normalized_hierarchy: dict = {}
            normalized_children: dict = {}
            # Build a level-lookup for each (level, cluster) → level-qualified key
            cluster_to_level: dict = {}
            for lvl, cmap in community_levels.items():
                for _ent, cid in cmap.items():
                    if cid not in cluster_to_level:
                        cluster_to_level[cid] = lvl
            for raw_cid, raw_parent in community_hierarchy.items():
                lvl = cluster_to_level.get(raw_cid, 0)
                norm_key = f"{lvl}_{raw_cid}"
                if raw_parent is None:
                    norm_parent = None
                else:
                    parent_lvl = cluster_to_level.get(raw_parent, max(0, lvl - 1))
                    norm_parent = f"{parent_lvl}_{raw_parent}"
                normalized_hierarchy[norm_key] = norm_parent
                if norm_parent is not None:
                    if norm_parent not in normalized_children:
                        normalized_children[norm_parent] = []
                    if norm_key not in normalized_children[norm_parent]:
                        normalized_children[norm_parent].append(norm_key)

            return community_levels, normalized_hierarchy, normalized_children

        except ImportError:
            logger.warning(
                "GraphRAG: graspologic not installed — falling back to leidenalg/Louvain. "
                "pip install axon[graphrag]"
            )

        # Tier-2: multi-resolution Leiden via leidenalg — skipped when backend="louvain"
        try:
            if _backend == "louvain":
                raise ImportError("backend override — skipping leidenalg")
            import igraph as _ig
            import leidenalg as _la

            try:
                import numpy as np

                resolutions = list(np.linspace(0.5, 1.5, n_levels)) if n_levels > 1 else [1.0]
            except ImportError:
                step = 1.0 / max(n_levels - 1, 1) if n_levels > 1 else 0
                resolutions = [0.5 + i * step for i in range(n_levels)]

            _G_ig = _ig.Graph.from_networkx(G)
            community_levels: dict = {}
            for level_idx, resolution in enumerate(resolutions):
                try:
                    partition = _la.find_partition(
                        _G_ig,
                        _la.ModularityVertexPartition,
                        seed=seed,
                    )
                    node_names = (
                        _G_ig.vs["_nx_name"]
                        if "_nx_name" in _G_ig.vs.attributes()
                        else list(G.nodes())
                    )
                    cmap = {}
                    for cid, members in enumerate(partition):
                        for idx in members:
                            cmap[node_names[idx]] = cid
                    community_levels[level_idx] = cmap
                except Exception as _e:
                    logger.debug("leidenalg at resolution %s failed: %s", resolution, _e)

            if community_levels:
                # Build synthetic hierarchy (same approach as Louvain fallback below)
                community_hierarchy: dict = {}
                community_children: dict = {}
                if len(community_levels) > 1:
                    _levels_sorted = sorted(community_levels.keys())
                    for _i in range(1, len(_levels_sorted)):
                        _fine = _levels_sorted[_i]
                        _coarse = _levels_sorted[_i - 1]
                        for _fine_cid in set(community_levels[_fine].values()):
                            _fine_key = f"{_fine}_{_fine_cid}"
                            _members = [
                                n for n, c in community_levels[_fine].items() if c == _fine_cid
                            ]
                            _votes: dict = {}
                            for _m in _members:
                                _p = community_levels[_coarse].get(_m)
                                if _p is not None:
                                    _votes[_p] = _votes.get(_p, 0) + 1
                            _parent_cid = max(_votes, key=_votes.get) if _votes else None
                            _parent_key = (
                                f"{_coarse}_{_parent_cid}" if _parent_cid is not None else None
                            )
                            community_hierarchy[_fine_key] = _parent_key
                            if _parent_key:
                                community_children.setdefault(_parent_key, [])
                                if _fine_key not in community_children[_parent_key]:
                                    community_children[_parent_key].append(_fine_key)
                    for _cid in set(community_levels[_levels_sorted[0]].values()):
                        _root_key = f"{_levels_sorted[0]}_{_cid}"
                        community_hierarchy.setdefault(_root_key, None)
                else:
                    for _cid in set(community_levels[0].values()):
                        community_hierarchy[f"0_{_cid}"] = None
                logger.debug(
                    "GraphRAG: leidenalg produced %d community levels.", len(community_levels)
                )
                return community_levels, community_hierarchy, community_children
        except ImportError:
            pass  # fall through to networkx Louvain

        # Synthetic parent-child mapping using multiple-resolution Louvain
        try:
            import networkx.algorithms.community as nx_comm
        except ImportError:
            return {0: dict.fromkeys(G.nodes(), 0)}, {0: None}, {0: []}

        try:
            import numpy as np

            resolutions = list(np.linspace(0.5, 1.5, n_levels)) if n_levels > 1 else [1.0]
        except ImportError:
            step = 1.0 / max(n_levels - 1, 1) if n_levels > 1 else 0
            resolutions = [0.5 + i * step for i in range(n_levels)]

        community_levels = {}

        for level_idx, resolution in enumerate(resolutions):
            try:
                partition = nx_comm.louvain_communities(G, seed=seed, resolution=resolution)
                cmap = {}
                for cid, nodes in enumerate(partition):
                    for node in nodes:
                        cmap[node] = cid
                community_levels[level_idx] = cmap
            except Exception as e:
                logger.debug(f"Louvain at resolution {resolution} failed: {e}")

        if not community_levels:
            return {0: dict.fromkeys(G.nodes(), 0)}, {0: None}, {0: []}

        # Synthetic parent mapping — use level-qualified keys to avoid cross-level collisions
        community_hierarchy = {}
        community_children = {}

        if len(community_levels) > 1:
            levels_sorted = sorted(community_levels.keys())
            for i in range(1, len(levels_sorted)):
                fine_level = levels_sorted[i]
                coarse_level = levels_sorted[i - 1]
                fine_map = community_levels[fine_level]
                coarse_map = community_levels[coarse_level]

                fine_clusters = set(fine_map.values())
                for fine_cid in fine_clusters:
                    fine_key = f"{fine_level}_{fine_cid}"
                    fine_members = [n for n, c in fine_map.items() if c == fine_cid]
                    coarse_votes: dict = {}
                    for m in fine_members:
                        parent_cid = coarse_map.get(m)
                        if parent_cid is not None:
                            coarse_votes[parent_cid] = coarse_votes.get(parent_cid, 0) + 1
                    parent_cid = max(coarse_votes, key=coarse_votes.get) if coarse_votes else None
                    parent_key = f"{coarse_level}_{parent_cid}" if parent_cid is not None else None
                    community_hierarchy[fine_key] = parent_key
                    if parent_key is not None:
                        if parent_key not in community_children:
                            community_children[parent_key] = []
                        if fine_key not in community_children[parent_key]:
                            community_children[parent_key].append(fine_key)

            for cid in set(community_levels[levels_sorted[0]].values()):
                root_key = f"{levels_sorted[0]}_{cid}"
                if root_key not in community_hierarchy:
                    community_hierarchy[root_key] = None
        else:
            for cid in set(community_levels[0].values()):
                community_hierarchy[f"0_{cid}"] = None

        return community_levels, community_hierarchy, community_children

    def _rebuild_communities(self) -> None:
        """Run community detection and generate summaries."""
        with self._community_rebuild_lock:
            # P1: Entity alias resolution — merge near-duplicates before community detection
            if getattr(self.config, "graph_rag_entity_resolve", False):
                self._resolve_entity_aliases()
            logger.info("GraphRAG: Running community detection...")
            result = self._run_hierarchical_community_detection()
            if isinstance(result, tuple) and len(result) == 3:
                community_levels, hierarchy, children = result
            else:
                # Legacy: _run_hierarchical_community_detection returned a plain dict
                community_levels = result if isinstance(result, dict) else {}
                hierarchy = {}
                children = {}
            self._community_levels = community_levels
            self._community_hierarchy = hierarchy
            self._community_children = children
            self._save_community_hierarchy()
            if self._community_levels:
                self._save_community_levels()
                total = sum(len(set(m.values())) for m in self._community_levels.values())
                logger.info(
                    f"GraphRAG: {len(self._community_levels)} community levels, "
                    f"{total} total communities detected."
                )
                if self.config.graph_rag_community:
                    _lazy = getattr(self.config, "graph_rag_community_lazy", False)
                    if _lazy:
                        logger.info(
                            "GraphRAG: community_lazy=True — skipping summarization; "
                            "will generate on first global query."
                        )
                    else:
                        self._generate_community_summaries()
                        if getattr(self.config, "graph_rag_index_community_reports", True):
                            self._index_community_reports_in_vector_store()

    def finalize_ingest(self) -> None:
        """Flush all deferred saves and trigger community rebuild.

        Call once after the last ``ingest()`` when ``ingest_batch_mode=True``.
        Flushes BM25, entity/relation/claims graphs, then delegates to
        ``finalize_graph()`` for community rebuild.
        Safe to call when ``ingest_batch_mode=False`` (flush is a no-op; community
        rebuild still runs as normal).
        """
        self._assert_write_allowed("finalize_ingest")
        if getattr(self.config, "ingest_batch_mode", False):
            if self._own_bm25:
                self._own_bm25.flush()
                logger.info("finalize_ingest: BM25 corpus flushed.")
            self._save_entity_graph()
            self._save_relation_graph()
            if getattr(self, "_claims_graph", None):
                self._save_claims_graph()
            logger.info("finalize_ingest: entity/relation/claims graphs saved.")
            if self._code_graph.get("nodes"):
                self._save_code_graph()
                logger.info("finalize_ingest: code graph saved.")
        self.finalize_graph()

    def finalize_graph(self, force: bool = False) -> None:
        """Explicitly trigger community rebuild.

        Use after batch ingest with ``graph_rag_community_defer=True`` to run
        community detection once when all documents have been ingested.
        Set ``force=True`` to rebuild even when the in-memory dirty flag is not
        set (for example, after a server restart or project switch).
        """
        self._assert_write_allowed("finalize_graph")
        if force or self._community_graph_dirty:
            self._community_graph_dirty = False
            self._rebuild_communities()

    # ── Graph visualization ──────────────────────────────────────────────────

    _VIZ_TYPE_COLORS: dict[str, str] = {
        "PERSON": "#4e79a7",
        "ORGANIZATION": "#f28e2b",
        "GEO": "#59a14f",
        "EVENT": "#e15759",
        "CONCEPT": "#76b7b2",
        "PRODUCT": "#edc948",
        "UNKNOWN": "#bab0ab",
    }

    def build_graph_payload(self) -> dict:
        """Return a renderer-neutral graph payload normalised from internal graph state.

        The payload shape is::

            {
                "nodes": [{"id", "name", "label", "type", "color", "val",
                           "chunk_count", "degree", "community", "description",
                           "tooltip",
                           "evidence": [{"chunk_id", "source", "start_line", "excerpt"}, ...]
                           }, ...],
                "links": [{"source", "target", "label", "relation",
                           "description", "value", "width"}, ...]
            }

        ``evidence`` is populated from the vector store for each chunk ID
        referenced by the node.  It may be empty if the store is unavailable or
        no chunks have been ingested yet.

        This method separates graph extraction from rendering.  Feed the result
        to :meth:`export_graph_html` or any other renderer.
        """
        from html import escape

        # community_levels level-0 schema: {entity -> community_id (int)}
        entity_to_community: dict[str, int] = {}
        if self._community_levels:
            for entity, cid in self._community_levels.get(0, {}).items():
                try:
                    entity_to_community[entity] = int(cid)
                except (TypeError, ValueError):
                    pass

        def _tooltip(name: str, node: dict, community: int | None) -> str:
            desc = (node.get("description") or "").strip()
            desc = escape(desc[:220]) if desc else "No description"
            ntype = escape(node.get("type") or "UNKNOWN")
            chunk_count = len(node.get("chunk_ids", []))
            degree = node.get("degree", 0)
            comm = "None" if community is None else str(community)
            return (
                f"<div style='max-width:320px'>"
                f"<div><b>{escape(name)}</b></div>"
                f"<div><b>Type:</b> {ntype}</div>"
                f"<div><b>Chunks:</b> {chunk_count}</div>"
                f"<div><b>Degree:</b> {degree}</div>"
                f"<div><b>Community:</b> {comm}</div>"
                f"<div style='margin-top:6px'>{desc}</div>"
                f"</div>"
            )

        # Build a chunk_id → metadata lookup for evidence population.
        all_chunk_ids: list[str] = []
        for _node in self._entity_graph.values():
            if isinstance(_node, dict):
                all_chunk_ids.extend(_node.get("chunk_ids", []))
        chunk_meta_lookup: dict[str, dict] = {}
        if all_chunk_ids and hasattr(self, "vector_store"):
            try:
                for _doc in self.vector_store.get_by_ids(list(dict.fromkeys(all_chunk_ids))):
                    _cid = _doc.get("id") or _doc.get("chunk_id", "")
                    if _cid:
                        chunk_meta_lookup[_cid] = _doc.get("metadata", _doc)
            except Exception:
                pass

        nodes: list[dict] = []
        node_ids: set[str] = set()
        for name, node in self._entity_graph.items():
            if not isinstance(node, dict):
                continue
            community = entity_to_community.get(name)
            chunk_count = len(node.get("chunk_ids", []))
            evidence = [
                {
                    "chunk_id": cid,
                    "source": meta.get("source", ""),
                    "start_line": meta.get("start_line"),
                    "excerpt": (meta.get("text") or meta.get("page_content") or "")[:200],
                }
                for cid in node.get("chunk_ids", [])
                if (meta := chunk_meta_lookup.get(cid)) is not None
            ]
            nodes.append(
                {
                    "id": name,
                    "name": name,
                    "label": name[:24],
                    "type": node.get("type") or "UNKNOWN",
                    "color": self._VIZ_TYPE_COLORS.get(node.get("type") or "UNKNOWN", "#aec7e8"),
                    "val": 4 + min(chunk_count, 18),
                    "chunk_count": chunk_count,
                    "degree": node.get("degree", 0),
                    "community": community,
                    "description": (node.get("description") or "")[:220],
                    "tooltip": _tooltip(name, node, community),
                    "evidence": evidence,
                }
            )
            node_ids.add(name)

        links: list[dict] = []
        seen_edges: set[tuple] = set()
        for src, rels in self._relation_graph.items():
            if src not in node_ids:
                continue
            for rel in rels:
                if not isinstance(rel, dict):
                    continue
                tgt = rel.get("target") or rel.get("object", "")
                if not tgt or tgt not in node_ids:
                    continue
                relation = rel.get("relation", "")
                key = (src, tgt, relation)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                strength = float(rel.get("weight") or rel.get("strength") or 5)
                links.append(
                    {
                        "source": src,
                        "target": tgt,
                        "label": relation[:32],
                        "relation": relation,
                        "description": rel.get("description", ""),
                        "value": strength,
                        "width": 1 + strength / 8,
                    }
                )

        return {"nodes": nodes, "links": links}

    def build_code_graph_payload(self) -> dict:
        """Return the code structure graph as {nodes, links} for VS Code webview.

        Node types: ``file``, ``class``, ``function``, ``method``, ``module``.
        Edge types: ``CONTAINS`` (file→symbol), ``IMPORTS`` (file→file).
        Returns ``{"nodes": [], "links": []}`` when no code graph has been built.
        """
        _COLORS = {
            "file": "#4ec9b0",
            "module": "#569cd6",
            "class": "#c586c0",
            "function": "#dcdcaa",
            "method": "#dcdcaa",
        }
        nodes: list[dict] = []
        for node_id, node in self._code_graph.get("nodes", {}).items():
            ntype = node.get("node_type", "unknown")
            sig = node.get("signature", "")
            label = node.get("name", node_id)
            nodes.append(
                {
                    "id": node_id,
                    "name": label,
                    "label": label[:28],
                    "type": ntype,
                    "color": _COLORS.get(ntype, "#888888"),
                    "val": 5 if ntype == "file" else 3,
                    "file_path": node.get("file_path", ""),
                    "start_line": node.get("start_line") or 1,
                    "chunk_ids": node.get("chunk_ids", []),
                    "tooltip": f"[{ntype}] {label}" + (f"\n{sig}" if sig else ""),
                }
            )
        links: list[dict] = []
        for edge in self._code_graph.get("edges", []):
            links.append(
                {
                    "source": edge.get("source", ""),
                    "target": edge.get("target", ""),
                    "label": edge.get("edge_type", ""),
                    "edge_type": edge.get("edge_type", ""),
                    "value": 1,
                    "width": 1,
                }
            )
        return {"nodes": nodes, "links": links}

    @staticmethod
    def _render_graph_html(graph: dict) -> str:
        """Render a graph payload (from :meth:`build_graph_payload`) as a 3D HTML viewer.

        The viewer loads three.js and 3d-force-graph from unpkg.com CDN — requires
        internet access to render.  The HTML file itself needs no server.
        """
        import json as _json

        # Escape </script> so a crafted entity string cannot break out of the script context.
        data_json = _json.dumps(graph, ensure_ascii=False).replace("</script>", "<\\/script>")
        n_nodes = len(graph["nodes"])
        n_links = len(graph["links"])
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>GraphRAG 3D Viewer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    html, body, #graph {{
      margin: 0; width: 100%; height: 100%;
      overflow: hidden;
      background: #0b1020; color: #e8edf7;
      font-family: ui-sans-serif, system-ui, sans-serif;
    }}
    .hud {{
      position: fixed; top: 14px; left: 14px; z-index: 20;
      max-width: 360px; padding: 12px 14px;
      border: 1px solid rgba(255,255,255,0.14); border-radius: 12px;
      background: rgba(8,12,20,0.84); backdrop-filter: blur(10px);
      box-shadow: 0 10px 30px rgba(0,0,0,0.28);
    }}
    .hud h1 {{ margin: 0 0 8px; font-size: 14px; }}
    .hud p  {{ margin: 4px 0; font-size: 12px; line-height: 1.45; opacity: 0.9; }}
    .legend {{ display: grid; grid-template-columns: auto 1fr;
               gap: 6px 10px; margin-top: 10px; font-size: 11px; }}
    .swatch {{ width: 10px; height: 10px; border-radius: 999px; margin-top: 3px; }}
  </style>
  <script src="https://unpkg.com/three"></script>
  <script src="https://unpkg.com/3d-force-graph"></script>
</head>
<body>
  <div class="hud">
    <h1>GraphRAG 3D Viewer</h1>
    <p>Left drag rotates &nbsp;·&nbsp; Right drag pans &nbsp;·&nbsp; Scroll zooms.</p>
    <p>Hover a node for details &nbsp;·&nbsp; Click to focus camera.</p>
    <p>Nodes: {n_nodes} &nbsp;|&nbsp; Edges: {n_links}</p>
    <div class="legend">
      <div class="swatch" style="background:#4e79a7"></div><div>PERSON</div>
      <div class="swatch" style="background:#f28e2b"></div><div>ORGANIZATION</div>
      <div class="swatch" style="background:#59a14f"></div><div>GEO</div>
      <div class="swatch" style="background:#e15759"></div><div>EVENT</div>
      <div class="swatch" style="background:#76b7b2"></div><div>CONCEPT</div>
      <div class="swatch" style="background:#edc948"></div><div>PRODUCT</div>
      <div class="swatch" style="background:#bab0ab"></div><div>UNKNOWN</div>
    </div>
  </div>
  <div id="graph"></div>
  <script>
    const graphData = {data_json};
    const elem = document.getElementById('graph');
    const Graph = ForceGraph3D()(elem)
      .graphData(graphData)
      .backgroundColor('#0b1020')
      .nodeLabel(node => node.tooltip)
      .nodeColor(node => node.color)
      .nodeVal(node => node.val)
      .nodeOpacity(0.95)
      .linkLabel(link => `<div><b>${{link.relation || 'relation'}}</b><br>${{link.description || ''}}</div>`)
      .linkWidth(link => link.width || 1)
      .linkOpacity(0.45)
      .linkDirectionalParticles(link => Math.min(4, Math.max(1, Math.round((link.value || 1) / 10))))
      .linkDirectionalParticleSpeed(0.004)
      .linkDirectionalParticleWidth(2)
      .d3Force('charge').strength(-140);
    Graph.d3Force('link').distance(90);
    Graph.onNodeClick(node => {{
      const distance = 120;
      const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
      Graph.cameraPosition(
        {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
        node, 1400
      );
    }});
    Graph.controls().autoRotate = false;
    Graph.controls().enableDamping = true;
    Graph.controls().dampingFactor = 0.12;
  </script>
</body>
</html>
"""

    def export_graph_html(
        self,
        path: str | None = None,
        json_path: str | None = None,
        open_browser: bool = True,
    ) -> str:
        """Export the entity–relation graph as a self-contained 3D interactive HTML viewer.

        Normalises internal graph state into a renderer-neutral payload via
        :meth:`build_graph_payload`, then renders it with three.js + 3d-force-graph.

        Args:
            path: File path to write the HTML to.  Defaults to a temp file when
                  *open_browser* is True and no path is provided.
            json_path: Optional path to also write the normalised graph JSON payload.
            open_browser: If True (default), open the generated HTML in the default
                          web browser immediately after writing.

        Returns:
            The rendered HTML string.
        """
        import json as _json
        import pathlib
        import tempfile

        graph = self.build_graph_payload()
        html = self._render_graph_html(graph)

        if json_path:
            pathlib.Path(json_path).write_text(
                _json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(
                "Graph JSON payload saved to %s (%d nodes, %d edges)",
                json_path,
                len(graph["nodes"]),
                len(graph["links"]),
            )

        if path:
            pathlib.Path(path).write_text(html, encoding="utf-8")
            logger.info(
                "Graph visualization saved to %s (%d nodes, %d edges)",
                path,
                len(graph["nodes"]),
                len(graph["links"]),
            )
            out_path = path
        elif open_browser:
            # Write to a temp file so the browser can load it
            tmp = tempfile.NamedTemporaryFile(
                suffix=".html", prefix="axon_graph_", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(html)
            tmp.close()
            out_path = tmp.name
            logger.info("Graph visualization written to temp file %s", out_path)
        else:
            out_path = None

        if open_browser and out_path:
            import webbrowser

            webbrowser.open(f"file://{pathlib.Path(out_path).resolve()}")
            logger.info("Opened graph visualization in default browser.")

        return html

    def _generate_community_summaries(self, query_hint: str = "") -> None:
        """Generate LLM summaries for each detected community cluster across all levels.

        When *query_hint* is provided (lazy mode), communities are ranked by relevance to
        the query first so the most useful ones get LLM treatment before the budget cap.
        The LLM cap is tightened to ``graph_rag_global_top_communities`` in lazy mode,
        replacing the full ``graph_rag_community_llm_max_total`` limit.
        """
        if not self._community_levels:
            return
        import hashlib as _hashlib
        import json as _json
        import re as _re
        from collections import defaultdict

        def _member_hash(level_idx: int, members: list) -> str:
            members_sorted = sorted(members)
            raw = f"{level_idx}|{'|'.join(members_sorted)}"
            return _hashlib.md5(raw.encode()).hexdigest()

        summaries = {}
        total_communities = sum(len(set(m.values())) for m in self._community_levels.values())

        _min_size = getattr(self.config, "graph_rag_community_min_size", 3)
        _top_n_per_level = getattr(self.config, "graph_rag_community_llm_top_n_per_level", 15)
        _max_total = getattr(self.config, "graph_rag_community_llm_max_total", 30)

        # Lazy mode: tighten cap to graph_rag_global_top_communities so only the most
        # query-relevant communities receive LLM treatment on the first global query.
        _lazy_cap = getattr(self.config, "graph_rag_global_top_communities", 0)
        if query_hint and _lazy_cap > 0:
            _max_total = min(_max_total, _lazy_cap)
            logger.info(
                "GraphRAG: lazy mode — generating community summaries for query "
                "(LLM cap: %d of %d communities).",
                _max_total,
                total_communities,
            )
        else:
            logger.info("GraphRAG: Generating summaries for %d communities...", total_communities)

        # Pre-compute query relevance scores when a hint is provided.
        _query_words: set = set(query_hint.lower().split()) if query_hint else set()

        def _query_relevance(members: list) -> float:
            """Fraction of query words present in community entity names."""
            if not _query_words:
                return 0.0
            member_words = {w for m in members for w in m.lower().split()}
            return len(_query_words & member_words) / len(_query_words)

        _llm_calls_issued = 0

        def _community_score(members: list) -> float:
            """Composite score = density × size. Used to rank which communities get LLM."""
            member_set = set(members)
            internal_edges = sum(
                1
                for src in members
                for entry in self._relation_graph.get(src, [])
                if entry.get("target", "") in member_set
            )
            return (internal_edges / max(len(members), 1)) * len(members)

        def _template_summary(level_idx: int, cid, members: list, new_hash: str) -> tuple:
            """Deterministic zero-LLM summary for small/capped communities."""
            size = len(members)
            sample = ", ".join(sorted(members)[:5])
            suffix = f" (+{size - 5} more)" if size > 5 else ""
            title = f"Community {cid}"
            summary = f"A cluster of {size} entities: {sample}{suffix}."
            return f"{level_idx}_{cid}", {
                "title": title,
                "summary": summary,
                "findings": [],
                "rank": float(size),
                "full_content": f"# {title}\n\n{summary}",
                "entities": members,
                "size": size,
                "level": level_idx,
                "member_hash": new_hash,
                "template": True,
            }

        def _summarise(args):
            level_idx, cid, members = args
            summary_key = f"{level_idx}_{cid}"
            new_hash = _member_hash(level_idx, members)
            existing = self._community_summaries.get(summary_key, {})
            if existing.get("member_hash") == new_hash and existing.get("full_content"):
                # Membership unchanged — reuse cached summary
                cached = dict(existing)
                cached["member_hash"] = new_hash
                return summary_key, cached
            ent_parts = []
            for e in members[:30]:
                node = self._entity_graph.get(e, {})
                desc = node.get("description", "") if isinstance(node, dict) else ""
                ent_type = node.get("type", "") if isinstance(node, dict) else ""
                if desc:
                    ent_parts.append(
                        f"- {e} [{ent_type}]: {desc}" if ent_type else f"- {e}: {desc}"
                    )
                else:
                    ent_parts.append(f"- {e}")
            member_set = set(members)
            rel_parts = []
            for src in members[:20]:
                for entry in self._relation_graph.get(src, [])[:5]:
                    tgt = entry.get("target", "")
                    if tgt in member_set:
                        rel_desc = entry.get("description") or (
                            f"{src} {entry.get('relation', '')} {tgt}"
                        )
                        rel_parts.append(f"- {rel_desc}")
            # GAP 3a: token-budget check — substitute sub-community reports when too large
            max_ctx_tokens = getattr(self.config, "graph_rag_community_max_context_tokens", 4000)
            entity_rel_text = "\n".join(ent_parts + rel_parts)
            if len(entity_rel_text) // 4 > max_ctx_tokens:
                # Children keys are level-qualified strings (e.g. "1_3") — look up directly
                parent_key = f"{level_idx}_{cid}"
                sub_community_keys = self._community_children.get(
                    parent_key, self._community_children.get(cid, [])
                )
                sub_reports = []
                ranked_sub = sorted(
                    [k for k in sub_community_keys if k in self._community_summaries],
                    key=lambda k: self._community_summaries[k].get("rank", 0.0),
                    reverse=True,
                )
                for sub_key in ranked_sub[:5]:
                    sub_reports.append(self._community_summaries[sub_key].get("full_content", ""))
                if sub_reports:
                    entity_rel_text = "\n\n".join(
                        f"Sub-community Report:\n{r}" for r in sub_reports
                    )
            else:
                entity_rel_text = "\n".join(ent_parts)
                if rel_parts:
                    entity_rel_text += "\n\nKey relationships:\n" + "\n".join(rel_parts[:20])
            # GAP 3c: Include claims in community context — auto-enabled if claims extraction is on
            include_claims = getattr(self.config, "graph_rag_community_include_claims", False)
            if not include_claims and getattr(self.config, "graph_rag_claims", False):
                include_claims = True
            claim_parts = []
            if include_claims and self._claims_graph:
                for entity in members[:10]:
                    for chunk_claims in self._claims_graph.values():
                        for claim in chunk_claims:
                            if claim.get("subject", "").lower() == entity.lower():
                                status = claim.get("status", "")
                                desc = claim.get("description", "")
                                claim_parts.append(f"  [{status}] {entity}: {desc}")
                if claim_parts:
                    claim_parts = claim_parts[:10]
            context = entity_rel_text
            if claim_parts:
                context += "\n\nClaims:\n" + "\n".join(claim_parts)
            prompt = (
                "You are analyzing a knowledge graph community.\n\n"
                "Community members and descriptions:\n"
                f"{context}\n\n"
                "Generate a community report as JSON with this exact structure:\n"
                "{\n"
                '  "title": "<2-5 word theme name>",\n'
                '  "summary": "<2-3 sentence overview>",\n'
                '  "findings": [\n'
                '    {"summary": "<headline>", "explanation": "<1-2 sentence detail>"},\n'
                "    ... up to 8 findings\n"
                "  ],\n"
                '  "rank": <float 0-10 importance>\n'
                "}\n"
                "Output only valid JSON. No markdown code blocks."
            )
            try:
                raw_text = self.llm.complete(
                    prompt,
                    system_prompt="You are an expert knowledge graph analyst.",
                )
                try:
                    raw_stripped = _re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()
                    parsed = _json.loads(raw_stripped)
                    title = str(parsed.get("title", f"Community {cid}"))
                    summary = str(parsed.get("summary", ""))
                    findings = parsed.get("findings", [])
                    if not isinstance(findings, list):
                        findings = []
                    rank = float(parsed.get("rank", 5.0))
                except Exception:
                    title = f"Community {cid}"
                    summary = raw_text.strip() if raw_text else ""
                    findings = []
                    rank = 5.0
                findings_text = "\n".join(
                    f"- {f.get('summary', '')}: {f.get('explanation', '')}"
                    for f in findings
                    if isinstance(f, dict)
                )
                full_content = f"# {title}\n\n{summary}"
                if findings_text:
                    full_content += f"\n\nFindings:\n{findings_text}"
                return summary_key, {
                    "title": title,
                    "summary": summary,
                    "findings": findings,
                    "rank": rank,
                    "full_content": full_content,
                    "entities": members,
                    "size": len(members),
                    "level": level_idx,
                    "member_hash": new_hash,
                }
            except Exception as e:
                logger.debug(f"Community summary failed for community {summary_key}: {e}")
                return summary_key, {
                    "title": f"Community {cid}",
                    "summary": "",
                    "findings": [],
                    "rank": 5.0,
                    "full_content": "",
                    "entities": members,
                    "size": len(members),
                    "level": level_idx,
                    "member_hash": new_hash,
                }

        # Process levels from finest (highest idx) to coarsest — SEQUENTIALLY so sub-community
        # reports are available when summarizing parent communities (true bottom-up composition).
        for level_idx in sorted(self._community_levels.keys(), reverse=True):
            level_map = self._community_levels[level_idx]
            community_entities: dict = defaultdict(list)
            for entity, cid in level_map.items():
                community_entities[cid].append(entity)

            llm_items, template_items = [], []
            if query_hint:
                # Prioritise query-relevant communities so they consume LLM budget first.
                ranked = sorted(
                    community_entities.items(),
                    key=lambda kv: (_query_relevance(kv[1]), _community_score(kv[1])),
                    reverse=True,
                )
            else:
                ranked = sorted(
                    community_entities.items(),
                    key=lambda kv: _community_score(kv[1]),
                    reverse=True,
                )

            for rank_pos, (cid, members) in enumerate(ranked):
                summary_key = f"{level_idx}_{cid}"
                new_hash = _member_hash(level_idx, members)
                # Cache hit: always reuse regardless of triage
                existing = self._community_summaries.get(summary_key, {})
                if existing.get("member_hash") == new_hash and existing.get("full_content"):
                    summaries[summary_key] = dict(existing)
                    continue
                # Triage gates (applied in order):
                if _min_size > 0 and len(members) < _min_size:
                    template_items.append((level_idx, cid, members, new_hash))
                    continue
                if _top_n_per_level > 0 and rank_pos >= _top_n_per_level:
                    template_items.append((level_idx, cid, members, new_hash))
                    continue
                if _max_total > 0 and _llm_calls_issued >= _max_total:
                    template_items.append((level_idx, cid, members, new_hash))
                    continue
                _llm_calls_issued += 1
                llm_items.append((level_idx, cid, members))

            for args in template_items:
                key, val = _template_summary(*args)
                summaries[key] = val
            for summary_key, summary_dict in self._executor.map(_summarise, llm_items):
                summaries[summary_key] = summary_dict
            # Make this level's summaries available before summarizing coarser levels
            self._community_summaries = dict(summaries)

        self._save_community_summaries()
        logger.info(f"GraphRAG: Community summaries generated for {len(summaries)} communities.")

    def _index_community_reports_in_vector_store(self) -> None:
        """Index community reports as synthetic documents in the vector store."""
        if not self._community_summaries:
            return
        docs_to_add = []
        for summary_key, cs in self._community_summaries.items():
            full_content = cs.get("full_content") or cs.get("summary", "")
            if not full_content:
                continue
            doc_id = f"__community__{summary_key}"
            content_hash = cs.get("member_hash", "")
            # Skip re-indexing if content is unchanged since last index
            if cs.get("indexed_hash") == content_hash and content_hash:
                continue
            cs["indexed_hash"] = content_hash
            docs_to_add.append(
                {
                    "id": doc_id,
                    "text": full_content,
                    "metadata": {
                        "graph_rag_type": "community_report",
                        "community_key": summary_key,
                        "level": cs.get("level", 0),
                        "rank": cs.get("rank", 5.0),
                        "title": cs.get("title", ""),
                        "source": "__community_report__",
                    },
                }
            )
        if docs_to_add:
            try:
                ids = [d["id"] for d in docs_to_add]
                texts = [d["text"] for d in docs_to_add]
                metadatas = [d["metadata"] for d in docs_to_add]
                embeddings = self.embedding.embed(texts)
                self._own_vector_store.add(ids, texts, embeddings, metadatas)
                logger.info(
                    f"GraphRAG: Indexed {len(docs_to_add)} community reports in vector store."
                )
            except Exception as e:
                logger.warning(f"Could not index community reports: {e}")

    def _global_search_map_reduce(self, query: str, cfg) -> str:
        """Map-reduce global search over community reports (Item 4)."""
        import json as _json
        import re as _re

        if not self._community_summaries:
            return ""

        min_score = getattr(cfg, "graph_rag_global_min_score", 20)
        top_points = getattr(cfg, "graph_rag_global_top_points", 50)

        # Filter summaries to the target level
        target_level = getattr(cfg, "graph_rag_community_level", 0)
        target_level_prefix = f"{target_level}_"
        level_summaries = {
            k: v for k, v in self._community_summaries.items() if k.startswith(target_level_prefix)
        }
        # Fall back to all summaries if nothing matches the target level
        if not level_summaries:
            level_summaries = self._community_summaries

        # TASK 12: Dynamic community pre-filter — cheap token-overlap relevance score
        _top_n_communities = getattr(cfg, "graph_rag_global_top_communities", 0)
        if _top_n_communities > 0 and len(level_summaries) > _top_n_communities:
            _query_words = set(query.lower().split())

            def _community_relevance(cs_item: tuple) -> float:
                _, cs = cs_item
                text = ((cs.get("title") or "") + " " + (cs.get("summary") or "")).lower()
                return len(_query_words & set(text.split())) / max(len(_query_words), 1)

            sorted_summaries = sorted(
                level_summaries.items(), key=_community_relevance, reverse=True
            )
            level_summaries = dict(sorted_summaries[:_top_n_communities])
            logger.debug("GraphRAG global: pre-filtered to top %d communities.", _top_n_communities)

        # Chunk reports so large reports don't get hard-truncated and later sections aren't lost
        _MAP_CHUNK_CHARS = int(getattr(cfg, "graph_rag_global_map_max_length", 500) or 500) * 4

        def _chunk_report(cid: str, cs: dict) -> list[tuple[str, str]]:
            """Split a community report into bounded chunks. Returns [(cid_chunk_id, text)]."""
            report = cs.get("full_content") or cs.get("summary", "")
            if not report:
                return []
            chunks = []
            for i in range(0, len(report), _MAP_CHUNK_CHARS):
                chunk_text = report[i : i + _MAP_CHUNK_CHARS]
                chunks.append((f"{cid}#{i}", chunk_text))
            return chunks

        # Build shuffled chunk list for unbiased map phase
        import random as _random

        all_chunks: list[tuple[str, str]] = []
        for cid, cs in level_summaries.items():
            all_chunks.extend(_chunk_report(cid, cs))
        rng = _random.Random(42)
        rng.shuffle(all_chunks)

        # TASK 14: Token-level compression of community report chunks before LLM map phase
        if getattr(cfg, "graph_rag_report_compress", False) is True and all_chunks:
            _ratio = getattr(cfg, "graph_rag_report_compress_ratio", 0.5)
            try:
                _lingua = self._ensure_llmlingua()
                _compressed_chunks = []
                for _cid, _text in all_chunks:
                    if not _text:
                        _compressed_chunks.append((_cid, _text))
                        continue
                    try:
                        _out = _lingua.compress_prompt(_text, rate=_ratio, force_tokens=["\n"])
                        _compressed_chunks.append((_cid, _out["compressed_prompt"]))
                    except Exception:
                        _compressed_chunks.append((_cid, _text))
                all_chunks = _compressed_chunks
                logger.info(
                    "GraphRAG map-reduce: compressed %d report chunks (ratio=%.2f)",
                    len(all_chunks),
                    _ratio,
                )
            except ImportError:
                logger.warning(
                    "graph_rag_report_compress=True but llmlingua not installed. "
                    "pip install axon[llmlingua]"
                )

        def _map_community(args):
            chunk_id, report_chunk = args
            if not report_chunk:
                return []
            prompt = (
                "---Role---\n"
                "You are a helpful assistant responding to questions about the dataset.\n\n"
                "---Goal---\n"
                "Generate a list of key points relevant to answering the question, "
                "based solely on the community report below.\n"
                "If the report is not relevant, return an empty JSON array.\n\n"
                f"---Question---\n{query}\n\n"
                f"---Community Report---\n{report_chunk}\n\n"
                "---Response Format---\n"
                'Return a JSON array: [{"point": "...", "score": 1-100}, ...]\n'
                "Higher score = more relevant. Output only valid JSON."
            )
            try:
                raw = self.llm.complete(
                    prompt,
                    system_prompt="You are a knowledge graph analysis assistant.",
                )
                raw_clean = _re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
                points = _json.loads(raw_clean)
                if not isinstance(points, list):
                    return []
                return [
                    (float(p.get("score", 0)), str(p.get("point", "")))
                    for p in points
                    if isinstance(p, dict) and p.get("point")
                ]
            except Exception:
                return []

        # Map phase — parallel over shuffled chunks
        # TASK 14: Use a dedicated pool when graph_rag_map_workers is set, so map-reduce
        # does not starve the shared executor during concurrent ingest.
        all_points = []
        _map_workers_cfg = getattr(cfg, "graph_rag_map_workers", 0)
        try:
            if isinstance(_map_workers_cfg, int) and _map_workers_cfg > 0:
                from concurrent.futures import ThreadPoolExecutor as _TPE

                with _TPE(max_workers=_map_workers_cfg) as _map_pool:
                    results = list(_map_pool.map(_map_community, all_chunks))
            else:
                results = list(self._executor.map(_map_community, all_chunks))
            for point_list in results:
                all_points.extend(point_list)
        except Exception:
            return ""

        # Filter and sort
        filtered = [(score, point) for score, point in all_points if score >= min_score]
        filtered.sort(reverse=True)
        top = filtered[:top_points]

        if not top:
            return _GRAPHRAG_NO_DATA_ANSWER

        # --- REDUCE PHASE: token-budget assembly ---
        reduce_max_tokens = getattr(cfg, "graph_rag_global_reduce_max_tokens", 8000)
        analyst_lines = []
        token_estimate = 0
        for idx, (score, point) in enumerate(top):
            line = f"----Analyst {idx + 1}----\nImportance Score: {score:.0f}\n{point}"
            # Rough token estimate: 1 token ≈ 4 characters
            token_estimate += len(line) // 4
            if token_estimate > reduce_max_tokens:
                break
            analyst_lines.append(line)

        if not analyst_lines:
            return _GRAPHRAG_NO_DATA_ANSWER

        reduce_context = "\n\n".join(analyst_lines)
        reduce_max_length = getattr(cfg, "graph_rag_global_reduce_max_length", 500)
        reduce_prompt = (
            f"The following analytic reports have been generated for the query:\n\n"
            f"Query: {query}\n\n"
            f"Reports:\n\n{reduce_context}\n\n"
            f"Using the reports above, generate a comprehensive response to the query."
            f"\nRespond in at most {reduce_max_length} tokens."
        )
        reduce_system_prompt = _GRAPHRAG_REDUCE_SYSTEM_PROMPT
        if getattr(cfg, "graph_rag_global_allow_general_knowledge", False):
            reduce_system_prompt = (
                reduce_system_prompt
                + " You may supplement the provided reports with your own general knowledge"
                " where relevant."
            )
        try:
            response = self.llm.complete(
                reduce_prompt,
                system_prompt=reduce_system_prompt,
            )
            return response.strip() if response else _GRAPHRAG_NO_DATA_ANSWER
        except Exception as e:
            logger.debug(f"GraphRAG global reduce failed: {e}")
            # Fallback: return raw analyst summaries
            return "**Knowledge Graph Findings:**\n\n" + reduce_context

    def _get_incoming_relations(self, entity: str) -> list[dict]:
        """Return all relation entries where entity is the target (GAP 4: incoming edges)."""
        results = []
        ent_lower = entity.lower()
        for src, entries in self._relation_graph.items():
            for entry in entries:
                if entry.get("target", "").lower() == ent_lower:
                    results.append({**entry, "source": src, "direction": "incoming"})
        return results

    def _local_search_context(self, query: str, matched_entities: list, cfg) -> str:
        """Build structured GraphRAG local context using unified candidate ranking.

        TASK 11: Replaces fixed 25/50/25 budget split with joint ranking across all
        artifact types, then greedy-fill up to total_budget tokens.
        """
        if not matched_entities:
            return ""

        import re as _re

        # --- Phase 1: Setup ---
        total_budget = getattr(cfg, "graph_rag_local_max_context_tokens", 8000)
        top_k_entities = getattr(cfg, "graph_rag_local_top_k_entities", 10)
        top_k_relationships = getattr(cfg, "graph_rag_local_top_k_relationships", 10)
        include_weight = getattr(cfg, "graph_rag_local_include_relationship_weight", False)

        entity_weight = getattr(cfg, "graph_rag_local_entity_weight", 3.0)
        relation_weight = getattr(cfg, "graph_rag_local_relation_weight", 2.0)
        community_weight = getattr(cfg, "graph_rag_local_community_weight", 1.5)
        text_unit_weight = getattr(cfg, "graph_rag_local_text_unit_weight", 1.0)

        _query_tokens = set(query.lower().split()) | set(_re.split(r"[\s\W_]+", query.lower()))
        _boost = getattr(self.config, "graph_rag_exact_entity_boost", 3.0)

        def _entity_degree(ent: str) -> int:
            outgoing = len(self._relation_graph.get(ent.lower(), []))
            incoming = len(self._get_incoming_relations(ent))
            return outgoing + incoming

        def _entity_raw(ent: str) -> float:
            return _entity_degree(ent) * (_boost if ent.lower() in _query_tokens else 1.0)

        ranked_entities = sorted(matched_entities, key=_entity_raw, reverse=True)
        ranked_entities = ranked_entities[:top_k_entities]

        # --- Phase 2: Collect all candidates into a flat list ---
        candidates: list[dict] = []

        # Entities
        raw_scores = [_entity_raw(e) for e in ranked_entities]
        max_raw = max(raw_scores, default=1.0) or 1.0
        for ent, raw in zip(ranked_entities, raw_scores):
            node = self._entity_graph.get(ent.lower(), {})
            desc = node.get("description", "") if isinstance(node, dict) else ""
            ent_type = node.get("type", "") if isinstance(node, dict) else ""
            if not desc:
                continue
            line = f"  - {ent} [{ent_type}]: {desc}" if ent_type else f"  - {ent}: {desc}"
            candidates.append(
                {
                    "type": "entity",
                    "score": entity_weight * (raw / max_raw),
                    "text": line,
                    "tokens": len(line) // 4 + 1,
                    "key": ent.lower(),
                }
            )

        # Relations — collect outgoing (top 3/entity) + incoming (top 2/entity)
        def _mutual_count(e_entry, ranked):
            tgt = e_entry.get("target", "")
            return sum(
                1
                for ee in ranked
                if tgt in {x.get("target", "") for x in self._relation_graph.get(ee.lower(), [])}
            )

        rel_candidates_raw: list[tuple[float, dict, str]] = []
        seen_rel_keys: set = set()
        for ent in ranked_entities:
            ent_lower = ent.lower()
            outgoing = self._relation_graph.get(ent_lower, [])
            sorted_outgoing = sorted(
                outgoing, key=lambda e: _mutual_count(e, ranked_entities), reverse=True
            )
            for entry in sorted_outgoing[:3]:
                mc = _mutual_count(entry, ranked_entities)
                rel_candidates_raw.append((mc, entry, ent))
            for entry in self._get_incoming_relations(ent)[:2]:
                mc = _mutual_count(entry, ranked_entities)
                rel_candidates_raw.append((mc, entry, ent))

        # Sort by mutual count, cap at top_k_relationships, normalize
        rel_candidates_raw.sort(key=lambda x: x[0], reverse=True)
        rel_candidates_raw = rel_candidates_raw[:top_k_relationships]
        max_mutual = max((x[0] for x in rel_candidates_raw), default=0)
        for mc, entry, ent in rel_candidates_raw:
            src = entry.get("source", ent)
            desc = entry.get("description") or (
                f"{src} {entry.get('relation', '')} {entry.get('target', '')}"
            )
            weight_str = ""
            if include_weight and entry.get("weight"):
                weight_str = f" (weight: {entry['weight']})"
            line = f"  - {desc}{weight_str}"
            rel_key = line[:80]
            if rel_key in seen_rel_keys:
                continue
            seen_rel_keys.add(rel_key)
            within = (mc + 1) / (max_mutual + 1)
            candidates.append(
                {
                    "type": "relation",
                    "score": relation_weight * within,
                    "text": line,
                    "tokens": len(line) // 4 + 1,
                }
            )

        # Communities — finest level, deduplicated by summary_key
        if self._community_levels and self._community_summaries:
            finest = max(self._community_levels.keys()) if self._community_levels else None
            seen_community_keys: set = set()
            community_raws: list[tuple[float, str, str]] = []
            for ent in ranked_entities:
                cid = (
                    self._community_levels.get(finest, {}).get(ent.lower())
                    if finest is not None
                    else None
                )
                if cid is None:
                    continue
                summary_key = f"{finest}_{cid}"
                if summary_key in seen_community_keys:
                    continue
                seen_community_keys.add(summary_key)
                cs = self._community_summaries.get(summary_key, {})
                if not cs:
                    cs = self._community_summaries.get(str(cid), {})
                summary = cs.get("summary", "")
                if not summary:
                    continue
                sentences = summary.split(". ")
                snippet = ". ".join(sentences[:3]).strip()
                if snippet and not snippet.endswith("."):
                    snippet += "."
                title = cs.get("title", f"Community {cid}")
                community_text = f"**Community Context ({title}):**\n  {snippet}"
                rank = cs.get("rank", 0.5)
                community_raws.append((rank, community_text, summary_key))
            max_rank = max((r[0] for r in community_raws), default=1.0) or 1.0
            for rank, ctext, _ in community_raws:
                within = rank / max_rank
                candidates.append(
                    {
                        "type": "community",
                        "score": community_weight * within,
                        "text": ctext,
                        "tokens": len(ctext) // 4 + 1,
                    }
                )

        # Text units — from entity chunk_ids, cap at 20, rank by relation count
        seen_chunks: set = set()
        text_unit_ids_all: list[str] = []
        for ent in ranked_entities:
            node = self._entity_graph.get(ent.lower(), {})
            chunk_ids = node.get("chunk_ids", []) if isinstance(node, dict) else []
            for cid in chunk_ids:
                if cid not in seen_chunks:
                    seen_chunks.add(cid)
                    text_unit_ids_all.append(cid)
                if len(text_unit_ids_all) >= 20:
                    break
            if len(text_unit_ids_all) >= 20:
                break

        def _tu_rel_count(cid: str) -> int:
            return len(self._text_unit_relation_map.get(cid, []))

        max_rel = max((_tu_rel_count(c) for c in text_unit_ids_all), default=0)
        for cid in text_unit_ids_all:
            try:
                docs = self.vector_store.get_by_ids([cid])
                if not docs:
                    continue
                text_snippet = docs[0].get("text", "")[:200].strip()
                if not text_snippet:
                    continue
                line = f"  [{cid}] {text_snippet}..."
                rc = _tu_rel_count(cid)
                within = (rc + 1) / (max_rel + 1)
                candidates.append(
                    {
                        "type": "text_unit",
                        "score": text_unit_weight * within,
                        "text": line,
                        "tokens": len(line) // 4 + 1,
                    }
                )
            except Exception:
                pass

        # Claims — high-signal factual assertions, fixed score
        if self._claims_graph:
            claim_lines: list[str] = []
            for ent in ranked_entities[:3]:
                for chunk_claims in self._claims_graph.values():
                    for claim in chunk_claims:
                        if claim.get("subject", "").lower() == ent.lower():
                            status = claim.get("status", "SUSPECTED")
                            desc = claim.get("description", "")
                            if desc:
                                claim_lines.append(f"  - [{status}] {desc}")
            for line in claim_lines[:10]:
                candidates.append(
                    {
                        "type": "claim",
                        "score": entity_weight * 1.0,
                        "text": line,
                        "tokens": len(line) // 4 + 1,
                    }
                )

        # --- Phase 3: Sort and greedy-fill ---
        candidates.sort(key=lambda c: c["score"], reverse=True)
        selected: list[dict] = []
        used = 0
        for c in candidates:
            if used + c["tokens"] <= total_budget:
                selected.append(c)
                used += c["tokens"]
            # continue (not break) — smaller items later may still fit

        # --- Phase 4: Reassemble into section-formatted output ---
        by_type: dict[str, list[str]] = {
            "entity": [],
            "relation": [],
            "community": [],
            "text_unit": [],
            "claim": [],
        }
        for c in selected:
            by_type[c["type"]].append(c["text"])

        parts: list[str] = []
        if by_type["entity"]:
            parts.append("**Matched Entities:**\n" + "\n".join(by_type["entity"]))
        if by_type["relation"]:
            parts.append("**Relevant Relationships:**\n" + "\n".join(by_type["relation"]))
        for ctext in by_type["community"]:
            parts.append(ctext)
        if by_type["text_unit"]:
            parts.append("**Source Text Units:**\n" + "\n".join(by_type["text_unit"]))
        if by_type["claim"]:
            parts.append("**Claims / Facts:**\n" + "\n".join(by_type["claim"]))

        return "\n\n".join(parts)

    def _entity_matches(self, q_entity: str, g_entity: str) -> float:
        """Return Jaccard-based match score between 0.0 and 1.0."""
        q = q_entity.lower().strip()
        g = g_entity.lower().strip()
        if q == g:
            return 1.0
        q_tokens = set(q.split())
        g_tokens = set(g.split())
        if len(q_tokens) == 1 and len(g_tokens) == 1:
            return 0.0  # single tokens must match exactly
        intersection = q_tokens & g_tokens
        union = q_tokens | g_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        return jaccard if jaccard >= 0.4 else 0.0

    _VALID_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "GEO", "EVENT", "CONCEPT", "PRODUCT"}

    _HOLISTIC_KEYWORDS = frozenset(
        [
            "summarize",
            "summary",
            "overview",
            "all",
            "every",
            "compare",
            "list all",
            "across",
            "throughout",
            "entire",
            "whole",
            "global",
            "what are all",
            "how many",
            "count",
            "trend",
            "pattern",
            "themes",
            "main topics",
        ]
    )

    def _classify_query_needs_graphrag(self, query: str, mode: str) -> bool:
        """Return True if GraphRAG global search is warranted for this query (TASK 14)."""
        if mode == "heuristic":
            q_lower = query.lower()
            if any(kw in q_lower for kw in AxonBrain._HOLISTIC_KEYWORDS):
                return True
            if len(query.split()) > 20:
                return True
            return False
        elif mode == "llm":
            prompt = (
                "Does the following question require a holistic, corpus-wide analysis "
                "(e.g. summarization, comparison, listing all topics) rather than a "
                "targeted factual lookup? Answer only YES or NO.\n\nQuestion: " + query
            )
            try:
                ans = self.llm.complete(prompt, max_tokens=5).strip().upper()
                return ans.startswith("Y")
            except Exception:
                return False
        return False

    # ── keyword sets for multi-class router ───────────────────────────────────
    _SYNTHESIS_KEYWORDS: frozenset = frozenset(
        {
            "summarize",
            "overview",
            "compare",
            "contrast",
            "explain",
            "discuss",
            "survey",
            "themes",
            "analysis",
        }
    )
    _TABLE_KEYWORDS: frozenset = frozenset(
        {
            "table",
            "row",
            "column",
            "value",
            "count",
            "average",
            "maximum",
            "minimum",
            "statistic",
            "how many",
            "list all",
        }
    )
    _ENTITY_KEYWORDS: frozenset = frozenset(
        {
            "relationship",
            "related to",
            "who",
            "works with",
            "connected",
            "linked",
            "colleague",
            "dependency",
        }
    )
    _CORPUS_KEYWORDS: frozenset = frozenset(
        {
            "all documents",
            "entire corpus",
            "everything",
            "main topics",
            "key themes",
            "across all",
        }
    )

    def _classify_query_route(self, query: str, cfg: "AxonConfig") -> str:
        """Return one of: factual | synthesis | table_lookup | entity_relation | corpus_exploration."""
        if cfg.query_router == "llm":
            return self._classify_query_route_llm(query)
        return self._classify_query_route_heuristic(query)

    def _classify_query_route_heuristic(self, query: str) -> str:
        q = query.lower()
        words = set(q.split())
        # corpus_exploration
        if any(kw in q for kw in self._CORPUS_KEYWORDS) or (
            len(query) > 120 and any(kw in words for kw in self._SYNTHESIS_KEYWORDS)
        ):
            return "corpus_exploration"
        # entity_relation
        if any(kw in q for kw in self._ENTITY_KEYWORDS):
            return "entity_relation"
        # table_lookup
        if any(kw in q for kw in self._TABLE_KEYWORDS):
            return "table_lookup"
        # synthesis
        if any(kw in words for kw in self._SYNTHESIS_KEYWORDS) or len(query) > 80:
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

    def _ensure_llmlingua(self):
        """Lazy-initialise the LLMLingua-2 prompt compressor (TASK 14)."""
        if not hasattr(self, "_llmlingua") or self._llmlingua is None:
            from llmlingua import PromptCompressor

            _model = getattr(
                self.config,
                "graph_rag_llmlingua_model",
                "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            )
            _local = os.path.isabs(_model) or os.path.isdir(_model)
            logger.info(
                "GraphRAG LLMLingua: loading model '%s'%s…", _model, " (local)" if _local else ""
            )
            self._llmlingua = PromptCompressor(
                model_name=_model,
                use_llmlingua2=True,
                device_map="cpu",
                **({"local_files_only": True} if _local else {}),
            )
        return self._llmlingua

    def _ensure_gliner(self):
        """Lazy-initialise the GLiNER NER model (TASK 14)."""
        if not hasattr(self, "_gliner_model") or self._gliner_model is None:
            from gliner import GLiNER

            _model = getattr(self.config, "graph_rag_gliner_model", "urchade/gliner_medium-v2.1")
            _local = os.path.isabs(_model) or os.path.isdir(_model)
            logger.info(
                "GraphRAG GLiNER: loading model '%s'%s…", _model, " (local)" if _local else ""
            )
            self._gliner_model = GLiNER.from_pretrained(_model, local_files_only=_local)
        return self._gliner_model

    _GLINER_LABELS = ["person", "organization", "location", "event", "concept", "product"]
    _GLINER_TYPE_MAP = {
        "person": "PERSON",
        "organization": "ORGANIZATION",
        "location": "GEO",
        "event": "EVENT",
        "concept": "CONCEPT",
        "product": "PRODUCT",
    }

    def _ensure_rebel(self):
        """Lazy-initialise the REBEL relation extraction pipeline (P2)."""
        if not hasattr(self, "_rebel_pipeline") or self._rebel_pipeline is None:
            try:
                from transformers import pipeline as _hf_pipeline
            except ImportError:
                raise ImportError("REBEL backend requires transformers. pip install axon[rebel]")
            _model = getattr(self.config, "graph_rag_rebel_model", "Babelscape/rebel-large")
            _local = os.path.isabs(_model) or os.path.isdir(_model)
            logger.info(
                "GraphRAG REBEL: loading model '%s'%s…",
                _model,
                " (local)" if _local else " (first-run download may take time)",
            )
            self._rebel_pipeline = _hf_pipeline(
                "text2text-generation",
                model=_model,
                tokenizer=_model,
                device=-1,  # CPU — avoids CUDA dependency
                **({"local_files_only": True} if _local else {}),
            )
        return self._rebel_pipeline

    @staticmethod
    def _parse_rebel_output(text: str) -> list[dict]:
        """Parse REBEL's ``<triplet> SUBJ <subj> OBJ <obj> REL`` output format.

        REBEL encodes triplets as:
            <triplet> head_entity <subj> tail_entity <obj> relation_type

        Returns a list of relation dicts compatible with ``_extract_relations``.
        """
        triplets: list[dict] = []
        subject = object_ = relation = ""
        current = "x"
        tokens = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
        for token in tokens:
            if token == "<triplet>":
                if subject and relation and object_:
                    triplets.append(
                        {
                            "subject": subject.strip(),
                            "relation": relation.strip(),
                            "object": object_.strip(),
                            "description": "",
                            "strength": 5,
                        }
                    )
                subject = object_ = relation = ""
                current = "t"
            elif token == "<subj>":
                current = "s"
            elif token == "<obj>":
                current = "o"
            elif current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
        # Flush final triplet
        if subject and relation and object_:
            triplets.append(
                {
                    "subject": subject.strip(),
                    "relation": relation.strip(),
                    "object": object_.strip(),
                    "description": "",
                    "strength": 5,
                }
            )
        return triplets

    def _extract_relations_rebel(self, text: str) -> list[dict]:
        """Extract relations using REBEL — no LLM call, structured triplet output (P2)."""
        try:
            pipe = self._ensure_rebel()
            # REBEL's tokeniser caps at 512 tokens; truncate input to ~1 000 chars
            outputs = pipe(text[:1000], max_length=512, return_tensors=False)
            raw = outputs[0]["generated_text"] if outputs else ""
            return AxonBrain._parse_rebel_output(raw)[:15]
        except ImportError:
            logger.warning(
                "graph_rag_relation_backend='rebel' but transformers is not installed. "
                "pip install axon[rebel]"
            )
            return []
        except Exception:
            return []

    def _extract_entities_gliner(self, text: str) -> list[dict]:
        """Extract entities using GLiNER (TASK 14 — NER-only; no LLM call)."""
        model = self._ensure_gliner()
        try:
            preds = model.predict_entities(text[:3000], AxonBrain._GLINER_LABELS, threshold=0.5)
            seen: set = set()
            entities = []
            for p in preds:
                name = p["text"].strip()
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                entities.append(
                    {
                        "name": name,
                        "type": AxonBrain._GLINER_TYPE_MAP.get(p["label"].lower(), "CONCEPT"),
                        "description": "",
                    }
                )
            return entities[:20]
        except Exception:
            return []

    def _extract_entities_light(self, text: str) -> list[dict]:
        """Lightweight entity extraction using regex noun-phrase heuristics (no LLM).

        Picks capitalized multi-word phrases (2–4 tokens) from the text.
        Returns up to 20 entities with type=CONCEPT and empty description.
        """
        import re as _re

        pattern = _re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\b")
        seen: set = set()
        entities: list[dict] = []
        for match in pattern.finditer(text):
            phrase = match.group(1)
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                entities.append({"name": phrase, "type": "CONCEPT", "description": ""})
            if len(entities) >= 20:
                break
        return entities

    def _extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from text using the LLM.

        Returns a list of dicts with shape:
          {"name": str, "type": str, "description": str}
        Returns an empty list on failure or when the LLM produces no output.
        """
        # A3: light tier — skip LLM entirely
        if getattr(self.config, "graph_rag_depth", "standard") == "light":
            return self._extract_entities_light(text)

        # TASK 14: GLiNER fast-path — skip LLM for NER when backend is "gliner"
        if getattr(self.config, "graph_rag_ner_backend", "llm") == "gliner":
            return self._extract_entities_gliner(text)

        prompt = (
            "Extract the key named entities from the following text.\n"
            "For each entity output one line:\n"
            "  ENTITY_NAME | ENTITY_TYPE | one-sentence description\n"
            "ENTITY_TYPE must be one of: PERSON, ORGANIZATION, GEO, EVENT, CONCEPT, PRODUCT\n"
            "No bullets, numbering, or extra text. If no entities, output nothing.\n\n"
            + text[:3000]
        )
        try:
            raw = self.llm.complete(
                prompt, system_prompt="You are a named entity extraction specialist."
            )
            entities = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    ent_type = parts[1].upper()
                    if ent_type not in AxonBrain._VALID_ENTITY_TYPES:
                        ent_type = "CONCEPT"
                    entities.append(
                        {
                            "name": parts[0],
                            "type": ent_type,
                            "description": parts[2],
                        }
                    )
                elif len(parts) == 2:
                    # Old 2-column format — no type
                    entities.append(
                        {
                            "name": parts[0],
                            "type": "UNKNOWN",
                            "description": parts[1],
                        }
                    )
                else:
                    entities.append(
                        {
                            "name": parts[0],
                            "type": "UNKNOWN",
                            "description": "",
                        }
                    )
            return entities[:20]
        except Exception:
            return []

    def _extract_relations(self, text: str) -> list[dict]:
        """Extract SUBJECT | RELATION | OBJECT | description quads from text via the LLM.

        Returns up to 15 relation dicts with shape
        {"subject": str, "relation": str, "object": str, "description": str}.
        Returns an empty list on failure or when the LLM produces no output.
        """
        # A3: light tier skips all relation extraction
        if getattr(self.config, "graph_rag_depth", "standard") == "light":
            return []

        # P2: REBEL fast-path — skip LLM when backend is "rebel"
        if getattr(self.config, "graph_rag_relation_backend", "llm") == "rebel":
            return self._extract_relations_rebel(text)

        prompt = (
            "Extract key relationships from the following text.\n"
            "For each relationship output one line:\n"
            "  SUBJECT | RELATION | OBJECT | one-sentence description | strength (1-10)\n"
            "Strength: 1=weak/incidental, 10=core/defining. "
            "No bullets or extra text. If no clear relationships, output nothing.\n\n" + text[:3000]
        )
        try:
            raw = self.llm.complete(
                prompt, system_prompt="You are a knowledge graph extraction specialist."
            )
            triples = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5:
                    try:
                        strength = max(1, min(10, int(parts[4])))
                    except (ValueError, IndexError):
                        strength = 5
                    triples.append(
                        {
                            "subject": parts[0],
                            "relation": parts[1],
                            "object": parts[2],
                            "description": parts[3],
                            "strength": strength,
                        }
                    )
                elif len(parts) >= 4:
                    triples.append(
                        {
                            "subject": parts[0],
                            "relation": parts[1],
                            "object": parts[2],
                            "description": parts[3],
                            "strength": 5,
                        }
                    )
                elif len(parts) == 3 and all(parts):
                    triples.append(
                        {
                            "subject": parts[0],
                            "relation": parts[1],
                            "object": parts[2],
                            "description": "",
                            "strength": 5,
                        }
                    )
            return triples[:15]
        except Exception:
            return []

    def _embed_entities(self, entity_keys: list) -> None:
        """Embed entity descriptions; store in _entity_embeddings (Item 5)."""
        to_embed = []
        for key in entity_keys:
            node = self._entity_graph.get(key, {})
            if isinstance(node, dict):
                desc = node.get("description", "")
                if desc and key not in self._entity_embeddings:
                    to_embed.append((key, f"{key}: {desc}"))
        if not to_embed:
            return
        keys, texts = zip(*to_embed)
        try:
            vectors = self.embedding.embed(list(texts))
            for k, v in zip(keys, vectors):
                self._entity_embeddings[k] = v if isinstance(v, list) else list(v)
            self._save_entity_embeddings()
            logger.info(f"GraphRAG: Embedded {len(to_embed)} entity descriptions.")
        except Exception as e:
            logger.debug(f"Entity embedding failed: {e}")

    def _match_entities_by_embedding(self, query: str, top_k: int = 5) -> list:
        """Return entity keys matching the query by embedding cosine similarity (Item 5)."""
        if not self._entity_embeddings:
            return []
        try:
            import numpy as np

            q_vec = np.array(self.embedding.embed_query(query))
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return []
            scored = []
            for entity_key, e_vec in self._entity_embeddings.items():
                ev = np.array(e_vec)
                ev_norm = np.linalg.norm(ev)
                if ev_norm == 0:
                    continue
                sim = float(np.dot(q_vec, ev) / (q_norm * ev_norm))
                scored.append((sim, entity_key))
            threshold = getattr(self.config, "graph_rag_entity_match_threshold", 0.5)
            scored = [(s, k) for s, k in scored if s >= threshold]
            scored.sort(reverse=True)
            return [k for _, k in scored[:top_k]]
        except Exception as e:
            logger.debug(f"Entity embedding match failed: {e}")
            return []

    def _extract_claims(self, text: str) -> list:
        """Extract factual claims from text (Item 11). Returns list of claim dicts."""
        prompt = (
            "Extract factual claims from the following text.\n"
            "For each claim output one line:\n"
            "  SUBJECT | OBJECT | CLAIM_TYPE | STATUS | DESCRIPTION | START_DATE | END_DATE | SOURCE_QUOTE\n"
            "  STATUS must be: TRUE, FALSE, or SUSPECTED\n"
            "  CLAIM_TYPE examples: acquisition, partnership, product_launch, regulatory_action, financial\n"
            "  START_DATE and END_DATE: ISO8601 date (YYYY-MM-DD) or 'unknown' if not mentioned\n"
            "  SOURCE_QUOTE: verbatim short quote from the text (max 100 chars) or 'none'\n"
            "No bullets or extra text. If no clear claims, output nothing.\n\n" + text[:3000]
        )
        try:
            raw = self.llm.complete(
                prompt,
                system_prompt="You are a fact extraction specialist.",
            )
            claims = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 8:
                    status = parts[3].upper()
                    if status not in ("TRUE", "FALSE", "SUSPECTED"):
                        status = "SUSPECTED"
                    claims.append(
                        {
                            "subject": parts[0],
                            "object": parts[1],
                            "type": parts[2],
                            "status": status,
                            "description": parts[4],
                            "start_date": parts[5] if parts[5] != "unknown" else None,
                            "end_date": parts[6] if parts[6] != "unknown" else None,
                            "source_text": parts[7] if parts[7] != "none" else None,
                            "text_unit_id": None,  # filled in by caller
                        }
                    )
                elif len(parts) >= 5:
                    status = parts[3].upper()
                    if status not in ("TRUE", "FALSE", "SUSPECTED"):
                        status = "SUSPECTED"
                    claims.append(
                        {
                            "subject": parts[0],
                            "object": parts[1],
                            "type": parts[2],
                            "status": status,
                            "description": parts[4],
                            "start_date": None,
                            "end_date": None,
                            "source_text": None,
                            "text_unit_id": None,
                        }
                    )
            return claims[:10]
        except Exception:
            return []

    def _resolve_entity_aliases(self) -> int:
        """Merge semantically equivalent entity nodes into a single canonical node (P1).

        Uses cosine similarity on entity-name embeddings from the active embedding model.
        Entities whose name-embeddings exceed ``graph_rag_entity_resolve_threshold`` are
        grouped; within each group the node with the most chunk_ids becomes canonical and
        all alias nodes are merged into it (chunk_ids, description, relations).

        Returns the number of alias nodes merged (0 if nothing to merge).
        """
        import numpy as np

        threshold = getattr(self.config, "graph_rag_entity_resolve_threshold", 0.92)
        max_entities = getattr(self.config, "graph_rag_entity_resolve_max", 5000)

        keys = [k for k, v in self._entity_graph.items() if isinstance(v, dict)]
        n = len(keys)
        if n < 2:
            return 0
        if n > max_entities:
            logger.warning(
                "GraphRAG entity resolution: entity graph has %d nodes (limit=%d). "
                "Skipping alias resolution — increase graph_rag_entity_resolve_max to enable.",
                n,
                max_entities,
            )
            return 0

        # Embed entity names (not descriptions) — aliases share similar surface forms
        try:
            raw_emb = self.embedding.embed(keys)
        except Exception as exc:
            logger.warning("GraphRAG entity resolution: embedding failed (%s). Skipping.", exc)
            return 0

        emb = np.array(raw_emb, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb = emb / norms  # unit-normalise for cosine via dot product

        # Union-Find for grouping similar entities
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        # O(n²) pairwise similarity — acceptable for n ≤ max_entities (default 5 000)
        sim = emb @ emb.T  # shape (n, n)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    _union(i, j)

        # Collect groups
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(_find(i), []).append(i)

        merged = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            # Canonical = entity with most chunk_ids (highest coverage)
            canon_idx = max(
                members,
                key=lambda i: len(self._entity_graph[keys[i]].get("chunk_ids", [])),
            )
            canon_key = keys[canon_idx]
            canon_node = self._entity_graph[canon_key]

            for idx in members:
                if idx == canon_idx:
                    continue
                alias_key = keys[idx]
                alias_node = self._entity_graph[alias_key]

                # Merge chunk_ids
                for cid in alias_node.get("chunk_ids", []):
                    if cid not in canon_node["chunk_ids"]:
                        canon_node["chunk_ids"].append(cid)

                # Inherit description if canonical has none
                if not canon_node.get("description") and alias_node.get("description"):
                    canon_node["description"] = alias_node["description"]

                # Migrate relation_graph entries keyed by the alias
                if alias_key in self._relation_graph:
                    canon_rels = self._relation_graph.setdefault(canon_key, [])
                    canon_rels.extend(self._relation_graph.pop(alias_key))

                # Rewrite subject/object references inside all relation lists
                for rel_list in self._relation_graph.values():
                    for rel in rel_list:
                        if isinstance(rel, dict):
                            if rel.get("subject", "").lower() == alias_key:
                                rel["subject"] = canon_key
                            if rel.get("object", "").lower() == alias_key:
                                rel["object"] = canon_key

                del self._entity_graph[alias_key]
                merged += 1

            # Refresh frequency for canonical
            canon_node["frequency"] = len(canon_node["chunk_ids"])

        if merged > 0:
            logger.info(
                "GraphRAG entity resolution: merged %d alias nodes into canonical entities "
                "(threshold=%.2f, %d nodes remaining).",
                merged,
                threshold,
                len(self._entity_graph),
            )
            self._community_graph_dirty = True

        return merged

    def _canonicalize_entity_descriptions(self) -> None:
        """Synthesize canonical descriptions for entities with multiple descriptions (Item 10)."""
        min_occ = getattr(self.config, "graph_rag_canonicalize_min_occurrences", 3)
        to_canonicalize = {
            k: descs
            for k, descs in self._entity_description_buffer.items()
            if len(set(descs)) > 1 and len(descs) >= min_occ
        }
        if not to_canonicalize:
            return

        logger.info(f"GraphRAG: Canonicalizing descriptions for {len(to_canonicalize)} entities...")

        def _synthesize(args):
            entity_key, descs = args
            unique_descs = list(dict.fromkeys(descs))[:10]  # deduplicate, cap at 10
            node = self._entity_graph.get(entity_key, {})
            entity_type = node.get("type", "UNKNOWN") if isinstance(node, dict) else "UNKNOWN"
            numbered = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(unique_descs))
            prompt = (
                f"Entity: {entity_key}\nType: {entity_type}\n\n"
                "The following descriptions were extracted from different documents:\n"
                f"{numbered}\n\n"
                "Write a single comprehensive description that synthesizes all of the above. "
                "Output only the description, no preamble."
            )
            try:
                result = self.llm.complete(
                    prompt,
                    system_prompt="You are a knowledge graph curation specialist.",
                )
                return entity_key, result.strip() if result else unique_descs[0]
            except Exception:
                return entity_key, unique_descs[0]

        results = list(self._executor.map(_synthesize, to_canonicalize.items()))
        changed = False
        for entity_key, canonical_desc in results:
            node = self._entity_graph.get(entity_key)
            if isinstance(node, dict) and canonical_desc:
                node["description"] = canonical_desc
                changed = True
        if changed:
            self._save_entity_graph()
        self._entity_description_buffer.clear()

    def _canonicalize_relation_descriptions(self) -> None:
        """Synthesize canonical descriptions for repeated (subject, object) pairs (GAP 3b)."""
        if not getattr(self.config, "graph_rag_canonicalize_relations", False):
            return
        min_occ = getattr(self.config, "graph_rag_canonicalize_relations_min_occurrences", 2)
        to_canonicalize = {
            pair: descs
            for pair, descs in self._relation_description_buffer.items()
            if len(descs) >= min_occ
        }
        if not to_canonicalize:
            return

        def _synthesize(args):
            (src, tgt), descs = args
            unique_descs = list(dict.fromkeys(descs))[:10]
            numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(unique_descs))
            prompt = (
                f"Relationship: {src} → {tgt}\n\n"
                f"The following descriptions were extracted from different documents:\n{numbered}\n\n"
                "Write a single comprehensive description that synthesizes all of the above. "
                "Output only the description, no preamble."
            )
            try:
                result = self.llm.complete(
                    prompt,
                    system_prompt="You are a knowledge graph curation specialist.",
                )
                return (src, tgt), result.strip() if result else unique_descs[0]
            except Exception:
                return (src, tgt), unique_descs[0]

        results = list(self._executor.map(_synthesize, to_canonicalize.items()))
        changed = False
        for (src, tgt), canonical_desc in results:
            if src in self._relation_graph:
                for entry in self._relation_graph[src]:
                    if entry.get("target", "").lower() == tgt and canonical_desc:
                        entry["description"] = canonical_desc
                        changed = True
        if changed:
            self._save_relation_graph()
        self._relation_description_buffer.clear()

    def _prune_entity_graph(self, deleted_ids: set) -> None:
        """Remove deleted chunk IDs from entity graph and relation graph.

        Entries that become empty are deleted entirely.  Persists changes to disk.
        """
        eg_changed = False
        for entity in list(self._entity_graph):
            node = self._entity_graph[entity]
            chunk_ids = node.get("chunk_ids", [])
            after = [d for d in chunk_ids if d not in deleted_ids]
            if len(after) != len(chunk_ids):
                eg_changed = True
                if after:
                    self._entity_graph[entity]["chunk_ids"] = after
                    # Recompute frequency after pruning
                    self._entity_graph[entity]["frequency"] = len(after)
                else:
                    del self._entity_graph[entity]
        if eg_changed:
            self._save_entity_graph()

        rg_changed = False
        for src in list(self._relation_graph):
            before = self._relation_graph[src]
            after = [entry for entry in before if entry.get("chunk_id") not in deleted_ids]
            if len(after) != len(before):
                rg_changed = True
                if after:
                    self._relation_graph[src] = after
                else:
                    del self._relation_graph[src]
        if rg_changed:
            self._save_relation_graph()

        # Prune claims graph (Item 11)
        if hasattr(self, "_claims_graph"):
            claims_changed = False
            for chunk_id in list(self._claims_graph):
                if chunk_id in deleted_ids:
                    del self._claims_graph[chunk_id]
                    claims_changed = True
            if claims_changed:
                self._save_claims_graph()

    def _raptor_group_by_structure(self, chunks: list[dict], n: int) -> list[list[dict]]:
        """Group chunks for RAPTOR summarization using section/heading boundaries.

        Detection priority (any one match = new section starts):
          1. metadata["heading"] or metadata["section"] is non-empty
          2. text starts with a Markdown heading (#, ##, ###)
          3. text starts with numbered/lettered heading pattern

        Fallback: if no heading found across any chunk, pure fixed windows of size n.
        Within detected sections: further split into sub-windows of size n if section > n.
        """
        import re as _re

        _heading_re = _re.compile(
            r"^(#{1,4}\s|chapter\s+\d|section\s+[\d.]+|\d+\.\s+[A-Z]|\b[IVX]+\.\s)",
            _re.IGNORECASE,
        )

        def _is_section_start(doc: dict) -> bool:
            meta = doc.get("metadata", {})
            if meta.get("heading") or meta.get("section"):
                return True
            return bool(_heading_re.match(doc.get("text", "").lstrip()[:80]))

        has_structure = any(_is_section_start(c) for c in chunks)
        if not has_structure:
            return [chunks[i : i + n] for i in range(0, len(chunks), n)]

        sections: list[list[dict]] = []
        current: list[dict] = []
        for chunk in chunks:
            if _is_section_start(chunk) and current:
                sections.append(current)
                current = [chunk]
            else:
                current.append(chunk)
        if current:
            sections.append(current)

        windows: list[list[dict]] = []
        for sec in sections:
            for i in range(0, len(sec), n):
                windows.append(sec[i : i + n])
        return windows

    def _generate_raptor_summaries(self, documents: list[dict]) -> list[dict]:
        """Generate RAPTOR summary nodes for a list of already-split chunks."""
        from itertools import groupby

        n = self.config.raptor_chunk_group_size
        if n < 1:
            logger.warning("raptor_chunk_group_size must be >= 1, skipping RAPTOR")
            return []

        def _source(doc):
            return doc.get("metadata", {}).get("source", doc["id"])

        sorted_docs = sorted(documents, key=_source)
        windows = []
        for source, group in groupby(sorted_docs, key=_source):
            chunks = list(group)
            for idx, window in enumerate(self._raptor_group_by_structure(chunks, n)):
                windows.append((source, idx, window))

        if not windows:
            return []

        logger.info(f"   RAPTOR: Generating summaries for {len(windows)} groups...")

        import hashlib as _hl

        def _summarise_window(source, i, window, level):
            """Call LLM (or return from cache) for one window at a given RAPTOR level."""
            combined = "\n\n".join(c["text"] for c in window)
            content_hash = _hl.md5(combined[:4000].encode("utf-8", errors="replace")).hexdigest()[
                :12
            ]
            cache_key = f"{source}|L{level}|{i}|{content_hash}"

            # P5: return cached summary when content is unchanged
            if self.config.raptor_cache_summaries and cache_key in self._raptor_summary_cache:
                logger.debug(f"RAPTOR cache hit for {source} L{level}[{i}]")
                return self._raptor_summary_cache[cache_key]

            prompt = (
                "Summarise the following passage into a concise but comprehensive paragraph "
                "that captures all key facts and concepts. "
                "Output only the summary paragraph.\n\n" + combined[:4000]
            )
            try:
                text = self.llm.complete(
                    prompt,
                    system_prompt="You are an expert at summarising technical documents.",
                )
                if not text or not text.strip():
                    return None
                text = text.strip()
                if self.config.raptor_cache_summaries:
                    self._raptor_summary_cache[cache_key] = text
                return text
            except Exception as e:
                logger.debug(f"RAPTOR L{level} summary failed for {source}[{i}]: {e}")
                return None

        def _proc_window(item):
            source, i, window = item
            summary_text = _summarise_window(source, i, window, level=1)
            if not summary_text or not isinstance(summary_text, str):
                return None
            try:
                content_sig = _hl.md5(summary_text.encode("utf-8", errors="replace")).hexdigest()[
                    :8
                ]
                uid = _hl.md5(f"{source}|raptor|{i}|{content_sig}".encode()).hexdigest()[:12]
                return {
                    "id": f"raptor_{uid}",
                    "text": summary_text,
                    "metadata": {
                        "source": source,
                        "raptor_level": 1,
                        "window_start": i,
                        "window_end": i + len(window) - 1,
                        "children_ids": [doc["id"] for doc in window],
                    },
                }
            except Exception as e:
                logger.debug(f"RAPTOR node build failed for {source}[{i}]: {e}")
                return None

        results = list(self._executor.map(_proc_window, windows))
        level_1_summaries = [r for r in results if r]
        if level_1_summaries:
            logger.info(f"   RAPTOR: generated {len(level_1_summaries)} level-1 summary node(s)")

        # P3: Recursive summarization up to raptor_max_levels
        max_levels = getattr(self.config, "raptor_max_levels", 2)
        all_summaries = list(level_1_summaries)
        prev_level_nodes = list(level_1_summaries)
        current_level = 1

        while current_level < max_levels and len(prev_level_nodes) > 1:
            next_level = current_level + 1
            logger.info(
                f"   RAPTOR: Building level-{next_level} summaries from "
                f"{len(prev_level_nodes)} level-{current_level} node(s)..."
            )
            sorted_prev = sorted(prev_level_nodes, key=_source)
            next_windows = []
            for source, group in groupby(sorted_prev, key=_source):
                group_list = list(group)
                for idx, window in enumerate(self._raptor_group_by_structure(group_list, n)):
                    next_windows.append((source, idx, window, next_level))

            if not next_windows:
                break

            def _proc_upper(item):
                source, i, window, lvl = item
                summary_text = _summarise_window(source, i, window, level=lvl)
                if not summary_text or not isinstance(summary_text, str):
                    return None
                try:
                    content_sig = _hl.md5(
                        summary_text.encode("utf-8", errors="replace")
                    ).hexdigest()[:8]
                    uid = _hl.md5(f"{source}|raptor|L{lvl}|{i}|{content_sig}".encode()).hexdigest()[
                        :12
                    ]
                    children_ids = [c["id"] for c in window]
                    node = {
                        "id": f"raptor_{uid}",
                        "text": summary_text,
                        "metadata": {
                            "source": source,
                            "raptor_level": lvl,
                            "window_start": i,
                            "window_end": i + len(window) - 1,
                            "children_ids": children_ids,
                        },
                    }
                    for child in window:
                        child["metadata"]["parent_id"] = node["id"]
                    return node
                except Exception as e:
                    logger.debug(f"RAPTOR L{lvl} node build failed for {source}[{i}]: {e}")
                    return None

            upper_results = list(self._executor.map(_proc_upper, next_windows))
            next_level_nodes = [r for r in upper_results if r]
            if next_level_nodes:
                logger.info(
                    f"   RAPTOR: generated {len(next_level_nodes)} level-{next_level} summary node(s)"
                )
            all_summaries.extend(next_level_nodes)
            prev_level_nodes = next_level_nodes
            current_level = next_level

        return all_summaries

    def _raptor_drilldown(self, query: str, results: list[dict], cfg=None) -> list[dict]:
        """Replace RAPTOR summary hits with their underlying leaf chunks.

        For each result whose metadata contains ``raptor_level >= 1``:

        * If ``children_ids`` is stored in the node's metadata, fetch those exact
          descendants via ``get_by_ids`` and recurse until leaf chunks are reached
          (P2+P3 — true tree traversal, no spurious cross-source contamination).
        * If ``children_ids`` is absent (level-1 nodes ingested before multi-level
          RAPTOR was added), fall back to a filtered ``search`` using the now-fixed
          Chroma ``where`` clause (P1).
        * After all substitutions, deduplicate by ID keeping the highest-scored
          occurrence (P4).

        Non-RAPTOR results pass through unchanged.  Falls back to keeping the
        summary when window metadata is missing or any fetch fails.
        """
        if cfg is None:
            cfg = self.config
        if not getattr(cfg, "raptor_drilldown", True):
            return results

        drilldown_top_k = getattr(cfg, "raptor_drilldown_top_k", 5)
        final: list[dict] = []

        def _collect_leaves(node_ids: list[str], depth: int = 0) -> list[dict]:
            """Recursively fetch children via get_by_ids until leaf chunks are reached."""
            if not node_ids or depth > 5:
                return []
            docs = self.vector_store.get_by_ids(node_ids)
            leaves = []
            to_recurse = []
            for doc in docs:
                if doc.get("metadata", {}).get("raptor_level"):
                    grandchildren = doc.get("metadata", {}).get("children_ids", [])
                    if grandchildren:
                        to_recurse.extend(grandchildren)
                    # RAPTOR node with no children_ids — treat as leaf to avoid data loss
                else:
                    leaves.append(doc)
            if to_recurse:
                leaves.extend(_collect_leaves(to_recurse, depth + 1))
            return leaves

        for r in results:
            meta = r.get("metadata", {})
            if not meta.get("raptor_level"):
                final.append(r)
                continue

            source = meta.get("source")
            window_start = meta.get("window_start")
            window_end = meta.get("window_end")

            if source is None or window_start is None or window_end is None:
                final.append(r)
                continue

            try:
                children_ids = meta.get("children_ids")
                if children_ids:
                    # P2+P3: walk the stored lineage tree
                    leaves = _collect_leaves(children_ids)
                    if not leaves:
                        # children_ids present but store returned nothing — fall back
                        query_vec = self.embedding.embed([query])[0]
                        fetch_k = (window_end - window_start + 1) * 3
                        raw = self.vector_store.search(
                            query_vec,
                            top_k=max(fetch_k, drilldown_top_k * 2),
                            filter_dict={"source": source},
                        )
                        leaves = [d for d in raw if not d.get("metadata", {}).get("raptor_level")]
                else:
                    # Legacy level-1 node or pre-P3 node: use filtered search
                    query_vec = self.embedding.embed([query])[0]
                    fetch_k = (window_end - window_start + 1) * 3
                    raw = self.vector_store.search(
                        query_vec,
                        top_k=max(fetch_k, drilldown_top_k * 2),
                        filter_dict={"source": source},
                    )
                    leaves = [d for d in raw if not d.get("metadata", {}).get("raptor_level")]
            except Exception as e:
                logger.debug(f"RAPTOR drilldown fetch failed for {source}: {e}")
                final.append(r)
                continue

            if not leaves:
                final.append(r)
                continue

            if cfg.rerank and self.reranker:
                leaves = self.reranker.rerank(query, leaves)
            else:
                leaves = sorted(leaves, key=lambda x: x.get("score", 0.0), reverse=True)

            final.extend(leaves[:drilldown_top_k])

        # P4: Deduplicate by ID, keeping highest-scored occurrence
        seen: dict[str, dict] = {}
        deduped: list[dict] = []
        for res in final:
            rid = res["id"]
            if rid not in seen:
                seen[rid] = res
                deduped.append(res)
            elif res.get("score", 0.0) > seen[rid].get("score", 0.0):
                idx = next(i for i, x in enumerate(deduped) if x["id"] == rid)
                deduped[idx] = res
                seen[rid] = res
        return deduped

    def _apply_artifact_ranking(self, results: list[dict], cfg=None) -> list[dict]:
        """Re-order results by artifact type according to ``raptor_retrieval_mode``.

        Applies a score multiplier per artifact type, then re-sorts:

        * ``tree_traversal`` (default): leaf ×1.5 > raptor ×1.0 > community ×0.7
        * ``summary_first``:            raptor ×1.5 > leaf ×1.0 > community ×0.7
        * ``corpus_overview``:          community ×1.5 > raptor ×1.0 > leaf ×0.7
        """
        if cfg is None:
            cfg = self.config
        mode = getattr(cfg, "raptor_retrieval_mode", "tree_traversal")
        if mode not in ("tree_traversal", "summary_first", "corpus_overview"):
            return results

        WEIGHTS = {
            "tree_traversal": {"leaf": 1.5, "raptor": 1.0, "community": 0.7},
            "summary_first": {"leaf": 1.0, "raptor": 1.5, "community": 0.7},
            "corpus_overview": {"leaf": 0.7, "raptor": 1.0, "community": 1.5},
        }
        w = WEIGHTS[mode]

        def _artifact_type(r: dict) -> str:
            meta = r.get("metadata", {})
            if meta.get("raptor_level"):
                return "raptor"
            if meta.get("graph_rag_type") == "community_report" or r.get("id", "").startswith(
                "__community__"
            ):
                return "community"
            return "leaf"

        for r in results:
            r["_artifact_score"] = r.get("score", 0.5) * w[_artifact_type(r)]

        ranked = sorted(results, key=lambda x: x["_artifact_score"], reverse=True)
        for r in ranked:
            r.pop("_artifact_score", None)
        return ranked

    # ------------------------------------------------------------------
    # Structural Code Graph (Phase 2 + Phase 3)
    # ------------------------------------------------------------------

    def _build_code_graph_from_chunks(self, chunks: list[dict]) -> None:
        """Build/update code graph nodes and CONTAINS/IMPORTS edges from codebase chunks.

        Nodes:
        - File node  — one per unique file_path
        - Symbol node — one per (file_path, symbol_name) with a real symbol_type

        Edges:
        - CONTAINS : File → Symbol
        - IMPORTS  : File → File  (resolved from imports metadata)
        """
        nodes: dict = self._code_graph.setdefault("nodes", {})
        edges_list: list = self._code_graph.setdefault("edges", [])
        existing_edges: set = {(e["source"], e["target"], e["edge_type"]) for e in edges_list}

        file_nodes_seen: set = set()

        for chunk in chunks:
            meta = chunk.get("metadata", {})
            if meta.get("source_class") != "code":
                continue

            file_path = meta.get("file_path") or meta.get("source", "")
            language = meta.get("language", "unknown")
            symbol_type = meta.get("symbol_type", "block")
            symbol_name = meta.get("symbol_name", "")
            chunk_id = chunk.get("id", "")
            file_node_id = file_path

            # ── File node ───────────────────────────────────────────────────
            if file_path and file_path not in file_nodes_seen:
                file_nodes_seen.add(file_path)
                if file_node_id not in nodes:
                    nodes[file_node_id] = {
                        "node_id": file_node_id,
                        "node_type": "file",
                        "name": os.path.basename(file_path),
                        "file_path": file_path,
                        "language": language,
                        "chunk_ids": [],
                        "signature": "",
                        "start_line": None,
                        "end_line": None,
                    }

            if file_path and chunk_id:
                cids = nodes[file_node_id]["chunk_ids"]
                if chunk_id not in cids:
                    cids.append(chunk_id)

            # ── Symbol node ──────────────────────────────────────────────────
            if symbol_type not in ("block", "") and symbol_name and file_path:
                sym_node_id = f"{file_path}::{symbol_name}"
                if sym_node_id not in nodes:
                    nodes[sym_node_id] = {
                        "node_id": sym_node_id,
                        "node_type": symbol_type,
                        "name": symbol_name,
                        "file_path": file_path,
                        "language": language,
                        "chunk_ids": [chunk_id] if chunk_id else [],
                        "signature": meta.get("signature", ""),
                        "start_line": meta.get("start_line"),
                        "end_line": meta.get("end_line"),
                    }
                else:
                    if chunk_id and chunk_id not in nodes[sym_node_id]["chunk_ids"]:
                        nodes[sym_node_id]["chunk_ids"].append(chunk_id)

                # CONTAINS edge: File → Symbol
                ek = (file_node_id, sym_node_id, "CONTAINS")
                if ek not in existing_edges and file_path:
                    edges_list.append(
                        {
                            "source": file_node_id,
                            "target": sym_node_id,
                            "edge_type": "CONTAINS",
                            "chunk_id": chunk_id,
                        }
                    )
                    existing_edges.add(ek)

            # ── IMPORTS edges ────────────────────────────────────────────────
            imports_raw = meta.get("imports", "")
            if isinstance(imports_raw, str):
                import_stmts = [s for s in imports_raw.split("|") if s.strip()]
            elif isinstance(imports_raw, list):
                import_stmts = imports_raw
            else:
                import_stmts = []

            for stmt in import_stmts:
                target_file_id = self._resolve_import_to_file(stmt.strip())
                if target_file_id and target_file_id != file_node_id:
                    ek = (file_node_id, target_file_id, "IMPORTS")
                    if ek not in existing_edges:
                        edges_list.append(
                            {
                                "source": file_node_id,
                                "target": target_file_id,
                                "edge_type": "IMPORTS",
                                "chunk_id": chunk_id,
                            }
                        )
                        existing_edges.add(ek)

    def _resolve_import_to_file(self, stmt: str) -> str | None:
        """Resolve an import statement to a file node_id in the code graph, or None."""
        m = re.match(r"^from\s+([\w.]+)\s+import", stmt)
        if not m:
            m = re.match(r"^import\s+([\w.]+)", stmt)
        if not m:
            return None
        module = m.group(1)
        # e.g. "axon.splitters" → look for file_path ending in axon/splitters.py
        module_rel = module.replace(".", "/") + ".py"
        for node_id, node in self._code_graph.get("nodes", {}).items():
            if node.get("node_type") == "file":
                fp = node.get("file_path", "").replace("\\", "/")
                if fp.endswith(module_rel):
                    return node_id
        return None

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
        diagnostics: "CodeRetrievalDiagnostics | None" = None,
        trace: "CodeRetrievalTrace | None" = None,
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
            "raptor": cfg.raptor,
            "raptor_chunk_group_size": cfg.raptor_chunk_group_size,
            "parent_chunk_size": cfg.parent_chunk_size,
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

    def _detect_dataset_type(self, doc: dict) -> tuple[str, bool]:
        """Detect the dataset type for a document using content-based heuristics.

        Returns:
            Tuple of (dataset_type, has_code) where dataset_type is one of
            'codebase', 'paper', 'doc', 'discussion', 'knowledge'
            and has_code is True when a doc-type document also contains code blocks.
        """
        # Use configured override if not "auto"
        if self.config.dataset_type != "auto":
            return self.config.dataset_type, False

        text = doc.get("text", "")
        source = doc.get("metadata", {}).get("source", "") or doc.get("id", "")

        # Priority 1: Code file extensions
        if source:
            ext = os.path.splitext(source)[1].lower()
            if ext in self._CODE_EXTENSIONS:
                return "codebase", False

        # Priority 1.5: Manifest / lockfile / generated-reference detection
        if source:
            _base = os.path.basename(source).lower()
            _manifest_names = {
                "package.json",
                "package-lock.json",
                "yarn.lock",
                "pnpm-lock.yaml",
                "requirements.txt",
                "requirements-dev.txt",
                "pipfile",
                "pipfile.lock",
                "pyproject.toml",
                "setup.cfg",
                "setup.py",
                "cargo.toml",
                "cargo.lock",
                "go.mod",
                "go.sum",
                "composer.json",
                "composer.lock",
                "gemfile",
                "gemfile.lock",
                "podfile",
                "podfile.lock",
                ".gitmodules",
                "cmakelists.txt",
                "makefile",
                "dockerfile",
            }
            if _base in _manifest_names:
                return "manifest", False
            _ext = os.path.splitext(source)[1].lower()
            if _ext in (".lock", ".sum"):
                return "manifest", False
            _ref_signals = (
                "/api/",
                "/apidocs/",
                "/api-docs/",
                "/swagger/",
                "/openapi/",
                "/reference/",
                "/javadoc/",
                "/doxygen/",
            )
            if any(sig in source.lower().replace("\\", "/") for sig in _ref_signals):
                return "reference", False

        # Priority 2: JSON with discussion keys
        if text.strip().startswith("{") or text.strip().startswith("["):
            try:
                import json as _json

                parsed = _json.loads(text[:2000])
                if isinstance(parsed, dict):
                    if any(k in parsed for k in ("role", "turn", "messages", "speaker")):
                        return "discussion", False
                elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                    if any(k in parsed[0] for k in ("role", "turn", "messages", "speaker")):
                        return "discussion", False
            except Exception:
                pass

        lines = text.splitlines()
        if not lines:
            return "doc", False

        # Priority 3: Tabular detection (avg commas or tabs per line)
        non_empty = [ln for ln in lines if ln.strip()]
        if non_empty:
            avg_commas = sum(ln.count(",") for ln in non_empty) / len(non_empty)
            avg_tabs = sum(ln.count("\t") for ln in non_empty) / len(non_empty)
            if avg_commas > 2.0 or avg_tabs > 1.5:
                return "knowledge", False

        # Priority 4: Code content heuristic (>15% of lines match code patterns)
        code_lines = sum(1 for ln in non_empty if self._CODE_LINE_PATTERNS.match(ln.strip()))
        code_ratio = code_lines / len(non_empty) if non_empty else 0.0
        if code_ratio > 0.15:
            # Mixed doc with heavy code
            if code_ratio < 0.5:
                return "doc", True
            return "codebase", False

        # Priority 5: Academic paper signals in first 2000 chars
        preview = text[:2000]
        paper_matches = len(self._PAPER_SIGNALS.findall(preview))
        if paper_matches >= 2:
            return "paper", False

        # Priority 6: Doc signals (markdown, numbered steps)
        if self._DOC_SIGNALS.search(text[:3000]):
            has_code = "```" in text or "    def " in text or "    class " in text
            return "doc", has_code

        # Priority 7: Extension-based fallback for common doc types
        if source:
            ext = os.path.splitext(source)[1].lower()
            if ext in (".pdf", ".docx", ".html", ".pptx", ".md", ".rst", ".txt"):
                return "doc", False

        return "doc", False

    def _get_splitter_for_type(self, dataset_type: str, has_code: bool, source: str = ""):
        """Return a splitter configured for the detected document type."""
        from axon.splitters import RecursiveCharacterTextSplitter, SemanticTextSplitter

        if dataset_type == "codebase":
            from axon.splitters import CodeAwareSplitter

            return CodeAwareSplitter()
        elif dataset_type == "paper":
            return SemanticTextSplitter(chunk_size=600, chunk_overlap=100)
        elif dataset_type == "discussion":
            return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        elif dataset_type == "knowledge":
            return SemanticTextSplitter(chunk_size=400, chunk_overlap=50)
        elif dataset_type == "doc" and has_code:
            return SemanticTextSplitter(chunk_size=500, chunk_overlap=75)
        elif dataset_type == "doc":
            # Auto-select MarkdownSplitter for .md sources when strategy is "semantic" (default)
            if (
                os.path.splitext(source)[1].lower() == ".md"
                and self.config.chunk_strategy == "semantic"
            ):
                from axon.splitters import MarkdownSplitter

                return MarkdownSplitter(
                    chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
                )
            return self.splitter
        else:
            return self.splitter  # Use default configured splitter

    def _split_with_parents(self, documents: list[dict]) -> list[dict]:
        """Split documents using parent-document (small-to-big) strategy.

        1. Split each raw document into large parent chunks (parent_chunk_size).
        2. Split each parent into small child chunks (chunk_size) for indexing.
        3. Store the parent text in every child's metadata so that at generation
           time _build_context() can return the richer parent passage instead of
           the small retrieval chunk.
        """
        from axon.splitters import RecursiveCharacterTextSplitter

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.parent_chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        all_chunks = []
        for doc in documents:
            # Detect and annotate dataset type before splitting
            dataset_type, has_code = self._detect_dataset_type(doc)
            doc.setdefault("metadata", {})["dataset_type"] = dataset_type
            if has_code:
                doc["metadata"]["has_code"] = True

            # Use type-specific child splitter so parent mode honours the same
            # chunking policy (e.g. code-aware splitting) as the normal ingest path.
            source = doc.get("metadata", {}).get("source", "")
            child_splitter = self._get_splitter_for_type(dataset_type, has_code, source=source)
            if child_splitter is None:
                child_splitter = self.splitter

            parent_texts = parent_splitter.split(doc["text"])
            for p_idx, parent_text in enumerate(parent_texts):
                p_doc = {
                    "id": f"{doc['id']}_p{p_idx}",
                    "text": parent_text,
                    "metadata": doc.get("metadata", {}).copy(),
                }
                child_chunks = child_splitter.transform_documents([p_doc])
                for child in child_chunks:
                    child["metadata"]["parent_text"] = parent_text
                all_chunks.extend(child_chunks)
        return all_chunks

    def ingest(self, documents: list[dict[str, Any]]) -> None:
        """Chunk, deduplicate, embed, and store *documents* in the knowledge base.

        Each document must be a dict with keys ``id`` (str), ``text`` (str), and
        optionally ``metadata`` (dict).  Chunking strategy and deduplication are
        governed by the active :class:`AxonConfig`.  When ``raptor=True``,
        summary nodes are generated and indexed alongside leaf chunks.  When
        ``graph_rag=True``, entities are extracted and added to the entity graph.
        """
        if not documents:
            return

        # Phase 6: block ingest on read-only scopes and mounted shares
        self._assert_write_allowed("ingest")

        # Phase 3: acquire a write lease so drain-mode can track in-flight writes.
        # _WriteLease.__del__ guarantees release even if an exception is raised.
        from axon.runtime import get_registry as _get_registry

        _ingest_lease = _get_registry().acquire(self._active_project)

        # Guard: raise immediately if the embedding model has changed since this
        # collection was created — mixing models silently corrupts retrieval.
        self._validate_embedding_meta(on_mismatch="raise")

        _defer_saves = getattr(self.config, "ingest_batch_mode", False)
        _policy_on = getattr(self.config, "source_policy_enabled", False)

        t0 = time.time()
        _ingest_fallback_count = 0
        from tqdm import tqdm

        logger.info(f"Ingesting {len(documents)} documents...")
        if self.splitter and self.config.parent_chunk_size > 0:
            documents = self._split_with_parents(documents)
        elif self.splitter:
            # Type-specific chunking: detect per-document and apply the right splitter
            chunked: list[dict] = []
            for doc in documents:
                dataset_type, has_code = self._detect_dataset_type(doc)
                # Store type metadata in each document
                doc.setdefault("metadata", {})["dataset_type"] = dataset_type
                if has_code:
                    doc["metadata"]["has_code"] = True
                source = doc.get("metadata", {}).get("source", "")
                splitter = self._get_splitter_for_type(dataset_type, has_code, source=source)
                if splitter is not None:
                    chunked.extend(splitter.transform_documents([doc]))
                    if hasattr(splitter, "fallback_chunks_produced"):
                        _ingest_fallback_count += splitter.fallback_chunks_produced
                else:
                    chunked.append(doc)
            documents = chunked

        # TASK 13B: Per-source chunk budget enforcement
        _max_chunks = getattr(self.config, "max_chunks_per_source", 0)
        if _max_chunks > 0:
            from collections import defaultdict as _dfl_b

            _chunks_by_source: dict = _dfl_b(list)
            for _d in documents:
                _src = _d.get("metadata", {}).get("source", _d["id"])
                _chunks_by_source[_src].append(_d)
            _capped: list = []
            for _src, _src_chunks in _chunks_by_source.items():
                if len(_src_chunks) > _max_chunks:
                    logger.info(
                        "   Chunk cap: '%s' truncated %d → %d chunks (max_chunks_per_source=%d)",
                        _src,
                        len(_src_chunks),
                        _max_chunks,
                        _max_chunks,
                    )
                    _capped.extend(_src_chunks[:_max_chunks])
                else:
                    _capped.extend(_src_chunks)
            documents = _capped

        if self.config.dedup_on_ingest:
            before = len(documents)
            new_docs = []
            new_hashes = []
            for doc in documents:
                h = self._doc_hash(doc)
                if h not in self._ingested_hashes:
                    new_docs.append(doc)
                    new_hashes.append(h)
            skipped = before - len(new_docs)
            if skipped:
                logger.info(
                    f"   Dedup: skipped {skipped} already-seen chunk(s), ingesting {len(new_docs)} new."
                )
            documents = new_docs
            if not documents:
                return
            self._ingested_hashes.update(new_hashes)
            self._save_hash_store()

        # Contextual retrieval: prepend LLM context to each chunk
        if self.config.contextual_retrieval and self.config.dataset_type in {
            "doc",
            "paper",
            "discussion",
        }:
            raw_docs = documents
            whole_doc_text = " ".join(d.get("text", "") for d in raw_docs)
            documents = [
                self._prepend_contextual_context(chunk, whole_doc_text) for chunk in raw_docs
            ]

        # RAPTOR: generate summarisation nodes for the deduplicated leaf chunks
        if self.config.raptor:
            # TASK 12: Source-size guard — skip RAPTOR for sources whose estimated text size exceeds threshold
            _raptor_max_mb = getattr(self.config, "raptor_max_source_size_mb", 0.0)
            if _raptor_max_mb > 0.0:
                from collections import defaultdict as _dfl

                _size_by_source: dict = _dfl(int)
                for _d in documents:
                    _src = _d.get("metadata", {}).get("source", _d["id"])
                    _size_by_source[_src] += len(_d.get("text", ""))
                _max_bytes = int(_raptor_max_mb * 1024 * 1024)
                _skipped_sources = {src for src, sz in _size_by_source.items() if sz > _max_bytes}
                if _skipped_sources:
                    logger.info(
                        "   RAPTOR: skipping %d large source(s) > %.1f MB",
                        len(_skipped_sources),
                        _raptor_max_mb,
                    )
                _raptor_eligible = [
                    _d
                    for _d in documents
                    if _d.get("metadata", {}).get("source", _d["id"]) not in _skipped_sources
                ]
            else:
                _raptor_eligible = documents
            if _policy_on:
                _raptor_ok_list: list = []
                _raptor_pol_skipped: set = set()
                for _d in _raptor_eligible:
                    _dtype = _d.get("metadata", {}).get("dataset_type", "doc")
                    _r_ok, _ = AxonBrain._SOURCE_POLICY.get(
                        _dtype, AxonBrain._SOURCE_POLICY_DEFAULT
                    )
                    if _r_ok:
                        _raptor_ok_list.append(_d)
                    else:
                        _raptor_pol_skipped.add(_d.get("metadata", {}).get("source", _d["id"]))
                if _raptor_pol_skipped:
                    logger.info(
                        "   RAPTOR: source_policy skipped %d source(s)",
                        len(_raptor_pol_skipped),
                    )
                _raptor_eligible = _raptor_ok_list
            if _raptor_eligible:
                raptor_docs = self._generate_raptor_summaries(_raptor_eligible)
                documents = documents + raptor_docs

        # GraphRAG: extract entities from new chunks and update entity graph
        if self.config.graph_rag:
            updated = False
            # Only extract entities from actual document chunks (optionally include RAPTOR level-1)
            _include_raptor = getattr(self.config, "graph_rag_include_raptor_summaries", False)

            # P2: Skip GraphRAG entity extraction for large sources when raptor=True.
            # Sources with >= raptor_graphrag_leaf_skip_threshold leaf chunks bypass
            # extraction; their RAPTOR summaries still enter GraphRAG if the include flag is set.
            _skip_threshold = getattr(self.config, "raptor_graphrag_leaf_skip_threshold", 20)
            _leaf_count_by_source: dict = {}
            for _doc in documents:
                if not _doc.get("metadata", {}).get("raptor_level"):
                    _src = _doc.get("metadata", {}).get("source", _doc["id"])
                    _leaf_count_by_source[_src] = _leaf_count_by_source.get(_src, 0) + 1

            _large_sources: set = set()
            if self.config.raptor and _skip_threshold > 0:
                _large_sources = {
                    src for src, cnt in _leaf_count_by_source.items() if cnt >= _skip_threshold
                }
                if _large_sources:
                    logger.info(
                        f"   GraphRAG: skipping leaf-chunk entity extraction for "
                        f"{len(_large_sources)} large source(s) (>= {_skip_threshold} leaf chunks)"
                    )

            chunks_to_process = []
            for _doc in documents:
                _lvl = _doc.get("metadata", {}).get("raptor_level")
                _src = _doc.get("metadata", {}).get("source", _doc["id"])
                if not _lvl:  # leaf chunk
                    if _src not in _large_sources:
                        chunks_to_process.append(_doc)
                    # else: leaf from large source → skip; RAPTOR summary will cover it
                elif _lvl == 1:  # RAPTOR level-1 summary
                    # Auto-include for large sources when RAPTOR is on (regardless of include flag)
                    _auto_raptor = self.config.raptor and _src in _large_sources
                    if _include_raptor or _auto_raptor:
                        chunks_to_process.append(_doc)

            if _policy_on:
                _grag_ok_list: list = []
                _grag_pol_skipped: set = set()
                for _d in chunks_to_process:
                    _dtype = _d.get("metadata", {}).get("dataset_type", "doc")
                    _, _g_ok = AxonBrain._SOURCE_POLICY.get(
                        _dtype, AxonBrain._SOURCE_POLICY_DEFAULT
                    )
                    if _g_ok:
                        _grag_ok_list.append(_d)
                    else:
                        _grag_pol_skipped.add(_d.get("metadata", {}).get("source", _d["id"]))
                if _grag_pol_skipped:
                    logger.info(
                        "   GraphRAG: source_policy skipped %d source(s)",
                        len(_grag_pol_skipped),
                    )
                chunks_to_process = _grag_ok_list

            logger.info(f"   GraphRAG: Extracting entities from {len(chunks_to_process)} chunks...")

            def _proc(doc):
                return doc["id"], self._extract_entities(doc["text"])

            results = list(self._executor.map(_proc, chunks_to_process))

            # Track entity keys extracted this run for embedding (Item 5)
            entities_extracted_this_run: list = []

            total_entities = 0
            # Build a lookup from doc_id to doc for metadata writing (Item 7)
            doc_by_id = {doc["id"]: doc for doc in chunks_to_process}
            for doc_id, entities in results:
                total_entities += len(entities)
                # Track entity keys for embedding
                for ent in entities:
                    if isinstance(ent, dict) and ent.get("name"):
                        entities_extracted_this_run.append(ent)

                for ent in entities:  # ent is now {"name": ..., "type": ..., "description": ...}
                    key = ent["name"].lower().strip() if isinstance(ent, dict) else ent.lower()
                    if not key:
                        continue
                    if key not in self._entity_graph:
                        desc = ent.get("description", "") if isinstance(ent, dict) else ""
                        ent_type = (
                            ent.get("type", "UNKNOWN") if isinstance(ent, dict) else "UNKNOWN"
                        )
                        self._entity_graph[key] = {
                            "description": desc,
                            "type": ent_type,
                            "chunk_ids": [],
                            "frequency": 0,
                            "degree": 0,
                        }
                    elif isinstance(self._entity_graph[key], dict):
                        # Update type if not yet set
                        if (
                            not self._entity_graph[key].get("type")
                            or self._entity_graph[key].get("type") == "UNKNOWN"
                        ):
                            new_type = (
                                ent.get("type", "UNKNOWN") if isinstance(ent, dict) else "UNKNOWN"
                            )
                            if new_type and new_type != "UNKNOWN":
                                self._entity_graph[key]["type"] = new_type
                        # Item 10: collect descriptions for canonicalization
                        if isinstance(ent, dict) and ent.get("description"):
                            desc_buf = self._entity_description_buffer.setdefault(key, [])
                            desc_buf.append(ent["description"])
                        if (
                            not self._entity_graph[key].get("description")
                            and isinstance(ent, dict)
                            and ent.get("description")
                        ):
                            self._entity_graph[key]["description"] = ent["description"]
                    if isinstance(self._entity_graph[key], dict):
                        self._entity_graph[key].setdefault("chunk_ids", [])
                        if doc_id not in self._entity_graph[key]["chunk_ids"]:
                            self._entity_graph[key]["chunk_ids"].append(doc_id)
                            updated = True
                            self._community_graph_dirty = True
                    else:
                        # Legacy list format — migrate on the fly
                        if doc_id not in self._entity_graph[key]:
                            self._entity_graph[key].append(doc_id)
                            updated = True
                            self._community_graph_dirty = True

                # Item 7: Write entity IDs back into chunk metadata for text-unit linkage
                doc = doc_by_id.get(doc_id)
                if doc is not None and entities and doc.get("metadata") is not None:
                    doc["metadata"]["entity_ids"] = [
                        e["name"].lower() for e in entities if isinstance(e, dict) and e.get("name")
                    ]
                # GAP 9: Update text_unit_entity_map
                self._text_unit_entity_map[doc_id] = [
                    e["name"] for e in entities if isinstance(e, dict) and e.get("name")
                ]

            # Item 2: Update frequency only for entities touched in this ingest run
            # (avoids O(|V|) scan of the full entity graph on every ingest batch)
            _touched_entity_keys = {
                ent["name"].lower() for ent in entities_extracted_this_run if ent.get("name")
            }
            for entity_key in _touched_entity_keys:
                node = self._entity_graph.get(entity_key)
                if isinstance(node, dict):
                    node["frequency"] = len(node.get("chunk_ids", []))

            if updated and not _defer_saves:
                self._save_entity_graph()
            if total_entities == 0:
                logger.warning(
                    "GraphRAG: entity extraction returned 0 entities across all chunks. "
                    "This may be caused by an LLM that is too small or refused to extract entities. "
                    "GraphRAG relationship expansion will have no effect for this ingestion."
                )

            # Relation extraction: build SUBJECT | RELATION | OBJECT triples
            if self.config.graph_rag_relations:
                # A2: skip relation extraction for chunks below the entity count threshold
                _min_ent = getattr(self.config, "graph_rag_min_entities_for_relations", 3)
                _entity_count_by_doc = {doc_id: len(ents) for doc_id, ents in results}
                if _min_ent > 0:
                    _rel_chunks = [
                        doc
                        for doc in chunks_to_process
                        if _entity_count_by_doc.get(doc["id"], 0) >= _min_ent
                    ]
                else:
                    _rel_chunks = chunks_to_process
                # Budget-based relation gating: rank by entity density, cap at budget
                _rel_budget = getattr(self.config, "graph_rag_relation_budget", 0)
                if _rel_budget > 0 and len(_rel_chunks) > _rel_budget:
                    _rel_chunks = sorted(
                        _rel_chunks,
                        key=lambda d: _entity_count_by_doc.get(d["id"], 0)
                        / max(len(d.get("text", "")), 1),
                        reverse=True,
                    )[:_rel_budget]
                    logger.info(
                        f"   GraphRAG: Extracting relations from {len(_rel_chunks)} chunks "
                        f"(budget cap; {len(chunks_to_process) - len(_rel_chunks)} skipped)..."
                    )
                else:
                    logger.info(
                        f"   GraphRAG: Extracting relations from {len(_rel_chunks)} chunks "
                        f"(skipped {len(chunks_to_process) - len(_rel_chunks)} below "
                        f"{_min_ent}-entity threshold)..."
                    )

                def _proc_rel(doc):
                    return doc["id"], self._extract_relations(doc["text"])

                rel_results = list(self._executor.map(_proc_rel, _rel_chunks))
                rg_updated = False
                for doc_id, triples in rel_results:
                    for triple in triples:
                        # triple is now a dict: {subject, relation, object, description}
                        if isinstance(triple, dict):
                            subject = triple.get("subject", "")
                            relation = triple.get("relation", "")
                            obj = triple.get("object", "")
                            description = triple.get("description", "")
                        else:
                            # Legacy tuple format fallback
                            subject, relation, obj = triple
                            description = ""
                        src_lower = subject.lower().strip()
                        if not src_lower:
                            continue
                        entry = {
                            "target": obj.lower().strip(),
                            "relation": relation.strip(),
                            "chunk_id": doc_id,
                            "description": description,
                            "strength": triple.get("strength", 5)
                            if isinstance(triple, dict)
                            else 5,
                            "support_count": 1,
                        }
                        if src_lower not in self._relation_graph:
                            self._relation_graph[src_lower] = []
                        # Item 8: weight tracking — increment weight for same (target, relation) pair
                        rel_tgt = entry["target"]
                        rel_relation = entry["relation"]
                        existing_entry = next(
                            (
                                e
                                for e in self._relation_graph[src_lower]
                                if e.get("target") == rel_tgt and e.get("relation") == rel_relation
                            ),
                            None,
                        )
                        if existing_entry:
                            # Accumulate strength-based weight (sum of LM-derived strengths)
                            existing_entry["weight"] = existing_entry.get("weight", 1) + entry.get(
                                "strength", 1
                            )
                            existing_entry["support_count"] = (
                                existing_entry.get("support_count", 1) + 1
                            )
                            # GAP 7: accumulate text_unit_ids
                            if "text_unit_ids" not in existing_entry:
                                existing_entry["text_unit_ids"] = [
                                    existing_entry.get("chunk_id", "")
                                ]
                            if doc_id not in existing_entry["text_unit_ids"]:
                                existing_entry["text_unit_ids"].append(doc_id)
                            rg_updated = True
                        else:
                            entry["weight"] = entry.get("strength", 1)
                            entry["text_unit_ids"] = [doc_id]
                            self._relation_graph[src_lower].append(entry)
                            rg_updated = True
                        # GAP 3b: update relation description buffer
                        if description:
                            pair = (src_lower, rel_tgt)
                            if pair not in self._relation_description_buffer:
                                self._relation_description_buffer[pair] = []
                            self._relation_description_buffer[pair].append(description)
                if rg_updated and not _defer_saves:
                    self._save_relation_graph()

                if getattr(self.config, "graph_rag_relation_backend", "llm") == "rebel":
                    _rg_edge_count = sum(len(v) for v in self._relation_graph.values())
                    if _rg_edge_count == 0 and len(_rel_chunks) > 0:
                        logger.warning(
                            "GraphRAG REBEL: processed %d chunks but produced 0 relation edges. "
                            "If using a local model path, verify the checkpoint contains pretrained weights "
                            "(a 'newly initialized weights' warning from transformers indicates an invalid checkpoint).",
                            len(_rel_chunks),
                        )
                    else:
                        logger.info(
                            "GraphRAG REBEL: %d relation edges from %d chunks.",
                            _rg_edge_count,
                            len(_rel_chunks),
                        )

                # TASK 11: Normalize relation targets into entity graph so traversal never KeyErrors
                if rg_updated or updated:
                    _stub_added = False
                    for _src, _entries in self._relation_graph.items():
                        for _entry in _entries:
                            _tgt = _entry.get("target", "").lower().strip()
                            if not _tgt:
                                continue
                            if _tgt not in self._entity_graph:
                                self._entity_graph[_tgt] = {
                                    "description": "",
                                    "type": "UNKNOWN",
                                    "chunk_ids": [],
                                    "frequency": 0,
                                    "degree": 0,
                                }
                                _stub_added = True
                            # Ensure the relation's source chunk is in the target's chunk_ids
                            _cid = _entry.get("chunk_id", "")
                            if _cid:
                                _tgt_node = self._entity_graph[_tgt]
                                if isinstance(_tgt_node, dict):
                                    _tgt_node.setdefault("chunk_ids", [])
                                    if _cid not in _tgt_node["chunk_ids"]:
                                        _tgt_node["chunk_ids"].append(_cid)
                                        _tgt_node["frequency"] = len(_tgt_node["chunk_ids"])
                                        _stub_added = True
                    if _stub_added and not _defer_saves:
                        self._save_entity_graph()

                # GAP 9: Update text_unit_relation_map
                for doc_id, triples in rel_results:
                    self._text_unit_relation_map[doc_id] = [
                        (t.get("subject", ""), t.get("object", ""))
                        if isinstance(t, dict)
                        else (t[0], t[2])
                        for t in triples
                    ]

            # Item 2: Recompute degree for entities touched by this ingest's relations only
            # (avoids O(|V|) scan of the full entity graph on every ingest batch)
            for entity_key in _touched_entity_keys:
                if isinstance(self._entity_graph.get(entity_key), dict):
                    self._entity_graph[entity_key]["degree"] = len(
                        self._relation_graph.get(entity_key, [])
                    )

            # Item 5: Embed entity descriptions for query-time matching
            if getattr(self.config, "graph_rag_entity_embedding_match", True):
                entity_keys_this_batch = list(
                    {ent["name"].lower() for ent in entities_extracted_this_run if ent.get("name")}
                )
                self._embed_entities(entity_keys_this_batch)

            # Item 10: Canonicalize entity descriptions
            # A3: also run for "deep" tier
            _depth = getattr(self.config, "graph_rag_depth", "standard")
            if self.config.graph_rag and (
                getattr(self.config, "graph_rag_canonicalize", False) or _depth == "deep"
            ):
                self._canonicalize_entity_descriptions()
            if self.config.graph_rag and getattr(
                self.config, "graph_rag_canonicalize_relations", False
            ):
                self._canonicalize_relation_descriptions()

            # Item 11: Extract claims
            # A3: also run for "deep" tier
            claims_changed = False
            if getattr(self.config, "graph_rag_claims", False) or _depth == "deep":
                logger.info(
                    f"   GraphRAG: Extracting claims from {len(chunks_to_process)} chunks..."
                )

                def _proc_claims(doc):
                    return doc["id"], self._extract_claims(doc["text"])

                claim_results = list(self._executor.map(_proc_claims, chunks_to_process))
                for doc_id, claims in claim_results:
                    if claims:
                        # GAP 5: set text_unit_id on each claim
                        for claim in claims:
                            if isinstance(claim, dict):
                                claim["text_unit_id"] = doc_id
                        self._claims_graph[doc_id] = claims
                        claims_changed = True
                if claims_changed and not _defer_saves:
                    self._save_claims_graph()

            if self.config.graph_rag_community and self._community_graph_dirty:
                if getattr(self.config, "graph_rag_community_defer", True):
                    pass  # leave dirty; caller must invoke finalize_graph()
                else:
                    self._community_graph_dirty = False
                    if self.config.graph_rag_community_async:

                        def _debounced_rebuild():
                            import time as _time

                            self._community_build_in_progress = True
                            try:
                                _time.sleep(self.config.graph_rag_community_rebuild_debounce_s)
                                self._rebuild_communities()
                            finally:
                                self._community_build_in_progress = False

                        self._executor.submit(_debounced_rebuild)
                    else:
                        self._rebuild_communities()

        n_chunks = len(documents)
        if self._own_bm25:
            self._own_bm25.add_documents(documents, save_deferred=_defer_saves)

        ids = [d["id"] for d in documents]
        texts = [d["text"] for d in documents]
        metadatas = [d.get("metadata", {}) for d in documents]

        logger.info("   Generating embeddings...")
        batch_size = 32
        all_embeddings = []
        t_embed = time.time()
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            all_embeddings.extend(self.embedding.embed(texts[i : i + batch_size]))
        embed_ms = (time.time() - t_embed) * 1000

        t_store = time.time()
        self._own_vector_store.add(ids, texts, all_embeddings, metadatas)
        store_ms = (time.time() - t_store) * 1000

        # Persist embedding meta after first successful ingest (idempotent on subsequent calls)
        self._save_embedding_meta()

        # Update doc-version tracking: record content hash + chunk count per source path.
        # This powers /tracked-docs and /ingest/refresh.
        import hashlib as _hl
        import time as _time_mod

        _chunks_by_src: dict = {}
        for _d in documents:
            _src = _d.get("metadata", {}).get("source", _d["id"])
            _chunks_by_src.setdefault(_src, []).append(_d)
        for _src, _src_docs in _chunks_by_src.items():
            _combined = "".join(d.get("text", "") for d in _src_docs)
            _content_hash = _hl.md5(_combined.encode("utf-8", errors="replace")).hexdigest()
            self._doc_versions[_src] = {
                "content_hash": _content_hash,
                "chunk_count": len(_src_docs),
                "ingested_at": _time_mod.strftime("%Y-%m-%dT%H:%M:%SZ", _time_mod.gmtime()),
            }
        self._save_doc_versions()

        # ── Code graph (Phase 2 + Phase 3) ──────────────────────────────
        if getattr(self.config, "code_graph", False):
            _code_chunks = [
                d for d in documents if d.get("metadata", {}).get("source_class") == "code"
            ]
            if _code_chunks:
                self._build_code_graph_from_chunks(_code_chunks)
                logger.info("   Code graph: %d code chunks indexed.", len(_code_chunks))
            if getattr(self.config, "code_graph_bridge", False):
                _prose_chunks = [
                    d for d in documents if d.get("metadata", {}).get("source_class") != "code"
                ]
                if _prose_chunks:
                    self._build_code_doc_bridge(_prose_chunks)
                    logger.info(
                        "   Code graph bridge: scanned %d prose chunks.", len(_prose_chunks)
                    )
            if not _defer_saves:
                self._save_code_graph()

        logger.info(
            {
                "event": "ingest_complete",
                "chunks": n_chunks,
                "embed_ms": round(embed_ms, 1),
                "store_ms": round(store_ms, 1),
                "total_ms": round((time.time() - t0) * 1000, 1),
                "entity_count": len(self._entity_graph),
                "relation_edge_count": sum(len(v) for v in self._relation_graph.values()),
                "fallback_chunks": _ingest_fallback_count,
            }
        )

        # Phase 7: ingest diagnostics — source IDs and collision check
        if n_chunks > 0:
            _batch_source_ids = {
                chunk.get("metadata", {}).get("source_id", "")
                for chunk in documents
                if chunk.get("metadata", {}).get("source_id", "")
            }
            _batch_chunk_ids = [chunk.get("id", "") for chunk in documents if chunk.get("id", "")]
            _collision_count = len(_batch_chunk_ids) - len(set(_batch_chunk_ids))
            logger.info(
                "Ingest diagnostics  |  source_ids: %d  |  chunk_ids: %d  |  collisions: %d",
                len(_batch_source_ids),
                len(_batch_chunk_ids),
                _collision_count,
            )
            if _collision_count > 0:
                logger.warning(
                    "Ingest: %d duplicate chunk IDs detected in this batch. "
                    "This may indicate basename-derived IDs colliding. Re-ingest with current version to fix.",
                    _collision_count,
                )

        # Phase 3: explicitly release lease (fallback: _WriteLease.__del__ handles it)
        _ingest_lease.close()

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

    def _compress_context(self, query: str, results: list[dict]) -> list[dict]:
        """Extract only query-relevant sentences from each retrieved chunk (parallel).

        Compresses non-web results by asking the LLM to strip irrelevant text.
        Falls back to the original chunk if compression fails or makes it longer.
        """
        if not results:
            return results

        from concurrent.futures import ThreadPoolExecutor

        def _compress_one(result: dict) -> dict:
            if result.get("is_web"):
                return result
            # Use parent_text if available (small-to-big path), else chunk text
            source_text = result.get("metadata", {}).get("parent_text") or result["text"]
            prompt = (
                "Extract only the sentences from the passage below that directly help answer "
                "the question. Output only those sentences verbatim, nothing else. "
                "If no sentence is relevant, keep the single most informative sentence.\n\n"
                f"Question: {query}\n\nPassage:\n{source_text}"
            )
            try:
                compressed = self.llm.complete(
                    prompt, system_prompt="You are an expert at extracting relevant information."
                )
                if compressed and len(compressed.strip()) < len(source_text):
                    r = {**result, "metadata": {**result.get("metadata", {})}}
                    # Overwrite whichever field _build_context reads
                    if "parent_text" in r["metadata"]:
                        r["metadata"]["parent_text"] = compressed.strip()
                    else:
                        r["text"] = compressed.strip()
                    r["metadata"]["compressed"] = True
                    return r
            except Exception as e:
                logger.debug(f"Context compression failed for {result.get('id')}: {e}")
            return result

        with ThreadPoolExecutor(max_workers=min(len(results), 4)) as pool:
            return list(pool.map(_compress_one, results))

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
            for item in data.get("web", {}).get("results", [])[:count]:
                snippet = item.get("description", "")
                title = item.get("title", "")
                url = item.get("url", "")
                web_results.append(
                    {
                        "id": url,
                        "text": f"{title}\n{snippet}",
                        "score": 1.0,
                        "metadata": {"source": url, "title": title},
                        "is_web": True,
                    }
                )
            logger.info(f"Brave Search returned {len(web_results)} results for: {query[:60]}")
            return web_results
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    def _execute_retrieval(self, query: str, filters: dict = None, cfg=None) -> dict:
        """Central retrieval execution logic supporting HyDE, Multi-Query, and Web Search (Parallelized)."""
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

        if not filtered_results and cfg.truth_grounding:
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
            base = [r for r in results if not r.get("_graph_expanded")][:_top_k]
            expanded = [r for r in results if r.get("_graph_expanded")]
            base_ids = {r["id"] for r in base}
            graph_slots = [r for r in expanded if r["id"] not in base_ids][: cfg.graph_rag_budget]
            results = base + graph_slots
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

    def _build_system_prompt(self, has_web: bool, cfg=None) -> str:
        """Return the system prompt based on discussion_fallback and web search state.

        When discussion_fallback is False, uses a strict context-only prompt.
        When True, uses the permissive prompt.
        When cite is False, the citation instruction is removed from the prompt.
        """
        if cfg is None:
            cfg = self.config
        base = self.SYSTEM_PROMPT if cfg.discussion_fallback else self.SYSTEM_PROMPT_STRICT

        if not cfg.cite:
            # Strip citation instruction lines from the prompt
            import re as _re

            base = _re.sub(r"\d+\. \*\*Mandatory Citations\*\*:.*?\n", "", base)

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

    def _apply_overrides(self, overrides: dict | None) -> "AxonConfig":
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
                return self.llm.complete(
                    query, self._build_system_prompt(False, cfg=cfg), chat_history=chat_history
                )

            return "I don't have any relevant information to answer that question."

        if cfg.rerank:
            results = self.reranker.rerank(query, results)

        _top_k = retrieval.get("_effective_top_k", cfg.top_k)
        if cfg.graph_rag and cfg.graph_rag_budget > 0:
            base = [r for r in results if not r.get("_graph_expanded")][:_top_k]
            expanded = [r for r in results if r.get("_graph_expanded")]
            base_ids = {r["id"] for r in base}
            graph_slots = [r for r in expanded if r["id"] not in base_ids][: cfg.graph_rag_budget]
            results = base + graph_slots
        else:
            results = results[:_top_k]
        for r in results:
            r.pop("_graph_expanded", None)

        # P1: RAPTOR drill-down — replace summary hits with grounded leaf chunks
        if self.config.raptor and getattr(cfg, "raptor_drilldown", True):
            results = self._raptor_drilldown(query, results, cfg=cfg)

        # P4: Artifact-type ranking pass
        if self.config.raptor or self.config.graph_rag:
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
            results = self._compress_context(query, results)
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
        # TASK 12: Lazy mode — generate summaries on first global query if not yet generated
        if (
            cfg.graph_rag
            and graph_mode in ("global", "hybrid")
            and not self._community_summaries
            and self._community_levels
            and getattr(self.config, "graph_rag_community_lazy", False)
        ):
            self._generate_community_summaries(query_hint=query)
            if getattr(self.config, "graph_rag_index_community_reports", True):
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
                yield from self.llm.stream(
                    query, self._build_system_prompt(False, cfg=cfg), chat_history=chat_history
                )
                return
            yield "I don't have any relevant information to answer that question."
            return

        if cfg.rerank:
            results = self.reranker.rerank(query, results)

        _top_k = retrieval.get("_effective_top_k", cfg.top_k)
        if cfg.graph_rag and cfg.graph_rag_budget > 0:
            base = [r for r in results if not r.get("_graph_expanded")][:_top_k]
            expanded = [r for r in results if r.get("_graph_expanded")]
            base_ids = {r["id"] for r in base}
            graph_slots = [r for r in expanded if r["id"] not in base_ids][: cfg.graph_rag_budget]
            results = base + graph_slots
        else:
            results = results[:_top_k]
        for r in results:
            r.pop("_graph_expanded", None)

        # P1: RAPTOR drill-down — replace summary hits with grounded leaf chunks
        if self.config.raptor and getattr(cfg, "raptor_drilldown", True):
            results = self._raptor_drilldown(query, results, cfg=cfg)

        # P4: Artifact-type ranking pass
        if self.config.raptor or self.config.graph_rag:
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
            results = self._compress_context(query, results)

        # A4: Exclude community report synthetic docs from citation candidates.
        citation_results = [
            r
            for r in results
            if r.get("metadata", {}).get("graph_rag_type") != "community_report"
            and not r.get("id", "").startswith("__community__")
        ]
        context, has_web = self._build_context(citation_results)

        # GraphRAG global context injection
        # TASK 12: Lazy mode — generate summaries on first global query if not yet generated
        if (
            cfg.graph_rag
            and graph_mode in ("global", "hybrid")
            and not self._community_summaries
            and self._community_levels
            and getattr(self.config, "graph_rag_community_lazy", False)
        ):
            self._generate_community_summaries(query_hint=query)
            if getattr(self.config, "graph_rag_index_community_reports", True):
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


# ---------------------------------------------------------------------------
# Phase 3 re-exports — backward-compat, existing callers need no changes
# ---------------------------------------------------------------------------
from axon.cli import _print_project_tree, _write_python_discovery, main  # noqa: E402,F401
from axon.repl import (  # noqa: E402,F401
    _AT_DIR_MAX_BYTES,
    _AT_FILE_MAX_BYTES,
    _AT_LOADER_EXTS,
    _AT_TEXT_EXTS,
    _AXON_ART,
    _AXON_BLUE,
    _AXON_RST,
    _BRAIN_ART,
    _HINT,
    _MODEL_CTX,
    _SLASH_COMMANDS,
    _anim_pad,
    _box_width,
    _brow,
    _build_header,
    _do_compact,
    _draw_header,
    _estimate_tokens,
    _expand_at_files,
    _get_brain_anim_row,
    _infer_provider,
    _InitDisplay,
    _interactive_repl,
    _make_completer,
    _print_recent_turns,
    _prompt_key_if_missing,
    _save_env_key,
    _show_context,
    _token_bar,
)
from axon.sessions import (  # noqa: E402,F401
    _SESSIONS_DIR,
    _list_sessions,
    _load_session,
    _new_session,
    _print_sessions,
    _save_session,
    _session_path,
    _sessions_dir,
)

if __name__ == "__main__":
    main()
