"""


src/axon/config.py


AxonConfig dataclass extracted from main.py for Phase 2 of the Axon refactor.


"""


import difflib
import getpass
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore

logger = logging.getLogger("Axon")


# XDG-style user config dir --' consistent across Linux / macOS / Windows


_USER_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".config", "axon", "config.yaml")


# ---------------------------------------------------------------------------


# Canonical default config — single source of truth used by every entry point


# (CLI --config-reset, REPL /config reset, POST /config/reset, wizard).


# ---------------------------------------------------------------------------


_DEFAULT_CONFIG_YAML = """\


# Axon Configuration — edit to customise behaviour.


# Full option reference: axon --help  or  docs/ADMIN_REFERENCE.md


embedding:


  # Model used to convert text into vectors.


  # sentence_transformers runs locally; openai/ollama/fastembed also supported.


  provider: sentence_transformers


  model: all-MiniLM-L6-v2            # Replace with a larger model for better recall


llm:


  # Language model used for answer generation and advanced RAG strategies.


  # ollama runs locally; openai / gemini / grok / vllm need API keys.


  provider: ollama


  model: llama3.1:8b


  temperature: 0.7                   # 0.0 = deterministic  1.0 = creative


  max_tokens: 2048                   # Max tokens in the generated answer


vector_store:


  # Persistent vector store for embeddings.


  # lancedb (default) requires no extra service; qdrant supports remote mode.


  provider: lancedb


  path: ~/.axon/projects/default/lancedb_data


bm25:


  path: ~/.axon/projects/default/bm25_index


rag:


  top_k: 10                          # Chunks retrieved per query (recommended: 5–30)


  similarity_threshold: 0.3          # Min cosine similarity to include a chunk (0.0–1.0)


  hybrid_search: true                # Combine vector + BM25 sparse retrieval (recommended)


  rerank: false                      # BGE cross-encoder reranker — better precision, slower


  sentence_window: false             # Expand retrieved chunks with surrounding sentences


  sentence_window_size: 3            # Sentences of context on each side (recommended: 1–5)


  # ── Advanced strategies — each adds extra LLM calls ─────────────────────────


  # raptor: builds a hierarchical summary tree at ingest time.


  # WARNING: raptor=true triggers many LLM calls during ingest. Suitable for


  #   large document sets (>5 MB); avoid for small corpora.


  raptor: false


  raptor_min_source_size_mb: 5.0     # Skip RAPTOR for sources smaller than this MB (0 = no filter)


  # graph_rag: extracts entity relationships and uses them at query time.


  # WARNING: graph_rag=true triggers many LLM calls during ingest (entity


  #   extraction) AND at query time (graph traversal). Enable only when


  #   reasoning over entity relationships is essential.


  graph_rag: false


  graph_rag_community: false         # Community-level summaries (even more LLM calls)


chunk:


  strategy: semantic                 # recursive | semantic | markdown | cosine_semantic


  size: 1000                         # Target chunk size in tokens (recommended: 400–2000)


  overlap: 200                       # Token overlap between adjacent chunks (recommended: 50–400)


rerank:


  enabled: false


  provider: cross-encoder


query_transformations:


  multi_query: false                 # Generate multiple query paraphrases (adds 1 LLM call)


  hyde: false                        # Hypothetical document embedding (adds 1 LLM call)


  discussion_fallback: true          # Fall back to general LLM answer when retrieval confidence is low


repl:


  shell_passthrough: local_only      # Allow ! shell commands: local_only | any | disabled


web_search:


  enabled: false


offline:


  enabled: false


  local_assets_only: false


  local_models_dir: ""               # Path to local HF model cache for offline embedding


"""


# ---------------------------------------------------------------------------


# Known YAML keys per section — used for structural validation.


# ---------------------------------------------------------------------------


_KNOWN_YAML_KEYS: dict[str, set[str]] = {
    "llm": {
        "provider",
        "model",
        "base_url",
        "models_dir",
        "openai_api_key",
        "grok_api_key",
        "gemini_api_key",
        "vllm_base_url",
        "temperature",
        "max_tokens",
        "api_key",
        "timeout",
        "ollama_cloud_key",
        "ollama_cloud_url",
    },
    "embedding": {"provider", "model", "model_path"},
    "vector_store": {
        "provider",
        "path",
        "qdrant_url",
        "qdrant_api_key",
        "qdrant_collection",
        "lancedb_path",
        "tqdb_bits",
    },
    "rag": {
        "hybrid_search",
        "hyde",
        "multi_query",
        "step_back",
        "decompose",
        "compress",
        "rerank",
        "sentence_window",
        "sentence_window_size",
        "crag_lite",
        "crag_lite_confidence_threshold",
        "cite",
        "raptor",
        "graph_rag",
        "discuss",
        "top_k",
        "similarity_threshold",
        "parent_doc",
        "parent_chunk_size",
        "query_router",
        "code_graph",
        "code_graph_bridge",
        "graph_rag_mode",
        "graph_rag_depth",
        "graph_rag_budget",
        "graph_rag_relations",
        "graph_rag_community",
        "graph_rag_community_backend",
        "graph_rag_relation_budget",
        "graph_rag_entity_min_frequency",
        "graph_rag_global_top_communities",
        "raptor_max_levels",
        "raptor_min_source_size_mb",
        "truth_grounding",
        "rerank_top_k",
        "discussion_fallback",
        "hybrid_weight",
        "hybrid_mode",
        "raptor_chunk_group_size",
        "dedup_on_ingest",
    },
    "chunk": {
        "strategy",
        "size",
        "overlap",
        "cosine_semantic_threshold",
        "parent_chunk_size",
        "cosine_semantic_max_size",
    },
    "rerank": {"enabled", "model", "top_k", "provider"},
    "offline": {
        "enabled",
        "local_assets_only",
        "local_models_dir",
        "embedding_models_dir",
        "hf_models_dir",
        "tokenizer_cache_dir",
    },
    "web_search": {"enabled", "brave_api_key", "num_results", "safe_search"},
    "store": {"base"},
    "repl": {"shell_passthrough"},
    "api": {"key", "allow_origins"},
    "bm25": {"path"},
    "query_transformations": {
        "multi_query",
        "hyde",
        "step_back",
        "query_decompose",
        "discussion_fallback",
    },
    "context_compression": {"enabled", "strategy", "token_budget"},
    "advanced": {"raptor", "graph_rag", "graph_rag_community"},
}


@dataclass
class ConfigIssue:

    """A single validation finding for a config.yaml field."""

    level: Literal["error", "warn", "info"]

    section: str

    field: str

    message: str

    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "section": self.section,
            "field": self.field,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass
class AxonConfig:

    """Configuration for Axon."""

    # Internal tracking

    _loaded_path: str | None = None

    # Embedding

    embedding_provider: Literal[
        "sentence_transformers", "ollama", "fastembed", "openai"
    ] = "sentence_transformers"

    embedding_model: str = "all-MiniLM-L6-v2"

    # Local path override for the embedding model (sentence_transformers / fastembed).

    # When set, this path is passed directly to the model loader instead of downloading.

    # sentence_transformers: absolute path to a local model folder.

    # fastembed: treated as cache_dir so the model is loaded from there.

    # Takes precedence over embedding_model when non-empty.

    embedding_model_path: str = ""

    ollama_base_url: str = "http://localhost:11434"

    # Local directory where Ollama stores its model blobs.

    # Equivalent to setting the OLLAMA_MODELS environment variable.

    # Useful when models live on a secondary disk or network share.

    # Can also be set via the OLLAMA_MODELS env var (env var takes priority).

    ollama_models_dir: str = ""

    # LLM

    llm_provider: Literal[
        "ollama", "gemini", "ollama_cloud", "openai", "vllm", "copilot", "github_copilot", "grok"
    ] = "ollama"

    llm_model: str = "llama3.1:8b"

    llm_temperature: float = 0.7

    llm_max_tokens: int = 2048

    api_key: str = ""  # legacy alias -- prefer openai_api_key

    openai_api_key: str = ""

    grok_api_key: str = ""

    gemini_api_key: str = ""

    ollama_cloud_key: str = ""

    ollama_cloud_url: str = ""

    vllm_base_url: str = "http://localhost:8000/v1"

    # GitHub OAuth token for the "github_copilot" provider.

    # Obtained via the OAuth device flow (/keys set github_copilot).

    # Classic PATs are NOT accepted by the Copilot API.

    # Can also be set via GITHUB_COPILOT_PAT env var.

    copilot_pat: str = ""

    # Projects

    # Root directory for all named projects.

    # Always derived from axon_store_base as {base}/AxonStore/{username}/.

    # Do not set directly; override axon_store_base instead.

    projects_root: str = ""

    # Vector Store

    vector_store: Literal["chroma", "qdrant", "lancedb", "turboquantdb"] = "lancedb"

    vector_store_path: str = ""

    tqdb_bits: int = 8

    # BM25 Settings

    bm25_path: str = ""

    def __post_init__(self) -> None:
        """Populate fields from environment variables and resolve storage paths."""

        # 1. API Keys and URLs

        if not self.api_key:
            self.api_key = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))

        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        if not self.grok_api_key:
            self.grok_api_key = os.getenv("XAI_API_KEY", os.getenv("GROK_API_KEY", ""))

        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")

        if not self.ollama_cloud_key:
            self.ollama_cloud_key = os.getenv("OLLAMA_CLOUD_KEY", "")

        if not self.ollama_cloud_url:
            self.ollama_cloud_url = os.getenv("OLLAMA_CLOUD_URL", "https://ollama.com/api")

        if self.vllm_base_url == "http://localhost:8000/v1":
            env_val = os.getenv("VLLM_BASE_URL")

            if env_val:
                self.vllm_base_url = env_val

        if not self.brave_api_key:
            self.brave_api_key = os.getenv("BRAVE_API_KEY", "")

        if not self.copilot_pat:
            self.copilot_pat = os.environ.get("GITHUB_COPILOT_PAT") or os.environ.get(
                "GITHUB_TOKEN", ""
            )

        # 2. Storage paths -- always derived from AxonStore layout.

        # Base defaults to ~/.axon; override via AXON_STORE_BASE env var or config.yaml store.base.

        env_store_base = os.getenv("AXON_STORE_BASE", "")

        if env_store_base and not self.axon_store_base:
            self.axon_store_base = env_store_base

        if not self.axon_store_base:
            self.axon_store_base = os.path.join(os.path.expanduser("~"), ".axon")

        import getpass

        username = getpass.getuser()

        store_root = Path(self.axon_store_base).expanduser().resolve() / "AxonStore"

        user_dir = store_root / username

        self.projects_root = str(user_dir)

        # Respect explicitly-provided absolute paths (e.g. in tests) — only set defaults.

        if not self.vector_store_path or not os.path.isabs(self.vector_store_path):
            self.vector_store_path = str(user_dir / "default" / "lancedb_data")

        if not self.bm25_path or not os.path.isabs(self.bm25_path):
            self.bm25_path = str(user_dir / "default" / "bm25_index")

    # RAG Settings

    top_k: int = 10

    similarity_threshold: float = 0.3

    hybrid_search: bool = True

    hybrid_weight: float = 0.7  # 1.0 = Pure Semantic, 0.0 = Pure Keyword

    hybrid_mode: Literal["weighted", "rrf"] = "rrf"  # Hybrid fusion mode (rrf is more robust)

    # Chunking

    chunk_strategy: Literal["recursive", "semantic", "markdown", "cosine_semantic"] = "semantic"

    chunk_size: int = 1000

    chunk_overlap: int = 200

    # Cosine semantic chunking (only active when chunk_strategy="cosine_semantic")

    cosine_semantic_threshold: float = 0.7

    cosine_semantic_max_size: int = 500

    # MMR deduplication --' reorders and removes near-duplicate retrieved chunks

    mmr: bool = False

    mmr_lambda: float = 0.5  # 1.0 = pure relevance, 0.0 = pure diversity

    # Sentence-Window Retrieval (Epic 1)

    # Indexes prose chunks at sentence granularity; retrieves by sentence but

    # expands each hit to Â+/-sentence_window_size surrounding sentences for LLM

    # context.  Only non-code, non-RAPTOR-summary leaf chunks are eligible.

    # Disabled by default; enable via config.yaml (rag.sentence_window: true).

    sentence_window: bool = False

    sentence_window_size: int = 3  # Â+/-N sentences around each sentence hit

    # CRAG-Lite Retrieval Correction (Epic 2)

    # Evaluates retrieval confidence before deciding whether to trust local

    # results or escalate to web fallback.  Operates without LLM calls.

    # Disabled by default; enable via config.yaml (rag.crag_lite: true).

    crag_lite: bool = False

    crag_lite_confidence_threshold: float = 0.4  # below â†' low-confidence fallback

    # Re-ranking

    rerank: bool = False

    reranker_provider: Literal["cross-encoder", "llm"] = "cross-encoder"

    # Default: fast ms-marco model (small, already commonly cached).

    # Upgrade to "BAAI/bge-reranker-v2-m3" for SOTA multilingual accuracy.

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Parent-Document / Small-to-Big Retrieval

    # Child chunks (chunk_size tokens) are indexed for precise retrieval;

    # parent chunks (parent_chunk_size tokens) are returned as LLM context.

    # Must be > chunk_size to have any effect.

    # 0 = disabled (retrieval chunks are used directly as LLM context).

    parent_chunk_size: int = 1500

    # Query Transformations

    multi_query: bool = False

    hyde: bool = False

    step_back: bool = False

    query_decompose: bool = False

    discussion_fallback: bool = True

    # Context Compression (Epic 3, Stories 3.1--'3.3)

    # compress_context: master on/off switch (backward-compatible).

    # compression_strategy selects the algorithm when compress_context is True:

    #   "sentence"   --' LLM-based sentence extraction (default, existing behaviour)

    #   "llmlingua"  --' LLMLingua-2 token compression (pip install axon[llmlingua])

    #   "none"       --' disabled (same as compress_context=False)

    # compression_token_budget: target output tokens for llmlingua (0 = use model default ratio).

    compress_context: bool = False

    compression_strategy: str = "sentence"  # "none" | "sentence" | "llmlingua"

    compression_token_budget: int = 0  # 0 = no explicit budget (llmlingua uses default ratio)

    # Inline Citations

    # When True, instructs the LLM to include [Document N (ID: ...)] citations in

    # its answers whenever it draws from retrieved context.

    cite: bool = True

    # Web Search / Truth Grounding

    truth_grounding: bool = False

    brave_api_key: str = ""

    # Query Result Caching

    query_cache: bool = False

    query_cache_size: int = 128

    # Ingest Deduplication

    dedup_on_ingest: bool = True

    # Offline Mode

    offline_mode: bool = False

    local_models_dir: str = ""  # legacy single-root, kept for backwards compat

    # Local-assets-only mode: enforce local model loading without disabling RAPTOR/GraphRAG.

    # Sets TRANSFORMERS_OFFLINE + HF_HUB_OFFLINE, resolves all model IDs to local paths,

    # but keeps GraphRAG and RAPTOR enabled (they use the LLM, not downloaded assets).

    # Use offline_mode for strict no-egress (also disables GraphRAG/RAPTOR LLM calls).

    local_assets_only: bool = False

    # Per-type model root directories (multi-root Phase 4).

    # Each takes precedence over local_models_dir when set.

    # embedding_models_dir: sentence-transformers / fastembed model root

    # hf_models_dir:        GLiNER, REBEL, LLMLingua, and cross-encoder reranker

    # tokenizer_cache_dir:  tiktoken BPE encoding cache (maps to TIKTOKEN_CACHE_DIR)

    embedding_models_dir: str = ""

    hf_models_dir: str = ""

    tokenizer_cache_dir: str = ""

    # RAPTOR Hierarchical Indexing

    # During ingest, groups every raptor_chunk_group_size consecutive chunks per source

    # and generates a summarisation node that is indexed alongside the leaf chunks.

    # At retrieval time these summary nodes surface high-level context for multi-hop

    # questions that no single leaf chunk can answer.

    raptor: bool = True

    raptor_chunk_group_size: int = 5

    raptor_max_levels: int = 2  # recursive summarization depth

    raptor_cache_summaries: bool = True  # skip LLM when window content unchanged

    raptor_drilldown: bool = True  # replace summary hits with leaf chunks

    raptor_drilldown_top_k: int = 5  # max leaves substituted per summary hit

    raptor_retrieval_mode: str = "tree_traversal"  # tree_traversal|summary_first|corpus_overview

    raptor_graphrag_leaf_skip_threshold: int = 3  # skip GraphRAG on sources >= N leaf chunks

    # GraphRAG Entity-Centric Retrieval

    # During ingest, named entities are extracted from each chunk via the LLM and

    # stored in an entityâ†'doc_id map.  At retrieval time, entities found in the

    # query are used to expand the result set with graph-connected documents.

    graph_rag: bool = True

    # Maximum number of graph-expanded (entity-linked) documents to inject beyond

    # the normal top_k slice.  Set to 0 to disable the guarantee and fall back to

    # the old behaviour of truncating the combined list to top_k.

    graph_rag_budget: int = 3

    # Relation extraction: extract SUBJECT | RELATION | OBJECT triples during ingest

    # and use 1-hop graph traversal at retrieval time for richer context expansion.

    graph_rag_relations: bool = True

    # Community detection: cluster the entity graph into thematic communities after ingest.

    # Requires: pip install networkx

    graph_rag_community: bool = True

    # Run community detection in the background (non-blocking) after ingest.

    graph_rag_community_async: bool = True

    # Number of top community summaries to inject into the prompt during global search.

    graph_rag_community_top_k: int = 5

    # GraphRAG query mode: "local" (entity/relation context), "global" (community summaries),

    # or "hybrid" (both).

    graph_rag_mode: str = "local"  # "local" | "global" | "hybrid"

    # Global search map-reduce parameters

    graph_rag_global_min_score: int = 20  # minimum map-phase score to include

    graph_rag_global_top_points: int = 50  # max points assembled in reduce phase

    graph_rag_community_level: int = 0  # which hierarchy level for global search

    # Hierarchical community detection

    graph_rag_community_levels: int = 2  # number of hierarchy levels

    # Entity embedding matching at query time

    graph_rag_entity_embedding_match: bool = True

    graph_rag_entity_match_threshold: float = 0.5

    # Community report vector store indexing

    graph_rag_index_community_reports: bool = True

    # Entity description canonicalization

    graph_rag_canonicalize: bool = False

    graph_rag_canonicalize_min_occurrences: int = 2  # GAP 8: was 3

    # Claim / covariate extraction (off by default)

    graph_rag_claims: bool = False

    # GAP 1: Global search reduce phase

    graph_rag_global_reduce_max_tokens: int = 8000

    graph_rag_global_map_max_length: int = 1000

    graph_rag_global_reduce_max_length: int = 2000

    graph_rag_global_allow_general_knowledge: bool = False

    # GAP 2: Hierarchical community detection parameters

    graph_rag_community_max_cluster_size: int = 10

    graph_rag_community_use_lcc: bool = False

    graph_rag_leiden_seed: int = 42

    # GAP 3a: Community summarization context budget

    graph_rag_community_max_context_tokens: int = 4000

    # GAP 3b: Relation description canonicalization

    graph_rag_canonicalize_relations: bool = False

    graph_rag_canonicalize_relations_min_occurrences: int = 2

    # GAP 3c: Include claims in community reports

    graph_rag_community_include_claims: bool = False

    # GAP 4: Local search token budget and ranking controls

    graph_rag_local_max_context_tokens: int = 8000

    graph_rag_local_community_prop: float = 0.25

    graph_rag_local_text_unit_prop: float = 0.5

    graph_rag_local_top_k_entities: int = 10

    graph_rag_local_top_k_relationships: int = 10

    graph_rag_local_include_relationship_weight: bool = False

    # Unified candidate ranking weights

    graph_rag_local_entity_weight: float = 3.0

    graph_rag_local_relation_weight: float = 2.0

    graph_rag_local_community_weight: float = 1.5

    graph_rag_local_text_unit_weight: float = 1.0

    # Runtime cost reduction --' community triage

    graph_rag_community_min_size: int = 3  # communities smaller than this â†' template only

    graph_rag_community_llm_top_n_per_level: int = 15  # max LLM-summarized per level (0=unlimited)

    graph_rag_community_llm_max_total: int = (
        30  # hard cap on LLM calls across all levels (0=unlimited)
    )

    # Lazy community generation --' skip summarization at finalize; generate on first global query

    graph_rag_community_lazy: bool = True

    # Global search pre-filter --' cap communities entering map-reduce (0=no cap)

    graph_rag_global_top_communities: int = 0

    # RAPTOR source-size guard — skip RAPTOR for sources smaller than this MB (0=no filter)

    raptor_min_source_size_mb: float = 5.0

    # Deferred batch saves --' suppress per-call disk writes during batch ingest.

    # When True: BM25, entity graph, and relation graph saves deferred to finalize_ingest().

    # Reduces O(NÂ²) disk writes to O(1) per session.

    # Crash recovery: in-memory state only; re-ingest affected sources on restart.

    ingest_batch_mode: bool = False

    # Per-source chunk count cap after splitting.

    # Prevents chunk explosion from large structured files (JSON, TSV, CSV).

    # Keeps first N chunks (document order). 0 = unlimited (current behavior).

    max_chunks_per_source: int = 0

    # When True, detected dataset_type gates RAPTOR and GraphRAG.

    # Tabular, manifest, and reference sources skip both enrichments.

    # False (default) = current behavior.

    source_policy_enabled: bool = False

    # GAP 6: Async rebuild debounce

    graph_rag_community_rebuild_debounce_s: float = 2.0

    # Exact-token entity boost in local search

    graph_rag_exact_entity_boost: float = 3.0

    # Deferred community rebuild (batch ingest mode)

    graph_rag_community_defer: bool = True

    # Include RAPTOR level-1 summaries in GraphRAG entity extraction.

    # Defaults to True so large-source RAPTOR summaries are used as GraphRAG units.

    graph_rag_include_raptor_summaries: bool = True

    # A2: Skip relation extraction for chunks with fewer than this many entities.

    # 0 = always extract. Saves ~30-50% of relation LLM calls on typical corpora.

    graph_rag_min_entities_for_relations: int = 3

    # Budget-based relation gating: max chunks to run relation extraction on per ingest batch.

    # Chunks are ranked by entity density (entities/text-length) and only the top-N are processed.

    # 0 = unlimited (fall back to entity-count threshold gate only).

    # Product default is 30: prevents unbounded LLM/REBEL calls on large corpora. Set to 0 only

    # for experiments where full coverage matters more than ingest cost.

    graph_rag_relation_budget: int = 30

    # Community detection backend preference.

    # "louvain"   = networkx Louvain only (default --' safe on all environments, fast for <10k nodes)

    # "leidenalg" = leidenalg/igraph multi-resolution Leiden (recommended when available)

    # "auto"      = graspologic â†' leidenalg â†' louvain fallback chain (legacy; unsafe on Python 3.13

    #               because graspologic's import can hang; use only when graspologic is verified safe)

    graph_rag_community_backend: str = "louvain"

    # Structural code graph.

    # Builds File/Symbol nodes and CONTAINS/IMPORTS edges from codebase chunk metadata.

    # code_graph_bridge: scans prose chunks for code symbol mentions â†' MENTIONED_IN edges.

    # Query time: traverses the code graph to expand retrieval results.

    code_graph: bool = False  # build + query structural code graph

    code_graph_bridge: bool = False  # link code symbols to prose chunks

    # Query-time lexical boost for code corpora.

    code_lexical_boost: bool = True  # apply identifier-aware re-scoring to code result sets

    code_top_k_multiplier: int = 2  # extra fetch_k factor when code query detected

    code_max_chunks_per_file: int = 3  # per-file cap in final top_k (diversity)

    # Code query mode tuning (active when code_lexical_boost=True and code query detected).

    # code_bm25_weight only affects weighted fusion mode --' silently ignored in RRF (default).

    code_bm25_weight: float = 0.7  # BM25 weight override for code queries (weighted mode only)

    code_top_k: int = 6  # top-K override when code mode active (0 = use top_k)

    # Retrieval dry-run: skip LLM, return ranked candidates + diagnostics only.

    retrieval_dry_run: bool = False

    # Minimum entity appearance frequency to include in community detection graph.

    # Entities appearing in fewer than this many chunks are pruned before building the graph.

    # 1 = no pruning (include all entities). 2 = prune singletons (recommended for non-trivial

    # corpora --' reduces noisy one-off entities; qualification studies used 2 for papers corpus).

    graph_rag_entity_min_frequency: int = 2

    # Dedicated thread pool size for map-reduce phase (0 = use max_workers).

    # When set, _global_search_map_reduce creates an isolated pool, preventing map-reduce

    # from starving the shared executor during concurrent ingest.

    graph_rag_map_workers: int = 0

    # Alternative NER backend. "gliner" skips LLM for entity extraction.

    # Relations and claims still use LLM. pip install axon[gliner]

    graph_rag_ner_backend: Literal["llm", "gliner"] = "llm"

    # A3: Extraction depth tier.

    # "light" = regex noun-phrase extractor, no LLM, no relations (fastest)

    # "standard" = current LLM-based NER (default)

    # "deep" = standard + claims + canonicalize

    graph_rag_depth: Literal["light", "standard", "deep"] = "standard"

    # Token-level compression of community reports before map-reduce LLM calls.

    # Uses LLMLingua-2. pip install axon[llmlingua]

    graph_rag_report_compress: bool = False

    graph_rag_report_compress_ratio: float = 0.5  # target compression (0.0--'1.0)

    # Auto-route queries based on complexity.

    # "heuristic": keyword-based, zero latency. "llm": one classifier LLM call.

    # "off" (default): use graph_rag_mode as configured.

    graph_rag_auto_route: Literal["off", "heuristic", "llm"] = "off"

    # Option B: Multi-class query router

    # "heuristic": keyword-based classifier (zero latency, default)

    # "llm": one LLM call per query to classify route

    # "off": skip router, use graph_rag_auto_route legacy behaviour

    query_router: str = "heuristic"

    # Contextual retrieval --' prepend LLM-generated situating context to each chunk at ingest time.

    # Based on Anthropic's contextual retrieval technique.

    contextual_retrieval: bool = False

    # Semantic entity alias resolution --' merge near-duplicate entity names (e.g.

    # "Apple" / "Apple Inc." / "Apple Corporation") into a single canonical node before

    # community detection.  Uses cosine similarity on entity-name embeddings.

    # pip install axon[graphrag]  (no extra deps --' uses the already-loaded embedding model)

    graph_rag_entity_resolve: bool = False

    graph_rag_entity_resolve_threshold: float = 0.92  # cosine similarity threshold (0--'1)

    graph_rag_entity_resolve_max: int = 5000  # skip if entity count exceeds this (perf guard)

    # Alternative relation extraction backend using REBEL (Babelscape/rebel-large).

    # "rebel" skips the LLM for relation extraction; produces structured (subject, relation,

    # object) triples directly from a fine-tuned seq2seq model.

    # pip install axon[rebel]

    graph_rag_relation_backend: Literal["llm", "rebel"] = "llm"

    graph_rag_rebel_model: str = "Babelscape/rebel-large"

    graph_rag_gliner_model: str = "urchade/gliner_medium-v2.1"

    graph_rag_llmlingua_model: str = (
        "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    )

    # LLM request timeout in seconds (applied where the provider client supports it)

    llm_timeout: int = 60

    # REPL shell passthrough policy for `!command`.

    # - local_only: allow only in local/default project modes (default)

    # - always: allow in all modes

    # - off: disable entirely

    repl_shell_passthrough: Literal["local_only", "always", "off"] = "local_only"

    # Maximum parallel worker threads for background ingestion and query tasks

    max_workers: int = 8

    # Dataset type for type-specific chunking. "auto" uses content-based heuristics.

    dataset_type: Literal[
        "auto", "codebase", "paper", "doc", "discussion", "knowledge", "manifest", "reference"
    ] = "auto"

    # Smart re-ingest: track doc versions and re-ingest only changed documents

    smart_ingest: bool = False

    # Qdrant remote connection settings (leave empty for local file mode)

    qdrant_url: str = ""

    qdrant_api_key: str = ""

    # AxonStore -- the only storage layout Axon uses.

    # projects_root is always {axon_store_base}/AxonStore/{os_username}/.

    # Default base: ~/.axon. Override via AXON_STORE_BASE env var or config.yaml store.base.

    # Call /store/init (or axon --store-init <path>) only when moving data to a shared drive.

    axon_store_base: str = ""

    @classmethod
    def load(cls, path: str | None = None) -> "AxonConfig":
        """Load configuration from a YAML file.

        Defaults to ``~/.config/axon/config.yaml``.  On first run (file absent)

        the directory is created and a starter config is written automatically,

        so the notice appears only once.  Passing an explicit *path* that does

        not exist still produces a WARNING.

        """

        using_default = path is None

        if using_default:
            path = str(_USER_CONFIG_PATH)

        assert path is not None

        if not os.path.exists(path):
            if using_default:
                try:
                    cfg_path = Path(path)

                    cfg_path.parent.mkdir(parents=True, exist_ok=True)

                    cfg_path.write_text(_DEFAULT_CONFIG_YAML, encoding="utf-8")

                    logger.info("Created default config at %s --' edit it to customise Axon.", path)

                    # Fall through so the newly-written file is parsed; do NOT return cls()

                    # here --' that would silently use the dataclass defaults (raptor=True etc.)

                    # instead of the file values (raptor=false etc.).

                except (OSError, PermissionError) as exc:
                    logger.warning(
                        "Could not create default config at %s (%s). Using in-memory defaults.",
                        path,
                        exc,
                    )

                    return cls()

            else:
                logger.warning("Config file %s not found. Using defaults.", path)

                return cls()

        try:
            with open(os.path.expanduser(path), encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        except (yaml.YAMLError, OSError) as exc:
            logger.warning(
                "Config file %s is unreadable or malformed (%s). Using defaults.", path, exc
            )

            return cls()

        # Flatten the YAML structure to match dataclass fields

        config_dict = {}

        if "embedding" in data:
            config_dict.update({f"embedding_{k}": v for k, v in data["embedding"].items()})

        if "llm" in data:
            config_dict.update({f"llm_{k}": v for k, v in data["llm"].items()})

        if "vector_store" in data:
            vs = data["vector_store"]

            if "provider" in vs:
                config_dict["vector_store"] = vs["provider"]

            if "tqdb_bits" in vs:
                config_dict["tqdb_bits"] = int(vs["tqdb_bits"])

            # vector_store_path is always derived from AxonStore in __post_init__
            # — ignore any path value in config.yaml.

        if "bm25" in data:
            pass  # bm25_path is always derived from AxonStore in __post_init__
            # — ignore any path value in config.yaml.

        if "rag" in data:
            config_dict.update(data["rag"])

        # Legacy: starter configs written before 2026-03-23 put raptor/graph_rag/graph_rag_community

        # under an "advanced:" section instead of "rag:". Parse it for backward compatibility.

        if "advanced" in data:
            for key in ("raptor", "graph_rag", "graph_rag_community"):
                if key in data["advanced"]:
                    config_dict[key] = data["advanced"][key]

        if "chunk" in data:
            config_dict.update({f"chunk_{k}": v for k, v in data["chunk"].items()})

        if "rerank" in data:
            if "enabled" in data["rerank"]:
                config_dict["rerank"] = data["rerank"]["enabled"]

            if "provider" in data["rerank"]:
                config_dict["reranker_provider"] = data["rerank"]["provider"]

            if "model" in data["rerank"]:
                config_dict["reranker_model"] = data["rerank"]["model"]

        if "query_transformations" in data:
            for key in (
                "multi_query",
                "hyde",
                "step_back",
                "query_decompose",
                "discussion_fallback",
            ):
                if key in data["query_transformations"]:
                    config_dict[key] = data["query_transformations"][key]

        if "repl" in data and isinstance(data["repl"], dict):
            if "shell_passthrough" in data["repl"]:
                config_dict["repl_shell_passthrough"] = data["repl"]["shell_passthrough"]

        if "context_compression" in data:
            config_dict["compress_context"] = data["context_compression"].get("enabled", False)

        if "web_search" in data:
            ws = data["web_search"]

            config_dict["truth_grounding"] = ws.get("enabled", False)

            if ws.get("brave_api_key"):
                config_dict["brave_api_key"] = ws["brave_api_key"]

        if "offline" in data:
            ol = data["offline"]

            config_dict["offline_mode"] = ol.get("enabled", False)

            if ol.get("local_models_dir"):
                config_dict["local_models_dir"] = ol["local_models_dir"]

            if ol.get("local_assets_only") is not None:
                config_dict["local_assets_only"] = ol["local_assets_only"]

            if ol.get("embedding_models_dir"):
                config_dict["embedding_models_dir"] = ol["embedding_models_dir"]

            if ol.get("hf_models_dir"):
                config_dict["hf_models_dir"] = ol["hf_models_dir"]

            if ol.get("tokenizer_cache_dir"):
                config_dict["tokenizer_cache_dir"] = ol["tokenizer_cache_dir"]

        # Map some specific names if they don't match exactly

        if "ollama_base_url" not in config_dict and "llm_base_url" in config_dict:
            config_dict["ollama_base_url"] = config_dict["llm_base_url"]

        # llm.models_dir â†' ollama_models_dir

        if "llm_models_dir" in config_dict and "ollama_models_dir" not in config_dict:
            config_dict["ollama_models_dir"] = config_dict.pop("llm_models_dir")

        if "api_key" not in config_dict and "llm_api_key" in config_dict:
            config_dict["api_key"] = config_dict["llm_api_key"]

        # llm.openai_api_key in YAML -> openai_api_key field

        if "llm_openai_api_key" in config_dict:
            config_dict["openai_api_key"] = config_dict.pop("llm_openai_api_key")

        # llm.grok_api_key in YAML -> grok_api_key field

        if "llm_grok_api_key" in config_dict:
            config_dict["grok_api_key"] = config_dict.pop("llm_grok_api_key")

        if "llm_vllm_base_url" in config_dict:
            config_dict["vllm_base_url"] = config_dict["llm_vllm_base_url"]

        if "projects_root" in data:
            config_dict["projects_root"] = data["projects_root"]

        if "max_workers" in data:
            config_dict["max_workers"] = data["max_workers"]

        if "ingest_batch_mode" in data:
            config_dict["ingest_batch_mode"] = data["ingest_batch_mode"]

        if "max_chunks_per_source" in data:
            config_dict["max_chunks_per_source"] = data["max_chunks_per_source"]

        if "source_policy_enabled" in data:
            config_dict["source_policy_enabled"] = data["source_policy_enabled"]

        if "projects_base" in data:
            config_dict["projects_root"] = data["projects_base"]

        if "store" in data:
            store_section = data["store"]

            if isinstance(store_section, dict) and store_section.get("base"):
                config_dict["axon_store_base"] = store_section["base"]

        if "qdrant_url" in data:
            config_dict["qdrant_url"] = data["qdrant_url"]

        if "qdrant_api_key" in data:
            config_dict["qdrant_api_key"] = data["qdrant_api_key"]

        if "repl_shell_passthrough" in data:
            config_dict["repl_shell_passthrough"] = data["repl_shell_passthrough"]

        # Environment Variable Overrides (High Priority --' wins over config.yaml)

        env_ollama_host = os.getenv("OLLAMA_HOST") or os.getenv("OLLAMA_BASE_URL")

        if env_ollama_host:
            config_dict["ollama_base_url"] = env_ollama_host

        env_vllm = os.getenv("VLLM_BASE_URL")

        if env_vllm:
            config_dict["vllm_base_url"] = env_vllm

        env_projects_root = os.getenv("AXON_PROJECTS_ROOT")

        if env_projects_root:
            config_dict["projects_root"] = env_projects_root

        env_ollama_models = os.getenv("OLLAMA_MODELS")

        if env_ollama_models:
            config_dict["ollama_models_dir"] = env_ollama_models

        # Filter only valid fields

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}

        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}

        cfg = cls(**filtered_dict)

        cfg._loaded_path = path

        return cfg

    def save(self, path: str | None = None) -> None:
        """Persist current configuration to disk in a structured format."""

        target = path or self._loaded_path or str(_USER_CONFIG_PATH)

        # Structure the data back into the expected nested groups

        from dataclasses import asdict

        flat = asdict(self)

        data = {
            "embedding": {
                "provider": flat["embedding_provider"],
                "model": flat["embedding_model"],
            },
            "llm": {
                "provider": flat["llm_provider"],
                "model": flat["llm_model"],
                "temperature": flat["llm_temperature"],
                "max_tokens": flat["llm_max_tokens"],
            },
            "vector_store": {
                "provider": flat["vector_store"],
                "path": flat["vector_store_path"],
                "tqdb_bits": flat["tqdb_bits"],
            },
            "bm25": {
                "path": flat["bm25_path"],
            },
            "rag": {
                "top_k": flat["top_k"],
                "similarity_threshold": flat["similarity_threshold"],
                "hybrid_search": flat["hybrid_search"],
                "hybrid_weight": flat["hybrid_weight"],
                "parent_chunk_size": flat["parent_chunk_size"],
                "raptor": flat["raptor"],
                "raptor_chunk_group_size": flat["raptor_chunk_group_size"],
                "graph_rag": flat["graph_rag"],
                "graph_rag_community": flat["graph_rag_community"],
                "dedup_on_ingest": flat["dedup_on_ingest"],
            },
            "chunk": {
                "strategy": flat["chunk_strategy"],
                "size": flat["chunk_size"],
                "overlap": flat["chunk_overlap"],
            },
            "rerank": {
                "enabled": flat["rerank"],
                "provider": flat["reranker_provider"],
                "model": flat["reranker_model"],
            },
            "query_transformations": {
                "multi_query": flat["multi_query"],
                "hyde": flat["hyde"],
                "step_back": flat["step_back"],
                "query_decompose": flat["query_decompose"],
                "discussion_fallback": flat["discussion_fallback"],
            },
            "repl": {
                "shell_passthrough": flat["repl_shell_passthrough"],
            },
            "context_compression": {
                "enabled": flat["compress_context"],
            },
            "web_search": {
                "enabled": flat["truth_grounding"],
                "brave_api_key": flat["brave_api_key"],
            },
            "offline": {
                "enabled": flat["offline_mode"],
                "local_models_dir": flat["local_models_dir"],
                "local_assets_only": flat["local_assets_only"],
                "embedding_models_dir": flat["embedding_models_dir"],
                "hf_models_dir": flat["hf_models_dir"],
                "tokenizer_cache_dir": flat["tokenizer_cache_dir"],
            },
            "projects_root": flat["projects_root"],
        }

        if flat.get("axon_store_base"):
            data["store"] = {"base": flat["axon_store_base"]}

            data.pop("projects_root", None)

            # Derived paths are recomputed in __post_init__ -- don't persist stale values

            data["vector_store"].pop("path", None)

            data["bm25"].pop("path", None)

        # Add provider-specific extras

        # Write under legacy "api_key" name for backwards compat; prefer openai_api_key if set

        _openai_key = flat["openai_api_key"] or flat["api_key"]

        if _openai_key:
            data["llm"]["api_key"] = _openai_key

        if flat["grok_api_key"]:
            data["llm"]["grok_api_key"] = flat["grok_api_key"]

        if flat["gemini_api_key"]:
            data["llm"]["gemini_api_key"] = flat["gemini_api_key"]

        if flat["ollama_cloud_key"]:
            data["llm"]["ollama_cloud_key"] = flat["ollama_cloud_key"]

        if flat["ollama_cloud_url"]:
            data["llm"]["ollama_cloud_url"] = flat["ollama_cloud_url"]

        if flat["vllm_base_url"]:
            data["llm"]["vllm_base_url"] = flat["vllm_base_url"]

        if flat["llm_timeout"]:
            data["llm"]["timeout"] = flat["llm_timeout"]

        import tempfile as _tempfile

        _resolved_target = os.path.expanduser(target)

        _tmp_root = os.path.realpath(_tempfile.gettempdir())

        if os.path.commonpath([os.path.realpath(_resolved_target), _tmp_root]) == _tmp_root:
            logger.warning(
                "AxonConfig.save() blocked: target path '%s' is inside the system temp "
                "directory. This usually means a test run is trying to overwrite your live "
                "config. Set path explicitly to save outside the temp tree.",
                _resolved_target,
            )

            return

        os.makedirs(os.path.dirname(_resolved_target), exist_ok=True)

        with open(_resolved_target, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Configuration saved to {target}")

    @classmethod
    def validate(cls, path: str | None = None) -> list["ConfigIssue"]:
        """Validate the config file at *path* (or the default user config path).

        Returns a list of :class:`ConfigIssue` objects grouped by severity.  The

        method never raises — it is designed to be called from CLI / API surfaces

        where a clean list of findings is more useful than an exception.

        Three passes are executed in order:

        1. **Structural pass** — unknown YAML keys are flagged; close matches are

           suggested via :func:`difflib.get_close_matches`.

        2. **Semantic pass** — known field values are range- and enum-checked on

           the constructed dataclass.

        3. **Store health pass** — the AxonStore directory layout is inspected on

           disk to surface mis-configuration or un-initialised stores.

        """

        issues: list[ConfigIssue] = []

        effective_path = path or str(_USER_CONFIG_PATH)

        # ------------------------------------------------------------------ #

        # 0. File existence                                                    #

        # ------------------------------------------------------------------ #

        if not os.path.exists(effective_path):
            issues.append(
                ConfigIssue(
                    level="info",
                    section="config",
                    field="path",
                    message=f"Config file not found at {effective_path}. Using dataclass defaults.",
                    suggestion="Run `axon --setup` to create a starter config.",
                )
            )

            return issues

        try:
            with open(os.path.expanduser(effective_path), encoding="utf-8") as _f:
                raw: dict = yaml.safe_load(_f) or {}

        except (yaml.YAMLError, OSError) as exc:
            issues.append(
                ConfigIssue(
                    level="error",
                    section="config",
                    field="path",
                    message=f"Could not read config file: {exc}",
                    suggestion="Check the file for YAML syntax errors.",
                )
            )

            return issues

        # ------------------------------------------------------------------ #

        # 1. Structural pass — unknown keys                                    #

        # ------------------------------------------------------------------ #

        for section, keys in raw.items():
            if not isinstance(keys, dict):
                continue

            known = _KNOWN_YAML_KEYS.get(section, set())

            if not known:
                continue

            for key in keys:
                if key not in known:
                    close = difflib.get_close_matches(key, known, n=1, cutoff=0.6)

                    suggestion = f"Did you mean '{close[0]}'?" if close else ""

                    issues.append(
                        ConfigIssue(
                            level="warn",
                            section=section,
                            field=key,
                            message=f"Unknown key '{key}' in section '{section}'.",
                            suggestion=suggestion,
                        )
                    )

        # ------------------------------------------------------------------ #

        # 2. Semantic pass — construct dataclass and check field values        #

        # ------------------------------------------------------------------ #

        try:
            cfg = cls.load(effective_path)

        except Exception as exc:  # pragma: no cover
            issues.append(
                ConfigIssue(
                    level="error",
                    section="config",
                    field="(load)",
                    message=f"Failed to load config: {exc}",
                )
            )

            return issues

        _VALID_CHUNK_STRATEGIES = {"recursive", "semantic", "markdown", "cosine_semantic"}

        if cfg.chunk_strategy not in _VALID_CHUNK_STRATEGIES:
            issues.append(
                ConfigIssue(
                    level="error",
                    section="chunk",
                    field="strategy",
                    message=(
                        f"Invalid chunk.strategy '{cfg.chunk_strategy}'. "
                        f"Must be one of: {', '.join(sorted(_VALID_CHUNK_STRATEGIES))}."
                    ),
                    suggestion="Set chunk.strategy to 'recursive', 'semantic', 'markdown', "
                    "or 'cosine_semantic'.",
                )
            )

        _VALID_GRAPH_RAG_MODES = {"local", "global", "hybrid"}

        if cfg.graph_rag_mode not in _VALID_GRAPH_RAG_MODES:
            issues.append(
                ConfigIssue(
                    level="error",
                    section="rag",
                    field="graph_rag_mode",
                    message=(
                        f"Invalid graph_rag_mode '{cfg.graph_rag_mode}'. "
                        f"Must be one of: {', '.join(sorted(_VALID_GRAPH_RAG_MODES))}."
                    ),
                )
            )

        _VALID_GRAPH_RAG_DEPTHS = {"light", "standard", "deep"}

        if cfg.graph_rag_depth not in _VALID_GRAPH_RAG_DEPTHS:
            issues.append(
                ConfigIssue(
                    level="error",
                    section="rag",
                    field="graph_rag_depth",
                    message=(
                        f"Invalid graph_rag_depth '{cfg.graph_rag_depth}'. "
                        f"Must be one of: {', '.join(sorted(_VALID_GRAPH_RAG_DEPTHS))}."
                    ),
                )
            )

        _VALID_QUERY_ROUTERS = {"heuristic", "llm", "off"}

        if cfg.query_router not in _VALID_QUERY_ROUTERS:
            issues.append(
                ConfigIssue(
                    level="error",
                    section="rag",
                    field="query_router",
                    message=(
                        f"Invalid query_router '{cfg.query_router}'. "
                        f"Must be one of: {', '.join(sorted(_VALID_QUERY_ROUTERS))}."
                    ),
                )
            )

        if cfg.top_k < 1:
            issues.append(
                ConfigIssue(
                    level="error",
                    section="rag",
                    field="top_k",
                    message=f"top_k must be >= 1, got {cfg.top_k}.",
                    suggestion="Set rag.top_k to a positive integer (e.g. 10).",
                )
            )

        if not (0.0 <= cfg.similarity_threshold <= 1.0):
            issues.append(
                ConfigIssue(
                    level="error",
                    section="rag",
                    field="similarity_threshold",
                    message=(
                        f"similarity_threshold must be between 0.0 and 1.0, "
                        f"got {cfg.similarity_threshold}."
                    ),
                )
            )

        if cfg.sentence_window_size < 1:
            issues.append(
                ConfigIssue(
                    level="error",
                    section="rag",
                    field="sentence_window_size",
                    message=f"sentence_window_size must be >= 1, got {cfg.sentence_window_size}.",
                )
            )

        # API key warnings

        if cfg.llm_provider == "openai":
            if not cfg.openai_api_key and not os.getenv("OPENAI_API_KEY"):
                issues.append(
                    ConfigIssue(
                        level="warn",
                        section="llm",
                        field="openai_api_key",
                        message="llm.provider is 'openai' but no OpenAI API key is configured.",
                        suggestion=(
                            "Set llm.openai_api_key in config.yaml or export OPENAI_API_KEY."
                        ),
                    )
                )

        if cfg.llm_provider == "gemini":
            if not cfg.gemini_api_key and not os.getenv("GEMINI_API_KEY"):
                issues.append(
                    ConfigIssue(
                        level="warn",
                        section="llm",
                        field="gemini_api_key",
                        message="llm.provider is 'gemini' but no Gemini API key is configured.",
                        suggestion=(
                            "Set llm.gemini_api_key in config.yaml or export GEMINI_API_KEY."
                        ),
                    )
                )

        if cfg.llm_provider == "grok":
            if (
                not cfg.grok_api_key
                and not os.getenv("XAI_API_KEY")
                and not os.getenv("GROK_API_KEY")
            ):
                issues.append(
                    ConfigIssue(
                        level="warn",
                        section="llm",
                        field="grok_api_key",
                        message="llm.provider is 'grok' but no Grok API key is configured.",
                        suggestion=("Set llm.grok_api_key in config.yaml or export XAI_API_KEY."),
                    )
                )

        # ------------------------------------------------------------------ #

        # 3. Store health pass                                                 #

        # ------------------------------------------------------------------ #

        env_store_base = os.getenv("AXON_STORE_BASE", "")

        config_store_base = ""

        if isinstance(raw.get("store"), dict):
            config_store_base = raw["store"].get("base", "")

        # Effective base: env wins, then config, then default

        effective_base = env_store_base or config_store_base or str(Path.home() / ".axon")

        if env_store_base and config_store_base and env_store_base != config_store_base:
            issues.append(
                ConfigIssue(
                    level="warn",
                    section="store",
                    field="base",
                    message=(
                        f"AXON_STORE_BASE env var ('{env_store_base}') overrides "
                        f"config store.base ('{config_store_base}'). Env wins."
                    ),
                    suggestion="Remove one of them to avoid confusion.",
                )
            )

        base_path = Path(effective_base).expanduser()

        # Only warn when the user has explicitly set a custom base that doesn't exist

        if not base_path.exists() and (env_store_base or config_store_base):
            issues.append(
                ConfigIssue(
                    level="warn",
                    section="store",
                    field="base",
                    message=f"Store base directory does not exist: {base_path}",
                    suggestion="Create the directory or update store.base in config.yaml.",
                )
            )

        elif base_path.exists():
            username = getpass.getuser()

            store_meta = base_path / "AxonStore" / username / "store_meta.json"

            if not store_meta.exists():
                issues.append(
                    ConfigIssue(
                        level="info",
                        section="store",
                        field="store_meta.json",
                        message=(
                            f"AxonStore not initialised at {base_path}. "
                            "No store_meta.json found."
                        ),
                        suggestion="Run `axon --store-init <path>` to initialise the store.",
                    )
                )

            else:
                try:
                    import json as _json

                    _meta = _json.loads(store_meta.read_text(encoding="utf-8"))

                    _store_id = _meta.get("store_id", "<unknown>")

                    issues.append(
                        ConfigIssue(
                            level="info",
                            section="store",
                            field="store_meta.json",
                            message=(
                                f"AxonStore initialised at {store_meta.parent} "
                                f"(store_id={_store_id})."
                            ),
                        )
                    )

                except Exception:
                    pass

        return issues
