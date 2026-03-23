"""
src/axon/config.py

AxonConfig dataclass extracted from main.py for Phase 2 of the Axon refactor.
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger("Axon")

# XDG-style user config dir — consistent across Linux / macOS / Windows
_USER_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".config", "axon", "config.yaml")


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
        "ollama", "gemini", "ollama_cloud", "openai", "vllm", "copilot", "github_copilot"
    ] = "ollama"
    llm_model: str = "gemma"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    api_key: str = ""
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
    # Root directory for all named projects. Defaults to ~/.axon/projects.
    # Override via config.yaml (projects_root: /path/to/dir) or the
    # AXON_PROJECTS_ROOT environment variable (env var wins over config.yaml).
    projects_root: str = os.path.join(os.path.expanduser("~"), ".axon", "projects")

    # Vector Store
    vector_store: Literal["chroma", "qdrant", "lancedb"] = "chroma"
    vector_store_path: str = os.path.join(
        os.path.expanduser("~"), ".axon", "projects", "default", "chroma_data"
    )

    # BM25 Settings
    bm25_path: str = os.path.join(
        os.path.expanduser("~"), ".axon", "projects", "default", "bm25_index"
    )

    def __post_init__(self) -> None:
        """Populate fields from environment variables and resolve storage paths."""
        # 1. API Keys and URLs
        if not self.api_key:
            self.api_key = os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", ""))
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

        # 2. Environment variable overrides for paths
        env_root = os.getenv("AXON_PROJECTS_ROOT")
        if env_root:
            self.projects_root = env_root

        env_vsp = os.getenv("CHROMA_DATA_PATH")
        if env_vsp:
            self.vector_store_path = env_vsp

        env_bm25 = os.getenv("BM25_INDEX_PATH")
        if env_bm25:
            self.bm25_path = env_bm25

        # 3. Aggressive WSL/Linux path resolution to avoid "readonly database"
        # errors on Windows-mounted drives (drvfs).
        is_linux = sys.platform == "linux"
        home = os.path.join(os.path.expanduser("~"), ".axon")

        def _resolve_safe(path_str: str, sub: str) -> str:
            if not path_str:
                return os.path.join(home, "projects", "default", sub)
            # Expand ~
            p_str = os.path.expanduser(path_str)
            # If it's the legacy relative default, force to home
            legacy_defaults = ("./chroma_data", "chroma_data", "./bm25_index", "bm25_index")
            if path_str in legacy_defaults:
                return os.path.join(home, "projects", "default", sub)

            # Check for absolute paths
            if is_linux:
                import posixpath

                is_abs = posixpath.isabs(p_str)
            else:
                is_abs = os.path.isabs(p_str)

            # If absolute but on a Windows mount in Linux, it will likely fail
            if is_linux and is_abs and p_str.startswith("/mnt/"):
                # Only redirect if it looks like the user didn't explicitly set a custom Linux path
                # (heuristic: if it contains 'studio_brain_open' or 'axon' in a Windows path)
                if any(x in p_str.lower() for x in ("axon", "studio_brain")):
                    safe_path = os.path.join(home, "projects", "default", sub)
                    return safe_path

            if is_linux:
                # On Windows host testing linux, abspath adds C:\
                return p_str
            return os.path.abspath(p_str)

        # Projects root special case (no sub-path)
        if not os.getenv("AXON_PROJECTS_ROOT"):
            self.projects_root = os.path.expanduser(self.projects_root)
            if (
                is_linux
                and os.path.isabs(self.projects_root)
                and self.projects_root.startswith("/mnt/")
            ):
                if any(x in self.projects_root.lower() for x in ("axon", "studio_brain")):
                    self.projects_root = os.path.join(home, "projects")

        self.vector_store_path = _resolve_safe(self.vector_store_path, "chroma_data")
        self.bm25_path = _resolve_safe(self.bm25_path, "bm25_index")

        # AxonStore mode: derive projects_root from store base
        env_store_base = os.getenv("AXON_STORE_BASE", "")
        if env_store_base and not self.axon_store_base:
            self.axon_store_base = env_store_base
        if self.axon_store_base:
            import getpass

            username = getpass.getuser()
            store_root = Path(self.axon_store_base).expanduser().resolve() / "AxonStore"
            user_dir = store_root / username
            self.projects_root = str(user_dir)
            self.vector_store_path = str(user_dir / "default" / "chroma_data")
            self.bm25_path = str(user_dir / "default" / "bm25_index")
            self.axon_store_mode = True

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
    # MMR deduplication — reorders and removes near-duplicate retrieved chunks
    mmr: bool = False
    mmr_lambda: float = 0.5  # 1.0 = pure relevance, 0.0 = pure diversity

    # Sentence-Window Retrieval (Epic 1)
    # Indexes prose chunks at sentence granularity; retrieves by sentence but
    # expands each hit to ±sentence_window_size surrounding sentences for LLM
    # context.  Only non-code, non-RAPTOR-summary leaf chunks are eligible.
    # Disabled by default; enable via config.yaml (rag.sentence_window: true).
    sentence_window: bool = False
    sentence_window_size: int = 3  # ±N sentences around each sentence hit

    # CRAG-Lite Retrieval Correction (Epic 2)
    # Evaluates retrieval confidence before deciding whether to trust local
    # results or escalate to web fallback.  Operates without LLM calls.
    # Disabled by default; enable via config.yaml (rag.crag_lite: true).
    crag_lite: bool = False
    crag_lite_confidence_threshold: float = 0.4  # below → low-confidence fallback

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

    # Context Compression (Epic 3, Stories 3.1–3.3)
    # compress_context: master on/off switch (backward-compatible).
    # compression_strategy selects the algorithm when compress_context is True:
    #   "sentence"   — LLM-based sentence extraction (default, existing behaviour)
    #   "llmlingua"  — LLMLingua-2 token compression (pip install axon[llmlingua])
    #   "none"       — disabled (same as compress_context=False)
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
    # stored in an entity→doc_id map.  At retrieval time, entities found in the
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

    # Runtime cost reduction — community triage
    graph_rag_community_min_size: int = 3  # communities smaller than this → template only
    graph_rag_community_llm_top_n_per_level: int = 15  # max LLM-summarized per level (0=unlimited)
    graph_rag_community_llm_max_total: int = (
        30  # hard cap on LLM calls across all levels (0=unlimited)
    )
    # Lazy community generation — skip summarization at finalize; generate on first global query
    graph_rag_community_lazy: bool = True
    # Global search pre-filter — cap communities entering map-reduce (0=no cap)
    graph_rag_global_top_communities: int = 0
    # RAPTOR source-size guard — skip RAPTOR for sources larger than this MB (0=no limit)
    raptor_max_source_size_mb: float = 0.0

    # Deferred batch saves — suppress per-call disk writes during batch ingest.
    # When True: BM25, entity graph, and relation graph saves deferred to finalize_ingest().
    # Reduces O(N²) disk writes to O(1) per session.
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
    # "louvain"   = networkx Louvain only (default — safe on all environments, fast for <10k nodes)
    # "leidenalg" = leidenalg/igraph multi-resolution Leiden (recommended when available)
    # "auto"      = graspologic → leidenalg → louvain fallback chain (legacy; unsafe on Python 3.13
    #               because graspologic's import can hang; use only when graspologic is verified safe)
    graph_rag_community_backend: str = "louvain"

    # Structural code graph.
    # Builds File/Symbol nodes and CONTAINS/IMPORTS edges from codebase chunk metadata.
    # code_graph_bridge: scans prose chunks for code symbol mentions → MENTIONED_IN edges.
    # Query time: traverses the code graph to expand retrieval results.
    code_graph: bool = False  # build + query structural code graph
    code_graph_bridge: bool = False  # link code symbols to prose chunks

    # Query-time lexical boost for code corpora.
    code_lexical_boost: bool = True  # apply identifier-aware re-scoring to code result sets
    code_top_k_multiplier: int = 2  # extra fetch_k factor when code query detected
    code_max_chunks_per_file: int = 3  # per-file cap in final top_k (diversity)
    # Code query mode tuning (active when code_lexical_boost=True and code query detected).
    # code_bm25_weight only affects weighted fusion mode — silently ignored in RRF (default).
    code_bm25_weight: float = 0.7  # BM25 weight override for code queries (weighted mode only)
    code_top_k: int = 6  # top-K override when code mode active (0 = use top_k)
    # Retrieval dry-run: skip LLM, return ranked candidates + diagnostics only.
    retrieval_dry_run: bool = False

    # Minimum entity appearance frequency to include in community detection graph.
    # Entities appearing in fewer than this many chunks are pruned before building the graph.
    # 1 = no pruning (include all entities). 2 = prune singletons (recommended for non-trivial
    # corpora — reduces noisy one-off entities; qualification studies used 2 for papers corpus).
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
    graph_rag_report_compress_ratio: float = 0.5  # target compression (0.0–1.0)

    # Auto-route queries based on complexity.
    # "heuristic": keyword-based, zero latency. "llm": one classifier LLM call.
    # "off" (default): use graph_rag_mode as configured.
    graph_rag_auto_route: Literal["off", "heuristic", "llm"] = "off"

    # Option B: Multi-class query router
    # "heuristic": keyword-based classifier (zero latency, default)
    # "llm": one LLM call per query to classify route
    # "off": skip router, use graph_rag_auto_route legacy behaviour
    query_router: str = "heuristic"

    # Contextual retrieval — prepend LLM-generated situating context to each chunk at ingest time.
    # Based on Anthropic's contextual retrieval technique.
    contextual_retrieval: bool = False

    # Semantic entity alias resolution — merge near-duplicate entity names (e.g.
    # "Apple" / "Apple Inc." / "Apple Corporation") into a single canonical node before
    # community detection.  Uses cosine similarity on entity-name embeddings.
    # pip install axon[graphrag]  (no extra deps — uses the already-loaded embedding model)
    graph_rag_entity_resolve: bool = False
    graph_rag_entity_resolve_threshold: float = 0.92  # cosine similarity threshold (0–1)
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

    # AxonStore — multi-user shared storage
    # When axon_store_base is set, projects_root is derived as:
    #   {axon_store_base}/AxonStore/{os_username}/
    # and the AxonStore namespace is initialised on first use.
    # Set via config.yaml store.base or AXON_STORE_BASE env var.
    axon_store_base: str = ""
    # Internal flag set by __post_init__ when store mode is active. Do not set directly.
    axon_store_mode: bool = False

    @classmethod
    def load(cls, path: str | None = None) -> "AxonConfig":
        """Load configuration from a YAML file.

        Defaults to ``~/.config/axon/config.yaml``.  On first run (file absent)
        the directory is created and a starter config is written automatically,
        so the notice appears only once.  Passing an explicit *path* that does
        not exist still produces a WARNING.
        """
        _DEFAULT_CONFIG_YAML = """\
# Axon Configuration — edit to customise behaviour.
# Full option reference: axon --help or https://github.com/...

embedding:
  provider: sentence_transformers
  model: all-MiniLM-L6-v2

llm:
  provider: ollama
  model: gemma
  temperature: 0.7
  max_tokens: 2048

vector_store:
  provider: chroma
  path: ~/.axon/projects/default/chroma_data

bm25:
  path: ~/.axon/projects/default/bm25_index

rag:
  top_k: 10
  similarity_threshold: 0.3
  hybrid_search: true

chunk:
  size: 1000
  overlap: 200

rerank:
  enabled: false
  provider: cross-encoder

query_transformations:
  multi_query: false
  hyde: false
  discussion_fallback: true

advanced:
  raptor: false
  graph_rag: false
  graph_rag_community: false

web_search:
  enabled: false

offline:
  enabled: false
  local_assets_only: false
  local_models_dir: ""
"""
        using_default = path is None
        if using_default:
            path = str(_USER_CONFIG_PATH)

        if not os.path.exists(path):
            if using_default:
                try:
                    cfg_path = Path(path)
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    cfg_path.write_text(_DEFAULT_CONFIG_YAML, encoding="utf-8")
                    logger.info("Created default config at %s — edit it to customise Axon.", path)
                except (OSError, PermissionError) as exc:
                    logger.warning(
                        "Could not create default config at %s (%s). Using in-memory defaults.",
                        path,
                        exc,
                    )
            else:
                logger.warning("Config file %s not found. Using defaults.", path)
            return cls()

        with open(os.path.expanduser(path), encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

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
            if "path" in vs:
                config_dict["vector_store_path"] = vs["path"]
        if "bm25" in data:
            if "path" in data["bm25"]:
                config_dict["bm25_path"] = data["bm25"]["path"]
        if "rag" in data:
            config_dict.update(data["rag"])
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

        # llm.models_dir → ollama_models_dir
        if "llm_models_dir" in config_dict and "ollama_models_dir" not in config_dict:
            config_dict["ollama_models_dir"] = config_dict.pop("llm_models_dir")

        if "api_key" not in config_dict and "llm_api_key" in config_dict:
            config_dict["api_key"] = config_dict["llm_api_key"]

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

        # Environment Variable Overrides (High Priority — wins over config.yaml)
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

        # Add provider-specific extras
        if flat["api_key"]:
            data["llm"]["api_key"] = flat["api_key"]
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
