"""
Core engine for Axon - Open Source RAG Interface.
"""

# Suppress TensorFlow/Keras noise before any imports that might trigger them
import os
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("USE_TF", "0")  # tell transformers to skip TF backend
# Disable ChromaDB / PostHog telemetry (avoids atexit noise on Ctrl+C)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")

import asyncio  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Literal  # noqa: E402

import yaml  # noqa: E402
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

# XDG-style user config dir — consistent across Linux / macOS / Windows
_USER_CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".config", "axon", "config.yaml")

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
    ollama_base_url: str = "http://localhost:11434"

    # LLM
    llm_provider: Literal[
        "ollama", "gemini", "ollama_cloud", "openai", "vllm", "copilot"
    ] = "ollama"
    llm_model: str = "gemma"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    api_key: str = ""
    gemini_api_key: str = ""
    ollama_cloud_key: str = ""
    ollama_cloud_url: str = ""
    vllm_base_url: str = "http://localhost:8000/v1"

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
            self.vector_store_path = str(user_dir / "_default" / "chroma_data")
            self.bm25_path = str(user_dir / "_default" / "bm25_index")
            self.axon_store_mode = True

    # RAG Settings
    top_k: int = 10
    similarity_threshold: float = 0.3
    hybrid_search: bool = True
    hybrid_weight: float = 0.7  # 1.0 = Pure Semantic, 0.0 = Pure Keyword
    hybrid_mode: Literal["weighted", "rrf"] = "weighted"  # Hybrid fusion mode

    # Chunking
    chunk_strategy: Literal["recursive", "semantic"] = "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 200

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

    # Context Compression
    # Uses the generation LLM to extract only query-relevant sentences from each
    # retrieved chunk before passing context to the final answer generation step.
    compress_context: bool = False

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
    local_models_dir: str = ""

    # RAPTOR Hierarchical Indexing
    # During ingest, groups every raptor_chunk_group_size consecutive chunks per source
    # and generates a summarisation node that is indexed alongside the leaf chunks.
    # At retrieval time these summary nodes surface high-level context for multi-hop
    # questions that no single leaf chunk can answer.
    raptor: bool = True
    raptor_chunk_group_size: int = 5

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

    # GAP 6: Async rebuild debounce
    graph_rag_community_rebuild_debounce_s: float = 2.0

    # LLM request timeout in seconds (applied where the provider client supports it)
    llm_timeout: int = 60

    # Maximum parallel worker threads for background ingestion and query tasks
    max_workers: int = 8

    # Dataset type for type-specific chunking. "auto" uses content-based heuristics.
    dataset_type: Literal["auto", "codebase", "paper", "doc", "discussion", "knowledge"] = "auto"

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
# Full option reference: see config.yaml in the project repository.

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

web_search:
  enabled: false

offline:
  enabled: false
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

        # Map some specific names if they don't match exactly
        if "ollama_base_url" not in config_dict and "llm_base_url" in config_dict:
            config_dict["ollama_base_url"] = config_dict["llm_base_url"]

        if "api_key" not in config_dict and "llm_api_key" in config_dict:
            config_dict["api_key"] = config_dict["llm_api_key"]

        if "llm_vllm_base_url" in config_dict:
            config_dict["vllm_base_url"] = config_dict["llm_vllm_base_url"]

        if "projects_root" in data:
            config_dict["projects_root"] = data["projects_root"]

        if "max_workers" in data:
            config_dict["max_workers"] = data["max_workers"]

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

        os.makedirs(os.path.dirname(os.path.expanduser(target)), exist_ok=True)
        with open(os.path.expanduser(target), "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Configuration saved to {target}")


class OpenReranker:
    """Document reranker supporting cross-encoder and LLM (RankGPT) providers.

    When ``reranker_provider="cross-encoder"`` a sentence-transformers CrossEncoder
    scores each (query, document) pair.  When ``reranker_provider="llm"`` the
    active LLM rates each document on a 1–10 scale in parallel threads.
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
                logger.info("Using LLM for Re-ranking (RankGPT)")
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
        """RankGPT pointwise scoring implementation."""
        system_prompt = "You are an expert relevance ranker. Rate the relevance of the document to the query on a scale from 1 to 10. Output ONLY the integer score."

        from concurrent.futures import ThreadPoolExecutor

        def score_doc(doc):
            prompt = f"Query: {query}\n\nDocument: {doc['text']}\n\nScore (1-10):"
            try:
                response = self.llm.complete(prompt, system_prompt=system_prompt).strip()
                return float(response) if response.replace(".", "", 1).isdigit() else 0.0
            except Exception:
                return 0.0

        # Batch concurrent requests to reduce overall latency
        with ThreadPoolExecutor(max_workers=min(10, len(documents))) as executor:
            scores = list(executor.map(score_doc, documents))

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score

        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked_docs


_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-m3": 1024,
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
}


class OpenEmbedding:
    """Unified embedding client supporting sentence_transformers, ollama, fastembed, and openai.

    Embedding dimensions for known models are resolved via :data:`_KNOWN_DIMS`
    without requiring a model download (useful for Ollama and FastEmbed).
    """

    def __init__(self, config: AxonConfig):
        self.config = config
        self.provider = config.embedding_provider
        self.model: Any = None
        self.dimension: int = 0
        self._load_model()

    def _load_model(self):
        """Load the embedding model."""
        if self.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading Sentence Transformers: {self.config.embedding_model}")
            self.model = SentenceTransformer(self.config.embedding_model)
            self.dimension = self.model.get_sentence_embedding_dimension()

        elif self.provider == "ollama":
            logger.info(f"Using Ollama Embedding: {self.config.embedding_model}")
            self.dimension = _KNOWN_DIMS.get(self.config.embedding_model, 768)

        elif self.provider == "fastembed":
            from fastembed import TextEmbedding

            logger.info(f"Loading FastEmbed: {self.config.embedding_model}")
            self.model = TextEmbedding(model_name=self.config.embedding_model)
            self.dimension = _KNOWN_DIMS.get(self.config.embedding_model, 384)

        elif self.provider == "openai":
            from openai import OpenAI

            logger.info(f"Using OpenAI API Embedding: {self.config.embedding_model}")
            kwargs = (
                {"api_key": self.config.api_key} if self.config.api_key else {"api_key": "sk-dummy"}
            )
            # ollama_base_url doubles as the generic base_url for OpenAI-compatible servers
            if (
                self.config.ollama_base_url
                and self.config.ollama_base_url != "http://localhost:11434"
            ):
                kwargs["base_url"] = self.config.ollama_base_url
            self.model = OpenAI(**kwargs)
            self.dimension = 1536

        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.provider == "sentence_transformers":
            embeddings = self.model.encode(texts, show_progress_bar=False)
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            return list(embeddings)

        elif self.provider == "ollama":
            from ollama import Client

            client = Client(host=self.config.ollama_base_url)
            embeddings = []
            for text in texts:
                response = client.embeddings(model=self.config.embedding_model, prompt=text)
                embeddings.append(response["embedding"])
            return embeddings

        elif self.provider == "fastembed":
            embeddings = list(self.model.embed(texts))
            return [e.tolist() for e in embeddings]

        elif self.provider == "openai":
            response = self.model.embeddings.create(input=texts, model=self.config.embedding_model)
            return [data.embedding for data in response.data]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]


# Bridge for VS Code Copilot LLM tasks
_copilot_task_queue: list[dict] = []
_copilot_responses: dict[str, dict] = {}
_copilot_bridge_lock = threading.Lock()


class OpenLLM:
    """Unified LLM client supporting ollama, gemini, ollama_cloud, openai, vllm, and copilot.

    The ``copilot`` provider routes completions through the VS Code extension
    bridge (poll ``GET /llm/copilot/tasks``, submit results via
    ``POST /llm/copilot/result/<task_id>``).
    """

    def __init__(self, config: AxonConfig):
        self.config = config
        self._openai_clients: dict = {}

    def _get_openai_client(self, base_url: str = None):
        """Return a cached OpenAI client. Pass base_url for vLLM or custom endpoints."""
        cache_key = base_url or "default"
        if cache_key not in self._openai_clients:
            from openai import OpenAI

            api_key = self.config.api_key if self.config.api_key else "sk-dummy"
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._openai_clients[cache_key] = OpenAI(**kwargs)
        return self._openai_clients[cache_key]

    def complete(
        self, prompt: str, system_prompt: str = None, chat_history: list[dict[str, str]] = None
    ) -> str:
        provider = self.config.llm_provider
        history = chat_history or []
        if provider == "ollama":
            from ollama import Client

            client = Client(host=self.config.ollama_base_url)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            # Add chat history
            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": prompt})

            response = client.chat(
                model=self.config.llm_model,
                messages=messages,
                options={"temperature": self.config.llm_temperature, "num_ctx": 8192},
            )
            return response["message"]["content"]

        elif provider == "gemini":
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import google.generativeai as genai
            if not getattr(self, "_gemini_configured", False):
                genai.configure(api_key=self.config.gemini_api_key)
                self._gemini_configured = True
            model_kwargs = {"model_name": self.config.llm_model}
            is_gemma = "gemma" in self.config.llm_model.lower()
            if system_prompt and not is_gemma:
                model_kwargs["system_instruction"] = system_prompt
            model = genai.GenerativeModel(**model_kwargs)

            contents = []
            for msg in history:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [msg["content"]]})

            user_text = prompt
            if system_prompt and is_gemma:
                user_text = f"{system_prompt}\n\n{prompt}"
            contents.append({"role": "user", "parts": [user_text]})

            response = model.generate_content(
                contents,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.llm_temperature,
                    max_output_tokens=self.config.llm_max_tokens,
                ),
            )
            return response.text

        elif provider == "ollama_cloud":
            import httpx

            headers = {
                "Authorization": f"Bearer {self.config.ollama_cloud_key}",
                "Content-Type": "application/json",
            }

            history_str = ""
            for msg in history:
                role = "Assistant" if msg["role"] == "assistant" else "User"
                history_str += f"{role}: {msg['content']}\n\n"

            full_prompt = f"{system_prompt}\n\n" if system_prompt else ""
            full_prompt += history_str
            full_prompt += f"User: {prompt}\n\nAssistant:"

            payload = {
                "model": self.config.llm_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": self.config.llm_temperature},
            }
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.config.ollama_cloud_url}/generate", json=payload, headers=headers
                )
                response.raise_for_status()
                return response.json()["response"]

        elif provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": prompt})

            # Pass base_url when pointing at an OpenAI-compatible local endpoint
            _openai_base = (
                self.config.ollama_base_url
                if self.config.ollama_base_url != "http://localhost:11434"
                else None
            )
            response = self._get_openai_client(base_url=_openai_base).chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                timeout=self.config.llm_timeout,
            )
            return response.choices[0].message.content

        elif provider == "vllm":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})

            response = self._get_openai_client(
                base_url=self.config.vllm_base_url
            ).chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                timeout=self.config.llm_timeout,
            )
            return response.choices[0].message.content

        elif provider == "copilot":
            task_id = f"task_{uuid.uuid4().hex[:12]}"
            event = threading.Event()

            with _copilot_bridge_lock:
                _copilot_task_queue.append(
                    {
                        "id": task_id,
                        "prompt": prompt,
                        "history": history,
                        "system_prompt": system_prompt,
                        "model": self.config.llm_model,
                        "temperature": self.config.llm_temperature,
                        "max_tokens": self.config.llm_max_tokens,
                    }
                )
                _copilot_responses[task_id] = {"event": event, "result": None, "error": None}

            # Wait for extension to fulfill (timeout from config)
            if event.wait(timeout=self.config.llm_timeout):
                res = _copilot_responses.get(task_id)
                if not res:
                    return "Error: Task response lost."
                if res["error"]:
                    return f"Error from Copilot: {res['error']}"
                result = res["result"] or ""
                with _copilot_bridge_lock:
                    _copilot_responses.pop(task_id, None)
                return result
            else:
                with _copilot_bridge_lock:
                    _copilot_responses.pop(task_id, None)
                return "Error: Copilot LLM bridge timed out."

        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

    def stream(
        self, prompt: str, system_prompt: str = None, chat_history: list[dict[str, str]] = None
    ):
        provider = self.config.llm_provider
        history = chat_history or []
        if provider == "ollama":
            from ollama import Client

            client = Client(host=self.config.ollama_base_url)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": prompt})

            stream_resp = client.chat(
                model=self.config.llm_model,
                messages=messages,
                stream=True,
                options={"temperature": self.config.llm_temperature, "num_ctx": 8192},
            )
            for chunk in stream_resp:
                yield chunk["message"]["content"]

        elif provider == "gemini":
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import google.generativeai as genai
            if not getattr(self, "_gemini_configured", False):
                genai.configure(api_key=self.config.gemini_api_key)
                self._gemini_configured = True
            model_kwargs = {"model_name": self.config.llm_model}
            is_gemma = "gemma" in self.config.llm_model.lower()
            if system_prompt and not is_gemma:
                model_kwargs["system_instruction"] = system_prompt
            model = genai.GenerativeModel(**model_kwargs)

            contents = []
            for msg in history:
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [msg["content"]]})

            user_text = prompt
            if system_prompt and is_gemma:
                user_text = f"{system_prompt}\n\n{prompt}"
            contents.append({"role": "user", "parts": [user_text]})

            response = model.generate_content(
                contents,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.llm_temperature,
                    max_output_tokens=self.config.llm_max_tokens,
                ),
            )
            for chunk in response:
                yield chunk.text

        elif provider == "ollama_cloud":
            import httpx

            headers = {
                "Authorization": f"Bearer {self.config.ollama_cloud_key}",
                "Content-Type": "application/json",
            }

            history_str = ""
            for msg in history:
                role = "Assistant" if msg["role"] == "assistant" else "User"
                history_str += f"{role}: {msg['content']}\n\n"

            full_prompt = f"{system_prompt}\n\n" if system_prompt else ""
            full_prompt += history_str
            full_prompt += f"User: {prompt}\n\nAssistant:"

            payload = {
                "model": self.config.llm_model,
                "prompt": full_prompt,
                "stream": True,
                "options": {"temperature": self.config.llm_temperature},
            }
            import json

            with httpx.Client(timeout=60.0) as client:
                with client.stream(
                    "POST",
                    f"{self.config.ollama_cloud_url}/generate",
                    json=payload,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]

        elif provider == "openai":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": prompt})

            # Pass base_url when pointing at an OpenAI-compatible local endpoint
            _openai_base = (
                self.config.ollama_base_url
                if self.config.ollama_base_url != "http://localhost:11434"
                else None
            )
            stream = self._get_openai_client(base_url=_openai_base).chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                stream=True,
                timeout=self.config.llm_timeout,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        elif provider == "vllm":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})

            stream = self._get_openai_client(
                base_url=self.config.vllm_base_url
            ).chat.completions.create(
                model=self.config.llm_model,
                messages=messages,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens,
                stream=True,
                timeout=self.config.llm_timeout,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content

        elif provider == "copilot":
            # For streaming, we'll just use the blocking complete() logic but yield it as one chunk.
            # True streaming via the bridge would require a more complex WebSocket setup.
            yield self.complete(prompt, system_prompt, chat_history)


class OpenVectorStore:
    """Unified interface over ChromaDB, Qdrant, and LanceDB vector stores.

    Initialized via :class:`AxonConfig`.  Supports ``add``, ``search``,
    ``get_by_ids``, ``delete_by_ids``, and ``list_documents`` operations
    across all three backends.  Qdrant can operate in local file mode or
    remote mode (set ``qdrant_url`` in config).
    """

    def __init__(self, config: AxonConfig):
        self.config = config
        self.provider = config.vector_store
        self.client: Any = None
        self.collection: Any = None
        self._init_store()

    def _init_store(self):
        if self.provider == "chroma":
            import chromadb

            logger.info(f"Initializing ChromaDB: {self.config.vector_store_path}")
            try:
                self.client = chromadb.PersistentClient(path=self.config.vector_store_path)
                self.collection = self.client.get_or_create_collection(
                    name="axon", metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                # Catch the specific WSL/SQLite readonly error (code 8)
                if "(code: 8)" in str(e) and "readonly database" in str(e).lower():
                    msg = (
                        f"\n\n[bold red]ERROR:[/bold red] ChromaDB failed to initialize at: {self.config.vector_store_path}\n"
                        "This typically happens in WSL when using a Windows-mounted drive (/mnt/c/...). \n"
                        "SQLite does not support locking on these mounts.\n\n"
                        "FIX: Store your data in the Linux filesystem instead:\n"
                        "  1. Set environment variable: [bold]CHROMA_DATA_PATH=~/axon_data[/bold]\n"
                        "  2. Or edit config.yaml to use a path like: [bold]~/axon_data[/bold]\n"
                    )
                    from rich.console import Console

                    Console().print(msg)
                    raise RuntimeError(
                        f"ChromaDB failed to initialize at: {self.config.vector_store_path}"
                    )
                raise e
        elif self.provider == "qdrant":
            from qdrant_client import QdrantClient

            if getattr(self.config, "qdrant_url", ""):
                logger.info(f"Initializing Qdrant (remote): {self.config.qdrant_url}")
                self.client = QdrantClient(
                    url=self.config.qdrant_url,
                    api_key=self.config.qdrant_api_key or None,
                )
            else:
                logger.info(f"Initializing Qdrant (local): {self.config.vector_store_path}")
                self.client = QdrantClient(path=self.config.vector_store_path)
        elif self.provider == "lancedb":
            import lancedb

            logger.info(f"Initializing LanceDB: {self.config.vector_store_path}")
            self.client = lancedb.connect(self.config.vector_store_path)
            try:
                self.collection = self.client.open_table("axon")
            except Exception:
                self.collection = None  # created lazily on first add()

    def close(self):
        """Release any open file handles or database connections."""
        if self.provider == "chroma" and self.client:
            # ChromaDB 0.4.x+ uses a persistent client that should be closed if supported
            if hasattr(self.client, "close"):
                try:
                    self.client.close()
                except Exception:
                    pass
            self.client = None
            self.collection = None
        elif self.provider == "lancedb" and self.client:
            self.client = None
            self.collection = None
        elif self.provider == "qdrant" and self.client:
            self.client = None

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] = None,
    ):
        if self.provider == "chroma":
            try:
                self.collection.add(
                    ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
                )
            except Exception as e:
                if "dimension" in str(e).lower():
                    logger.error(
                        f"Embedding dimension mismatch in ChromaDB! Expected: {self.config.embedding_model}. "
                        "Try clearing the project data or switch to a different project."
                    )
                raise
        elif self.provider == "qdrant":
            from qdrant_client.models import PointStruct

            points = []
            for i, (id, embedding, text) in enumerate(zip(ids, embeddings, texts)):
                payload = {"text": text}
                if metadatas:
                    payload.update(metadatas[i])
                points.append(PointStruct(id=id, vector=embedding, payload=payload))
            self.client.upsert(collection_name="axon", points=points)
        elif self.provider == "lancedb":
            import json

            rows = []
            for i, (doc_id, emb, text) in enumerate(zip(ids, embeddings, texts)):
                meta = metadatas[i] if metadatas else {}
                rows.append(
                    {
                        "id": doc_id,
                        "vector": emb,
                        "text": text,
                        "source": meta.get("source", ""),
                        "metadata_json": json.dumps(meta),
                    }
                )
            if self.collection is None:
                self.collection = self.client.create_table(
                    "axon", data=rows, mode="overwrite", metric="cosine"
                )
            else:
                self.collection.add(rows)

    def list_documents(self) -> list[dict[str, Any]]:
        """Return all unique source files stored in the knowledge base with chunk counts.

        Returns:
            List of dicts sorted by source name, each with keys:
                - source (str): The metadata 'source' value, or 'unknown' if not set.
                - chunks (int): Number of chunks stored for that source.
                - doc_ids (List[str]): All chunk IDs belonging to this source.
        """
        if self.provider == "chroma":
            result = self.collection.get(include=["metadatas"])
            sources: dict[str, dict[str, Any]] = {}
            for doc_id, meta in zip(
                result["ids"], result["metadatas"] or [{}] * len(result["ids"])
            ):
                source = (meta or {}).get("source", "unknown")
                if source not in sources:
                    sources[source] = {"source": source, "chunks": 0, "doc_ids": []}
                sources[source]["chunks"] += 1
                sources[source]["doc_ids"].append(doc_id)
            return sorted(sources.values(), key=lambda x: x["source"])
        elif self.provider == "qdrant":
            results, _ = self.client.scroll(collection_name="axon", limit=10000, with_payload=True)
            sources: dict[str, dict[str, Any]] = {}
            for point in results:
                source = point.payload.get("source", "unknown")
                if source not in sources:
                    sources[source] = {"source": source, "chunks": 0, "doc_ids": []}
                sources[source]["chunks"] += 1
                sources[source]["doc_ids"].append(str(point.id))
            return sorted(sources.values(), key=lambda x: x["source"])
        elif self.provider == "lancedb":
            if self.collection is None:
                return []
            rows = self.collection.to_arrow().to_pydict()
            sources: dict[str, dict[str, Any]] = {}
            for doc_id, source in zip(rows.get("id", []), rows.get("source", [])):
                src = source or "unknown"
                if src not in sources:
                    sources[src] = {"source": src, "chunks": 0, "doc_ids": []}
                sources[src]["chunks"] += 1
                sources[src]["doc_ids"].append(doc_id)
            return sorted(sources.values(), key=lambda x: x["source"])
        return []

    def search(
        self, query_embedding: list[float], top_k: int = 10, filter_dict: dict = None
    ) -> list[dict]:
        if self.provider == "chroma":
            results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
            return [
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "score": 1 - results["distances"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                }
                for i in range(len(results["ids"][0]))
            ]
        elif self.provider == "qdrant":
            results = self.client.search(
                collection_name="axon", query_vector=query_embedding, limit=top_k
            )
            return [
                {
                    "id": str(r.id),
                    "text": r.payload.get("text", ""),
                    "score": r.score,
                    "metadata": {k: v for k, v in r.payload.items() if k != "text"},
                }
                for r in results
            ]
        elif self.provider == "lancedb":
            import json

            if self.collection is None:
                return []
            results = self.collection.search(query_embedding).limit(top_k).to_list()
            return [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "score": max(0.0, 1.0 - r.get("_distance", 1.0)),
                    "metadata": json.loads(r.get("metadata_json", "{}")),
                }
                for r in results
            ]

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        """Fetch stored documents by their IDs (used by GraphRAG expansion).

        Returns a list of result dicts in the same format as search(), with score=1.0
        since these docs are fetched by exact ID (not scored).
        """
        if not ids:
            return []
        if self.provider == "chroma":
            result = self.collection.get(ids=ids, include=["documents", "metadatas"])
            result_ids = result.get("ids") or []
            result_docs = result.get("documents") or []
            result_metas = result.get("metadatas") or []
            num_ids = len(result_ids)
            if len(result_docs) < num_ids:
                result_docs = list(result_docs) + [""] * (num_ids - len(result_docs))
            if len(result_metas) < num_ids:
                result_metas = list(result_metas) + [{}] * (num_ids - len(result_metas))
            docs = []
            for i in range(num_ids):
                docs.append(
                    {
                        "id": result_ids[i],
                        "text": result_docs[i] or "",
                        "score": 1.0,
                        "metadata": result_metas[i] or {},
                    }
                )
            return docs
        if self.provider == "qdrant":
            try:
                points = self.client.retrieve(
                    collection_name="axon",
                    ids=ids,
                    with_payload=True,
                )
                return [
                    {
                        "id": str(p.id),
                        "text": p.payload.get("text", ""),
                        "score": 1.0,
                        "metadata": {k: v for k, v in p.payload.items() if k != "text"},
                    }
                    for p in points
                ]
            except Exception as e:
                logger.debug(f"GraphRAG get_by_ids (Qdrant) failed: {e}")
                return []
        if self.provider == "lancedb":
            import json

            if self.collection is None:
                return []
            try:
                id_str = ", ".join(f"'{i}'" for i in ids)
                rows = self.collection.search().where(f"id IN ({id_str})", prefilter=True).to_list()
                return [
                    {
                        "id": r["id"],
                        "text": r["text"],
                        "score": 1.0,
                        "metadata": json.loads(r.get("metadata_json", "{}")),
                    }
                    for r in rows
                ]
            except Exception as e:
                logger.debug(f"GraphRAG get_by_ids (LanceDB) failed: {e}")
                return []
        return []

    def delete_by_ids(self, ids: list[str]) -> None:
        """Delete documents by ID from the vector store."""
        if not ids:
            return
        if self.provider == "chroma":
            self.collection.delete(ids=ids)
        elif self.provider == "qdrant":
            from qdrant_client.models import PointIdsList

            self.client.delete(
                collection_name="axon",
                points_selector=PointIdsList(points=ids),
            )
        elif self.provider == "lancedb":
            if self.collection is None:
                return
            id_str = ", ".join(f"'{i}'" for i in ids)
            self.collection.delete(f"id IN ({id_str})")


_MERGED_VIEW_WRITE_ERROR = (
    "Cannot write to a merged parent project view. " "Switch to a specific sub-project first."
)


class MultiVectorStore:
    """Read-only fan-out wrapper over multiple OpenVectorStore instances.

    Used when a parent project is active: queries are dispatched to all
    descendant stores and the top-k results (by score) are returned merged.
    Writes are NOT supported — use the project's own OpenVectorStore for that.
    """

    def __init__(self, stores: list[OpenVectorStore]):
        self._stores = stores
        # Expose provider/collection from the first store so callers that
        # inspect brain.vector_store.provider / .collection still work.
        self.provider = stores[0].provider if stores else "chroma"
        self.collection = stores[0].collection if stores else None

    def search(
        self, query_embedding: list[float], top_k: int = 10, filter_dict: dict = None
    ) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor

        seen: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(self._stores))) as ex:
            futures = [
                ex.submit(store.search, query_embedding, top_k, filter_dict)
                for store in self._stores
            ]
            for fut in futures:
                for doc in fut.result():
                    doc_id = doc["id"]
                    if doc_id not in seen or doc["score"] > seen[doc_id]["score"]:
                        seen[doc_id] = doc
        return sorted(seen.values(), key=lambda d: d["score"], reverse=True)[:top_k]

    def add(self, *args, **kwargs):
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)

    def list_documents(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for store in self._stores:
            for doc in store.list_documents():
                src = doc["source"]
                if src not in seen:
                    seen[src] = doc.copy()
                else:
                    seen[src]["chunks"] += doc["chunks"]
                    seen[src]["doc_ids"].extend(doc["doc_ids"])
        return sorted(seen.values(), key=lambda x: x["source"])

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        seen: dict[str, dict] = {}
        for store in self._stores:
            for doc in store.get_by_ids(ids):
                seen[doc["id"]] = doc
        return list(seen.values())

    def delete_by_ids(self, ids: list[str]) -> None:
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)

    def delete_documents(self, ids: list[str]) -> None:
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)


class MultiBM25Retriever:
    """Read-only fan-out wrapper over multiple BM25Retriever instances.

    Merges BM25 results from all descendant stores and returns the top-k
    by score. Writes are NOT supported.
    """

    def __init__(self, retrievers: list) -> None:
        self._retrievers = retrievers

    def close(self):
        for r in self._retrievers:
            if hasattr(r, "close"):
                r.close()

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor

        seen: dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(4, len(self._retrievers))) as ex:
            futures = [ex.submit(r.search, query, top_k) for r in self._retrievers]
            for fut in futures:
                for doc in fut.result():
                    doc_id = doc["id"]
                    if doc_id not in seen or doc["score"] > seen[doc_id]["score"]:
                        seen[doc_id] = doc
        return sorted(seen.values(), key=lambda d: d["score"], reverse=True)[:top_k]

    def delete_documents(self, doc_ids: list[str]) -> None:
        """Disallow deletes in merged read-only views."""
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)

    def add_documents(self, *args, **kwargs):
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)

    # Alias used by ingest code
    batch_add_documents = add_documents


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

    def _resolve_model_path(self, model_name: str) -> str:
        """Resolve a HuggingFace model ID to a local path when offline_mode is on.

        If the name is already an absolute path or starts with '.' it is returned
        unchanged.  Otherwise the last path component (after the optional org/)
        is looked up under local_models_dir, then the full name with '/' → '--'.
        A warning is logged when the directory cannot be found.
        """
        if not self.config.offline_mode or not self.config.local_models_dir:
            return model_name
        if os.path.isabs(model_name) or model_name.startswith("."):
            return model_name
        base = self.config.local_models_dir
        # Try bare name: "BAAI/bge-reranker-base" → "<dir>/bge-reranker-base"
        short_name = model_name.split("/")[-1]
        candidate = os.path.join(base, short_name)
        if os.path.isdir(candidate):
            return candidate
        # Try HF cache style: "BAAI/bge-reranker-base" → "<dir>/BAAI--bge-reranker-base"
        hf_name = model_name.replace("/", "--")
        candidate2 = os.path.join(base, hf_name)
        if os.path.isdir(candidate2):
            return candidate2
        logger.warning(
            f"Offline mode: '{model_name}' not found in {base} "
            f"(tried '{short_name}' and '{hf_name}')"
        )
        return model_name

    def __init__(self, config: AxonConfig | None = None):
        self.config = config or AxonConfig.load()

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
            self.config.embedding_model = self._resolve_model_path(self.config.embedding_model)
            self.config.reranker_model = self._resolve_model_path(self.config.reranker_model)
            logger.info(
                f"Offline mode ON  |  models dir: {self.config.local_models_dir or '(not set)'}"
            )

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

        # Content hash store for ingest deduplication
        self._ingested_hashes: set = self._load_hash_store()

        # Doc versions store for smart re-ingest
        self._doc_versions: dict = {}
        self._doc_versions_path = os.path.join(self.config.bm25_path, ".doc_versions.json")
        self._load_doc_versions()

        # GraphRAG entity → doc_id mapping (entity name -> list of chunk IDs)
        self._entity_graph: dict[str, list[str]] = self._load_entity_graph()

        # GraphRAG relation graph: {source_entity_lower: [{target, relation, chunk_id, description}]}
        self._relation_graph: dict = self._load_relation_graph()

        # GraphRAG community detection state
        self._community_levels: dict = self._load_community_levels()
        self._community_summaries: dict = self._load_community_summaries()
        self._community_graph_dirty: bool = False
        self._last_matched_entities: list = []

        # GraphRAG entity embeddings (for embedding-based entity matching at query time)
        self._entity_embeddings: dict = self._load_entity_embeddings()

        # GraphRAG entity description buffer (transient, not persisted)
        self._entity_description_buffer: dict = {}

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

        logger.info("Axon ready!")

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
        from axon.projects import (
            list_descendants,
            project_bm25_path,
            project_dir,
            project_vector_path,
            set_active_project,
        )

        if name == "default":
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
        if name != "default":
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
                                if isinstance(node, dict):
                                    doc_ids = node.get("chunk_ids", [])
                                elif isinstance(node, list):
                                    doc_ids = node
                                else:
                                    continue
                                if not doc_ids:
                                    continue
                                existing = self._entity_graph.get(entity)
                                if existing is None:
                                    if isinstance(node, dict):
                                        self._entity_graph[entity] = {
                                            "description": node.get("description", ""),
                                            "type": node.get("type", "UNKNOWN"),
                                            "chunk_ids": [d for d in doc_ids if isinstance(d, str)],
                                            "frequency": len(
                                                [d for d in doc_ids if isinstance(d, str)]
                                            ),
                                            "degree": node.get("degree", 0),
                                        }
                                    else:
                                        self._entity_graph[entity] = [
                                            d for d in doc_ids if isinstance(d, str)
                                        ]
                                elif isinstance(existing, dict):
                                    existing_ids = set(existing.get("chunk_ids", []))
                                    new_ids = [
                                        d
                                        for d in doc_ids
                                        if isinstance(d, str) and d not in existing_ids
                                    ]
                                    if new_ids:
                                        existing["chunk_ids"].extend(new_ids)
                                        existing["frequency"] = len(existing["chunk_ids"])
                                else:
                                    # existing is legacy list
                                    existing_ids = set(existing)
                                    for doc_id in doc_ids:
                                        if isinstance(doc_id, str) and doc_id not in existing_ids:
                                            existing.append(doc_id)
                                            existing_ids.add(doc_id)
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
        set_active_project(name)
        logger.info(f"Switched to project '{name}'")

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

        Shape (new): {entity_lower: {"description": str, "chunk_ids": list[str]}}
        Migrates legacy flat-list format: {entity_lower: [chunk_id, ...]}
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
                    if isinstance(value, list):
                        # Legacy flat-list format — migrate
                        cleaned[key] = {
                            "description": "",
                            "type": "UNKNOWN",
                            "chunk_ids": [v for v in value if isinstance(v, str)],
                            "frequency": len([v for v in value if isinstance(v, str)]),
                            "degree": 0,
                        }
                    elif isinstance(value, dict) and "chunk_ids" in value:
                        node = value
                        # Migration: add missing fields with defaults
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

    def _load_community_map(self) -> dict:
        """Backward-compat: returns finest community level map."""
        return self._community_map  # uses the property

    def _save_community_map(self) -> None:
        """Backward-compat: delegates to _save_community_levels."""
        self._save_community_levels()

    def _load_community_levels(self) -> dict:
        """Load persisted hierarchical community levels from disk.

        Shape: {level_int: {entity_lower: community_id}}
        Falls back to old .community_map.json if levels file is missing.
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
        # Fall back to old .community_map.json if levels file missing
        old_path = pathlib.Path(self.config.bm25_path) / ".community_map.json"
        try:
            if old_path.exists():
                raw = json.loads(old_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    filtered = {
                        k: v for k, v in raw.items() if isinstance(k, str) and isinstance(v, int)
                    }
                    if filtered:
                        return {0: filtered}
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
                    # JSON keys are strings; convert to int; None values stay None
                    return {int(k): (None if v is None else int(v)) for k, v in raw.items()}
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

    @property
    def _community_map(self) -> dict:
        """Alias for the finest-grained community level (backward compat)."""
        if not self._community_levels:
            return {}
        finest = max(self._community_levels.keys())
        return self._community_levels.get(finest, {})

    @_community_map.setter
    def _community_map(self, value: dict) -> None:
        """For backward compat: setting _community_map sets level 0 in a single-level structure."""
        if value:
            self._community_levels = {0: value}
        else:
            self._community_levels = {}

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

        G = nx.Graph()
        for entity, node in self._entity_graph.items():
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

        try:
            # Try graspologic hierarchical Leiden
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

            return community_levels, community_hierarchy, community_children

        except ImportError:
            pass  # Fall through to synthetic mapping

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
                    self._generate_community_summaries()
                    if getattr(self.config, "graph_rag_index_community_reports", True):
                        self._index_community_reports_in_vector_store()

    def _generate_community_summaries(self) -> None:
        """Generate LLM summaries for each detected community cluster across all levels."""
        if not self._community_levels:
            return
        import json as _json
        import re as _re
        from collections import defaultdict

        summaries = {}
        total_communities = sum(len(set(m.values())) for m in self._community_levels.values())
        logger.info(f"GraphRAG: Generating summaries for {total_communities} communities...")

        def _summarise(args):
            level_idx, cid, members = args
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
                for sub_key in sub_community_keys:
                    if sub_key in self._community_summaries:
                        sub_reports.append(
                            self._community_summaries[sub_key].get("full_content", "")
                        )
                if sub_reports:
                    entity_rel_text = "\n\n".join(
                        f"Sub-community Report:\n{r}" for r in sub_reports[:3]
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
            summary_key = f"{level_idx}_{cid}"
            prompt = (
                "You are analyzing a knowledge graph community.\n\n"
                "Community members and descriptions:\n"
                f"{context[:3000]}\n\n"
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
                }

        # Process levels from finest (highest idx) to coarsest — SEQUENTIALLY so sub-community
        # reports are available when summarizing parent communities (true bottom-up composition).
        for level_idx in sorted(self._community_levels.keys(), reverse=True):
            level_map = self._community_levels[level_idx]
            community_entities: dict = defaultdict(list)
            for entity, cid in level_map.items():
                community_entities[cid].append(entity)
            level_work_items = [
                (level_idx, cid, members) for cid, members in community_entities.items()
            ]
            level_results = list(self._executor.map(_summarise, level_work_items))
            for summary_key, summary_dict in level_results:
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

    def _global_search_context(self, query: str, cfg) -> str:
        """Score community summaries against the query; return top-K as context (legacy method)."""
        if not self._community_summaries:
            return ""
        top_k = getattr(cfg, "graph_rag_community_top_k", 5)
        query_tokens = set(query.lower().split())
        scored = []
        for cid, cs in self._community_summaries.items():
            summary = cs.get("full_content") or cs.get("summary", "(no summary)")
            if not summary:
                continue
            # Try embedding similarity first, fall back to token overlap
            try:
                import numpy as np

                q_vec = self.embedding.embed_query(query)
                s_vec = self.embedding.embed([summary])[0]
                sim = float(
                    np.dot(q_vec, s_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(s_vec) + 1e-9)
                )
            except Exception:
                summary_tokens = set(summary.lower().split())
                sim = len(query_tokens & summary_tokens) / max(len(query_tokens), 1)
            scored.append((sim, cid, cs))
        scored.sort(reverse=True)
        parts = []
        for rank, (_, _cid, cs) in enumerate(scored[:top_k]):
            entities_preview = ", ".join(cs.get("entities", [])[:10])
            parts.append(
                f"[Community Report {rank + 1} — {cs.get('size', 0)} entities: "
                f"{entities_preview}]\n{cs.get('full_content') or cs.get('summary', '(no summary)')}"
            )
        return "\n\n".join(parts)

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

        # Chunk reports so large reports don't get hard-truncated and later sections aren't lost
        _MAP_CHUNK_CHARS = 2000

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
        all_points = []
        try:
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
        reduce_prompt = (
            f"The following analytic reports have been generated for the query:\n\n"
            f"Query: {query}\n\n"
            f"Reports:\n\n{reduce_context}\n\n"
            f"Using the reports above, generate a comprehensive response to the query."
        )
        try:
            response = self.llm.complete(
                reduce_prompt,
                system_prompt=_GRAPHRAG_REDUCE_SYSTEM_PROMPT,
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
        """Build structured GraphRAG local context: entity descriptions + relations + community snippet."""
        if not matched_entities:
            return ""

        # GAP 4: Token budget per section
        total_budget = getattr(cfg, "graph_rag_local_max_context_tokens", 8000)
        community_budget = int(total_budget * getattr(cfg, "graph_rag_local_community_prop", 0.25))
        text_unit_budget = int(total_budget * getattr(cfg, "graph_rag_local_text_unit_prop", 0.5))
        entity_rel_budget = total_budget - community_budget - text_unit_budget
        top_k_entities = getattr(cfg, "graph_rag_local_top_k_entities", 10)
        top_k_relationships = getattr(cfg, "graph_rag_local_top_k_relationships", 10)
        include_weight = getattr(cfg, "graph_rag_local_include_relationship_weight", False)

        # GAP 4: Rank entities by total degree (outgoing + incoming relations)
        def _entity_degree(ent: str) -> int:
            outgoing = len(self._relation_graph.get(ent.lower(), []))
            incoming = len(self._get_incoming_relations(ent))
            return outgoing + incoming

        ranked_entities = sorted(matched_entities, key=_entity_degree, reverse=True)
        ranked_entities = ranked_entities[:top_k_entities]

        parts = []
        used_entity_rel_tokens = 0

        ent_parts = []
        for ent in ranked_entities:
            if used_entity_rel_tokens > entity_rel_budget:
                break
            node = self._entity_graph.get(ent.lower(), {})
            desc = node.get("description", "") if isinstance(node, dict) else ""
            ent_type = node.get("type", "") if isinstance(node, dict) else ""
            if desc:
                line = f"  - {ent} [{ent_type}]: {desc}" if ent_type else f"  - {ent}: {desc}"
                ent_parts.append(line)
                used_entity_rel_tokens += len(line) // 4
        if ent_parts:
            parts.append("**Matched Entities:**\n" + "\n".join(ent_parts))

        rel_parts = []
        rel_count = 0
        for ent in ranked_entities:
            if rel_count >= top_k_relationships or used_entity_rel_tokens > entity_rel_budget:
                break
            ent_lower = ent.lower()
            # Outgoing relations
            outgoing = self._relation_graph.get(ent_lower, [])

            # GAP 4: Out-of-network mutual-links prioritization (count shared targets)
            def _mutual_count(e):
                return sum(
                    1
                    for ee in ranked_entities
                    if e.get("target", "")
                    in {
                        entry2.get("target", "")
                        for entry2 in self._relation_graph.get(ee.lower(), [])
                    }
                )

            sorted_outgoing = sorted(outgoing, key=_mutual_count, reverse=True)
            for entry in sorted_outgoing[:3]:
                if rel_count >= top_k_relationships or used_entity_rel_tokens > entity_rel_budget:
                    break
                desc = entry.get("description") or (
                    f"{ent} {entry.get('relation', '')} {entry.get('target', '')}"
                )
                weight_str = ""
                if include_weight and entry.get("weight"):
                    weight_str = f" (weight: {entry['weight']})"
                line = f"  - {desc}{weight_str}"
                rel_parts.append(line)
                used_entity_rel_tokens += len(line) // 4
                rel_count += 1
            # GAP 4: Incoming edges
            for entry in self._get_incoming_relations(ent)[:2]:
                if rel_count >= top_k_relationships or used_entity_rel_tokens > entity_rel_budget:
                    break
                src = entry.get("source", "")
                desc = entry.get("description") or (f"{src} {entry.get('relation', '')} {ent}")
                weight_str = ""
                if include_weight and entry.get("weight"):
                    weight_str = f" (weight: {entry['weight']})"
                line = f"  - {desc}{weight_str}"
                rel_parts.append(line)
                used_entity_rel_tokens += len(line) // 4
                rel_count += 1
        if rel_parts:
            parts.append("**Relevant Relationships:**\n" + "\n".join(rel_parts))

        # Community section (budget-capped) — collect up to budget across all matched entities
        if self._community_levels and self._community_summaries:
            finest = max(self._community_levels.keys()) if self._community_levels else None
            used_community_tokens = 0
            seen_community_keys: set = set()
            community_parts = []
            for ent in ranked_entities[:top_k_entities]:
                if used_community_tokens > community_budget:
                    break
                cid = (
                    self._community_levels.get(finest, {}).get(ent.lower())
                    if finest is not None
                    else None
                )
                if cid is not None:
                    summary_key = f"{finest}_{cid}"
                    if summary_key in seen_community_keys:
                        continue
                    seen_community_keys.add(summary_key)
                    cs = self._community_summaries.get(summary_key, {})
                    # Also check old-format key for backward compat
                    if not cs:
                        cs = self._community_summaries.get(str(cid), {})
                    summary = cs.get("summary", "")
                    if summary:
                        sentences = summary.split(". ")
                        snippet = ". ".join(sentences[:3]).strip()
                        if snippet and not snippet.endswith("."):
                            snippet += "."
                        title = cs.get("title", f"Community {cid}")
                        community_text = f"**Community Context ({title}):**\n  {snippet}"
                        used_community_tokens += len(community_text) // 4
                        community_parts.append(community_text)
            if community_parts:
                parts.extend(community_parts)

        # Source text units (budget-capped, ranked by relationship count)
        text_unit_parts = []
        seen_chunks: set = set()
        used_text_unit_tokens = 0
        text_unit_ids_all = []
        for ent in ranked_entities[:3]:
            node = self._entity_graph.get(ent.lower(), {})
            chunk_ids = node.get("chunk_ids", []) if isinstance(node, dict) else []
            for cid in chunk_ids[:2]:
                if cid not in seen_chunks:
                    seen_chunks.add(cid)
                    text_unit_ids_all.append(cid)

        # GAP 9: Sort by relationship count (descending)
        def _text_unit_rel_count(chunk_id: str) -> int:
            return len(self._text_unit_relation_map.get(chunk_id, []))

        text_unit_ids_sorted = sorted(text_unit_ids_all, key=_text_unit_rel_count, reverse=True)

        for cid in text_unit_ids_sorted:
            if used_text_unit_tokens > text_unit_budget:
                break
            try:
                docs = self.vector_store.get_by_ids([cid])
                if docs:
                    text_snippet = docs[0].get("text", "")[:200].strip()
                    if text_snippet:
                        line = f"  [{cid}] {text_snippet}..."
                        text_unit_parts.append(line)
                        used_text_unit_tokens += len(line) // 4
            except Exception:
                pass
        if text_unit_parts:
            parts.append("**Source Text Units:**\n" + "\n".join(text_unit_parts))

        # GAP 4 (covariate fix): Claims matched by subject name, not chunk_ids
        if self._claims_graph:
            claim_parts = []
            for ent in ranked_entities[:3]:
                for chunk_claims in self._claims_graph.values():
                    for claim in chunk_claims:
                        if claim.get("subject", "").lower() == ent.lower():
                            status = claim.get("status", "SUSPECTED")
                            desc = claim.get("description", "")
                            if desc:
                                claim_parts.append(f"  - [{status}] {desc}")
            if claim_parts:
                parts.append("**Claims / Facts:**\n" + "\n".join(claim_parts[:10]))

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

    def _extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from text using the LLM.

        Returns a list of dicts with shape:
          {"name": str, "type": str, "description": str}
        Returns an empty list on failure or when the LLM produces no output.
        """
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
        Handles both new dict-node format and legacy list format.
        """
        eg_changed = False
        for entity in list(self._entity_graph):
            node = self._entity_graph[entity]
            chunk_ids = node["chunk_ids"] if isinstance(node, dict) else node
            after = [d for d in chunk_ids if d not in deleted_ids]
            if len(after) != len(chunk_ids):
                eg_changed = True
                if after:
                    if isinstance(node, dict):
                        self._entity_graph[entity]["chunk_ids"] = after
                        # Recompute frequency after pruning (Item 2)
                        self._entity_graph[entity]["frequency"] = len(after)
                    else:
                        self._entity_graph[entity] = {"description": "", "chunk_ids": after}
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
            for i in range(0, len(chunks), n):
                windows.append((source, i, chunks[i : i + n]))

        if not windows:
            return []

        logger.info(f"   RAPTOR: Generating summaries for {len(windows)} groups...")

        def _proc_window(item):
            source, i, window = item
            combined = "\n\n".join(c["text"] for c in window)
            prompt = (
                "Summarise the following passage into a concise but comprehensive paragraph "
                "that captures all key facts and concepts. "
                "Output only the summary paragraph.\n\n" + combined[:4000]
            )
            try:
                summary_text = self.llm.complete(
                    prompt,
                    system_prompt="You are an expert at summarising technical documents.",
                )
                if not summary_text or not summary_text.strip():
                    return None

                import hashlib

                content_sig = hashlib.md5(
                    summary_text.strip().encode("utf-8", errors="replace")
                ).hexdigest()[:8]
                uid = hashlib.md5(f"{source}|raptor|{i}|{content_sig}".encode()).hexdigest()[:12]
                return {
                    "id": f"raptor_{uid}",
                    "text": summary_text.strip(),
                    "metadata": {
                        "source": source,
                        "raptor_level": 1,
                        "window_start": i,
                        "window_end": i + len(window) - 1,
                    },
                }
            except Exception as e:
                logger.debug(f"RAPTOR summary failed for {source}[{i}]: {e}")
                return None

        results = list(self._executor.map(_proc_window, windows))

        summaries = [r for r in results if r]
        if summaries:
            logger.info(f"   RAPTOR: generated {len(summaries)} summary node(s)")
        return summaries

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
                doc_ids = node["chunk_ids"] if isinstance(node, dict) else node
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
                    target_chunk_ids = (
                        target_node["chunk_ids"] if isinstance(target_node, dict) else target_node
                    )
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
            "rerank": cfg.rerank,
            "hyde": cfg.hyde,
            "multi_query": cfg.multi_query,
            "step_back": cfg.step_back,
            "query_decompose": cfg.query_decompose,
            "threshold": cfg.similarity_threshold,
            "discuss": cfg.discussion_fallback,
            "compress_context": cfg.compress_context,
            "truth_grounding": cfg.truth_grounding,
            "graph_rag": cfg.graph_rag,
            "llm_provider": cfg.llm_provider,
            "llm_model": cfg.llm_model,
            "embedding_provider": cfg.embedding_provider,
            "embedding_model": cfg.embedding_model,
            "reranker_model": cfg.reranker_model,
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
                "top_score": round(top_score, 4) if top_score else None,
                "hybrid": active.hybrid_search,
                "rerank": active.rerank,
                "transformations": transformations or {},
            }
        )

    # ── Dataset type detection ────────────────────────────────────────────────
    _CODE_EXTENSIONS = {
        ".py",
        ".ts",
        ".js",
        ".go",
        ".rs",
        ".java",
        ".cpp",
        ".c",
        ".sh",
        ".rb",
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

    def _get_splitter_for_type(self, dataset_type: str, has_code: bool):
        """Return a splitter configured for the detected document type."""
        from axon.splitters import RecursiveCharacterTextSplitter, SemanticTextSplitter

        if dataset_type == "codebase":
            return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        elif dataset_type == "paper":
            return SemanticTextSplitter(chunk_size=600, chunk_overlap=100)
        elif dataset_type == "discussion":
            return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        elif dataset_type == "knowledge":
            return SemanticTextSplitter(chunk_size=400, chunk_overlap=50)
        elif dataset_type == "doc" and has_code:
            return SemanticTextSplitter(chunk_size=500, chunk_overlap=75)
        else:  # doc default
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

            parent_texts = parent_splitter.split(doc["text"])
            for p_idx, parent_text in enumerate(parent_texts):
                p_doc = {
                    "id": f"{doc['id']}_p{p_idx}",
                    "text": parent_text,
                    "metadata": doc.get("metadata", {}).copy(),
                }
                child_chunks = self.splitter.transform_documents([p_doc])
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

        # Guard: raise immediately if the embedding model has changed since this
        # collection was created — mixing models silently corrupts retrieval.
        self._validate_embedding_meta(on_mismatch="raise")

        t0 = time.time()
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
                splitter = self._get_splitter_for_type(dataset_type, has_code)
                if splitter is not None:
                    chunked.extend(splitter.transform_documents([doc]))
                else:
                    chunked.append(doc)
            documents = chunked

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

        # RAPTOR: generate summarisation nodes for the deduplicated leaf chunks
        if self.config.raptor:
            raptor_docs = self._generate_raptor_summaries(documents)
            documents = documents + raptor_docs

        # GraphRAG: extract entities from new chunks and update entity graph
        if self.config.graph_rag:
            updated = False
            # Only extract entities from actual document chunks
            chunks_to_process = [
                doc for doc in documents if not doc.get("metadata", {}).get("raptor_level")
            ]

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

            # Item 2: Update frequency for all entities touched during this ingest
            for entity_key in self._entity_graph:
                node = self._entity_graph[entity_key]
                if isinstance(node, dict):
                    node["frequency"] = len(node.get("chunk_ids", []))

            if updated:
                self._save_entity_graph()
            if total_entities == 0:
                logger.warning(
                    "GraphRAG: entity extraction returned 0 entities across all chunks. "
                    "This may be caused by an LLM that is too small or refused to extract entities. "
                    "GraphRAG relationship expansion will have no effect for this ingestion."
                )

            # Relation extraction: build SUBJECT | RELATION | OBJECT triples
            if self.config.graph_rag_relations:
                logger.info(
                    f"   GraphRAG: Extracting relations from {len(chunks_to_process)} chunks..."
                )

                def _proc_rel(doc):
                    return doc["id"], self._extract_relations(doc["text"])

                rel_results = list(self._executor.map(_proc_rel, chunks_to_process))
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
                if rg_updated:
                    self._save_relation_graph()

                # GAP 9: Update text_unit_relation_map
                for doc_id, triples in rel_results:
                    self._text_unit_relation_map[doc_id] = [
                        (t.get("subject", ""), t.get("object", ""))
                        if isinstance(t, dict)
                        else (t[0], t[2])
                        for t in triples
                    ]

            # Item 2: Recompute degree for all entities after relation-build
            for entity_key in self._entity_graph:
                if isinstance(self._entity_graph[entity_key], dict):
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
            if self.config.graph_rag and getattr(self.config, "graph_rag_canonicalize", False):
                self._canonicalize_entity_descriptions()

            # Item 11: Extract claims
            claims_changed = False
            if getattr(self.config, "graph_rag_claims", False):
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
                if claims_changed:
                    self._save_claims_graph()

            if self.config.graph_rag_community and self._community_graph_dirty:
                self._community_graph_dirty = False
                if self.config.graph_rag_community_async:

                    def _debounced_rebuild():
                        import time as _time

                        _time.sleep(self.config.graph_rag_community_rebuild_debounce_s)
                        self._rebuild_communities()

                    self._executor.submit(_debounced_rebuild)
                else:
                    self._rebuild_communities()

        n_chunks = len(documents)
        if self._own_bm25:
            self._own_bm25.add_documents(documents)

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

        logger.info(
            {
                "event": "ingest_complete",
                "chunks": n_chunks,
                "embed_ms": round(embed_ms, 1),
                "store_ms": round(store_ms, 1),
                "total_ms": round((time.time() - t0) * 1000, 1),
            }
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

        # --- Phase 2: Retrieval ---
        fetch_k = cfg.top_k * 3 if (cfg.rerank or cfg.hybrid_search) else cfg.top_k

        all_vector_results = []
        all_bm25_results = []

        # Vector Search
        if cfg.hyde:
            # HyDE uses a single hypothetical document as the vector query
            all_vector_results.extend(
                self.vector_store.search(
                    self.embedding.embed_query(vector_query), top_k=fetch_k, filter_dict=filters
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

        # Hybrid Search (Keyword component)
        bm25_count = 0
        if cfg.hybrid_search and self.bm25:
            # Parallelize BM25 searches across multiple queries
            def _bm25_search(q):
                return self.bm25.search(q, top_k=fetch_k)

            all_bm25_lists = list(self._executor.map(_bm25_search, search_queries))
            for b_list in all_bm25_lists:
                all_bm25_results.extend(b_list)

            dedup_bm25 = {}
            for r in all_bm25_results:
                if r["id"] not in dedup_bm25 or r["score"] > dedup_bm25[r["id"]]["score"]:
                    dedup_bm25[r["id"]] = r
            bm25_results = list(dedup_bm25.values())
            bm25_count = len(bm25_results)

            if cfg.hybrid_mode == "rrf":
                from axon.retrievers import reciprocal_rank_fusion

                results = reciprocal_rank_fusion(vector_results, bm25_results)
            else:
                from axon.retrievers import weighted_score_fusion

                results = weighted_score_fusion(
                    vector_results, bm25_results, weight=cfg.hybrid_weight
                )
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

        # Save base count before GraphRAG expansion for accurate metrics
        base_count = len(results)

        # GraphRAG: expand results with entity-linked documents
        _matched_entities: list = []
        if cfg.graph_rag and self._entity_graph:
            results, _matched_entities = self._expand_with_entity_graph(query, results, cfg=cfg)

        return {
            "results": results,
            "vector_count": vector_count,
            "bm25_count": bm25_count,
            "filtered_count": base_count,
            "graph_expanded_count": len(results) - base_count,
            "matched_entities": _matched_entities,
            "transforms": transforms,
        }

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
                parts.append(f"[Document {i+1} (ID: {r['id']})]\n{context_text}")
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

        if cfg.graph_rag and cfg.graph_rag_budget > 0:
            base = [r for r in results if not r.get("_graph_expanded")][: cfg.top_k]
            expanded = [r for r in results if r.get("_graph_expanded")]
            base_ids = {r["id"] for r in base}
            graph_slots = [r for r in expanded if r["id"] not in base_ids][: cfg.graph_rag_budget]
            results = base + graph_slots
        else:
            results = results[: cfg.top_k]
        for r in results:
            r.pop("_graph_expanded", None)

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
        top_score = results[0].get("score", 0) if results else 0
        context, has_web = self._build_context(results)

        # GraphRAG global context injection
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

        if cfg.graph_rag and cfg.graph_rag_budget > 0:
            base = [r for r in results if not r.get("_graph_expanded")][: cfg.top_k]
            expanded = [r for r in results if r.get("_graph_expanded")]
            base_ids = {r["id"] for r in base}
            graph_slots = [r for r in expanded if r["id"] not in base_ids][: cfg.graph_rag_budget]
            results = base + graph_slots
        else:
            results = results[: cfg.top_k]
        for r in results:
            r.pop("_graph_expanded", None)

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
        context, has_web = self._build_context(results)

        # GraphRAG global context injection
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
# Shared helpers used by both CLI (main()) and REPL
# ---------------------------------------------------------------------------


def _print_project_tree(proj_list: list, active: str, indent: int = 0) -> None:
    """Print a recursive project tree with active-project marker and metadata.

    Uses the already-fetched ``children`` list from list_projects() rather than
    calling has_children() again, avoiding redundant directory traversals.
    """
    pad = "  " * indent
    for p in proj_list:
        marker = "●" if p["name"] == active else " "
        ts = p["created_at"][:10] if p["created_at"] else ""
        desc = f"  {p.get('description', '')}" if p.get("description") else ""
        merged = "  [merged]" if p.get("children") else ""
        short = p["name"].split("/")[-1]
        print(f"  {pad}{marker} {short:<22} {ts}{merged}{desc}")
        _print_project_tree(p.get("children", []), active, indent + 1)


def _write_python_discovery() -> None:
    """Write current Python executable path to ~/.axon/.python_path.

    Called once at startup so the VS Code extension can auto-detect the Python
    interpreter regardless of whether axon was installed via pip, venv, or pipx.
    Failures are silently ignored — this is a best-effort helper.
    """
    try:
        discovery_dir = Path.home() / ".axon"
        discovery_dir.mkdir(parents=True, exist_ok=True)
        (discovery_dir / ".python_path").write_text(sys.executable, encoding="utf-8")
    except Exception:
        pass


def main():
    import argparse

    # On Windows, switch the console to UTF-8 (codepage 65001) so that
    # box-drawing characters and emoji render correctly.
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    _write_python_discovery()
    parser = argparse.ArgumentParser(description="Axon CLI")
    parser.add_argument("query", nargs="?", help="Question to ask")
    parser.add_argument("--ingest", help="Path to file or directory to ingest")
    parser.add_argument(
        "--list", action="store_true", help="List all ingested sources in the knowledge base"
    )
    parser.add_argument(
        "--project",
        metavar="NAME",
        help="Project to use (must exist; use --project-new to create). "
        'Use "default" for the global knowledge base.',
    )
    parser.add_argument(
        "--project-new",
        metavar="NAME",
        help="Create a new project (if it does not exist) and use it. "
        "Combine with --ingest to populate in one step.",
    )
    parser.add_argument("--project-list", action="store_true", help="List all projects and exit")
    parser.add_argument(
        "--project-delete", metavar="NAME", help="Delete a project and all its data, then exit"
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to config YAML (default: ~/.config/axon/config.yaml)",
    )
    parser.add_argument("--stream", action="store_true", help="Stream the response")
    parser.add_argument(
        "--provider",
        choices=["ollama", "gemini", "ollama_cloud", "openai", "vllm"],
        help="LLM provider to use (overrides config)",
    )
    parser.add_argument(
        "--model",
        help="Model name to use (overrides config), e.g. gemma:2b, gemini-1.5-flash, gpt-4o",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available Ollama models and supported cloud providers",
    )
    parser.add_argument(
        "--pull", metavar="MODEL", help="Pull an Ollama model by name, e.g. --pull gemma:2b"
    )
    parser.add_argument(
        "--embed",
        metavar="MODEL",
        help="Embedding model to use, e.g. all-MiniLM-L6-v2 or ollama/nomic-embed-text",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        metavar="F",
        help="LLM temperature for generation (0.0–2.0, default: from config, usually 0.7). "
        "Lower = more deterministic, higher = more creative.",
    )
    parser.add_argument(
        "--discuss",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable discussion fallback (answer from general knowledge when no docs match). "
        "Use --discuss to enable, --no-discuss to disable.",
    )
    parser.add_argument(
        "--search",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable Brave web search fallback (requires BRAVE_API_KEY). "
        "Use --search to enable, --no-search to disable.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        metavar="N",
        help="Number of chunks to retrieve (default: from config, usually 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        metavar="F",
        help="Similarity threshold for retrieval, 0.0–1.0 (default: from config, usually 0.3)",
    )
    parser.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable hybrid BM25+vector search",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable cross-encoder reranking",
    )
    parser.add_argument(
        "--reranker-model",
        metavar="MODEL",
        help="Re-ranker model to use (e.g. BAAI/bge-reranker-v2-m3 for SOTA accuracy, "
        "default: cross-encoder/ms-marco-MiniLM-L-6-v2)",
    )
    parser.add_argument(
        "--hyde",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable HyDE (hypothetical document embedding)",
    )
    parser.add_argument(
        "--multi-query",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable multi-query retrieval",
    )
    parser.add_argument(
        "--step-back",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable step-back prompting (abstracts query before retrieval)",
    )
    parser.add_argument(
        "--decompose",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable query decomposition (breaks complex questions into sub-questions)",
    )
    parser.add_argument(
        "--compress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable LLM context compression (extracts only relevant sentences before generation)",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable in-memory query result caching",
    )
    parser.add_argument(
        "--raptor",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable RAPTOR hierarchical summarisation nodes during ingest",
    )
    parser.add_argument(
        "--raptor-group-size",
        type=int,
        metavar="N",
        help="Number of consecutive chunks to group per RAPTOR summary (default: 5)",
    )
    parser.add_argument(
        "--graph-rag",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable GraphRAG entity-centric retrieval expansion",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable ingest deduplication (allow re-ingesting identical content)",
    )
    parser.add_argument(
        "--chunk-strategy",
        choices=["recursive", "semantic"],
        help="Chunking strategy for ingest (recursive or semantic)",
    )
    parser.add_argument(
        "--parent-chunk-size",
        type=int,
        metavar="N",
        help="Enable small-to-big retrieval: index child chunks of --chunk-size tokens "
        "but return parent passages of N tokens as LLM context. 0 = disabled.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress spinners and progress (auto-enabled when stdin is not a TTY)",
    )
    args = parser.parse_args()

    # Suppress httpx INFO noise before _InitDisplay is active (ollama.list fires early)
    if sys.stdin.isatty():
        logging.getLogger("httpx").propagate = False
        logging.getLogger("httpx").setLevel(logging.WARNING)

    config = AxonConfig.load(args.config)
    if args.provider:
        config.llm_provider = args.provider
    if args.model:
        _PROVIDERS = ("ollama", "gemini", "openai", "ollama_cloud", "vllm")
        if "/" in args.model:
            _prov, _mdl = args.model.split("/", 1)
            if _prov in _PROVIDERS:
                config.llm_provider = _prov
                config.llm_model = _mdl
            else:
                # Not a provider prefix — treat whole string as model name
                config.llm_provider = _infer_provider(args.model)
                config.llm_model = args.model
        else:
            config.llm_provider = _infer_provider(args.model)
            config.llm_model = args.model

    if args.embed:
        _EMBED_PROVIDERS = ("sentence_transformers", "ollama", "fastembed", "openai")
        if "/" in args.embed:
            _eprov, _emdl = args.embed.split("/", 1)
            if _eprov in _EMBED_PROVIDERS:
                config.embedding_provider = _eprov
                config.embedding_model = _emdl
            else:
                config.embedding_model = args.embed
        else:
            config.embedding_model = args.embed

    if args.temperature is not None:
        config.llm_temperature = args.temperature
    if args.discuss is not None:
        config.discussion_fallback = args.discuss
    if args.search is not None:
        config.truth_grounding = args.search
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.threshold is not None:
        config.similarity_threshold = args.threshold
    if args.hybrid is not None:
        config.hybrid_search = args.hybrid
    if args.rerank is not None:
        config.rerank = args.rerank
    if args.reranker_model:
        config.reranker_model = args.reranker_model
    if args.chunk_strategy:
        config.chunk_strategy = args.chunk_strategy
    if args.parent_chunk_size is not None:
        config.parent_chunk_size = args.parent_chunk_size
    if args.hyde is not None:
        config.hyde = args.hyde
    if args.multi_query is not None:
        config.multi_query = args.multi_query
    if args.step_back is not None:
        config.step_back = args.step_back
    if args.decompose is not None:
        config.query_decompose = args.decompose
    if args.compress is not None:
        config.compress_context = args.compress
    if args.cache is not None:
        config.query_cache = args.cache
    if args.no_dedup:
        config.dedup_on_ingest = False
    if args.raptor is not None:
        config.raptor = args.raptor
    if args.raptor_group_size is not None:
        config.raptor_chunk_group_size = args.raptor_group_size
    if args.graph_rag is not None:
        config.graph_rag = args.graph_rag

    if args.list_models:
        print("\n  Supported LLM providers and example models:\n")
        print("  ollama       (local)  — gemma:2b, gemma, llama3.1, mistral, phi3")
        print("  gemini       (cloud)  — gemini-1.5-flash, gemini-1.5-pro, gemini-2.0-flash")
        print("  ollama_cloud (cloud)  — any model hosted at your OLLAMA_CLOUD_URL")
        print("  openai       (cloud)  — gpt-4o, gpt-4o-mini, gpt-3.5-turbo")
        print(
            "  vllm         (local)  — any model served by vLLM (e.g., meta-llama/Llama-3.1-8B-Instruct)"
        )
        print(
            f"               URL: {config.vllm_base_url}  (set vllm_base_url in config or VLLM_BASE_URL env)\n"
        )
        try:
            import ollama as _ollama

            response = _ollama.list()
            models = response.models if hasattr(response, "models") else response.get("models", [])
            if models:
                print("  Locally available Ollama models:")
                for m in models:
                    name = m.model if hasattr(m, "model") else m.get("name", str(m))
                    size_gb = m.size / 1e9 if hasattr(m, "size") and m.size else 0
                    size_str = f"  ({size_gb:.1f} GB)" if size_gb else ""
                    print(f"     • {name}{size_str}")
        except Exception:
            print("  (Ollama not reachable — cannot list local models)")
        print()
        return

    if args.pull:
        try:
            import ollama as _ollama

            print(f"  Pulling '{args.pull}'...")
            for chunk in _ollama.pull(args.pull, stream=True):
                status = (
                    chunk.get("status", "")
                    if isinstance(chunk, dict)
                    else getattr(chunk, "status", "")
                )
                total = (
                    chunk.get("total", 0) if isinstance(chunk, dict) else getattr(chunk, "total", 0)
                )
                completed = (
                    chunk.get("completed", 0)
                    if isinstance(chunk, dict)
                    else getattr(chunk, "completed", 0)
                )
                if total and completed:
                    pct = int(completed / total * 100)
                    print(f"\r  {status}: {pct}%  ", end="", flush=True)
                elif status:
                    print(f"\r  {status}...    ", end="", flush=True)
            print(f"\n  '{args.pull}' is ready.\n")
        except Exception as e:
            print(f"\n  Error: Failed to pull '{args.pull}': {e}")
        return

    # Auto-pull Ollama model if not available locally
    if config.llm_provider == "ollama" and config.llm_model:
        try:
            import ollama as _ollama

            response = _ollama.list()
            models = response.models if hasattr(response, "models") else response.get("models", [])
            local_names = set()
            for m in models:
                name = m.model if hasattr(m, "model") else m.get("name", "")
                local_names.add(name)
                local_names.add(name.split(":")[0])  # also match without tag
            model_tag = (
                config.llm_model if ":" in config.llm_model else f"{config.llm_model}:latest"
            )
            if model_tag not in local_names and config.llm_model not in local_names:
                print(f"  Model '{config.llm_model}' not found locally — pulling from Ollama...")
                for chunk in _ollama.pull(config.llm_model, stream=True):
                    status = (
                        chunk.get("status", "")
                        if isinstance(chunk, dict)
                        else getattr(chunk, "status", "")
                    )
                    total = (
                        chunk.get("total", 0)
                        if isinstance(chunk, dict)
                        else getattr(chunk, "total", 0)
                    )
                    completed = (
                        chunk.get("completed", 0)
                        if isinstance(chunk, dict)
                        else getattr(chunk, "completed", 0)
                    )
                    if total and completed:
                        pct = int(completed / total * 100)
                        print(f"\r  {status}: {pct}%", end="", flush=True)
                    elif status:
                        print(f"\r  {status}...", end="", flush=True)
                print(f"\n  Model '{config.llm_model}' ready.\n")
        except Exception as e:
            logger.warning(f"Could not auto-pull model '{config.llm_model}': {e}")

    # Animated init display — only when entering interactive REPL
    _entering_repl = (
        not args.query
        and not getattr(args, "ingest", None)
        and not args.list
        and not args.list_models
        and not getattr(args, "pull", None)
        and sys.stdin.isatty()
    )
    _init_display: _InitDisplay | None = None
    _saved_propagate: dict = {}
    _INIT_LOGGER_NAMES = [
        "Axon",
        "Axon.Retrievers",
        "sentence_transformers.SentenceTransformer",
        "sentence_transformers",
        "chromadb",
        "chromadb.telemetry.product.posthog",
        "httpx",
    ]
    if _entering_repl:
        print()
        _init_display = _InitDisplay()
        for _n in _INIT_LOGGER_NAMES:
            _lg = logging.getLogger(_n)
            _saved_propagate[_n] = _lg.propagate
            _lg.propagate = False  # suppress default stderr handler
            _lg.setLevel(logging.INFO)
            _lg.addHandler(_init_display)

    brain = AxonBrain(config)

    if _init_display:
        _init_display.stop()
        for _n in _INIT_LOGGER_NAMES:
            _lg = logging.getLogger(_n)
            _lg.removeHandler(_init_display)
            _lg.propagate = _saved_propagate.get(_n, True)

    # --- Project CLI handling ---
    from axon.projects import (
        ProjectHasChildrenError,
        delete_project,
        ensure_project,
        list_projects,
        project_dir,
    )

    if args.project_list:
        projects = list_projects()
        if not projects:
            print("  No projects yet. Use --project-new <name> to create one.")
        else:
            print()
            active = brain._active_project

            _print_project_tree(projects, active)
            print(f"\n  Active: {active}")
        return

    if args.project_delete:
        proj_name = args.project_delete.lower()
        try:
            if brain._active_project == proj_name:
                brain.switch_project("default")
            delete_project(proj_name)
            print(f"  Deleted project '{proj_name}'.")
        except ProjectHasChildrenError as e:
            print(f"  {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"  {e}")
            sys.exit(1)
        return

    # Switch to an existing project
    if args.project:
        proj_name = args.project.lower()
        try:
            brain.switch_project(proj_name)
        except ValueError as e:
            print(f"  {e}")
            sys.exit(1)

    # Create (if needed) and switch to new project
    if args.project_new:
        proj_name = args.project_new.lower()
        ensure_project(proj_name)
        brain.switch_project(proj_name)
        print(f"  Using project '{proj_name}'  ({project_dir(proj_name)})")

    if args.ingest:
        if os.path.isdir(args.ingest):
            asyncio.run(brain.load_directory(args.ingest))
        else:
            from axon.loaders import DirectoryLoader

            ext = os.path.splitext(args.ingest)[1].lower()
            loader_mgr = DirectoryLoader()
            if ext in loader_mgr.loaders:
                docs = loader_mgr.loaders[ext].load(args.ingest)
                # Add [File Path:] breadcrumb to match directory ingest metadata
                abs_path = os.path.abspath(args.ingest)
                for doc in docs:
                    if doc.get("metadata", {}).get("type") not in ("csv", "tsv", "image"):
                        doc["text"] = f"[File Path: {abs_path}]\n{doc['text']}"
                brain.ingest(docs)

    if args.list:
        docs = brain.list_documents()
        if not docs:
            print("  Knowledge base is empty.")
        else:
            total_chunks = sum(d["chunks"] for d in docs)
            print(f"\n  Knowledge Base — {len(docs)} file(s), {total_chunks} chunk(s)\n")
            print(f"  {'Source':<60} {'Chunks':>6}")
            print(f"  {'-'*60} {'-'*6}")
            for d in docs:
                print(f"  {d['source']:<60} {d['chunks']:>6}")
        return

    if args.query:
        if args.stream:
            for chunk in brain.query_stream(args.query):
                if isinstance(chunk, dict):
                    continue
                print(chunk, end="", flush=True)
            print()
        else:
            print(f"\n  Response:\n{brain.query(args.query)}")
        return

    # No query supplied — enter interactive REPL (streaming on by default)
    _quiet = args.quiet or not sys.stdin.isatty()
    try:
        _interactive_repl(brain, stream=True, init_display=_init_display, quiet=_quiet)
    except (KeyboardInterrupt, EOFError):
        pass
    print("\n  Bye!")
    # Manually flush readline history, then hard-exit to skip atexit handlers
    # (colorama/posthog atexit callbacks raise tracebacks on double Ctrl+C)
    try:
        import readline as _rl

        _hist = os.path.expanduser("~/.axon_history")
        _rl.write_history_file(_hist)
    except Exception:
        pass
    os._exit(0)


_SLASH_COMMANDS = [
    "/clear",
    "/compact",
    "/context",
    "/discuss",
    "/embed ",
    "/exit",
    "/help",
    "/ingest ",
    "/keys",
    "/list",
    "/llm ",
    "/model ",
    "/project ",
    "/pull ",
    "/quit",
    "/rag ",
    "/resume ",
    "/retry",
    "/search",
    "/sessions",
    "/vllm-url ",
]


def _make_completer(brain: "AxonBrain"):
    """Return a readline completer for slash commands, paths, and model names."""

    def completer(text: str, state: int):
        try:
            import readline

            full_line = readline.get_line_buffer()

            # Completing a slash command name
            if full_line.startswith("/") and " " not in full_line:
                matches = [c for c in _SLASH_COMMANDS if c.startswith(full_line)]
                return matches[state] if state < len(matches) else None

            # /ingest <path|glob> — complete filesystem paths
            if full_line.startswith("/ingest "):
                path_prefix = full_line[len("/ingest ") :]
                import glob as _glob

                matches = _glob.glob(path_prefix + "*")
                # Append / to directories
                matches = [m + "/" if os.path.isdir(m) else m for m in matches]
                return matches[state] if state < len(matches) else None

            # /model or /pull — complete Ollama model names
            if full_line.startswith("/model ") or full_line.startswith("/pull "):
                model_prefix = full_line.split(" ", 1)[1]
                try:
                    import ollama as _ollama

                    response = _ollama.list()
                    all_models = (
                        response.models
                        if hasattr(response, "models")
                        else response.get("models", [])
                    )
                    names = [
                        m.model if hasattr(m, "model") else m.get("name", "") for m in all_models
                    ]
                    matches = [n for n in names if n.startswith(model_prefix)]
                    return matches[state] if state < len(matches) else None
                except Exception:
                    return None

        except Exception:
            return None
        return None

    return completer


# ── Session persistence ────────────────────────────────────────────────────────
import json as _json  # noqa: E402
from datetime import datetime as _dt  # noqa: E402
from datetime import timezone as _tz  # noqa: E402

_SESSIONS_DIR = os.path.join(os.path.expanduser("~"), ".axon", "sessions")


def _sessions_dir(project: str | None = None) -> str:
    """Return the sessions directory for *project*, or the global fallback."""
    if project and project != "default":
        from axon.projects import project_sessions_path

        d = project_sessions_path(project)
    else:
        d = _SESSIONS_DIR
    os.makedirs(d, exist_ok=True)
    return d


def _new_session(brain: "AxonBrain") -> dict:
    return {
        "id": _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%S%f")[:-3],
        "started_at": _dt.now(_tz.utc).isoformat(),
        "provider": brain.config.llm_provider,
        "model": brain.config.llm_model,
        "project": getattr(brain, "_active_project", "default"),
        "history": [],
    }


def _session_path(session_id: str, project: str | None = None) -> str:
    return os.path.join(_sessions_dir(project), f"session_{session_id}.json")


def _save_session(session: dict) -> None:
    try:
        project = session.get("project")
        with open(_session_path(session["id"], project), "w", encoding="utf-8") as f:
            _json.dump(session, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _list_sessions(limit: int = 20, project: str | None = None) -> list:
    d = _sessions_dir(project)
    files = sorted(
        [f for f in os.listdir(d) if f.startswith("session_") and f.endswith(".json")],
        reverse=True,
    )[:limit]
    sessions = []
    for fn in files:
        try:
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                s = _json.load(f)
            sessions.append(s)
        except Exception:
            pass
    return sessions


def _load_session(session_id: str, project: str | None = None) -> dict | None:
    p = _session_path(session_id, project)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None


def _print_sessions(sessions: list) -> None:
    if not sessions:
        print("  (no saved sessions)")
        return
    print(f"\n  {'ID':<18}  {'Model':<30}  {'Turns':<6}  Started")
    print(f"  {'─'*18}  {'─'*30}  {'─'*6}  {'─'*20}")
    for s in sessions:
        turns = len(s.get("history", [])) // 2
        ts = s.get("started_at", "")[:16].replace("T", " ")
        model = f"{s.get('provider','?')}/{s.get('model','?')}"
        print(f"  {s['id']:<18}  {model:<30}  {turns:<6}  {ts}")
    print()


_MODEL_CTX: dict[str, int] = {
    "gemma": 8192,
    "gemma:2b": 8192,
    "gemma:7b": 8192,
    "llama3.1": 131072,
    "llama3.1:8b": 131072,
    "llama3.1:70b": 131072,
    "mistral": 32768,
    "mistral:7b": 32768,
    "phi3": 131072,
    "phi3:mini": 131072,
    "gemini-1.5-flash": 1048576,
    "gemini-1.5-pro": 2097152,
    "gemini-2.0-flash": 1048576,
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-flash-lite": 1048576,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-3.5-turbo": 16385,
}


def _infer_provider(model: str) -> str:
    """Guess LLM provider from model name.

    Returns "gemini" for gemini-* models, "openai" for gpt-*/o1-*/o3-*/o4-*
    models (without a colon, since Ollama uses name:tag format), and "ollama"
    for everything else (local models, including gpt-oss:tag Ollama models).
    """
    m = model.lower()
    if m.startswith("gemini-"):
        return "gemini"
    # OpenAI model names never contain ':'; Ollama uses name:tag format.
    if ":" not in m and m.startswith(("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    return "ollama"


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token for English text)."""
    return max(1, len(text) // 4)


def _token_bar(used: int, total: int, width: int = 20) -> str:
    """Return a visual fill bar: ████░░░░ 2,340 / 8,192 (28%)."""
    pct = min(used / total, 1.0) if total > 0 else 0
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    color = "[ok]" if pct < 0.6 else ("[!]" if pct < 0.85 else "[!!]")
    return f"{color} {bar}  {used:,} / {total:,} ({int(pct*100)}%)"


def _show_context(
    brain: "AxonBrain",
    chat_history: list,
    last_sources: list,
    last_query: str,
) -> None:
    """Display a formatted context window panel with token usage, model info, and chat history.

    Shows:
    - Model info: LLM provider/model and context window size; embedding provider/model
    - Token usage: Rough estimates (4 chars/token) with visual bar and color indicator
    - RAG settings: top_k, similarity_threshold, hybrid_search, rerank, hyde, multi_query toggles
    - Chat history: Last 10 turns (user/assistant messages)
    - Last retrieved sources: Up to 8 chunks with similarity scores and source names
    - System prompt: Full text (word-wrapped)

    All content is wrapped in a box with section separators for readability.

    Args:
        brain: AxonBrain instance to extract model and config info.
        chat_history: List of message dicts {"role": "user"|"assistant", "content": str}.
        last_sources: List of document dicts from last retrieval (with "vector_score", "metadata", "text").
        last_query: The user query that was used for the last retrieval.
    """
    W = _box_width()  # match main header box width
    TOP = f"  ╭{'─' * W}╮"
    BOTTOM = f"  ╰{'─' * W}╯"
    SEP = f"  ├{'─' * W}┤"
    BLANK = f"  │{' ' * W}│"

    def _wlen(s: str) -> int:
        """Terminal display width: wide Unicode chars (emoji, CJK) count as 2 columns."""
        extra = sum(
            1
            for c in s
            if "\U00001100" <= c <= "\U00001fff"
            or "\U00002e80" <= c <= "\U00002eff"
            or "\U00002f00" <= c <= "\U00002fdf"
            or "\U00003000" <= c <= "\U00003fff"
            or "\U00004e00" <= c <= "\U00009fff"
            or "\U0000a000" <= c <= "\U0000abff"
            or "\U0000ac00" <= c <= "\U0000d7ff"
            or "\U0000f900" <= c <= "\U0000faff"
            or "\U0000fe10" <= c <= "\U0000fe1f"
            or "\U0000fe30" <= c <= "\U0000fe4f"
            or "\U0000ff00" <= c <= "\U0000ff60"
            or "\U0000ffe0" <= c <= "\U0000ffe6"
            or "\U0001f000" <= c <= "\U0001ffff"
            or "\U00020000" <= c <= "\U0002ffff"
        )
        return len(s) + extra

    def row(text: str = "", indent: int = 4) -> str:
        content = " " * indent + text
        display_w = _wlen(content)
        if display_w > W:
            content = content[: W - 1] + "…"
            display_w = _wlen(content)
        pad = W - display_w
        return f"  │{content}{' ' * pad}│"

    def section(title: str) -> str:
        content = f"  ▸  {title}"
        pad = W - _wlen(content)
        return f"  │{content}{' ' * pad}│"

    def wrap_row(text: str, indent: int = 4, max_lines: int = 0) -> list:
        """Word-wrap text into multiple box rows. 0 = no limit."""
        avail = W - indent - 2  # 2-char right margin so text never crowds the border
        words = text.split()
        lines_out, current = [], ""
        for w in words:
            if len(current) + len(w) + (1 if current else 0) <= avail:
                current = f"{current} {w}" if current else w
            else:
                if current:
                    lines_out.append(row(current, indent))
                current = w
        if current:
            lines_out.append(row(current, indent))
        return lines_out if not max_lines else lines_out[:max_lines]

    # ── Token estimates ────────────────────────────────────────────────────────
    system_text = brain._build_system_prompt(False)
    sys_tokens = _estimate_tokens(system_text)
    hist_tokens = sum(_estimate_tokens(m["content"]) for m in chat_history)
    src_tokens = sum(_estimate_tokens(s.get("text", "")) for s in last_sources)
    total_used = sys_tokens + hist_tokens + src_tokens

    model_key = brain.config.llm_model.split(":")[0].lower()
    ctx_size = _MODEL_CTX.get(brain.config.llm_model, _MODEL_CTX.get(model_key, 8192))
    remaining = max(0, ctx_size - total_used)
    pct = min(total_used / ctx_size, 1.0) if ctx_size > 0 else 0
    bar_w = 40
    filled = int(pct * bar_w)
    bar = "█" * filled + "░" * (bar_w - filled)
    indicator = "[ok]" if pct < 0.6 else ("[!]" if pct < 0.85 else "[!!]")

    lines = [TOP, BLANK]
    lines.append(row("Context Window", indent=4))
    lines.append(BLANK)

    # ── Model section ──────────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("Model"))
    lines.append(SEP)
    lines.append(BLANK)
    lines.append(
        row(
            f"LLM    ·  {brain.config.llm_provider}/{brain.config.llm_model}"
            f"   ({ctx_size:,} token context window)"
        )
    )
    lines.append(row(f"Embed  ·  {brain.config.embedding_provider}/{brain.config.embedding_model}"))
    lines.append(BLANK)

    # ── Token usage ───────────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("Token Usage  (rough estimate — ~4 chars/token)"))
    lines.append(SEP)
    lines.append(BLANK)
    lines.append(
        row(f"{indicator} {bar}  {total_used:,} / {ctx_size:,}  ({int(pct*100)}%)", indent=4)
    )
    lines.append(BLANK)
    lines.append(row(f"{'System prompt':<22}{sys_tokens:>7,} tokens"))
    lines.append(
        row(f"{'Chat history':<22}{hist_tokens:>7,} tokens    ({len(chat_history) // 2} turns)")
    )
    lines.append(
        row(f"{'Retrieved context':<22}{src_tokens:>7,} tokens    ({len(last_sources)} chunks)")
    )
    lines.append(row("─" * 40))
    lines.append(row(f"{'Total':<22}{total_used:>7,} tokens    ({remaining:,} remaining)"))
    lines.append(BLANK)

    # ── RAG settings ──────────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("RAG Settings"))
    lines.append(SEP)
    lines.append(BLANK)
    lines.append(
        row(
            f"top-k · {brain.config.top_k}    "
            f"threshold · {brain.config.similarity_threshold}    "
            f"hybrid · {'ON' if brain.config.hybrid_search else 'OFF'}    "
            f"rerank · {'ON' if brain.config.rerank else 'OFF'}    "
            f"hyde · {'ON' if brain.config.hyde else 'OFF'}    "
            f"multi-query · {'ON' if brain.config.multi_query else 'OFF'}"
        )
    )
    lines.append(BLANK)

    # ── Chat history ───────────────────────────────────────────────────────────
    lines.append(SEP)
    turns = len(chat_history) // 2
    lines.append(section(f"Chat History  ({turns} turns)"))
    lines.append(SEP)
    lines.append(BLANK)
    if not chat_history:
        lines.append(row("(empty)"))
    else:
        shown = chat_history[-10:]
        for msg in shown:
            tag = "You   " if msg["role"] == "user" else "Brain "
            snip = msg["content"].replace("\n", " ")
            avail = W - 14
            if len(snip) > avail:
                snip = snip[:avail] + "…"
            lines.append(row(f"{tag}  {snip}"))
        if len(chat_history) > 10:
            lines.append(row(f"… {len(chat_history) - 10} earlier messages not shown"))
    lines.append(BLANK)

    # ── Last retrieved sources ─────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section(f"Last Retrieved Sources  ({len(last_sources)} chunks)"))
    lines.append(SEP)
    lines.append(BLANK)
    if last_query:
        lines.append(row(f'query · "{last_query[:W - 14]}"'))
        lines.append(BLANK)
    if not last_sources:
        lines.append(row("(no retrieval yet)"))
    else:
        for i, src in enumerate(last_sources[:8], 1):
            meta = src.get("metadata", {})
            name = os.path.basename(meta.get("source", src.get("id", "?")))
            score = src.get("vector_score", src.get("score", 0))
            kind = "web" if src.get("is_web") else "doc"
            score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            lines.append(row(f"{i:>2}. {kind} {score_bar} {score:.3f}   {name}"))
    lines.append(BLANK)

    # ── System prompt ──────────────────────────────────────────────────────────
    lines.append(SEP)
    lines.append(section("System Prompt"))
    lines.append(SEP)
    lines.append(BLANK)
    lines.extend(wrap_row(system_text.replace("\n", " "), indent=4))
    lines.append(BLANK)
    lines.append(BOTTOM)

    for line in lines:
        print(line)
    print()


def _do_compact(brain: "AxonBrain", chat_history: list) -> None:
    """Summarize chat history via LLM and replace it with a single summary turn.

    Condenses all messages in chat_history into a 4-6 sentence summary using the configured LLM.
    The original conversation is replaced with a single message prefixed with
    "[Conversation summary]: " to preserve context while freeing up token space.

    If chat_history is empty, prints a message and returns without action.

    Args:
        brain: AxonBrain instance used to call the LLM for summarization.
        chat_history: List of message dicts to summarize (modified in-place; emptied and refilled with summary).
    """
    if not chat_history:
        print("  Nothing to compact — chat history is empty.")
        return

    turns_before = len(chat_history)
    print(f"  ⠿ Compacting {turns_before} turns…", end="", flush=True)

    conversation = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in chat_history
    )
    summary_prompt = (
        "Summarize the following conversation in 4-6 concise sentences. "
        "Preserve all key facts, decisions, and topics discussed. "
        "Write in third person ('The user asked…'). "
        "Output only the summary, no preamble.\n\n"
        f"{conversation}"
    )
    try:
        summary = brain.llm.complete(summary_prompt, system_prompt=None, chat_history=[])
        chat_history.clear()
        chat_history.append({"role": "assistant", "content": f"[Conversation summary]: {summary}"})
        tokens_saved = _estimate_tokens(conversation) - _estimate_tokens(summary)
        print(f"\r  Compacted {turns_before} turns -> 1 summary  (~{tokens_saved:,} tokens freed)")
    except Exception as e:
        print(f"\r  Compact failed: {e}")


# ── Banner constants ───────────────────────────────────────────────────────────
_HINT = "  Type your question  ·  /help for commands  ·  Tab to autocomplete  ·  @file or @folder/ to attach context"


def _box_width() -> int:
    """Return inner box width: terminal columns minus 4, minimum 43."""
    return max(43, shutil.get_terminal_size((120, 24)).columns - 4)


# FIGlet "Big" ASCII art for AXON — all chars are 1-col wide, each line is 35 cols
_AXON_ART = [
    " █████╗ ██╗  ██╗ ██████╗ ███╗   ██╗",
    "██╔══██╗╚██╗██╔╝██╔═══██╗████╗  ██║",
    "███████║ ╚███╔╝ ██║   ██║██╔██╗ ██║",
    "██╔══██║ ██╔██╗ ██║   ██║██║╚██╗██║",
    "██║  ██║██╔╝ ██╗╚██████╔╝██║ ╚████║",
    "╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝",
]
# 24-bit blue gradient: light sky → cornflower → dodger → royal → medium → cobalt
_AXON_BLUE = [
    "\x1b[38;2;173;216;230m",  # light blue
    "\x1b[38;2;135;206;250m",  # light sky blue
    "\x1b[38;2;100;149;237m",  # cornflower blue
    "\x1b[38;2;30;144;255m",  # dodger blue
    "\x1b[38;2;65;105;225m",  # royal blue
    "\x1b[38;2;0;71;171m",  # cobalt blue
]
_AXON_RST = "\x1b[0m"


# Symmetrical brain/axon hub design (25 columns wide)
_BRAIN_ART = [
    "(O)~~.             .~~(O)",
    "  \\   *~._     _.~*   /  ",
    "[#]--.    ( O )    .--[#]",
    "[#]--'    ( O )    '--[#]",
    "  /   *~.'     '.~*   \\  ",
    "(O)~~'             '~~(O)",
]


def _get_brain_anim_row(row_idx: int, frame: int, width: int) -> str:
    """Return one row of the animated brain design, centered in `width`."""
    if width < 25:
        return " " * width

    pad = (width - 25) // 2
    l_pad = " " * pad
    r_pad = " " * (width - 25 - pad)

    line = _BRAIN_ART[row_idx]
    RST = "\x1b[0m"
    DIM = "\x1b[38;2;100;110;130m"  # Muted steel-blue (base connection)
    PULSE = "\x1b[38;2;180;210;255m"  # Pastel Blue (moving signal)
    GLOW = "\x1b[38;2;255;255;255m"  # Soft White (peak flash)

    # Staggered animation: 6 paths pulsing at different offsets in a 24-frame cycle
    total_cycle = 24
    # Paths: (phase, row, start, end)
    # Phase 0: Center, Phase 1: Inner path, Phase 2: Outer path, Phase 3: Cell/Dot
    tl_path = [(1, 1, 6, 10), (2, 0, 3, 6), (3, 0, 0, 3), (3, 1, 2, 3)]
    tr_path = [(1, 1, 15, 19), (2, 0, 19, 22), (3, 0, 22, 25), (3, 1, 22, 23)]
    ml_path = [(1, 2, 3, 6), (1, 3, 3, 6), (3, 2, 0, 3), (3, 3, 0, 3)]  # Mid-left
    mr_path = [(1, 2, 19, 22), (1, 3, 19, 22), (3, 2, 22, 25), (3, 3, 22, 25)]  # Mid-right
    bl_path = [(1, 4, 6, 10), (2, 5, 3, 6), (3, 5, 0, 3), (3, 4, 2, 3)]
    br_path = [(1, 4, 15, 19), (2, 5, 19, 22), (3, 5, 22, 25), (3, 4, 22, 23)]

    # (offset, path_segments)
    groups = [
        (0, tl_path),
        (4, br_path),
        (8, tr_path),
        (12, bl_path),
        (16, ml_path),
        (20, mr_path),
    ]

    char_levels = [0] * 25  # 0: DIM, 1: PULSE, 2: GLOW
    if frame == -1:
        # Fully connected state: all paths lit, center and cells glowing
        for _, path in groups:
            for phase, r, s, e in path + [(0, 2, 10, 15), (0, 3, 10, 15)]:
                if r == row_idx:
                    level = 2 if phase in (0, 3) else 1
                    for i in range(s, e):
                        char_levels[i] = max(char_levels[i], level)
    else:
        for offset, path in groups:
            rel = (frame - offset) % total_cycle
            # Add center to every path's start (phase 0)
            for phase, r, s, e in path + [(0, 2, 10, 15), (0, 3, 10, 15)]:
                if r == row_idx:
                    level = 0
                    if rel == phase:
                        level = 2
                    elif rel == (phase + 1) % total_cycle:
                        level = 1
                    for i in range(s, e):
                        char_levels[i] = max(char_levels[i], level)

    char_colors = [DIM] * 25
    for i, level in enumerate(char_levels):
        if level == 2:
            char_colors[i] = GLOW
        elif level == 1:
            char_colors[i] = PULSE

    res = ""
    curr_col = ""
    for i, char in enumerate(line):
        if char == " ":
            res += char
            continue
        if char_colors[i] != curr_col:
            res += char_colors[i]
            curr_col = char_colors[i]
        res += char
    return l_pad + res + RST + r_pad


def _brow(content: str, emoji_extra: int = 0) -> str:
    """One box row: pads/truncates content to exactly _box_width() terminal columns."""
    bw = _box_width()
    vis = len(content) + emoji_extra
    if vis > bw:
        content = content[: bw - emoji_extra - 1] + "…"
        vis = bw
    pad = bw - vis
    return f"  ┃{content}{' ' * pad}┃"


def _anim_pad(row_idx: int, frame: int, width: int) -> str:
    """Return an animated brain design centered in `width`."""
    return _get_brain_anim_row(row_idx, frame, width)


def _build_header(brain: "AxonBrain", tick_lines: list | None = None) -> list:
    """Return lines of the pinned header box (airy layout)."""
    model_s = f"{brain.config.llm_provider}/{brain.config.llm_model}"
    embed_s = f"{brain.config.embedding_provider}/{brain.config.embedding_model}"
    search_s = "ON  (Brave Search)" if brain.config.truth_grounding else "OFF"
    discuss_s = "ON" if brain.config.discussion_fallback else "OFF"
    hybrid_s = "ON" if brain.config.hybrid_search else "OFF"
    topk_s = str(brain.config.top_k)
    thr_s = str(brain.config.similarity_threshold)
    try:
        docs = brain.list_documents()
        doc_s = f"{sum(d['chunks'] for d in docs)} chunks  ({len(docs)} files)"
    except Exception:
        doc_s = "unknown"

    bw = _box_width()
    apad_w = max(0, bw - 39)  # 4 indent + 35 art cols = 39 vis cols

    # Build tick status — wrap onto a second row if too wide
    tick_items = [f"✓ {t}" for t in tick_lines] if tick_lines else ["✓ Ready"]
    ticks_s = "   ".join(tick_items)
    inner_w = bw - 4  # 4-char left indent "    "
    if len(ticks_s) > inner_w:
        # Split into two roughly equal halves at a separator boundary
        mid = len(tick_items) // 2
        ticks_s = "   ".join(tick_items[:mid])
        ticks_s2 = "   ".join(tick_items[mid:])
    else:
        ticks_s2 = None

    blank = f"  ┃{' ' * bw}┃"
    rows = [
        f"  \x1b[1m╭\x1b[22m{'━' * bw}\x1b[1m╮\x1b[22m",  # 1
        blank,  # 2
        *[  # 3-8  blue-shaded art lines + brain design
            f"  ┃    {_AXON_BLUE[i]}{line}{_AXON_RST}{_get_brain_anim_row(i, -1, apad_w)}┃"
            for i, line in enumerate(_AXON_ART)
        ],
        blank,  # 9
        _brow(f"    LLM    ·  {model_s}"),  # 6
        _brow(f"    Embed  ·  {embed_s}"),  # 7
        blank,  # 8
        _brow(f"    Search ·  {search_s:<26}  Discuss  ·  {discuss_s}"),  # 9
        _brow(
            f"    Docs   ·  {doc_s:<26}  Hybrid   ·  {hybrid_s}   top-k · {topk_s}   threshold · {thr_s}"
        ),  # 10
        blank,  # 11
        _brow(f"    {ticks_s}"),  # 12
    ]
    if ticks_s2:
        rows.append(_brow(f"    {ticks_s2}"))  # 12b (overflow)
    rows.append(f"  \x1b[1m╰\x1b[22m{'━' * bw}\x1b[1m╯\x1b[22m")  # 13
    return rows


def _draw_header(brain: "AxonBrain", tick_lines: list | None = None) -> None:
    """Clear screen and draw the welcome header box with LLM and embedding model info.

    Displays initialization status lines (e.g., "✓ Embedding ready [CPU]", "✓ BM25 · 42 docs").
    Clears the entire screen and redraws the header with hints for available REPL commands.
    Uses ANSI codes to clear and position the cursor — no scroll region (natural terminal scrollback).

    Args:
        brain: AxonBrain instance to extract model and provider information.
        tick_lines: Optional list of status messages (e.g., ["Starting", "Embedding ready [CPU]"])
                   to display in the header box.
    """
    bw = _box_width()
    sep = "  " + "─" * (bw + 2)
    lines = _build_header(brain, tick_lines)
    sys.stdout.write("\033[2J\033[H")  # clear screen, cursor to top-left
    for line in lines:
        sys.stdout.write(line + "\n")
    sys.stdout.write("\n" + _HINT + "\n" + sep + "\n\n")
    sys.stdout.flush()


def _print_recent_turns(history: list, n_turns: int = 2) -> None:
    """Print the last n_turns of Q&A below the header so context is visible.

    Args:
        history: chat_history list of {"role": ..., "content": ...} dicts.
        n_turns: Number of complete Q&A turns to show (each turn = 1 user + 1 assistant message).
    """
    if not history:
        return
    recent = history[-(n_turns * 2) :]
    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            sys.stdout.write(f"  \033[1;32mYou\033[0m: {content}\n")
        elif role == "assistant":
            # Cap very long responses so they don't flood the screen
            if len(content) > 600:
                content = content[:600] + "…"
            sys.stdout.write(f"\n  \033[1;33mAxon\033[0m:\n  {content}\n")
        sys.stdout.write("\n")
    sys.stdout.flush()


class _InitDisplay(logging.Handler):
    """Intercepts initialization log messages and renders animated status in a box.

    Displays a 7-line box with title and status line updated in-place using ANSI cursor positioning.
    Uses a braille spinner (⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏) that rotates every 0.08 seconds.
    Collects completed steps as checkmarks (✓) for the final banner display.

    The box is printed once at initialization, then the step line (line 5) is updated in-place
    as different initialization phases complete (Starting, Loading models, Vector store ready, etc.).
    """

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self) -> None:
        super().__init__()
        self._step: str = ""
        self._idx: int = 0
        self._anim_frame: int = 0
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.tick_lines: list = []  # collected for the final banner
        # Print CLOSED 7-line box immediately — step line updated in-place
        bw = _box_width()
        art_pad = " " * max(0, bw - 39)  # 4 indent + 35 art cols = 39 vis cols
        _art_rows = "".join(
            f"  ┃    {_AXON_BLUE[i]}{line}{_AXON_RST}{art_pad}┃\n"
            for i, line in enumerate(_AXON_ART)
        )
        sys.stdout.write(
            f"\n  \x1b[1m╭\x1b[22m{'━' * bw}\x1b[1m╮\x1b[22m\n"
            f"  ┃{' ' * bw}┃\n" + _art_rows + f"  ┃{' ' * bw}┃\n"
            f"  ┃{'    ⠿  Initializing…'.ljust(bw)}┃\n"  # step line (3rd from bottom)
            f"  ┃{' ' * bw}┃\n"
            f"  \x1b[1m╰\x1b[22m{'━' * bw}\x1b[1m╯\x1b[22m\n"
        )
        sys.stdout.flush()
        self._thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._thread.start()

    def _spin_loop(self) -> None:
        while not self._done.wait(0.08):
            with self._lock:
                self._anim_frame += 1
                bw = _box_width()
                apad_w = max(0, bw - 39)

                # Rebuild the 6 animated art rows (AXON text + signal animation)
                art_rows = [
                    f"  ┃    {_AXON_BLUE[i]}{_AXON_ART[i]}{_AXON_RST}"
                    f"{_anim_pad(i, self._anim_frame, apad_w)}┃"
                    for i in range(6)
                ]

                # Box line layout (1-indexed, cursor rests at line 14 after init print):
                #  1=blank  2=╭╮  3=┃blank┃  4-9=art  10=┃blank┃  11=┃step┃  12=┃blank┃  13=╰╯
                # Go up 10 from line 14 → line 4 (first art row), rewrite all 6.
                sys.stdout.write("\033[10A")
                for arow in art_rows:
                    sys.stdout.write(f"\r{arow}\n")
                # Cursor now at line 10 (blank row after art).

                if self._step:
                    spinner = self._FRAMES[self._idx % len(self._FRAMES)]
                    line = _brow(f"    {spinner}  {self._step}")
                    # Down 1 → line 11 (step), write, newline → 12, down 2 → 14.
                    sys.stdout.write(f"\033[1B\r{line}\n\033[2B")
                    self._idx += 1
                else:
                    # Skip blank(10) step(11) blank(12) bottom(13) → back to line 14.
                    sys.stdout.write("\033[4B")

                sys.stdout.flush()

    def _tick(self, label: str) -> None:
        with self._lock:
            self._step = ""
            self.tick_lines.append(label)
            line = _brow(f"    ✓  {label}")
            sys.stdout.write(f"\033[3A\r{line}\n\033[2B")
            sys.stdout.flush()

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "Initializing Axon" in msg:
            with self._lock:
                self._step = "Starting…"
        elif "Loading Sentence Transformers" in msg:
            m = re.search(r":\s*(.+)$", msg)
            with self._lock:
                self._step = f"Loading {m.group(1).strip() if m else 'model'}…"
        elif "Use pytorch device_name" in msg:
            m = re.search(r":\s*(.+)$", msg)
            self._tick(f"Embedding ready  [{m.group(1).strip() if m else 'cpu'}]")
        elif "Initializing ChromaDB" in msg:
            with self._lock:
                self._step = "Vector store…"
        elif "Loaded BM25 corpus" in msg:
            m = re.search(r"(\d+) documents", msg)
            self._tick("Vector store ready")
            self._tick(f"BM25  ·  {m.group(1) if m else '?'} docs")
        elif "Axon ready" in msg:
            self._done.set()

    def stop(self) -> None:
        self._done.set()
        with self._lock:
            self._step = ""
        self._thread.join(timeout=0.5)


_AT_TEXT_EXTS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".html",
    ".htm",
    ".css",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rs",
    ".go",
    ".rb",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".xml",
    ".ini",
    ".cfg",
    ".env",
    ".tf",
    ".proto",
    ".graphql",
}
# Extensions handled by dedicated loaders (extract clean text from binary formats)
_AT_LOADER_EXTS = {
    ".docx",
    ".pptx",
    ".pdf",
    ".bmp",
    ".png",
    ".jpg",
    ".jpeg",
}
_AT_DIR_MAX_BYTES = 120_000  # ~120 KB total across all files in a folder
_AT_FILE_MAX_BYTES = 40_000  # ~40 KB per single file


def _expand_at_files(text: str) -> str:
    """Expand @path references in user input with file/folder contents (read-only).

    - @file.txt / @file.docx / @file.pdf → inlines extracted text
    - @folder/   → recursively reads all supported files in the folder (capped at
                   _AT_DIR_MAX_BYTES total; unsupported / oversized files are skipped)
    """
    # Imported at call time so tests can patch axon.loaders.DOCXLoader / PDFLoader.
    from axon.loaders import DOCXLoader, PDFLoader

    def _read_text_file(path: str, max_bytes: int = _AT_FILE_MAX_BYTES) -> str:
        try:
            with open(path, "rb") as f:
                raw = f.read(max_bytes)
            text = raw.decode("utf-8", errors="ignore")
            truncated = os.path.getsize(path) > max_bytes
            return text + ("\n… (truncated)" if truncated else "")
        except OSError:
            return ""

    def _read_via_loader(path: str) -> str:
        try:
            ext = os.path.splitext(path)[1].lower()
            loader = DOCXLoader() if ext == ".docx" else PDFLoader()
            docs = loader.load(path)
            chunks = [d.get("text", "") for d in docs if d.get("text")]
            joined = "\n\n".join(chunks)
            if len(joined) > _AT_FILE_MAX_BYTES:
                joined = joined[:_AT_FILE_MAX_BYTES] + "\n… (truncated)"
            return joined
        except Exception as e:
            return f"(could not extract text: {e})"

    def _read_file(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext in _AT_LOADER_EXTS:
            return _read_via_loader(path)
        return _read_text_file(path)

    def _expand_dir(dirpath: str) -> str:
        supported = _AT_TEXT_EXTS | _AT_LOADER_EXTS
        parts: list[str] = []
        total = 0
        for root, dirs, files in os.walk(dirpath):
            dirs.sort()
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in sorted(files):
                if os.path.splitext(fname)[1].lower() not in supported:
                    continue
                fpath = os.path.join(root, fname)
                rel = os.path.relpath(fpath, dirpath)
                if total >= _AT_DIR_MAX_BYTES:
                    # Budget exhausted — skip remaining files without reading them
                    parts.append(f"\n--- @{rel} (skipped: context limit reached) ---")
                    continue
                content = _read_file(fpath)
                if not content:
                    continue
                encoded = content.encode("utf-8", errors="ignore")
                next_size = len(encoded)
                if total + next_size > _AT_DIR_MAX_BYTES:
                    remaining = _AT_DIR_MAX_BYTES - total
                    truncated_text = encoded[:remaining].decode("utf-8", errors="ignore")
                    parts.append(f"\n--- @{rel} ---\n{truncated_text}\n… (truncated)\n--- end ---")
                    total = _AT_DIR_MAX_BYTES
                else:
                    parts.append(f"\n--- @{rel} ---\n{content}\n--- end ---")
                    total += next_size
        return "\n".join(parts) if parts else f"\n(no readable files found in {dirpath})"

    def _replace(m: re.Match) -> str:
        path = m.group(1).rstrip("/\\")
        if os.path.isdir(path):
            return f"\n\n=== folder: {path} ===\n{_expand_dir(path)}\n=== end folder ===\n"
        if os.path.isfile(path):
            content = _read_file(path)
            if content:
                return f"\n\n--- @{path} ---\n{content}\n--- end ---\n"
        return m.group(0)

    return re.sub(r"@(\S+)", _replace, text)


def _interactive_repl(
    brain: "AxonBrain",
    stream: bool = True,
    init_display: "_InitDisplay | None" = None,
    quiet: bool = False,
) -> None:
    """Interactive REPL chat session with session persistence and live tab completion.

    Features:
    - Session persistence: auto-saves to ~/.axon/sessions/session_<timestamp>.json
    - Live tab completion: slash commands, filesystem paths, Ollama model names via prompt_toolkit
    - Animated spinners: braille spinner during init and LLM generation (disabled in quiet mode)
    - Slash commands: /help, /list, /ingest, /model, /embed, /pull, /search, /discuss, /rag,
      /compact, /context, /sessions, /resume, /retry, /clear, /project, /keys, /vllm-url, /quit, /exit
    - @file/folder context: type @path/file.txt or @path/folder/ to inline contents into your query (read-only)
    - Shell passthrough: !command runs a shell command without leaving the REPL
    - Pinned status info: token usage, model info, RAG settings visible at terminal bottom

    Args:
        brain: AxonBrain instance to use for queries.
        stream: If True, streams LLM response token-by-token; if False, waits for full response.
        init_display: Optional _InitDisplay handler to stop after initialization.
        quiet: Suppress spinners and progress bars (auto-enabled for non-TTY stdin).
    """
    import warnings

    warnings.filterwarnings("ignore", category=FutureWarning, module="google")

    # Silence INFO logs — they clutter the interactive UI
    import logging as _logging

    for _log in (
        "Axon",
        "Axon.Retrievers",
        "httpx",
        "sentence_transformers",
        "chromadb",
        "httpcore",
    ):
        _lg = _logging.getLogger(_log)
        _lg.setLevel(_logging.WARNING)
        _lg.propagate = False  # prevent bubbling to root logger

    # ── Input: prefer prompt_toolkit (live completions), fall back to readline ──
    _pt_session = None
    try:
        import glob as _pglob

        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.formatted_text import ANSI as _PTANSI
        from prompt_toolkit.formatted_text import HTML as _PThtml
        from prompt_toolkit.formatted_text import FormattedText as _PTFT  # noqa: F401
        from prompt_toolkit.history import FileHistory as _FileHistory
        from prompt_toolkit.styles import Style

        _HIST_DIR = os.path.expanduser("~/.axon")
        os.makedirs(_HIST_DIR, exist_ok=True)
        _HIST_FILE = os.path.join(_HIST_DIR, "repl_history")

        _PT_STYLE = Style.from_dict(
            {
                "": "",
                "completion-menu.completion.current": "bg:#444466 #ffffff",
                "bottom-toolbar": "bg:#0a2a5e #c8d8f0",
            }
        )

        class _PTCompleter(Completer):
            def __init__(self, brain_ref):
                self._brain = brain_ref

            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                # ── slash command name ─────────────────────────────────────
                if text.startswith("/") and " " not in text:
                    for cmd in _SLASH_COMMANDS:
                        c = cmd.rstrip()
                        if c.startswith(text):
                            yield Completion(c[len(text) :], display=c, display_meta="command")
                # ── /ingest path / glob ───────────────────────────────────
                elif text.startswith("/ingest "):
                    prefix = text[len("/ingest ") :]
                    for p in _pglob.glob(prefix + "*"):
                        disp = p + ("/" if os.path.isdir(p) else "")
                        yield Completion(p[len(prefix) :], display=disp)
                # ── /model <provider/model> ───────────────────────────────
                elif text.startswith("/model ") or text.startswith("/embed "):
                    cmd_len = len("/model ") if text.startswith("/model ") else len("/embed ")
                    prefix = text[cmd_len:]
                    try:
                        import ollama as _ol

                        resp = _ol.list()
                        mods = resp.models if hasattr(resp, "models") else resp.get("models", [])
                        for m in mods:
                            name = m.model if hasattr(m, "model") else m.get("name", "")
                            if name.startswith(prefix):
                                yield Completion(name[len(prefix) :], display=name)
                    except Exception:
                        pass
                # ── /resume <session-id> ──────────────────────────────────
                elif text.startswith("/resume "):
                    prefix = text[len("/resume ") :]
                    for s in _list_sessions(project=brain._active_project):
                        sid = s["id"]
                        if sid.startswith(prefix):
                            turns = len(s.get("history", [])) // 2
                            yield Completion(
                                sid[len(prefix) :], display=sid, display_meta=f"{turns} turns"
                            )
                # ── /llm <option> ─────────────────────────────────────────
                elif text.startswith("/llm "):
                    opts = ["temperature "]
                    prefix = text[len("/llm ") :]
                    for o in opts:
                        if o.startswith(prefix):
                            yield Completion(o[len(prefix) :], display=o)
                # ── /rag <option> ─────────────────────────────────────────
                elif text.startswith("/rag "):
                    opts = [
                        "topk ",
                        "threshold ",
                        "hybrid",
                        "rerank",
                        "rerank-model ",
                        "hyde",
                        "multi",
                        "step-back",
                        "decompose",
                        "compress",
                        "cite",
                        "raptor",
                        "graph-rag",
                    ]
                    prefix = text[len("/rag ") :]
                    for o in opts:
                        if o.startswith(prefix):
                            yield Completion(o[len(prefix) :], display=o)
                # ── @file context attachment ──────────────────────────────
                elif "@" in text:
                    at_pos = text.rfind("@")
                    prefix = text[at_pos + 1 :]
                    for p in _pglob.glob(prefix + "*"):
                        disp = p + ("/" if os.path.isdir(p) else "")
                        yield Completion(
                            p[len(prefix) :], display=disp, display_meta="file context"
                        )
                # ── /project <subcommand> ──────────────────────────────────
                elif text.startswith("/project "):
                    sub_opts = ["list", "new ", "switch ", "delete ", "folder"]
                    prefix = text[len("/project ") :]
                    for o in sub_opts:
                        if o.startswith(prefix):
                            yield Completion(o[len(prefix) :], display=o)
                    # also complete project names for switch/delete
                    if text.startswith(("/project switch ", "/project delete ")):
                        cmd_len = (
                            len("/project switch ")
                            if text.startswith("/project switch ")
                            else len("/project delete ")
                        )
                        pfx = text[cmd_len:]
                        try:
                            from axon.projects import list_projects

                            for p in list_projects():
                                n = p["name"]
                                if n.startswith(pfx):
                                    yield Completion(n[len(pfx) :], display=n)
                        except Exception:
                            pass

        def _toolbar():
            def _t(s: str, w: int) -> str:
                return s if len(s) <= w else s[: w - 1] + "…"

            # No explicit background codes — the bottom-toolbar class paints
            # bg:#0a2a5e for every cell uniformly.  Bold labels only change
            # font weight; the class background is never overridden.
            _BON = "\x1b[1m"  # bold on
            _BOF = "\x1b[22m"  # bold off
            _RST = "\x1b[0m"

            def _lbl(text: str) -> str:
                return f"{_BON}{text}{_BOF}"

            def _pad(label: str, val: str, width: int) -> str:
                return " " * max(0, width - len(label) - 1 - len(str(val)))

            m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
            emb = f"{brain.config.embedding_provider}/{brain.config.embedding_model}"
            try:
                docs = brain.vector_store.collection.count()
                doc_s = f"{docs} chunks"
            except Exception:
                doc_s = "?"
            proj = getattr(brain, "_active_project", "default")
            _proj_display = proj.replace("/", " > ") if proj != "default" else ""
            _merged = isinstance(brain.vector_store, MultiVectorStore)
            _merged_tag = " [merged]" if _merged else ""
            proj_s = f"  │  {_proj_display}{_merged_tag}" if proj != "default" else ""
            sep = "  │  "
            W1, W2 = 28, 30
            C1 = len("LLM  ") + W1  # 33
            C2 = len("Embed  ") + W2  # 37
            s_state = "ON" if brain.config.truth_grounding else "off"
            d_state = "ON" if brain.config.discussion_fallback else "off"
            h_state = "ON" if brain.config.hybrid_search else "off"

            row1 = (
                f"  {_lbl('LLM')}  {_t(m, W1):{W1}}{sep}"
                f"{_lbl('Embed')}  {_t(emb, W2):{W2}}{sep}"
                f"{_lbl('Docs')}  {doc_s}"
            )
            row2 = (
                f"  {_lbl('search')}:{s_state}{_pad('search', s_state, C1)}{sep}"
                f"{_lbl('discuss')}:{d_state}{_pad('discuss', d_state, C2)}{sep}"
                f"{_lbl('hybrid')}:{h_state}  "
                f"{_lbl('top-k')}:{brain.config.top_k}  "
                f"{_lbl('thr')}:{brain.config.similarity_threshold}"
                f"{proj_s}"
            )
            return _PTANSI(f"{row1}\n{row2}{_RST}")

        _pt_session = PromptSession(
            completer=_PTCompleter(brain),
            auto_suggest=AutoSuggestFromHistory(),
            style=_PT_STYLE,
            complete_while_typing=True,
            bottom_toolbar=_toolbar,
            history=_FileHistory(_HIST_FILE),
        )
    except ImportError:
        # Fall back to readline with history persistence
        try:
            import atexit
            import readline

            _hist_file = os.path.expanduser("~/.axon/repl_history")
            os.makedirs(os.path.dirname(_hist_file), exist_ok=True)
            try:
                readline.read_history_file(_hist_file)
            except FileNotFoundError:
                pass
            readline.set_history_length(2000)
            atexit.register(readline.write_history_file, _hist_file)
            readline.set_completer(_make_completer(brain))
            readline.set_completer_delims("")
            readline.parse_and_bind("tab: complete")
            readline.parse_and_bind(r'"\C-l": clear-screen')
        except ImportError:
            pass

    def _print_status_bar() -> None:
        """Reprint the 2-row status bar to stdout after each response."""

        def _t(s: str, w: int) -> str:
            return s if len(s) <= w else s[: w - 1] + "…"

        m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
        emb = f"{brain.config.embedding_provider}/{brain.config.embedding_model}"
        try:
            docs = brain.vector_store.collection.count()
            doc_s = f"{docs} chunks"
        except Exception:
            doc_s = "?"
        s_val = "search:ON" if brain.config.truth_grounding else "search:off"
        d_val = "discuss:ON" if brain.config.discussion_fallback else "discuss:off"
        h_val = "hybrid:ON" if brain.config.hybrid_search else "hybrid:off"
        tk = f"top-k:{brain.config.top_k}  thr:{brain.config.similarity_threshold}"
        proj = getattr(brain, "_active_project", "default")
        _proj_display = proj.replace("/", " > ") if proj != "default" else ""
        _merged = isinstance(brain.vector_store, MultiVectorStore)
        _merged_tag = " [merged]" if _merged else ""
        proj_s = f"  │  {_proj_display}{_merged_tag}" if proj != "default" else ""
        sep = "  │  "
        W1, W2 = 28, 30
        C1 = len("LLM  ") + W1  # 33
        C2 = len("Embed  ") + W2  # 37
        row1 = (
            f"  \033[1mLLM\033[0m\033[2m  {_t(m, W1):{W1}}"
            f"{sep}\033[0m\033[1mEmbed\033[0m\033[2m  {_t(emb, W2):{W2}}"
            f"{sep}\033[0m\033[1mDocs\033[0m\033[2m  {doc_s}\033[0m"
        )
        row2 = f"\033[2m  {s_val:<{C1}}" f"{sep}{d_val:<{C2}}" f"{sep}{h_val}  {tk}{proj_s}\033[0m"
        print(row1)
        print(row2)

    def _read_input(prompt: str = "") -> str:
        if _pt_session:
            _p = _PThtml("<ansigreen><b>You</b></ansigreen>: ") if not prompt else prompt
            return _pt_session.prompt(_p)
        return input(prompt if prompt else "\033[1;32mYou\033[0m: ")

    # REPL is conversational — always let the LLM answer even with no RAG hits
    brain.config.discussion_fallback = True

    _tick_lines = init_display.tick_lines if init_display else []
    _draw_header(brain, _tick_lines)

    # ── Session init ───────────────────────────────────────────────────────────
    session: dict = _new_session(brain)
    chat_history: list = session["history"]

    _last_sources: list = []
    _last_query: str = ""

    # Initial snapshot to avoid printing status on the very first query
    m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
    s_v = "search:ON" if brain.config.truth_grounding else "search:off"
    d_v = "discuss:ON" if brain.config.discussion_fallback else "discuss:off"
    h_v = "hybrid:ON" if brain.config.hybrid_search else "hybrid:off"
    tk = f"top-k:{brain.config.top_k}"
    thr = f"thr:{brain.config.similarity_threshold}"
    _last_config_snapshot: tuple = (m, s_v, d_v, h_v, tk, thr)

    while True:
        try:
            user_input = _read_input().strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        # --- Shell passthrough: !command ---
        if user_input.startswith("!"):
            shell_cmd = user_input[1:].strip()
            if shell_cmd:
                import subprocess

                subprocess.run(shell_cmd, shell=True)
            continue

        # --- Slash commands ---
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            # Allow "/<cmd> help" as alias for "/help <cmd>"
            if arg.strip().lower() == "help" and cmd not in ("/help",):
                arg = cmd.lstrip("/")
                cmd = "/help"
            if cmd in ("/quit", "/exit", "/q"):
                break

            elif cmd == "/help":
                if arg:
                    # Per-command detail
                    _detail = {
                        "model": "  /model <model>              keep current provider\n"
                        "  /model <provider>/<model>   switch provider + model\n"
                        "  providers: ollama, gemini, openai, ollama_cloud, vllm\n"
                        "  e.g.  /model gemini/gemini-2.0-flash\n"
                        "        /model ollama/gemma:2b\n"
                        "        /model openai/gpt-4o\n"
                        "        /model vllm/meta-llama/Llama-3.1-8B-Instruct",
                        "embed": "  /embed <model>              keep current provider\n"
                        "  /embed <provider>/<model>   switch provider + model\n"
                        "  /embed /path/to/local        local HuggingFace folder\n"
                        "  providers: sentence_transformers, ollama, fastembed, openai\n"
                        "  !  Re-ingest after changing embedding model.",
                        "ingest": "  /ingest <path>              ingest a directory\n"
                        "  /ingest ./src/*.py           glob pattern\n"
                        "  /ingest ./notes/**/*.md      recursive glob",
                        "llm": "  /llm                         show LLM settings (provider, model, temperature)\n"
                        "  /llm temperature <0.0–2.0>   set generation temperature\n"
                        "  Lower temperature = more deterministic; higher = more creative.",
                        "rag": "  /rag                         show all RAG settings\n"
                        "  /rag topk <n>                results to retrieve (1–20)\n"
                        "  /rag threshold <0.0–1.0>     min similarity score\n"
                        "  /rag hybrid                  toggle hybrid BM25+vector\n"
                        "  /rag rerank                  toggle cross-encoder reranker\n"
                        "  /rag rerank-model <model>    set reranker model (HF ID or local path)\n"
                        "  /rag hyde                    toggle HyDE query expansion\n"
                        "  /rag multi                   toggle multi-query expansion\n"
                        "  /rag step-back               toggle step-back prompting\n"
                        "  /rag decompose               toggle query decomposition\n"
                        "  /rag compress                toggle LLM context compression\n"
                        "  /rag cite                    toggle inline [Document N] citations\n"
                        "  /rag raptor                  toggle RAPTOR hierarchical indexing\n"
                        "  /rag graph-rag               toggle GraphRAG entity retrieval",
                        "sessions": "  /sessions                    list recent saved sessions\n"
                        "  /resume <id>                 load a session by ID\n"
                        "  Sessions auto-save after each turn.",
                        "keys": "  /keys                        show API key status for all providers\n"
                        "  /keys set <provider>         interactively set an API key\n"
                        "  providers: gemini, openai, brave, ollama_cloud\n"
                        "  Keys are saved to ~/.axon/.env and loaded at startup.",
                        "project": "  /project                          show active project + list all\n"
                        "  /project list                      list all projects (tree view)\n"
                        "  /project new <name>                create a new project and switch to it\n"
                        "  /project new <name> <desc>         create with description\n"
                        "  /project new <parent>/<child>      create a sub-project (up to 3 levels)\n"
                        "  /project switch <name>             switch to an existing project\n"
                        "  /project switch <parent>/<child>   switch to a sub-project\n"
                        "  /project switch default            return to the global knowledge base\n"
                        "  /project delete <name>             delete a leaf project and its data\n"
                        "  /project folder                    open the active project folder\n"
                        "\n"
                        "  Projects are stored in ~/.axon/projects/<name>/\n"
                        "  Sub-projects use nested subs/ directories (max depth: 3).\n"
                        "  Switching to a parent project shows merged data across all sub-projects.\n"
                        "  Use /ingest after switching to add documents to the current project.",
                    }
                    key = arg.lstrip("/")
                    if key in _detail:
                        print(f"\n{_detail[key]}\n")
                    else:
                        print(f"  No detail for '{arg}'. Available: {', '.join(_detail)}")
                else:
                    print(
                        "\n"
                        "  /clear          clear knowledge base for current project\n"
                        "  /compact        summarise conversation to free context\n"
                        "  /context        show current conversation context size\n"
                        "  /discuss        toggle discussion fallback (general knowledge)\n"
                        "  /embed [model]  show or switch embedding model\n"
                        "  /help [cmd]     show this help or details for a command\n"
                        "  /ingest <path>  ingest a file, directory, or glob\n"
                        "  /keys           show/set API keys (gemini, openai, brave, ollama_cloud)\n"
                        "  /list           list ingested documents\n"
                        "  /llm [opt val]  show or set LLM settings (temperature)\n"
                        "  /model [model]  show or switch LLM model\n"
                        "  /project [sub]  manage projects (list, new, switch, delete, folder)\n"
                        "  /pull <name>    pull an Ollama model\n"
                        "  /quit           exit Axon\n"
                        "  /rag [opt val]  show or set retrieval settings (topk, threshold, hybrid, …)\n"
                        "  /resume <id>    load a saved session\n"
                        "  /retry          retry the last query\n"
                        "  /search         toggle Brave web search fallback\n"
                        "  /sessions       list recent saved sessions\n"
                        "\n"
                        "  Shell:   !<cmd>  run a shell command\n"
                        "  Files:   @<file>  attach file context  ·  @<folder>/  attach all text files\n"
                        "\n"
                        "  /help <cmd>  for details  ·  e.g.  /help rag   /help llm   /help project\n"
                        "  Tab  autocomplete  ·  ↑↓  history  ·  Ctrl+C  cancel  ·  Ctrl+D  exit\n"
                    )

            elif cmd == "/list":
                docs = brain.list_documents()
                if not docs:
                    print("  Knowledge base is empty.")
                else:
                    total = sum(d["chunks"] for d in docs)
                    print(f"\n  {len(docs)} file(s), {total} chunk(s)\n")
                    for d in docs:
                        print(f"  {d['source']:<60} {d['chunks']:>6}")
                    print()

            elif cmd == "/ingest":
                if not arg:
                    print("  Usage: /ingest <path|glob>  e.g. /ingest ./docs  /ingest ./src/*.py")
                else:
                    from axon.projects import ensure_project

                    # Prompt to create a project if none exist and currently in 'default'
                    if brain.should_recommend_project():
                        try:
                            print(
                                "\n  \033[1mNote\033[0m: You are about to ingest into the 'default' project."
                            )
                            print(
                                "  It is recommended to create a dedicated project to keep your data organized."
                            )
                            confirm = (
                                _read_input("  Create a new project now? [y/N]: ").strip().lower()
                            )
                            if confirm == "y":
                                new_name = _read_input("  New project name: ").strip().lower()
                                if new_name:
                                    try:
                                        ensure_project(new_name)
                                        brain.switch_project(new_name)
                                        print(f"  Switched to project '{new_name}'.\n")
                                    except ValueError as e:
                                        print(f"  {e}")
                        except (EOFError, KeyboardInterrupt):
                            print("\n  Cancelled project check.")

                    import glob as _glob

                    from axon.loaders import DirectoryLoader

                    # Expand glob pattern; fallback to literal path
                    matched = sorted(_glob.glob(arg, recursive=True))
                    if not matched:
                        # No glob match — try as plain directory
                        if os.path.isdir(arg):
                            matched = [arg]
                        else:
                            print(f"  No files matched: {arg}")
                    if matched:
                        loader_mgr = DirectoryLoader()
                        ingested, skipped = 0, 0
                        for path in matched:
                            if os.path.isdir(path):
                                print(f"  {path} …", end="", flush=True)
                                asyncio.run(brain.load_directory(path))
                                print("  done")
                                ingested += 1
                            elif os.path.isfile(path):
                                ext = os.path.splitext(path)[1].lower()
                                if ext in loader_mgr.loaders:
                                    brain.ingest(loader_mgr.loaders[ext].load(path))
                                    print(f"  {path}")
                                    ingested += 1
                                else:
                                    print(f"  !  Skipped (unsupported type): {path}")
                                    skipped += 1
                        print(f"  Done — {ingested} ingested, {skipped} skipped.")

            elif cmd == "/model":
                _PROVIDERS = ("ollama", "gemini", "openai", "ollama_cloud", "vllm")
                if not arg:
                    print(f"  LLM:       {brain.config.llm_provider}/{brain.config.llm_model}")
                    print(
                        f"  Embedding: {brain.config.embedding_provider}/{brain.config.embedding_model}"
                    )
                    print("  Usage:   /model <model>              (auto-detect provider)")
                    print("           /model <provider>/<model>   (switch provider too)")
                    print(f"  Providers: {', '.join(_PROVIDERS)}")
                    print(
                        f"  vLLM URL:  {brain.config.vllm_base_url}  (change with /vllm-url <url>)"
                    )
                elif "/" in arg:
                    provider, model = arg.split("/", 1)
                    if provider not in _PROVIDERS:
                        print(
                            f"  Unknown provider '{provider}'. Choose from: {', '.join(_PROVIDERS)}"
                        )
                    else:
                        brain.config.llm_provider = provider
                        brain.config.llm_model = model
                        brain.llm = OpenLLM(brain.config)
                        print(f"  Switched LLM to {provider}/{model}")
                        if provider == "vllm":
                            print(
                                f"  ℹ️  vLLM server: {brain.config.vllm_base_url}  (change with /vllm-url <url>)"
                            )
                        elif provider != "ollama":
                            print("  ℹ️  Make sure the required API key env var is set.")
                else:
                    inferred = _infer_provider(arg)
                    brain.config.llm_provider = inferred
                    brain.config.llm_model = arg
                    brain.llm = OpenLLM(brain.config)
                    print(f"  Switched LLM to {inferred}/{arg}")
                    if inferred != "ollama":
                        print("  ℹ️  Make sure the required API key env var is set.")

            elif cmd == "/vllm-url":
                if not arg:
                    print(f"  Current vLLM base URL: {brain.config.vllm_base_url}")
                    print("  Usage: /vllm-url http://host:port/v1")
                else:
                    brain.config.vllm_base_url = arg
                    brain.llm._openai_clients = {}  # invalidate cached client
                    print(f"  vLLM base URL set to {arg}")

            elif cmd == "/embed":
                _EMBED_PROVIDERS = ("sentence_transformers", "ollama", "fastembed", "openai")
                if not arg:
                    print(
                        f"  Current:   {brain.config.embedding_provider}/{brain.config.embedding_model}"
                    )
                    print("  Usage:   /embed <model>              (keep current provider)")
                    print("           /embed <provider>/<model>   (switch provider too)")
                    print(f"  Providers: {', '.join(_EMBED_PROVIDERS)}")
                    print("  Examples:")
                    print("    /embed all-MiniLM-L6-v2                    (sentence_transformers)")
                    print("    /embed /path/to/local/model                (local folder)")
                    print("    /embed ollama/nomic-embed-text")
                    print("    /embed fastembed/BAAI/bge-small-en")
                    print("  !  Changing embedding model invalidates existing indexed documents.")
                else:
                    if "/" in arg:
                        provider, model = arg.split("/", 1)
                        if provider not in _EMBED_PROVIDERS:
                            # Could be a path like /home/user/model — treat as local st path
                            provider = brain.config.embedding_provider
                            model = arg
                        else:
                            brain.config.embedding_provider = provider
                            brain.config.embedding_model = model
                    else:
                        provider = brain.config.embedding_provider
                        model = arg
                        brain.config.embedding_model = model
                    try:
                        print("  ⠿ Loading embedding model…", end="", flush=True)
                        brain.embedding = OpenEmbedding(brain.config)
                        print(
                            f"\r  Embedding switched to {brain.config.embedding_provider}/{brain.config.embedding_model}"
                        )
                        print("  Re-ingest your documents so they use the new embedding model.")
                    except Exception as e:
                        print(f"\r  Failed to load embedding: {e}")

            elif cmd == "/pull":
                if not arg:
                    print("  Usage: /pull <model-name>")
                else:
                    try:
                        import ollama as _ollama

                        print(f"  Pulling '{arg}' …")
                        last_status = ""
                        for chunk in _ollama.pull(arg, stream=True):
                            status = (
                                chunk.get("status", "")
                                if isinstance(chunk, dict)
                                else getattr(chunk, "status", "")
                            )
                            total = (
                                chunk.get("total", 0)
                                if isinstance(chunk, dict)
                                else getattr(chunk, "total", 0)
                            )
                            completed = (
                                chunk.get("completed", 0)
                                if isinstance(chunk, dict)
                                else getattr(chunk, "completed", 0)
                            )
                            if total and completed:
                                line = f"  {status}: {int(completed/total*100)}%"
                            elif status:
                                line = f"  {status}"
                            else:
                                continue
                            # Pad to clear previous longer line
                            print(f"\r{line:<60}", end="", flush=True)
                            last_status = line  # noqa: F841
                        print(f"\r  '{arg}' ready.{' ' * 50}")
                    except Exception as e:
                        print(f"  Pull failed: {e}")

            elif cmd == "/clear":
                chat_history.clear()
                print("  Chat history cleared.")

            elif cmd == "/search":
                if brain.config.offline_mode:
                    print("  Offline mode is ON — web search is disabled.")
                elif brain.config.truth_grounding:
                    brain.config.truth_grounding = False
                    print("  Web search OFF — answers from local knowledge only.")
                else:
                    if not brain.config.brave_api_key:
                        print("  BRAVE_API_KEY is not set. Export it and restart, or set it with:")
                        print("     export BRAVE_API_KEY=your_key")
                    else:
                        brain.config.truth_grounding = True
                        print(
                            "  Web search ON — Brave Search will be used as fallback when local knowledge is insufficient."
                        )

            elif cmd == "/discuss":
                brain.config.discussion_fallback = not brain.config.discussion_fallback
                state = "ON" if brain.config.discussion_fallback else "OFF"
                print(f"  Discussion mode {state}.")

            elif cmd == "/rag":
                if not arg:
                    print(
                        f"\n  top-k        · {brain.config.top_k}\n"
                        f"  threshold    · {brain.config.similarity_threshold}\n"
                        f"  hybrid       · {'ON' if brain.config.hybrid_search else 'OFF'}\n"
                        f"  rerank       · {'ON' if brain.config.rerank else 'OFF'}"
                        + (f"  [{brain.config.reranker_model}]" if brain.config.rerank else "")
                        + "\n"
                        f"  hyde         · {'ON' if brain.config.hyde else 'OFF'}\n"
                        f"  multi-query  · {'ON' if brain.config.multi_query else 'OFF'}\n"
                        f"  step-back    · {'ON' if brain.config.step_back else 'OFF'}\n"
                        f"  decompose    · {'ON' if brain.config.query_decompose else 'OFF'}\n"
                        f"  compress     · {'ON' if brain.config.compress_context else 'OFF'}\n"
                        f"  raptor       · {'ON' if brain.config.raptor else 'OFF'}\n"
                        f"  graph-rag    · {'ON' if brain.config.graph_rag else 'OFF'}\n"
                        f"\n  /help rag   for usage details\n"
                    )
                else:
                    rag_parts = arg.split(maxsplit=1)
                    rag_opt = rag_parts[0].lower()
                    rag_val = rag_parts[1] if len(rag_parts) > 1 else ""
                    if rag_opt == "topk":
                        try:
                            n = int(rag_val)
                            assert 1 <= n <= 50
                            brain.config.top_k = n
                            print(f"  top-k set to {n}")
                        except Exception:
                            print("  Usage: /rag topk <integer 1–50>")
                    elif rag_opt == "threshold":
                        try:
                            v = float(rag_val)
                            assert 0.0 <= v <= 1.0
                            brain.config.similarity_threshold = v
                            print(f"  threshold set to {v}")
                        except Exception:
                            print("  Usage: /rag threshold <float 0.0–1.0>")
                    elif rag_opt == "hybrid":
                        brain.config.hybrid_search = not brain.config.hybrid_search
                        print(f"  Hybrid search {'ON' if brain.config.hybrid_search else 'OFF'}")
                    elif rag_opt == "rerank":
                        brain.config.rerank = not brain.config.rerank
                        print(f"  Reranker {'ON' if brain.config.rerank else 'OFF'}")
                    elif rag_opt == "hyde":
                        brain.config.hyde = not brain.config.hyde
                        print(f"  HyDE {'ON' if brain.config.hyde else 'OFF'}")
                    elif rag_opt == "multi":
                        brain.config.multi_query = not brain.config.multi_query
                        print(f"  Multi-query {'ON' if brain.config.multi_query else 'OFF'}")
                    elif rag_opt == "step-back":
                        brain.config.step_back = not brain.config.step_back
                        print(f"  Step-back prompting {'ON' if brain.config.step_back else 'OFF'}")
                    elif rag_opt == "decompose":
                        brain.config.query_decompose = not brain.config.query_decompose
                        print(
                            f"  Query decomposition {'ON' if brain.config.query_decompose else 'OFF'}"
                        )
                    elif rag_opt == "compress":
                        brain.config.compress_context = not brain.config.compress_context
                        print(
                            f"  Context compression {'ON' if brain.config.compress_context else 'OFF'}"
                        )
                    elif rag_opt == "cite":
                        brain.config.cite = not brain.config.cite
                        print(f"  Inline citations {'ON' if brain.config.cite else 'OFF'}")
                    elif rag_opt == "raptor":
                        brain.config.raptor = not brain.config.raptor
                        print(
                            f"  RAPTOR hierarchical indexing {'ON' if brain.config.raptor else 'OFF'}"
                        )
                    elif rag_opt in ("graph-rag", "graph_rag", "graphrag"):
                        brain.config.graph_rag = not brain.config.graph_rag
                        print(
                            f"  GraphRAG entity retrieval {'ON' if brain.config.graph_rag else 'OFF'}"
                        )
                    elif rag_opt == "rerank-model":
                        if not rag_val:
                            print(f"  Current reranker: {brain.config.reranker_model}")
                            print("  Usage: /rag rerank-model <HuggingFace ID or local path>")
                            print("  e.g.  /rag rerank-model BAAI/bge-reranker-base")
                            print("        /rag rerank-model ./models/bge-reranker-base")
                        else:
                            resolved = brain._resolve_model_path(rag_val)
                            if resolved != rag_val:
                                print(f"  Resolved to local path: {resolved}")
                            brain.config.reranker_model = resolved
                            brain.config.rerank = True  # auto-enable when setting a model
                            print(f"  Loading reranker '{resolved}'…")
                            try:
                                brain.reranker = OpenReranker(brain.config)
                                print(f"  Reranker → {resolved}  (rerank: ON)")
                            except Exception as e:
                                brain.config.rerank = False
                                print(f"  Failed to load reranker: {e}")
                    else:
                        print(
                            f"  Unknown option '{rag_opt}'. Try: topk, threshold, hybrid, rerank, rerank-model, hyde, multi, step-back, decompose, compress, cite, raptor, graph-rag"
                        )

            elif cmd == "/llm":
                if not arg:
                    print(
                        f"\n  temperature  · {brain.config.llm_temperature}\n"
                        f"  provider     · {brain.config.llm_provider}\n"
                        f"  model        · {brain.config.llm_model}\n"
                        f"\n  /llm temperature <0.0–2.0>   set generation temperature\n"
                    )
                else:
                    llm_parts = arg.split(maxsplit=1)
                    llm_opt = llm_parts[0].lower()
                    llm_val = llm_parts[1] if len(llm_parts) > 1 else ""
                    if llm_opt == "temperature":
                        try:
                            v = float(llm_val)
                            assert 0.0 <= v <= 2.0
                            brain.config.llm_temperature = v
                            print(f"  Temperature set to {v}")
                        except Exception:
                            print("  Usage: /llm temperature <float 0.0–2.0>")
                    else:
                        print(f"  Unknown option '{llm_opt}'. Available: temperature")

            elif cmd == "/compact":
                _do_compact(brain, chat_history)
                _save_session(session)

            elif cmd == "/project":
                from axon.projects import (
                    ProjectHasChildrenError,
                    delete_project,
                    ensure_project,
                    list_projects,
                    project_dir,
                )

                sub_parts = arg.split(maxsplit=1)
                sub = sub_parts[0].lower() if sub_parts else ""
                sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

                if not sub or sub == "list":
                    projects = list_projects()
                    active = brain._active_project

                    if not projects:
                        print("  No projects yet. Use /project new <name> to create one.")
                    else:
                        print()
                        _print_project_tree(projects, active)
                    print(f"\n  Active: {active}")
                    print("  /project new <name>           create + switch")
                    print("  /project new <parent>/<name>  create sub-project")
                    print("  /project switch <name>        switch to existing")
                    print("  /project folder               open active project folder\n")

                elif sub == "new":
                    if not sub_arg:
                        print("  Usage: /project new <name>  [description]")
                        print("         /project new research/papers  (sub-project)")
                    else:
                        name_parts = sub_arg.split(maxsplit=1)
                        proj_name = name_parts[0].lower()
                        proj_desc = name_parts[1] if len(name_parts) > 1 else ""
                        try:
                            ensure_project(proj_name, proj_desc)
                            brain.switch_project(proj_name)
                            print(f"  Created and switched to project '{proj_name}'")
                            print(f"  {project_dir(proj_name)}")
                            print("  Use /ingest to add documents to this project.\n")
                        except ValueError as e:
                            print(f"  {e}")

                elif sub == "switch":
                    if not sub_arg:
                        print("  Usage: /project switch <name>")
                    else:
                        proj_name = sub_arg.strip().lower()
                        if proj_name == "default" or project_dir(proj_name).exists():
                            try:
                                brain.switch_project(proj_name)
                                is_merged = isinstance(brain.vector_store, MultiVectorStore)
                                if is_merged:
                                    print(f"  Switched to project '{proj_name}'  [merged view]\n")
                                elif brain.vector_store.provider == "chroma":
                                    count = brain.vector_store.collection.count()
                                    print(
                                        f"  Switched to project '{proj_name}'  ({count} chunks)\n"
                                    )
                                else:
                                    print(f"  Switched to project '{proj_name}'\n")
                            except Exception as e:
                                print(f"  {e}")
                        else:
                            print(
                                f"  Project '{proj_name}' not found. Use /project list or /project new {proj_name}"
                            )

                elif sub == "delete":
                    if not sub_arg:
                        print("  Usage: /project delete <name>")
                    else:
                        proj_name = sub_arg.strip().lower()
                        try:
                            confirm = (
                                _read_input(
                                    f"  !  Delete project '{proj_name}' and ALL its data? [y/N]: "
                                )
                                .strip()
                                .lower()
                            )
                        except (EOFError, KeyboardInterrupt):
                            confirm = "n"
                        if confirm == "y":
                            try:
                                if brain._active_project == proj_name:
                                    brain.switch_project("default")
                                    print("  ↩️  Switched back to default project.")
                                delete_project(proj_name)
                                print(f"  Deleted project '{proj_name}'.\n")
                            except ProjectHasChildrenError as e:
                                print(f"  {e}")
                            except ValueError as e:
                                print(f"  {e}")
                        else:
                            print("  Cancelled.")

                elif sub == "folder":
                    active = brain._active_project
                    if active == "default":
                        print("  Default project uses config paths:")
                        print(f"    Vector store: {brain.config.vector_store_path}")
                        print(f"    BM25 index:   {brain.config.bm25_path}\n")
                    else:
                        folder = str(project_dir(active))
                        print(f"  {folder}")
                        import subprocess

                        try:
                            if os.name == "nt":
                                subprocess.Popen(["explorer", folder])
                            elif sys.platform == "darwin":
                                subprocess.Popen(["open", folder])
                            else:
                                subprocess.Popen(["xdg-open", folder])
                        except Exception:
                            pass

                else:
                    print(f"  Unknown sub-command '{sub}'. Try: list, new, switch, delete, folder")

            elif cmd == "/retry":
                if not _last_query:
                    print("  Nothing to retry — no previous query.")
                else:
                    user_input = _last_query
                    print(f"  ↩️  Retrying: {user_input}")

            elif cmd == "/context":
                _show_context(brain, chat_history, _last_sources, _last_query)

            elif cmd == "/sessions":
                _print_sessions(_list_sessions(project=brain._active_project))

            elif cmd == "/resume":
                if not arg:
                    print("  Usage: /resume <session-id>")
                else:
                    loaded = _load_session(arg, project=brain._active_project)
                    if loaded is None:
                        print(f"  Session '{arg}' not found. Use /sessions to list.")
                    else:
                        session = loaded
                        chat_history.clear()
                        chat_history.extend(session["history"])
                        turns = len(chat_history) // 2
                        print(f"  Loaded session {session['id']}  ({turns} turns)\n")

            elif cmd == "/keys":
                _env_file = Path.home() / ".axon" / ".env"
                _provider_keys = {
                    "gemini": ("GEMINI_API_KEY", "https://aistudio.google.com/app/apikey"),
                    "openai": ("OPENAI_API_KEY", "https://platform.openai.com/api-keys"),
                    "brave": ("BRAVE_API_KEY", "https://api.search.brave.com/app/keys"),
                    "ollama_cloud": ("OLLAMA_CLOUD_KEY", "https://ollama.com/settings"),
                }
                if arg.lower().startswith("set"):
                    set_parts = arg.split(maxsplit=1)
                    prov = set_parts[1].lower().strip() if len(set_parts) > 1 else ""
                    if not prov or prov not in _provider_keys:
                        print("  Usage: /keys set <provider>")
                        print(f"  Providers: {', '.join(_provider_keys)}")
                    else:
                        env_name, url = _provider_keys[prov]
                        print(f"  Get your key at: {url}")
                        try:
                            import getpass

                            new_key = getpass.getpass(f"  Enter {env_name} (hidden): ").strip()
                        except (EOFError, KeyboardInterrupt):
                            print("\n  Cancelled.")
                        else:
                            if new_key:
                                _env_file.parent.mkdir(parents=True, exist_ok=True)
                                existing = (
                                    _env_file.read_text(encoding="utf-8")
                                    if _env_file.exists()
                                    else ""
                                )
                                lines = [
                                    ln
                                    for ln in existing.splitlines()
                                    if not ln.startswith(f"{env_name}=")
                                ]
                                lines.append(f"{env_name}={new_key}")
                                _env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                                os.environ[env_name] = new_key
                                if prov == "brave":
                                    brain.config.brave_api_key = new_key
                                elif prov == "gemini":
                                    brain.config.gemini_api_key = new_key
                                elif prov == "openai":
                                    brain.config.api_key = new_key
                                elif prov == "ollama_cloud":
                                    brain.config.ollama_cloud_key = new_key
                                print(f"  {env_name} saved to {_env_file} and applied.")
                                print(f"  Switch provider: /model {prov}/<model-name>")
                            else:
                                print("  No key entered — nothing saved.")
                else:
                    print("\n  API Key Status\n  " + "─" * 50)
                    for prov, (env_name, _url) in _provider_keys.items():
                        val = os.environ.get(env_name, "")
                        if val:
                            masked = val[:4] + "****" + val[-2:] if len(val) > 6 else "****"
                            status = f"set ({masked})"
                        else:
                            status = "not set"
                        print(f"  {prov:<14} {env_name:<22} {status}")
                    if _env_file.exists():
                        print(f"\n  Keys file: {_env_file}")
                    else:
                        print("\n  No keys file yet. Use /keys set <provider> to add keys.")
                    print("  /keys set <provider>  to set a key interactively")
                    print("  /help keys            for provider URLs and usage\n")

            else:
                print(f"  Unknown command: {cmd}. Type /help for options.")

            if cmd != "/retry":
                continue

        # --- @file expansion: replace @path references with file contents ---
        query_text = _expand_at_files(user_input)
        if query_text != user_input:
            at_files = re.findall(r"@(\S+)", user_input)
            print(f"  Attached: {', '.join(at_files)}")

        # --- Regular query — use Rich Live for spinner + streaming response ---
        response_parts: list = []
        _cancelled = False
        try:
            from rich.console import Console as _RC
            from rich.live import Live as _RL
            from rich.markdown import Markdown as _RM
            from rich.text import Text as _RT

            _console = _RC()

            # ── Spinner phase (transient=True removes it cleanly when stopped) ──
            _spin_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            _spin_stop = threading.Event()
            _spin_idx = [0]

            def _spin_update(live: "_RL") -> None:
                while not _spin_stop.wait(0.1):
                    f = _spin_frames[_spin_idx[0] % len(_spin_frames)]
                    live.update(_RT.from_markup(f"[bold yellow]Axon:[/bold yellow] {f} thinking…"))
                    _spin_idx[0] += 1

            if stream:
                token_gen = brain.query_stream(query_text, chat_history=chat_history)

                if not quiet:
                    m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
                    s_v = "search:ON" if brain.config.truth_grounding else "search:off"
                    d_v = "discuss:ON" if brain.config.discussion_fallback else "discuss:off"
                    h_v = "hybrid:ON" if brain.config.hybrid_search else "hybrid:off"
                    tk = f"top-k:{brain.config.top_k}"
                    thr = f"thr:{brain.config.similarity_threshold}"
                    _snap = (m, s_v, d_v, h_v, tk, thr)

                    if _snap != _last_config_snapshot:
                        # Only show the parts that changed
                        changes = []
                        for i, val in enumerate(_snap):
                            if i >= len(_last_config_snapshot) or val != _last_config_snapshot[i]:
                                changes.append(val)
                        if changes:
                            print(f"\033[2m  {'  │  '.join(changes)}\033[0m")
                        _last_config_snapshot = _snap

                    print()
                    # Spinner until first real token arrives
                    with _RL(
                        _RT.from_markup("[bold yellow]Axon:[/bold yellow] ⠋ thinking…"),
                        console=_console,
                        transient=True,
                        refresh_per_second=10,
                    ) as _spin_live:
                        _st = threading.Thread(target=_spin_update, args=(_spin_live,), daemon=True)
                        _st.start()
                        for chunk in token_gen:
                            if isinstance(chunk, dict):
                                if chunk.get("type") == "sources":
                                    _last_sources = chunk.get("sources", [])
                                continue
                            response_parts.append(chunk)
                            break  # first token → exit spinner
                        _spin_stop.set()
                        _st.join(timeout=0.3)
                    # spinner gone (transient); cursor is at a clean line

                # Stream remaining tokens via Rich Live (plain text + cursor),
                # then swap to full Markdown on completion — no raw cursor
                # save/restore so the terminal scrollback is never corrupted.
                try:
                    _console.print("[bold yellow]Axon:[/bold yellow]")
                    _accumulated = "".join(response_parts)
                    with _RL(
                        _RT(_accumulated + " ▋"),
                        console=_console,
                        transient=False,
                        refresh_per_second=15,
                    ) as live:
                        for chunk in token_gen:
                            if isinstance(chunk, dict):
                                if chunk.get("type") == "sources":
                                    _last_sources = chunk.get("sources", [])
                                continue
                            _accumulated += chunk
                            response_parts.append(chunk)
                            live.update(_RT(_accumulated + " ▋"))
                        # All tokens received — swap plain text for Markdown
                        live.update(_RM(_accumulated))
                    print()
                except KeyboardInterrupt:
                    _cancelled = True
                    if _accumulated:
                        _console.print(_RM(_accumulated))
                    print("\n  !  Cancelled.\n")
            else:
                # Non-streaming: spinner while brain.query() blocks
                _spin_stop2 = threading.Event()
                _result: list = []
                _err: list = []

                if not quiet:
                    m = f"{brain.config.llm_provider}/{brain.config.llm_model}"
                    s_v = "search:ON" if brain.config.truth_grounding else "search:off"
                    d_v = "discuss:ON" if brain.config.discussion_fallback else "discuss:off"
                    h_v = "hybrid:ON" if brain.config.hybrid_search else "hybrid:off"
                    tk = f"top-k:{brain.config.top_k}"
                    thr = f"thr:{brain.config.similarity_threshold}"
                    _snap = (m, s_v, d_v, h_v, tk, thr)

                    if _snap != _last_config_snapshot:
                        # Only show the parts that changed
                        changes = []
                        for i, val in enumerate(_snap):
                            if i >= len(_last_config_snapshot) or val != _last_config_snapshot[i]:
                                changes.append(val)
                        if changes:
                            print(f"\033[2m  {'  │  '.join(changes)}\033[0m")
                        _last_config_snapshot = _snap

                def _run_query() -> None:
                    try:
                        _result.append(brain.query(query_text, chat_history=chat_history))
                    except Exception as exc:
                        _err.append(exc)
                    finally:
                        _spin_stop2.set()

                _qt = threading.Thread(target=_run_query, daemon=True)
                _qt.start()

                if not quiet:
                    print()
                    with _RL(
                        _RT.from_markup("[bold yellow]Axon:[/bold yellow] ⠋ thinking…"),
                        console=_console,
                        transient=True,
                        refresh_per_second=10,
                    ) as _spin_live2:
                        _st2 = threading.Thread(
                            target=_spin_update, args=(_spin_live2,), daemon=True
                        )
                        _st2.start()
                        _spin_stop2.wait()
                        _spin_stop.set()
                        _st2.join(timeout=0.3)
                else:
                    _qt.join()

                if _err:
                    raise _err[0]
                response = _result[0] if _result else ""
                print()  # blank line between You: and Axon:
                _console.print("[bold yellow]Axon:[/bold yellow]")
                _console.print(_RM(response))
                print()  # blank line after Brain response, before next You:
                response_parts = [response]

        except ImportError:
            # rich not available — plain fallback
            _spin_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            _spin_stop = threading.Event()
            _spin_idx = [0]

            def _spin_plain() -> None:
                while not _spin_stop.wait(0.1):
                    f = _spin_frames[_spin_idx[0] % len(_spin_frames)]
                    sys.stdout.write(f"\r  Axon: {f} thinking…")
                    sys.stdout.flush()
                    _spin_idx[0] += 1

            if not quiet:
                print()
                _spt = threading.Thread(target=_spin_plain, daemon=True)
                _spt.start()
            response = brain.query(query_text, chat_history=chat_history)
            if not quiet:
                _spin_stop.set()
            print(f"\n\033[1;33mAxon:\033[0m {response}\n")
            response_parts = [response]

        response = "".join(response_parts)
        if not _cancelled:
            # Append both turns so future queries have full context
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})
            _last_query = user_input
            _save_session(session)  # persist after every turn


# Backwards-compatible aliases (Deprecated)
def __getattr__(name):
    import warnings

    if name == "OpenStudioBrain":
        warnings.warn(
            "OpenStudioBrain is deprecated and will be removed in a future version. Use AxonBrain instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AxonBrain
    if name == "OpenStudioConfig":
        warnings.warn(
            "OpenStudioConfig is deprecated and will be removed in a future version. Use AxonConfig instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AxonConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
