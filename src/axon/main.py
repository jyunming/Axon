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


import concurrent.futures  # noqa: E402
import hashlib  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
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


from axon.code_graph import CodeGraphMixin  # noqa: E402,F401
from axon.code_retrieval import CodeRetrievalMixin  # noqa: E402,F401
from axon.config import _USER_CONFIG_PATH, AxonConfig  # noqa: E402,F401
from axon.embeddings import _KNOWN_DIMS, OpenEmbedding  # noqa: E402,F401
from axon.graph_rag import GraphRagMixin  # noqa: E402,F401
from axon.graph_render import GraphRenderMixin  # noqa: E402,F401
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
from axon.query_router import QueryRouterMixin  # noqa: E402,F401
from axon.rerank import OpenReranker  # noqa: E402,F401
from axon.vector_store import (  # noqa: E402,F401
    _MERGED_VIEW_WRITE_ERROR,
    MultiBM25Retriever,
    MultiVectorStore,
    OpenVectorStore,
)

# GraphRAG reduce-phase system prompt (GAP 1)


# ---------------------------------------------------------------------------


class _BloomHashStore:
    """Memory-efficient probabilistic hash store (bloom filter).
    False positive rate ~0.1% at capacity. A false positive means a chunk that
    was never ingested is treated as already-ingested (silently skipped).
    Acceptable trade-off for RAM-constrained deployments. Default: disabled.
    """

    def __init__(self, capacity: int = 500_000, fp_rate: float = 0.001):
        m_bits = math.ceil(-(capacity * math.log(fp_rate)) / (math.log(2) ** 2))
        self._k = max(1, round((m_bits / capacity) * math.log(2)))
        self._n_bytes = math.ceil(m_bits / 8)
        self._bits = bytearray(self._n_bytes)
        self._count = 0
        # Tracks only hashes added via add() this session (not loaded from disk).
        # Kept small so disk persistence remains possible without reading the full
        # bloom filter back out (bloom filters are write-only / non-enumerable).
        self._session_hashes: set[str] = set()

    def _hashes(self, item: str):
        h1 = int(hashlib.md5(item.encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(item.encode()).hexdigest(), 16)
        for i in range(self._k):
            yield (h1 + i * h2) % (self._n_bytes * 8)

    def _load_item(self, item: str) -> None:
        """Set bloom bits for *item* without recording it in session_hashes.
        Used during startup to replay persisted hashes from disk so they are
        recognised as already-ingested without inflating the session delta that
        will be written back to disk on the next save.
        """
        for bit in self._hashes(item):
            self._bits[bit >> 3] |= 1 << (bit & 7)
        self._count += 1

    def add(self, item: str) -> None:
        for bit in self._hashes(item):
            self._bits[bit >> 3] |= 1 << (bit & 7)
        self._session_hashes.add(item)
        self._count += 1

    def update(self, items) -> None:
        for item in items:
            self.add(item)

    def __contains__(self, item: str) -> bool:
        return all(self._bits[bit >> 3] >> (bit & 7) & 1 for bit in self._hashes(item))

    def __len__(self) -> int:
        return self._count

    def discard(self, item: str) -> None:
        # Bloom filter bits cannot be cleared; remove from session set so the
        # hash is not re-persisted to disk on the next save.
        self._session_hashes.discard(item)


# ---------------------------------------------------------------------------


class AxonBrain(
    GraphRagMixin,
    CodeGraphMixin,
    CodeRetrievalMixin,
    GraphRenderMixin,
    QueryRouterMixin,
):
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
        # Ensure the user namespace directories exist
        from axon.projects import ensure_user_project

        ensure_user_project(Path(self.config.projects_root))
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
        self._active_project_kind: str = "default"  # "default" | "local" | "mounted" | "scope"
        self._active_mount_descriptor: dict | None = None
        try:
            from axon.retrievers import BM25Retriever

            self.bm25 = BM25Retriever(
                storage_path=self.config.bm25_path,
                engine=getattr(self.config, "bm25_engine", "python"),
                rust_fallback_enabled=getattr(self.config, "rust_fallback_enabled", True),
            )
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
        # Eagerly initialise mixin state that would otherwise be created lazily
        # via @property getters. Lazy init in those getters has a TOCTOU race
        # under concurrent first-access (two threads both pass the hasattr
        # check, both create a new lock, one overwrites the other — leaving
        # threads holding "different" locks for the same critical section).
        # Doing it once here, single-threaded in __init__, removes the race.
        self._graph_lock_internal: threading.RLock = threading.RLock()
        self._traversal_cache_lock_internal: threading.Lock = threading.Lock()
        self._entity_token_index_internal: dict[str, set[str]] = {}
        self._pending_persist_futures_internal: list[concurrent.futures.Future] = []
        self._persist_executor_internal: concurrent.futures.ThreadPoolExecutor | None = None
        # BFS traversal cache for GraphRAG multi-hop expansion (LRU + TTL).
        # Keyed by (frozenset(matched_entities), max_hops, hop_decay); stores
        # {chunk_id: hop_score} dicts discovered by BFS.  Invalidated whenever
        # _entity_graph or _relation_graph are mutated.
        self._traversal_cache: OrderedDict = OrderedDict()
        self._traversal_cache_maxsize: int = 512
        self._traversal_cache_ttl: float = 900.0  # 15 minutes
        self._last_diagnostics: CodeRetrievalDiagnostics = CodeRetrievalDiagnostics()
        self._last_provenance: dict = {}
        # Content hash store for ingest deduplication
        self._ingested_hashes: set = self._load_hash_store()
        # Doc versions store for smart re-ingest
        self._doc_versions: dict = {}
        self._doc_versions_path = os.path.join(self.config.bm25_path, ".doc_versions.json")
        self._load_doc_versions()
        # GraphRAG entity → doc_id mapping (entity name -> list of chunk IDs)
        self._entity_graph: dict[str, list[str]] = self._load_entity_graph()
        self._rebuild_entity_token_index()
        # Persisted GraphRAG extraction cache (entities/relations keyed by chunk hash)
        self._graph_rag_cache: dict = self._load_graph_rag_extraction_cache()
        self._graph_rag_cache_dirty: bool = False
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
        # In-memory RAPTOR summary cache {cache_key: summary_text}
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
        # Sentence-Window Retrieval secondary index (Epic 1, Stories 1.1–1.2)
        from axon.sentence_window import SentenceVectorStore, SentenceWindowIndex

        self._sw_index_path: Path = Path(self.config.bm25_path)
        self._sw_index: SentenceWindowIndex = SentenceWindowIndex()
        self._sw_vs: SentenceVectorStore = SentenceVectorStore(self._sw_index_path)
        if self.config.sentence_window:
            self._sw_index.load(self._sw_index_path)
            self._sw_vs.load()
            logger.info(
                "Sentence-Window: loaded %d sentence records, %d embeddings",
                len(self._sw_index),
                len(self._sw_vs),
            )
        # Shared executor for background/parallel tasks
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        # Sealed-project cache slot. Held across switch_project calls so
        # close() can wipe it on shutdown. None whenever the active
        # project is plaintext (the common case).
        self._sealed_cache: Any = None
        # Two-phase materialisation handoff between the path-setup arm
        # of switch_project and the post-close block. Tuple is
        # ``(project_name, project_root, share_key_id_or_None)`` —
        # ``share_key_id`` is None for the owner's own sealed projects
        # and a non-empty key_id for sealed mounts (grantee path).
        self._pending_seal_mount: tuple[str, Path, str | None] | None = None
        # Wipe stale sealed caches from previous crashed sessions. Cheap
        # (just lists temp dir) and only fires when the optional
        # [sealed] extra is installed.
        try:
            from axon.security.cache import cleanup_orphans as _cleanup_sealed_orphans

            wiped = _cleanup_sealed_orphans()
            if wiped:
                logger.info(
                    "Wiped %d orphaned sealed-cache directories from a previous session",
                    wiped,
                )
        except ImportError:
            pass  # [sealed] extra not installed — nothing to clean up
        except Exception as _orphan_exc:
            logger.debug("Sealed-cache orphan cleanup raised: %s", _orphan_exc)
        self._log_startup_summary()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Explicitly release all resources (connections, file handles)."""
        if getattr(self, "_graph_rag_cache_dirty", False):
            try:
                self._save_graph_rag_extraction_cache()
            except Exception as e:
                logger.debug("Could not flush graph_rag extraction cache on close: %s", e)
        # Flush any in-flight background graph persist operations before shutting
        # down the executor so that data is not silently dropped on close.
        try:
            self._flush_pending_saves()
        except Exception as e:
            logger.debug("Could not flush pending graph saves on close: %s", e)
        # Wait briefly for in-flight tasks instead of abandoning futures.
        # Long-running tasks (>30s) will still be cancelled when the process
        # exits, but the common case (single-second persists, embedding flushes)
        # gets a chance to finish so we don't silently drop user data.
        if hasattr(self, "_executor") and self._executor:
            try:
                # cancel_futures requires Python 3.9+; we're on 3.11+.
                self._executor.shutdown(wait=True, cancel_futures=True)
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("Shared executor shutdown raised: %s", exc)
        # Shut down the dedicated graph-persist executor too. Without this it
        # leaks one thread per switch_project / brain instance.
        persist_exec = getattr(self, "_persist_executor_internal", None)
        if persist_exec is not None:
            try:
                persist_exec.shutdown(wait=True, cancel_futures=False)
            except Exception as exc:  # pragma: no cover — defensive
                logger.debug("Persist executor shutdown raised: %s", exc)
            self._persist_executor_internal = None
        # Close all unique store objects to avoid double-closure
        seen_stores: set[int] = set()
        for attr in ("vector_store", "_own_vector_store", "bm25", "_own_bm25"):
            store = getattr(self, attr, None)
            if store and hasattr(store, "close") and id(store) not in seen_stores:
                store.close()
                seen_stores.add(id(store))
        # Force GC so Windows file handles into the sealed cache are
        # released BEFORE we try to overwrite + unlink the cache files.
        # Without this, wipe() may fail on Windows with PermissionError.
        sealed_cache = getattr(self, "_sealed_cache", None)
        if sealed_cache is not None:
            import gc as _gc

            _gc.collect()
            try:
                from axon.security.mount import release_cache as _release_sealed

                _release_sealed(sealed_cache)
            except Exception as _wipe_exc:
                logger.debug("Sealed-cache wipe on close raised: %s", _wipe_exc)
            self._sealed_cache = None

    def _log_startup_summary(self) -> None:
        """Log a one-line startup summary: active project, namespace ID, scope type."""
        from axon.projects import get_project_id

        scope = "read-only merged" if getattr(self, "_read_only_scope", False) else "authoritative"
        ns_id = get_project_id(self._active_project)
        if ns_id is None:
            # Try reading from the actual project dir (for default which maps to config path)
            try:
                pdir = Path(self.config.bm25_path).parent
                meta_file = pdir / "meta.json"
                if meta_file.exists():
                    import json as _json

                    meta = _json.loads(meta_file.read_text(encoding="utf-8"))
                    ns_id = meta.get("project_id", "none")
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
    _SCOPE_RESERVED_DIRS: frozenset = frozenset(
        {"default", "projects", "mounts", "sharemount", "_default", ".shares"}
    )

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
                            str(Path(target) / "vector_store_data"),
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
                all_bm25.append(
                    _BM25(
                        storage_path=bpath,
                        engine=getattr(cfg, "bm25_engine", "python"),
                        rust_fallback_enabled=getattr(cfg, "rust_fallback_enabled", True),
                    )
                )
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
        with self._traversal_cache_lock:
            self._traversal_cache.clear()
        self._ingested_hashes = set()
        self._entity_graph = {}
        self._rebuild_entity_token_index()
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
        logger.info(
            "Switched to %s scope  |  %d store(s) merged  |  read-only",
            scope,
            len(all_vs),
        )

    def _is_mounted_share(self) -> bool:
        """Return True if the active project is a received share mount (always read-only)."""
        kind = getattr(self, "_active_project_kind", None)
        if kind is not None:
            return kind == "mounted"
        # Backward compatibility for older instances/tests that only set _mounted_share.
        return bool(getattr(self, "_mounted_share", False))

    def refresh_mount(self) -> bool:
        """Re-read the owner's version marker; reopen project handles if newer.
        Called automatically before retrieval when ``cfg.mount_refresh_mode``
        is ``"per_query"`` (see :meth:`QueryRouterMixin._maybe_refresh_mount`)
        and available as a public API for explicit refresh.
        Returns:
            ``True`` if handles were reopened (the owner had advanced),
            ``False`` if no refresh was needed or the brain is not on a mount.
        Raises:
            MountSyncPendingError: when the owner's marker has advanced but
                the underlying index files have not yet replicated to this
                grantee — typical for cloud-sync where ``version.json``
                arrives before larger index files. Callers should retry
                later or surface a transient ``X-Axon-Mount-Sync-Pending``
                signal to the user.
        """
        import time as _time

        from axon.version_marker import (
            MountSyncPendingError,
            artifacts_match,
            is_newer_than,
        )
        from axon.version_marker import (
            read as _read_marker,
        )

        if not self._is_mounted_share():
            return False
        desc = getattr(self, "_active_mount_descriptor", None)
        if not desc:
            return False
        target = Path(desc.get("target_project_dir", ""))
        if not target.exists():
            return False
        cached = getattr(self, "_mount_version_marker", None)
        current = _read_marker(target)
        if not is_newer_than(current, cached):
            return False
        # Mid-sync mismatch protection: re-hash the actual files and compare
        # to the marker's expected hashes. If they don't match yet, the
        # index files are still in flight; back off and retry.
        retry_max = max(0, int(getattr(self.config, "mount_sync_retry_max", 5)))
        backoff = float(getattr(self.config, "mount_sync_retry_backoff_s", 0.5))
        for attempt in range(retry_max + 1):
            if artifacts_match(target, current):
                break
            if attempt == retry_max:
                raise MountSyncPendingError(
                    f"Mount '{self._active_project}' has a newer version marker "
                    f"(seq={current.get('seq')}) but its index files have not "
                    f"finished replicating after {retry_max} retries. "
                    f"Try again in a few seconds."
                )
            _time.sleep(backoff * (2**attempt))
            current = _read_marker(target) or current
        # Files are coherent with the marker — reopen handles via the
        # existing switch logic. switch_project resets caches, GCs file
        # handles, and rebuilds vector_store + bm25 from the new on-disk
        # state. The marker we just read is re-cached during that call.
        logger.info(
            "Refreshing mount '%s': owner advanced to seq=%s",
            self._active_project,
            current.get("seq"),
        )
        self.switch_project(self._active_project)
        return True

    def _assert_write_allowed(self, operation: str = "write") -> None:
        """Raise PermissionError if current project is read-only (scope, mounted share, or maintenance state)."""
        from axon.access import check_write_allowed

        check_write_allowed(
            operation,
            self._active_project,
            getattr(self, "_read_only_scope", False),
            self._is_mounted_share(),
        )

    # ------------------------------------------------------------------
    # Sealed-project routing (lazy — only fires when [sealed] installed)
    # ------------------------------------------------------------------
    def _project_is_sealed(self, project_root: Path) -> bool:
        """Return True if *project_root* has a ``.security/.sealed`` marker.
        Cheap probe — does not require an unlocked store. Returns False
        when the optional ``[sealed]`` extra is not installed (in which
        case sealed projects can never have been created on this
        install anyway).
        """
        try:
            from axon.security.seal import is_project_sealed
        except ImportError:
            return False
        try:
            return is_project_sealed(project_root)
        except Exception as exc:
            logger.debug("is_project_sealed raised on %s: %s", project_root, exc)
            return False

    def _mount_sealed_project(
        self,
        name: str,
        project_root: Path,
        share_key_id: str | None = None,
    ) -> Path:
        """Decrypt *project_root* into an ephemeral cache; return its path.
        Wipes any pre-existing sealed cache before creating the new one,
        so back-to-back ``switch_project`` calls don't leak plaintext.
        Stashes the new cache on ``self._sealed_cache`` so ``close()``
        can wipe it on shutdown.
        Args:
            name: Project name (for logging).
            project_root: Directory containing the sealed files.
            share_key_id: When set, this is a grantee mount; the DEK is
                fetched from the OS keyring at ``axon.share.<key_id>``.
                When None, this is the owner's own sealed project; the
                DEK is unwrapped from ``dek.wrapped`` via the master.
        Raises:
            SecurityError: store is locked (owner path) or DEK missing
                from keyring (grantee path), sealed marker missing /
                malformed, or cache materialisation failed (capacity,
                I/O, or decryption error).
        """
        from axon.security.mount import materialize_for_read, release_cache

        # Wipe stale cache from a previously-active sealed project.
        prior = getattr(self, "_sealed_cache", None)
        if prior is not None:
            try:
                release_cache(prior)
            except Exception as exc:
                logger.debug("Wiping prior sealed cache raised: %s", exc)
            self._sealed_cache = None
        user_dir = Path(self.config.projects_root)
        if share_key_id is not None:
            # Grantee path — DEK from keyring, not from disk.
            from axon.security.share import get_grantee_dek

            grantee_dek = get_grantee_dek(share_key_id)
            cache = materialize_for_read(project_root, user_dir, dek=grantee_dek)
        else:
            cache = materialize_for_read(project_root, user_dir)
        self._sealed_cache = cache
        logger.info(
            "Sealed project '%s' mounted at %s (key_id=%s)",
            name,
            cache.path,
            share_key_id or "owner",
        )
        return Path(cache.path)

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
        # Clear any stale sealed-mount intent from a previous switch
        # that didn't reach the post-close materialisation block.
        self._pending_seal_mount = None
        if getattr(self, "_graph_rag_cache_dirty", False):
            self._save_graph_rag_extraction_cache()
        from axon.projects import (
            is_reserved_top_level_name,
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
            # Sealed mounts route through the cache: descriptor carries
            # mount_type="sealed" + share_key_id; fetch the DEK from the
            # grantee's keyring, materialise the cache after close().
            if desc.get("mount_type") == "sealed":
                share_key_id = desc.get("share_key_id", "")
                if not share_key_id:
                    raise ValueError(
                        f"Sealed mount '{name}' has no share_key_id in its descriptor; "
                        "the redeem flow may be from an incompatible older version."
                    )
                # Defer cache materialisation until AFTER close() — see
                # the local-project arm for the same two-phase reason.
                self._pending_seal_mount = (name, target, share_key_id)
                self.config.vector_store_path = str(target / "vector_store_data")
                self.config.bm25_path = str(target / "bm25_index")
            else:
                self.config.vector_store_path = str(target / "vector_store_data")
                self.config.bm25_path = str(target / "bm25_index")
            # Cache the owner's version marker so a future TTL/per-query
            # check (issue #53 follow-up) can detect when the owner has
            # re-ingested and we need to reopen handles. For now this is
            # informational — logged at INFO so operators can see what
            # snapshot of the owner's state they bound to.
            try:
                from axon.version_marker import read as _read_marker

                self._mount_version_marker = _read_marker(target)
                if self._mount_version_marker is not None:
                    logger.info(
                        "Mounted '%s' at owner version seq=%s (generated_at=%s)",
                        name,
                        self._mount_version_marker.get("seq"),
                        self._mount_version_marker.get("generated_at"),
                    )
                else:
                    self._mount_version_marker = None
                    logger.info(
                        "Mounted '%s' — owner has not yet emitted a version marker.",
                        name,
                    )
            except Exception as _vm_exc:
                logger.debug("Could not read mount version marker: %s", _vm_exc)
                self._mount_version_marker = None
        elif name == "default":
            self.config.vector_store_path = self._base_vector_store_path
            self.config.bm25_path = self._base_bm25_path
        else:
            if is_reserved_top_level_name(name):
                raise ValueError(
                    f"Project '{name}' is reserved and cannot be activated as a local project."
                )
            root = project_dir(name)
            # A sealed project has every content file as AES-GCM
            # ciphertext, so meta.json on disk is unreadable plaintext.
            # Skip the meta.json existence check for sealed projects —
            # the sealed marker (.security/.sealed) is the proof the
            # project exists.
            sealed = self._project_is_sealed(root)
            if not root.exists() or (not sealed and not (root / "meta.json").exists()):
                raise ValueError(
                    f"Project '{name}' does not exist. Create it first with /project new {name}"
                )
            if sealed:
                # Defer cache materialisation until AFTER self.close()
                # below — close() wipes self._sealed_cache, so creating
                # the cache before close() would wipe it before any
                # backend can read it. Stash the project root on the
                # instance so the post-close block can pick it up.
                # Owner path: share_key_id is None — DEK comes from
                # the master via get_project_dek.
                self._pending_seal_mount = (name, root, None)
                # Sentinel paths — overwritten right after close().
                self.config.vector_store_path = str(root / "vector_store_data")
                self.config.bm25_path = str(root / "bm25_index")
            else:
                self._pending_seal_mount = None
                self.config.vector_store_path = project_vector_path(name)
                self.config.bm25_path = project_bm25_path(name)
        # Close existing stores before replacing them; force GC to release Windows
        # file handles on ChromaDB's SQLite database before opening the new path.
        self.close()
        self.vector_store = None  # type: ignore[assignment]
        self._own_vector_store = None  # type: ignore[assignment]
        self.bm25 = None
        self._own_bm25 = None
        import gc as _gc

        _gc.collect()
        # Sealed-project routing — close() wiped any prior _sealed_cache,
        # so it's safe to materialise the new one now. The sentinel
        # paths set in the local-project / sealed-mount arm above are
        # overwritten with the cache directory.
        pending = getattr(self, "_pending_seal_mount", None)
        if pending is not None:
            seal_name, seal_root, seal_share_key_id = pending
            self._pending_seal_mount = None
            cache_path = self._mount_sealed_project(seal_name, seal_root, seal_share_key_id)
            self.config.vector_store_path = str(cache_path / "vector_store_data")
            self.config.bm25_path = str(cache_path / "bm25_index")
        # Recreate the executor — close() shuts it down and it cannot be reused
        from concurrent.futures import ThreadPoolExecutor

        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        # Own store: always the project's own data (used for writes / dedup / GraphRAG)
        own_vs = OpenVectorStore(self.config)
        try:
            from axon.retrievers import BM25Retriever

            own_bm25 = BM25Retriever(
                storage_path=self.config.bm25_path,
                engine=getattr(self.config, "bm25_engine", "python"),
                rust_fallback_enabled=getattr(self.config, "rust_fallback_enabled", True),
            )
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
                    all_bm25.append(
                        _BM25(
                            storage_path=cfg.bm25_path,
                            engine=getattr(cfg, "bm25_engine", "python"),
                            rust_fallback_enabled=getattr(cfg, "rust_fallback_enabled", True),
                        )
                    )
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
        with self._traversal_cache_lock:
            self._traversal_cache.clear()
        self._ingested_hashes = self._load_hash_store()
        self._graph_rag_cache = self._load_graph_rag_extraction_cache()
        self._graph_rag_cache_dirty = False
        self._code_graph = self._load_code_graph()
        self._entity_graph = self._load_entity_graph()
        self._rebuild_entity_token_index()
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

            from axon.rust_bridge import get_rust_bridge

            bridge = get_rust_bridge()
            with self._graph_lock:
                for desc in descendants:
                    desc_bm25_path = project_bm25_path(desc)
                    desc_base = pathlib.Path(desc_bm25_path)
                    # --- entity graph ---
                    desc_graph_path = desc_base / ".entity_graph.json"
                    desc_mp_path = desc_base / ".entity_graph.msgpack"
                    raw = None
                    if desc_mp_path.exists() and bridge.can_entity_graph_codec():
                        try:
                            raw = bridge.decode_entity_graph(desc_mp_path.read_bytes())
                        except Exception:
                            raw = None
                    if raw is None and desc_graph_path.exists():
                        try:
                            import json as _json

                            raw = _json.loads(desc_graph_path.read_text(encoding="utf-8"))
                        except Exception:
                            raw = None
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
                                    "frequency": len([d for d in doc_ids if isinstance(d, str)]),
                                    "degree": node.get("degree", 0),
                                }
                                self._token_index_add(entity)
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
                    # --- relation graph ---
                    desc_rel_path = desc_base / ".relation_graph.json"
                    desc_rel_mp_path = desc_base / ".relation_graph.msgpack"
                    raw_rel = None
                    if desc_rel_mp_path.exists() and bridge.can_relation_graph_codec():
                        try:
                            raw_rel = bridge.decode_relation_graph(desc_rel_mp_path.read_bytes())
                        except Exception:
                            raw_rel = None
                    if raw_rel is None and desc_rel_path.exists():
                        try:
                            import json as _json

                            raw_rel = _json.loads(desc_rel_path.read_text(encoding="utf-8"))
                        except Exception:
                            raw_rel = None
                    if isinstance(raw_rel, dict):
                        for src, entries in raw_rel.items():
                            if isinstance(src, str) and isinstance(entries, list):
                                if src not in self._relation_graph:
                                    self._relation_graph[src] = []
                                existing_keys = {
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
                                        if key not in existing_keys:
                                            self._relation_graph[src].append(entry)
                                            existing_keys.add(key)
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
            # Mounts can be revoked; persist "default" so startup never reopens a broken mount
            set_active_project("default")
        elif name == "default":
            self._active_project_kind = "default"
            self._active_mount_descriptor = None
            set_active_project("default")
        elif name.startswith("@"):
            self._active_project_kind = "scope"
            self._active_mount_descriptor = None
            # Scopes are transient; revert on-disk pointer to "default"
            set_active_project("default")
        else:
            self._active_project_kind = "local"
            self._active_mount_descriptor = None
            set_active_project(name)
        logger.info(f"Switched to project '{name}'")
        # Bump epoch on the old project to fence any stale in-flight writers.
        if _prev_project != "default" and not _prev_project.startswith("@"):
            from axon.runtime import get_registry as _get_registry

            _get_registry().bump_epoch(_prev_project)

    def _load_hash_store(self) -> set:
        """Load persisted content hashes for ingest deduplication."""
        import pathlib

        from axon.rust_bridge import get_rust_bridge

        path = pathlib.Path(self.config.bm25_path) / ".content_hashes"
        bin_path = pathlib.Path(self.config.bm25_path) / ".content_hashes.bin"
        bridge = get_rust_bridge()
        if bin_path.exists() and bridge.can_hash_store_binary():
            result = bridge.load_hash_store_binary(str(bin_path))
            if result is not None:
                return result
            # Corrupt binary — delete so it gets recreated cleanly on next save
            logger.warning(
                "Corrupt hash store binary deleted: %s (will rebuild on next ingest)", bin_path
            )
            bin_path.unlink(missing_ok=True)
        if path.exists():
            return set(path.read_text(encoding="utf-8").splitlines())
        return set()

    def _save_hash_store(self) -> None:
        """Persist content hashes to disk."""
        import pathlib

        from axon.rust_bridge import get_rust_bridge

        path = pathlib.Path(self.config.bm25_path) / ".content_hashes"
        bin_path = pathlib.Path(self.config.bm25_path) / ".content_hashes.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        bridge = get_rust_bridge()
        if bridge.can_hash_store_binary():
            ok = bridge.save_hash_store_binary(str(bin_path), self._ingested_hashes)
            if ok:
                # Remove old text file to reclaim disk space
                if path.exists():
                    try:
                        path.unlink()
                    except OSError:
                        pass
                return
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
        if getattr(self, "_graph_rag_cache_dirty", False):
            self._save_graph_rag_extraction_cache()
        self.finalize_graph()

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
            # return cached summary when content is unchanged
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
        # Recursive summarization up to raptor_max_levels
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
        # Deduplicate by ID, keeping highest-scored occurrence
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

    def _index_sentence_windows(self, documents: list[dict]) -> None:
        """Segment eligible chunks into sentences and add to the sentence index.
        Called at the end of :meth:`ingest` when ``config.sentence_window`` is
        enabled.  Only non-code, non-RAPTOR-summary leaf chunks are eligible.
        Overhead (embedding time in ms) is logged at INFO level.
        """
        import time as _sw_time

        from axon.sentence_window import is_eligible, segment_chunk

        eligible = [d for d in documents if is_eligible(d)]
        if not eligible:
            return
        all_records = []
        for chunk in eligible:
            recs = segment_chunk(chunk)
            if recs:
                self._sw_index.add_records(recs)
                all_records.extend(recs)
        if not all_records:
            return
        logger.info(
            "   Sentence-Window: indexing %d sentences from %d eligible chunks",
            len(all_records),
            len(eligible),
        )
        # Batch-embed all sentence texts
        t_sw = _sw_time.time()
        texts = [r.text for r in all_records]
        batch_size = 64
        all_embeddings: list = []
        for i in range(0, len(texts), batch_size):
            all_embeddings.extend(self.embedding.embed(texts[i : i + batch_size]))
        sw_ms = (_sw_time.time() - t_sw) * 1000
        ids = [r.sentence_id for r in all_records]
        metadatas = [
            {
                "chunk_id": r.chunk_id,
                "source": r.source,
                "sentence_idx": r.sentence_idx,
            }
            for r in all_records
        ]
        self._sw_vs.add(ids, all_embeddings, metadatas)
        self._sw_index.save(self._sw_index_path)
        self._sw_vs.save()
        logger.info(
            "   Sentence-Window: %d sentences indexed (embedding %.0f ms)",
            len(all_records),
            sw_ms,
        )

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

    def _maybe_rust_preprocess_documents(self, documents: list[dict]) -> list[dict]:
        """Optionally preprocess ingest documents via Rust, with Python fallback."""
        if getattr(self.config, "ingest_engine", "python") != "rust":
            return documents
        from axon.rust_bridge import get_rust_bridge

        bridge = get_rust_bridge()
        if not bridge.can_ingest_preprocess():
            if getattr(self.config, "rust_fallback_enabled", True):
                return documents
            raise RuntimeError(
                "ingest_engine is set to 'rust' but Rust ingest preprocessing is unavailable."
            )
        batch_size = int(getattr(self.config, "rust_batch_size", 512))
        processed = bridge.preprocess_documents(documents, batch_size=batch_size)
        if processed is not None:
            return processed
        if getattr(self.config, "rust_fallback_enabled", True):
            return documents
        raise RuntimeError("Rust ingest preprocessing failed and fallback is disabled.")

    def ingest(
        self,
        documents: list[dict[str, Any]],
        progress_callback: Any | None = None,
    ) -> None:
        """Chunk, deduplicate, embed, and store *documents* in the knowledge base.
        Each document must be a dict with keys ``id`` (str), ``text`` (str), and
        optionally ``metadata`` (dict).  Chunking strategy and deduplication are
        governed by the active :class:`AxonConfig`.  When ``raptor=True``,
        summary nodes are generated and indexed alongside leaf chunks.  When
        ``graph_rag=True``, entities are extracted and added to the entity graph.
        Args:
            documents: List of document dicts with 'id', 'text', optional 'metadata'.
            progress_callback: Optional callable(phase: str, **kwargs) invoked at each
                pipeline phase transition.  Phases: loading, chunking, raptor, graph_build,
                embedding, code_graph, finalizing.
        """

        def _progress(phase: str, **kwargs: Any) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(phase, **kwargs)
                except Exception:
                    pass

        if not documents:
            return
        # Block ingest on read-only scopes and mounted shares
        self._assert_write_allowed("ingest")
        # Acquire a write lease so drain-mode can track in-flight writes.
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
        _progress("chunking", files_total=len(documents))
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
        # Per-source chunk budget enforcement
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
        # Namespace chunk IDs by project to prevent cross-project collisions in
        # MultiVectorStore / MultiBM25Retriever dedup. The default project is left
        # unchanged so existing single-project deployments are unaffected.
        if self._active_project and self._active_project != "default":
            from axon.projects import get_project_id

            _ns = get_project_id(self._active_project) or self._active_project
            _ns_prefix = f"{_ns}::"
            for _doc in documents:
                if not _doc["id"].startswith(_ns_prefix):
                    _doc["id"] = _ns_prefix + _doc["id"]
                _meta = _doc.get("metadata", {})
                if "source_id" in _meta and not _meta["source_id"].startswith(_ns_prefix):
                    _meta["source_id"] = _ns_prefix + _meta["source_id"]
        documents = self._maybe_rust_preprocess_documents(documents)
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
        _progress("raptor", chunks_total=len(documents))
        if self.config.raptor:
            # Source-size guard — skip RAPTOR for sources whose estimated text size is below threshold
            _raptor_min_mb = self.config.raptor_min_source_size_mb
            if _raptor_min_mb > 0.0:
                from collections import defaultdict as _dfl

                _size_by_source: dict = _dfl(int)
                for _d in documents:
                    _src = _d.get("metadata", {}).get("source", _d["id"])
                    _size_by_source[_src] += len(_d.get("text", ""))
                _max_bytes = int(_raptor_min_mb * 1024 * 1024)
                _skipped_sources = {src for src, sz in _size_by_source.items() if sz < _max_bytes}
                if _skipped_sources:
                    logger.info(
                        "   RAPTOR: skipping %d small source(s) < %.1f MB",
                        len(_skipped_sources),
                        _raptor_min_mb,
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
                    _r_ok, _ = self._SOURCE_POLICY.get(_dtype, self._SOURCE_POLICY_DEFAULT)
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
        # Commit chunks to stores BEFORE entity extraction so /collection shows progress
        # immediately. GraphRAG entity extraction is a pure second pass that only reads
        # chunk text — it does not need to write back to the vector store.
        # Abort if a project switch happened mid-ingest (stale epoch).
        if _ingest_lease.is_stale():
            logger.warning(
                "Ingest abandoned for '%s': project was switched mid-ingest "
                "(epoch mismatch). Data was NOT committed to prevent "
                "cross-project contamination.",
                _ingest_lease._project,
            )
            return
        n_chunks = len(documents)
        if self._own_bm25:
            self._own_bm25.add_documents(documents, save_deferred=_defer_saves)
        ids = [d["id"] for d in documents]
        texts = [d["text"] for d in documents]
        metadatas = [d.get("metadata", {}) for d in documents]
        logger.info("   Generating embeddings...")
        _progress("embedding", chunks_total=len(texts), chunks_embedded=0)
        batch_size = 32
        all_embeddings = []
        t_embed = time.time()
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            all_embeddings.extend(self.embedding.embed(texts[i : i + batch_size]))
            _progress(
                "embedding",
                chunks_total=len(texts),
                chunks_embedded=min(i + batch_size, len(texts)),
            )
        embed_ms = (time.time() - t_embed) * 1000
        t_store = time.time()
        self._own_vector_store.add(ids, texts, all_embeddings, metadatas)
        store_ms = (time.time() - t_store) * 1000
        # Persist dedup hashes only after a successful vector store write so that a
        # failed write does not permanently mark documents as "seen".
        if self.config.dedup_on_ingest and new_hashes:
            self._ingested_hashes.update(new_hashes)
            self._save_hash_store()
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
        # ── Sentence-Window secondary index (Epic 1, Story 1.2) ─────────
        if self.config.sentence_window:
            self._index_sentence_windows(documents)
        # ── Code graph (Phase 2 + Phase 3) ──────────────────────────────
        _progress("code_graph")
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
        # GraphRAG: extract entities from new chunks (second pass — after vector store commit)
        _progress("graph_build", chunks_total=len(documents))
        if self.config.graph_rag:
            updated = False
            # Only extract entities from actual document chunks (optionally include RAPTOR level-1)
            _include_raptor = getattr(self.config, "graph_rag_include_raptor_summaries", False)
            # Skip GraphRAG entity extraction for large sources when raptor=True.
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
                    _, _g_ok = self._SOURCE_POLICY.get(_dtype, self._SOURCE_POLICY_DEFAULT)
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
            # Skip chunks already present in the entity graph (cross-restart dedup)
            _already_extracted = self._build_extracted_chunk_ids()
            if _already_extracted:
                _before = len(chunks_to_process)
                chunks_to_process = [
                    c for c in chunks_to_process if c["id"] not in _already_extracted
                ]
                _skipped = _before - len(chunks_to_process)
                if _skipped:
                    logger.info("   GraphRAG: skipping %d already-extracted chunk(s).", _skipped)
            if not chunks_to_process:
                logger.info("   GraphRAG: all chunks already extracted — nothing to do.")
            else:
                logger.info(
                    f"   GraphRAG: Extracting entities from {len(chunks_to_process)} chunks..."
                )
            _relations_enabled = bool(self.config.graph_rag_relations)
            _min_ent = getattr(self.config, "graph_rag_min_entities_for_relations", 3)
            _rel_budget = getattr(self.config, "graph_rag_relation_budget", 0)
            (
                results,
                rel_results,
                _rel_chunks,
                _relations_pipelined,
            ) = self._extract_graph_llm_batches(
                chunks_to_process,
                relations_enabled=_relations_enabled,
                min_entities_for_relations=_min_ent,
                relation_budget=_rel_budget,
            )
            # Track entity keys extracted this run for embedding (Item 5)
            from axon.rust_bridge import get_rust_bridge

            _rust_bridge = get_rust_bridge()
            with self._graph_lock:
                entities_extracted_this_run: list = []
                total_entities = 0
                _touched_entity_keys: set[str] = set()
                # Build a lookup from doc_id to doc for metadata writing (Item 7)
                doc_by_id = {doc["id"]: doc for doc in chunks_to_process}
                _entity_graph_changed = False
                _use_rust_entity_merge = (
                    bool(results)
                    and bool(getattr(self.config, "graph_rag_rust_merge_entities", False))
                    and _rust_bridge.can_merge_entities_into_graph()
                )
                for doc_id, entities in results:
                    total_entities += len(entities)
                    for ent in entities:
                        if not isinstance(ent, dict) or not ent.get("name"):
                            continue
                        entities_extracted_this_run.append(ent)
                        key = ent["name"].lower().strip()
                        if not key:
                            continue
                        _touched_entity_keys.add(key)
                        existing = self._entity_graph.get(key)
                        if existing is None:
                            _entity_graph_changed = True
                        elif isinstance(existing, dict):
                            chunk_ids = existing.get("chunk_ids", [])
                            if doc_id not in chunk_ids:
                                _entity_graph_changed = True
                        else:
                            _use_rust_entity_merge = False
                            if doc_id not in existing:
                                _entity_graph_changed = True
                _merged_entities_in_rust = False
                if _use_rust_entity_merge:
                    _merged_entities_in_rust = (
                        _rust_bridge.merge_entities_into_graph(self._entity_graph, results)
                        is not None
                    )
                if _merged_entities_in_rust and _entity_graph_changed:
                    updated = True
                    self._community_graph_dirty = True
                for doc_id, entities in results:
                    for (
                        ent
                    ) in entities:  # ent is now {"name": ..., "type": ..., "description": ...}
                        key = ent["name"].lower().strip() if isinstance(ent, dict) else ent.lower()
                        if not key:
                            continue
                        if not _merged_entities_in_rust:
                            if key not in self._entity_graph:
                                desc = ent.get("description", "") if isinstance(ent, dict) else ""
                                ent_type = (
                                    ent.get("type", "UNKNOWN")
                                    if isinstance(ent, dict)
                                    else "UNKNOWN"
                                )
                                self._entity_graph[key] = {
                                    "description": desc,
                                    "type": ent_type,
                                    "chunk_ids": [],
                                    "frequency": 0,
                                    "degree": 0,
                                }
                                self._token_index_add(key)
                            elif isinstance(self._entity_graph[key], dict):
                                # Update type if not yet set
                                if (
                                    not self._entity_graph[key].get("type")
                                    or self._entity_graph[key].get("type") == "UNKNOWN"
                                ):
                                    new_type = (
                                        ent.get("type", "UNKNOWN")
                                        if isinstance(ent, dict)
                                        else "UNKNOWN"
                                    )
                                    if new_type and new_type != "UNKNOWN":
                                        self._entity_graph[key]["type"] = new_type
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
                        if isinstance(self._entity_graph.get(key), dict):
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
                    # Item 7: Write entity IDs back into chunk metadata for text-unit linkage
                    doc = doc_by_id.get(doc_id)
                    if doc is not None and entities and doc.get("metadata") is not None:
                        doc["metadata"]["entity_ids"] = [
                            e["name"].lower()
                            for e in entities
                            if isinstance(e, dict) and e.get("name")
                        ]
                    # GAP 9: Update text_unit_entity_map
                    self._text_unit_entity_map[doc_id] = [
                        e["name"] for e in entities if isinstance(e, dict) and e.get("name")
                    ]
            # Item 2: Update frequency only for entities touched in this ingest run
            # (avoids O(|V|) scan of the full entity graph on every ingest batch)
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
            if _relations_enabled:
                _entity_count_by_doc = {doc_id: len(ents) for doc_id, ents in results}
                _rel_candidate_count = sum(
                    1
                    for doc in chunks_to_process
                    if _entity_count_by_doc.get(doc["id"], 0) >= _min_ent
                )
                if _relations_pipelined:
                    logger.info(
                        f"   GraphRAG: Pipelined relation extraction for {len(_rel_chunks)} chunks "
                        f"(skipped {len(chunks_to_process) - len(_rel_chunks)} below "
                        f"{_min_ent}-entity threshold)..."
                    )
                elif _rel_budget > 0 and _rel_candidate_count > _rel_budget:
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
                rg_updated = False
                # Rust fast-path for relation graph merge
                if rel_results and _rust_bridge.can_relation_merge():
                    _added = _rust_bridge.merge_relations_into_graph(
                        self._relation_graph, rel_results
                    )
                    if _added > 0:
                        rg_updated = True
                        self._community_graph_dirty = True
                    # Still run Python loop for _relation_description_buffer (side-effect only)
                    for _doc_id, triples in rel_results:
                        for triple in triples:
                            if not isinstance(triple, dict):
                                continue
                            description = triple.get("description", "")
                            if not description:
                                continue
                            src_lower = triple.get("subject", "").lower().strip()
                            tgt_lower = triple.get("object", "").lower().strip()
                            if src_lower:
                                pair = (src_lower, tgt_lower)
                                if pair not in self._relation_description_buffer:
                                    self._relation_description_buffer[pair] = []
                                self._relation_description_buffer[pair].append(description)
                else:
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
                                    if e.get("target") == rel_tgt
                                    and e.get("relation") == rel_relation
                                ),
                                None,
                            )
                            if existing_entry:
                                # Accumulate strength-based weight (sum of LM-derived strengths)
                                existing_entry["weight"] = existing_entry.get(
                                    "weight", 1
                                ) + entry.get("strength", 1)
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
                # Normalize relation targets into entity graph so traversal never KeyErrors
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
                                self._token_index_add(_tgt)
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
            if getattr(self, "_graph_rag_cache_dirty", False) and not _defer_saves:
                self._save_graph_rag_extraction_cache()
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
                with self._graph_lock:
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
        # Flush deferred sidecar writes (e.g., TurboQuantDB doc index) at ingest end.
        try:
            if hasattr(self._own_vector_store, "flush_pending_writes"):
                self._own_vector_store.flush_pending_writes()
        except Exception:
            pass
        _progress("finalizing", chunks_total=n_chunks)
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
        # Ingest diagnostics — source IDs and collision check
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
        # Bump the cross-machine version marker LAST. Grantees read this
        # file on mount switch (and, in a future iteration, on a TTL
        # background poll) to detect when the owner has re-indexed and
        # in-memory handles need to be reopened. Atomic write — if any
        # part of this fails, the previous marker stays intact.
        try:
            from axon.version_marker import bump as _bump_marker

            _project_dir = Path(self.config.bm25_path).parent
            _bump_marker(_project_dir)
        except Exception as _vm_exc:
            logger.debug("version_marker bump failed (non-fatal): %s", _vm_exc)
        # Explicitly release lease (fallback: _WriteLease.__del__ handles it)
        _ingest_lease.close()


# ---------------------------------------------------------------------------

# Phase 3 re-exports — backward-compat, existing callers need no changes

# ---------------------------------------------------------------------------

from axon.cli import _print_project_tree, _write_python_discovery, main  # noqa: E402,F401

# ---------------------------------------------------------------------------
# Phase 4 re-exports — mixin classes available from axon.main for compat
# ---------------------------------------------------------------------------
from axon.code_graph import CodeGraphMixin  # noqa: E402,F401,F811
from axon.code_retrieval import (  # noqa: E402,F401,F811
    CodeRetrievalDiagnostics,
    CodeRetrievalMixin,
    CodeRetrievalTrace,
    _build_code_bm25_queries,
    _classify_retrieval_failure,
    _extract_code_query_tokens,
    _looks_like_code_query,
)
from axon.graph_rag import (  # noqa: E402,F401,F811
    _GRAPHRAG_NO_DATA_ANSWER,
    _GRAPHRAG_REDUCE_SYSTEM_PROMPT,
    GraphRagMixin,
)
from axon.graph_render import GraphRenderMixin  # noqa: E402,F401,F811
from axon.query_router import _ROUTE_PROFILES, QueryRouterMixin  # noqa: E402,F401,F811
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
