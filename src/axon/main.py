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

import asyncio
import logging
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv

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
_USER_CONFIG_PATH = Path.home() / ".config" / "axon" / "config.yaml"


@dataclass
class AxonConfig:
    """Configuration for Axon."""

    # Embedding
    embedding_provider: Literal[
        "sentence_transformers", "ollama", "fastembed", "openai"
    ] = "sentence_transformers"
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_base_url: str = "http://localhost:11434"

    # LLM
    llm_provider: Literal["ollama", "gemini", "ollama_cloud", "openai", "vllm"] = "ollama"
    llm_model: str = "gemma"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048
    api_key: str = ""
    gemini_api_key: str = ""
    ollama_cloud_key: str = ""
    ollama_cloud_url: str = ""
    vllm_base_url: str = "http://localhost:8000/v1"

    def __post_init__(self) -> None:
        """Populate API-related fields from environment variables when not set."""
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

        # 2. Path resolution: default to home-based directories (~/.axon)
        # to avoid filesystem permission/locking issues in WSL/Linux.
        home = Path.home() / ".axon"

        # Projects Root
        if not self.projects_root:
            self.projects_root = os.getenv("AXON_PROJECTS_ROOT", str(home / "projects"))

        # Vector Store Path
        if self.vector_store_path == "./chroma_data":
            env_vsp = os.getenv("CHROMA_DATA_PATH")
            if env_vsp:
                self.vector_store_path = env_vsp
            else:
                self.vector_store_path = str(home / "data" / "chroma")

        # BM25 Path
        if self.bm25_path == "./bm25_index":
            env_bm25 = os.getenv("BM25_INDEX_PATH")
            if env_bm25:
                self.bm25_path = env_bm25
            else:
                self.bm25_path = str(home / "data" / "bm25")

        # Always expand user paths for safety
        self.projects_root = os.path.expanduser(self.projects_root)
        self.vector_store_path = os.path.expanduser(self.vector_store_path)
        self.bm25_path = os.path.expanduser(self.bm25_path)

    # Projects
    # Root directory for all named projects. Defaults to ~/.axon/projects.
    # Override via config.yaml (projects_root: /path/to/dir) or the
    # AXON_PROJECTS_ROOT environment variable (env var wins over config.yaml).
    projects_root: str = ""

    # Vector Store
    vector_store: Literal["chroma", "qdrant", "lancedb"] = "chroma"
    vector_store_path: str = "./chroma_data"

    # BM25 Settings
    bm25_path: str = "./bm25_index"

    # RAG Settings
    top_k: int = 10
    similarity_threshold: float = 0.3
    hybrid_search: bool = True

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Re-ranking
    rerank: bool = False
    reranker_provider: Literal["cross-encoder", "llm"] = "cross-encoder"
    # Default: fast ms-marco model (small, already commonly cached).
    # Upgrade to "BAAI/bge-reranker-v2-m3" for SOTA multilingual accuracy.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Parent-Document / Small-to-Big Retrieval
    # Set parent_chunk_size > chunk_size to enable. Child chunks (chunk_size)
    # are indexed for precise retrieval; parent chunks are returned as context.
    # 0 = disabled (use retrieval chunks directly).
    parent_chunk_size: int = 0

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

    # Inline Source Citations
    # When enabled, the LLM is instructed to cite [Doc N] inline in its answer
    # matching the document labels injected into the context block.
    cite_sources: bool = False

    # RAPTOR Hierarchical Indexing
    # During ingest, groups every raptor_chunk_group_size consecutive chunks per source
    # and generates a summarisation node that is indexed alongside the leaf chunks.
    # At retrieval time these summary nodes surface high-level context for multi-hop
    # questions that no single leaf chunk can answer.
    raptor: bool = False
    raptor_chunk_group_size: int = 5

    # GraphRAG Entity-Centric Retrieval
    # During ingest, named entities are extracted from each chunk via the LLM and
    # stored in an entity→doc_id map.  At retrieval time, entities found in the
    # query are used to expand the result set with graph-connected documents.
    graph_rag: bool = False

    # LLM request timeout in seconds (applied where the provider client supports it)
    llm_timeout: int = 60

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
  path: ./chroma_data

bm25:
  path: ./bm25_index

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

        return cls(**filtered_dict)


class OpenReranker:
    """
    Open-source reranking using Cross-Encoders.
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


class OpenEmbedding:
    """
    Open-source embedding provider.
    """

    def __init__(self, config: AxonConfig):
        self.config = config
        self.provider = config.embedding_provider
        self.model = None
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
            self.dimension = 768  # nomic-embed-text dimension

        elif self.provider == "fastembed":
            from fastembed import TextEmbedding

            logger.info(f"Loading FastEmbed: {self.config.embedding_model}")
            self.model = TextEmbedding(model_name=self.config.embedding_model)
            if "bge-m3" in self.config.embedding_model.lower():
                self.dimension = 1024
            else:
                self.dimension = 384

        elif self.provider == "openai":
            from openai import OpenAI

            logger.info(f"Using OpenAI API Embedding: {self.config.embedding_model}")
            kwargs = (
                {"api_key": self.config.api_key} if self.config.api_key else {"api_key": "sk-dummy"}
            )
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


class OpenLLM:
    """
    Open-source LLM provider.
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


class OpenVectorStore:
    """
    Open-source vector store interface.
    """

    def __init__(self, config: AxonConfig):
        self.config = config
        self.provider = config.vector_store
        self.client = None
        self.collection = None
        self._init_store()

    def _init_store(self):
        if self.provider == "chroma":
            import chromadb

            logger.info(f"Initializing ChromaDB: {self.config.vector_store_path}")
            self.client = chromadb.PersistentClient(path=self.config.vector_store_path)
            self.collection = self.client.get_or_create_collection(
                name="axon", metadata={"hnsw:space": "cosine"}
            )
        elif self.provider == "qdrant":
            from qdrant_client import QdrantClient

            logger.info(f"Initializing Qdrant: {self.config.vector_store_path}")
            self.client = QdrantClient(path=self.config.vector_store_path)
        elif self.provider == "lancedb":
            import lancedb

            logger.info(f"Initializing LanceDB: {self.config.vector_store_path}")
            self.client = lancedb.connect(self.config.vector_store_path)
            try:
                self.collection = self.client.open_table("axon")
            except Exception:
                self.collection = None  # created lazily on first add()

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] = None,
    ):
        if self.provider == "chroma":
            self.collection.add(
                ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
            )
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

    def add_documents(self, *args, **kwargs):
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)

    def delete_documents(self, doc_ids: list[str]) -> None:
        raise RuntimeError(_MERGED_VIEW_WRITE_ERROR)

    # Alias used by ingest code
    batch_add_documents = add_documents


class AxonBrain:
    """
    Main interface for Axon.
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

        # GraphRAG entity → doc_id mapping (entity name -> list of chunk IDs)
        self._entity_graph: dict[str, list[str]] = self._load_entity_graph()

        logger.info("Axon ready!")

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

    def _load_entity_graph(self) -> dict[str, list[str]]:
        """Load persisted entity→doc_id graph from disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_graph.json"
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    return {}
                cleaned: dict[str, list[str]] = {}
                for key, value in raw.items():
                    if not isinstance(key, str) or not isinstance(value, list):
                        continue
                    cleaned[key] = [v for v in value if isinstance(v, str)]
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

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text using the LLM.

        Returns a list of entity strings (people, organisations, places, concepts).
        Returns an empty list on failure.
        """
        prompt = (
            "Extract the key named entities (people, organisations, places, products, "
            "technical concepts) from the following text. "
            "Output one entity per line, no bullets or numbering. "
            "If there are no entities, output nothing.\n\n"
            + text[:1500]  # cap to avoid huge prompts
        )
        try:
            raw = self.llm.complete(
                prompt, system_prompt="You are a named entity extraction specialist."
            )
            return [e.strip() for e in raw.splitlines() if e.strip()][:20]
        except Exception:
            return []

    def _generate_raptor_summaries(self, documents: list[dict]) -> list[dict]:
        """Generate RAPTOR summary nodes for a list of already-split chunks.

        Groups consecutive chunks by source in windows of raptor_chunk_group_size,
        then calls the LLM to produce a summary passage.  Each summary is returned
        as a new synthetic document (raptor_level=1 in metadata) that will be
        embedded and stored alongside the leaf chunks.
        """
        from itertools import groupby

        n = self.config.raptor_chunk_group_size
        if n < 1:
            logger.warning("raptor_chunk_group_size must be >= 1, skipping RAPTOR")
            return []
        summaries = []

        # Group chunks by their source file so we only summarise within one document
        def _source(doc):
            return doc.get("metadata", {}).get("source", doc["id"])

        sorted_docs = sorted(documents, key=_source)
        for source, group in groupby(sorted_docs, key=_source):
            chunks = list(group)
            for i in range(0, len(chunks), n):
                window = chunks[i : i + n]
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
                        continue
                    # Build a deterministic ID from source + window position + content
                    # so that re-ingesting updated content produces a new ID and
                    # does not silently collide with a stale RAPTOR node in the store.
                    import hashlib

                    content_sig = hashlib.md5(
                        summary_text.strip().encode("utf-8", errors="replace")
                    ).hexdigest()[:8]
                    uid = hashlib.md5(f"{source}|raptor|{i}|{content_sig}".encode()).hexdigest()[
                        :12
                    ]
                    summaries.append(
                        {
                            "id": f"raptor_{uid}",
                            "text": summary_text.strip(),
                            "metadata": {
                                "source": source,
                                "raptor_level": 1,
                                "window_start": i,
                                "window_end": i + len(window) - 1,
                            },
                        }
                    )
                except Exception as e:
                    logger.debug(f"RAPTOR summary failed for {source}[{i}]: {e}")

        if summaries:
            logger.info(f"   RAPTOR: generated {len(summaries)} summary node(s)")
        return summaries

    def _expand_with_entity_graph(self, query: str, results: list[dict], cfg=None) -> list[dict]:
        """Expand retrieval results using GraphRAG entity linkage.

        1. Extract entities from the query.
        2. Look up all chunk IDs linked to those entities in the entity graph.
        3. Fetch any chunks not already in results from the vector store.
        4. Append them (capped to avoid bloat), returning the expanded list.
        """
        query_entities = self._extract_entities(query)
        if not query_entities:
            return results

        active_top_k = (cfg.top_k if cfg is not None else None) or self.config.top_k

        existing_ids = {r["id"] for r in results}
        extra_ids: list[str] = []
        for entity in query_entities:
            for eid, doc_ids in self._entity_graph.items():
                if entity.lower() in eid.lower() or eid.lower() in entity.lower():
                    for did in doc_ids:
                        if did not in existing_ids and did not in extra_ids:
                            extra_ids.append(did)

        if not extra_ids:
            return results

        # Fetch the extra chunks from the vector store (capped to the active top_k)
        try:
            extra_results = self.vector_store.get_by_ids(extra_ids[:active_top_k])
            if extra_results:
                logger.info(
                    f"   GraphRAG: expanded results by {len(extra_results)} entity-linked doc(s)"
                )
                results = list(results) + extra_results
        except Exception as e:
            logger.debug(f"GraphRAG expansion failed: {e}")

        return results

    def _doc_hash(self, doc: dict) -> str:
        """Return an MD5 hex digest of the document's text content."""
        import hashlib

        return hashlib.md5(doc.get("text", "").encode("utf-8", errors="replace")).hexdigest()

    def _make_cache_key(self, query: str, filters, cfg) -> str:
        """Return a stable cache key for the given query + active config.

        All flags that change retrieval or generation are included so two
        requests that differ only by (e.g.) cite_sources or compress_context
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
            "cite_sources": cfg.cite_sources,
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

    def ingest(self, documents: list[dict[str, Any]]):
        if not documents:
            return
        t0 = time.time()
        from tqdm import tqdm

        logger.info(f"Ingesting {len(documents)} documents...")
        if self.splitter and self.config.parent_chunk_size > 0:
            documents = self._split_with_parents(documents)
        elif self.splitter:
            documents = self.splitter.transform_documents(documents)

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
            for doc in documents:
                if doc.get("metadata", {}).get("raptor_level"):
                    continue  # skip synthetic RAPTOR nodes
                entities = self._extract_entities(doc["text"])
                for entity in entities:
                    key = entity.lower()
                    if key not in self._entity_graph:
                        self._entity_graph[key] = []
                    if doc["id"] not in self._entity_graph[key]:
                        self._entity_graph[key].append(doc["id"])
                        updated = True
            if updated:
                self._save_entity_graph()

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
        """Central retrieval execution logic supporting HyDE, Multi-Query, and Web Search."""
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

        search_queries = [query]
        vector_query = query

        if cfg.multi_query:
            search_queries = self._get_multi_queries(query)
            transforms["multi_query_applied"] = True
            transforms["queries"] = search_queries

        if cfg.step_back:
            abstract_query = self._get_step_back_query(query)
            if abstract_query not in search_queries:
                search_queries = search_queries + [abstract_query]
            transforms["step_back_applied"] = True
            transforms["queries"] = search_queries

        if cfg.query_decompose:
            sub_questions = self._decompose_query(query)
            # Merge without duplicates, preserving order
            existing = set(search_queries)
            for sq in sub_questions:
                if sq not in existing:
                    search_queries = search_queries + [sq]
                    existing.add(sq)
            transforms["decompose_applied"] = True
            transforms["queries"] = search_queries

        if cfg.hyde:
            # We construct a single hypothetical document based on the primary query
            vector_query = self._get_hyde_document(query)
            transforms["hyde_applied"] = True

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
            # When multi_query is enabled, batch embed all queries to avoid
            # multiple sequential embedding calls (latency/cost optimization).
            if len(search_queries) == 1:
                sq = search_queries[0]
                all_vector_results.extend(
                    self.vector_store.search(
                        self.embedding.embed_query(sq), top_k=fetch_k, filter_dict=filters
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

        # Hybrid Search
        bm25_count = 0
        if cfg.hybrid_search and self.bm25:
            for sq in search_queries:
                all_bm25_results.extend(self.bm25.search(sq, top_k=fetch_k))

            dedup_bm25 = {}
            for r in all_bm25_results:
                if r["id"] not in dedup_bm25 or r["score"] > dedup_bm25[r["id"]]["score"]:
                    dedup_bm25[r["id"]] = r
            bm25_results = list(dedup_bm25.values())
            bm25_count = len(bm25_results)

            from axon.retrievers import reciprocal_rank_fusion

            results = reciprocal_rank_fusion(vector_results, bm25_results)
        else:
            results = vector_results

        results = [
            r
            for r in results
            if r.get("score", 1.0) >= cfg.similarity_threshold or "fused_only" in r
        ]

        # GraphRAG: expand results with entity-linked documents
        if cfg.graph_rag and self._entity_graph:
            results = self._expand_with_entity_graph(query, results, cfg=cfg)

        # Web Search (Truth Grounding)
        # Trigger when the best cosine similarity from vector search is below the threshold,
        # meaning local knowledge is genuinely insufficient (even if BM25 returned fused_only docs).
        web_count = 0
        if cfg.truth_grounding and cfg.brave_api_key:
            max_vector_score = max((r["score"] for r in vector_results), default=0.0)
            local_sufficient = max_vector_score >= cfg.similarity_threshold
            if not local_sufficient:
                logger.info(
                    f"Local knowledge insufficient (best vector score {max_vector_score:.3f} < "
                    f"threshold {cfg.similarity_threshold}) — falling back to Brave web search"
                )
                web_results = self._execute_web_search(query)
                web_count = len(web_results)
                # Replace low-relevance local results with web results
                results = web_results
                transforms["web_search_applied"] = True

        return {
            "results": results,
            "vector_count": vector_count,
            "bm25_count": bm25_count,
            "web_count": web_count,
            "filtered_count": len(results),
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
        """
        if cfg is None:
            cfg = self.config
        base = self.SYSTEM_PROMPT if cfg.discussion_fallback else self.SYSTEM_PROMPT_STRICT

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

        results = results[: cfg.top_k]
        if cfg.compress_context:
            results = self._compress_context(query, results)
        final_count = len(results)
        top_score = results[0].get("score", 0) if results else 0
        context, has_web = self._build_context(results)
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

        results = results[: cfg.top_k]
        if cfg.compress_context:
            results = self._compress_context(query, results)
        context, has_web = self._build_context(results)
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

_MERGED_VIEW_WRITE_ERROR = (
    "Cannot write to a merged parent project view. " "Switch to a specific sub-project first."
)


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
        "--cite",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable inline source citations ([Doc N]) in LLM answers",
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
    if args.cite is not None:
        config.cite_sources = args.cite
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

        if os.path.isdir(args.ingest):
            asyncio.run(brain.load_directory(args.ingest))
        else:
            from axon.loaders import DirectoryLoader

            ext = os.path.splitext(args.ingest)[1].lower()
            loader_mgr = DirectoryLoader()
            if ext in loader_mgr.loaders:
                brain.ingest(loader_mgr.loaders[ext].load(args.ingest))

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
    "/help",
    "/list",
    "/ingest ",
    "/model ",
    "/embed ",
    "/pull ",
    "/search",
    "/discuss",
    "/rag ",
    "/compact",
    "/context",
    "/sessions",
    "/resume ",
    "/clear",
    "/retry",
    "/project ",
    "/keys",
    "/vllm-url ",
    "/quit",
    "/exit",
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


def _sessions_dir() -> str:
    os.makedirs(_SESSIONS_DIR, exist_ok=True)
    return _SESSIONS_DIR


def _new_session(brain: "AxonBrain") -> dict:
    return {
        "id": _dt.now(_tz.utc).strftime("%Y%m%dT%H%M%S%f")[:-3],
        "started_at": _dt.now(_tz.utc).isoformat(),
        "provider": brain.config.llm_provider,
        "model": brain.config.llm_model,
        "history": [],
    }


def _session_path(session_id: str) -> str:
    return os.path.join(_sessions_dir(), f"session_{session_id}.json")


def _save_session(session: dict) -> None:
    try:
        with open(_session_path(session["id"]), "w", encoding="utf-8") as f:
            _json.dump(session, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _list_sessions(limit: int = 20) -> list:
    d = _sessions_dir()
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


def _load_session(session_id: str) -> dict | None:
    p = _session_path(session_id)
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
_AT_LOADER_EXTS = {".docx", ".pdf"}
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
                    for s in _list_sessions():
                        sid = s["id"]
                        if sid.startswith(prefix):
                            turns = len(s.get("history", [])) // 2
                            yield Completion(
                                sid[len(prefix) :], display=sid, display_meta=f"{turns} turns"
                            )
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
                        "  Knowledge:  /list   /ingest <path>   /clear\n"
                        "  Model:      /model [<prov/>model]   /embed [<prov/>model]   /pull <name>\n"
                        "  Mode:       /search   /discuss   /rag [option value]\n"
                        "  Session:    /sessions   /resume <id>   /compact   /context   /retry\n"
                        "  Projects:   /project [list|new|switch|delete|folder]\n"
                        "  API Keys:   /keys   /keys set <provider>\n"
                        "  Other:      /help [cmd]   /quit\n"
                        "  Shell:      !<cmd>  run a shell command  ·  @<file>  attach file  ·  @<folder>/  attach all text files in folder\n"
                        "\n"
                        "  /help <cmd>  for details (model, embed, ingest, rag, sessions, keys, project)\n"
                        "  /<cmd> help  also works  ·  e.g. /project help   /rag help\n"
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
                        f"  cite         · {'ON' if brain.config.cite_sources else 'OFF'}\n"
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
                        brain.config.cite_sources = not brain.config.cite_sources
                        print(f"  Inline citations {'ON' if brain.config.cite_sources else 'OFF'}")
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
                _print_sessions(_list_sessions())

            elif cmd == "/resume":
                if not arg:
                    print("  Usage: /resume <session-id>")
                else:
                    loaded = _load_session(arg)
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
                                    brain.config.openai_api_key = new_key
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


if __name__ == "__main__":
    main()
