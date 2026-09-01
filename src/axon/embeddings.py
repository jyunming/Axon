"""
src/axon/embeddings.py

OpenEmbedding client extracted from main.py for Phase 2 of the Axon refactor.
"""

import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from axon.config import AxonConfig

logger = logging.getLogger("Axon")

_T = TypeVar("_T")


def _retry_call(
    fn: Callable[[], _T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 4.0,
) -> _T:
    """Call *fn* with exponential backoff + jitter.
    Retries on any exception except those signalling a permanent failure
    (TypeError / AttributeError / ValueError) — those bubble up so a
    typo in the call site doesn't get hidden behind 3 retries (audit
    P1: previously zero retry framework, transient 503s killed long
    ingests).
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (TypeError, AttributeError, ValueError):
            raise
        except Exception as exc:  # network / provider transient failure
            last_exc = exc
            if attempt == attempts:
                break
            sleep_s = min(max_delay, base_delay * (2 ** (attempt - 1)))
            sleep_s += random.uniform(0, sleep_s * 0.1)
            logger.debug(
                "Provider call attempt %d/%d failed (%s); sleeping %.2fs",
                attempt,
                attempts,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)
    assert last_exc is not None
    raise last_exc


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
    # fastembed uses the org-prefixed hub id even for models sourced from the
    # sentence-transformers org — same weights/dims as the bare name above.
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    # OpenAI models
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_ST_ORG_PREFIX = "sentence-transformers/"


def _hf_hub_cache_dir() -> str:
    return os.path.join(os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def is_hf_model_cached(model_id: str, *, guess_st_prefix: bool = True) -> bool:
    """Check whether an HF hub model id is present in the local hub cache.
    When *guess_st_prefix* (default), also handles sentence-transformers'
    short-name auto-resolution: a bare name like ``"all-MiniLM-L6-v2"`` is
    actually cached on disk as ``models--sentence-transformers--all-MiniLM-L6-v2``,
    not ``models--all-MiniLM-L6-v2``. That guess is only valid for models
    sentence_transformers itself resolves this way — pass ``guess_st_prefix=False``
    for other model kinds (reranker, GLiNER, REBEL, LLMLingua) where a bare id
    coincidentally matching a cached sentence-transformers-org repo would be a
    false positive, not a real cache hit.
    """
    if not model_id or os.path.isabs(model_id) or model_id.startswith("."):
        return False
    cache_dir = _hf_hub_cache_dir()
    candidates = [model_id]
    if guess_st_prefix and "/" not in model_id:
        candidates.append(_ST_ORG_PREFIX + model_id)
    return any(
        os.path.isdir(os.path.join(cache_dir, "models--" + candidate.replace("/", "--")))
        for candidate in candidates
    )


def fastembed_default_cache_dir(axon_store_base: str) -> str:
    """Stable fastembed cache dir under the AxonStore root (not fastembed's own
    default, the OS temp dir — Windows/macOS periodically sweep temp, which
    would silently re-trigger the first-run download)."""
    return os.path.join(axon_store_base, "model_cache", "fastembed")


# Known fastembed catalog-id -> actual backing HF repo, so the common case (the
# shipped default model) doesn't need to import fastembed just to answer "is
# this cached?" — that import alone costs ~1s, which the whole point of the
# preflight audit calling this on every boot is to avoid paying twice.
_KNOWN_FASTEMBED_SOURCES = {
    "sentence-transformers/all-MiniLM-L6-v2": "qdrant/all-MiniLM-L6-v2-onnx",
}


def is_fastembed_model_cached(model_id: str, cache_dir: str) -> bool:
    """Check whether *model_id* is already present in a fastembed cache_dir.
    fastembed's public catalog name and the HF repo it actually downloads from
    often differ (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"`` is fetched
    from ``"qdrant/all-MiniLM-L6-v2-onnx"``), so a plain ``models--<model_id>``
    slug check would false-negative — resolve the real source repo first, via
    the static map above for the shipped default, else fastembed's own
    (offline, bundled) catalog.
    """
    if not os.path.isdir(cache_dir):
        return False
    repo = _KNOWN_FASTEMBED_SOURCES.get(model_id)
    if repo is None:
        repo = model_id
        try:
            from fastembed import TextEmbedding

            for entry in TextEmbedding.list_supported_models():
                if isinstance(entry, dict) and entry.get("model") == model_id:
                    repo = entry.get("sources", {}).get("hf") or model_id
                    break
        except Exception:
            pass  # fall back to the public model_id; worst case under-detects a cache hit
    slug = "models--" + repo.replace("/", "--")
    return os.path.isdir(os.path.join(cache_dir, slug))


# Provider/model pairs verified numerically identical across sentence_transformers
# and fastembed (cosine similarity 1.0 on real embeddings, both providers, same
# text). Do NOT add an entry without actually verifying it — a wrong equivalence
# here would let embedding_identity() silently pass a real mismatch, corrupting
# a collection with vectors from two different models.
_VERIFIED_CROSS_PROVIDER_MODELS = frozenset({"all-MiniLM-L6-v2"})


def embedding_identity(provider: str, model: str) -> tuple[str, str]:
    """Canonicalize (provider, model) so numerically-identical embeddings
    compare equal across providers.  Scoped to the specific models in
    :data:`_VERIFIED_CROSS_PROVIDER_MODELS` — everything else (including other
    bare sentence-transformers-org model names never verified against their
    fastembed ONNX export) is returned unchanged, i.e. treated as a real
    mismatch if the provider/model string differs at all.
    """
    if provider in ("sentence_transformers", "fastembed"):
        bare = model[len(_ST_ORG_PREFIX) :] if model.startswith(_ST_ORG_PREFIX) else model
        if bare in _VERIFIED_CROSS_PROVIDER_MODELS:
            return ("st-family", bare)
    return (provider, model)


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
        _model_path = getattr(self.config, "embedding_model_path", "")
        if self.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            _src = _model_path or self.config.embedding_model
            logger.info(f"Loading Sentence Transformers: {_src}")
            # Skip the "check the hub for a newer revision" network round-trip
            # (~4s of serialized HEAD/GET requests) when the model is already
            # cached — it's pure overhead for a model we're not going to
            # re-download anyway.
            _local_only = not _model_path and is_hf_model_cached(_src)
            try:
                self.model = SentenceTransformer(_src, local_files_only=_local_only)
            except Exception:
                if not _local_only:
                    raise
                # Cache dir existed but the load still failed (e.g. an
                # interrupted earlier download left a partial models--* dir).
                # Fall back to the normal online path instead of hard-failing.
                logger.warning(
                    "Local-only load of '%s' failed despite a cache hit; retrying online.",
                    _src,
                )
                self.model = SentenceTransformer(_src)
            self.dimension = (
                getattr(self.config, "embedding_dim", 0)
                or self.model.get_sentence_embedding_dimension()
            )
        elif self.provider == "ollama":
            logger.info(f"Using Ollama Embedding: {self.config.embedding_model}")
            _cfg_dim = getattr(self.config, "embedding_dim", 0)
            if _cfg_dim:
                self.dimension = _cfg_dim
            else:
                if self.config.embedding_model not in _KNOWN_DIMS:
                    logger.warning(
                        "Ollama embedding model '%s' is not in the dimension registry; "
                        "defaulting to 768-dim.  If this is wrong, set embedding_dim in config.",
                        self.config.embedding_model,
                    )
                self.dimension = _KNOWN_DIMS.get(self.config.embedding_model, 768)
        elif self.provider == "fastembed":
            try:
                from fastembed import TextEmbedding
            except ImportError as exc:
                raise ImportError(
                    "FastEmbed is not installed. It ships as a base dependency of "
                    "axon-rag since 0.4.6, so this usually means the install is "
                    "incomplete or was made without dependencies — try: "
                    "pip install --upgrade axon-rag"
                ) from exc
            _cache_dir = _model_path or fastembed_default_cache_dir(self.config.axon_store_base)
            try:
                os.makedirs(_cache_dir, exist_ok=True)
            except OSError:
                pass  # fastembed's own downloader will surface a clearer error if unwritable
            _kwargs: dict = {"model_name": self.config.embedding_model, "cache_dir": _cache_dir}
            logger.info(
                f"Loading FastEmbed: {self.config.embedding_model} (cache_dir={_cache_dir})"
            )
            self.model = TextEmbedding(**_kwargs)
            _cfg_dim = getattr(self.config, "embedding_dim", 0)
            if _cfg_dim:
                self.dimension = _cfg_dim
            elif self.config.embedding_model in _KNOWN_DIMS:
                self.dimension = _KNOWN_DIMS[self.config.embedding_model]
            else:
                # Auto-detect dimension by probing the model with a short sentinel string.
                # This avoids a silent 384-dim fallback that can corrupt existing collections.
                _probe = list(self.model.embed(["dim-probe"]))
                self.dimension = len(_probe[0]) if _probe else 384
                logger.info(
                    "FastEmbed: auto-detected dimension %d for model '%s'. "
                    "Add it to _KNOWN_DIMS to skip this probe on future loads.",
                    self.dimension,
                    self.config.embedding_model,
                )
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
            _cfg_dim = getattr(self.config, "embedding_dim", 0)
            if _cfg_dim:
                self.dimension = _cfg_dim
            else:
                if self.config.embedding_model not in _KNOWN_DIMS:
                    logger.warning(
                        "OpenAI embedding model '%s' is not in the dimension registry; "
                        "defaulting to 1536-dim. Add it to _KNOWN_DIMS if this is wrong.",
                        self.config.embedding_model,
                    )
                self.dimension = _KNOWN_DIMS.get(self.config.embedding_model, 1536)
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if os.getenv("AXON_DRY_RUN"):
            return [[0.0] * self.dimension for _ in texts]
        if self.provider == "sentence_transformers":
            embeddings = self.model.encode(texts, show_progress_bar=False)
            if hasattr(embeddings, "tolist"):
                return embeddings.tolist()
            return list(embeddings)
        elif self.provider == "ollama":
            from ollama import Client

            client = Client(host=self.config.ollama_base_url)
            try:
                # Batch API (ollama-python >= 0.4): single round-trip for all texts.
                response = _retry_call(
                    lambda: client.embed(model=self.config.embedding_model, input=texts),
                )
                return response["embeddings"]
            except (AttributeError, TypeError):
                # Older client without batch embed — fall back to sequential calls.
                embeddings = []
                for text in texts:
                    response = _retry_call(
                        lambda t=text: client.embeddings(
                            model=self.config.embedding_model, prompt=t
                        ),
                    )
                    embeddings.append(response["embedding"])
                return embeddings
        elif self.provider == "fastembed":
            embeddings = list(self.model.embed(texts))
            return [e.tolist() for e in embeddings]
        elif self.provider == "openai":
            response = _retry_call(
                lambda: self.model.embeddings.create(
                    input=texts, model=self.config.embedding_model
                ),
            )
            return [data.embedding for data in response.data]
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}")

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]
