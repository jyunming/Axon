"""
src/axon/embeddings.py

OpenEmbedding client extracted from main.py for Phase 2 of the Axon refactor.
"""

import logging
from typing import Any

from axon.config import AxonConfig

logger = logging.getLogger("Axon")

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
        _model_path = getattr(self.config, "embedding_model_path", "")
        if self.provider == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            _src = _model_path or self.config.embedding_model
            logger.info(f"Loading Sentence Transformers: {_src}")
            self.model = SentenceTransformer(_src)
            self.dimension = self.model.get_sentence_embedding_dimension()

        elif self.provider == "ollama":
            logger.info(f"Using Ollama Embedding: {self.config.embedding_model}")
            self.dimension = _KNOWN_DIMS.get(self.config.embedding_model, 768)

        elif self.provider == "fastembed":
            from fastembed import TextEmbedding

            _kwargs: dict = {"model_name": self.config.embedding_model}
            if _model_path:
                _kwargs["cache_dir"] = _model_path
            logger.info(
                f"Loading FastEmbed: {self.config.embedding_model}"
                + (f" (cache_dir={_model_path})" if _model_path else "")
            )
            self.model = TextEmbedding(**_kwargs)
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
