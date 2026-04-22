"""


Axon - Open Source RAG Interface


A fully open-source local-first RAG implementation.


Uses sentence-transformers, Ollama, and LanceDB/ChromaDB/Qdrant.


"""


from pathlib import Path

from ._rust_loader import bootstrap_dev_rust_module

bootstrap_dev_rust_module(__name__, Path(__file__).resolve().parent)

from .config import AxonConfig  # noqa: E402
from .embeddings import OpenEmbedding  # noqa: E402
from .llm import OpenLLM  # noqa: E402
from .main import AxonBrain  # noqa: E402
from .vector_store import OpenVectorStore  # noqa: E402

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("axon-rag")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"


__all__ = [
    "AxonBrain",
    "AxonConfig",
    "OpenEmbedding",
    "OpenLLM",
    "OpenVectorStore",
]
