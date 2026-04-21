"""


Axon - Open Source RAG Interface


A fully open-source local-first RAG implementation.


Uses sentence-transformers, Ollama, and LanceDB/ChromaDB/Qdrant.


"""


from pathlib import Path

from ._rust_loader import bootstrap_dev_rust_module

bootstrap_dev_rust_module(__name__, Path(__file__).resolve().parent)

from .config import AxonConfig
from .embeddings import OpenEmbedding
from .llm import OpenLLM
from .main import AxonBrain
from .vector_store import OpenVectorStore

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
