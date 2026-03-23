"""
Axon - Open Source RAG Interface

A fully open-source local-first RAG implementation.
Uses sentence-transformers, Ollama, and ChromaDB/Qdrant.
"""

from .config import AxonConfig
from .embeddings import OpenEmbedding
from .llm import OpenLLM
from .main import AxonBrain
from .vector_store import OpenVectorStore

__version__ = "0.9.0"
__all__ = [
    "AxonBrain",
    "AxonConfig",
    "OpenEmbedding",
    "OpenLLM",
    "OpenVectorStore",
]
