"""
Axon - Open Source RAG Interface

A fully open-source local-first RAG implementation.
Uses sentence-transformers, Ollama, and ChromaDB/Qdrant.
"""

from .main import OpenEmbedding, OpenLLM, AxonBrain, AxonConfig, OpenVectorStore

__version__ = "2.0.0"
__all__ = ["AxonBrain", "AxonConfig", "OpenEmbedding", "OpenLLM", "OpenVectorStore"]
