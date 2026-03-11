"""
Axon - Open Source RAG Interface

A fully open-source local-first RAG implementation.
Uses sentence-transformers, Ollama, and ChromaDB/Qdrant.
"""

from .main import OpenEmbedding, OpenLLM, OpenStudioBrain, OpenStudioConfig, OpenVectorStore

__version__ = "2.0.0"
__all__ = [
    "OpenStudioBrain",
    "OpenStudioConfig",
    "OpenEmbedding",
    "OpenLLM",
    "OpenVectorStore"
]
