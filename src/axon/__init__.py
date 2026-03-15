"""
Axon - Open Source RAG Interface

A fully open-source local-first RAG implementation.
Uses sentence-transformers, Ollama, and ChromaDB/Qdrant.
"""

from .main import AxonBrain, AxonConfig, OpenEmbedding, OpenLLM, OpenVectorStore


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


__version__ = "1.0.0"
__all__ = [
    "AxonBrain",
    "AxonConfig",
    "OpenEmbedding",
    "OpenLLM",
    "OpenVectorStore",
    "OpenStudioBrain",
    "OpenStudioConfig",
]
