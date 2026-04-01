# Axon source package
from axon import AxonBrain, AxonConfig, OpenEmbedding, OpenLLM, OpenVectorStore

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
