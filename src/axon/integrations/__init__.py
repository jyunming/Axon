"""Third-party framework integrations for Axon.

Each submodule is gated behind an optional extra (declared in ``pyproject.toml``)
so the base ``axon-rag`` install stays minimal:

- ``axon-rag[langchain]`` → :mod:`axon.integrations.langchain`
- ``axon-rag[llama-index]`` → :mod:`axon.integrations.llama_index`

All adapters wrap :meth:`AxonBrain.search_raw` so they share a single retrieval
codepath and inherit reranking, hybrid search, HyDE, multi-query, and the
graph-rag budget exactly as the REST and REPL surfaces see them.
"""
from __future__ import annotations
