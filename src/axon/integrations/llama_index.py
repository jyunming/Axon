"""LlamaIndex ``BaseRetriever`` adapter for :class:`AxonBrain`.

Install with::

    pip install "axon-rag[llama-index]"

Usage::

    from axon import AxonBrain, AxonConfig
    from axon.integrations.llama_index import AxonLlamaRetriever

    brain = AxonBrain(AxonConfig.from_yaml("config.yaml"))
    retriever = AxonLlamaRetriever(brain=brain, top_k=5)

    # Drop-in for any LlamaIndex query engine that takes a Retriever:
    nodes = retriever.retrieve("what does the project do?")
    # → list[NodeWithScore]

The adapter delegates to :meth:`AxonBrain.search_raw`, so all the project's
retrieval features (hybrid, rerank, HyDE, multi-query, GraphRAG budget,
sentence-window) apply without extra configuration.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

    _LLAMA_INDEX_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised by extra-not-installed path
    _LLAMA_INDEX_AVAILABLE = False
    # Names imported above are NOT reassigned here — when the extra is
    # missing we never enter the `if _LLAMA_INDEX_AVAILABLE:` branch that
    # references them. Reassigning to a sentinel triggered "Unused type:
    # ignore" warnings under mypy when the extra IS installed.

if TYPE_CHECKING:
    from axon.main import AxonBrain


_INSTALL_HINT = (
    "LlamaIndex integration requires the [llama-index] extra. "
    "Install it with: pip install 'axon-rag[llama-index]'"
)


def _check_available() -> None:
    """Raise a clear ImportError when ``llama-index-core`` is missing."""
    if not _LLAMA_INDEX_AVAILABLE:
        raise ImportError(_INSTALL_HINT)


def _result_to_node_with_score(result: dict[str, Any]) -> Any:
    """Convert one ``brain.search_raw`` row to a LlamaIndex ``NodeWithScore``."""
    text = result.get("text", "") or ""
    meta = dict(result.get("metadata", {}) or {})
    node_id = result.get("id") or meta.get("id") or ""
    if result.get("is_web"):
        meta.setdefault("source_kind", "web")
    score = result.get("score")
    score_val: float | None = float(score) if isinstance(score, int | float) else None
    node = TextNode(text=text, id_=str(node_id), metadata=meta)
    return NodeWithScore(node=node, score=score_val)


if _LLAMA_INDEX_AVAILABLE:

    class AxonLlamaRetriever(BaseRetriever):
        """LlamaIndex :class:`BaseRetriever` backed by :meth:`AxonBrain.search_raw`.

        Args:
            brain: A live :class:`AxonBrain` instance.
            top_k: Override the project's default ``top_k``.
            filters: Optional metadata filter dict applied to every call.
            overrides: Per-retriever RAG flag overrides forwarded verbatim to
                ``search_raw``.
        """

        def __init__(
            self,
            brain: AxonBrain,
            top_k: int | None = None,
            filters: dict[str, Any] | None = None,
            overrides: dict[str, Any] | None = None,
        ) -> None:
            super().__init__()
            self._brain = brain
            self._top_k = top_k
            self._filters = filters
            self._overrides = overrides

        def _build_overrides(self) -> dict[str, Any] | None:
            ov = dict(self._overrides or {})
            if self._top_k is not None:
                ov.setdefault("top_k", self._top_k)
            return ov or None

        def _retrieve(self, query_bundle: Any) -> list[Any]:
            query = (
                query_bundle.query_str if hasattr(query_bundle, "query_str") else str(query_bundle)
            )
            results, _diag, _trace = self._brain.search_raw(
                query,
                filters=self._filters,  # type: ignore[arg-type]
                overrides=self._build_overrides(),  # type: ignore[arg-type]
            )
            return [_result_to_node_with_score(r) for r in results]

        async def _aretrieve(self, query_bundle: Any) -> list[Any]:
            import asyncio

            return await asyncio.to_thread(self._retrieve, query_bundle)

else:

    def AxonLlamaRetriever(*_args, **_kwargs):  # type: ignore[no-redef]
        """Stub raised when ``llama-index-core`` is not installed."""
        _check_available()
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["AxonLlamaRetriever"]
