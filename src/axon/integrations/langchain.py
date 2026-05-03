"""LangChain ``BaseRetriever`` adapter for :class:`AxonBrain`.

Install with::

    pip install "axon-rag[langchain]"

Usage::

    from axon import AxonBrain, AxonConfig
    from axon.integrations.langchain import AxonRetriever

    brain = AxonBrain(AxonConfig.from_yaml("config.yaml"))
    retriever = AxonRetriever(brain=brain, top_k=5)

    # Drop-in for any LangChain chain that takes a Retriever:
    docs = retriever.invoke("what does the project do?")
    # → list[langchain_core.documents.Document]

The adapter delegates to :meth:`AxonBrain.search_raw`, so all the project's
retrieval features (hybrid, rerank, HyDE, multi-query, GraphRAG budget,
sentence-window) apply without extra configuration. Per-call overrides can be
passed via the ``overrides`` constructor argument or by using
``AxonRetriever.with_overrides({...}).invoke(query)`` for a one-off override.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from pydantic import ConfigDict

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised by extra-not-installed path
    _LANGCHAIN_AVAILABLE = False
    # Names imported above are NOT reassigned here — when the extra is
    # missing we never enter the `if _LANGCHAIN_AVAILABLE:` branch that
    # references them, so leaving them unbound is correct. Reassigning
    # to a sentinel triggered "Unused type: ignore" warnings under mypy
    # in environments that DO have the extra installed.

if TYPE_CHECKING:
    pass


_INSTALL_HINT = (
    "LangChain integration requires the [langchain] extra. "
    "Install it with: pip install 'axon-rag[langchain]'"
)


def _check_available() -> None:
    """Raise a clear ImportError when ``langchain-core`` is missing."""
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(_INSTALL_HINT)


def _result_to_document(result: dict[str, Any]) -> Any:
    """Convert one ``brain.search_raw`` result row to a LangChain ``Document``.

    Web-search rows (``is_web=True``) are kept; the source URL lands in
    metadata so downstream chains can render citations.
    """
    text = result.get("text", "") or ""
    meta = dict(result.get("metadata", {}) or {})
    # Normalise the most useful identifiers into the metadata dict so chains
    # that consume Document.metadata (e.g. citation chains) can find them
    # without reaching back into Axon-specific keys.
    if "id" in result and "id" not in meta:
        meta["id"] = result["id"]
    if "score" in result and "score" not in meta:
        meta["score"] = result["score"]
    if result.get("is_web"):
        meta.setdefault("source_kind", "web")
    return Document(page_content=text, metadata=meta)


if _LANGCHAIN_AVAILABLE:

    class AxonRetriever(BaseRetriever):
        """LangChain :class:`BaseRetriever` backed by :meth:`AxonBrain.search_raw`.

        Args:
            brain: A live :class:`AxonBrain` instance.
            top_k: Override the project's default ``top_k`` for this retriever.
                ``None`` falls through to ``brain.config.top_k``.
            filters: Optional metadata filter dict applied to every call.
            overrides: Per-retriever RAG flag overrides (``hybrid``, ``rerank``,
                ``hyde``, ``multi_query``, ``step_back``, ``threshold``,
                ``graph_rag``, etc.). Forwarded verbatim to ``search_raw``.
        """

        # Pydantic v2 / langchain-core BaseModel needs explicit field declarations.
        brain: Any
        top_k: int | None = None
        filters: dict[str, Any] | None = None
        overrides: dict[str, Any] | None = None

        model_config = ConfigDict(arbitrary_types_allowed=True)

        def with_overrides(self, overrides: dict[str, Any]) -> AxonRetriever:
            """Return a copy of the retriever with merged overrides.

            Original retriever is left untouched so callers can keep a base
            instance and derive per-question variants without state leakage.
            """
            merged = dict(self.overrides or {})
            merged.update(overrides or {})
            return AxonRetriever(
                brain=self.brain,
                top_k=self.top_k,
                filters=self.filters,
                overrides=merged,
            )

        def _build_overrides(self) -> dict[str, Any] | None:
            ov = dict(self.overrides or {})
            if self.top_k is not None:
                ov.setdefault("top_k", self.top_k)
            return ov or None

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
        ) -> list[Any]:
            results, _diag, _trace = self.brain.search_raw(
                query,
                filters=self.filters,
                overrides=self._build_overrides(),
            )
            return [_result_to_document(r) for r in results]

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
        ) -> list[Any]:
            # AxonBrain.search_raw is synchronous; defer to a thread to keep
            # the event loop free. We don't pass run_manager into the sync
            # path because its callback shape is async; the sync method
            # only needs the brain reference and config from `self`.
            import asyncio

            def _run() -> list[Any]:
                results, _diag, _trace = self.brain.search_raw(
                    query,
                    filters=self.filters,
                    overrides=self._build_overrides(),
                )
                return [_result_to_document(r) for r in results]

            return await asyncio.to_thread(_run)

else:

    def AxonRetriever(*_args, **_kwargs):  # type: ignore[no-redef]
        """Stub raised when ``langchain-core`` is not installed."""
        _check_available()
        raise AssertionError("unreachable")  # pragma: no cover


__all__ = ["AxonRetriever"]
