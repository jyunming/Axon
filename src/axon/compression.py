"""
axon/compression.py — General context-compression abstraction (Epic 3, Story 3.1).

Strategies
----------
- ``none``       : pass-through; chunks returned unchanged
- ``sentence``   : LLM-based sentence extraction (existing behaviour, re-homed here)
- ``llmlingua``  : LLMLingua-2 token-level compression; soft dependency (optional install)

All strategies return a :class:`CompressionResult` that carries pre-/post-token estimates
and the strategy actually used, so callers can record telemetry without knowing
which code path ran.

Fallback chain
--------------
``llmlingua`` → falls back to ``sentence`` on import error or runtime failure
``sentence``  → falls back to original chunk on LLM failure per-chunk
``none``      → always succeeds (trivial pass-through)
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from axon.llm import OpenLLM  # noqa: F401 — type hint only

logger = logging.getLogger(__name__)

_DEFAULT_LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token count: ~4 chars per token for English prose."""
    return max(1, len(text) // 4)


def _chunk_text(chunk: dict) -> str:
    """Return the text that _build_context will use for this chunk."""
    parent = chunk.get("metadata", {}).get("parent_text")
    return parent if parent else chunk.get("text", "")


def _total_tokens(chunks: list[dict]) -> int:
    return sum(_estimate_tokens(_chunk_text(c)) for c in chunks)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class CompressionResult:
    """Output of :class:`ContextCompressor.compress`.

    Carries the (possibly compressed) chunks plus telemetry fields that map
    directly to the diagnostics contract (Epic 3, Story 3.3).
    """

    chunks: list[dict]
    strategy_used: str  # "none" | "sentence" | "llmlingua"
    pre_tokens: int
    post_tokens: int
    compression_ratio: float  # post / pre; 1.0 = no reduction
    fallback_reason: str = ""  # non-empty when fell back from requested strategy


# ---------------------------------------------------------------------------
# ContextCompressor
# ---------------------------------------------------------------------------


class ContextCompressor:
    """Centralised context-compression gateway (Story 3.1).

    Instantiate once per query (or cache on the router instance).
    Thread-safe: ``compress()`` uses an internal ThreadPoolExecutor for the
    sentence strategy; LLMLingua is single-threaded by default.

    Parameters
    ----------
    llm:
        The generation LLM, used only by the ``sentence`` strategy.
        May be ``None`` if that strategy will never be requested.
    llmlingua_model:
        HuggingFace model identifier or local path for LLMLingua-2.
        Defaults to the ``microsoft/llmlingua-2-…-meetingbank`` checkpoint.
    """

    def __init__(
        self,
        llm: Any = None,
        llmlingua_model: str = _DEFAULT_LLMLINGUA_MODEL,
    ) -> None:
        self._llm = llm
        self._llmlingua_model = llmlingua_model or _DEFAULT_LLMLINGUA_MODEL
        self._lingua: Any = None  # lazy-init on first LLMLingua request

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        query: str,
        chunks: list[dict],
        strategy: str = "sentence",
        token_budget: int = 0,
    ) -> CompressionResult:
        """Compress *chunks* and return a :class:`CompressionResult`.

        Parameters
        ----------
        query:
            The user query; used by sentence and LLMLingua strategies.
        chunks:
            Retrieved document chunks (same shape as returned by the retriever).
        strategy:
            One of ``"none"``, ``"sentence"``, ``"llmlingua"``.
            Unknown values are treated as ``"none"``.
        token_budget:
            Target maximum tokens for the compressed context (LLMLingua only).
            ``0`` means no explicit budget; the model uses its default ratio.
        """
        if not chunks:
            return CompressionResult(
                chunks=chunks,
                strategy_used="none",
                pre_tokens=0,
                post_tokens=0,
                compression_ratio=1.0,
            )

        pre_tokens = _total_tokens(chunks)
        fallback_reason = ""

        if strategy == "llmlingua":
            compressed, fallback_reason = self._llmlingua_compress(query, chunks, token_budget)
            if fallback_reason:
                # Fell back — compressed may be the sentence result or the originals
                strategy_used = "sentence" if self._llm else "none"
            else:
                strategy_used = "llmlingua"
        elif strategy == "sentence":
            compressed = self._sentence_compress(query, chunks)
            strategy_used = "sentence"
        else:
            # "none" or unknown
            compressed = chunks
            strategy_used = "none"

        post_tokens = _total_tokens(compressed)
        ratio = round(post_tokens / pre_tokens, 4) if pre_tokens else 1.0

        logger.info(
            "Compression: strategy=%s pre=%d post=%d ratio=%.3f fallback=%r",
            strategy_used,
            pre_tokens,
            post_tokens,
            ratio,
            fallback_reason or "none",
        )
        return CompressionResult(
            chunks=compressed,
            strategy_used=strategy_used,
            pre_tokens=pre_tokens,
            post_tokens=post_tokens,
            compression_ratio=ratio,
            fallback_reason=fallback_reason,
        )

    # ------------------------------------------------------------------
    # Sentence-extraction strategy (Story 3.2 — re-homed from query_router)
    # ------------------------------------------------------------------

    def _sentence_compress(self, query: str, chunks: list[dict]) -> list[dict]:
        """Ask the LLM to extract only query-relevant sentences from each chunk.

        Skips web results. Falls back to the original chunk on error or if
        compression makes the text longer.  Runs chunks in parallel (≤4 workers).
        """
        if self._llm is None:
            logger.debug("Sentence compression skipped — no LLM attached")
            return chunks

        def _compress_one(chunk: dict) -> dict:
            if chunk.get("is_web"):
                return chunk
            source_text = _chunk_text(chunk)
            prompt = (
                "Extract only the sentences from the passage below that directly help answer "
                "the question. Output only those sentences verbatim, nothing else. "
                "If no sentence is relevant, keep the single most informative sentence.\n\n"
                f"Question: {query}\n\nPassage:\n{source_text}"
            )
            try:
                compressed = self._llm.complete(
                    prompt,
                    system_prompt="You are an expert at extracting relevant information.",
                )
                if compressed and len(compressed.strip()) < len(source_text):
                    r = {**chunk, "metadata": {**chunk.get("metadata", {})}}
                    if "parent_text" in r["metadata"]:
                        r["metadata"]["parent_text"] = compressed.strip()
                    else:
                        r["text"] = compressed.strip()
                    r["metadata"]["compressed"] = True
                    return r
            except Exception as exc:
                logger.debug("Sentence compression failed for %s: %s", chunk.get("id"), exc)
            return chunk

        with ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as pool:
            return list(pool.map(_compress_one, chunks))

    # ------------------------------------------------------------------
    # LLMLingua strategy (Story 3.2)
    # ------------------------------------------------------------------

    def _llmlingua_compress(
        self,
        query: str,
        chunks: list[dict],
        token_budget: int,
    ) -> tuple[list[dict], str]:
        """Token-level LLMLingua-2 compression.

        Returns ``(compressed_chunks, fallback_reason)``.
        On any error, falls back to sentence-extraction and records the reason.
        """
        try:
            lingua = self._ensure_llmlingua()
        except Exception as exc:
            reason = f"llmlingua_unavailable: {exc}"
            logger.warning("LLMLingua unavailable, falling back to sentence: %s", exc)
            fallback = self._sentence_compress(query, chunks) if self._llm else chunks
            return fallback, reason

        # Determine compression rate from token_budget
        if token_budget and token_budget > 0:
            current = _total_tokens(chunks)
            rate = max(0.1, min(0.95, token_budget / current)) if current else 0.5
        else:
            rate = 0.5  # default 50 % target

        compressed_chunks: list[dict] = []
        for chunk in chunks:
            if chunk.get("is_web"):
                compressed_chunks.append(chunk)
                continue
            source_text = _chunk_text(chunk)
            try:
                out = lingua.compress_prompt(
                    source_text,
                    rate=rate,
                    force_tokens=["\n"],
                    question=query,
                )
                compressed_text = out.get("compressed_prompt", "").strip()
                if compressed_text and len(compressed_text) < len(source_text):
                    r = {**chunk, "metadata": {**chunk.get("metadata", {})}}
                    if "parent_text" in r["metadata"]:
                        r["metadata"]["parent_text"] = compressed_text
                    else:
                        r["text"] = compressed_text
                    r["metadata"]["compressed"] = True
                    r["metadata"]["compression_backend"] = "llmlingua"
                    compressed_chunks.append(r)
                else:
                    compressed_chunks.append(chunk)
            except Exception as exc:
                logger.debug("LLMLingua chunk compression failed for %s: %s", chunk.get("id"), exc)
                compressed_chunks.append(chunk)

        return compressed_chunks, ""

    def _ensure_llmlingua(self) -> Any:
        """Lazy-init the LLMLingua-2 PromptCompressor (soft dependency)."""
        if self._lingua is None:
            from llmlingua import PromptCompressor  # soft dep — axon[llmlingua]

            _local = os.path.isabs(self._llmlingua_model) or os.path.isdir(self._llmlingua_model)
            logger.info(
                "Compression: loading LLMLingua model '%s'%s…",
                self._llmlingua_model,
                " (local)" if _local else "",
            )
            self._lingua = PromptCompressor(
                model_name=self._llmlingua_model,
                use_llmlingua2=True,
                device_map="cpu",
                **({"local_files_only": True} if _local else {}),
            )
        return self._lingua
