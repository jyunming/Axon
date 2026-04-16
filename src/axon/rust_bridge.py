from __future__ import annotations

import importlib
import logging
import os
from typing import Any

logger = logging.getLogger("Axon.RustBridge")


class RustBridge:
    """Optional runtime bridge to Rust acceleration module."""

    def __init__(self) -> None:
        self._attempted = False
        self._module: Any | None = None
        self._load_error: str | None = None

    def _load(self) -> Any | None:
        if self._attempted:
            return self._module
        self._attempted = True

        module_name = os.getenv("AXON_RUST_MODULE", "axon_rust")
        try:
            self._module = importlib.import_module(module_name)
            logger.info("Rust bridge loaded module '%s'.", module_name)
        except Exception as exc:
            self._module = None
            self._load_error = str(exc)
            logger.debug("Rust bridge unavailable (%s): %s", module_name, exc)
        return self._module

    def is_available(self) -> bool:
        return self._load() is not None

    def can_bm25(self) -> bool:
        mod = self._load()
        return bool(mod and callable(getattr(mod, "build_bm25_index", None)))

    def can_ingest_preprocess(self) -> bool:
        mod = self._load()
        return bool(mod and callable(getattr(mod, "preprocess_documents", None)))

    def can_symbol_search(self) -> bool:
        mod = self._load()
        return bool(mod and callable(getattr(mod, "symbol_channel_search", None)))

    def can_symbol_index(self) -> bool:
        mod = self._load()
        return bool(
            mod
            and callable(getattr(mod, "build_symbol_index", None))
            and callable(getattr(mod, "search_symbol_index", None))
        )

    def preprocess_documents(self, documents: list[dict], batch_size: int) -> list[dict] | None:
        mod = self._load()
        fn = getattr(mod, "preprocess_documents", None) if mod else None
        if not callable(fn):
            return None
        try:
            out = fn(documents=documents, batch_size=batch_size)
        except TypeError:
            out = fn(documents, batch_size)
        except Exception as exc:
            logger.warning("Rust preprocess failed; fallback to Python: %s", exc)
            return None
        return out if isinstance(out, list) else None

    def build_bm25_index(self, corpus: list[dict]) -> Any | None:
        mod = self._load()
        fn = getattr(mod, "build_bm25_index", None) if mod else None
        if not callable(fn):
            return None
        try:
            return fn(corpus=corpus)
        except TypeError:
            try:
                return fn(corpus)
            except Exception as exc:
                logger.warning("Rust BM25 build failed; fallback to Python: %s", exc)
                return None
        except Exception as exc:
            logger.warning("Rust BM25 build failed; fallback to Python: %s", exc)
            return None

    def search_bm25(self, index: Any, query: str, top_k: int) -> list[dict[str, Any]] | None:
        mod = self._load()
        fn = getattr(mod, "search_bm25", None) if mod else None
        if not callable(fn):
            return None
        try:
            raw = fn(index=index, query=query, top_k=top_k)
        except TypeError:
            try:
                raw = fn(index, query, top_k)
            except Exception as exc:
                logger.warning("Rust BM25 search failed; fallback to Python: %s", exc)
                return None
        except Exception as exc:
            logger.warning("Rust BM25 search failed; fallback to Python: %s", exc)
            return None

        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                idx = item.get("index", item.get("doc_index"))
                score = item.get("score", 0.0)
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                idx, score = item[0], item[1]
            else:
                continue
            if isinstance(idx, int):
                try:
                    out.append({"index": idx, "score": float(score)})
                except (TypeError, ValueError):
                    continue
        return out

    def symbol_channel_search(
        self,
        corpora: list[list[dict]],
        query_tokens: list[str],
        top_k: int,
        filters: dict | None,
    ) -> list[dict] | None:
        mod = self._load()
        fn = getattr(mod, "symbol_channel_search", None) if mod else None
        if not callable(fn):
            return None
        try:
            raw = fn(
                corpora=corpora,
                query_tokens=query_tokens,
                top_k=top_k,
                filters=filters or {},
            )
        except TypeError:
            try:
                raw = fn(corpora, query_tokens, top_k, filters or {})
            except Exception as exc:
                logger.warning("Rust symbol search failed; fallback to Python: %s", exc)
                return None
        except Exception as exc:
            logger.warning("Rust symbol search failed; fallback to Python: %s", exc)
            return None
        return raw if isinstance(raw, list) else None

    def build_symbol_index(self, corpora: list[list[dict]]) -> Any | None:
        mod = self._load()
        fn = getattr(mod, "build_symbol_index", None) if mod else None
        if not callable(fn):
            return None
        try:
            return fn(corpora=corpora)
        except TypeError:
            try:
                return fn(corpora)
            except Exception as exc:
                logger.warning("Rust symbol index build failed; fallback to scan: %s", exc)
                return None
        except Exception as exc:
            logger.warning("Rust symbol index build failed; fallback to scan: %s", exc)
            return None

    def search_symbol_index(
        self, index: Any, query_tokens: list[str], top_k: int
    ) -> list[dict[str, Any]] | None:
        mod = self._load()
        fn = getattr(mod, "search_symbol_index", None) if mod else None
        if not callable(fn):
            return None
        try:
            raw = fn(index=index, query_tokens=query_tokens, top_k=top_k)
        except TypeError:
            try:
                raw = fn(index, query_tokens, top_k)
            except Exception as exc:
                logger.warning("Rust symbol index search failed; fallback to scan: %s", exc)
                return None
        except Exception as exc:
            logger.warning("Rust symbol index search failed; fallback to scan: %s", exc)
            return None

        out: list[dict[str, Any]] = []
        if not isinstance(raw, list):
            return out
        for item in raw:
            if isinstance(item, dict):
                idx = item.get("index", item.get("doc_index"))
                score = item.get("score", 0.0)
                channel = item.get("channel", "symbol_name")
            elif isinstance(item, (tuple, list)) and len(item) >= 3:
                idx, score, channel = item[0], item[1], item[2]
            else:
                continue
            if not isinstance(idx, int):
                continue
            try:
                out.append(
                    {"index": idx, "score": float(score), "channel": str(channel or "symbol_name")}
                )
            except (TypeError, ValueError):
                continue
        return out


_SINGLETON = RustBridge()


def get_rust_bridge() -> RustBridge:
    return _SINGLETON
