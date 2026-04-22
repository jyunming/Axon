from __future__ import annotations

import hashlib
import importlib
import logging
import os
from pathlib import Path
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

        module_name = os.getenv("AXON_RUST_MODULE", "axon.axon_rust")
        try:
            self._module = importlib.import_module(module_name)
            logger.info("Rust bridge loaded module '%s'.", module_name)
        except Exception as exc:
            self._module = None
            self._load_error = str(exc)
            logger.debug("Rust bridge unavailable (%s): %s", module_name, exc)
        return self._module

    def _has(self, name: str) -> bool:
        mod = self._load()
        return bool(mod and callable(getattr(mod, name, None)))

    def _call(self, name: str, *args, **kwargs):
        mod = self._load()
        fn = getattr(mod, name, None) if mod else None
        if not callable(fn):
            return None
        try:
            return fn(**kwargs)
        except TypeError:
            try:
                return fn(*args)
            except (ValueError, RuntimeError) as exc:
                # Specific data-integrity or runtime errors from Rust should be logged loudly
                logger.error("Rust %s critical failure: %s", name, exc)
                return None
            except Exception as exc:
                logger.warning("Rust %s failed; fallback to Python: %s", name, exc)
                return None
        except (ValueError, RuntimeError) as exc:
            logger.error("Rust %s critical failure: %s", name, exc)
            return None
        except Exception as exc:
            logger.warning("Rust %s failed; fallback to Python: %s", name, exc)
            return None

    def py(self) -> Any | None:
        return self._load()

    def is_available(self) -> bool:
        return self._load() is not None

    def can_bm25(self) -> bool:
        return self._has("build_bm25_index") and self._has("search_bm25")

    def build_bm25_index(self, corpus: list[dict]) -> Any | None:
        return self._call("build_bm25_index", corpus, corpus=corpus)

    def search_bm25(self, index: Any, query: str, top_k: int) -> list[dict[str, Any]] | None:
        raw = self._call("search_bm25", index, query, top_k, index=index, query=query, top_k=top_k)
        if not isinstance(raw, list):
            return [] if raw is not None else None
        out: list[dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                idx = item.get("index", item.get("doc_index"))
                score = item.get("score", 0.0)
            elif isinstance(item, tuple | list) and len(item) >= 2:
                idx, score = item[0], item[1]
            else:
                continue
            if isinstance(idx, int):
                try:
                    out.append({"index": idx, "score": float(score)})
                except (TypeError, ValueError):
                    continue
        return out

    def can_ingest_preprocess(self) -> bool:
        return self._has("preprocess_documents")

    def preprocess_documents(self, documents: list[dict], batch_size: int) -> list[dict] | None:
        out = self._call(
            "preprocess_documents",
            documents,
            batch_size,
            documents=documents,
            batch_size=batch_size,
        )
        return out if isinstance(out, list) else None

    def can_symbol_search(self) -> bool:
        return self._has("symbol_channel_search")

    def symbol_channel_search(
        self,
        corpora: list[list[dict]],
        query_tokens: list[str],
        top_k: int,
        filters: dict | None,
    ) -> list[dict] | None:
        raw = self._call(
            "symbol_channel_search",
            corpora,
            query_tokens,
            top_k,
            filters or {},
            corpora=corpora,
            query_tokens=query_tokens,
            top_k=top_k,
            filters=filters or {},
        )
        return raw if isinstance(raw, list) else None

    def can_symbol_index(self) -> bool:
        return self._has("build_symbol_index") and self._has("search_symbol_index")

    def build_symbol_index(self, corpora: list[list[dict]]) -> Any | None:
        return self._call("build_symbol_index", corpora, corpora=corpora)

    def search_symbol_index(
        self, index: Any, query_tokens: list[str], top_k: int
    ) -> list[dict[str, Any]] | None:
        raw = self._call(
            "search_symbol_index",
            index,
            query_tokens,
            top_k,
            index=index,
            query_tokens=query_tokens,
            top_k=top_k,
        )
        out: list[dict[str, Any]] = []
        if not isinstance(raw, list):
            return out if raw is not None else None
        for item in raw:
            if isinstance(item, dict):
                idx = item.get("index", item.get("doc_index"))
                score = item.get("score", 0.0)
                channel = item.get("channel", "symbol_name")
            elif isinstance(item, tuple | list) and len(item) >= 3:
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

    def can_doc_hash(self) -> bool:
        return self._has("compute_doc_hash") or True

    def compute_doc_hash(self, text: str) -> str:
        out = self._call("compute_doc_hash", text)
        if isinstance(out, str):
            return out
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def can_extract_code_tokens(self) -> bool:
        return self._has("extract_code_query_tokens") or True

    def extract_code_query_tokens(self, query: str) -> frozenset[str] | None:
        out = self._call("extract_code_query_tokens", query)
        if isinstance(out, list):
            return frozenset(str(x) for x in out if isinstance(x, str))
        try:
            from axon.code_retrieval import _extract_code_query_tokens

            return _extract_code_query_tokens(query)
        except Exception:
            return None

    def can_code_lexical_scores(self) -> bool:
        return self._has("code_lexical_scores") or True

    def code_lexical_scores(
        self, results: list[dict], query_tokens: list[str]
    ) -> tuple[list[float], float] | None:
        out = self._call("code_lexical_scores", results, query_tokens)
        if (
            isinstance(out, tuple | list)
            and len(out) == 2
            and isinstance(out[0], list)
            and isinstance(out[1], int | float)
        ):
            return [float(x) for x in out[0]], float(out[1])
        if not query_tokens:
            return [0.0 for _ in results], 0.0
        long_tokens = frozenset(str(t).lower() for t in query_tokens if len(str(t)) >= 4)
        scores: list[float] = []
        for result in results:
            meta = result.get("metadata", {})
            if meta.get("source_class") != "code":
                scores.append(0.0)
                continue
            score = 0.0
            sym_name = (meta.get("symbol_name") or "").lower()
            sym_type = (meta.get("symbol_type") or "").lower()
            file_path = meta.get("file_path") or meta.get("source") or ""
            basename = os.path.splitext(os.path.basename(file_path))[0].lower()
            qualified = f"{basename}.{sym_name}" if sym_name and basename else ""
            text_lower = (result.get("text") or "").lower()
            if sym_name and sym_name in query_tokens:
                score += 1.0
            elif sym_name:
                for tok in long_tokens:
                    if tok in sym_name:
                        score += 0.5
                        break
            if basename and basename in query_tokens:
                score += 0.4
            if qualified and qualified in query_tokens:
                score += 1.0
            text_hits = sum(1 for tok in long_tokens if tok in text_lower)
            score += min(text_hits * 0.08, 0.32)
            if score > 0.0 and sym_type in {"function", "method"}:
                score *= 1.1
            start_line = meta.get("start_line")
            end_line = meta.get("end_line")
            if start_line is not None and end_line is not None:
                span = max(int(end_line) - int(start_line), 1)
                if span <= 30:
                    score += 0.05
                elif span <= 80:
                    score += 0.02
            scores.append(score)
        max_lex = max(scores) if scores else 0.0
        return scores, max_lex

    def can_decode_corpus_json(self) -> bool:
        return self._has("decode_corpus_json")

    def decode_corpus_json(self, raw: bytes):
        return self._call("decode_corpus_json", raw)

    def can_corpus_msgpack(self) -> bool:
        return self._has("encode_corpus_msgpack") and self._has("decode_corpus_msgpack")

    def encode_corpus_msgpack(self, texts: list[str], docs: list[dict]):
        return self._call("encode_corpus_msgpack", texts, docs)

    def decode_corpus_msgpack(self, raw: bytes):
        return self._call("decode_corpus_msgpack", raw)

    def can_sha256(self) -> bool:
        return self._has("compute_sha256") or True

    def compute_sha256(self, text: str) -> str:
        out = self._call("compute_sha256", text)
        if isinstance(out, str):
            return out
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    def can_hash_store_binary(self) -> bool:
        return (self._has("save_hash_store_binary") and self._has("load_hash_store_binary")) or True

    def save_hash_store_binary(self, path: str, hashes) -> bool:
        mod = self._load()
        fn = getattr(mod, "save_hash_store_binary", None) if mod else None
        if callable(fn):
            try:
                fn(path, list(hashes))
                return True  # Rust returns unit — success means no exception
            except Exception as exc:
                logger.debug("save_hash_store_binary Rust failed: %s", exc)
        try:
            Path(path).write_bytes("\n".join(sorted({str(x) for x in hashes})).encode("utf-8"))
            return True
        except Exception:
            return False

    def load_hash_store_binary(self, path: str) -> set[str] | None:
        mod = self._load()
        fn = getattr(mod, "load_hash_store_binary", None) if mod else None
        if not callable(fn):
            try:
                return set(Path(path).read_text("utf-8").splitlines())
            except OSError:
                return None
        try:
            out = fn(path)
        except Exception as exc:
            # Old-format binary files fail on schema upgrades — expected, rebuild at next ingest
            logger.debug("load_hash_store_binary: %s (file will be rebuilt)", exc)
            return None
        if isinstance(out, list):
            return {str(x) for x in out if isinstance(x, str)}
        return None

    def can_sentence_codec(self) -> bool:
        return self._has("encode_sentence_index") and self._has("decode_sentence_index")

    def encode_sentence_index(self, records: dict, chunk_to_sentences: dict):
        return self._call("encode_sentence_index", records, chunk_to_sentences)

    def decode_sentence_index(self, raw: bytes):
        return self._call("decode_sentence_index", raw)

    def encode_sentence_meta(self, ids: list[str], meta: list[dict]):
        return self._call("encode_sentence_meta", ids, meta, ids=ids, meta=meta)

    def decode_sentence_meta(self, raw: bytes):
        return self._call("decode_sentence_meta", raw, raw=raw)

    def can_segment_text(self) -> bool:
        return self._has("segment_text")

    def segment_text(self, text: str, max_tokens: int):
        return self._call("segment_text", text, max_tokens, text=text, max_tokens=max_tokens)

    def can_cosine_similarity(self) -> bool:
        return self._has("cosine_similarity")

    def cosine_similarity(self, a: list[float], b: list[float]):
        out = self._call("cosine_similarity", a, b)
        return float(out) if isinstance(out, int | float) else None

    def can_score_fusion(self) -> bool:
        return self._has("score_fusion_weighted") and self._has("score_fusion_rrf")

    def score_fusion_weighted(
        self, vector_scores: list[float], bm25_scores: list[float], alpha: float
    ):
        return self._call(
            "score_fusion_weighted",
            vector_scores,
            bm25_scores,
            alpha,
            vector_scores=vector_scores,
            bm25_scores=bm25_scores,
            alpha=alpha,
        )

    def score_fusion_rrf(self, vector_ranks: list[int], bm25_ranks: list[int], k: int):
        return self._call(
            "score_fusion_rrf",
            vector_ranks,
            bm25_ranks,
            k,
            vector_ranks=vector_ranks,
            bm25_ranks=bm25_ranks,
            k=k,
        )

    def can_mmr_rerank(self) -> bool:
        return self._has("mmr_rerank")

    def mmr_rerank(self, results: list[dict], lambda_mult: float, diversity_bias: float):
        return self._call(
            "mmr_rerank",
            results,
            lambda_mult,
            diversity_bias,
            results=results,
            lambda_mult=lambda_mult,
            diversity_bias=diversity_bias,
        )

    def can_result_postprocess(self) -> bool:
        return self._has("dedupe_best_by_id") and self._has("filter_results_by_threshold")

    def dedupe_best_by_id(self, results: list[dict]) -> list[dict[str, Any]] | None:
        out = self._call("dedupe_best_by_id", results, results=results)
        return out if isinstance(out, list) else None

    def filter_results_by_threshold(
        self,
        results: list[dict],
        threshold: float,
        score_field: str = "vector_score",
    ) -> list[dict[str, Any]] | None:
        out = self._call(
            "filter_results_by_threshold",
            results,
            threshold,
            score_field,
            results=results,
            threshold=threshold,
            score_field=score_field,
        )
        return out if isinstance(out, list) else None

    def can_code_doc_bridge(self) -> bool:
        return self._has("build_code_doc_bridge_edges")

    def build_code_doc_bridge_edges(
        self, symbol_lookup: dict, chunks: list[dict], relations: list[dict]
    ):
        return self._call(
            "build_code_doc_bridge_edges",
            symbol_lookup,
            chunks,
            relations,
            symbol_lookup=symbol_lookup,
            chunks=chunks,
            relations=relations,
        )

    def can_entity_graph_codec(self) -> bool:
        return self._has("encode_entity_graph") and self._has("decode_entity_graph")

    def encode_entity_graph(self, graph: dict):
        return self._call("encode_entity_graph", graph)

    def decode_entity_graph(self, raw: bytes):
        return self._call("decode_entity_graph", raw)

    def can_entity_embeddings_codec(self) -> bool:
        return self._has("encode_entity_embeddings") and self._has("decode_entity_embeddings")

    def encode_entity_embeddings(self, embeddings: dict):
        return self._call("encode_entity_embeddings", embeddings)

    def decode_entity_embeddings(self, raw: bytes):
        return self._call("decode_entity_embeddings", raw)

    def can_relation_graph_codec(self) -> bool:
        return self._has("encode_relation_graph") and self._has("decode_relation_graph")

    def encode_relation_graph(self, graph: dict):
        return self._call("encode_relation_graph", graph)

    def decode_relation_graph(self, raw: bytes):
        return self._call("decode_relation_graph", raw)

    def can_dedup_corpus_payload(self) -> bool:
        return self._has("build_dedup_corpus_payload")

    def build_dedup_corpus_payload(self, corpus: list[dict]):
        return self._call("build_dedup_corpus_payload", corpus, corpus=corpus)

    def can_build_graph_edges(self) -> bool:
        return self._has("build_graph_edges")

    def build_graph_edges(
        self, entity_graph: dict, relation_graph: dict
    ) -> tuple[list[str], list[tuple[str, str, float]]] | None:
        out = self._call(
            "build_graph_edges",
            entity_graph,
            relation_graph,
            entity_graph=entity_graph,
            relation_graph=relation_graph,
        )
        if isinstance(out, tuple | list) and len(out) == 2:
            nodes = out[0] if isinstance(out[0], list) else []
            edges = out[1] if isinstance(out[1], list) else []
            return nodes, edges
        return None

    def can_run_louvain(self) -> bool:
        return self._has("run_louvain")

    def run_louvain(
        self,
        nodes: list[str],
        edges: list[tuple[str, str, float]],
        resolution: float = 1.0,
    ) -> dict[str, int] | None:
        out = self._call(
            "run_louvain",
            nodes,
            edges,
            resolution,
            nodes=nodes,
            edges=edges,
            resolution=resolution,
        )
        if isinstance(out, dict):
            return {str(k): int(v) for k, v in out.items()}
        return None

    def can_merge_entities_into_graph(self) -> bool:
        return self._has("merge_entities_into_graph")

    def merge_entities_into_graph(self, entity_graph: dict, results: list) -> int | None:
        out = self._call(
            "merge_entities_into_graph",
            entity_graph,
            results,
            entity_graph=entity_graph,
            results=results,
        )
        if isinstance(out, int):
            return out
        inserted = 0
        for doc_id, entities in results:
            for ent in entities:
                if not isinstance(ent, dict) or not ent.get("name"):
                    continue
                key = str(ent["name"]).lower().strip()
                if not key:
                    continue
                node = entity_graph.get(key)
                if not isinstance(node, dict):
                    node = {
                        "description": ent.get("description", "") if isinstance(ent, dict) else "",
                        "type": ent.get("type", "UNKNOWN") if isinstance(ent, dict) else "UNKNOWN",
                        "chunk_ids": [],
                        "frequency": 0,
                        "degree": 0,
                    }
                    entity_graph[key] = node
                    inserted += 1
                if not node.get("description") and ent.get("description"):
                    node["description"] = ent["description"]
                if (not node.get("type") or node.get("type") == "UNKNOWN") and ent.get("type"):
                    node["type"] = ent["type"]
                node.setdefault("chunk_ids", [])
                if doc_id not in node["chunk_ids"]:
                    node["chunk_ids"].append(doc_id)
                node["frequency"] = len(node.get("chunk_ids", []))
        return inserted

    def can_entity_merge(self) -> bool:
        return self._has("merge_entities_into_graph")

    def can_relation_merge(self) -> bool:
        return self._has("merge_relations_into_graph")

    def merge_relations_into_graph(self, relation_graph: dict, results: list) -> int | None:
        out = self._call(
            "merge_relations_into_graph",
            relation_graph,
            results,
            relation_graph=relation_graph,
            results=results,
        )
        return int(out) if isinstance(out, int) else None

    def can_resolve_entity_alias_groups(self) -> bool:
        return self._has("resolve_entity_alias_groups")

    def resolve_entity_alias_groups(self, embeddings: list[list[float]], threshold: float):
        out = self._call(
            "resolve_entity_alias_groups",
            embeddings,
            threshold,
            embeddings=embeddings,
            threshold=threshold,
        )
        if not isinstance(out, list):
            return None
        groups: list[list[int]] = []
        for item in out:
            if not isinstance(item, list):
                continue
            try:
                groups.append([int(v) for v in item])
            except (TypeError, ValueError):
                continue
        return groups


_SINGLETON = RustBridge()


def get_rust_bridge() -> RustBridge:
    return _SINGLETON
