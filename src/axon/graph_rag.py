"""GraphRAG entity/community graph management and retrieval (GraphRagMixin)."""
from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger("Axon")

_GRAPHRAG_REDUCE_SYSTEM_PROMPT = (
    "You are a helpful assistant responding to questions about a dataset by synthesizing the "
    "perspectives from multiple data analysts.\n\n"
    "Generate a response of the target length and format that responds to the user's question, "
    "summarize all the reports from multiple analysts who focused on different parts of the "
    "dataset.\n\n"
    "Note that the analysts' reports provided below are ranked in the importance of addressing "
    "the user's question.\n\n"
    "If you don't know the answer or if the input reports do not contain sufficient information "
    "to provide an answer, just say so. Do not make anything up.\n\n"
    "The response should preserve the original meaning and use of modal verbs such as 'shall', "
    "'may' or 'will'.\n\n"
    "Points supported by data should list their data references as follows:\n"
    '"This is an example sentence supported by multiple data references [Data: Reports (report1), (report2)]."\n\n'
    "**Do not list more than 5 data references in a single reference**. Instead, list the top 5 "
    'most relevant references and add "+more" to indicate that there are more.\n\n'
    "Everything needs to be explained at length and in detail, with specific quotes and passages "
    "from the data providing evidence to back up each and every claim."
)

_GRAPHRAG_NO_DATA_ANSWER = (
    "I am sorry but I am unable to answer this question given the provided data."
)


class GraphRagMixin:
    # Share heavyweight local model instances across threads and brain instances.
    _shared_gliner_models: dict[tuple[str, bool], object] = {}
    _shared_rebel_pipelines: dict[tuple[str, bool], object] = {}
    _shared_gliner_lock = threading.Lock()
    _shared_rebel_lock = threading.Lock()
    _GR_CS_KEY_COMPACT = {
        "title": "t",
        "summary": "s",
        "full_content": "f",
        "rank": "r",
        "level": "l",
        "member_hash": "mh",
        "indexed_hash": "ih",
        "findings": "fd",
        "rating": "rt",
        "rating_explanation": "re",
    }
    _GR_CS_KEY_EXPAND = {v: k for k, v in _GR_CS_KEY_COMPACT.items()}

    def _gr_profile_enabled(self) -> bool:
        return bool(getattr(self.config, "graph_rag_profile", False))

    def _gr_log_profile(self, section: str, elapsed_s: float, **extra) -> None:
        if not self._gr_profile_enabled():
            return
        suffix = " ".join(f"{k}={v}" for k, v in extra.items())
        logger.info(
            "GraphRAG profile: %s %.2fms%s",
            section,
            elapsed_s * 1000.0,
            f" {suffix}" if suffix else "",
        )

    def _gr_cache_get(self, bucket: str, key: str):
        store = getattr(self, "_graph_rag_cache", None)
        if not isinstance(store, dict):
            store = {}
            self._graph_rag_cache = store
        return store.get(bucket, {}).get(key)

    def _gr_cache_put(self, bucket: str, key: str, value) -> None:
        if bucket == "global_answer":
            if not getattr(self.config, "graph_rag_global_answer_cache", True):
                return
            cap = int(getattr(self.config, "graph_rag_global_answer_cache_size", 500))
            if cap <= 0:
                return
        elif bucket == "global_map":
            if not getattr(self.config, "graph_rag_global_map_cache", True):
                return
            cap = int(getattr(self.config, "graph_rag_global_map_cache_size", 2000))
            if cap <= 0:
                return
        else:
            cap = 0
        _is_llm_bucket = bucket.startswith("llm:")
        if cap == 0 and _is_llm_bucket:
            if not getattr(self.config, "graph_rag_llm_cache", True):
                return
            cap = int(getattr(self.config, "graph_rag_llm_cache_size", 2000))
        elif cap == 0:
            if not getattr(self.config, "graph_rag_extraction_cache", True):
                return
            cap = int(getattr(self.config, "graph_rag_extraction_cache_size", 5000))
        if cap <= 0:
            return
        store = getattr(self, "_graph_rag_cache", None)
        if not isinstance(store, dict):
            store = {}
            self._graph_rag_cache = store
        bucket_map = store.setdefault(bucket, {})
        bucket_map[key] = value
        if len(bucket_map) > cap:
            # dict is insertion ordered; pop oldest inserted
            oldest = next(iter(bucket_map))
            bucket_map.pop(oldest, None)

    def _gr_text_hash(self, text: str) -> str:
        import hashlib as _hashlib

        return _hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()

    def _gr_llm_complete_cached(
        self,
        bucket: str,
        prompt: str,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """LLM completion with semantic response cache keyed by prompt+options."""
        if not getattr(self.config, "graph_rag_llm_cache", True):
            return self.llm.complete(prompt, system_prompt=system_prompt, **kwargs)
        _llm_name = getattr(getattr(self, "llm", None), "__class__", type("x", (), {})).__name__
        _kwargs_key = "|".join(f"{k}={kwargs[k]}" for k in sorted(kwargs.keys()))
        _key = self._gr_text_hash(
            f"{_llm_name}|{bucket}|{system_prompt or ''}|{_kwargs_key}|{prompt}"
        )
        _cache_bucket = f"llm:{bucket}"
        _cached = self._gr_cache_get(_cache_bucket, _key)
        if _cached is not None:
            return _cached
        _out = self.llm.complete(prompt, system_prompt=system_prompt, **kwargs)
        self._gr_cache_put(_cache_bucket, _key, _out)
        return _out

    def _gr_write_json_if_changed(self, path, payload, *, sort_keys: bool = False) -> bool:
        """Atomically write JSON only when content changed. Returns True if written."""
        import hashlib as _hashlib
        import json as _json
        import os as _os
        import pathlib as _pathlib

        p = _pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = _json.dumps(payload, sort_keys=sort_keys, separators=(",", ":"))
        digest = _hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
        cache = getattr(self, "_gr_persist_hashes", None)
        if not isinstance(cache, dict):
            cache = {}
            self._gr_persist_hashes = cache
        p_key = str(p)
        if cache.get(p_key) == digest and p.exists():
            return False
        if p.exists() and cache.get(p_key) is None:
            try:
                existing = p.read_text(encoding="utf-8")
                existing_digest = _hashlib.sha1(
                    existing.encode("utf-8", errors="replace")
                ).hexdigest()
                cache[p_key] = existing_digest
                if existing_digest == digest:
                    return False
            except Exception:
                pass
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        _os.replace(tmp, p)
        cache[p_key] = digest
        return True

    def _gr_json_load_path(self, path):
        """Load JSON from path using orjson when available, else stdlib json."""
        import json as _json
        import os as _os
        import pathlib as _pathlib

        p = _pathlib.Path(path)
        raw = p.read_bytes()
        _prefer_orjson = _os.getenv("AXON_GRAPHRAG_ORJSON_LOAD", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not _prefer_orjson:
            return _json.loads(raw.decode("utf-8"))
        _orjson = getattr(self, "_gr_orjson", None)
        if _orjson is None:
            try:
                import orjson as _orjson_mod  # type: ignore

                _orjson = _orjson_mod
            except Exception:
                _orjson = False
            self._gr_orjson = _orjson
        if _orjson and _orjson is not False:
            return _orjson.loads(raw)
        return _json.loads(raw.decode("utf-8"))

    def _get_incoming_relation_index(self) -> dict[str, list]:
        if not getattr(self.config, "graph_rag_local_cached_incoming", True):
            return {}
        rel = self._relation_graph
        sig = (len(rel), sum(len(v) for v in rel.values()))
        if getattr(self, "_incoming_rel_sig", None) == sig and hasattr(self, "_incoming_rel_index"):
            return self._incoming_rel_index
        idx: dict[str, list] = {}
        for src, entries in rel.items():
            for entry in entries:
                tgt = (entry.get("target", "") or "").lower()
                if not tgt:
                    continue
                # Store (source, original_entry_ref) to avoid duplicating relation dicts.
                idx.setdefault(tgt, []).append((src, entry))
        self._incoming_rel_sig = sig
        self._incoming_rel_index = idx
        return idx

    def _get_incoming_relation_count_map(self) -> dict[str, int]:
        """Return cached incoming edge counts per entity keyed to relation-graph signature."""
        if not bool(getattr(self.config, "graph_rag_local_cached_incoming_counts", True)):
            return {}
        idx = self._get_incoming_relation_index()
        if not idx:
            return {}
        sig = getattr(self, "_incoming_rel_sig", None)
        if sig is None:
            sig = (len(self._relation_graph), sum(len(v) for v in self._relation_graph.values()))
        if getattr(self, "_incoming_count_sig", None) == sig and hasattr(
            self, "_incoming_count_map"
        ):
            return self._incoming_count_map
        cmap = {k: len(v) for k, v in idx.items()}
        self._incoming_count_sig = sig
        self._incoming_count_map = cmap
        return cmap

    def _load_entity_graph(self) -> dict:
        """Load persisted entity→doc_id graph from disk.

        Shape: {entity_lower: {"description": str, "chunk_ids": list[str]}}
        """
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_graph.json"
        if path.exists():
            try:
                raw = self._gr_json_load_path(path)
                if not isinstance(raw, dict):
                    return {}
                cleaned: dict = {}
                for key, value in raw.items():
                    if not isinstance(key, str):
                        continue
                    if isinstance(value, dict) and "chunk_ids" in value:
                        node = value
                        node.setdefault("type", "UNKNOWN")
                        node.setdefault("frequency", len(node.get("chunk_ids", [])))
                        node.setdefault("degree", 0)
                        cleaned[key] = node
                    # Otherwise skip malformed entries
                return cleaned
            except Exception:
                pass
        return {}

    def _save_entity_graph(self) -> None:
        """Persist entity graph to disk."""
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_graph.json"
        self._gr_write_json_if_changed(path, self._entity_graph)

    def _load_code_graph(self) -> dict:
        """Load code graph from disk. Returns empty graph if not found."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".code_graph.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "nodes" in data and "edges" in data:
                    return data
            except Exception:
                pass
        return {"nodes": {}, "edges": []}

    def _save_code_graph(self) -> None:
        """Persist code graph to disk."""
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".code_graph.json"
        self._gr_write_json_if_changed(path, self._code_graph)

    def _load_relation_graph(self) -> dict:
        """Load persisted relation graph from disk.

        Shape: {source_entity_lower: [{target: str, relation: str, chunk_id: str}]}
        """
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".relation_graph.json"
        shards_manifest = pathlib.Path(self.config.bm25_path) / ".relation_graph.shards.json"
        shards_list_manifest = pathlib.Path(self.config.bm25_path) / ".relation_graph.shards.lst"
        shard_state_path = pathlib.Path(self.config.bm25_path) / ".relation_graph.shard_state.json"
        shard_cache_path = pathlib.Path(self.config.bm25_path) / ".relation_graph.cache.pkl"
        shard_cache_meta_path = (
            pathlib.Path(self.config.bm25_path) / ".relation_graph.cache.meta.json"
        )
        if shards_manifest.exists():
            try:
                cache_key = None
                if bool(getattr(self.config, "graph_rag_relation_pickle_cache", False)):
                    try:
                        if shard_state_path.exists():
                            _state = self._gr_json_load_path(shard_state_path)
                            if isinstance(_state, dict) and isinstance(
                                _state.get("signatures"), list
                            ):
                                cache_key = self._gr_text_hash(
                                    "|".join(str(x) for x in _state.get("signatures", []))
                                )
                    except Exception:
                        cache_key = None
                    if cache_key and shard_cache_path.exists() and shard_cache_meta_path.exists():
                        try:
                            _meta = self._gr_json_load_path(shard_cache_meta_path)
                            if isinstance(_meta, dict) and _meta.get("key") == cache_key:
                                import pickle as _pickle

                                with open(shard_cache_path, "rb") as _f:
                                    _cached = _pickle.load(_f)
                                if isinstance(_cached, dict):
                                    return _cached
                        except Exception:
                            pass
                shard_names: list[str] = []
                try:
                    sig = (
                        shards_manifest.stat().st_mtime_ns if shards_manifest.exists() else 0,
                        shards_list_manifest.stat().st_mtime_ns
                        if shards_list_manifest.exists()
                        else 0,
                    )
                except Exception:
                    sig = None
                if (
                    sig is not None
                    and getattr(self, "_rel_shard_names_sig", None) == sig
                    and isinstance(getattr(self, "_rel_shard_names_cache", None), list)
                ):
                    shard_names = list(self._rel_shard_names_cache)
                else:
                    if (
                        bool(getattr(self.config, "graph_rag_relation_shard_list_manifest", True))
                        and shards_list_manifest.exists()
                    ):
                        shard_names = [
                            ln.strip()
                            for ln in shards_list_manifest.read_text(encoding="utf-8").splitlines()
                            if ln.strip()
                        ]
                    if not shard_names:
                        raw_manifest = self._gr_json_load_path(shards_manifest)
                        if (
                            isinstance(raw_manifest, dict)
                            and raw_manifest.get("format") == "rg_rel_shard_v1"
                            and isinstance(raw_manifest.get("shards"), list)
                        ):
                            shard_names = [x for x in raw_manifest["shards"] if isinstance(x, str)]
                    if sig is not None:
                        self._rel_shard_names_sig = sig
                        self._rel_shard_names_cache = list(shard_names)
                if shard_names:
                    merged: dict = {}

                    def _decode_shard(shard_name: str) -> dict:
                        shard_path = pathlib.Path(self.config.bm25_path) / shard_name
                        if not shard_path.exists():
                            return {}
                        shard_raw = self._gr_json_load_path(shard_path)
                        if not isinstance(shard_raw, dict):
                            return {}
                        out: dict = {}
                        if shard_raw.get("format") == "rg_rel_v2" and isinstance(
                            shard_raw.get("g"), dict
                        ):
                            for key, value in shard_raw["g"].items():
                                if not isinstance(key, str) or not isinstance(value, list):
                                    continue
                                out_entries = []
                                for entry in value:
                                    if (
                                        isinstance(entry, list | tuple)
                                        and len(entry) >= 3
                                        and isinstance(entry[0], str)
                                        and isinstance(entry[1], str)
                                        and isinstance(entry[2], str)
                                    ):
                                        out_entries.append(
                                            {
                                                "target": entry[0],
                                                "relation": entry[1],
                                                "chunk_id": entry[2],
                                            }
                                        )
                                if out_entries:
                                    out.setdefault(key, []).extend(out_entries)
                        else:
                            for key, value in shard_raw.items():
                                if not isinstance(key, str) or not isinstance(value, list):
                                    continue
                                out_entries = [
                                    entry
                                    for entry in value
                                    if isinstance(entry, dict)
                                    and "target" in entry
                                    and "relation" in entry
                                    and "chunk_id" in entry
                                ]
                                if out_entries:
                                    out.setdefault(key, []).extend(out_entries)
                        return out

                    use_parallel = (
                        bool(getattr(self.config, "graph_rag_relation_shard_parallel_load", True))
                        and len(shard_names) > 1
                    )
                    if use_parallel:
                        from concurrent.futures import ThreadPoolExecutor

                        workers = min(
                            int(
                                getattr(self.config, "graph_rag_relation_shard_load_workers", 4)
                                or 4
                            ),
                            len(shard_names),
                        )
                        workers = max(1, workers)
                        with ThreadPoolExecutor(max_workers=workers) as ex:
                            for part in ex.map(_decode_shard, shard_names):
                                for key, value in part.items():
                                    merged.setdefault(key, []).extend(value)
                    else:
                        for n in shard_names:
                            part = _decode_shard(n)
                            for key, value in part.items():
                                merged.setdefault(key, []).extend(value)
                    if (
                        bool(getattr(self.config, "graph_rag_relation_pickle_cache", False))
                        and cache_key
                    ):
                        try:
                            import json as _json
                            import os as _os
                            import pickle as _pickle

                            proto = int(
                                getattr(self.config, "graph_rag_relation_pickle_cache_protocol", 4)
                                or 4
                            )
                            proto = min(max(1, proto), 5)
                            _tmp = shard_cache_path.with_suffix(shard_cache_path.suffix + ".tmp")
                            with open(_tmp, "wb") as _f:
                                _pickle.dump(merged, _f, protocol=proto)
                            _os.replace(_tmp, shard_cache_path)
                            _tmp_meta = shard_cache_meta_path.with_suffix(
                                shard_cache_meta_path.suffix + ".tmp"
                            )
                            _tmp_meta.write_text(_json.dumps({"key": cache_key}), encoding="utf-8")
                            _os.replace(_tmp_meta, shard_cache_meta_path)
                        except Exception:
                            pass
                    return merged
            except Exception:
                pass
        if path.exists():
            try:
                raw = self._gr_json_load_path(path)
                if not isinstance(raw, dict):
                    return {}
                # Compact format: {"format":"rg_rel_v2","g": {src:[[tgt,rel,cid],...]}}
                if raw.get("format") == "rg_rel_v2" and isinstance(raw.get("g"), dict):
                    graph_raw = raw.get("g", {})
                    cleaned_v2: dict = {}
                    for key, value in graph_raw.items():
                        if not isinstance(key, str) or not isinstance(value, list):
                            continue
                        out_entries = []
                        for entry in value:
                            if (
                                isinstance(entry, list | tuple)
                                and len(entry) >= 3
                                and isinstance(entry[0], str)
                                and isinstance(entry[1], str)
                                and isinstance(entry[2], str)
                            ):
                                out_entries.append(
                                    {
                                        "target": entry[0],
                                        "relation": entry[1],
                                        "chunk_id": entry[2],
                                    }
                                )
                        if out_entries:
                            cleaned_v2[key] = out_entries
                    return cleaned_v2
                cleaned: dict = {}
                for key, value in raw.items():
                    if not isinstance(key, str) or not isinstance(value, list):
                        continue
                    cleaned[key] = [
                        entry
                        for entry in value
                        if isinstance(entry, dict)
                        and "target" in entry
                        and "relation" in entry
                        and "chunk_id" in entry
                    ]
                return cleaned
            except Exception:
                pass
        return {}

    def _save_relation_graph(self) -> None:
        """Persist relation graph to disk."""
        import hashlib
        import os
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".relation_graph.json"
        if bool(getattr(self.config, "graph_rag_relation_shard_persist", False)):
            shard_count = int(getattr(self.config, "graph_rag_relation_shard_count", 16) or 16)
            shard_count = max(1, shard_count)
            src_keys = sorted(k for k in self._relation_graph.keys() if isinstance(k, str))
            buckets: list[dict] = [{} for _ in range(shard_count)]
            for i, src in enumerate(src_keys):
                buckets[i % shard_count][src] = self._relation_graph[src]
            state_path = pathlib.Path(self.config.bm25_path) / ".relation_graph.shard_state.json"
            prev_state = {}
            try:
                if state_path.exists():
                    _raw_state = self._gr_json_load_path(state_path)
                    if isinstance(_raw_state, dict):
                        prev_state = _raw_state
            except Exception:
                prev_state = {}
            prev_sigs = prev_state.get("signatures", []) if isinstance(prev_state, dict) else []
            prev_compact = (
                bool(prev_state.get("compact", False)) if isinstance(prev_state, dict) else False
            )
            prev_shard_count = (
                int(prev_state.get("shard_count", 0)) if isinstance(prev_state, dict) else 0
            )
            compact_mode = bool(getattr(self.config, "graph_rag_relation_compact_persist", True))
            selective = bool(
                getattr(self.config, "graph_rag_relation_shard_selective_rewrite", True)
            )

            def _bucket_sig(bucket: dict) -> str:
                h = hashlib.sha1()
                for src in sorted(bucket.keys()):
                    h.update(src.encode("utf-8", errors="replace"))
                    entries = bucket.get(src, [])
                    h.update(str(len(entries)).encode("utf-8"))
                    for e in entries:
                        if not isinstance(e, dict):
                            continue
                        h.update(str(e.get("target", "")).encode("utf-8", errors="replace"))
                        h.update(str(e.get("relation", "")).encode("utf-8", errors="replace"))
                        h.update(str(e.get("chunk_id", "")).encode("utf-8", errors="replace"))
                return h.hexdigest()

            if (
                bool(getattr(self.config, "graph_rag_relation_shard_parallel_signatures", True))
                and len(buckets) > 1
            ):
                from concurrent.futures import ThreadPoolExecutor

                workers = min(
                    int(getattr(self.config, "graph_rag_relation_shard_signature_workers", 4) or 4),
                    len(buckets),
                )
                workers = max(1, workers)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    curr_sigs = list(ex.map(_bucket_sig, buckets))
            else:
                curr_sigs = [_bucket_sig(b) for b in buckets]
            shard_names: list[str] = []
            shard_tasks: list[tuple[str, object]] = []
            for i, bucket in enumerate(buckets):
                shard_name = f".relation_graph.shard.{i:03d}.json"
                shard_path = pathlib.Path(self.config.bm25_path) / shard_name
                shard_names.append(shard_name)
                if (
                    selective
                    and isinstance(prev_sigs, list)
                    and i < len(prev_sigs)
                    and prev_sigs[i] == curr_sigs[i]
                    and prev_compact == compact_mode
                    and prev_shard_count == shard_count
                    and shard_path.exists()
                ):
                    continue
                if compact_mode:
                    compact_graph = {}
                    for src, entries in bucket.items():
                        compact_entries = []
                        for e in entries:
                            if not isinstance(e, dict):
                                continue
                            tgt = e.get("target")
                            rel = e.get("relation")
                            cid = e.get("chunk_id")
                            if (
                                isinstance(tgt, str)
                                and isinstance(rel, str)
                                and isinstance(cid, str)
                            ):
                                compact_entries.append([tgt, rel, cid])
                        if compact_entries:
                            compact_graph[src] = compact_entries
                    payload = {"format": "rg_rel_v2", "g": compact_graph}
                else:
                    payload = bucket
                shard_tasks.append((str(shard_path), payload))
            if (
                bool(getattr(self.config, "graph_rag_relation_shard_parallel_writes", True))
                and len(shard_tasks) > 1
            ):
                from concurrent.futures import ThreadPoolExecutor

                workers = min(
                    int(getattr(self.config, "graph_rag_relation_shard_write_workers", 4) or 4),
                    len(shard_tasks),
                )
                workers = max(1, workers)

                def _write_task(task):
                    p, pl = task
                    self._gr_write_json_if_changed(p, pl)

                with ThreadPoolExecutor(max_workers=workers) as ex:
                    list(ex.map(_write_task, shard_tasks))
            else:
                for p, pl in shard_tasks:
                    self._gr_write_json_if_changed(p, pl)
            manifest = {
                "format": "rg_rel_shard_v1",
                "compact": compact_mode,
                "shards": shard_names,
            }
            manifest_path = pathlib.Path(self.config.bm25_path) / ".relation_graph.shards.json"
            self._gr_write_json_if_changed(manifest_path, manifest, sort_keys=True)
            if bool(getattr(self.config, "graph_rag_relation_shard_list_manifest", True)):
                list_manifest_path = (
                    pathlib.Path(self.config.bm25_path) / ".relation_graph.shards.lst"
                )
                list_payload = "\n".join(shard_names) + ("\n" if shard_names else "")
                try:
                    existing = (
                        list_manifest_path.read_text(encoding="utf-8")
                        if list_manifest_path.exists()
                        else None
                    )
                    if existing != list_payload:
                        tmp = list_manifest_path.with_suffix(list_manifest_path.suffix + ".tmp")
                        tmp.write_text(list_payload, encoding="utf-8")
                        os.replace(tmp, list_manifest_path)
                except Exception:
                    pass
            state_payload = {
                "format": "rg_rel_shard_state_v1",
                "compact": compact_mode,
                "shard_count": shard_count,
                "signatures": curr_sigs,
            }
            self._gr_write_json_if_changed(state_path, state_payload, sort_keys=True)
            # Best effort cleanup of monolithic file to avoid duplicate storage.
            try:
                if path.exists():
                    os.remove(path)
            except OSError:
                pass
            return
        if bool(getattr(self.config, "graph_rag_relation_compact_persist", True)):
            compact_graph = {}
            for src, entries in self._relation_graph.items():
                if not isinstance(src, str) or not isinstance(entries, list):
                    continue
                compact_entries = []
                for e in entries:
                    if not isinstance(e, dict):
                        continue
                    tgt = e.get("target")
                    rel = e.get("relation")
                    cid = e.get("chunk_id")
                    if isinstance(tgt, str) and isinstance(rel, str) and isinstance(cid, str):
                        compact_entries.append([tgt, rel, cid])
                if compact_entries:
                    compact_graph[src] = compact_entries
            payload = {"format": "rg_rel_v2", "g": compact_graph}
            self._gr_write_json_if_changed(path, payload)
        else:
            self._gr_write_json_if_changed(path, self._relation_graph)

    def _load_community_levels(self) -> dict:
        """Load persisted hierarchical community levels from disk.

        Shape: {level_int: {entity_lower: community_id}}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_levels.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return {int(k): v for k, v in raw.items() if isinstance(v, dict)}
        except Exception:
            pass
        return {}

    def _save_community_levels(self) -> None:
        """Persist hierarchical community levels to disk."""
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_levels.json"
        try:
            self._gr_write_json_if_changed(
                path, {str(k): v for k, v in self._community_levels.items()}, sort_keys=True
            )
        except Exception as e:
            logger.debug(f"Could not save community levels: {e}")

    def _load_community_hierarchy(self) -> dict:
        """Load persisted community hierarchy from disk.

        Shape: {cluster_id_int: parent_cluster_id_int_or_None}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_hierarchy.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    # JSON keys are strings; support both legacy int keys and
                    # level-qualified string keys like "1_3".
                    result = {}
                    for k, v in raw.items():
                        key = k if ("_" in str(k)) else (int(k) if str(k).isdigit() else k)
                        val = (
                            None
                            if v is None
                            else (
                                v
                                if (isinstance(v, str) and "_" in v)
                                else (int(str(v)) if str(v).isdigit() else v)
                            )
                        )
                        result[key] = val
                    return result
        except Exception:
            pass
        return {}

    def _save_community_hierarchy(self) -> None:
        """Persist community hierarchy to disk."""
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_hierarchy.json"
        try:
            self._gr_write_json_if_changed(
                path,
                {str(k): v for k, v in self._community_hierarchy.items()},
                sort_keys=True,
            )
        except Exception as e:
            logger.debug(f"Could not save community hierarchy: {e}")

    def _load_community_summaries(self) -> dict:
        """Load persisted community summaries from disk."""
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_summaries.json"
        try:
            if path.exists():
                raw = self._gr_json_load_path(path)
                if isinstance(raw, dict):
                    if raw.get("format") == "gr_cs_v2" and isinstance(raw.get("s"), dict):
                        out = {}
                        for cid, item in raw["s"].items():
                            if not isinstance(cid, str) or not isinstance(item, dict):
                                continue
                            expanded = {}
                            for k, v in item.items():
                                expanded[self._GR_CS_KEY_EXPAND.get(k, k)] = v
                            out[cid] = expanded
                        return out
                    return raw
        except Exception:
            pass
        return {}

    def _save_community_summaries(self) -> None:
        """Persist community summaries to disk."""
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".community_summaries.json"
        try:
            if bool(getattr(self.config, "graph_rag_community_summary_compact_persist", True)):
                compact = {}
                for cid, item in self._community_summaries.items():
                    if not isinstance(cid, str) or not isinstance(item, dict):
                        continue
                    citem = {}
                    for k, v in item.items():
                        citem[self._GR_CS_KEY_COMPACT.get(k, k)] = v
                    compact[cid] = citem
                payload = {"format": "gr_cs_v2", "s": compact}
                self._gr_write_json_if_changed(path, payload, sort_keys=True)
            else:
                self._gr_write_json_if_changed(path, self._community_summaries, sort_keys=True)
        except Exception as e:
            logger.debug(f"Could not save community summaries: {e}")

    def _load_entity_embeddings(self) -> dict:
        """Load persisted entity embeddings from disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_embeddings.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            pass
        return {}

    def _save_entity_embeddings(self) -> None:
        """Persist entity embeddings to disk."""
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".entity_embeddings.json"
        try:
            self._gr_write_json_if_changed(path, self._entity_embeddings)
        except Exception as e:
            logger.debug(f"Could not save entity embeddings: {e}")

    def _load_claims_graph(self) -> dict:
        """Load persisted claims graph from disk.

        Shape: {chunk_id: [claim_dict, ...]}
        """
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".claims_graph.json"
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    return raw
        except Exception:
            pass
        return {}

    def _save_claims_graph(self) -> None:
        """Persist claims graph to disk."""
        import json
        import pathlib

        path = pathlib.Path(self.config.bm25_path) / ".claims_graph.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._claims_graph), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Could not save claims graph: {e}")

    def _build_networkx_graph(self):
        """Build a NetworkX undirected graph from entity and relation data."""
        import networkx as nx

        _min_freq_raw = getattr(self.config, "graph_rag_entity_min_frequency", 1)
        _min_freq = int(_min_freq_raw) if isinstance(_min_freq_raw, int | float) else 1
        G = nx.Graph()
        for entity, node in self._entity_graph.items():
            if isinstance(node, dict) and node.get("frequency", 1) < _min_freq:
                continue
            desc = node.get("description", "") if isinstance(node, dict) else ""
            G.add_node(entity, description=desc)
        for src, entries in self._relation_graph.items():
            for entry in entries:
                tgt = entry.get("target", "")
                if src and tgt:
                    edge_weight = entry.get("weight", 1)
                    if G.has_edge(src, tgt):
                        G[src][tgt]["weight"] += edge_weight
                    else:
                        G.add_edge(
                            src,
                            tgt,
                            weight=edge_weight,
                            relation=entry.get("relation", ""),
                            description=entry.get("description", ""),
                        )
        return G

    def _run_community_detection(self) -> dict:
        """Run Louvain community detection. Returns {entity_lower: community_id}."""
        try:
            import networkx as nx  # noqa: F401 — ensures ImportError fires when networkx is absent
            import networkx.algorithms.community as nx_comm
        except ImportError:
            logger.warning(
                "GraphRAG community detection requires networkx. "
                "Install with: pip install networkx"
            )
            return {}
        G = self._build_networkx_graph()
        if len(G.nodes) < 2:
            return dict.fromkeys(G.nodes, 0)
        try:
            communities = nx_comm.louvain_communities(G, seed=42)
            mapping = {}
            for cid, members in enumerate(communities):
                for entity in members:
                    mapping[entity] = cid
            return mapping
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return {}

    def _run_hierarchical_community_detection(self) -> tuple:
        """Run hierarchical community detection.

        Returns:
            (community_levels, community_hierarchy, community_children)
            - community_levels: {level_int: {entity_lower: cluster_id}}
            - community_hierarchy: {cluster_id: parent_cluster_id}  (None for root)
            - community_children: {cluster_id: [child_cluster_ids]}
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning(
                "GraphRAG community detection requires networkx. "
                "Install with: pip install networkx"
            )
            return {}, {}, {}

        G = self._build_networkx_graph()
        if G.number_of_nodes() < 2:
            nodes = list(G.nodes())
            return {0: dict.fromkeys(nodes, 0)}, {0: None}, {0: []}

        n_levels = max(1, getattr(self.config, "graph_rag_community_levels", 2))
        max_cluster_size = getattr(self.config, "graph_rag_community_max_cluster_size", 10)
        seed = getattr(self.config, "graph_rag_leiden_seed", 42)
        use_lcc = getattr(self.config, "graph_rag_community_use_lcc", True)

        components = list(nx.connected_components(G))
        if use_lcc and components:
            try:
                lcc_nodes = max(components, key=len)
                dropped = G.number_of_nodes() - len(lcc_nodes)
                if dropped > 0:
                    logger.info(
                        f"GraphRAG: use_lcc=True — clustering {len(lcc_nodes)} nodes, "
                        f"dropping {dropped} nodes in {len(components) - 1} smaller components"
                    )
                G = G.subgraph(lcc_nodes).copy()
            except Exception:
                pass
        elif not use_lcc and len(components) > 1:
            logger.debug(
                f"GraphRAG: clustering all {len(components)} connected components "
                f"({G.number_of_nodes()} total nodes)"
            )

        _backend = getattr(self.config, "graph_rag_community_backend", "louvain")

        # Tier-1: graspologic hierarchical Leiden — only attempted when backend="auto"
        # The import is probed inside a thread with a 10-second timeout to avoid hanging the
        # main process on Python 3.13+ where graspologic's numba/scipy initialisation can block.
        if _backend == "auto":
            _graspologic_available = False
            try:
                import subprocess as _sp
                import sys as _sys

                _result = _sp.run(
                    [_sys.executable, "-c", "import graspologic.partition"],
                    timeout=10,
                    capture_output=True,
                )
                _graspologic_available = _result.returncode == 0
            except _sp.TimeoutExpired:
                logger.warning(
                    "GraphRAG: graspologic import probe timed out (10 s) — "
                    "treating as unavailable. Set graph_rag_community_backend: leidenalg "
                    "to skip the probe."
                )
            except Exception:
                pass

            if not _graspologic_available:
                logger.warning(
                    "GraphRAG: graspologic not available — falling back to leidenalg/Louvain. "
                    "pip install graspologic  (or set graph_rag_community_backend: leidenalg)"
                )
            else:
                from graspologic.partition import hierarchical_leiden

                partitions = hierarchical_leiden(
                    G, max_cluster_size=max_cluster_size, random_seed=seed
                )
                community_levels: dict = {}
                community_hierarchy: dict = {}
                community_children: dict = {}

                for triple in partitions:
                    level = triple.level
                    entity = triple.node
                    cluster = triple.cluster
                    parent = getattr(triple, "parent_cluster", None)

                    if level not in community_levels:
                        community_levels[level] = {}
                    community_levels[level][entity] = cluster

                    if cluster not in community_hierarchy:
                        community_hierarchy[cluster] = parent
                    if parent is not None:
                        if parent not in community_children:
                            community_children[parent] = []
                        if cluster not in community_children[parent]:
                            community_children[parent].append(cluster)

                # Normalize Leiden hierarchy to level-qualified string keys to match Louvain format
                # and avoid cross-level integer collisions.
                normalized_hierarchy: dict = {}
                normalized_children: dict = {}
                # Build a level-lookup for each (level, cluster) → level-qualified key
                cluster_to_level: dict = {}
                for lvl, cmap in community_levels.items():
                    for _ent, cid in cmap.items():
                        if cid not in cluster_to_level:
                            cluster_to_level[cid] = lvl
                for raw_cid, raw_parent in community_hierarchy.items():
                    lvl = cluster_to_level.get(raw_cid, 0)
                    norm_key = f"{lvl}_{raw_cid}"
                    if raw_parent is None:
                        norm_parent = None
                    else:
                        parent_lvl = cluster_to_level.get(raw_parent, max(0, lvl - 1))
                        norm_parent = f"{parent_lvl}_{raw_parent}"
                    normalized_hierarchy[norm_key] = norm_parent
                    if norm_parent is not None:
                        if norm_parent not in normalized_children:
                            normalized_children[norm_parent] = []
                        if norm_key not in normalized_children[norm_parent]:
                            normalized_children[norm_parent].append(norm_key)

                return community_levels, normalized_hierarchy, normalized_children
        else:
            logger.debug("GraphRAG: community backend='%s' — skipping graspologic.", _backend)

        # Tier-2: multi-resolution Leiden via leidenalg — used when backend="leidenalg" or
        # when backend="auto" and graspologic was unavailable
        if _backend in ("auto", "leidenalg"):
            try:
                import igraph as _ig
                import leidenalg as _la

                try:
                    import numpy as np

                    resolutions = list(np.linspace(0.5, 1.5, n_levels)) if n_levels > 1 else [1.0]
                except ImportError:
                    step = 1.0 / max(n_levels - 1, 1) if n_levels > 1 else 0
                    resolutions = [0.5 + i * step for i in range(n_levels)]

                _G_ig = _ig.Graph.from_networkx(G)
                community_levels: dict = {}
                for level_idx, resolution in enumerate(resolutions):
                    try:
                        partition = _la.find_partition(
                            _G_ig,
                            _la.CPMVertexPartition,
                            resolution_parameter=resolution,
                            seed=seed,
                        )
                        node_names = (
                            _G_ig.vs["_nx_name"]
                            if "_nx_name" in _G_ig.vs.attributes()
                            else list(G.nodes())
                        )
                        cmap = {}
                        for cid, members in enumerate(partition):
                            for idx in members:
                                cmap[node_names[idx]] = cid
                        community_levels[level_idx] = cmap
                    except Exception as _e:
                        logger.debug("leidenalg at resolution %s failed: %s", resolution, _e)

                if community_levels:
                    # Build synthetic hierarchy (same approach as Louvain fallback below)
                    community_hierarchy: dict = {}
                    community_children: dict = {}
                    if len(community_levels) > 1:
                        _levels_sorted = sorted(community_levels.keys())
                        for _i in range(1, len(_levels_sorted)):
                            _fine = _levels_sorted[_i]
                            _coarse = _levels_sorted[_i - 1]
                            for _fine_cid in set(community_levels[_fine].values()):
                                _fine_key = f"{_fine}_{_fine_cid}"
                                _members = [
                                    n for n, c in community_levels[_fine].items() if c == _fine_cid
                                ]
                                _votes: dict = {}
                                for _m in _members:
                                    _p = community_levels[_coarse].get(_m)
                                    if _p is not None:
                                        _votes[_p] = _votes.get(_p, 0) + 1
                                _parent_cid = max(_votes, key=_votes.get) if _votes else None
                                _parent_key = (
                                    f"{_coarse}_{_parent_cid}" if _parent_cid is not None else None
                                )
                                community_hierarchy[_fine_key] = _parent_key
                                if _parent_key:
                                    community_children.setdefault(_parent_key, [])
                                    if _fine_key not in community_children[_parent_key]:
                                        community_children[_parent_key].append(_fine_key)
                        for _cid in set(community_levels[_levels_sorted[0]].values()):
                            _root_key = f"{_levels_sorted[0]}_{_cid}"
                            community_hierarchy.setdefault(_root_key, None)
                    else:
                        for _cid in set(community_levels[0].values()):
                            community_hierarchy[f"0_{_cid}"] = None
                    logger.debug(
                        "GraphRAG: leidenalg produced %d community levels.", len(community_levels)
                    )
                    return community_levels, community_hierarchy, community_children
            except ImportError:
                logger.warning(
                    "GraphRAG: leidenalg/igraph not available — falling back to networkx Louvain. "
                    "pip install axon[graphrag]"
                )
        else:
            logger.debug("GraphRAG: community backend='%s' — skipping leidenalg.", _backend)

        # Tier-3: NetworkX Louvain — always available; used when backend="louvain" or as final fallback
        try:
            import networkx.algorithms.community as nx_comm
        except ImportError:
            return {0: dict.fromkeys(G.nodes(), 0)}, {0: None}, {0: []}

        try:
            import numpy as np

            resolutions = list(np.linspace(0.5, 1.5, n_levels)) if n_levels > 1 else [1.0]
        except ImportError:
            step = 1.0 / max(n_levels - 1, 1) if n_levels > 1 else 0
            resolutions = [0.5 + i * step for i in range(n_levels)]

        community_levels = {}

        for level_idx, resolution in enumerate(resolutions):
            try:
                partition = nx_comm.louvain_communities(G, seed=seed, resolution=resolution)
                cmap = {}
                for cid, nodes in enumerate(partition):
                    for node in nodes:
                        cmap[node] = cid
                community_levels[level_idx] = cmap
            except Exception as e:
                logger.debug(f"Louvain at resolution {resolution} failed: {e}")

        if not community_levels:
            return {0: dict.fromkeys(G.nodes(), 0)}, {0: None}, {0: []}

        # Synthetic parent mapping — use level-qualified keys to avoid cross-level collisions
        community_hierarchy = {}
        community_children = {}

        if len(community_levels) > 1:
            levels_sorted = sorted(community_levels.keys())
            for i in range(1, len(levels_sorted)):
                fine_level = levels_sorted[i]
                coarse_level = levels_sorted[i - 1]
                fine_map = community_levels[fine_level]
                coarse_map = community_levels[coarse_level]

                fine_clusters = set(fine_map.values())
                for fine_cid in fine_clusters:
                    fine_key = f"{fine_level}_{fine_cid}"
                    fine_members = [n for n, c in fine_map.items() if c == fine_cid]
                    coarse_votes: dict = {}
                    for m in fine_members:
                        parent_cid = coarse_map.get(m)
                        if parent_cid is not None:
                            coarse_votes[parent_cid] = coarse_votes.get(parent_cid, 0) + 1
                    parent_cid = max(coarse_votes, key=coarse_votes.get) if coarse_votes else None
                    parent_key = f"{coarse_level}_{parent_cid}" if parent_cid is not None else None
                    community_hierarchy[fine_key] = parent_key
                    if parent_key is not None:
                        if parent_key not in community_children:
                            community_children[parent_key] = []
                        if fine_key not in community_children[parent_key]:
                            community_children[parent_key].append(fine_key)

            for cid in set(community_levels[levels_sorted[0]].values()):
                root_key = f"{levels_sorted[0]}_{cid}"
                if root_key not in community_hierarchy:
                    community_hierarchy[root_key] = None
        else:
            for cid in set(community_levels[0].values()):
                community_hierarchy[f"0_{cid}"] = None

        return community_levels, community_hierarchy, community_children

    def _rebuild_communities(self) -> None:
        """Run community detection and generate summaries."""
        with self._community_rebuild_lock:
            if bool(getattr(self.config, "graph_rag_rebuild_skip_if_unchanged", True)):
                _sig = (
                    len(self._entity_graph),
                    len(self._relation_graph),
                    sum(len(v.get("chunk_ids", [])) for v in self._entity_graph.values()),
                    sum(len(v) for v in self._relation_graph.values()),
                )
                if _sig == getattr(self, "_gr_last_community_rebuild_sig", None):
                    logger.debug("GraphRAG: skip community rebuild (graph unchanged).")
                    return
            # Entity alias resolution — merge near-duplicates before community detection
            if getattr(self.config, "graph_rag_entity_resolve", False):
                self._resolve_entity_aliases()
            logger.info("GraphRAG: Running community detection...")
            result = self._run_hierarchical_community_detection()
            if isinstance(result, tuple) and len(result) == 3:
                community_levels, hierarchy, children = result
            else:
                # Legacy: _run_hierarchical_community_detection returned a plain dict
                community_levels = result if isinstance(result, dict) else {}
                hierarchy = {}
                children = {}
            self._community_levels = community_levels
            self._community_hierarchy = hierarchy
            self._community_children = children
            self._save_community_hierarchy()
            if self._community_levels:
                self._save_community_levels()
                total = sum(len(set(m.values())) for m in self._community_levels.values())
                logger.info(
                    f"GraphRAG: {len(self._community_levels)} community levels, "
                    f"{total} total communities detected."
                )
                if self.config.graph_rag_community:
                    _lazy = getattr(self.config, "graph_rag_community_lazy", False)
                    if _lazy:
                        logger.info(
                            "GraphRAG: community_lazy=True — skipping summarization; "
                            "will generate on first global query."
                        )
                    else:
                        self._generate_community_summaries()
                        if getattr(self.config, "graph_rag_index_community_reports", True):
                            self._index_community_reports_in_vector_store()
            if bool(getattr(self.config, "graph_rag_rebuild_skip_if_unchanged", True)):
                self._gr_last_community_rebuild_sig = (
                    len(self._entity_graph),
                    len(self._relation_graph),
                    sum(len(v.get("chunk_ids", [])) for v in self._entity_graph.values()),
                    sum(len(v) for v in self._relation_graph.values()),
                )

    def finalize_graph(self, force: bool = False) -> None:
        """Explicitly trigger community rebuild.

        Use after batch ingest with ``graph_rag_community_defer=True`` to run
        community detection once when all documents have been ingested.
        Set ``force=True`` to rebuild even when the in-memory dirty flag is not
        set (for example, after a server restart or project switch).
        """
        self._assert_write_allowed("finalize_graph")
        if force or self._community_graph_dirty:
            self._community_graph_dirty = False
            self._rebuild_communities()

    # ── Graph visualization ──────────────────────────────────────────────────

    _VIZ_TYPE_COLORS: dict[str, str] = {
        "PERSON": "#4e79a7",
        "ORGANIZATION": "#f28e2b",
        "GEO": "#59a14f",
        "EVENT": "#e15759",
        "CONCEPT": "#76b7b2",
        "PRODUCT": "#edc948",
        "UNKNOWN": "#bab0ab",
    }

    def _generate_community_summaries(self, query_hint: str = "") -> None:
        """Generate LLM summaries for each detected community cluster across all levels.

        When *query_hint* is provided (lazy mode), communities are ranked by relevance to
        the query first so the most useful ones get LLM treatment before the budget cap.
        The LLM cap is tightened to ``graph_rag_global_top_communities`` in lazy mode,
        replacing the full ``graph_rag_community_llm_max_total`` limit.
        """
        if not self._community_levels:
            return
        import hashlib as _hashlib
        import json as _json
        import re as _re
        from collections import defaultdict

        def _member_hash(level_idx: int, members: list) -> str:
            members_sorted = sorted(members)
            raw = f"{level_idx}|{'|'.join(members_sorted)}"
            return _hashlib.md5(raw.encode()).hexdigest()

        summaries = {}
        total_communities = sum(len(set(m.values())) for m in self._community_levels.values())

        _min_size = getattr(self.config, "graph_rag_community_min_size", 3)
        _top_n_per_level = getattr(self.config, "graph_rag_community_llm_top_n_per_level", 15)
        _max_total = getattr(self.config, "graph_rag_community_llm_max_total", 30)

        # Lazy mode: tighten cap to graph_rag_global_top_communities so only the most
        # query-relevant communities receive LLM treatment on the first global query.
        _lazy_cap = getattr(self.config, "graph_rag_global_top_communities", 0)
        if query_hint and _lazy_cap > 0:
            _max_total = min(_max_total, _lazy_cap)
            logger.info(
                "GraphRAG: lazy mode — generating community summaries for query "
                "(LLM cap: %d of %d communities).",
                _max_total,
                total_communities,
            )
        else:
            logger.info("GraphRAG: Generating summaries for %d communities...", total_communities)

        # Pre-compute query relevance scores when a hint is provided.
        _query_words: set = set(query_hint.lower().split()) if query_hint else set()

        def _query_relevance(members: list) -> float:
            """Fraction of query words present in community entity names."""
            if not _query_words:
                return 0.0
            member_words = {w for m in members for w in m.lower().split()}
            return len(_query_words & member_words) / len(_query_words)

        _llm_calls_issued = 0

        def _community_score(members: list) -> float:
            """Composite score = density × size. Used to rank which communities get LLM."""
            member_set = set(members)
            internal_edges = sum(
                1
                for src in members
                for entry in self._relation_graph.get(src, [])
                if entry.get("target", "") in member_set
            )
            return (internal_edges / max(len(members), 1)) * len(members)

        def _template_summary(level_idx: int, cid, members: list, new_hash: str) -> tuple:
            """Deterministic zero-LLM summary for small/capped communities."""
            size = len(members)
            sample = ", ".join(sorted(members)[:5])
            suffix = f" (+{size - 5} more)" if size > 5 else ""
            title = f"Community {cid}"
            summary = f"A cluster of {size} entities: {sample}{suffix}."
            return f"{level_idx}_{cid}", {
                "title": title,
                "summary": summary,
                "findings": [],
                "rank": float(size),
                "full_content": f"# {title}\n\n{summary}",
                "entities": members,
                "size": size,
                "level": level_idx,
                "member_hash": new_hash,
                "template": True,
            }

        def _summarise(args):
            level_idx, cid, members = args
            summary_key = f"{level_idx}_{cid}"
            new_hash = _member_hash(level_idx, members)
            existing = self._community_summaries.get(summary_key, {})
            if existing.get("member_hash") == new_hash and existing.get("full_content"):
                # Membership unchanged — reuse cached summary
                cached = dict(existing)
                cached["member_hash"] = new_hash
                return summary_key, cached
            ent_parts = []
            for e in members[:30]:
                node = self._entity_graph.get(e, {})
                desc = node.get("description", "") if isinstance(node, dict) else ""
                ent_type = node.get("type", "") if isinstance(node, dict) else ""
                if desc:
                    ent_parts.append(
                        f"- {e} [{ent_type}]: {desc}" if ent_type else f"- {e}: {desc}"
                    )
                else:
                    ent_parts.append(f"- {e}")
            member_set = set(members)
            rel_parts = []
            for src in members[:20]:
                for entry in self._relation_graph.get(src, [])[:5]:
                    tgt = entry.get("target", "")
                    if tgt in member_set:
                        rel_desc = entry.get("description") or (
                            f"{src} {entry.get('relation', '')} {tgt}"
                        )
                        rel_parts.append(f"- {rel_desc}")
            # GAP 3a: token-budget check — substitute sub-community reports when too large
            max_ctx_tokens = getattr(self.config, "graph_rag_community_max_context_tokens", 4000)
            entity_rel_text = "\n".join(ent_parts + rel_parts)
            if len(entity_rel_text) // 4 > max_ctx_tokens:
                # Children keys are level-qualified strings (e.g. "1_3") — look up directly
                parent_key = f"{level_idx}_{cid}"
                sub_community_keys = self._community_children.get(
                    parent_key, self._community_children.get(cid, [])
                )
                sub_reports = []
                ranked_sub = sorted(
                    [k for k in sub_community_keys if k in self._community_summaries],
                    key=lambda k: self._community_summaries[k].get("rank", 0.0),
                    reverse=True,
                )
                for sub_key in ranked_sub[:5]:
                    sub_reports.append(self._community_summaries[sub_key].get("full_content", ""))
                if sub_reports:
                    entity_rel_text = "\n\n".join(
                        f"Sub-community Report:\n{r}" for r in sub_reports
                    )
            else:
                entity_rel_text = "\n".join(ent_parts)
                if rel_parts:
                    entity_rel_text += "\n\nKey relationships:\n" + "\n".join(rel_parts[:20])
            # GAP 3c: Include claims in community context — auto-enabled if claims extraction is on
            include_claims = getattr(self.config, "graph_rag_community_include_claims", False)
            if not include_claims and getattr(self.config, "graph_rag_claims", False):
                include_claims = True
            claim_parts = []
            if include_claims and self._claims_graph:
                for entity in members[:10]:
                    for chunk_claims in self._claims_graph.values():
                        for claim in chunk_claims:
                            if claim.get("subject", "").lower() == entity.lower():
                                status = claim.get("status", "")
                                desc = claim.get("description", "")
                                claim_parts.append(f"  [{status}] {entity}: {desc}")
                if claim_parts:
                    claim_parts = claim_parts[:10]
            context = entity_rel_text
            if claim_parts:
                context += "\n\nClaims:\n" + "\n".join(claim_parts)
            prompt = (
                "You are analyzing a knowledge graph community.\n\n"
                "Community members and descriptions:\n"
                f"{context}\n\n"
                "Generate a community report as JSON with this exact structure:\n"
                "{\n"
                '  "title": "<2-5 word theme name>",\n'
                '  "summary": "<2-3 sentence overview>",\n'
                '  "findings": [\n'
                '    {"summary": "<headline>", "explanation": "<1-2 sentence detail>"},\n'
                "    ... up to 8 findings\n"
                "  ],\n"
                '  "rank": <float 0-10 importance>\n'
                "}\n"
                "Output only valid JSON. No markdown code blocks."
            )
            try:
                raw_text = self._gr_llm_complete_cached(
                    "community_summary",
                    prompt,
                    system_prompt="You are an expert knowledge graph analyst.",
                )
                try:
                    raw_stripped = _re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()
                    parsed = _json.loads(raw_stripped)
                    title = str(parsed.get("title", f"Community {cid}"))
                    summary = str(parsed.get("summary", ""))
                    findings = parsed.get("findings", [])
                    if not isinstance(findings, list):
                        findings = []
                    rank = float(parsed.get("rank", 5.0))
                except Exception:
                    title = f"Community {cid}"
                    summary = raw_text.strip() if raw_text else ""
                    findings = []
                    rank = 5.0
                findings_text = "\n".join(
                    f"- {f.get('summary', '')}: {f.get('explanation', '')}"
                    for f in findings
                    if isinstance(f, dict)
                )
                full_content = f"# {title}\n\n{summary}"
                if findings_text:
                    full_content += f"\n\nFindings:\n{findings_text}"
                return summary_key, {
                    "title": title,
                    "summary": summary,
                    "findings": findings,
                    "rank": rank,
                    "full_content": full_content,
                    "entities": members,
                    "size": len(members),
                    "level": level_idx,
                    "member_hash": new_hash,
                }
            except Exception as e:
                logger.debug(f"Community summary failed for community {summary_key}: {e}")
                return summary_key, {
                    "title": f"Community {cid}",
                    "summary": "",
                    "findings": [],
                    "rank": 5.0,
                    "full_content": "",
                    "entities": members,
                    "size": len(members),
                    "level": level_idx,
                    "member_hash": new_hash,
                }

        # Process levels from finest (highest idx) to coarsest — SEQUENTIALLY so sub-community
        # reports are available when summarizing parent communities (true bottom-up composition).
        for level_idx in sorted(self._community_levels.keys(), reverse=True):
            level_map = self._community_levels[level_idx]
            community_entities: dict = defaultdict(list)
            for entity, cid in level_map.items():
                community_entities[cid].append(entity)

            llm_items, template_items = [], []
            if query_hint:
                # Prioritise query-relevant communities so they consume LLM budget first.
                ranked = sorted(
                    community_entities.items(),
                    key=lambda kv: (_query_relevance(kv[1]), _community_score(kv[1])),
                    reverse=True,
                )
            else:
                ranked = sorted(
                    community_entities.items(),
                    key=lambda kv: _community_score(kv[1]),
                    reverse=True,
                )

            for rank_pos, (cid, members) in enumerate(ranked):
                summary_key = f"{level_idx}_{cid}"
                new_hash = _member_hash(level_idx, members)
                # Cache hit: always reuse regardless of triage
                existing = self._community_summaries.get(summary_key, {})
                if existing.get("member_hash") == new_hash and existing.get("full_content"):
                    summaries[summary_key] = dict(existing)
                    continue
                # Triage gates (applied in order):
                if _min_size > 0 and len(members) < _min_size:
                    template_items.append((level_idx, cid, members, new_hash))
                    continue
                if _top_n_per_level > 0 and rank_pos >= _top_n_per_level:
                    template_items.append((level_idx, cid, members, new_hash))
                    continue
                if _max_total > 0 and _llm_calls_issued >= _max_total:
                    template_items.append((level_idx, cid, members, new_hash))
                    continue
                _llm_calls_issued += 1
                llm_items.append((level_idx, cid, members))

            for args in template_items:
                key, val = _template_summary(*args)
                summaries[key] = val
            for summary_key, summary_dict in self._executor.map(_summarise, llm_items):
                summaries[summary_key] = summary_dict
            # Make this level's summaries available before summarizing coarser levels
            self._community_summaries = dict(summaries)

        self._save_community_summaries()
        logger.info(f"GraphRAG: Community summaries generated for {len(summaries)} communities.")

    def _index_community_reports_in_vector_store(self) -> None:
        """Index community reports as synthetic documents in the vector store."""
        if not self._community_summaries:
            return
        docs_to_add = []
        for summary_key, cs in self._community_summaries.items():
            full_content = cs.get("full_content") or cs.get("summary", "")
            if not full_content:
                continue
            doc_id = f"__community__{summary_key}"
            content_hash = cs.get("member_hash", "")
            # Skip re-indexing if content is unchanged since last index
            if cs.get("indexed_hash") == content_hash and content_hash:
                continue
            cs["indexed_hash"] = content_hash
            docs_to_add.append(
                {
                    "id": doc_id,
                    "text": full_content,
                    "metadata": {
                        "graph_rag_type": "community_report",
                        "community_key": summary_key,
                        "level": cs.get("level", 0),
                        "rank": cs.get("rank", 5.0),
                        "title": cs.get("title", ""),
                        "source": "__community_report__",
                    },
                }
            )
        if docs_to_add:
            try:
                ids = [d["id"] for d in docs_to_add]
                texts = [d["text"] for d in docs_to_add]
                metadatas = [d["metadata"] for d in docs_to_add]
                embeddings = self.embedding.embed(texts)
                self._own_vector_store.add(ids, texts, embeddings, metadatas)
                logger.info(
                    f"GraphRAG: Indexed {len(docs_to_add)} community reports in vector store."
                )
            except Exception as e:
                logger.warning(f"Could not index community reports: {e}")

    def _global_search_map_reduce(self, query: str, cfg) -> str:
        """Map-reduce global search over community reports (Item 4)."""
        import json as _json
        import re as _re
        import time as _time

        _t0_total = _time.perf_counter()

        if not self._community_summaries:
            return ""

        min_score = getattr(cfg, "graph_rag_global_min_score", 20)
        top_points = getattr(cfg, "graph_rag_global_top_points", 50)

        # Filter summaries to the target level
        target_level = getattr(cfg, "graph_rag_community_level", 0)
        target_level_prefix = f"{target_level}_"
        level_summaries = {
            k: v for k, v in self._community_summaries.items() if k.startswith(target_level_prefix)
        }
        # Fall back to all summaries if nothing matches the target level
        if not level_summaries:
            level_summaries = self._community_summaries

        # Dynamic community pre-filter — cheap token-overlap relevance score
        _top_n_communities = getattr(cfg, "graph_rag_global_top_communities", 0)
        if _top_n_communities > 0 and len(level_summaries) > _top_n_communities:
            import heapq as _heapq

            _query_words = set(query.lower().split())

            def _community_relevance(cs_item: tuple) -> float:
                _, cs = cs_item
                text = ((cs.get("title") or "") + " " + (cs.get("summary") or "")).lower()
                return len(_query_words & set(text.split())) / max(len(_query_words), 1)

            top_summaries = _heapq.nlargest(
                _top_n_communities, level_summaries.items(), key=_community_relevance
            )
            level_summaries = dict(top_summaries)
            logger.debug("GraphRAG global: pre-filtered to top %d communities.", _top_n_communities)

        # Global answer cache: short-circuit repeated query+graph-signature requests.
        if bool(getattr(cfg, "graph_rag_global_answer_cache", True)):
            _sig_parts = []
            for _k, _cs in sorted(level_summaries.items(), key=lambda kv: kv[0]):
                _sig_parts.append(
                    f"{_k}:{_cs.get('member_hash','')}:{_cs.get('indexed_hash','')}:{_cs.get('rank',0)}"
                )
            _ans_key = self._gr_text_hash(
                f"{query}|lvl={target_level}|topc={_top_n_communities}|{'|'.join(_sig_parts)}"
            )
            _ans_cached = self._gr_cache_get("global_answer", _ans_key)
            if _ans_cached is not None:
                self._gr_log_profile(
                    "global_search.answer_cache_hit", _time.perf_counter() - _t0_total
                )
                return _ans_cached
        else:
            _ans_key = None

        # Chunk reports so large reports don't get hard-truncated and later sections aren't lost
        _MAP_CHUNK_CHARS = int(getattr(cfg, "graph_rag_global_map_max_length", 500) or 500) * 4

        def _chunk_report(cid: str, cs: dict) -> list[tuple[str, str]]:
            """Split a community report into bounded chunks. Returns [(cid_chunk_id, text)]."""
            report = cs.get("full_content") or cs.get("summary", "")
            if not report:
                return []
            chunks = []
            for i in range(0, len(report), _MAP_CHUNK_CHARS):
                chunk_text = report[i : i + _MAP_CHUNK_CHARS]
                chunks.append((f"{cid}#{i}", chunk_text))
            return chunks

        # Build shuffled chunk list for unbiased map phase
        import random as _random

        all_chunks: list[tuple[str, str]] = []
        for cid, cs in level_summaries.items():
            all_chunks.extend(_chunk_report(cid, cs))
        _max_map_chunks = int(getattr(cfg, "graph_rag_global_max_map_chunks", 0) or 0)
        if _max_map_chunks > 0 and len(all_chunks) > _max_map_chunks:
            all_chunks = all_chunks[:_max_map_chunks]
        rng = _random.Random(42)
        rng.shuffle(all_chunks)
        self._gr_log_profile(
            "global_search.prepare_chunks", _time.perf_counter() - _t0_total, chunks=len(all_chunks)
        )

        # Token-level compression of community report chunks before LLM map phase
        if getattr(cfg, "graph_rag_report_compress", False) is True and all_chunks:
            _ratio = getattr(cfg, "graph_rag_report_compress_ratio", 0.5)
            try:
                _lingua = self._ensure_llmlingua()
                _compressed_chunks = []
                for _cid, _text in all_chunks:
                    if not _text:
                        _compressed_chunks.append((_cid, _text))
                        continue
                    try:
                        _out = _lingua.compress_prompt(_text, rate=_ratio, force_tokens=["\n"])
                        _compressed_chunks.append((_cid, _out["compressed_prompt"]))
                    except Exception:
                        _compressed_chunks.append((_cid, _text))
                all_chunks = _compressed_chunks
                logger.info(
                    "GraphRAG map-reduce: compressed %d report chunks (ratio=%.2f)",
                    len(all_chunks),
                    _ratio,
                )
            except ImportError:
                logger.warning(
                    "graph_rag_report_compress=True but llmlingua not installed. "
                    "pip install axon[llmlingua]"
                )

        def _map_community(args):
            chunk_id, report_chunk = args
            if not report_chunk:
                return []
            _chunk_sig = self._gr_text_hash(report_chunk)
            _map_key = self._gr_text_hash(f"{query}|{chunk_id}|{_chunk_sig}")
            _map_cached = self._gr_cache_get("global_map", _map_key)
            if isinstance(_map_cached, list):
                return _map_cached
            prompt = (
                "---Role---\n"
                "You are a helpful assistant responding to questions about the dataset.\n\n"
                "---Goal---\n"
                "Generate a list of key points relevant to answering the question, "
                "based solely on the community report below.\n"
                "If the report is not relevant, return an empty JSON array.\n\n"
                f"---Question---\n{query}\n\n"
                f"---Community Report---\n{report_chunk}\n\n"
                "---Response Format---\n"
                'Return a JSON array: [{"point": "...", "score": 1-100}, ...]\n'
                "Higher score = more relevant. Output only valid JSON."
            )
            try:
                raw = self._gr_llm_complete_cached(
                    "global_map_chunk",
                    prompt,
                    system_prompt="You are a knowledge graph analysis assistant.",
                )
                raw_clean = _re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
                points = _json.loads(raw_clean)
                if not isinstance(points, list):
                    return []
                out = [
                    (float(p.get("score", 0)), str(p.get("point", "")))
                    for p in points
                    if isinstance(p, dict) and p.get("point")
                ]
                self._gr_cache_put("global_map", _map_key, out)
                return out
            except Exception:
                return []

        # Map phase — parallel over shuffled chunks; use a dedicated pool when
        # graph_rag_map_workers is set, so map-reduce does not starve the shared executor
        # during concurrent ingest.
        import heapq as _heapq

        if top_points <= 0:
            return _GRAPHRAG_NO_DATA_ANSWER
        top_heap: list[tuple[float, str]] = []
        _points_seen = 0

        def _consume_point(_score: float, _point: str) -> None:
            nonlocal top_heap, _points_seen
            _points_seen += 1
            if _score < min_score:
                return
            if len(top_heap) < top_points:
                _heapq.heappush(top_heap, (_score, _point))
            elif _score > top_heap[0][0]:
                _heapq.heapreplace(top_heap, (_score, _point))

        _map_workers_cfg = int(getattr(cfg, "graph_rag_map_workers", 0) or 0)
        _map_auto_workers = int(getattr(cfg, "graph_rag_map_auto_workers", 4) or 0)
        _map_use_dedicated_pool = bool(getattr(cfg, "graph_rag_map_use_dedicated_pool", True))
        _map_workers_effective = 0
        if _map_workers_cfg > 0:
            _map_workers_effective = min(_map_workers_cfg, max(1, len(all_chunks)))
        elif _map_use_dedicated_pool and _map_auto_workers > 0:
            import os as _os

            _cpu_cap = max(1, int(_os.cpu_count() or 4))
            _map_workers_effective = min(_map_auto_workers, _cpu_cap, max(1, len(all_chunks)))
        _t0_map = _time.perf_counter()
        try:
            if _map_workers_effective > 0:
                from concurrent.futures import ThreadPoolExecutor as _TPE

                with _TPE(max_workers=_map_workers_effective) as _map_pool:
                    for point_list in _map_pool.map(_map_community, all_chunks):
                        for score, point in point_list:
                            _consume_point(score, point)
            else:
                for point_list in self._executor.map(_map_community, all_chunks):
                    for score, point in point_list:
                        _consume_point(score, point)
        except Exception:
            return ""
        self._gr_log_profile(
            "global_search.map_phase",
            _time.perf_counter() - _t0_map,
            points_seen=_points_seen,
            heap_size=len(top_heap),
            workers=_map_workers_effective,
        )

        # Extract top points in descending score order
        top = _heapq.nlargest(top_points, top_heap, key=lambda x: x[0])

        if not top:
            return _GRAPHRAG_NO_DATA_ANSWER
        _skip_reduce_points_le = int(
            getattr(cfg, "graph_rag_global_reduce_skip_if_top_points_le", 1) or 0
        )
        _skip_reduce_score_gte = float(
            getattr(cfg, "graph_rag_global_reduce_skip_if_top_score_gte", 95.0) or 95.0
        )
        if (
            _skip_reduce_points_le > 0
            and len(top) <= _skip_reduce_points_le
            and top[0][0] >= _skip_reduce_score_gte
        ):
            self._gr_log_profile(
                "global_search.reduce_skipped",
                _time.perf_counter() - _t0_total,
                top_points=len(top),
                top_score=top[0][0],
            )
            out = top[0][1]
            if _ans_key is not None:
                self._gr_cache_put("global_answer", _ans_key, out)
            return out

        # --- REDUCE PHASE: token-budget assembly ---
        _t0_reduce = _time.perf_counter()
        reduce_max_tokens = getattr(cfg, "graph_rag_global_reduce_max_tokens", 8000)
        analyst_lines = []
        token_estimate = 0
        for idx, (score, point) in enumerate(top):
            line = f"----Analyst {idx + 1}----\nImportance Score: {score:.0f}\n{point}"
            # Rough token estimate: 1 token ≈ 4 characters
            token_estimate += len(line) // 4
            if token_estimate > reduce_max_tokens:
                break
            analyst_lines.append(line)

        if not analyst_lines:
            return _GRAPHRAG_NO_DATA_ANSWER

        reduce_context = "\n\n".join(analyst_lines)
        reduce_max_length = getattr(cfg, "graph_rag_global_reduce_max_length", 500)
        reduce_prompt = (
            f"The following analytic reports have been generated for the query:\n\n"
            f"Query: {query}\n\n"
            f"Reports:\n\n{reduce_context}\n\n"
            f"Using the reports above, generate a comprehensive response to the query."
            f"\nRespond in at most {reduce_max_length} tokens."
        )
        reduce_system_prompt = _GRAPHRAG_REDUCE_SYSTEM_PROMPT
        if getattr(cfg, "graph_rag_global_allow_general_knowledge", False):
            reduce_system_prompt = (
                reduce_system_prompt
                + " You may supplement the provided reports with your own general knowledge"
                " where relevant."
            )
        try:
            response = self._gr_llm_complete_cached(
                "global_reduce",
                reduce_prompt,
                system_prompt=reduce_system_prompt,
            )
            out = response.strip() if response else _GRAPHRAG_NO_DATA_ANSWER
            if _ans_key is not None:
                self._gr_cache_put("global_answer", _ans_key, out)
            self._gr_log_profile(
                "global_search.reduce_phase",
                _time.perf_counter() - _t0_reduce,
                analysts=len(analyst_lines),
            )
            self._gr_log_profile("global_search.total", _time.perf_counter() - _t0_total)
            return out
        except Exception as e:
            logger.debug(f"GraphRAG global reduce failed: {e}")
            # Fallback: return raw analyst summaries
            self._gr_log_profile(
                "global_search.reduce_phase(error)",
                _time.perf_counter() - _t0_reduce,
                analysts=len(analyst_lines),
            )
            self._gr_log_profile("global_search.total", _time.perf_counter() - _t0_total)
            out = "**Knowledge Graph Findings:**\n\n" + reduce_context
            if _ans_key is not None:
                self._gr_cache_put("global_answer", _ans_key, out)
            return out

    def _get_incoming_relations(self, entity: str) -> list[dict]:
        """Return all relation entries where entity is the target (GAP 4: incoming edges)."""
        ent_lower = entity.lower()
        idx = self._get_incoming_relation_index()
        if idx:
            out = []
            for item in idx.get(ent_lower, []):
                if isinstance(item, tuple) and len(item) == 2:
                    src, entry = item
                    if isinstance(entry, dict):
                        out.append({**entry, "source": src, "direction": "incoming"})
                elif isinstance(item, dict):
                    # Backward-compat with older in-memory cache shape.
                    out.append(item)
            return out
        results = []
        for src, entries in self._relation_graph.items():
            for entry in entries:
                if entry.get("target", "").lower() == ent_lower:
                    results.append({**entry, "source": src, "direction": "incoming"})
        return results

    def _local_search_context(self, query: str, matched_entities: list, cfg) -> str:
        """Build structured GraphRAG local context using unified candidate ranking.

        TASK 11: Replaces fixed 25/50/25 budget split with joint ranking across all
        artifact types, then greedy-fill up to total_budget tokens.
        """
        import time as _time

        _t0_total = _time.perf_counter()
        if not matched_entities:
            return ""

        import re as _re

        # --- Phase 1: Setup ---
        _t0_setup = _time.perf_counter()
        total_budget = getattr(cfg, "graph_rag_local_max_context_tokens", 8000)
        top_k_entities = getattr(cfg, "graph_rag_local_top_k_entities", 10)
        top_k_relationships = getattr(cfg, "graph_rag_local_top_k_relationships", 10)
        include_weight = getattr(cfg, "graph_rag_local_include_relationship_weight", False)

        entity_weight = getattr(cfg, "graph_rag_local_entity_weight", 3.0)
        relation_weight = getattr(cfg, "graph_rag_local_relation_weight", 2.0)
        community_weight = getattr(cfg, "graph_rag_local_community_weight", 1.5)
        text_unit_weight = getattr(cfg, "graph_rag_local_text_unit_weight", 1.0)

        _query_tokens = set(query.lower().split()) | set(_re.split(r"[\s\W_]+", query.lower()))
        _boost = getattr(self.config, "graph_rag_exact_entity_boost", 3.0)
        _fast_degree = bool(getattr(cfg, "graph_rag_local_entity_degree_fast", True))
        _incoming_count_map: dict[str, int] = {}
        if _fast_degree:
            _incoming_count_map = self._get_incoming_relation_count_map()

        def _entity_degree(ent: str) -> int:
            ent_lower = ent.lower()
            outgoing = len(self._relation_graph.get(ent_lower, []))
            if _incoming_count_map:
                incoming = _incoming_count_map.get(ent_lower, 0)
            else:
                incoming = len(self._get_incoming_relations(ent))
            return outgoing + incoming

        def _entity_raw(ent: str) -> float:
            return _entity_degree(ent) * (_boost if ent.lower() in _query_tokens else 1.0)

        ranked_entities = sorted(matched_entities, key=_entity_raw, reverse=True)
        ranked_entities = ranked_entities[:top_k_entities]
        self._gr_log_profile("local_search.setup", _time.perf_counter() - _t0_setup)

        # --- Phase 2: Collect all candidates into a flat list ---
        _t0_collect = _time.perf_counter()
        # Store candidates as lightweight tuples to reduce per-candidate allocations:
        # (score, tokens, type, text)
        candidates: list[tuple[float, int, str, str]] = []

        # Entities
        raw_scores = [_entity_raw(e) for e in ranked_entities]
        max_raw = max(raw_scores, default=1.0) or 1.0
        for ent, raw in zip(ranked_entities, raw_scores):
            node = self._entity_graph.get(ent.lower(), {})
            desc = node.get("description", "") if isinstance(node, dict) else ""
            ent_type = node.get("type", "") if isinstance(node, dict) else ""
            if not desc:
                continue
            line = f"  - {ent} [{ent_type}]: {desc}" if ent_type else f"  - {ent}: {desc}"
            candidates.append((entity_weight * (raw / max_raw), len(line) // 4 + 1, "entity", line))

        # Relations — collect outgoing (top 3/entity) + incoming (top 2/entity)
        _use_fast_rel_support = bool(getattr(cfg, "graph_rag_local_relation_support_fast", True))
        if _use_fast_rel_support:
            target_support_count: dict[str, int] = {}
            for ee in ranked_entities:
                tset: set[str] = set()
                for rel in self._relation_graph.get(ee.lower(), []):
                    tgt = rel.get("target", "")
                    if tgt:
                        tset.add(tgt)
                for tgt in tset:
                    target_support_count[tgt] = target_support_count.get(tgt, 0) + 1

            def _mutual_count(e_entry, ranked):
                tgt = e_entry.get("target", "")
                return target_support_count.get(tgt, 0)

        else:

            def _mutual_count(e_entry, ranked):
                tgt = e_entry.get("target", "")
                return sum(
                    1
                    for ee in ranked
                    if tgt
                    in {x.get("target", "") for x in self._relation_graph.get(ee.lower(), [])}
                )

        rel_candidates_raw: list[tuple[float, dict, str]] = []
        seen_rel_keys: set = set()
        for ent in ranked_entities:
            ent_lower = ent.lower()
            outgoing = self._relation_graph.get(ent_lower, [])
            out_scored: list[tuple[float, dict]] = []
            for entry in outgoing:
                out_scored.append((_mutual_count(entry, ranked_entities), entry))
            out_scored.sort(key=lambda x: x[0], reverse=True)
            for mc, entry in out_scored[:3]:
                rel_candidates_raw.append((mc, entry, ent))

            in_scored: list[tuple[float, dict]] = []
            for entry in self._get_incoming_relations(ent):
                in_scored.append((_mutual_count(entry, ranked_entities), entry))
            in_scored.sort(key=lambda x: x[0], reverse=True)
            for mc, entry in in_scored[:2]:
                rel_candidates_raw.append((mc, entry, ent))

        # Sort by mutual count, cap at top_k_relationships, normalize
        rel_candidates_raw.sort(key=lambda x: x[0], reverse=True)
        rel_candidates_raw = rel_candidates_raw[:top_k_relationships]
        max_mutual = max((x[0] for x in rel_candidates_raw), default=0)
        for mc, entry, ent in rel_candidates_raw:
            src = entry.get("source", ent)
            desc = entry.get("description") or (
                f"{src} {entry.get('relation', '')} {entry.get('target', '')}"
            )
            weight_str = ""
            if include_weight and entry.get("weight"):
                weight_str = f" (weight: {entry['weight']})"
            line = f"  - {desc}{weight_str}"
            rel_key = line[:80]
            if rel_key in seen_rel_keys:
                continue
            seen_rel_keys.add(rel_key)
            within = (mc + 1) / (max_mutual + 1)
            candidates.append((relation_weight * within, len(line) // 4 + 1, "relation", line))

        # Communities — finest level, deduplicated by summary_key
        if self._community_levels and self._community_summaries:
            finest = max(self._community_levels.keys()) if self._community_levels else None
            seen_community_keys: set = set()
            community_raws: list[tuple[float, str, str]] = []
            for ent in ranked_entities:
                cid = (
                    self._community_levels.get(finest, {}).get(ent.lower())
                    if finest is not None
                    else None
                )
                if cid is None:
                    continue
                summary_key = f"{finest}_{cid}"
                if summary_key in seen_community_keys:
                    continue
                seen_community_keys.add(summary_key)
                cs = self._community_summaries.get(summary_key, {})
                if not cs:
                    cs = self._community_summaries.get(str(cid), {})
                summary = cs.get("summary", "")
                if not summary:
                    continue
                sentences = summary.split(". ")
                snippet = ". ".join(sentences[:3]).strip()
                if snippet and not snippet.endswith("."):
                    snippet += "."
                title = cs.get("title", f"Community {cid}")
                community_text = f"**Community Context ({title}):**\n  {snippet}"
                rank = cs.get("rank", 0.5)
                community_raws.append((rank, community_text, summary_key))
            max_rank = max((r[0] for r in community_raws), default=1.0) or 1.0
            for rank, ctext, _ in community_raws:
                within = rank / max_rank
                candidates.append(
                    (community_weight * within, len(ctext) // 4 + 1, "community", ctext)
                )

        # Early cut-off: if current highest-scored candidates already fill budget and
        # all selected scores are strictly above max text-unit score, skip text-unit fetch.
        _skip_text_units = False
        if (
            bool(getattr(cfg, "graph_rag_local_early_cutoff", True))
            and candidates
            and total_budget > 0
        ):
            _factor = float(getattr(cfg, "graph_rag_local_early_cutoff_factor", 0.2))
            if _factor < 0.0:
                _factor = 0.0
            _ordered = sorted(candidates, key=lambda c: c[0], reverse=True)
            _used_probe = 0
            _min_score = None
            for _c in _ordered:
                _tok = _c[1]
                if _used_probe + _tok <= total_budget:
                    _used_probe += _tok
                    _min_score = _c[0] if _min_score is None else min(_min_score, _c[0])
            # If the selected score floor is above text-unit max score with a margin,
            # lower-scored text-unit candidates cannot improve the final selection.
            if (
                _used_probe >= total_budget
                and _min_score is not None
                and _min_score >= (text_unit_weight * (1.0 + _factor))
            ):
                _skip_text_units = True

        # Text units — from entity chunk_ids, cap at 20, rank by relation count
        if not _skip_text_units:
            seen_chunks: set = set()
            text_unit_ids_all: list[str] = []
            for ent in ranked_entities:
                node = self._entity_graph.get(ent.lower(), {})
                chunk_ids = node.get("chunk_ids", []) if isinstance(node, dict) else []
                for cid in chunk_ids:
                    if cid not in seen_chunks:
                        seen_chunks.add(cid)
                        text_unit_ids_all.append(cid)
                    if len(text_unit_ids_all) >= 20:
                        break
                if len(text_unit_ids_all) >= 20:
                    break

            def _tu_rel_count(cid: str) -> int:
                return len(self._text_unit_relation_map.get(cid, []))

            max_rel = max((_tu_rel_count(c) for c in text_unit_ids_all), default=0)
            if text_unit_ids_all:
                try:
                    _batch_fetch = bool(getattr(cfg, "graph_rag_local_batch_fetch", True))
                    if _batch_fetch:
                        docs_all = self.vector_store.get_by_ids(text_unit_ids_all)
                        doc_map = {}
                        if isinstance(docs_all, list):
                            # Preferred path: fetch by document id when available.
                            for d in docs_all:
                                if isinstance(d, dict) and d.get("id"):
                                    doc_map[str(d.get("id"))] = d
                            # Backward-compatible fallback: some stores return docs without id but
                            # preserve order for the requested ids.
                            if not doc_map and len(docs_all) == len(text_unit_ids_all):
                                for _cid, _doc in zip(text_unit_ids_all, docs_all):
                                    if isinstance(_doc, dict):
                                        doc_map[_cid] = _doc
                        _iter_ids = text_unit_ids_all

                        def _fetch(_cid, _m=doc_map):
                            return _m.get(_cid)

                    else:
                        _iter_ids = text_unit_ids_all

                        def _fetch(_cid):  # noqa: F811
                            return None

                    for cid in _iter_ids:
                        if _batch_fetch:
                            doc = _fetch(cid)
                        else:
                            docs_one = self.vector_store.get_by_ids([cid])
                            doc = docs_one[0] if docs_one else None
                        if not doc:
                            continue
                        text_snippet = doc.get("text", "")[:200].strip()
                        if not text_snippet:
                            continue
                        line = f"  [{cid}] {text_snippet}..."
                        rc = _tu_rel_count(cid)
                        within = (rc + 1) / (max_rel + 1)
                        candidates.append(
                            (
                                text_unit_weight * within,
                                len(line) // 4 + 1,
                                "text_unit",
                                line,
                            )
                        )
                except Exception:
                    pass
        self._gr_log_profile("local_search.collect", _time.perf_counter() - _t0_collect)

        # Claims — high-signal factual assertions, fixed score
        if self._claims_graph:
            claim_lines: list[str] = []
            for ent in ranked_entities[:3]:
                for chunk_claims in self._claims_graph.values():
                    for claim in chunk_claims:
                        if claim.get("subject", "").lower() == ent.lower():
                            status = claim.get("status", "SUSPECTED")
                            desc = claim.get("description", "")
                            if desc:
                                claim_lines.append(f"  - [{status}] {desc}")
            for line in claim_lines[:10]:
                candidates.append((entity_weight * 1.0, len(line) // 4 + 1, "claim", line))

        # --- Phase 3: Sort and greedy-fill ---
        _t0_select = _time.perf_counter()
        candidates.sort(key=lambda c: c[0], reverse=True)
        selected: list[tuple[float, int, str, str]] = []
        used = 0
        for c in candidates:
            if used + c[1] <= total_budget:
                selected.append(c)
                used += c[1]
            # continue (not break) — smaller items later may still fit
        self._gr_log_profile("local_search.select", _time.perf_counter() - _t0_select)

        # --- Phase 4: Reassemble into section-formatted output ---
        by_type: dict[str, list[str]] = {
            "entity": [],
            "relation": [],
            "community": [],
            "text_unit": [],
            "claim": [],
        }
        for c in selected:
            by_type[c[2]].append(c[3])

        parts: list[str] = []
        if by_type["entity"]:
            parts.append("**Matched Entities:**\n" + "\n".join(by_type["entity"]))
        if by_type["relation"]:
            parts.append("**Relevant Relationships:**\n" + "\n".join(by_type["relation"]))
        for ctext in by_type["community"]:
            parts.append(ctext)
        if by_type["text_unit"]:
            parts.append("**Source Text Units:**\n" + "\n".join(by_type["text_unit"]))
        if by_type["claim"]:
            parts.append("**Claims / Facts:**\n" + "\n".join(by_type["claim"]))

        out = "\n\n".join(parts)
        self._gr_log_profile(
            "local_search.total",
            _time.perf_counter() - _t0_total,
            matched=len(matched_entities),
            candidates=len(candidates),
            selected=len(selected),
        )
        return out

    def _entity_matches(self, q_entity: str, g_entity: str) -> float:
        """Return Jaccard-based match score between 0.0 and 1.0."""
        q = q_entity.lower().strip()
        g = g_entity.lower().strip()
        if q == g:
            return 1.0
        q_tokens = set(q.split())
        g_tokens = set(g.split())
        if len(q_tokens) == 1 and len(g_tokens) == 1:
            return 0.0  # single tokens must match exactly
        intersection = q_tokens & g_tokens
        union = q_tokens | g_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        return jaccard if jaccard >= 0.4 else 0.0

    _VALID_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "GEO", "EVENT", "CONCEPT", "PRODUCT"}

    _HOLISTIC_KEYWORDS = frozenset(
        [
            "summarize",
            "summary",
            "overview",
            "all",
            "every",
            "compare",
            "list all",
            "across",
            "throughout",
            "entire",
            "whole",
            "global",
            "what are all",
            "how many",
            "count",
            "trend",
            "pattern",
            "themes",
            "main topics",
        ]
    )

    def _classify_query_needs_graphrag(self, query: str, mode: str) -> bool:
        """Return True if GraphRAG global search is warranted for this query (TASK 14)."""
        if mode == "heuristic":
            q_lower = query.lower()
            if any(kw in q_lower for kw in self._HOLISTIC_KEYWORDS):
                return True
            if len(query.split()) > 20:
                return True
            return False
        elif mode == "llm":
            prompt = (
                "Does the following question require a holistic, corpus-wide analysis "
                "(e.g. summarization, comparison, listing all topics) rather than a "
                "targeted factual lookup? Answer only YES or NO.\n\nQuestion: " + query
            )
            try:
                ans = (
                    self._gr_llm_complete_cached(
                        "query_router_llm",
                        prompt,
                        max_tokens=5,
                    )
                    .strip()
                    .upper()
                )
                return ans.startswith("Y")
            except Exception:
                return False
        return False

    # ── keyword sets for multi-class router ───────────────────────────────────
    _SYNTHESIS_KEYWORDS: frozenset = frozenset(
        {
            "summarize",
            "overview",
            "compare",
            "contrast",
            "explain",
            "discuss",
            "survey",
            "themes",
            "analysis",
        }
    )
    _TABLE_KEYWORDS: frozenset = frozenset(
        {
            "table",
            "row",
            "column",
            "value",
            "count",
            "average",
            "maximum",
            "minimum",
            "statistic",
            "how many",
            "list all",
        }
    )
    _ENTITY_KEYWORDS: frozenset = frozenset(
        {
            "relationship",
            "related to",
            "who",
            "works with",
            "connected",
            "linked",
            "colleague",
            "dependency",
            "relate",
        }
    )
    _CORPUS_KEYWORDS: frozenset = frozenset(
        {
            "all documents",
            "entire corpus",
            "everything",
            "main topics",
            "key themes",
            "across all",
        }
    )

    def _ensure_llmlingua(self):
        """Lazy-initialise the LLMLingua-2 prompt compressor (TASK 14)."""
        if not hasattr(self, "_llmlingua") or self._llmlingua is None:
            from llmlingua import PromptCompressor

            _model = getattr(
                self.config,
                "graph_rag_llmlingua_model",
                "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            )
            _local = os.path.isabs(_model) or os.path.isdir(_model)
            logger.info(
                "GraphRAG LLMLingua: loading model '%s'%s…", _model, " (local)" if _local else ""
            )
            self._llmlingua = PromptCompressor(
                model_name=_model,
                use_llmlingua2=True,
                device_map="cpu",
                **({"local_files_only": True} if _local else {}),
            )
        return self._llmlingua

    def _ensure_gliner(self):
        """Lazy-initialise the GLiNER NER model (TASK 14)."""
        cached = getattr(self, "_gliner_model", None)
        if cached is not None:
            return cached

        from gliner import GLiNER

        _model = getattr(self.config, "graph_rag_gliner_model", "urchade/gliner_medium-v2.1")
        _local = os.path.isabs(_model) or os.path.isdir(_model)
        _cache_key = (_model, bool(_local))
        with GraphRagMixin._shared_gliner_lock:
            cached = GraphRagMixin._shared_gliner_models.get(_cache_key)
            if cached is None:
                logger.info(
                    "GraphRAG GLiNER: loading model '%s'%s…", _model, " (local)" if _local else ""
                )
                cached = GLiNER.from_pretrained(_model, local_files_only=_local)
                GraphRagMixin._shared_gliner_models[_cache_key] = cached
        self._gliner_model = cached
        return cached

    _GLINER_LABELS = ["person", "organization", "location", "event", "concept", "product"]
    _GLINER_TYPE_MAP = {
        "person": "PERSON",
        "organization": "ORGANIZATION",
        "location": "GEO",
        "event": "EVENT",
        "concept": "CONCEPT",
        "product": "PRODUCT",
    }

    def _ensure_rebel(self):
        """Lazy-initialise the REBEL relation extraction pipeline (P2)."""
        cached = getattr(self, "_rebel_pipeline", None)
        if cached is not None:
            return cached

        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            from transformers import pipeline as _hf_pipeline
        except ImportError:
            raise ImportError("REBEL backend requires transformers. pip install axon[rebel]")
        _model = getattr(self.config, "graph_rag_rebel_model", "Babelscape/rebel-large")
        _local = os.path.isabs(_model) or os.path.isdir(_model)
        _cache_key = (_model, bool(_local))
        with GraphRagMixin._shared_rebel_lock:
            cached = GraphRagMixin._shared_rebel_pipelines.get(_cache_key)
            if cached is None:
                logger.info(
                    "GraphRAG REBEL: loading model '%s'%s…",
                    _model,
                    " (local)" if _local else " (first-run download may take time)",
                )
                _load_kwargs = {"local_files_only": True} if _local else {}
                _model_obj = AutoModelForSeq2SeqLM.from_pretrained(_model, **_load_kwargs)
                _tokenizer = AutoTokenizer.from_pretrained(_model, **_load_kwargs)
                cached = _hf_pipeline(
                    "text2text-generation",
                    model=_model_obj,
                    tokenizer=_tokenizer,
                    device=-1,  # CPU — avoids CUDA dependency
                )
                GraphRagMixin._shared_rebel_pipelines[_cache_key] = cached
        self._rebel_pipeline = cached
        return cached

    @staticmethod
    def _parse_rebel_output(text: str) -> list[dict]:
        """Parse REBEL's ``<triplet> SUBJ <subj> OBJ <obj> REL`` output format.

        REBEL encodes triplets as:
            <triplet> head_entity <subj> tail_entity <obj> relation_type

        Returns a list of relation dicts compatible with ``_extract_relations``.
        """
        triplets: list[dict] = []
        subject = object_ = relation = ""
        current = "x"
        tokens = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
        for token in tokens:
            if token == "<triplet>":
                if subject and relation and object_:
                    triplets.append(
                        {
                            "subject": subject.strip(),
                            "relation": relation.strip(),
                            "object": object_.strip(),
                            "description": "",
                            "strength": 5,
                        }
                    )
                subject = object_ = relation = ""
                current = "t"
            elif token == "<subj>":
                current = "s"
            elif token == "<obj>":
                current = "o"
            elif current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
        # Flush final triplet
        if subject and relation and object_:
            triplets.append(
                {
                    "subject": subject.strip(),
                    "relation": relation.strip(),
                    "object": object_.strip(),
                    "description": "",
                    "strength": 5,
                }
            )
        return triplets

    def _extract_relations_rebel(self, text: str) -> list[dict]:
        """Extract relations using REBEL — no LLM call, structured triplet output (P2)."""
        try:
            pipe = self._ensure_rebel()
            # Preserve REBEL special tokens; the plain generated_text output strips them
            # and makes the triplet parser return no relations.
            _gen_cfg = dict(
                getattr(getattr(pipe, "model", None), "config", {}).task_specific_params.get(
                    "relation_extraction", {}
                )
                if getattr(getattr(pipe, "model", None), "config", None) is not None
                and getattr(pipe.model.config, "task_specific_params", None)
                else {}
            )
            _gen_cfg.setdefault("max_length", 256)
            outputs = pipe(text[:1000], return_tensors=True, return_text=False, **_gen_cfg)
            token_ids = outputs[0].get("generated_token_ids", []) if outputs else []
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if not token_ids:
                logger.debug(
                    "REBEL: pipeline returned empty token_ids for input (len=%d).", len(text)
                )
                return []
            raw = pipe.tokenizer.batch_decode([token_ids], skip_special_tokens=False)[0]
            logger.debug("REBEL decoded output (first 200 chars): %s", raw[:200])
            triplets = self._parse_rebel_output(raw)[:15]
            if not triplets and text.strip():
                logger.warning(
                    "REBEL: decoded %d tokens but parsed 0 relation triplets. "
                    "Check that the model checkpoint is a relation-extraction variant "
                    "(e.g. Babelscape/rebel-large) and that special tokens are preserved "
                    "in decoding. Decoded prefix: %.120r",
                    len(token_ids),
                    raw,
                )
            return triplets
        except ImportError:
            logger.warning(
                "graph_rag_relation_backend='rebel' but transformers is not installed. "
                "pip install axon[rebel]"
            )
            return []
        except Exception as e:
            logger.debug("REBEL extraction failed: %s", e)
            return []

    def _extract_entities_gliner(self, text: str) -> list[dict]:
        """Extract entities using GLiNER (TASK 14 — NER-only; no LLM call)."""
        model = self._ensure_gliner()
        try:
            preds = model.predict_entities(text[:3000], self._GLINER_LABELS, threshold=0.5)
            seen: set = set()
            entities = []
            for p in preds:
                name = p["text"].strip()
                if name.lower() in seen:
                    continue
                seen.add(name.lower())
                entities.append(
                    {
                        "name": name,
                        "type": self._GLINER_TYPE_MAP.get(p["label"].lower(), "CONCEPT"),
                        "description": "",
                    }
                )
            return entities[:20]
        except Exception:
            return []

    def _extract_entities_light(self, text: str) -> list[dict]:
        """Lightweight entity extraction using regex noun-phrase heuristics (no LLM).

        Picks capitalized multi-word phrases (2–4 tokens) from the text.
        Returns up to 20 entities with type=CONCEPT and empty description.
        """
        import re as _re

        pattern = _re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\b")
        seen: set = set()
        entities: list[dict] = []
        for match in pattern.finditer(text):
            phrase = match.group(1)
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                entities.append({"name": phrase, "type": "CONCEPT", "description": ""})
            if len(entities) >= 20:
                break
        return entities

    def _extract_entities(self, text: str) -> list[dict]:
        """Extract named entities from text using the LLM.

        Returns a list of dicts with shape:
          {"name": str, "type": str, "description": str}
        Returns an empty list on failure or when the LLM produces no output.
        """
        import time as _time

        _t0 = _time.perf_counter()
        _depth = getattr(self.config, "graph_rag_depth", "standard")
        _ner_backend = getattr(self.config, "graph_rag_ner_backend", "llm")
        _cache_key = f"{_depth}|{_ner_backend}|{self._gr_text_hash(text[:3000])}"
        if getattr(self.config, "graph_rag_extraction_cache", True):
            _cached = self._gr_cache_get("entities", _cache_key)
            if _cached is not None:
                self._gr_log_profile("extract_entities(cache_hit)", _time.perf_counter() - _t0)
                return _cached

        # A3: light tier — skip LLM entirely
        if _depth == "light":
            _out = self._extract_entities_light(text)
            self._gr_cache_put("entities", _cache_key, _out)
            self._gr_log_profile("extract_entities(light)", _time.perf_counter() - _t0)
            return _out

        # GLiNER fast-path — skip LLM for NER when backend is "gliner"
        if _ner_backend == "gliner":
            _out = self._extract_entities_gliner(text)
            self._gr_cache_put("entities", _cache_key, _out)
            self._gr_log_profile("extract_entities(gliner)", _time.perf_counter() - _t0)
            return _out

        prompt = (
            "Extract the key named entities from the following text.\n"
            "For each entity output one line:\n"
            "  ENTITY_NAME | ENTITY_TYPE | one-sentence description\n"
            "ENTITY_TYPE must be one of: PERSON, ORGANIZATION, GEO, EVENT, CONCEPT, PRODUCT\n"
            "No bullets, numbering, or extra text. If no entities, output nothing.\n\n"
            + text[:3000]
        )
        try:
            raw = self._gr_llm_complete_cached(
                "extract_entities_llm",
                prompt,
                system_prompt="You are a named entity extraction specialist.",
            )
            entities = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3:
                    ent_type = parts[1].upper()
                    if ent_type not in self._VALID_ENTITY_TYPES:
                        ent_type = "CONCEPT"
                    entities.append(
                        {
                            "name": parts[0],
                            "type": ent_type,
                            "description": parts[2],
                        }
                    )
                elif len(parts) == 2:
                    # Old 2-column format — no type
                    entities.append(
                        {
                            "name": parts[0],
                            "type": "UNKNOWN",
                            "description": parts[1],
                        }
                    )
                else:
                    entities.append(
                        {
                            "name": parts[0],
                            "type": "UNKNOWN",
                            "description": "",
                        }
                    )
            _out = entities[:20]
            self._gr_cache_put("entities", _cache_key, _out)
            self._gr_log_profile("extract_entities(llm)", _time.perf_counter() - _t0)
            return _out
        except Exception:
            self._gr_log_profile("extract_entities(error)", _time.perf_counter() - _t0)
            return []

    def _extract_relations(self, text: str) -> list[dict]:
        """Extract SUBJECT | RELATION | OBJECT | description quads from text via the LLM.

        Returns up to 15 relation dicts with shape
        {"subject": str, "relation": str, "object": str, "description": str}.
        Returns an empty list on failure or when the LLM produces no output.
        """
        import time as _time

        _t0 = _time.perf_counter()
        _depth = getattr(self.config, "graph_rag_depth", "standard")
        _rel_backend = getattr(self.config, "graph_rag_relation_backend", "llm")
        _cache_key = f"{_depth}|{_rel_backend}|{self._gr_text_hash(text[:3000])}"
        if getattr(self.config, "graph_rag_extraction_cache", True):
            _cached = self._gr_cache_get("relations", _cache_key)
            if _cached is not None:
                self._gr_log_profile("extract_relations(cache_hit)", _time.perf_counter() - _t0)
                return _cached

        # A3: light tier skips all relation extraction
        if _depth == "light":
            self._gr_cache_put("relations", _cache_key, [])
            self._gr_log_profile("extract_relations(light_skip)", _time.perf_counter() - _t0)
            return []

        # REBEL fast-path — skip LLM when backend is "rebel"
        if _rel_backend == "rebel":
            _out = self._extract_relations_rebel(text)
            self._gr_cache_put("relations", _cache_key, _out)
            self._gr_log_profile("extract_relations(rebel)", _time.perf_counter() - _t0)
            return _out

        prompt = (
            "Extract key relationships from the following text.\n"
            "For each relationship output one line:\n"
            "  SUBJECT | RELATION | OBJECT | one-sentence description | strength (1-10)\n"
            "Strength: 1=weak/incidental, 10=core/defining. "
            "No bullets or extra text. If no clear relationships, output nothing.\n\n" + text[:3000]
        )
        try:
            raw = self._gr_llm_complete_cached(
                "extract_relations_llm",
                prompt,
                system_prompt="You are a knowledge graph extraction specialist.",
            )
            triples = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 5:
                    try:
                        strength = max(1, min(10, int(parts[4])))
                    except (ValueError, IndexError):
                        strength = 5
                    triples.append(
                        {
                            "subject": parts[0],
                            "relation": parts[1],
                            "object": parts[2],
                            "description": parts[3],
                            "strength": strength,
                        }
                    )
                elif len(parts) >= 4:
                    triples.append(
                        {
                            "subject": parts[0],
                            "relation": parts[1],
                            "object": parts[2],
                            "description": parts[3],
                            "strength": 5,
                        }
                    )
                elif len(parts) == 3 and all(parts):
                    triples.append(
                        {
                            "subject": parts[0],
                            "relation": parts[1],
                            "object": parts[2],
                            "description": "",
                            "strength": 5,
                        }
                    )
            _out = triples[:15]
            self._gr_cache_put("relations", _cache_key, _out)
            self._gr_log_profile("extract_relations(llm)", _time.perf_counter() - _t0)
            return _out
        except Exception:
            self._gr_log_profile("extract_relations(error)", _time.perf_counter() - _t0)
            return []

    def _embed_entities(self, entity_keys: list) -> None:
        """Embed entity descriptions; store in _entity_embeddings (Item 5)."""
        to_embed = []
        for key in entity_keys:
            node = self._entity_graph.get(key, {})
            if isinstance(node, dict):
                desc = node.get("description", "")
                if desc and key not in self._entity_embeddings:
                    to_embed.append((key, f"{key}: {desc}"))
        if not to_embed:
            return
        keys, texts = zip(*to_embed)
        try:
            vectors = self.embedding.embed(list(texts))
            for k, v in zip(keys, vectors):
                self._entity_embeddings[k] = v if isinstance(v, list) else list(v)
            self._save_entity_embeddings()
            logger.info(f"GraphRAG: Embedded {len(to_embed)} entity descriptions.")
        except Exception as e:
            logger.debug(f"Entity embedding failed: {e}")

    def _match_entities_by_embedding(self, query: str, top_k: int = 5) -> list:
        """Return entity keys matching the query by embedding cosine similarity (Item 5)."""
        if not self._entity_embeddings:
            return []
        try:
            import numpy as np

            q_vec = np.array(self.embedding.embed_query(query))
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return []
            scored = []
            for entity_key, e_vec in self._entity_embeddings.items():
                ev = np.array(e_vec)
                ev_norm = np.linalg.norm(ev)
                if ev_norm == 0:
                    continue
                sim = float(np.dot(q_vec, ev) / (q_norm * ev_norm))
                scored.append((sim, entity_key))
            threshold = getattr(self.config, "graph_rag_entity_match_threshold", 0.5)
            scored = [(s, k) for s, k in scored if s >= threshold]
            scored.sort(reverse=True)
            return [k for _, k in scored[:top_k]]
        except Exception as e:
            logger.debug(f"Entity embedding match failed: {e}")
            return []

    def _extract_claims(self, text: str) -> list:
        """Extract factual claims from text (Item 11). Returns list of claim dicts."""
        prompt = (
            "Extract factual claims from the following text.\n"
            "For each claim output one line:\n"
            "  SUBJECT | OBJECT | CLAIM_TYPE | STATUS | DESCRIPTION | START_DATE | END_DATE | SOURCE_QUOTE\n"
            "  STATUS must be: TRUE, FALSE, or SUSPECTED\n"
            "  CLAIM_TYPE examples: acquisition, partnership, product_launch, regulatory_action, financial\n"
            "  START_DATE and END_DATE: ISO8601 date (YYYY-MM-DD) or 'unknown' if not mentioned\n"
            "  SOURCE_QUOTE: verbatim short quote from the text (max 100 chars) or 'none'\n"
            "No bullets or extra text. If no clear claims, output nothing.\n\n" + text[:3000]
        )
        try:
            raw = self._gr_llm_complete_cached(
                "extract_claims_llm",
                prompt,
                system_prompt="You are a fact extraction specialist.",
            )
            claims = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 8:
                    status = parts[3].upper()
                    if status not in ("TRUE", "FALSE", "SUSPECTED"):
                        status = "SUSPECTED"
                    claims.append(
                        {
                            "subject": parts[0],
                            "object": parts[1],
                            "type": parts[2],
                            "status": status,
                            "description": parts[4],
                            "start_date": parts[5] if parts[5] != "unknown" else None,
                            "end_date": parts[6] if parts[6] != "unknown" else None,
                            "source_text": parts[7] if parts[7] != "none" else None,
                            "text_unit_id": None,  # filled in by caller
                        }
                    )
                elif len(parts) >= 5:
                    status = parts[3].upper()
                    if status not in ("TRUE", "FALSE", "SUSPECTED"):
                        status = "SUSPECTED"
                    claims.append(
                        {
                            "subject": parts[0],
                            "object": parts[1],
                            "type": parts[2],
                            "status": status,
                            "description": parts[4],
                            "start_date": None,
                            "end_date": None,
                            "source_text": None,
                            "text_unit_id": None,
                        }
                    )
            return claims[:10]
        except Exception:
            return []

    def _resolve_entity_aliases(self) -> int:
        """Merge semantically equivalent entity nodes into a single canonical node (P1).

        Uses cosine similarity on entity-name embeddings from the active embedding model.
        Entities whose name-embeddings exceed ``graph_rag_entity_resolve_threshold`` are
        grouped; within each group the node with the most chunk_ids becomes canonical and
        all alias nodes are merged into it (chunk_ids, description, relations).

        Returns the number of alias nodes merged (0 if nothing to merge).
        """
        import numpy as np

        threshold = getattr(self.config, "graph_rag_entity_resolve_threshold", 0.92)
        max_entities = getattr(self.config, "graph_rag_entity_resolve_max", 5000)

        keys = [k for k, v in self._entity_graph.items() if isinstance(v, dict)]
        n = len(keys)
        if n < 2:
            return 0
        if n > max_entities:
            logger.warning(
                "GraphRAG entity resolution: entity graph has %d nodes (limit=%d). "
                "Skipping alias resolution — increase graph_rag_entity_resolve_max to enable.",
                n,
                max_entities,
            )
            return 0

        # Embed entity names (not descriptions) — aliases share similar surface forms
        try:
            raw_emb = self.embedding.embed(keys)
        except Exception as exc:
            logger.warning("GraphRAG entity resolution: embedding failed (%s). Skipping.", exc)
            return 0

        emb = np.array(raw_emb, dtype=np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        emb = emb / norms  # unit-normalise for cosine via dot product

        # Union-Find for grouping similar entities
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: int, b: int) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[rb] = ra

        # O(n²) pairwise similarity — acceptable for n ≤ max_entities (default 5 000)
        sim = emb @ emb.T  # shape (n, n)
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    _union(i, j)

        # Collect groups
        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(_find(i), []).append(i)

        merged = 0
        for members in groups.values():
            if len(members) < 2:
                continue
            # Canonical = entity with most chunk_ids (highest coverage)
            canon_idx = max(
                members,
                key=lambda i: len(self._entity_graph[keys[i]].get("chunk_ids", [])),
            )
            canon_key = keys[canon_idx]
            canon_node = self._entity_graph[canon_key]

            for idx in members:
                if idx == canon_idx:
                    continue
                alias_key = keys[idx]
                alias_node = self._entity_graph[alias_key]

                # Merge chunk_ids
                for cid in alias_node.get("chunk_ids", []):
                    if cid not in canon_node["chunk_ids"]:
                        canon_node["chunk_ids"].append(cid)

                # Inherit description if canonical has none
                if not canon_node.get("description") and alias_node.get("description"):
                    canon_node["description"] = alias_node["description"]

                # Migrate relation_graph entries keyed by the alias
                if alias_key in self._relation_graph:
                    canon_rels = self._relation_graph.setdefault(canon_key, [])
                    canon_rels.extend(self._relation_graph.pop(alias_key))

                # Rewrite subject/object references inside all relation lists
                for rel_list in self._relation_graph.values():
                    for rel in rel_list:
                        if isinstance(rel, dict):
                            if rel.get("subject", "").lower() == alias_key:
                                rel["subject"] = canon_key
                            if rel.get("object", "").lower() == alias_key:
                                rel["object"] = canon_key

                del self._entity_graph[alias_key]
                merged += 1

            # Refresh frequency for canonical
            canon_node["frequency"] = len(canon_node["chunk_ids"])

        if merged > 0:
            logger.info(
                "GraphRAG entity resolution: merged %d alias nodes into canonical entities "
                "(threshold=%.2f, %d nodes remaining).",
                merged,
                threshold,
                len(self._entity_graph),
            )
            self._community_graph_dirty = True

        return merged

    def _canonicalize_entity_descriptions(self) -> None:
        """Synthesize canonical descriptions for entities with multiple descriptions (Item 10)."""
        min_occ = getattr(self.config, "graph_rag_canonicalize_min_occurrences", 3)
        to_canonicalize = {
            k: descs
            for k, descs in self._entity_description_buffer.items()
            if len(set(descs)) > 1 and len(descs) >= min_occ
        }
        if not to_canonicalize:
            return

        logger.info(f"GraphRAG: Canonicalizing descriptions for {len(to_canonicalize)} entities...")

        def _synthesize(args):
            entity_key, descs = args
            unique_descs = list(dict.fromkeys(descs))[:10]  # deduplicate, cap at 10
            node = self._entity_graph.get(entity_key, {})
            entity_type = node.get("type", "UNKNOWN") if isinstance(node, dict) else "UNKNOWN"
            numbered = "\n".join(f"{i + 1}. {d}" for i, d in enumerate(unique_descs))
            prompt = (
                f"Entity: {entity_key}\nType: {entity_type}\n\n"
                "The following descriptions were extracted from different documents:\n"
                f"{numbered}\n\n"
                "Write a single comprehensive description that synthesizes all of the above. "
                "Output only the description, no preamble."
            )
            try:
                result = self._gr_llm_complete_cached(
                    "canonicalize_entity_desc",
                    prompt,
                    system_prompt="You are a knowledge graph curation specialist.",
                )
                return entity_key, result.strip() if result else unique_descs[0]
            except Exception:
                return entity_key, unique_descs[0]

        results = list(self._executor.map(_synthesize, to_canonicalize.items()))
        changed = False
        for entity_key, canonical_desc in results:
            node = self._entity_graph.get(entity_key)
            if isinstance(node, dict) and canonical_desc:
                node["description"] = canonical_desc
                changed = True
        if changed:
            self._save_entity_graph()
        self._entity_description_buffer.clear()

    def _canonicalize_relation_descriptions(self) -> None:
        """Synthesize canonical descriptions for repeated (subject, object) pairs (GAP 3b)."""
        if not getattr(self.config, "graph_rag_canonicalize_relations", False):
            return
        min_occ = getattr(self.config, "graph_rag_canonicalize_relations_min_occurrences", 2)
        to_canonicalize = {
            pair: descs
            for pair, descs in self._relation_description_buffer.items()
            if len(descs) >= min_occ
        }
        if not to_canonicalize:
            return

        def _synthesize(args):
            (src, tgt), descs = args
            unique_descs = list(dict.fromkeys(descs))[:10]
            numbered = "\n".join(f"{i+1}. {d}" for i, d in enumerate(unique_descs))
            prompt = (
                f"Relationship: {src} → {tgt}\n\n"
                f"The following descriptions were extracted from different documents:\n{numbered}\n\n"
                "Write a single comprehensive description that synthesizes all of the above. "
                "Output only the description, no preamble."
            )
            try:
                result = self._gr_llm_complete_cached(
                    "canonicalize_relation_desc",
                    prompt,
                    system_prompt="You are a knowledge graph curation specialist.",
                )
                return (src, tgt), result.strip() if result else unique_descs[0]
            except Exception:
                return (src, tgt), unique_descs[0]

        results = list(self._executor.map(_synthesize, to_canonicalize.items()))
        changed = False
        for (src, tgt), canonical_desc in results:
            if src in self._relation_graph:
                for entry in self._relation_graph[src]:
                    if entry.get("target", "").lower() == tgt and canonical_desc:
                        entry["description"] = canonical_desc
                        changed = True
        if changed:
            self._save_relation_graph()
        self._relation_description_buffer.clear()

    def _prune_entity_graph(self, deleted_ids: set) -> None:
        """Remove deleted chunk IDs from entity graph and relation graph.

        Entries that become empty are deleted entirely.  Persists changes to disk.
        """
        eg_changed = False
        for entity in list(self._entity_graph):
            node = self._entity_graph[entity]
            chunk_ids = node.get("chunk_ids", [])
            after = [d for d in chunk_ids if d not in deleted_ids]
            if len(after) != len(chunk_ids):
                eg_changed = True
                if after:
                    self._entity_graph[entity]["chunk_ids"] = after
                    # Recompute frequency after pruning
                    self._entity_graph[entity]["frequency"] = len(after)
                else:
                    del self._entity_graph[entity]
        if eg_changed:
            self._save_entity_graph()

        rg_changed = False
        for src in list(self._relation_graph):
            before = self._relation_graph[src]
            after = [entry for entry in before if entry.get("chunk_id") not in deleted_ids]
            if len(after) != len(before):
                rg_changed = True
                if after:
                    self._relation_graph[src] = after
                else:
                    del self._relation_graph[src]
        if rg_changed:
            self._save_relation_graph()

        # Prune claims graph (Item 11)
        if hasattr(self, "_claims_graph"):
            claims_changed = False
            for chunk_id in list(self._claims_graph):
                if chunk_id in deleted_ids:
                    del self._claims_graph[chunk_id]
                    claims_changed = True
            if claims_changed:
                self._save_claims_graph()
