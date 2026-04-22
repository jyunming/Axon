from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from axon.config import AxonConfig
from axon.graph_rag import GraphRagMixin


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        self.calls += 1
        p = prompt.lower()
        if "response format" in p and "json array" in p:
            return '[{"point":"Key community evidence","score":99}]'
        if "using the reports above" in p:
            return "Final synthesized answer."
        if "answer only yes or no" in p:
            return "YES"
        if "extract key relationships" in p:
            return "Alice|works_at|Acme|employment|8"
        if "extract factual claims" in p:
            return "Alice|Acme|employment|TRUE|Alice works at Acme|unknown|unknown|none"
        if "extract the key named entities" in p:
            return "Alice|PERSON|Research lead"
        return "Generic response"


class _Brain(GraphRagMixin):
    pass


def _make_cfg(tmp_dir: Path, **kwargs) -> AxonConfig:
    base = dict(
        bm25_path=str(tmp_dir / "bm25"),
        vector_store_path=str(tmp_dir / "vs"),
        graph_rag=True,
        graph_rag_community=True,
        graph_rag_mode="global",
        graph_rag_community_level=0,
        graph_rag_global_top_points=20,
        graph_rag_global_min_score=0,
    )
    base.update(kwargs)
    return AxonConfig(**base)


def _make_brain(cfg: AxonConfig) -> _Brain:
    b = _Brain()
    b.config = cfg
    b.llm = _FakeLLM()
    b.embedding = None
    b._executor = type(
        "SyncExecutor",
        (),
        {
            "map": staticmethod(lambda fn, items: list(map(fn, items))),
            "submit": staticmethod(lambda fn, *a, **k: fn(*a, **k)),
        },
    )()
    b._community_rebuild_lock = type(
        "Lock", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: False}
    )()
    b._community_levels = {0: {"alice": 0}}
    b._community_hierarchy = {"0_0": None}
    b._community_children = {}
    b._community_summaries = {
        "0_0": {
            "title": "Graph Community",
            "summary": "Alice is central to the knowledge graph operations.",
            "full_content": (
                "Alice is central to the knowledge graph operations. "
                "She maintains relation extraction and local search optimization."
            ),
            "rank": 9.0,
            "level": 0,
            "member_hash": "x",
        }
    }
    b._entity_graph = {"alice": {"description": "lead", "chunk_ids": ["c1"], "type": "PERSON"}}
    b._relation_graph = {"alice": [{"target": "acme", "relation": "works_at", "chunk_id": "c1"}]}
    b._claims_graph = {}
    b._text_unit_relation_map = {"c1": ["r1"]}
    b._entity_description_buffer = {}
    b._relation_description_buffer = {}
    return b


def bench_llm_cache_global() -> dict:
    out = {}
    for enabled in (False, True):
        with tempfile.TemporaryDirectory(prefix="axon_cost_") as tmp_str:
            tmp = Path(tmp_str)
            cfg = _make_cfg(
                tmp,
                graph_rag_llm_cache=enabled,
                graph_rag_llm_cache_size=2000,
                graph_rag_global_max_map_chunks=8,
                graph_rag_global_answer_cache=False,
                graph_rag_global_map_cache=False,
            )
            brain = _make_brain(cfg)
            t0 = time.perf_counter()
            for _ in range(40):
                _ = brain._global_search_map_reduce("What are the key graph findings?", cfg)
            dt = time.perf_counter() - t0
            out["cache_on" if enabled else "cache_off"] = {
                "seconds_total": dt,
                "llm_calls": brain.llm.calls,
            }
    out["llm_call_reduction_x"] = out["cache_off"]["llm_calls"] / max(
        out["cache_on"]["llm_calls"], 1
    )
    out["speedup"] = out["cache_off"]["seconds_total"] / max(
        out["cache_on"]["seconds_total"], 1e-12
    )
    return out


def bench_reduce_skip() -> dict:
    out = {}
    for enabled in (False, True):
        with tempfile.TemporaryDirectory(prefix="axon_cost_") as tmp_str:
            tmp = Path(tmp_str)
            cfg = _make_cfg(
                tmp,
                graph_rag_llm_cache=False,
                graph_rag_global_answer_cache=False,
                graph_rag_global_map_cache=False,
                graph_rag_global_reduce_skip_if_top_points_le=(1 if enabled else 0),
                graph_rag_global_reduce_skip_if_top_score_gte=95.0,
                graph_rag_global_max_map_chunks=4,
            )
            brain = _make_brain(cfg)
            t0 = time.perf_counter()
            for _ in range(40):
                _ = brain._global_search_map_reduce("What are the key graph findings?", cfg)
            dt = time.perf_counter() - t0
            out["skip_on" if enabled else "skip_off"] = {
                "seconds_total": dt,
                "llm_calls": brain.llm.calls,
            }
    out["llm_call_reduction_x"] = out["skip_off"]["llm_calls"] / max(out["skip_on"]["llm_calls"], 1)
    out["speedup"] = out["skip_off"]["seconds_total"] / max(out["skip_on"]["seconds_total"], 1e-12)
    return out


def bench_global_answer_cache() -> dict:
    out = {}
    for enabled in (False, True):
        with tempfile.TemporaryDirectory(prefix="axon_cost_") as tmp_str:
            tmp = Path(tmp_str)
            cfg = _make_cfg(
                tmp,
                graph_rag_llm_cache=False,
                graph_rag_global_answer_cache=enabled,
                graph_rag_global_answer_cache_size=500,
                graph_rag_global_map_cache=False,
                graph_rag_global_max_map_chunks=8,
                graph_rag_global_reduce_skip_if_top_points_le=0,
            )
            brain = _make_brain(cfg)
            t0 = time.perf_counter()
            for _ in range(40):
                _ = brain._global_search_map_reduce("What are the key graph findings?", cfg)
            dt = time.perf_counter() - t0
            out["answer_cache_on" if enabled else "answer_cache_off"] = {
                "seconds_total": dt,
                "llm_calls": brain.llm.calls,
            }
    out["llm_call_reduction_x"] = out["answer_cache_off"]["llm_calls"] / max(
        out["answer_cache_on"]["llm_calls"], 1
    )
    out["speedup"] = out["answer_cache_off"]["seconds_total"] / max(
        out["answer_cache_on"]["seconds_total"], 1e-12
    )
    return out


def bench_persist_if_changed() -> dict:
    with tempfile.TemporaryDirectory(prefix="axon_cost_") as tmp_str:
        tmp = Path(tmp_str)
        cfg = _make_cfg(tmp)
        brain = _make_brain(cfg)
        # warm write
        brain._save_entity_graph()
        t0 = time.perf_counter()
        for _ in range(300):
            brain._save_entity_graph()
        dt_cached = time.perf_counter() - t0

        # baseline direct-write simulation for comparison
        import json as _json

        p = Path(cfg.bm25_path) / ".entity_graph_baseline.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        t1 = time.perf_counter()
        for _ in range(300):
            p.write_text(_json.dumps(brain._entity_graph), encoding="utf-8")
        dt_baseline = time.perf_counter() - t1

        return {
            "persist_if_changed_seconds": dt_cached,
            "baseline_write_every_time_seconds": dt_baseline,
            "speedup": dt_baseline / max(dt_cached, 1e-12),
        }


def bench_global_map_cache() -> dict:
    out = {}
    for enabled in (False, True):
        with tempfile.TemporaryDirectory(prefix="axon_cost_") as tmp_str:
            tmp = Path(tmp_str)
            cfg = _make_cfg(
                tmp,
                graph_rag_llm_cache=False,
                graph_rag_global_answer_cache=False,
                graph_rag_global_map_cache=enabled,
                graph_rag_global_map_cache_size=2000,
                graph_rag_global_max_map_chunks=8,
                graph_rag_global_reduce_skip_if_top_points_le=0,
            )
            brain = _make_brain(cfg)
            t0 = time.perf_counter()
            for _ in range(40):
                _ = brain._global_search_map_reduce("What are the key graph findings?", cfg)
            dt = time.perf_counter() - t0
            out["map_cache_on" if enabled else "map_cache_off"] = {
                "seconds_total": dt,
                "llm_calls": brain.llm.calls,
            }
    out["llm_call_reduction_x"] = out["map_cache_off"]["llm_calls"] / max(
        out["map_cache_on"]["llm_calls"], 1
    )
    out["speedup"] = out["map_cache_off"]["seconds_total"] / max(
        out["map_cache_on"]["seconds_total"], 1e-12
    )
    return out


def main() -> int:
    result = {
        "llm_cache_global_map_reduce": bench_llm_cache_global(),
        "global_reduce_skip": bench_reduce_skip(),
        "global_answer_cache": bench_global_answer_cache(),
        "global_map_cache": bench_global_map_cache(),
        "persist_write_coalescing": bench_persist_if_changed(),
    }
    out_path = Path("C:/dev/studio_brain_open/bench_results_graphrag_cost_profile.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
