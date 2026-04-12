from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from axon.config import AxonConfig
from axon.graph_rag import GraphRagMixin


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls += 1
        if "Extract key relationships" in prompt:
            return "Alice|works_at|Acme|employment|5"
        return "Alice|PERSON|Research lead"


class _FakeVectorStore:
    def __init__(self, docs: dict[str, dict], per_call_overhead_s: float = 0.0001) -> None:
        self._docs = docs
        self._overhead = per_call_overhead_s
        self.calls = 0

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        self.calls += 1
        if self._overhead > 0:
            time.sleep(self._overhead)
        out = []
        for cid in ids:
            doc = self._docs.get(cid)
            if doc is not None:
                out.append(doc)
        return out


class _BenchBrain(GraphRagMixin):
    pass


def _make_cfg(**kwargs) -> AxonConfig:
    tmp = Path(tempfile.mkdtemp(prefix="axon_bench_"))
    defaults = {
        "graph_rag": True,
        "graph_rag_mode": "local",
        "graph_rag_depth": "standard",
    }
    defaults.update(kwargs)
    return AxonConfig(
        bm25_path=str(tmp / "bm25"),
        vector_store_path=str(tmp / "vs"),
        **defaults,
    )


def _make_brain(cfg: AxonConfig, n_chunks: int = 2000) -> _BenchBrain:
    brain = _BenchBrain()
    brain.config = cfg
    brain._entity_graph = {
        "alice": {
            "description": "Research lead in graph retrieval systems",
            "chunk_ids": [f"c{i}" for i in range(n_chunks)],
            "type": "PERSON",
            "frequency": n_chunks,
        }
    }
    brain._relation_graph = {
        "alice": [
            {
                "target": "knowledge_graph",
                "relation": "maintains",
                "chunk_id": "c1",
                "description": "Alice maintains a production knowledge graph stack",
                "weight": 1.0,
            }
        ]
    }
    brain._community_levels = {0: {"alice": 0}}
    brain._community_summaries = {
        "0_0": {
            "title": "Graph Platform",
            "summary": "Entity retrieval, relation extraction, and query routing stack",
            "rank": 5.0,
            "level": 0,
        }
    }
    brain._claims_graph = {}
    brain._text_unit_relation_map = {f"c{i}": ["r1"] for i in range(n_chunks)}
    brain._entity_description_buffer = {}
    brain._relation_description_buffer = {}
    brain._community_children = {}
    brain._community_hierarchy = {}
    brain._graph_rag_cache = {}
    brain._incoming_rel_sig = None

    docs = {
        f"c{i}": {
            "id": f"c{i}",
            "text": f"Chunk {i} discusses entity and relation features in depth.",
        }
        for i in range(n_chunks)
    }
    brain.vector_store = _FakeVectorStore(docs)
    brain._own_vector_store = SimpleNamespace()
    brain.llm = _FakeLLM()
    return brain


def _wire_dense_relations(
    brain: _BenchBrain, n_entities: int = 40, edges_per_entity: int = 60
) -> list[str]:
    ents = [f"e{i}" for i in range(n_entities)]
    brain._entity_graph = {}
    brain._relation_graph = {}
    for i, ent in enumerate(ents):
        chunk_ids = [f"c{i}_{j}" for j in range(8)]
        brain._entity_graph[ent] = {
            "description": f"Entity {ent} in dense relation benchmark",
            "chunk_ids": chunk_ids,
            "type": "CONCEPT",
            "frequency": len(chunk_ids),
        }
        outs = []
        for j in range(edges_per_entity):
            tgt = ents[(i + j + 1) % n_entities]
            outs.append(
                {
                    "target": tgt,
                    "relation": "links_to",
                    "chunk_id": chunk_ids[j % len(chunk_ids)],
                    "description": f"{ent} links_to {tgt}",
                    "weight": 1.0,
                }
            )
        brain._relation_graph[ent] = outs

    docs = {}
    for ent, node in brain._entity_graph.items():
        for cid in node["chunk_ids"]:
            docs[cid] = {"id": cid, "text": f"Text unit {cid} for {ent} dense benchmark."}
    brain.vector_store = _FakeVectorStore(docs, per_call_overhead_s=0.0)
    brain._text_unit_relation_map = {
        cid: ["r1", "r2"] for node in brain._entity_graph.values() for cid in node["chunk_ids"]
    }
    brain._community_levels = {}
    brain._community_summaries = {}
    brain._claims_graph = {}
    return ents


def _bench_local_context(iterations: int = 20) -> dict:
    cfg = _make_cfg(graph_rag_local_max_context_tokens=8000)
    out = {}
    for batch_fetch in (False, True):
        cfg.graph_rag_local_batch_fetch = batch_fetch
        brain = _make_brain(cfg, n_chunks=2500)
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = brain._local_search_context("who is alice?", ["alice"], cfg)
        dt = time.perf_counter() - t0
        out["batch" if batch_fetch else "single"] = {
            "seconds_total": dt,
            "seconds_avg": dt / iterations,
            "vector_store_calls": brain.vector_store.calls,
        }
    out["speedup_avg"] = out["single"]["seconds_avg"] / max(out["batch"]["seconds_avg"], 1e-12)
    return out


def _bench_relation_support_fast(iterations: int = 80) -> dict:
    cfg = _make_cfg(
        graph_rag_local_max_context_tokens=12000,
        graph_rag_local_top_k_entities=30,
        graph_rag_local_top_k_relationships=40,
        graph_rag_local_batch_fetch=True,
    )
    out = {}
    for enabled in (False, True):
        cfg.graph_rag_local_relation_support_fast = enabled
        brain = _make_brain(cfg, n_chunks=50)
        matched = _wire_dense_relations(brain, n_entities=45, edges_per_entity=70)
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = brain._local_search_context("dense relation query", matched, cfg)
        dt = time.perf_counter() - t0
        out["fast_on" if enabled else "fast_off"] = {
            "seconds_total": dt,
            "seconds_avg": dt / iterations,
            "vector_store_calls": brain.vector_store.calls,
        }
    out["speedup_avg"] = out["fast_off"]["seconds_avg"] / max(out["fast_on"]["seconds_avg"], 1e-12)
    return out


def _bench_entity_degree_fast(iterations: int = 120) -> dict:
    cfg = _make_cfg(
        graph_rag_local_max_context_tokens=12000,
        graph_rag_local_top_k_entities=35,
        graph_rag_local_top_k_relationships=40,
        graph_rag_local_batch_fetch=True,
        graph_rag_local_relation_support_fast=True,
    )
    out = {}
    for enabled in (False, True):
        cfg.graph_rag_local_entity_degree_fast = enabled
        brain = _make_brain(cfg, n_chunks=50)
        matched = _wire_dense_relations(brain, n_entities=60, edges_per_entity=80)
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = brain._local_search_context("entity degree benchmark", matched, cfg)
        dt = time.perf_counter() - t0
        out["fast_on" if enabled else "fast_off"] = {
            "seconds_total": dt,
            "seconds_avg": dt / iterations,
            "vector_store_calls": brain.vector_store.calls,
        }
    out["speedup_avg"] = out["fast_off"]["seconds_avg"] / max(out["fast_on"]["seconds_avg"], 1e-12)
    return out


def _bench_early_cutoff(iterations: int = 120) -> dict:
    cfg = _make_cfg(
        graph_rag_local_max_context_tokens=2200,
        graph_rag_local_top_k_entities=40,
        graph_rag_local_top_k_relationships=60,
        graph_rag_local_batch_fetch=True,
        graph_rag_local_relation_support_fast=True,
        graph_rag_local_entity_degree_fast=True,
    )
    out = {}
    for enabled in (False, True):
        cfg.graph_rag_local_early_cutoff = enabled
        cfg.graph_rag_local_early_cutoff_factor = 0.2
        brain = _make_brain(cfg, n_chunks=50)
        matched = _wire_dense_relations(brain, n_entities=70, edges_per_entity=90)
        # add richer communities so non-text-unit candidates can saturate budget
        brain._community_levels = {0: {e: i % 8 for i, e in enumerate(matched)}}
        brain._community_summaries = {
            f"0_{i}": {
                "title": f"Community {i}",
                "summary": "Dense graph neighborhood with multiple strong shared relations and entities. "
                "Supports context without requiring many raw text units.",
                "rank": float(10 - i),
                "level": 0,
            }
            for i in range(8)
        }
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = brain._local_search_context("early cutoff benchmark", matched, cfg)
        dt = time.perf_counter() - t0
        out["cutoff_on" if enabled else "cutoff_off"] = {
            "seconds_total": dt,
            "seconds_avg": dt / iterations,
            "vector_store_calls": brain.vector_store.calls,
        }
    out["speedup_avg"] = out["cutoff_off"]["seconds_avg"] / max(
        out["cutoff_on"]["seconds_avg"], 1e-12
    )
    return out


def _bench_extraction_cache(iterations: int = 2000) -> dict:
    out = {}
    sample_text = "Alice leads graph retrieval and optimization."
    for enabled in (False, True):
        cfg = _make_cfg(
            graph_rag_extraction_cache=enabled,
            graph_rag_extraction_cache_size=5000,
            graph_rag_depth="standard",
            graph_rag_llm_cache=False,
        )
        brain = _make_brain(cfg, n_chunks=50)
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = brain._extract_entities(sample_text)
            _ = brain._extract_relations(sample_text)
        dt = time.perf_counter() - t0
        key = "cache_on" if enabled else "cache_off"
        out[key] = {
            "seconds_total": dt,
            "seconds_avg_pair": dt / iterations,
            "llm_calls": brain.llm.calls,
        }
    out["speedup_avg_pair"] = out["cache_off"]["seconds_avg_pair"] / max(
        out["cache_on"]["seconds_avg_pair"], 1e-12
    )
    return out


def main() -> int:
    result = {
        "graph_rag_local_context": _bench_local_context(),
        "graph_rag_relation_support_fast": _bench_relation_support_fast(),
        "graph_rag_entity_degree_fast": _bench_entity_degree_fast(),
        "graph_rag_early_cutoff": _bench_early_cutoff(),
        "graph_rag_extraction_cache": _bench_extraction_cache(),
    }
    out_path = Path("C:/dev/studio_brain_open/bench_results_graphrag_perf.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
