from __future__ import annotations

import json
import random
import tempfile
import time
from pathlib import Path

from axon.config import AxonConfig
from axon.graph_rag import GraphRagMixin


class _VS:
    def get_by_ids(self, ids):
        return []


class _Brain(GraphRagMixin):
    pass


def _make_graph(n_entities: int = 8000, rel_per_entity: int = 6):
    rng = random.Random(42)
    ent = {}
    rel = {}
    for i in range(n_entities):
        name = f"e{i}"
        ent[name] = {"description": f"Entity {i}", "chunk_ids": [], "type": "ENTITY"}
        rows = []
        for j in range(rel_per_entity):
            rows.append(
                {
                    "target": f"e{rng.randrange(n_entities)}",
                    "relation": f"r{(i+j)%53}",
                    "chunk_id": f"c{i}_{j}",
                }
            )
        rel[name] = rows
    return ent, rel


def _make_brain(cfg: AxonConfig, ent: dict, rel: dict):
    b = _Brain()
    b.config = cfg
    b._entity_graph = ent
    b._relation_graph = rel
    b._claims_graph = {}
    b._text_unit_relation_map = {}
    b._community_hierarchy = {}
    b._community_children = {}
    b._entity_description_buffer = {}
    b._relation_description_buffer = {}
    b._community_levels = {}
    b._community_summaries = {}
    b.vector_store = _VS()
    return b


def _run(enabled: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="axon_incoming_count_") as tmp_str:
        tmp = Path(tmp_str)
        cfg = AxonConfig(
            bm25_path=str(tmp / "bm25"),
            vector_store_path=str(tmp / "vs"),
            graph_rag=True,
            graph_rag_mode="local",
            graph_rag_local_entity_degree_fast=True,
            graph_rag_local_cached_incoming=True,
            graph_rag_local_cached_incoming_counts=enabled,
            graph_rag_local_top_k_entities=25,
            graph_rag_local_top_k_relationships=25,
        )
        ent, rel = _make_graph()
        b = _make_brain(cfg, ent, rel)
        matched = [f"e{i}" for i in range(120)]
        q = "entity relation impact"
        t0 = time.perf_counter()
        for _ in range(80):
            _ = b._local_search_context(q, matched, cfg)
        dt = time.perf_counter() - t0
        return {"seconds_total": dt}


def main() -> int:
    off = _run(False)
    on = _run(True)
    result = {
        "count_cache_off": off,
        "count_cache_on": on,
        "summary": {
            "speedup_x": off["seconds_total"] / max(on["seconds_total"], 1e-12),
        },
    }
    out = Path("C:/dev/studio_brain_open/bench_results_graphrag_incoming_count_cache_profile.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
