from __future__ import annotations

import json
import random
import tempfile
import time
from pathlib import Path


def _make_relation_graph(n_entities: int = 8000, rel_per_entity: int = 6) -> dict:
    rng = random.Random(42)
    g = {}
    for i in range(n_entities):
        src = f"entity_{i}"
        entries = []
        for j in range(rel_per_entity):
            tgt = f"entity_{rng.randrange(n_entities)}"
            rel = f"rel_{(i + j) % 37}"
            cid = f"c_{i}_{j}"
            entries.append({"target": tgt, "relation": rel, "chunk_id": cid})
        g[src] = entries
    return g


def _to_compact(g: dict) -> dict:
    out = {}
    for src, entries in g.items():
        out[src] = [[e["target"], e["relation"], e["chunk_id"]] for e in entries]
    return {"format": "rg_rel_v2", "g": out}


def _bench_write_read(payload: dict, path: Path) -> dict:
    t0 = time.perf_counter()
    text = json.dumps(payload, separators=(",", ":"))
    write_prepare = time.perf_counter() - t0

    t1 = time.perf_counter()
    path.write_text(text, encoding="utf-8")
    write_io = time.perf_counter() - t1

    t2 = time.perf_counter()
    raw = json.loads(path.read_text(encoding="utf-8"))
    read_parse = time.perf_counter() - t2

    return {
        "bytes": path.stat().st_size,
        "prepare_seconds": write_prepare,
        "write_io_seconds": write_io,
        "read_parse_seconds": read_parse,
        "top_keys": len(raw) if isinstance(raw, dict) else 0,
    }


def main() -> int:
    graph = _make_relation_graph()
    legacy = graph
    compact = _to_compact(graph)
    tmp = Path(tempfile.mkdtemp(prefix="axon_rel_store_"))
    legacy_path = tmp / "relation_graph_legacy.json"
    compact_path = tmp / "relation_graph_compact.json"

    m_legacy = _bench_write_read(legacy, legacy_path)
    m_compact = _bench_write_read(compact, compact_path)

    result = {
        "legacy": m_legacy,
        "compact_v2": m_compact,
        "summary": {
            "size_reduction_x": m_legacy["bytes"] / max(m_compact["bytes"], 1),
            "prepare_speedup_x": m_legacy["prepare_seconds"]
            / max(m_compact["prepare_seconds"], 1e-12),
            "read_parse_speedup_x": m_legacy["read_parse_seconds"]
            / max(m_compact["read_parse_seconds"], 1e-12),
        },
    }
    out_path = Path("C:/dev/studio_brain_open/bench_results_graphrag_relation_storage_profile.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
