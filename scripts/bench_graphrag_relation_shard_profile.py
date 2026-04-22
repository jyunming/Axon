from __future__ import annotations

import json
import random
import tempfile
import time
from pathlib import Path


def _make_graph(n_entities: int = 10000, rel_per_entity: int = 6) -> dict:
    rng = random.Random(42)
    out = {}
    for i in range(n_entities):
        src = f"entity_{i}"
        rows = []
        for j in range(rel_per_entity):
            rows.append(
                {
                    "target": f"entity_{rng.randrange(n_entities)}",
                    "relation": f"r{(i+j)%47}",
                    "chunk_id": f"c{i}_{j}",
                }
            )
        out[src] = rows
    return out


def _compact_payload(graph: dict) -> dict:
    cg = {}
    for src, entries in graph.items():
        cg[src] = [[e["target"], e["relation"], e["chunk_id"]] for e in entries]
    return {"format": "rg_rel_v2", "g": cg}


def _write_json(path: Path, payload: dict) -> int:
    blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    path.write_bytes(blob)
    return len(blob)


def main() -> int:
    g = _make_graph()
    with tempfile.TemporaryDirectory(prefix="axon_rel_shard_") as tmp_str:
        tmp = Path(tmp_str)

        # Baseline: monolithic compact file.
        mono_path = tmp / ".relation_graph.json"
        t0 = time.perf_counter()
        mono_bytes = _write_json(mono_path, _compact_payload(g))
        mono_full_write = time.perf_counter() - t0

        # Sharded: initial full write (16 shards + manifest).
        shard_count = 16
        keys = sorted(g.keys())
        buckets = [dict() for _ in range(shard_count)]
        for i, k in enumerate(keys):
            buckets[i % shard_count][k] = g[k]
        t1 = time.perf_counter()
        shard_bytes_total = 0
        shard_files = []
        for i, b in enumerate(buckets):
            p = tmp / f".relation_graph.shard.{i:03d}.json"
            shard_files.append(p)
            shard_bytes_total += _write_json(p, _compact_payload(b))
        manifest = {
            "format": "rg_rel_shard_v1",
            "compact": True,
            "shards": [p.name for p in shard_files],
        }
        manifest_bytes = _write_json(tmp / ".relation_graph.shards.json", manifest)
        shard_full_write = time.perf_counter() - t1

        # Incremental update simulation: modify ~1/16 sources.
        updated = dict(g)
        touched = set(keys[::16])
        for k in touched:
            rows = list(updated[k])
            rows.append({"target": "entity_0", "relation": "new_rel", "chunk_id": f"{k}_new"})
            updated[k] = rows

        # Monolithic rewrite after update.
        t2 = time.perf_counter()
        mono_update_bytes = _write_json(mono_path, _compact_payload(updated))
        mono_update_write = time.perf_counter() - t2

        # Sharded rewrite after update: only touched shard 0 in this deterministic partition.
        bucket0 = {k: updated[k] for k in keys[::16]}
        t3 = time.perf_counter()
        shard_update_bytes = _write_json(shard_files[0], _compact_payload(bucket0))
        shard_update_write = time.perf_counter() - t3

    result = {
        "baseline_monolithic": {
            "initial_write_seconds": mono_full_write,
            "initial_bytes": mono_bytes,
            "update_write_seconds": mono_update_write,
            "update_bytes": mono_update_bytes,
        },
        "sharded_16": {
            "initial_write_seconds": shard_full_write,
            "initial_bytes_total": shard_bytes_total + manifest_bytes,
            "update_write_seconds": shard_update_write,
            "update_bytes": shard_update_bytes,
        },
        "summary": {
            "initial_size_overhead_ratio": (shard_bytes_total + manifest_bytes)
            / max(mono_bytes, 1),
            "incremental_write_size_reduction_x": mono_update_bytes / max(shard_update_bytes, 1),
            "incremental_write_time_speedup_x": mono_update_write / max(shard_update_write, 1e-12),
        },
    }

    out = Path("C:/dev/studio_brain_open/bench_results_graphrag_relation_shard_profile.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
