from __future__ import annotations

import json
import pickle
import random
import tempfile
import time
from pathlib import Path


def _make_shards(root: Path, shard_count: int = 16, rows: int = 3200) -> list[Path]:
    rng = random.Random(42)
    paths = []
    for i in range(shard_count):
        g = {}
        for j in range(rows):
            src = f"e{i}_{j}"
            g[src] = [
                [f"e{rng.randrange(shard_count*rows*3)}", f"r{(i+j)%37}", f"c{i}_{j}_{k}"]
                for k in range(3)
            ]
        p = root / f".relation_graph.shard.{i:03d}.json"
        p.write_text(
            json.dumps({"format": "rg_rel_v2", "g": g}, separators=(",", ":")), encoding="utf-8"
        )
        paths.append(p)
    return paths


def _load_json_merge(paths: list[Path]) -> dict:
    merged = {}
    for p in paths:
        raw = json.loads(p.read_text(encoding="utf-8"))
        for key, value in raw.get("g", {}).items():
            out = [{"target": e[0], "relation": e[1], "chunk_id": e[2]} for e in value]
            merged.setdefault(key, []).extend(out)
    return merged


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="axon_rel_pickle_cache_") as tmp_str:
        root = Path(tmp_str)
        paths = _make_shards(root)

        t0 = time.perf_counter()
        merged = _load_json_merge(paths)
        json_load_dt = time.perf_counter() - t0

        cache_path = root / ".relation_graph.cache.pkl"
        t1 = time.perf_counter()
        with open(cache_path, "wb") as f:
            pickle.dump(merged, f, protocol=4)
        pickle_write_dt = time.perf_counter() - t1

        t2 = time.perf_counter()
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        pickle_load_dt = time.perf_counter() - t2

        result = {
            "json_merge_load": {"seconds": json_load_dt, "entries": len(merged)},
            "pickle_cache_write": {"seconds": pickle_write_dt, "bytes": cache_path.stat().st_size},
            "pickle_cache_load": {"seconds": pickle_load_dt, "entries": len(cached)},
            "summary": {"reload_speedup_x": json_load_dt / max(pickle_load_dt, 1e-12)},
        }
    out = Path("C:/dev/studio_brain_open/bench_results_graphrag_relation_pickle_cache_profile.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
