from __future__ import annotations

import json
import random
import tempfile
import time
import tracemalloc
from pathlib import Path


def _write_shards(root: Path, n: int = 16, rows: int = 3500) -> list[Path]:
    rng = random.Random(42)
    paths = []
    for i in range(n):
        g = {}
        for j in range(rows):
            src = f"e{i}_{j}"
            g[src] = [
                [f"e{rng.randrange(n*rows*3)}", f"r{(i+j)%43}", f"c{i}_{j}_{k}"] for k in range(3)
            ]
        p = root / f".relation_graph.shard.{i:03d}.json"
        p.write_text(
            json.dumps({"format": "rg_rel_v2", "g": g}, separators=(",", ":")), encoding="utf-8"
        )
        paths.append(p)
    return paths


def _decode(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for key, value in raw.get("g", {}).items():
        out[key] = [{"target": x[0], "relation": x[1], "chunk_id": x[2]} for x in value]
    return out


def _merge_materialized(paths: list[Path]) -> tuple[float, int]:
    tracemalloc.start()
    t0 = time.perf_counter()
    parts = [_decode(p) for p in paths]
    merged = {}
    for part in parts:
        for k, v in part.items():
            merged.setdefault(k, []).extend(v)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dt, int(peak)


def _merge_stream(paths: list[Path]) -> tuple[float, int]:
    tracemalloc.start()
    t0 = time.perf_counter()
    merged = {}
    for p in paths:
        part = _decode(p)
        for k, v in part.items():
            merged.setdefault(k, []).extend(v)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return dt, int(peak)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="axon_shard_merge_stream_") as tmp_str:
        root = Path(tmp_str)
        paths = _write_shards(root)
        mat_dt, mat_peak = _merge_materialized(paths)
        str_dt, str_peak = _merge_stream(paths)
    result = {
        "materialized_merge": {"seconds": mat_dt, "peak_bytes": mat_peak},
        "stream_merge": {"seconds": str_dt, "peak_bytes": str_peak},
        "summary": {
            "peak_memory_reduction_x": mat_peak / max(str_peak, 1),
            "speedup_x": mat_dt / max(str_dt, 1e-12),
        },
    }
    out = Path("C:/dev/studio_brain_open/bench_results_graphrag_shard_merge_stream_profile.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
