from __future__ import annotations

import json
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _make_shards(shard_count: int = 16, rows_per_shard: int = 4500) -> list[dict]:
    rng = random.Random(42)
    shards: list[dict] = []
    for s in range(shard_count):
        g = {}
        for i in range(rows_per_shard):
            src = f"e{s}_{i}"
            g[src] = [
                [f"e{rng.randrange(100000)}", f"r{(i+j)%47}", f"c{s}_{i}_{j}"] for j in range(3)
            ]
        shards.append({"format": "rg_rel_v2", "g": g})
    return shards


def _write(path: Path, payload: dict) -> int:
    blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    path.write_bytes(blob)
    return len(blob)


def main() -> int:
    shards = _make_shards()
    with tempfile.TemporaryDirectory(prefix="axon_rel_shard_parallel_") as tmp_str:
        tmp = Path(tmp_str)
        paths = [tmp / f".relation_graph.shard.{i:03d}.json" for i in range(len(shards))]

        t0 = time.perf_counter()
        seq_bytes = 0
        for p, payload in zip(paths, shards):
            seq_bytes += _write(p, payload)
        seq_dt = time.perf_counter() - t0

        t1 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as ex:
            sizes = list(ex.map(lambda pair: _write(pair[0], pair[1]), zip(paths, shards)))
        par_dt = time.perf_counter() - t1
        par_bytes = sum(sizes)

    result = {
        "sequential": {"seconds": seq_dt, "bytes": seq_bytes},
        "parallel_4w": {"seconds": par_dt, "bytes": par_bytes},
        "summary": {
            "speedup_x": seq_dt / max(par_dt, 1e-12),
            "bytes_equal": seq_bytes == par_bytes,
        },
    }
    out = Path(
        "C:/dev/studio_brain_open/bench_results_graphrag_relation_shard_parallel_profile.json"
    )
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
