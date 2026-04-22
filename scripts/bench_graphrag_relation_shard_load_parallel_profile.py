from __future__ import annotations

import json
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _make_shard_payload(rows: int = 4200) -> dict:
    rng = random.Random(42)
    g = {}
    for i in range(rows):
        src = f"e{i}"
        g[src] = [[f"e{rng.randrange(rows*8)}", f"r{(i+j)%37}", f"c{i}_{j}"] for j in range(3)]
    return {"format": "rg_rel_v2", "g": g}


def _write_shards(root: Path, n: int = 16) -> list[Path]:
    paths = []
    for i in range(n):
        p = root / f".relation_graph.shard.{i:03d}.json"
        payload = _make_shard_payload()
        p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        paths.append(p)
    manifest = {"format": "rg_rel_shard_v1", "compact": True, "shards": [p.name for p in paths]}
    (root / ".relation_graph.shards.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )
    return paths


def _decode_shard(path: Path) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return 0
    if raw.get("format") == "rg_rel_v2" and isinstance(raw.get("g"), dict):
        total = 0
        for _, value in raw["g"].items():
            if isinstance(value, list):
                total += len(value)
        return total
    return 0


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="axon_rel_load_parallel_") as tmp_str:
        tmp = Path(tmp_str)
        shard_paths = _write_shards(tmp, n=16)

        t0 = time.perf_counter()
        seq_total = 0
        for p in shard_paths:
            seq_total += _decode_shard(p)
        seq_dt = time.perf_counter() - t0

        t1 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as ex:
            par_total = sum(ex.map(_decode_shard, shard_paths))
        par_dt = time.perf_counter() - t1

    result = {
        "sequential": {"seconds": seq_dt, "decoded_entries": seq_total},
        "parallel_4w": {"seconds": par_dt, "decoded_entries": par_total},
        "summary": {
            "speedup_x": seq_dt / max(par_dt, 1e-12),
            "totals_equal": seq_total == par_total,
        },
    }
    out = Path(
        "C:/dev/studio_brain_open/bench_results_graphrag_relation_shard_load_parallel_profile.json"
    )
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
