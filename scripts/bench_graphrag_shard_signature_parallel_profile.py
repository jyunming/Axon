from __future__ import annotations

import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _make_buckets(
    shard_count: int = 16, src_per_shard: int = 700, rel_per_src: int = 6
) -> list[dict]:
    rng = random.Random(42)
    buckets = []
    for s in range(shard_count):
        b = {}
        for i in range(src_per_shard):
            src = f"e{s}_{i}"
            rows = []
            for j in range(rel_per_src):
                rows.append(
                    {
                        "target": f"e{rng.randrange(shard_count * src_per_shard)}",
                        "relation": f"r{(i+j)%41}",
                        "chunk_id": f"c{s}_{i}_{j}",
                    }
                )
            b[src] = rows
        buckets.append(b)
    return buckets


def _sig(bucket: dict) -> str:
    h = hashlib.sha1()
    for src in sorted(bucket.keys()):
        h.update(src.encode("utf-8"))
        entries = bucket[src]
        h.update(str(len(entries)).encode("utf-8"))
        for e in entries:
            h.update(e["target"].encode("utf-8"))
            h.update(e["relation"].encode("utf-8"))
            h.update(e["chunk_id"].encode("utf-8"))
    return h.hexdigest()


def main() -> int:
    buckets = _make_buckets()
    t0 = time.perf_counter()
    seq = [_sig(b) for b in buckets]
    seq_dt = time.perf_counter() - t0

    t1 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as ex:
        par = list(ex.map(_sig, buckets))
    par_dt = time.perf_counter() - t1

    result = {
        "sequential": {"seconds": seq_dt, "count": len(seq)},
        "parallel_4w": {"seconds": par_dt, "count": len(par)},
        "summary": {"speedup_x": seq_dt / max(par_dt, 1e-12), "same_count": len(seq) == len(par)},
    }
    out = Path(
        "C:/dev/studio_brain_open/bench_results_graphrag_shard_signature_parallel_profile.json"
    )
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
