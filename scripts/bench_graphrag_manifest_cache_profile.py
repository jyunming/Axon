from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path


def _make_manifest(tmp: Path, n: int = 8000) -> tuple[Path, Path]:
    shards = [f".relation_graph.shard.{i:05d}.json" for i in range(n)]
    j = tmp / ".relation_graph.shards.json"
    l = tmp / ".relation_graph.shards.lst"
    j.write_text(
        json.dumps({"format": "rg_rel_shard_v1", "compact": True, "shards": shards}),
        encoding="utf-8",
    )
    l.write_text("\n".join(shards) + "\n", encoding="utf-8")
    return j, l


def _load_uncached(j: Path, l: Path, use_list: bool) -> list[str]:
    if use_list:
        return [ln.strip() for ln in l.read_text(encoding="utf-8").splitlines() if ln.strip()]
    raw = json.loads(j.read_text(encoding="utf-8"))
    return [x for x in raw.get("shards", []) if isinstance(x, str)]


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="axon_manifest_cache_") as tmp_str:
        tmp = Path(tmp_str)
        j, l = _make_manifest(tmp)

        t0 = time.perf_counter()
        for _ in range(200):
            _ = _load_uncached(j, l, use_list=False)
        uncached_dt = time.perf_counter() - t0

        sig = (j.stat().st_mtime_ns, l.stat().st_mtime_ns)
        cached_names = _load_uncached(j, l, use_list=False)
        t1 = time.perf_counter()
        for _ in range(200):
            cur_sig = (j.stat().st_mtime_ns, l.stat().st_mtime_ns)
            if cur_sig == sig:
                _ = cached_names
            else:
                _ = _load_uncached(j, l, use_list=False)
        cached_dt = time.perf_counter() - t1

    result = {
        "uncached_json_loop_seconds": uncached_dt,
        "cached_sig_loop_seconds": cached_dt,
        "summary": {"speedup_x": uncached_dt / max(cached_dt, 1e-12), "entries": len(cached_names)},
    }
    out = Path("C:/dev/studio_brain_open/bench_results_graphrag_manifest_cache_profile.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
