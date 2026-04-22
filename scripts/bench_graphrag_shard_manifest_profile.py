from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path


def _make_names(n: int = 10000) -> list[str]:
    return [f".relation_graph.shard.{i:05d}.json" for i in range(n)]


def main() -> int:
    names = _make_names()
    with tempfile.TemporaryDirectory(prefix="axon_manifest_bench_") as tmp_str:
        tmp = Path(tmp_str)
        json_path = tmp / ".relation_graph.shards.json"
        lst_path = tmp / ".relation_graph.shards.lst"

        manifest = {"format": "rg_rel_shard_v1", "compact": True, "shards": names}
        json_path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
        lst_path.write_text("\n".join(names) + "\n", encoding="utf-8")

        t0 = time.perf_counter()
        raw = json.loads(json_path.read_text(encoding="utf-8"))
        _json_names = [x for x in raw.get("shards", []) if isinstance(x, str)]
        json_dt = time.perf_counter() - t0

        t1 = time.perf_counter()
        _lst_names = [
            ln.strip() for ln in lst_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        ]
        lst_dt = time.perf_counter() - t1

    result = {
        "json_manifest": {
            "seconds": json_dt,
            "count": len(_json_names),
            "bytes": json_path.stat().st_size,
        },
        "list_manifest": {
            "seconds": lst_dt,
            "count": len(_lst_names),
            "bytes": lst_path.stat().st_size,
        },
        "summary": {
            "parse_speedup_x": json_dt / max(lst_dt, 1e-12),
            "sizes_equal_count": len(_json_names) == len(_lst_names),
        },
    }
    out = Path("C:/dev/studio_brain_open/bench_results_graphrag_shard_manifest_profile.json")
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
