from __future__ import annotations

import json
import time
from pathlib import Path

_KEY_MAP = {
    "source": "s",
    "graph_rag_type": "grt",
    "community_key": "ck",
    "level": "lv",
    "dataset_type": "dt",
}
_REV = {v: k for k, v in _KEY_MAP.items()}


def _make_meta(n: int = 120000) -> list[dict]:
    out = []
    for i in range(n):
        out.append(
            {
                "source": f"doc_{i%5000}",
                "graph_rag_type": "community_report" if i % 7 == 0 else "text_unit",
                "community_key": f"0_{i%2000}",
                "level": i % 4,
                "dataset_type": "knowledge",
                "owner": f"u{i%50}",
                "tag": f"t{i%17}",
            }
        )
    return out


def _compact(meta: dict) -> dict:
    return {_KEY_MAP.get(k, k): v for k, v in meta.items()}


def _expand(meta: dict) -> dict:
    return {_REV.get(k, k): v for k, v in meta.items()}


def main() -> int:
    metas = _make_meta()
    t0 = time.perf_counter()
    plain_json = json.dumps(metas, separators=(",", ":"))
    dt_plain = time.perf_counter() - t0

    t1 = time.perf_counter()
    compacted = [_compact(m) for m in metas]
    dt_compact_map = time.perf_counter() - t1

    t2 = time.perf_counter()
    compact_json = json.dumps(compacted, separators=(",", ":"))
    dt_compact_dump = time.perf_counter() - t2

    t3 = time.perf_counter()
    _ = [_expand(m) for m in compacted]
    dt_expand_map = time.perf_counter() - t3

    result = {
        "plain": {
            "json_bytes": len(plain_json.encode("utf-8")),
            "json_dump_seconds": dt_plain,
        },
        "compact": {
            "json_bytes": len(compact_json.encode("utf-8")),
            "map_compact_seconds": dt_compact_map,
            "json_dump_seconds": dt_compact_dump,
            "map_expand_seconds": dt_expand_map,
        },
        "summary": {
            "rows": len(metas),
            "size_reduction_x": len(plain_json.encode("utf-8"))
            / max(len(compact_json.encode("utf-8")), 1),
            "json_dump_speedup_x": dt_plain / max(dt_compact_dump, 1e-12),
        },
    }
    out_path = Path("C:/dev/studio_brain_open/bench_results_tqdb_metadata_compaction_profile.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
