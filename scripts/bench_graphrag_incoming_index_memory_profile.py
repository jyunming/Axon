from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def _make_relation_graph(n_entities: int = 12000, rel_per_entity: int = 6) -> dict:
    rng = random.Random(42)
    out = {}
    for i in range(n_entities):
        src = f"e{i}"
        rows = []
        for j in range(rel_per_entity):
            rows.append(
                {
                    "target": f"e{rng.randrange(n_entities)}",
                    "relation": f"r{(i + j) % 53}",
                    "chunk_id": f"c{i}_{j}",
                    "weight": (j % 5) + 1,
                }
            )
        out[src] = rows
    return out


def _index_legacy(rel: dict) -> dict:
    idx = {}
    for src, entries in rel.items():
        for e in entries:
            tgt = e.get("target", "")
            if not tgt:
                continue
            idx.setdefault(tgt, []).append({**e, "source": src, "direction": "incoming"})
    return idx


def _index_compact(rel: dict) -> dict:
    idx = {}
    for src, entries in rel.items():
        for e in entries:
            tgt = e.get("target", "")
            if not tgt:
                continue
            idx.setdefault(tgt, []).append((src, e))
    return idx


def _json_bytes(obj: object) -> int:
    return len(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


def main() -> int:
    rel = _make_relation_graph()
    legacy = _index_legacy(rel)
    compact = _index_compact(rel)
    # JSON bytes are used as a stable proxy for relative in-memory footprint.
    legacy_bytes = _json_bytes(legacy)
    # Tuple is not JSON-serializable; convert to list for byte proxy.
    compact_for_json = {k: [[src, e] for src, e in v] for k, v in compact.items()}
    compact_bytes = _json_bytes(compact_for_json)
    result = {
        "legacy_index": {
            "targets": len(legacy),
            "entry_lists": sum(len(v) for v in legacy.values()),
            "json_proxy_bytes": legacy_bytes,
        },
        "compact_index": {
            "targets": len(compact),
            "entry_lists": sum(len(v) for v in compact.values()),
            "json_proxy_bytes": compact_bytes,
        },
        "summary": {
            "size_reduction_x": legacy_bytes / max(compact_bytes, 1),
            "legacy_index_obj_size_bytes": sys.getsizeof(legacy),
            "compact_index_obj_size_bytes": sys.getsizeof(compact),
        },
    }
    out_path = Path(
        "C:/dev/studio_brain_open/bench_results_graphrag_incoming_index_memory_profile.json"
    )
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
