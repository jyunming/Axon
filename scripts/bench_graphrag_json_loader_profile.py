from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path


def _make_payload(n: int = 50000) -> dict:
    out = {}
    for i in range(n):
        out[f"entity_{i}"] = {
            "description": f"Entity {i} description",
            "chunk_ids": [f"c{i}_{j}" for j in range(3)],
            "type": "ENTITY",
            "frequency": (i % 7) + 1,
            "degree": i % 13,
        }
    return out


def main() -> int:
    payload = _make_payload()
    tmp = Path(tempfile.mkdtemp(prefix="axon_json_profile_")) / "entity_graph.json"
    tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    raw = tmp.read_bytes()

    t0 = time.perf_counter()
    _ = json.loads(raw.decode("utf-8"))
    std_dt = time.perf_counter() - t0

    try:
        import orjson  # type: ignore

        t1 = time.perf_counter()
        _ = orjson.loads(raw)
        orj_dt = time.perf_counter() - t1
        summary = {
            "orjson_available": True,
            "stdlib_parse_seconds": std_dt,
            "orjson_parse_seconds": orj_dt,
            "parse_speedup_x": std_dt / max(orj_dt, 1e-12),
        }
    except Exception:
        summary = {
            "orjson_available": False,
            "stdlib_parse_seconds": std_dt,
            "orjson_parse_seconds": None,
            "parse_speedup_x": None,
        }

    result = {
        "file_bytes": len(raw),
        "records": len(payload),
        "summary": summary,
    }
    out_path = Path("C:/dev/studio_brain_open/bench_results_graphrag_json_loader_profile.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
