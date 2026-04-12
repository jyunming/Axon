from __future__ import annotations

import json
import random
import time
from pathlib import Path

_KEY_MAP = {
    "title": "t",
    "summary": "s",
    "full_content": "f",
    "rank": "r",
    "level": "l",
    "member_hash": "mh",
    "indexed_hash": "ih",
    "findings": "fd",
    "rating": "rt",
    "rating_explanation": "re",
}


def _make_summaries(n: int = 2500) -> dict:
    rng = random.Random(42)
    out = {}
    for i in range(n):
        cid = f"0_{i}"
        findings = [
            {"summary": f"Finding {j} for {cid}", "explanation": f"Evidence block {j} for {cid}"}
            for j in range(3)
        ]
        out[cid] = {
            "title": f"Community {cid}",
            "summary": f"Summary for {cid}",
            "full_content": (
                f"Community report {cid}. " * 10
                + f"Key entities: e{rng.randrange(1000)}, e{rng.randrange(1000)}."
            ),
            "rank": float((i % 100) / 10.0),
            "level": 0,
            "member_hash": f"mh_{i%500}",
            "indexed_hash": f"ih_{i%800}",
            "findings": findings,
            "rating": rng.randrange(1, 11),
            "rating_explanation": "Synthetic rating rationale.",
        }
    return out


def _compact(summaries: dict) -> dict:
    return {
        "format": "gr_cs_v2",
        "s": {
            cid: {_KEY_MAP.get(k, k): v for k, v in item.items()} for cid, item in summaries.items()
        },
    }


def _bench(payload: dict) -> dict:
    t0 = time.perf_counter()
    text = json.dumps(payload, separators=(",", ":"))
    dump_dt = time.perf_counter() - t0
    t1 = time.perf_counter()
    _ = json.loads(text)
    parse_dt = time.perf_counter() - t1
    return {"bytes": len(text.encode("utf-8")), "dump_seconds": dump_dt, "parse_seconds": parse_dt}


def main() -> int:
    base = _make_summaries()
    compact = _compact(base)
    m_base = _bench(base)
    m_compact = _bench(compact)
    result = {
        "legacy": m_base,
        "compact_v2": m_compact,
        "summary": {
            "size_reduction_x": m_base["bytes"] / max(m_compact["bytes"], 1),
            "dump_speedup_x": m_base["dump_seconds"] / max(m_compact["dump_seconds"], 1e-12),
            "parse_speedup_x": m_base["parse_seconds"] / max(m_compact["parse_seconds"], 1e-12),
        },
    }
    out_path = Path(
        "C:/dev/studio_brain_open/bench_results_graphrag_community_storage_profile.json"
    )
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
