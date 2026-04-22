from __future__ import annotations

import hashlib
import json
import random
import tempfile
import time
from pathlib import Path


def _make_graph(n_entities: int = 10000, rel_per_entity: int = 6) -> dict:
    rng = random.Random(42)
    out = {}
    for i in range(n_entities):
        src = f"entity_{i}"
        rows = []
        for j in range(rel_per_entity):
            rows.append(
                {
                    "target": f"entity_{rng.randrange(n_entities)}",
                    "relation": f"r{(i+j)%47}",
                    "chunk_id": f"c{i}_{j}",
                }
            )
        out[src] = rows
    return out


def _bucket_sig(bucket: dict) -> str:
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


def _compact_payload(bucket: dict) -> dict:
    g = {}
    for src, entries in bucket.items():
        g[src] = [[e["target"], e["relation"], e["chunk_id"]] for e in entries]
    return {"format": "rg_rel_v2", "g": g}


def _save_once(
    root: Path, graph: dict, selective: bool, prev_sigs: list[str] | None
) -> tuple[float, int, list[str]]:
    shard_count = 16
    keys = sorted(graph.keys())
    buckets = [dict() for _ in range(shard_count)]
    for i, k in enumerate(keys):
        buckets[i % shard_count][k] = graph[k]
    sigs = [_bucket_sig(b) for b in buckets]

    t0 = time.perf_counter()
    bytes_written = 0
    for i, b in enumerate(buckets):
        p = root / f".relation_graph.shard.{i:03d}.json"
        if (
            selective
            and prev_sigs is not None
            and i < len(prev_sigs)
            and prev_sigs[i] == sigs[i]
            and p.exists()
        ):
            continue
        blob = json.dumps(_compact_payload(b), separators=(",", ":")).encode("utf-8")
        p.write_bytes(blob)
        bytes_written += len(blob)
    manifest = {
        "format": "rg_rel_shard_v1",
        "compact": True,
        "shards": [f".relation_graph.shard.{i:03d}.json" for i in range(shard_count)],
    }
    manifest_blob = json.dumps(manifest, separators=(",", ":")).encode("utf-8")
    (root / ".relation_graph.shards.json").write_bytes(manifest_blob)
    bytes_written += len(manifest_blob)
    state = {
        "format": "rg_rel_shard_state_v1",
        "compact": True,
        "shard_count": shard_count,
        "signatures": sigs,
    }
    state_blob = json.dumps(state, separators=(",", ":")).encode("utf-8")
    (root / ".relation_graph.shard_state.json").write_bytes(state_blob)
    bytes_written += len(state_blob)
    dt = time.perf_counter() - t0
    return dt, bytes_written, sigs


def main() -> int:
    g = _make_graph()
    with (
        tempfile.TemporaryDirectory(prefix="axon_rel_sel_off_") as off_str,
        tempfile.TemporaryDirectory(prefix="axon_rel_sel_on_") as on_str,
    ):
        root_off = Path(off_str)
        root_on = Path(on_str)

        off_first_dt, off_first_bytes, off_sigs = _save_once(root_off, g, False, None)
        off_second_dt, off_second_bytes, _ = _save_once(root_off, g, False, off_sigs)

        on_first_dt, on_first_bytes, on_sigs = _save_once(root_on, g, True, None)
        on_second_dt, on_second_bytes, _ = _save_once(root_on, g, True, on_sigs)

    result = {
        "selective_off": {
            "first_save_seconds": off_first_dt,
            "first_save_bytes": off_first_bytes,
            "second_save_seconds": off_second_dt,
            "second_save_bytes": off_second_bytes,
        },
        "selective_on": {
            "first_save_seconds": on_first_dt,
            "first_save_bytes": on_first_bytes,
            "second_save_seconds": on_second_dt,
            "second_save_bytes": on_second_bytes,
        },
        "summary": {
            "second_save_speedup_x": off_second_dt / max(on_second_dt, 1e-12),
            "second_save_bytes_reduction_x": off_second_bytes / max(on_second_bytes, 1),
        },
    }
    out = Path(
        "C:/dev/studio_brain_open/bench_results_graphrag_relation_shard_selective_rewrite_profile.json"
    )
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
