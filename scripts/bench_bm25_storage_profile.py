from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from axon.retrievers import BM25Retriever


def _make_docs(n: int = 15000) -> list[dict]:
    docs = []
    for i in range(n):
        txt = (
            f"Document {i} about graph retrieval optimization and bm25 ranking. "
            "This line repeats to create compressible payload. "
            "This line repeats to create compressible payload. "
            "This line repeats to create compressible payload."
        )
        docs.append({"id": f"d{i}", "text": txt, "metadata": {"source": f"s{i%100}"}})
    return docs


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def bench() -> dict:
    docs = _make_docs()
    out = {}

    for mode in ("json", "zstd"):
        with tempfile.TemporaryDirectory(prefix="axon_bm25_store_") as tmp_str:
            tmp = Path(tmp_str)
            if mode == "zstd":
                os.environ["AXON_BM25_COMPRESS"] = "1"
                os.environ["AXON_BM25_COMPRESS_MIN_BYTES"] = "1"
                os.environ["AXON_BM25_COMPRESS_LEVEL"] = "6"
            else:
                os.environ["AXON_BM25_COMPRESS"] = "0"
            r = BM25Retriever(storage_path=str(tmp), engine="python")
            r.corpus = docs
            t0 = time.perf_counter()
            r.save()
            save_dt = time.perf_counter() - t0
            json_path = tmp / "bm25_corpus.json"
            zst_path = tmp / "bm25_corpus.json.zst"
            size_json = _size(json_path)
            size_zst = _size(zst_path)

            t1 = time.perf_counter()
            r2 = BM25Retriever(storage_path=str(tmp), engine="python")
            load_dt = time.perf_counter() - t1
            out[mode] = {
                "save_seconds": save_dt,
                "load_seconds": load_dt,
                "file_bytes_json": size_json,
                "file_bytes_zst": size_zst,
                "docs_loaded": len(r2.corpus),
            }

    json_bytes = out["json"]["file_bytes_json"] or 1
    zstd_bytes = out["zstd"]["file_bytes_zst"]
    if zstd_bytes and zstd_bytes > 0:
        out["summary"] = {
            "compression_available": True,
            "size_reduction_x": json_bytes / zstd_bytes,
            "save_speedup_x": out["json"]["save_seconds"] / max(out["zstd"]["save_seconds"], 1e-12),
            "load_speedup_x": out["json"]["load_seconds"] / max(out["zstd"]["load_seconds"], 1e-12),
        }
    else:
        out["summary"] = {
            "compression_available": False,
            "size_reduction_x": None,
            "save_speedup_x": None,
            "load_speedup_x": None,
            "note": "zstandard module not available; zstd path fell back to JSON.",
        }
    return out


def main() -> int:
    result = bench()
    out_path = Path("C:/dev/studio_brain_open/bench_results_bm25_storage_profile.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
