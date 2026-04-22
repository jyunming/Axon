from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from axon.retrievers import BM25Retriever


def _make_docs(n_docs: int = 20000, n_unique_texts: int = 200) -> list[dict]:
    docs: list[dict] = []
    for i in range(n_docs):
        t = i % n_unique_texts
        text = (
            f"Template {t}: graph retrieval pipeline with bm25 token_{t%37} "
            f"entity_{t%23} relation_{t%29}. Shared text to simulate chunk duplication."
        )
        docs.append({"id": f"d{i}", "text": text, "metadata": {"source": f"s{i%250}"}})
    return docs


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _run_case(*, dedup: bool, compress: bool, docs: list[dict]) -> dict:
    os.environ["AXON_BM25_CORPUS_DEDUP"] = "1" if dedup else "0"
    os.environ["AXON_BM25_COMPRESS"] = "1" if compress else "0"
    os.environ["AXON_BM25_COMPRESS_MIN_BYTES"] = "1"
    os.environ["AXON_BM25_COMPRESS_LEVEL"] = "6"

    with tempfile.TemporaryDirectory(prefix="axon_bm25_dedup_") as tmp_str:
        tmp = Path(tmp_str)
        r = BM25Retriever(storage_path=str(tmp), engine="python")
        r.corpus = docs

        t0 = time.perf_counter()
        r.save()
        save_dt = time.perf_counter() - t0

        json_path = tmp / "bm25_corpus.json"
        zst_path = tmp / "bm25_corpus.json.zst"
        bytes_json = _size(json_path)
        bytes_zst = _size(zst_path)
        on_disk = bytes_zst if bytes_zst > 0 else bytes_json

        t1 = time.perf_counter()
        r2 = BM25Retriever(storage_path=str(tmp), engine="python")
        load_dt = time.perf_counter() - t1
        docs_loaded = len(r2.corpus)
        dedup_lazy_docs = int(getattr(r2, "_dedup_doc_count", 0) or 0)
        docs_logical = docs_loaded + dedup_lazy_docs

        return {
            "dedup_enabled": dedup,
            "compress_enabled": compress,
            "save_seconds": save_dt,
            "load_seconds": load_dt,
            "file_bytes_json": bytes_json,
            "file_bytes_zst": bytes_zst,
            "file_bytes_effective": on_disk,
            "docs_loaded": docs_loaded,
            "docs_loaded_logical": docs_logical,
        }


def main() -> int:
    docs = _make_docs()
    cases = {
        "json_plain": _run_case(dedup=False, compress=False, docs=docs),
        "json_dedup": _run_case(dedup=True, compress=False, docs=docs),
        "zstd_plain": _run_case(dedup=False, compress=True, docs=docs),
        "zstd_dedup": _run_case(dedup=True, compress=True, docs=docs),
    }

    summary = {
        "dataset_docs": len(docs),
        "dataset_unique_texts": len({d["text"] for d in docs}),
        "json_size_reduction_x": cases["json_plain"]["file_bytes_effective"]
        / max(cases["json_dedup"]["file_bytes_effective"], 1),
        "zstd_size_reduction_x": cases["zstd_plain"]["file_bytes_effective"]
        / max(cases["zstd_dedup"]["file_bytes_effective"], 1),
        "json_load_speedup_x": cases["json_plain"]["load_seconds"]
        / max(cases["json_dedup"]["load_seconds"], 1e-12),
        "zstd_load_speedup_x": cases["zstd_plain"]["load_seconds"]
        / max(cases["zstd_dedup"]["load_seconds"], 1e-12),
    }
    result = {"cases": cases, "summary": summary}
    out_path = Path("C:/dev/studio_brain_open/bench_results_bm25_dedup_profile.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
