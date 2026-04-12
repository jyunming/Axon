from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from axon.retrievers import BM25Retriever


def _make_docs(n_docs: int = 12000) -> list[dict]:
    out = []
    for i in range(n_docs):
        text = (
            f"doc {i} graph retrieval optimization bm25 ranking token_{i%97} "
            f"entity_{i%53} relation_{i%71} context_{i%37}"
        )
        out.append({"id": f"d{i}", "text": text, "metadata": {"source": f"s{i%100}"}})
    return out


def _make_queries(n: int = 500) -> list[str]:
    return [f"graph token_{i%97} relation_{i%71}" for i in range(n)]


def _run(numpy_topk: bool) -> dict:
    os.environ["AXON_BM25_NUMPY_TOPK"] = "1" if numpy_topk else "0"
    tmp = Path(tempfile.mkdtemp(prefix="axon_bm25_topk_"))
    r = BM25Retriever(storage_path=str(tmp), engine="python")
    r.corpus = _make_docs()
    r._rebuild_index()
    queries = _make_queries()
    t0 = time.perf_counter()
    total_hits = 0
    for q in queries:
        res = r.search(q, top_k=10)
        total_hits += len(res)
    dt = time.perf_counter() - t0
    return {
        "seconds_total": dt,
        "seconds_avg_query": dt / len(queries),
        "total_hits": total_hits,
    }


def main() -> int:
    baseline = _run(numpy_topk=False)
    optimized = _run(numpy_topk=True)
    result = {
        "python_heap_topk": baseline,
        "python_numpy_topk": optimized,
        "summary": {
            "speedup_avg_query_x": baseline["seconds_avg_query"]
            / max(optimized["seconds_avg_query"], 1e-12)
        },
    }
    out_path = Path("C:/dev/studio_brain_open/bench_results_bm25_python_topk_profile.json")
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
