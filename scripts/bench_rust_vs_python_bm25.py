from __future__ import annotations

import argparse
import json
import random
import string
import tempfile
import time
import tracemalloc
from pathlib import Path

from axon.retrievers import BM25Retriever


def _rand_word(rng: random.Random, k: int = 6) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(k))


def make_corpus(n_docs: int, vocab_size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    vocab = [_rand_word(rng) for _ in range(vocab_size)]
    docs: list[dict] = []
    for i in range(n_docs):
        toks = [rng.choice(vocab) for _ in range(40)]
        # Inject stable terms for queryability.
        if i % 13 == 0:
            toks += ["critical", "path", "latency"]
        docs.append({"id": f"d{i}", "text": " ".join(toks), "metadata": {"src": "bench"}})
    return docs


def make_queries(n_queries: int) -> list[str]:
    qs = []
    for i in range(n_queries):
        if i % 3 == 0:
            qs.append("critical latency")
        elif i % 3 == 1:
            qs.append("path")
        else:
            qs.append("critical path latency")
    return qs


def run_once(
    engine: str,
    docs: list[dict],
    queries: list[str],
    top_k: int,
    rust_fallback_enabled: bool,
) -> dict:
    with tempfile.TemporaryDirectory(prefix=f"bm25_{engine}_") as td:
        tracemalloc.start()
        t0 = time.perf_counter()
        r = BM25Retriever(
            storage_path=td,
            engine=engine,
            rust_fallback_enabled=rust_fallback_enabled,
        )
        r.add_documents(docs, save_deferred=True)
        # Force index build.
        r.search(queries[0], top_k=top_k)
        build_s = time.perf_counter() - t0
        _, peak_build = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        t1 = time.perf_counter()
        for q in queries:
            r.search(q, top_k=top_k)
        search_s = time.perf_counter() - t1
        _, peak_search = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return {
            "engine": engine,
            "backend": getattr(r, "_bm25_backend", engine),
            "docs": len(docs),
            "queries": len(queries),
            "top_k": top_k,
            "build_seconds": build_s,
            "search_total_seconds": search_s,
            "search_avg_ms": (search_s / max(1, len(queries))) * 1000.0,
            "py_peak_build_bytes": int(peak_build),
            "py_peak_search_bytes": int(peak_search),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=15000)
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--vocab-size", type=int, default=2000)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("bench_results_rust_vs_python_bm25.json"),
    )
    args = ap.parse_args()

    docs = make_corpus(args.docs, args.vocab_size, args.seed)
    queries = make_queries(args.queries)

    py = run_once("python", docs, queries, args.top_k, rust_fallback_enabled=True)
    rust = run_once("rust", docs, queries, args.top_k, rust_fallback_enabled=False)

    summary = {
        "python": py,
        "rust": rust,
        "speedup_build_x": (py["build_seconds"] / rust["build_seconds"])
        if rust["build_seconds"] > 0
        else None,
        "speedup_search_x": (py["search_total_seconds"] / rust["search_total_seconds"])
        if rust["search_total_seconds"] > 0
        else None,
        "python_peak_build_mb": py["py_peak_build_bytes"] / (1024 * 1024),
        "rust_peak_build_mb": rust["py_peak_build_bytes"] / (1024 * 1024),
        "python_peak_search_mb": py["py_peak_search_bytes"] / (1024 * 1024),
        "rust_peak_search_mb": rust["py_peak_search_bytes"] / (1024 * 1024),
    }

    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
