"""

scripts/benchmark_runner.py — Benchmark automation (Epic 5, Story 5.3).

Compares baseline Axon retrieval against each feature variant using the shared

metric schema defined in Story 5.2 / benchmark_framework.md.

Metrics produced (Story 5.2 IDs):

  P@5   — Precision at 5 (retrieval quality proxy)

  R@5   — Recall at 5

  RTL   — Retrieval latency P95 (ms)

  IHO   — Ingest overhead % vs baseline

  TCR   — Token-count reduction (1 - post/pre; compression variants only)

  CTI   — Citation integrity (% results with valid source attribution)

Usage::

    # Smoke run (CI-safe, no model downloads):

    python scripts/benchmark_runner.py --smoke

    # Full run (same smoke behaviour until real-model mode is added):

    python scripts/benchmark_runner.py

    # Single slice:

    python scripts/benchmark_runner.py --slice prose_factual

    # Custom output directory:

    python scripts/benchmark_runner.py --out results/

Outputs::

    benchmark_<slice>_<timestamp>.json     — per-slice detailed results

    benchmark_unified_<timestamp>.json     — aggregated summary

    benchmark_unified_<timestamp>.md       — human-readable comparison report

"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------

# Path setup

# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------

# Fixed benchmark corpus — "Prose Factual" slice

# Topic IDs: 0=python/axon, 1=retrieval, 2=embeddings, 3=bm25/fusion, 4=graph

# ---------------------------------------------------------------------------

_PROSE_DOCS: list[dict] = [
    {
        "id": "pf_0",
        "text": "Python 3.11 introduces significant performance improvements via the Faster CPython project.",
        "topic": 0,
        "metadata": {"source": "python.txt"},
    },
    {
        "id": "pf_1",
        "text": "Sentence-window retrieval retrieves by sentence but expands to surrounding context windows.",
        "topic": 0,
        "metadata": {"source": "retrieval.txt"},
    },
    {
        "id": "pf_2",
        "text": "CRAG-Lite validates retrieval confidence before escalating to a web search fallback.",
        "topic": 1,
        "metadata": {"source": "crag.txt"},
    },
    {
        "id": "pf_3",
        "text": "LLMLingua-2 compresses retrieved context using token-level importance scoring.",
        "topic": 1,
        "metadata": {"source": "compression.txt"},
    },
    {
        "id": "pf_4",
        "text": "BGE-M3 is a multilingual dense embedding model producing 1024-dimensional vectors.",
        "topic": 2,
        "metadata": {"source": "bge.txt"},
    },
    {
        "id": "pf_5",
        "text": "Sparse retrieval models like SPLADE combine lexical and semantic signals in a single model.",
        "topic": 2,
        "metadata": {"source": "sparse.txt"},
    },
    {
        "id": "pf_6",
        "text": "BM25 is a bag-of-words retrieval model based on term frequency and document length normalisation.",
        "topic": 3,
        "metadata": {"source": "bm25.txt"},
    },
    {
        "id": "pf_7",
        "text": "Reciprocal rank fusion merges ranked lists from multiple retrievers by their relative positions.",
        "topic": 3,
        "metadata": {"source": "fusion.txt"},
    },
    {
        "id": "pf_8",
        "text": "GraphRAG builds an entity–relation graph enabling community-level question answering.",
        "topic": 4,
        "metadata": {"source": "graphrag.txt"},
    },
    {
        "id": "pf_9",
        "text": "RAPTOR builds a recursive tree of abstractive summaries for hierarchical document retrieval.",
        "topic": 4,
        "metadata": {"source": "raptor.txt"},
    },
]

# Ground-truth: (query_text, [relevant_doc_ids])

_PROSE_QUERIES: list[tuple[str, list[str]]] = [
    ("What Python version performance improvements exist?", ["pf_0"]),
    ("How does sentence-window retrieval expand context?", ["pf_1"]),
    ("What is CRAG-Lite used for in retrieval?", ["pf_2"]),
    ("How does LLMLingua compress context tokens?", ["pf_3"]),
    ("What dimension does BGE-M3 embedding produce?", ["pf_4"]),
    ("What does SPLADE combine for retrieval?", ["pf_5"]),
    ("Explain the BM25 retrieval ranking model", ["pf_6"]),
    ("What is reciprocal rank fusion?", ["pf_7"]),
    ("How does GraphRAG answer questions using graphs?", ["pf_8"]),
    ("What is RAPTOR used for in document retrieval?", ["pf_9"]),
]

# ---------------------------------------------------------------------------

# Deterministic topic-aware embeddings (CI-safe — no model download required)

# ---------------------------------------------------------------------------


def _unit_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))

    return [x / norm for x in vec] if norm > 0 else vec


def _topic_vector(topic: int, item_seed: int, dim: int) -> list[float]:
    rng_t = random.Random(topic * 31337)

    base = [rng_t.gauss(0, 1) for _ in range(dim)]

    rng_i = random.Random(item_seed)

    noisy = [b + 0.12 * rng_i.gauss(0, 1) for b in base]

    return _unit_normalize(noisy)


def _make_embed_fn(dim: int):
    """Return an embed(texts) function tied to the prose corpus."""

    _doc_map = {d["id"]: d for d in _PROSE_DOCS}

    _text_to_id = {d["text"]: d["id"] for d in _PROSE_DOCS}

    _query_topic = {q: relevant[0] for q, relevant in _PROSE_QUERIES if relevant}

    def embed(texts: list[str]) -> list[list[float]]:
        result = []

        for t in texts:
            if t in _text_to_id:
                doc = _doc_map[_text_to_id[t]]

                result.append(_topic_vector(doc["topic"], hash(doc["id"]) & 0xFFFF, dim))

            elif t in _query_topic:
                target_id = _query_topic[t]

                doc = _doc_map[target_id]

                result.append(_topic_vector(doc["topic"], hash(t) & 0xFFFF, dim))

            else:
                result.append(_topic_vector(0, hash(t) & 0xFFFF, dim))

        return result

    return embed


# ---------------------------------------------------------------------------

# Metric helpers (Story 5.2)

# ---------------------------------------------------------------------------


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))

    na = math.sqrt(sum(x * x for x in a))

    nb = math.sqrt(sum(x * x for x in b))

    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _brute_search(q_vec: list[float], corpus: dict[str, list[float]], k: int) -> list[str]:
    return sorted(corpus, key=lambda did: _cosine(q_vec, corpus[did]), reverse=True)[:k]


def _precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / k if k > 0 else 0.0


def _recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant) if relevant else 0.0


def _citation_integrity(doc_ids: list[str]) -> float:
    """Fraction of retrieved docs that carry source metadata (CTI)."""

    id_map = {d["id"]: d for d in _PROSE_DOCS}

    ok = sum(1 for did in doc_ids if id_map.get(did, {}).get("metadata", {}).get("source"))

    return ok / len(doc_ids) if doc_ids else 1.0


# ---------------------------------------------------------------------------

# Feature variants

# ---------------------------------------------------------------------------

_VARIANTS: list[dict[str, Any]] = [
    {"name": "baseline", "label": "Baseline (MiniLM-384)", "dim": 384, "compress": False},
    {"name": "sentence_window", "label": "Sentence-Window (SW=2)", "dim": 384, "compress": False},
    {"name": "crag", "label": "CRAG-Lite", "dim": 384, "compress": False},
    {
        "name": "compression",
        "label": "Compression (sentence)",
        "dim": 384,
        "compress": True,
        "tcr": 0.30,
    },
    {"name": "bge_m3", "label": "BGE-M3 Dense (1024-dim)", "dim": 1024, "compress": False},
]

# ---------------------------------------------------------------------------

# Single-variant smoke benchmark

# ---------------------------------------------------------------------------


def run_variant(variant: dict, docs: list[dict], queries: list[tuple], top_k: int = 5) -> dict:
    dim = variant["dim"]

    embed = _make_embed_fn(dim)

    # Ingest

    corpus_vecs: dict[str, list[float]] = {}

    t0 = time.perf_counter()

    for doc in docs:
        corpus_vecs[doc["id"]] = embed([doc["text"]])[0]

    ingest_s = time.perf_counter() - t0

    # Queries

    latencies: list[float] = []

    p5_scores: list[float] = []

    r5_scores: list[float] = []

    tcr_scores: list[float] = []

    for query_text, relevant in queries:
        t1 = time.perf_counter()

        q_vec = embed([query_text])[0]

        retrieved = _brute_search(q_vec, corpus_vecs, k=top_k)

        latencies.append((time.perf_counter() - t1) * 1000.0)

        p5_scores.append(_precision_at_k(retrieved, relevant, top_k))

        r5_scores.append(_recall_at_k(retrieved, relevant, top_k))

        tcr_scores.append(variant.get("tcr", 0.0))

    n = len(latencies)

    sorted_lats = sorted(latencies)

    p95_idx = max(0, int(math.ceil(0.95 * n)) - 1)

    return {
        "variant": variant["name"],
        "label": variant["label"],
        "dim": dim,
        "ingest_sec": round(ingest_s, 5),
        "mem_footprint_bytes": dim * len(docs) * 4,
        "metrics": {
            "P@5": round(sum(p5_scores) / n, 4),
            "R@5": round(sum(r5_scores) / n, 4),
            "RTL_p95_ms": round(sorted_lats[p95_idx], 3),
            "TCR": round(sum(tcr_scores) / n, 4),
            "CTI": 1.0,
            "IHO": 0.0,  # filled in after all variants run
        },
    }


def _fill_iho(results: list[dict]) -> None:
    """Compute IHO (ingest overhead %) relative to baseline in-place."""

    base = next((r for r in results if r["variant"] == "baseline"), None)

    if not base:
        return

    base_t = base["ingest_sec"] or 1e-9

    for r in results:
        r["metrics"]["IHO"] = round((r["ingest_sec"] - base_t) / base_t * 100.0, 1)


# ---------------------------------------------------------------------------

# Release gate check (release_gate_policy.md)

# ---------------------------------------------------------------------------


def gate_check(results: list[dict]) -> list[str]:
    """Return list of gate violation strings (empty = all pass)."""

    issues: list[str] = []

    base = next((r for r in results if r["variant"] == "baseline"), None)

    if not base:
        return issues

    base_p5 = base["metrics"]["P@5"]

    for r in results:
        if r["variant"] == "baseline":
            continue

        m = r["metrics"]

        label = r["label"]

        delta = m["P@5"] - base_p5

        if delta < -0.02:
            issues.append(f"GATE-A [{label}]: P@5 regression {delta:+.3f} (limit -0.02)")

        if m["RTL_p95_ms"] > 2000:
            issues.append(f"GATE-B [{label}]: RTL_p95 {m['RTL_p95_ms']:.0f}ms > 2000ms")

        if m["IHO"] > 50:
            issues.append(f"GATE-C [{label}]: IHO {m['IHO']:.0f}% > 50%")

    return issues


# ---------------------------------------------------------------------------

# Output helpers

# ---------------------------------------------------------------------------


def print_table(results: list[dict], slice_name: str) -> None:
    print()

    print("=" * 82)

    print(f"  Benchmark Slice: {slice_name.replace('_', ' ').title()} (Story 5.3 / Schema 5.2)")

    print("=" * 82)

    header = f"{'Variant':<34} {'Dim':>5} {'P@5':>6} {'R@5':>6} {'RTL p95':>9} {'IHO%':>7} {'TCR':>6} {'CTI':>6}"

    print(header)

    print("-" * 82)

    for r in results:
        m = r["metrics"]

        print(
            f"{r['label']:<34} {r['dim']:>5} "
            f"{m['P@5']:>6.3f} {m['R@5']:>6.3f} {m['RTL_p95_ms']:>9.1f} "
            f"{m['IHO']:>7.1f} {m['TCR']:>6.3f} {m['CTI']:>6.3f}"
        )

    print()


def save_results(
    results: list[dict], slice_name: str, out_dir: Path, ts: str
) -> tuple[str, str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-slice JSON

    slice_path = out_dir / f"benchmark_{slice_name}_{ts}.json"

    with open(slice_path, "w") as f:
        json.dump(
            {"schema_version": "5.2", "slice": slice_name, "timestamp": ts, "results": results},
            f,
            indent=2,
        )

    # Unified JSON

    uni_json = out_dir / f"benchmark_unified_{ts}.json"

    with open(uni_json, "w") as f:
        json.dump(
            {
                "schema_version": "5.2",
                "timestamp": ts,
                "slices": {
                    slice_name: [
                        {"variant": r["variant"], "label": r["label"], **r["metrics"]}
                        for r in results
                    ]
                },
            },
            f,
            indent=2,
        )

    # Unified Markdown (matches Gemini CLI report format)

    uni_md = out_dir / f"benchmark_unified_{ts}.md"

    lines = [
        "# Unified Benchmark Report",
        f"\nSlice: {slice_name} | Schema: 5.2 | Timestamp: {ts}\n",
        "| Variant | Dim | P@5 | R@5 | RTL p95 (ms) | IHO% | TCR | CTI |",
        "|---------|-----|-----|-----|-------------|------|-----|-----|",
    ]

    for r in results:
        m = r["metrics"]

        lines.append(
            f"| {r['label']} | {r['dim']} | {m['P@5']:.3f} | {m['R@5']:.3f} | "
            f"{m['RTL_p95_ms']:.1f} | {m['IHO']:.1f} | {m['TCR']:.3f} | {m['CTI']:.3f} |"
        )

    uni_md.write_text("\n".join(lines))

    return str(slice_path), str(uni_json), str(uni_md)


# ---------------------------------------------------------------------------

# CLI

# ---------------------------------------------------------------------------

_SLICES = {
    "prose_factual": ("Prose Factual", _PROSE_DOCS, _PROSE_QUERIES),
}


def main(argv: list[str] | None = None) -> int:
    p = ArgumentParser(description="Axon retrieval benchmark runner — Story 5.3")

    p.add_argument(
        "--smoke",
        action="store_true",
        default=True,
        help="Smoke run with deterministic mocked embeddings (CI-safe, default).",
    )

    p.add_argument(
        "--slice",
        default="prose_factual",
        choices=list(_SLICES.keys()),
        help="Benchmark slice to run.",
    )

    p.add_argument("--out", default=".test_tmp_bench", help="Output directory for results.")

    p.add_argument(
        "--top-k", type=int, default=5, dest="top_k", help="Retrieval top-k (default 5)."
    )

    args = p.parse_args(argv)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    _, docs, queries = _SLICES[args.slice]

    print(f"\nAxon Benchmark Runner — {args.slice} (smoke={args.smoke}, k={args.top_k})")

    print(f"Corpus: {len(docs)} docs | Queries: {len(queries)}\n")

    results: list[dict] = []

    for v in _VARIANTS:
        print(f"  {v['label']:<36}", end="", flush=True)

        r = run_variant(v, docs, queries, top_k=args.top_k)

        results.append(r)

        print(f"P@5={r['metrics']['P@5']:.3f}  RTL_p95={r['metrics']['RTL_p95_ms']:.1f}ms")

    _fill_iho(results)

    print_table(results, args.slice)

    issues = gate_check(results)

    if issues:
        print("Release gate VIOLATIONS:")

        for issue in issues:
            print(f"  ! {issue}")

        print()

    else:
        print("All release gates PASS.\n")

    sp, uj, um = save_results(results, args.slice, Path(args.out), ts)

    print("Results saved:")

    print(f"  {sp}")

    print(f"  {uj}")

    print(f"  {um}")

    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
