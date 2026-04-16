"""
scripts/bench_stores.py
-----------------------
Benchmark: tqdb-2bit vs tqdb-4bit vs ChromaDB (rerank=True for all)

Metrics reported:
  Thruput vps  -- vectors (chunks) ingested per second
  Ingest       -- wall-clock ingest time (s)
  Disk MB      -- total project directory size
  dRSS MB      -- resident-set-size growth during ingest
  p50 ms       -- median query latency (post-warmup)
  p99 ms       -- 99th-percentile query latency
  MRR          -- Mean Reciprocal Rank across all queries

Run:
    cd c:\\dev\\studio_brain_open
    $env:GEMINI_API_KEY="YOUR_KEY"
    python scripts/bench_stores.py
"""

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# -- Make src/ importable ---------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "chromadb.telemetry.product.posthog.Posthog")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["ALLOW_RESET"] = "true"

try:
    import psutil as _psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ---------------------------------------------------------------------------
# Configurations to benchmark
# ---------------------------------------------------------------------------
BENCH_CONFIGS = [
    {"label": "tqdb-2bit", "provider": "turboquantdb", "bits": 2, "rerank": True},
    {"label": "tqdb-4bit", "provider": "turboquantdb", "bits": 4, "rerank": True},
    {"label": "chroma", "provider": "chroma", "bits": 8, "rerank": True},
]

# ---------------------------------------------------------------------------
# Queries with relevant source-filename fragments for MRR
# ---------------------------------------------------------------------------
RECALL_QUERIES = [
    {
        "query": "How does acid diffusion limit resolution in chemically amplified photoresists?",
        "relevant": ["Mack_1995_Acid_Diffusion"],
    },
    {
        "query": "How is mask topography modeled in EUV lithography simulation?",
        "relevant": ["Mack_1999_Mask_Topography"],
    },
    {
        "query": "What are the key stochastic variation sources in EUV exposure?",
        "relevant": [
            "Mack_2013_Stochastic",
            "Mack_2018_Shot_Noise",
            "Mack_2019_Metrics_Stochastic",
            "Mack_2019_Will_Stochastics",
        ],
    },
    {
        "query": "What shot noise models apply to optical lithography across 100 years of history?",
        "relevant": ["Mack_2018_Shot_Noise"],
    },
    {
        "query": "What metrics quantify stochastic scaling in EUV processes?",
        "relevant": ["Mack_2019_Metrics_Stochastic"],
    },
    {
        "query": "How can line edge roughness be reduced in EUV lithography?",
        "relevant": ["Mack_2018_Reducing_Roughness"],
    },
    {
        "query": "What is the history and scope of lithography simulation from 1975 to 2005?",
        "relevant": ["Mack_2005_Thirty_Years", "Mack_2001_Lithographic", "Mack_2003_Lithography"],
    },
    {
        "query": "How does differentiable lithography enable inverse mask optimization?",
        "relevant": ["2024_OpenSource_Differentiable"],
    },
    {
        "query": "What crosslinking mechanisms are important for metal oxide resists at high-NA EUV?",
        "relevant": ["Huang_2025_Crosslinking", "Narasimhan_2025_MOR"],
    },
    {
        "query": "What NIST metrology standards apply to nanoscale lithography measurements?",
        "relevant": ["NIST_Lithography_Metrology", "NIST_SP_1500"],
    },
    {
        "query": "What are the new optical limits for lithography beyond NA=1.35?",
        "relevant": ["Mack_2004_New_Limits"],
    },
    {
        "query": "What semiconductor manufacturing process optimization approaches use machine learning?",
        "relevant": ["2025_Semiconductor_Manufacturing"],
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dir_size_mb(path):
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return total / (1024 * 1024)


def rss_mb():
    if not _HAS_PSUTIL:
        return 0.0
    return _psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def reciprocal_rank(results, relevant_fragments):
    """Return 1/rank of first relevant result, 0 if none found."""
    for rank, r in enumerate(results, start=1):
        src = r.get("metadata", {}).get("source", r.get("id", ""))
        for frag in relevant_fragments:
            if frag.lower() in src.lower():
                return 1.0 / rank
    return 0.0


def percentile(values, pct):
    if not values:
        return 0.0
    s = sorted(values)
    idx = (pct / 100) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


@dataclass
class BenchResult:
    label: str
    provider: str
    bits: int
    rerank: bool
    ingest_time_s: float = 0.0
    chunk_count: int = 0
    disk_mb: float = 0.0
    delta_rss_mb: float = 0.0
    latencies_ms: list = field(default_factory=list)
    rr_scores: list = field(default_factory=list)

    @property
    def thruput_vps(self):
        return self.chunk_count / self.ingest_time_s if self.ingest_time_s > 0 else 0.0

    @property
    def p50_ms(self):
        return percentile(self.latencies_ms, 50)

    @property
    def p99_ms(self):
        return percentile(self.latencies_ms, 99)

    @property
    def mrr(self):
        return sum(self.rr_scores) / len(self.rr_scores) if self.rr_scores else 0.0


# ---------------------------------------------------------------------------
# Build AxonConfig
# ---------------------------------------------------------------------------


def make_config(provider, project_path, store_base, gemini_api_key, tqdb_bits, rerank):
    from axon.config import AxonConfig

    bm25_path = str(Path(project_path) / "bm25_index")
    vector_path = str(Path(project_path) / "vector_data")

    cfg = AxonConfig(
        projects_root=store_base,
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        llm_provider="gemini",
        llm_model="gemini-2.0-flash",
        gemini_api_key=gemini_api_key,
        vector_store=provider,
        vector_store_path=vector_path,
        tqdb_bits=tqdb_bits,
        bm25_path=bm25_path,
        top_k=10,
        similarity_threshold=0.0,
        hybrid_search=True,
        hybrid_weight=0.7,
        rerank=rerank,
        reranker_provider="cross-encoder",
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        raptor=False,
        graph_rag=False,
        graph_rag_community=False,
        hyde=False,
        multi_query=False,
        dedup_on_ingest=False,
        chunk_size=1000,
        chunk_overlap=200,
        chunk_strategy="semantic",
        parent_chunk_size=0,
        discussion_fallback=False,
    )
    Path(bm25_path).mkdir(parents=True, exist_ok=True)
    Path(vector_path).mkdir(parents=True, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# Run one benchmark configuration
# ---------------------------------------------------------------------------


def run_benchmark(cfg_spec, store_base, papers_dir, gemini_api_key):
    label = cfg_spec["label"]
    provider = cfg_spec["provider"]
    bits = cfg_spec["bits"]
    rerank = cfg_spec["rerank"]

    project_dir = Path(store_base) / f"bench-{label}"

    print(f"\n{'='*64}")
    print(f"  [{label}]  provider={provider}  bits={bits}  rerank={rerank}")
    print(f"  dir: {project_dir}")
    print(f"{'='*64}")

    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)

    cfg = make_config(provider, str(project_dir), store_base, gemini_api_key, bits, rerank)

    from axon.loaders import DirectoryLoader
    from axon.main import AxonBrain

    print(f"  Loading from {papers_dir} ...")
    loader = DirectoryLoader()
    raw_docs = loader.load(papers_dir)
    print(f"  Loaded {len(raw_docs)} raw documents.")

    brain = AxonBrain(config=cfg)

    rss_before = rss_mb()
    t0 = time.perf_counter()
    brain.ingest(raw_docs)
    t1 = time.perf_counter()
    rss_after = rss_mb()

    ingest_time = t1 - t0
    chunk_count = sum(d["chunks"] for d in brain.vector_store.list_documents())
    disk_mb_val = dir_size_mb(project_dir)
    delta_rss = rss_after - rss_before

    print(
        f"  Ingest: {ingest_time:.1f}s  |  {chunk_count} chunks  |  "
        f"{disk_mb_val:.1f} MB  |  dRSS {delta_rss:+.1f} MB"
    )

    # warmup: 2 queries to prime BM25 index and model caches
    print("  Warming up ...")
    for wq in RECALL_QUERIES[:2]:
        brain.search_raw(wq["query"])

    print(f"  Timing {len(RECALL_QUERIES)} queries ...")
    latencies = []
    rr_scores = []

    for i, q in enumerate(RECALL_QUERIES):
        qt0 = time.perf_counter()
        results, _, _ = brain.search_raw(q["query"])
        qt1 = time.perf_counter()

        lat_ms = (qt1 - qt0) * 1000
        latencies.append(lat_ms)

        rr = reciprocal_rank(results, q["relevant"])
        rr_scores.append(rr)

        hit = "HIT" if rr > 0 else "---"
        print(f"    [{i+1:2d}] {lat_ms:6.1f}ms  RR={rr:.3f}  {hit}  {q['query'][:52]}")

    if hasattr(brain.vector_store, "close"):
        brain.vector_store.close()

    return BenchResult(
        label=label,
        provider=provider,
        bits=bits,
        rerank=rerank,
        ingest_time_s=ingest_time,
        chunk_count=chunk_count,
        disk_mb=disk_mb_val,
        delta_rss_mb=delta_rss,
        latencies_ms=latencies,
        rr_scores=rr_scores,
    )


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------


def print_report(results):
    col = 16
    pad = 18

    print("\n" + "=" * (pad + col * len(results)))
    print("  BENCHMARK RESULTS  (rerank=True, all configs)")
    print("=" * (pad + col * len(results)))

    print(f"  {'Metric':<{pad-2}}" + "".join(f"{r.label:>{col}}" for r in results))
    print("-" * (pad + col * len(results)))

    def row(label, values):
        print(f"  {label:<{pad-2}}" + "".join(f"{v:>{col}}" for v in values))

    row("Thruput vps", [f"{r.thruput_vps:.0f}" for r in results])
    row("Ingest (s)", [f"{r.ingest_time_s:.1f}" for r in results])
    row("Chunks", [f"{r.chunk_count}" for r in results])
    row("Disk MB", [f"{r.disk_mb:.1f}" for r in results])
    row("dRSS MB", [f"{r.delta_rss_mb:+.1f}" if _HAS_PSUTIL else "n/a" for r in results])
    row("p50 ms", [f"{r.p50_ms:.1f}" for r in results])
    row("p99 ms", [f"{r.p99_ms:.1f}" for r in results])
    row("MRR", [f"{r.mrr:.3f}" for r in results])

    print("=" * (pad + col * len(results)))

    print("\n  Per-query latency & reciprocal rank:")
    print("-" * (pad + col * len(results)))
    for i, q in enumerate(RECALL_QUERIES):
        cells = "".join(f"  {r.latencies_ms[i]:5.0f}ms/{r.rr_scores[i]:.2f}" for r in results)
        print(f"  [{i+1:2d}]{cells}  {q['query'][:42]}")
    print("=" * (pad + col * len(results)))

    out = ROOT / "bench_results.json"
    out.write_text(
        json.dumps(
            [
                {
                    "label": r.label,
                    "provider": r.provider,
                    "bits": r.bits,
                    "rerank": r.rerank,
                    "thruput_vps": r.thruput_vps,
                    "ingest_s": r.ingest_time_s,
                    "chunks": r.chunk_count,
                    "disk_mb": r.disk_mb,
                    "delta_rss_mb": r.delta_rss_mb,
                    "p50_ms": r.p50_ms,
                    "p99_ms": r.p99_ms,
                    "mrr": r.mrr,
                    "latencies_ms": r.latencies_ms,
                    "rr_scores": r.rr_scores,
                }
                for r in results
            ],
            indent=2,
        )
    )
    print(f"\n  Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tqdb-2bit / tqdb-4bit / ChromaDB with rerank"
    )
    parser.add_argument(
        "--papers-dir",
        default=str(ROOT / "Qualification" / "papers"),
    )
    parser.add_argument(
        "--store-base",
        default=os.getenv("AXON_STORE_ROOT", ""),
    )
    parser.add_argument(
        "--gemini-key",
        default=os.getenv("GEMINI_API_KEY", ""),
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="Subset of labels to run, e.g. --configs tqdb-2bit chroma",
    )
    args = parser.parse_args()

    papers_dir = args.papers_dir
    store_base = args.store_base

    if not Path(papers_dir).exists():
        print(f"ERROR: papers-dir not found: {papers_dir}", file=sys.stderr)
        sys.exit(1)

    configs = BENCH_CONFIGS
    if args.configs:
        labels = set(args.configs)
        configs = [c for c in BENCH_CONFIGS if c["label"] in labels]
        if not configs:
            print(f"ERROR: no matching configs for {args.configs}", file=sys.stderr)
            sys.exit(1)

    print(f"Papers : {papers_dir}")
    print(f"Store  : {store_base}")
    print(f"psutil : {'yes' if _HAS_PSUTIL else 'no (pip install psutil for dRSS)'}")
    print(f"Configs: {[c['label'] for c in configs]}")

    results = []
    for spec in configs:
        results.append(run_benchmark(spec, store_base, papers_dir, args.gemini_key))

    print_report(results)


if __name__ == "__main__":
    main()
