# Rust Migration Plan (Detailed)

## 1. Goal and Constraints
- Goal: improve throughput and reduce peak memory usage without destabilizing product behavior.
- Constraint: preserve current public API and operational workflows.
- Constraint: migration must be incremental, reversible, and benchmark-driven.
- Constraint: Python remains orchestration layer until final stage; Rust introduced in bounded modules first.

## 2. Current Architecture Assumption
- Control plane: Python (config, orchestration, API routing, workflow).
- Data plane hot paths: ingest preprocessing, lexical indexing/search, graph transforms, heavy in-memory scans.
- LLM inference: currently Python wrappers over providers or local runtimes.

## 3. Decision Framework (What to migrate first)
Prioritize by:
1. CPU-bound loops in Python with large N.
2. High object overhead (many dict/list allocations).
3. Repeated JSON/serialization churn.
4. Operations with clear I/O contracts that can be isolated behind FFI boundary.

Do not prioritize first:
1. Network-bound provider calls.
2. Thin wrappers around third-party inference engines.
3. Areas where bottleneck is external service latency.

## 4. Candidate Components and Expected Impact

### 4.1 BM25 subsystem (index build + search)
- Why: pure CPU/text-processing path; highly suited for Rust.
- Expected gain:
  - Build time: 2-6x faster (depends corpus size/tokenization).
  - Query latency: 1.5-4x faster for lexical-heavy queries.
  - Memory: lower overhead due compact structs vs Python dict-heavy corpus.
- Migration shape:
  - Rust crate implementing corpus store, tokenizer, index build, query top-k.
  - Python wrapper keeps same result schema.
  - Dual-run mode during validation: compare rank consistency.

### 4.2 Ingest preprocessing pipeline
- Why: repeated chunk metadata transforms, normalization, hashing, and per-doc passes are CPU/memory intensive.
- Expected gain:
  - Throughput: 1.5-3x for preprocessing-heavy ingest.
  - Peak RAM: 20-40% reduction if streaming batches and fewer Python intermediates.
- Migration shape:
  - Rust functions for text normalization, tokenization helpers, metadata transforms.
  - Batch-oriented API (`Vec<DocumentIn> -> Vec<DocumentOut>`).
  - Keep embedding calls in Python initially.

### 4.3 Symbol/channel search over code metadata
- Why: currently full corpus scans in Python are expensive with scale.
- Expected gain:
  - Latency: 2-8x improvement depending corpus and query token count.
  - RAM: lower transient allocation from Rust iterators and compact indices.
- Migration shape:
  - Build auxiliary symbol index in Rust.
  - Support exact + partial match semantics to preserve behavior.

### 4.4 GraphRAG in-memory graph transformations
- Why: graph operations on large dict-based structures are allocation-heavy.
- Expected gain:
  - Selected transforms: 1.5-4x.
  - RAM: material reduction from typed adjacency structures.
- Risk: higher complexity and correctness risk; schedule after BM25/ingest.

### 4.5 LLM loading/inference using Rust (feasibility)
- Feasible, with caveats:
  - If using llama.cpp/ggml: Rust bindings can run local inference well.
  - If using remote APIs (OpenAI/Gemini/etc.): Rust gives little speed benefit; network dominates.
  - If current stack uses Python-only ecosystem features (callbacks, tracing, adapters), migration cost increases.
- Recommendation:
  - Keep LLM orchestration in Python initially.
  - Move only local inference runtime invocation to Rust when profiling proves Python runtime overhead is meaningful (>10-15% of end-to-end latency).

## 5. Target End-State Architecture
- Python: API, workflow, configuration, fallbacks, provider abstraction.
- Rust modules:
  - `rust_bm25`
  - `rust_ingest_prep`
  - `rust_symbol_index`
  - optional `rust_graph_ops` (later phase)
- Interop: `pyo3` + `maturin` wheels.
- Feature flags per module in config:
  - `bm25_engine = python|rust`
  - `ingest_engine = python|rust`
  - `symbol_index_engine = python|rust`
  - `rust_fallback_enabled = true|false`

## 6. Implementation Phases

### Phase 0: Baseline and Profiling (1-2 weeks)
- Establish representative benchmark suites:
  - ingest throughput (docs/min, chunks/s)
  - query latency p50/p95/p99 (semantic-only, hybrid, code queries)
  - memory peak (RSS) for ingest and large query batches
- Capture baseline on fixed datasets and hardware profile.
- Output: locked baseline report and perf gates.

### Phase 1: Rust foundation and FFI scaffolding (1 week)
- Create Rust workspace and Python packaging integration (`maturin`).
- Define stable data contracts (input/output schemas).
- Build health-check API for capability detection.
- Output: importable wheel + CI build/test on target OS matrix.

### Phase 2: BM25 migration (2-3 weeks)
- Implement Rust BM25 index build and search.
- Add Python adapter preserving exact output schema.
- Add dual-run shadow mode:
  - compare top-k overlap (e.g., Jaccard@k)
  - compare score monotonicity and result stability.
- Rollout:
  - Canary: 5% traffic/jobs.
  - Ramp to 50%, then 100% if SLOs pass.
- Exit criteria:
  - p95 lexical query latency improves >=30%.
  - no functional regressions in retrieval tests.

### Phase 3: Ingest preprocessing migration (2-3 weeks)
- Implement Rust transformations for document normalization/chunk metadata prep.
- Use streaming batches to reduce memory spikes.
- Validate idempotence and schema compatibility.
- Exit criteria:
  - ingest throughput +25% minimum.
  - peak RSS down >=20% on large ingest benchmarks.

### Phase 4: Symbol index migration (1-2 weeks)
- Build Rust symbol index and query path.
- Maintain current match semantics (exact/partial, qualified/symbol fields).
- Exit criteria:
  - code-symbol query p95 down >=40%.
  - result parity thresholds met.

### Phase 5: Graph operations pilot (optional, 3-5 weeks)
- Move only highest-cost transforms first (not entire GraphRAG stack).
- Keep Python orchestration and serialization boundary stable.
- Exit criteria:
  - measurable gains in targeted graph workloads.

### Phase 6: LLM runtime decision (1 week analysis + optional 2-4 weeks implementation)
- Run profiler to quantify Python overhead in LLM lifecycle.
- If overhead insignificant: keep Python.
- If significant in local inference mode: migrate invocation path to Rust.
- Keep provider API integrations in Python unless proven bottleneck.

## 7. Testing and Validation Strategy
- Unit parity tests:
  - same inputs -> structurally identical outputs.
- Differential tests (Python vs Rust engines).
- Property-based tests for tokenizer/index invariants.
- Soak tests on long ingest runs.
- Memory leak tests (repeat ingest/query cycles).
- Fuzz tests for malformed metadata payloads.

## 8. Observability and Safety Controls
- Metrics:
  - per-engine latency, throughput, error rate, memory usage.
- Runtime flags:
  - instant fallback to Python on error.
- Structured logging:
  - engine selected, fallback reason, exception class, operation ID.
- Rollback:
  - single config flip to force Python engines globally.

## 9. Risks and Mitigations
- FFI serialization overhead can erase gains:
  - Mitigate with coarse-grained batch APIs and minimal copying.
- Semantic drift in retrieval ranking:
  - Mitigate via shadow runs + acceptance thresholds.
- Build/distribution complexity across platforms:
  - Mitigate with CI wheel matrix and pinned toolchains.
- Team skill gap in Rust:
  - Mitigate with module ownership, code templates, and strict review checklist.

## 10. Team Plan
- Role A: Rust core (BM25, ingest prep).
- Role B: Python integration + feature flags + fallback logic.
- Role C: Benchmarking/observability + CI release pipeline.
- Weekly checkpoints:
  - perf delta report
  - regression report
  - rollout decision

## 11. Timeline (Conservative)
- Phase 0-1: Weeks 1-3
- Phase 2: Weeks 4-6
- Phase 3: Weeks 7-9
- Phase 4: Weeks 10-11
- Phase 5-6 (optional): Weeks 12-16

## 12. Go/No-Go Criteria
Proceed with broader migration only if all are true:
- End-to-end p95 query latency improves >=25% overall.
- Ingest throughput improves >=25%.
- Peak RSS reduced >=20% on target workloads.
- Regression failure rate remains within agreed threshold.

## 13. Immediate Next Actions (No code changes)
1. Freeze benchmark datasets and hardware profile.
2. Define parity acceptance thresholds per component.
3. Draft FFI schema contracts for BM25 and ingest preprocessing.
4. Prepare CI matrix for Rust wheel builds.
5. Schedule Phase 0 profiling sprint.
