# Sparse Retrieval Milestone Proposal

**Status:** Deferred — awaiting Story 4.2 benchmark evidence
**Author:** Engineering team
**Date:** 2026-03-21
**Depends on:** Epic 4 Stories 4.1 (BGE-M3 support), 4.2 (embedding benchmarks)

---

## Scope

Add a learned sparse retrieval path to Axon's hybrid retriever so that
BM25 (lexical), dense (semantic), and learned-sparse (semantic + lexical) signals
can all participate in score fusion.

The interface and integration hook are already designed in
`src/axon/sparse_retrieval.py` (Story 4.3).  This milestone is about
**implementing** the retriever and the ingest-time sparse indexing, and deploying
it behind a feature flag.

This is **not** in the current release.  The current release delivers:
- BGE-M3 dense support hardening (Story 4.1)
- Embedding comparison benchmarks (Story 4.2)
- Interface design only (Story 4.3, `sparse_retrieval.py`)

---

## Motivation

BM25 excels on exact keyword matches but misses paraphrases.  Dense retrieval
captures semantics but degrades on rare terms and specialised vocabulary.
Learned sparse models (SPLADE, BGE-M3 sparse head) produce token-weighted vectors
that combine both strengths.  Early benchmarks on passage retrieval show +10–15%
MRR improvement over BM25 alone, with smaller (though still positive) gains over
well-tuned BM25 + dense hybrids.

**Axon's workloads most likely to benefit:**
- Code corpora with rare identifiers (exact-match critical)
- Technical documentation with domain-specific vocabulary
- Multi-lingual ingestion where BGE-M3 covers both sparse and dense in one model

---

## Proposed Implementation

### Phase A — Indexing (ingest path)

1. **Model selection**: Use BGE-M3's sparse head via `FlagEmbedding` or the
   `fastembed` sparse API.  Alternatively use SPLADE-v3 as a standalone checkpoint.

2. **Storage**: Two options evaluated by cost and query latency:
   - **Qdrant sparse vectors** (preferred): native support for sparse dot-product
     queries via `SparseVector` in `qdrant_client`.  Enables single-store hybrid.
   - **Custom inverted index** (fallback for non-Qdrant users): gzip-compressed
     JSON file alongside the BM25 corpus, loaded into memory on query.

3. **Schema**: `SparseVector` dataclass defined in `sparse_retrieval.py`.

4. **Config flag**: `sparse_retrieval: bool = False` in `AxonConfig`.

5. **Ingest overhead estimate**: SPLADE-v3 ≈ 80–120 ms/doc on CPU.  BGE-M3 sparse
   head ≈ 40–60 ms/doc sharing the dense encoding pass.  Target: < 2× current
   ingest time.

### Phase B — Query path

6. **Integration point**: Hook already defined in `sparse_retrieval.fuse_sparse()`.
   Called in `QueryRouterMixin._execute_retrieval()` when `brain._sparse_retriever`
   is not None.

7. **Fusion**: Weighted score fusion (default `sparse_weight=0.3`).  Tunable via
   `AxonConfig.sparse_weight`.

8. **Fallback**: Sparse retrieval failure is silently caught; dense + BM25 results
   are returned unchanged (contract enforced in `fuse_sparse()`).

### Phase C — Evaluation

9. **Benchmark gate**: Run `tests/benchmark_embeddings.py` with sparse enabled.
   Must show ≥ 5% HR@5 improvement over dense-only baseline before enabling
   `sparse_retrieval: True` as the default.

10. **Regression guard**: Add `tests/test_sparse_retrieval.py` covering:
    - `SparseVector` contract
    - `fuse_sparse()` with a real and a failing retriever
    - Config flag routing

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| HR@5 improvement over dense-only | ≥ 5 percentage points on benchmark corpus |
| HR@5 regression vs current hybrid (dense+BM25) | ≤ 2 percentage points (within noise) |
| Ingest overhead vs current baseline | ≤ 2× per document |
| Query latency increase | ≤ 20 ms p99 on test hardware |
| ChromaDB users unaffected | Zero behaviour change when `sparse_retrieval: false` |
| Qdrant-only feature | Sparse index only available with `vector_store: qdrant` |

---

## Rollout Phases

| Phase | Condition | Config |
|-------|-----------|--------|
| Experimental | After Phase A + B implementation | `sparse_retrieval: true` (opt-in) |
| Opt-in recommended | After benchmark evidence (Phase C passes) | Documented in SETUP.md |
| Default | After two release cycles at opt-in with no regressions reported | `sparse_retrieval: true` by default |

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| BGE-M3 sparse head not exposed in `fastembed` | Medium | Fall back to standalone SPLADE-v3 checkpoint |
| Qdrant-only constraint excludes ChromaDB users | High | Document clearly; offer inverted index fallback in Phase B |
| Ingest overhead too high for large corpora | Medium | Profile on 10k-doc corpus; add async ingest path if needed |
| Sparse fusion hurts quality on short-query workloads | Low | Tunable `sparse_weight`; default 0.3 is conservative |
| SPLADE model download adds install friction | Medium | Lazy load; warn user on first use; add to `prefetch_models.py` |

---

## Cost Estimate

| Item | Effort |
|------|--------|
| Phase A — ingest-time sparse indexing (Qdrant) | ~3 days |
| Phase B — query path integration + config flag | ~1 day |
| Phase C — benchmark + test suite | ~1 day |
| Documentation (SETUP.md update) | ~0.5 days |
| **Total** | **~5–6 days** |

---

## Decision Gate

This milestone proceeds when:

1. Story 4.2 (`benchmark_embeddings.py`) results show BGE-M3 provides ≥ 5% quality
   lift on at least one representative corpus type.
2. Engineering capacity is available after current release (Epics 1–4 dense work).
3. Management decides the memory + ingest overhead tradeoff is acceptable.

Until the gate is cleared, `src/axon/sparse_retrieval.py` exists as design
documentation only.  The integration slot in `AxonBrain` (`_sparse_retriever = None`)
ensures no code path is affected.
