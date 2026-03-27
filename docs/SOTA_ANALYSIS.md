# SOTA RAG — Open Gaps

**Last updated: March 2026**

Everything in the original gap analysis has shipped except the items below. This document is the living backlog for what remains.

> **Shipped since last update (Epics 1–6, March 2026):**
> - Sentence-Window Retrieval — `src/axon/sentence_window.py`, config `sentence_window: true`
> - CRAG-Lite Corrective RAG — `src/axon/crag.py`, config `crag_lite: true`
> - LLMLingua token-level compression — `src/axon/compression.py`, config `compress: true`
> - BGE-M3 dense embedding — FastEmbed provider, 1024-dim, documented in SETUP.md
> - BM25 async safety — lazy rebuild (`_dirty` flag) + ThreadPoolExecutor offload on API path

---

## 1. Indexing

### Proposition Indexing
Use an LLM to decompose each chunk into discrete factual claims ("propositions") before indexing. Each proposition is indexed independently.

- **Why:** Factual propositions are semantically denser than raw passages; retrieval precision improves significantly.
- **Effort:** High — extra LLM pass at ingest time; cost scales with corpus size.
- **Status:** Deferred — benchmark evidence from sentence-window needed before committing ingest cost.
- **Paper:** Chen et al., 2023 — "Dense X Retrieval: What Retrieval Granularity Should We Use?"

---

## 2. Agentic / Adaptive Retrieval

The current pipeline is static (retrieve once, then generate). These techniques make retrieval dynamic.

### Self-RAG
Fine-tune the LLM to emit reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) that control whether to retrieve at all, whether retrieved docs are relevant, and whether the answer is grounded.

- **Why:** Avoids unnecessary retrieval for questions the LLM can answer directly; filters irrelevant docs before generation.
- **Effort:** High — requires fine-tuning; not plug-and-play with any model.
- **Status:** Deferred — not compatible with the local-first model-agnostic design.
- **Paper:** Asai et al., 2023 — "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"

### FLARE — Forward-Looking Active Retrieval
During generation, monitor token probabilities. When the model becomes uncertain (low-probability next tokens), pause and retrieve additional context before continuing.

- **Why:** Retrieves exactly when and what is needed — mid-generation retrieval instead of a single upfront search.
- **Effort:** High — requires streaming token probability access (not available in all providers).
- **Status:** Deferred — requires per-token logprob access; Ollama/Gemini do not expose this uniformly.
- **Paper:** Jiang et al., 2023 — "Active Retrieval Augmented Generation"

---

## 3. Embedding Models

Current defaults (`all-MiniLM-L6-v2` / `BAAI/bge-small-en-v1.5`) are good but not top-of-leaderboard. BGE-M3 is now fully supported. One model remains:

| Model | Dims | MTEB Avg | Notes |
|-------|------|----------|-------|
| ~~**`BAAI/bge-m3`**~~ | ~~1024~~ | ~~65~~ | **Shipped** — FastEmbed, fully wired, 8192 ctx |
| **`intfloat/e5-mistral-7b-instruct`** | 4096 | ~66 | LLM-based; SOTA local; requires ~14 GB VRAM |

**E5-Mistral** is SOTA but GPU-only; lower priority for the local-first use case. No blocking dependency — can be added as a FastEmbed/Ollama model entry when hardware targets are confirmed.

---

## 4. Evaluation Frameworks

Axon Eval smoke tests and the benchmark runner (`scripts/benchmark_runner.py`) cover P@k, R@k, RTL, IHO, TCR, CTI.

**Shipped:**

| Framework | What It Adds | Status |
|-----------|-------------|--------|
| **DeepEval** | Pytest-style CI integration; 25+ metrics (faithfulness, contextual precision/recall, hallucination, MMLU) | **Shipped** — integrated in `tests/` eval suite |

**Not yet integrated:**

| Framework | What It Adds | Effort |
|-----------|-------------|--------|
| **ARES** | Fine-tuned LLM judges; statistically confident scores; confidence intervals | High |
| **TruLens** | OpenTelemetry tracing; RAG Triad (answer relevance, context relevance, groundedness) | Medium |

---

## 5. Learned Sparse Retrieval (BGE-Sparse)

The `SparseRetriever` Protocol and `fuse_sparse()` hook exist in `src/axon/sparse_retrieval.py` (interface only, `NoOpSparseRetriever` placeholder). The full milestone is scoped in `SPARSE_RETRIEVAL_MILESTONE.md`.

- **Phase 1:** BGE-M3 sparse vector encoding + Qdrant sparse payload storage
- **Phase 2:** Hybrid fusion (dense + BM25 + sparse) in query path
- **Phase 3:** Benchmark validation gate before default rollout
- **Blocking:** Needs benchmark evidence from embedding comparison (Story 4.2 data) to justify ingest cost.

---

## References

- [Self-RAG (2023)](https://arxiv.org/abs/2310.11511)
- [CRAG (2024)](https://arxiv.org/abs/2401.15884)
- [FLARE (2023)](https://arxiv.org/abs/2305.06983)
- [Dense X Retrieval / Propositions (2023)](https://arxiv.org/abs/2312.06648)
- [LLMLingua](https://github.com/microsoft/LLMLingua)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [DeepEval](https://github.com/confident-ai/deepeval)
- [TruLens](https://github.com/truera/trulens)
- [ARES](https://github.com/stanford-futuredata/ARES)
