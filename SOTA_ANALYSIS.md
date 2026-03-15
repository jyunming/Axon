# SOTA RAG — Open Gaps

**Last updated: March 2026**

Everything in the original gap analysis has shipped except the items below. This document is the living backlog for what remains.

---

## 1. Indexing

### Sentence-Window Retrieval
Embed individual sentences for precise semantic matching, but return a sliding window of ±2–3 surrounding sentences as the actual LLM context.

- **Why:** Sentence embeddings capture meaning more precisely than passage embeddings; the window keeps answers coherent.
- **Effort:** Medium — needs a separate sentence-index alongside the chunk index.

### Proposition Indexing
Use an LLM to decompose each chunk into discrete factual claims ("propositions") before indexing. Each proposition is indexed independently.

- **Why:** Factual propositions are semantically denser than raw passages; retrieval precision improves significantly.
- **Effort:** High — extra LLM pass at ingest time; cost scales with corpus size.
- **Paper:** Chen et al., 2023 — "Dense X Retrieval: What Retrieval Granularity Should We Use?"

---

## 2. Agentic / Adaptive Retrieval

The current pipeline is static (retrieve once, then generate). These techniques make retrieval dynamic.

### Self-RAG
Fine-tune the LLM to emit reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) that control whether to retrieve at all, whether retrieved docs are relevant, and whether the answer is grounded.

- **Why:** Avoids unnecessary retrieval for questions the LLM can answer directly; filters irrelevant docs before generation.
- **Effort:** High — requires fine-tuning; not plug-and-play with any model.
- **Paper:** Asai et al., 2023 — "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"

### Corrective RAG (CRAG)
Evaluate retrieved document relevance with a lightweight classifier. If confidence is low, trigger web search as a supplement.

- **Why:** Gracefully handles KB gaps without hallucination.
- **Effort:** Medium — classifier + fallback routing.
- **Paper:** Yan et al., 2024 — "Corrective Retrieval Augmented Generation"

### FLARE — Forward-Looking Active Retrieval
During generation, monitor token probabilities. When the model becomes uncertain (low-probability next tokens), pause and retrieve additional context before continuing.

- **Why:** Retrieves exactly when and what is needed — mid-generation retrieval instead of a single upfront search.
- **Effort:** High — requires streaming token probability access (not available in all providers).
- **Paper:** Jiang et al., 2023 — "Active Retrieval Augmented Generation"

---

## 3. Context Compression — Token-Level (LLMLingua)

LLM-based sentence extraction is implemented (`--compress`). What's missing is **token-level** compression.

LLMLingua uses a small dedicated LM to score and drop low-information tokens from retrieved context before passing to the main LLM. Achieves 2–20× compression with minimal accuracy loss.

- **Why:** Fits more retrieved chunks into the context window; reduces latency and cost proportional to compression ratio.
- **Effort:** Medium — add optional dependency; runs as a pre-generation pass.
- **Project:** [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)

---

## 4. Embedding Models

Current defaults (`all-MiniLM-L6-v2` / `BAAI/bge-small-en-v1.5`) are good but not top-of-leaderboard. Two models stand out:

| Model | Dims | MTEB Avg | Notes |
|-------|------|----------|-------|
| **`BAAI/bge-m3`** | 1024 | ~65 | Dense + sparse + ColBERT from one model; 100+ languages; 8192 ctx |
| **`intfloat/e5-mistral-7b-instruct`** | 4096 | ~66 | LLM-based; SOTA local; requires ~14 GB VRAM |

**BGE-M3** is the higher-priority target — its sparse vector output could replace BM25 with a learned sparse retriever, which is typically more accurate. FastEmbed already supports it; the gap is wiring it through `OpenEmbedding` and updating the hybrid retriever to consume its sparse vectors.

**E5-Mistral** is SOTA but GPU-only; lower priority for the local-first use case.

---

## 5. Evaluation Frameworks

Axon Eval smoke tests run in CI. The following richer frameworks are not yet integrated:

| Framework | What It Adds |
|-----------|-------------|
| **ARES** | Fine-tuned LLM judges; statistically confident scores |
| **DeepEval** | Pytest-style CI integration; 25+ metrics; MMLU/HumanEval benchmarks |
| **TruLens** | OpenTelemetry tracing; RAG Triad; execution-flow inspection |

---

## 6. Infrastructure

### Async Streaming for Hybrid Search
BM25 currently runs synchronously and blocks the async event loop on the API path. Moving it to a thread pool or making it truly async would improve throughput under concurrent load.

- **Effort:** Low-Medium — `run_in_executor` wrapper or rewrite `BM25Retriever.search()` as async.

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
