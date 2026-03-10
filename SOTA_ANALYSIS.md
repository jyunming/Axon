# SOTA RAG Gap Analysis

**Local RAG Brain vs. State-of-the-Art RAG Systems (2026)**

This document benchmarks the current system against state-of-the-art RAG research and production systems, identifies gaps, and prioritizes improvements. It is intended to guide contributors on what to build next.

---

## Current System Capabilities (Baseline)

The following features are already implemented:

| Category | What We Have |
|----------|-------------|
| **Retrieval** | Hybrid search: dense vector + BM25 keyword, fused via Reciprocal Rank Fusion (RRF) |
| **Re-ranking** | Cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) **and** LLM-based pointwise re-ranking via `reranker_provider: llm` |
| **Query transformations** | HyDE, Multi-Query, Step-Back, Query Decomposition — all available via CLI flags, REPL slash commands (`/rag hyde`, `/rag multi`, `/rag step-back`, `/rag decompose`, `/rag compress`, `/rag cite`), and REST API overrides |
| **Embedding providers** | sentence-transformers (`all-MiniLM-L6-v2`), Ollama (`nomic-embed-text`), FastEmbed (`BAAI/bge-small-en-v1.5`, BGE-M3 via fastembed), OpenAI |
| **LLM** | Ollama (local), Gemini, OpenAI, Ollama Cloud, vLLM — all with streaming |
| **Vector stores** | ChromaDB (default), Qdrant |
| **Document formats** | PDF, DOCX, HTML, CSV/TSV, Markdown, JSON, plain text, BMP/PNG/TIF/TIFF/PGM images |
| **Multimodal** | Raster image ingestion (BMP/PNG/TIF/PGM) with VLM auto-captioning (llava) |
| **Agent interface** | FastAPI with per-request RAG flag overrides (full CLI parity); 8+ REST endpoints |
| **Ingest quality** | Hash-based content deduplication on ingest (skips re-ingesting identical chunks) |
| **Performance** | In-memory query result cache (`query_cache: true`) with true LRU eviction (`OrderedDict` + move-to-end on hit) |
| **Web search** | Brave Search fallback when local KB is insufficient (truth grounding) |
| **Evaluation** | RAGAS evaluation script (`scripts/evaluate.py`) |
| **Chunking** | Recursive character text splitter with configurable size/overlap |
| **Projects** | Named knowledge base namespaces with isolated vector/BM25 stores |

---

## Gaps vs. State of the Art

### 1. Query Transformation — ✅ Mostly Implemented

Standard RAG sends the raw user query directly to the retriever. SOTA systems transform the query first.

#### 1.1 HyDE — Hypothetical Document Embeddings ✅ DONE
- **Status:** Fully implemented. Enable with `--hyde` (CLI), `/rag hyde on` (REPL), or `"hyde": true` (API).
- **What it does:** Generates a hypothetical "ideal" answer to the query using the LLM, then embeds that answer (not the original query) for retrieval.
- **Impact:** ~5–15% recall improvement on knowledge-intensive tasks.
- **Papers:** Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"

#### 1.2 Multi-Query Retrieval ✅ DONE
- **Status:** Fully implemented. Enable with `--multi-query` (CLI) or `"multi_query": true` (API).
- **What it does:** Generates 3 alternative phrasings of the user query, runs retrieval for each, deduplicates and merges with the original.
- **Impact:** Consistently improves recall with minimal added latency.

#### 1.3 Step-Back Prompting ✅ DONE
- **Status:** Fully implemented. Enable with `--step-back` (CLI), `/rag step-back` (REPL), or `"step_back": true` (API).
- **What it does:** Generates an abstract, higher-level version of the query and runs retrieval for both the specific and abstract queries. Useful when KB has general background but query is highly specific.
- **Papers:** Zheng et al., 2023 — "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"

#### 1.4 Query Decomposition ✅ DONE
- **Status:** Implemented. Enable with `--decompose` (CLI), `/rag decompose` (REPL), `"decompose": true` (API), or `query_decompose: true` in `config.yaml` under `query_transformations:`.
- **What it does:** Uses the LLM to break complex multi-part questions into 2–4 atomic sub-questions, runs retrieval for each, merges and deduplicates results. Composes naturally with `multi_query` and `step_back`.
- **Synthesis:** Retrieval merges across all sub-questions; a single LLM call generates the answer with the combined context (no extra LLM calls beyond decomposition itself).

---

### 2. Advanced Indexing Strategies — Not Implemented

Current system indexes fixed-size chunks. SOTA systems use more sophisticated indexing hierarchies.

#### 2.1 RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval
- **What it does:** Clusters document chunks, generates LLM summaries for each cluster, then clusters those summaries — building a hierarchical tree. Retrieval can happen at any level.
- **Why it matters:** Enables both fine-grained chunk retrieval (leaf level) and broad thematic retrieval (summary level). Especially effective for long documents and thematic questions.
- **Papers:** Sarthi et al., 2024 — "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"

#### 2.2 Parent-Document / Small-to-Big Retrieval ✅ DONE
- **Status:** Implemented. Set `parent_chunk_size: 2000` (or any value > `chunk_size`) in `config.yaml` under `rag:`, or use `--parent-chunk-size 2000` CLI flag.
- **What it does:** Indexes small child chunks (`chunk_size`) for precise semantic matching. At generation time, returns the full parent passage (`parent_chunk_size`) as LLM context. Parent text stored in chunk metadata — no separate store needed.
- **Typical config:** `chunk_size: 256`, `parent_chunk_size: 1500` gives precise retrieval with rich context windows.

#### 2.3 Sentence-Window Retrieval
- **What it does:** Embeds individual sentences for retrieval, but returns a sliding window of ±2–3 surrounding sentences as the actual context.
- **Why it matters:** Embedding single sentences captures semantic meaning precisely; the window provides enough context for coherent answers.

#### 2.4 Proposition Indexing
- **What it does:** Uses an LLM to extract discrete factual claims ("propositions") from document chunks before indexing. Each proposition is indexed independently.
- **Why it matters:** Factual propositions are more semantically dense and precise than raw passage chunks. Retrieval quality improves significantly.
- **Papers:** Chen et al., 2023 — "Dense X Retrieval: What Retrieval Granularity Should We Use?"

---

### 3. Advanced Re-ranking — ✅ Partially Implemented

#### 3.1 LLM-Based Pointwise Re-ranking (RankGPT) ✅ DONE
- **Status:** Implemented. Set `reranker_provider: llm` in config or pass `"rerank": true` to the API. Uses the configured LLM to score each document with a ThreadPoolExecutor for speed.
- **Papers:** Sun et al., 2023 — "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents"

#### 3.2 BGE Reranker v2-m3 ✅ DONE
- **Status:** Fully accessible. Use `--reranker-model BAAI/bge-reranker-v2-m3` (CLI) or set `reranker_model: "BAAI/bge-reranker-v2-m3"` in `config.yaml` under `rerank:`. No code changes needed; `sentence-transformers` CrossEncoder loads it automatically.
- **Why it matters:** Significantly outperforms the default `ms-marco-MiniLM-L-6-v2` on BEIR benchmarks; supports 100+ languages.
- **Default kept as** `ms-marco-MiniLM-L-6-v2` (smaller, faster, already commonly cached). Upgrade when accuracy matters more than speed.

#### 3.3 Contextual Compression ✅ DONE
- **Status:** Implemented (LLM-based). Enable with `--compress` (CLI) or `"compress": true` (API) or `compress_context: true` in `config.yaml`.
- **What it does:** After reranking, uses the generation LLM to extract only query-relevant sentences from each retrieved chunk (parallel via ThreadPoolExecutor). Falls back to the original chunk if compression fails or would expand the text. Integrates with small-to-big: compresses the parent passage, not the small child chunk.
- **Note:** LLMLingua (token-level compression with a separate small LM) would be more efficient at scale but requires an additional model download. The LLM-based approach works out of the box with any configured provider.

---

### 4. GraphRAG — Not Implemented

#### 4.1 Microsoft GraphRAG
- **What it does:** Builds a knowledge graph from documents (entity extraction + relationship mapping), then runs community detection (Leiden algorithm) and generates hierarchical community summaries. Retrieval can use either graph traversal ("local" search) or community summaries ("global" search).
- **Why it matters:** Answers global/thematic questions that chunk-based RAG cannot (e.g., "What are the main themes across all these documents?"). Enables relationship-aware reasoning.
- **Project:** [microsoft/graphrag](https://github.com/microsoft/graphrag) (MIT License)
- **Cost note:** Community summary generation requires significant upfront LLM calls — feasible with local Ollama.

#### 4.2 Entity-Centric Retrieval
- **What it does:** Indexes entity-specific knowledge separately; retrieval traverses entity relationships rather than just chunk similarity.
- **Why it matters:** Supports multi-hop reasoning (e.g., "Who owns the company that acquired X?").

---

### 5. Agentic / Adaptive RAG — Partially Implemented

Current system has a static retrieval pipeline. The agent API exposes tools but doesn't make autonomous retrieval decisions.

#### 5.1 Self-RAG
- **What it does:** Fine-tunes the LLM to generate special reflection tokens (`[Retrieve]`, `[IsRel]`, `[IsSup]`, `[IsUse]`) that control whether to retrieve, whether retrieved docs are relevant, and whether the generation is grounded.
- **Why it matters:** Removes unnecessary retrieval for queries the LLM can answer directly; filters out irrelevant documents before generation.
- **Papers:** Asai et al., 2023 — "Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"

#### 5.2 Corrective RAG (CRAG)
- **What it does:** Evaluates the relevance of retrieved documents with a lightweight classifier. If confidence is low, triggers web search as a fallback to supplement the local KB.
- **Why it matters:** Gracefully handles KB gaps without hallucination.
- **Papers:** Yan et al., 2024 — "Corrective Retrieval Augmented Generation"

#### 5.3 FLARE — Forward-Looking Active Retrieval
- **What it does:** During generation, monitors token probabilities. When the model becomes uncertain (low-probability next tokens), it pauses and retrieves additional context before continuing.
- **Why it matters:** Retrieves exactly when and what is needed — dynamic, mid-generation retrieval rather than a single upfront search.
- **Papers:** Jiang et al., 2023 — "Active Retrieval Augmented Generation"

---

### 6. Context Compression — Not Implemented

#### 6.1 LLMLingua / LLMLingua-2
- **What it does:** Uses a small LLM (e.g., Llama-2-7B) to score and remove low-information tokens from retrieved context before passing to the main LLM. Achieves 2–20x compression with minimal accuracy loss.
- **Why it matters:** Allows fitting more retrieved chunks into the context window; reduces latency and cost proportional to compression ratio.
- **Project:** [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)

---

### 7. Better Local Embedding Models — Partially Covered

Current embedding options are good but not SOTA. Higher-quality models are available locally.

| Model | Context | Dims | MTEB Avg | Notes |
|-------|---------|------|----------|-------|
| `all-MiniLM-L6-v2` *(current default)* | 256 | 384 | ~56 | Fast, small |
| `BAAI/bge-small-en-v1.5` *(current FastEmbed)* | 512 | 384 | ~62 | Balanced |
| `nomic-embed-text` *(current Ollama option)* | 8192 | 768 | ~62 | Long context |
| **`BAAI/bge-m3`** *(missing)* | 8192 | 1024 | ~65 | Multi-lingual, dense+sparse+ColBERT |
| **`intfloat/e5-mistral-7b-instruct`** *(missing)* | 32768 | 4096 | ~66 | LLM-based, SOTA local |

#### 7.1 BGE-M3
- **Multi-functionality:** Produces dense vectors (for vector search), sparse vectors (for keyword search — replaces BM25), and multi-vector ColBERT representations (late interaction) — all from a single model.
- **Multi-lingual:** 100+ languages.
- **Long context:** 8192 token context window.
- **Adding BGE-M3 could replace BM25 with a learned sparse retriever**, which is typically more accurate.

#### 7.2 E5-Mistral-7b-Instruct
- LLM-based embedder — follows task-specific instructions before encoding.
- SOTA on MTEB for local (non-API) models but requires ~14GB VRAM.
- Better suited for GPU setups.

---

### 8. Evaluation Framework — Partially Implemented

RAGAS script exists but is limited.

| Framework | What It Adds | Status |
|-----------|-------------|--------|
| **RAGAS** | Faithfulness, Answer Relevancy, Context Recall, Context Precision | Implemented (basic) |
| **ARES** | Fine-tuned LLM judges, statistically confident scores | Missing |
| **DeepEval** | Pytest-style CI/CD integration, 25+ metrics, MMLU/HumanEval benchmarks | Missing |
| **TruLens** | OpenTelemetry tracing, RAG Triad, execution-flow inspection | Missing |

Current gap: No evaluation runs in CI. No hallucination/groundedness metric is tracked over time.

---

### 9. Infrastructure Gaps

| Gap | Impact | Effort | Status |
|-----|--------|--------|--------|
| Query result caching | Every identical query hits the LLM | Low | ✅ DONE — `query_cache: true` in config or `--cache` CLI |
| Metadata filtering in vector search | Can't filter by date, source, tag | Medium | ✅ DONE — `filters` field on `/query` and `/search` endpoints |
| Document deduplication on ingest | Re-ingesting same doc creates duplicates | Low | ✅ DONE — hash-based, enabled by default (`dedup_on_ingest: true`) |
| Async streaming for hybrid path | BM25 runs sync, blocks | Medium | ⬜ Missing |
| Chunk-level source citations in answers | Answers don't show which chunk they came from | Medium | ⚙️ Partial — sources returned in stream (`type: sources`); not yet inline in text |

---

## Priority Roadmap

Ordered by **impact / effort** ratio. ✅ = shipped, ⬜ = open.

| Priority | Feature | Effort | Impact | Status |
|----------|---------|--------|--------|--------|
| ~~1~~ | ~~Multi-query retrieval~~ | Low | High | ✅ Shipped |
| ~~2~~ | ~~BGE-M3 embedding support~~ | Low | High | ✅ Via FastEmbed |
| ~~3~~ | ~~HyDE query transformation~~ | Medium | High | ✅ Shipped |
| ~~4~~ | ~~LLM-based re-ranking (RankGPT)~~ | Medium | High | ✅ Shipped |
| ~~5~~ | ~~Query result caching~~ | Low | Medium | ✅ Shipped |
| ~~6~~ | ~~Ingest deduplication~~ | Low | Medium | ✅ Shipped (default on) |
| ~~7~~ | ~~API parity with CLI~~ | Low | High | ✅ Shipped |
| ~~8~~ | ~~Step-back prompting~~ | Low | Medium | ✅ Shipped |
| ~~9~~ | ~~BGE Reranker v2-m3~~ | Low | Medium | ✅ Shipped — `--reranker-model BAAI/bge-reranker-v2-m3` or config |
| ~~10~~ | ~~Parent-document / small-to-big retrieval~~ | Medium | High | ✅ Shipped — `parent_chunk_size` config / `--parent-chunk-size` CLI |
| ~~1~~ | ~~Query decomposition~~ | Medium | Medium | ✅ Shipped — `--decompose` CLI / `"decompose": true` API |
| ~~2~~ | ~~Context compression (LLMLingua)~~ | Medium | Medium | ✅ Shipped (LLM-based) — `--compress` CLI / `"compress": true` API |
| ~~3~~ | ~~Inline chunk-level citations in answers~~ | Medium | Medium | ✅ Shipped — `--cite` CLI / `"cite": true` API / `cite_sources` config |
| ~~4~~ | ~~RAPTOR hierarchical indexing~~ | High | High | ✅ Shipped — `--raptor` CLI / `rag.raptor` config; LLM summary nodes indexed alongside leaf chunks |
| ~~5~~ | ~~GraphRAG integration~~ | High | High | ✅ Shipped — `--graph-rag` CLI / `rag.graph_rag` config; entity graph persisted to `.entity_graph.json`; result expansion at query time |
| ~~6~~ | ~~Evaluation in CI~~ | Medium | Medium | ✅ Shipped — `tests/test_eval_smoke.py` (precision@k, context relevance, faithfulness, recall@k); `@pytest.mark.eval`; dedicated CI step |

---

## References

- [RAPTOR Paper (2024)](https://arxiv.org/abs/2401.18059)
- [HyDE Paper (2022)](https://arxiv.org/abs/2212.10496)
- [Self-RAG Paper (2023)](https://arxiv.org/abs/2310.11511)
- [CRAG Paper (2024)](https://arxiv.org/abs/2401.15884)
- [FLARE Paper (2023)](https://arxiv.org/abs/2305.06983)
- [RankGPT Paper (2023)](https://arxiv.org/abs/2304.09542)
- [Dense X Retrieval / Propositions (2023)](https://arxiv.org/abs/2312.06648)
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag)
- [LLMLingua](https://github.com/microsoft/LLMLingua)
- [BGE-M3 Model](https://huggingface.co/BAAI/bge-m3)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [RAGAS Framework](https://github.com/explodinggradients/ragas)
- [DeepEval Framework](https://github.com/confident-ai/deepeval)

---

*Last updated: March 2026 — Sprint 4 (query decomposition + LLM context compression)*
