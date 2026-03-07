# SOTA RAG Gap Analysis

**Local RAG Brain vs. State-of-the-Art RAG Systems (2026)**

This document benchmarks the current system against state-of-the-art RAG research and production systems, identifies gaps, and prioritizes improvements. It is intended to guide contributors on what to build next.

---

## Current System Capabilities (Baseline)

The following features are already implemented:

| Category | What We Have |
|----------|-------------|
| **Retrieval** | Hybrid search: dense vector + BM25 keyword, fused via Reciprocal Rank Fusion (RRF) |
| **Re-ranking** | Optional cross-encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| **Embedding providers** | sentence-transformers (`all-MiniLM-L6-v2`), Ollama (`nomic-embed-text`), FastEmbed (`BAAI/bge-small-en-v1.5`) |
| **LLM** | Ollama-backed LLM with streaming (llama3.1, qwen2.5, phi3, mistral) |
| **Vector stores** | ChromaDB (default), Qdrant |
| **Document formats** | PDF, DOCX, HTML, CSV/TSV, Markdown, JSON, plain text, BMP/PNG/TIF/TIFF/PGM images |
| **Multimodal** | BMP image ingestion with VLM auto-captioning (llava) |
| **Agent interface** | FastAPI with 6 OpenAI-compatible tools |
| **Evaluation** | RAGAS evaluation script (`scripts/evaluate.py`) |
| **Chunking** | Recursive character text splitter with configurable size/overlap |

---

## Gaps vs. State of the Art

### 1. Query Transformation — Not Implemented

Standard RAG sends the raw user query directly to the retriever. SOTA systems transform the query first.

#### 1.1 HyDE — Hypothetical Document Embeddings
- **What it does:** Generates a hypothetical "ideal" answer to the query using the LLM, then embeds that answer (not the original query) for retrieval. The hypothesis is semantically closer to relevant documents than a short question.
- **Why it matters:** Short questions and long document chunks exist in very different embedding spaces. HyDE bridges this gap.
- **Impact:** ~5–15% recall improvement on knowledge-intensive tasks.
- **Papers:** Gao et al., 2022 — "Precise Zero-Shot Dense Retrieval without Relevance Labels"

#### 1.2 Multi-Query Retrieval
- **What it does:** Generates N alternative phrasings of the user query (via LLM), runs retrieval for each, then deduplicates and merges results.
- **Why it matters:** Eliminates single-phrasing bias; catches documents that use different vocabulary.
- **Impact:** Consistently improves recall with minimal added latency.

#### 1.3 Step-Back Prompting
- **What it does:** Prompts the LLM to first identify a higher-level concept or principle behind the user's question, then retrieves context for both the abstract and specific questions.
- **Why it matters:** Useful when the question is highly specific and the KB has general background information.
- **Papers:** Zheng et al., 2023 — "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models"

#### 1.4 Query Decomposition
- **What it does:** Breaks multi-part or complex questions into atomic sub-questions, retrieves independently, and synthesizes a combined answer.
- **Why it matters:** Without decomposition, the retriever tries to satisfy multiple requirements with a single search — often poorly.

---

### 2. Advanced Indexing Strategies — Not Implemented

Current system indexes fixed-size chunks. SOTA systems use more sophisticated indexing hierarchies.

#### 2.1 RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval
- **What it does:** Clusters document chunks, generates LLM summaries for each cluster, then clusters those summaries — building a hierarchical tree. Retrieval can happen at any level.
- **Why it matters:** Enables both fine-grained chunk retrieval (leaf level) and broad thematic retrieval (summary level). Especially effective for long documents and thematic questions.
- **Papers:** Sarthi et al., 2024 — "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"

#### 2.2 Parent-Document / Small-to-Big Retrieval
- **What it does:** Indexes small, granular chunks (sentences or short passages) for precise semantic matching, but when retrieved, returns the full parent paragraph or section as context.
- **Why it matters:** Small chunks = precise retrieval; large context = better answer generation. Best of both worlds.
- **Implementation note:** Requires storing chunk-to-parent mapping in metadata.

#### 2.3 Sentence-Window Retrieval
- **What it does:** Embeds individual sentences for retrieval, but returns a sliding window of ±2–3 surrounding sentences as the actual context.
- **Why it matters:** Embedding single sentences captures semantic meaning precisely; the window provides enough context for coherent answers.

#### 2.4 Proposition Indexing
- **What it does:** Uses an LLM to extract discrete factual claims ("propositions") from document chunks before indexing. Each proposition is indexed independently.
- **Why it matters:** Factual propositions are more semantically dense and precise than raw passage chunks. Retrieval quality improves significantly.
- **Papers:** Chen et al., 2023 — "Dense X Retrieval: What Retrieval Granularity Should We Use?"

---

### 3. Advanced Re-ranking — Partially Implemented

We have a single cross-encoder re-ranker. SOTA systems have multiple re-ranking options.

#### 3.1 LLM-Based Listwise / Pointwise Re-ranking (RankGPT)
- **What it does:** Uses the generation LLM itself to score or sort retrieved documents by relevance to the query.
- **Why it matters:** LLMs have strong natural language understanding that cross-encoders lack. Can leverage existing Ollama infrastructure — no new model needed.
- **Implementation note:** Pointwise scoring is simpler; listwise (permutation-based) is more accurate.
- **Papers:** Sun et al., 2023 — "Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents"

#### 3.2 BGE Reranker v2-m3
- **What it does:** A multilingual cross-encoder trained for retrieval re-ranking, significantly outperforming `ms-marco-MiniLM-L-6-v2` on BEIR benchmarks.
- **Why it matters:** Drop-in replacement for the current re-ranker with substantially better accuracy; supports 100+ languages.
- **Model:** `BAAI/bge-reranker-v2-m3` from HuggingFace

#### 3.3 Contextual Compression
- **What it does:** After retrieval, uses an LLM to extract only the specific sentences or passages from each retrieved chunk that are directly relevant to the query. Discards irrelevant surrounding text.
- **Why it matters:** Reduces context noise; allows fitting more unique sources into the LLM context window.

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

| Gap | Impact | Effort |
|-----|--------|--------|
| No query result caching | Every identical query hits the LLM | Low — add `functools.lru_cache` or Redis |
| No metadata filtering in vector search | Can't filter by date, source, tag | Medium — ChromaDB/Qdrant support `where` filters |
| No document deduplication | Re-ingesting same doc creates duplicates | Low — hash-based dedup on ingest |
| Only BMP images supported | JPG/PNG not indexed | Low — expand loaders.py |
| No async streaming for hybrid path | BM25 runs sync, blocks | Medium |
| No chunk-level source citations | Answers don't show which chunk they came from | Medium |

---

## Priority Roadmap

Ordered by **impact / effort** ratio:

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| 1 | **Multi-query retrieval** | Low | High — recall improvement |
| 2 | **BGE-M3 embedding support** | Low | High — SOTA local embeddings |
| 3 | **HyDE query transformation** | Medium | High — precision improvement |
| 4 | **Parent-document / small-to-big retrieval** | Medium | High — context quality |
| 5 | **LLM-based re-ranking (RankGPT)** | Medium | High — uses existing Ollama |
| 6 | **Query result caching** | Low | Medium — latency/cost |
| 7 | **Metadata filtering** | Medium | Medium — precision queries |
| 8 | **BGE Reranker v2-m3** | Low | Medium — better re-ranking |
| 9 | **Context compression (LLMLingua)** | Medium | Medium — context efficiency |
| 10 | **RAPTOR hierarchical indexing** | High | High — thematic queries |
| 11 | **GraphRAG integration** | High | High — new query types |
| 12 | **CRAG (web fallback)** | High | Medium — KB gap handling |

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

*Last updated: March 2026*
