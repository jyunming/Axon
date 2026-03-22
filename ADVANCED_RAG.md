# Advanced RAG Pipeline Guide

Axon's retrieval pipeline is modular. Every stage can be enabled or disabled independently via
`config.yaml`, CLI flags, REPL `/rag` commands, or the REST API `overrides` field.

---

## 1. Overview

The standard pipeline runs in four stages:

```
Query → [Augmentation] → Embed → Retrieve → [Post-process] → Generate
```

| Stage | Options |
|-------|---------|
| Routing | Smart Query Router (heuristic or LLM; auto-selects strategy per query) |
| Augmentation | HyDE, Multi-Query, Step-Back, Decompose |
| Retrieval | Vector (dense), BM25 (sparse), Hybrid, RAPTOR, GraphRAG |
| Post-processing | Reranking, Context Compression |
| Generation | Inline Citations, Discussion Fallback |

All options default to `false` in the shipped `config.yaml` for fast first-run.

---

## 2. HyDE — Hypothetical Document Embeddings

**Flag:** `hyde: true`
**CLI:** `axon --hyde "your query"`
**REPL:** `/rag hyde`

HyDE generates a *hypothetical* answer to the query via an LLM call, then embeds that answer
for retrieval instead of (or alongside) the original query string. This aligns the embedding
space with document content rather than question phrasing.

**When to use:** Queries that are phrased as questions rather than keyword-dense statements.
Dense corpora where the vocabulary gap between questions and documents is large.

**Cost:** One additional LLM call per query.

---

## 3. Multi-Query

**Flag:** `multi_query: true`
**CLI:** `axon --multi-query "your query"`
**REPL:** `/rag multi`

Generates N paraphrased variants of the original query (default N=3), retrieves chunks for
each, then deduplicates and merges results before reranking or synthesis.

**When to use:** When single-query retrieval misses relevant documents due to wording
differences. Broad or exploratory questions.

**Cost:** N additional embed calls (no LLM calls unless combined with HyDE).

---

## 4. Step-Back Prompting

**Flag:** `step_back: true`
**CLI:** `axon --step-back "your query"`
**REPL:** `/rag step-back`

Before retrieval, an LLM call abstracts the query to a higher-level concept (e.g.
"How do transformers handle long contexts?" → "Transformer architecture and attention").
Both the original and abstracted queries are used for retrieval; results are merged.

**When to use:** Narrow or overly specific queries that miss conceptually related documents.
Scientific or technical domains with deep taxonomy.

**Cost:** One additional LLM call per query.

---

## 5. Query Decomposition

**Flag:** `decompose: true`
**CLI:** `axon --decompose "your query"`
**REPL:** `/rag decompose`

Splits compound or multi-part queries into independent sub-questions, retrieves separately for
each, then synthesises a unified answer across all sub-results.

**When to use:** Queries with multiple distinct sub-questions (e.g. "Compare X and Y and explain
how they differ from Z"). Reports or summaries requiring multiple document sections.

**Cost:** One LLM call to decompose + one retrieval pass per sub-question.

---

## 6. Context Compression

**Flag:** `compress: true`
**CLI:** `axon --compress "your query"`
**REPL:** `/rag compress`

After retrieval, each chunk is filtered by an LLM (or a lightweight compressor model) to keep
only the sentences relevant to the query. This reduces prompt length and noise before synthesis.

**When to use:** Large top_k values. Documents with significant off-topic content. When hitting
LLM context limits.

**Cost:** One LLM call per retrieved chunk (or one batch call if the compressor supports it).

---

## 7. BGE Reranking

**Flag:** `rerank: true`
**CLI:** `axon --rerank "your query"`
**REPL:** `/rag rerank`

A cross-encoder model (default: `BAAI/bge-reranker-base`) re-scores the top-K retrieved chunks
by jointly encoding the query and each chunk. The re-scored order replaces the original ranking
before synthesis.

**When to use:** Any production deployment. Reranking consistently reduces hallucination by
surfacing the most relevant chunks. Works best with `top_k ≥ 20` to give the reranker a wide
candidate pool.

**Reranker model:** Set via `/rag rerank-model <model>` or `rerank_model` in config.

**Cost:** One cross-encoder forward pass per candidate chunk (CPU-bound; fast on modern hardware).

---

## 8. Sentence-Window Retrieval

**Flag:** `sentence_window: true`
**Config:** `rag.sentence_window: true`, `rag.sentence_window_size: 3`

Indexes documents at sentence granularity for precise semantic matching, then expands each hit to
±N surrounding sentences before passing context to the LLM. Retrieval precision improves because
sentence embeddings capture meaning more tightly than full-passage embeddings; the window keeps
answers coherent.

**When to use:** Prose-heavy corpora (articles, reports, manuals) where a single sentence contains
the key fact but isolated context is insufficient for a coherent answer.

**Config options:**
```yaml
rag:
  sentence_window: true
  sentence_window_size: 3   # ±3 sentences around each hit (default)
```

**Cost:** Slightly higher ingest overhead (sentence segmentation pass); query latency similar to
chunk retrieval.

---

## 8b. CRAG-Lite — Corrective Retrieval

**Flag:** `crag_lite: true`
**Config:** `rag.crag_lite: true`, `rag.crag_lite_confidence_threshold: 0.4`

After retrieval, a lightweight heuristic confidence assessment evaluates the result set using
signals such as score spread, result count, source diversity, and threshold proximity. If
confidence falls below the threshold, CRAG-Lite triggers the configured fallback (web search via
`truth_grounding`, or a no-answer response). High-confidence retrievals proceed unmodified.

**When to use:** Deployments where the KB may have gaps and silent hallucination is worse than an
explicit fallback. Works well with `truth_grounding: true` for automatic web escalation.

**Config options:**
```yaml
rag:
  crag_lite: true
  crag_lite_confidence_threshold: 0.4   # 0.0–1.0; lower = less aggressive fallback
```

**Diagnostics:** The `/query` response includes `confidence`, `fallback_triggered`, and
`fallback_reason` fields when `crag_lite` is active.

**Cost:** Negligible — heuristic only, no extra LLM call.

---

## 10. RAPTOR — Hierarchical Clustering Summaries

**Flag:** `raptor: true` (also set at ingest time)
**CLI:** `axon --raptor --ingest ./docs/`
**REPL:** `/rag raptor`

RAPTOR builds a *tree* of abstractive summaries over the document corpus during ingest. Leaf
nodes are the original chunks; higher-level nodes are LLM-generated summaries of clusters of
chunks. At query time, retrieval can reach both leaf and summary nodes.

**When to use:** Long documents (books, specifications, large codebases) where important
information is spread across many chunks. Queries that require understanding at different levels
of granularity.

**Config options:**
```yaml
rag:
  raptor: true
  raptor_max_levels: 1          # number of summary levels to build (1 is usually sufficient)
  raptor_max_source_size_mb: 5.0  # skip RAPTOR for sources smaller than this
```

**Cost:** Significant additional LLM calls at ingest time (one per cluster per level).
Query time cost is negligible.

---

## 11. GraphRAG — Entity Graph with Community Summaries

**Flag:** `graph_rag: true` (also set at ingest time)
**CLI:** `axon --graph-rag "your query"`
**REPL:** `/rag graph-rag`

GraphRAG builds an entity–relation graph during ingest: named entities and SUBJECT|RELATION|OBJECT
triples are extracted per chunk, then clustered into communities. LLM-generated summaries are
produced per community. At query time, graph context (entity descriptions, relations, community
reports) is prepended to the retrieved chunks before synthesis.

**Retrieval modes** (`graph_rag_mode`):
- `local` — entity + relation descriptions + community snippet (default; best for specific queries)
- `global` — top community summaries ranked by embedding similarity (best for corpus-wide questions)
- `hybrid` — combines community reports with document excerpts

**Config options:**
```yaml
rag:
  graph_rag: true
  graph_rag_depth: standard       # light (no LLM) | standard | deep (+ claims)
  graph_rag_mode: hybrid
  graph_rag_budget: 3
  graph_rag_relations: true
  graph_rag_community: true
  graph_rag_community_backend: louvain  # safe default; set to leidenalg or auto if graspologic is verified
  graph_rag_auto_route: heuristic    # auto-select local/global/hybrid per query
```

**When to use:** Multi-hop reasoning questions. Queries about relationships between entities.
Corpus-wide summaries and thematic analysis.

**Cost:** LLM calls per chunk at ingest time. Community detection runs async after ingest.
Query time overhead is low (graph lookup + community snippet injection).

---

## 12. Recommended Configurations

### Lean (default — fastest, no extra LLM calls)
```yaml
rag:
  hybrid_search: true
  rerank: true
  top_k: 20
  # all other flags: false
```

### Balanced (good recall, moderate cost)
```yaml
rag:
  hybrid_search: true
  rerank: true
  hyde: true
  multi_query: true
  top_k: 20
  cite: true
```
*Adds ~2 LLM calls per query. Recommended for most production use cases.*

### Maximum (best recall, highest cost and latency)
```yaml
rag:
  hybrid_search: true
  rerank: true
  hyde: true
  multi_query: true
  step_back: true
  compress: true
  raptor: true
  graph_rag: true
  graph_rag_mode: hybrid
  top_k: 30
  cite: true
```
*Adds ~5–10 LLM calls per query plus cross-encoder reranking. Best for complex knowledge bases
where answer quality is paramount over latency.*

---

## 13. Smart Query Routing

**Config:** `graph_rag_auto_route: heuristic` (default) or `graph_rag_auto_route: llm`

Axon's `QueryRouterMixin` automatically profiles every query before retrieval and selects the
optimal RAG strategy for it. This happens transparently — no user action is required.

### Query Profiles

| Profile | Description | Strategy applied |
|---------|-------------|-----------------|
| `factual` | Short lookups, specific named facts or definitions | Standard vector + BM25 |
| `synthesis` | Broad questions drawing from multiple document sections | Multi-query + RAPTOR summaries |
| `table_lookup` | Structured data, statistics, numerical queries | BM25-heavy; parent-doc retrieval |
| `entity_relation` | Relationships and connections between entities | GraphRAG local mode |
| `corpus_exploration` | Themes, topics, corpus-wide summaries | GraphRAG global mode |

### Routing Modes

**Heuristic (default, zero LLM cost):**

The router inspects query length and matches against keyword sets:
- Queries > 80 chars with synthesis keywords → `synthesis`
- Relation/connection keywords → `entity_relation`
- Statistics/table keywords → `table_lookup`
- Corpus/overview keywords → `corpus_exploration`
- Otherwise → `factual`

**LLM routing (higher accuracy, one extra LLM call):**

```yaml
rag:
  graph_rag_auto_route: llm
```

The configured LLM classifies the query into one of the five profiles. More accurate for
ambiguous queries; adds ~1 LLM call per query.

### Disabling Auto-Routing

Set `graph_rag_auto_route: "off"` to use fixed RAG flags from your config or `overrides`
without any per-query profile override.

---

## Runtime Overrides

All flags can be toggled without restarting:

**REPL:**
```
/rag hyde          # toggle HyDE on/off
/rag rerank        # toggle reranking
/rag topk 20       # set top_k to 20
```

**CLI (single query):**
```bash
axon --hyde --rerank --multi-query "Your question"
```

**REST API:**
```json
POST /query
{
  "query": "Your question",
  "overrides": {"hyde": true, "rerank": true, "top_k": 20}
}
```
