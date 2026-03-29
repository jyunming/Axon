# RAG for Code — Engineering Guide

Axon includes a dedicated code-retrieval pipeline that significantly outperforms generic RAG
when the knowledge base contains source code. This guide explains how to enable it and what
it does internally.

---

## 1. Code Graph Indexing

During ingest, Axon can build a structural **code graph** that maps the relationships between
files, classes, and functions.

**Enable at ingest time:**
```bash
axon --code-graph --ingest ./src/
```

Or in `config.yaml`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rag.code_graph` | bool | `false` | Build a structural code graph during ingest (file/class/function relationships) |

**What it indexes:**
- `CONTAINS` edges — file → class, class → method, file → function
- `IMPORTS` edges — module-level import statements between files
- `MENTIONED_IN` edges — prose chunks (README, comments) that reference known symbols

The graph is stored alongside the BM25 index and persists across restarts. It is queryable
via the VS Code "Show Graph for Selection" command or the `/graph-viz` REPL export.

---

## 2. Lexical Code Boost

When a query mentions code identifiers, the standard semantic search score is blended with
a **lexical identifier score** that rewards exact or partial token matches against symbol
names in the index.

### How it works

Axon's `_extract_code_query_tokens()` function tokenises the query:

| Input style | Example query fragment | Tokens extracted |
|-------------|----------------------|-----------------|
| CamelCase | `CodeAwareSplitter` | `code`, `aware`, `splitter` |
| snake_case | `split_python_ast` | `split`, `python`, `ast` |
| Qualified name | `brain.query` | `brain`, `query`, `brain.query` |
| File basename | `loaders.py` | `loaders` |

Tokens shorter than 3–4 characters are filtered out to avoid false positives.

### Scoring blend

For each retrieved code chunk, the final score is:

```
final_score = 0.55 × lexical_score + 0.45 × semantic_score
```

where `lexical_score` is 1.0 for an exact symbol match, 0.6 for a partial token match, and
0.0 otherwise. A **per-file diversity cap** prevents a single large file from monopolising
all top-K slots.

---

## 3. Symbol Channel Search

In addition to the standard vector + BM25 retrieval, Axon runs a dedicated **symbol channel**
that searches directly against the `symbol_name` and `qualified_name` metadata fields stored
with each code chunk.

- Exact token match → score 1.0
- Partial token match → score 0.6

Results from the symbol channel are merged with the main retrieval pool before reranking.
This ensures that a query like "find the `_apply_overrides` method" returns the correct chunk
even if the embedding distance is mediocre.

---

## 4. Code Graph Expansion

After initial retrieval, Axon traverses the code graph from matched node names to collect
**related chunks** beyond the initial top-K:

- `CONTAINS` edges — fetch the parent file or class when a child method is matched
- `IMPORTS` edges — include modules imported by the matched file
- `MENTIONED_IN` edges — include prose chunks (docs, READMEs) that reference the matched symbol

This means a query about `AxonBrain` will surface not just the class definition but also
its methods, the files it imports, and any documentation that mentions it by name.

---

## 5. AST-Aware Chunking

For source code files, Axon automatically uses a `CodeAwareSplitter` that respects syntactic boundaries — no config change needed:

- Functions and methods are never split mid-body
- Class definitions are kept together with their docstring and `__init__`
- Module-level code is chunked separately from class bodies

This is triggered automatically whenever a file is loaded by `CodeFileLoader` (i.e. any extension Axon recognises as source code: `.py`, `.ts`, `.js`, `.go`, `.rs`, etc.). It is not controlled by `chunk.strategy`.

The `chunk.strategy` setting controls chunking for **non-code documents** only. Valid values:

| Value | Description |
|-------|-------------|
| `semantic` | *(default)* Sentence-boundary semantic splitting |
| `recursive` | Character-based recursive splitting |
| `markdown` | Header-aware Markdown splitting |
| `cosine_semantic` | Cosine-similarity-based semantic splitting |

---

## 6. VS Code Integration

When the code graph is built, the VS Code extension exposes two additional features:

**Show Graph for Selection:**
Select any symbol in the editor, then run `Axon: Show Graph for Selection` from the Command
Palette (`Ctrl+Shift+P`). The graph panel highlights the selected symbol and its immediate
neighbours (imports, callers, callees via `CONTAINS`/`IMPORTS` edges).

**Graph Panel — KG tab:**
The Knowledge Graph tab in the Axon side panel shows both the entity graph (from GraphRAG)
and the code graph overlaid. Code nodes are rendered as squares; document nodes as circles.

---

## 7. Recommended Config for Code Corpora

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rag.hybrid_search` | bool | `false` | Enable BM25 + vector hybrid search — especially effective for code identifiers |
| `rag.rerank` | bool | `false` | Cross-encoder reranking; improves symbol disambiguation |
| `rag.code_graph` | bool | `false` | Structural code graph + lexical boost |
| `rag.graph_rag` | bool | `false` | GraphRAG entity graph; enable for architecture-level queries |
| `rag.top_k` | int | `10` | Candidate pool size; `20` recommended to give the lexical booster more to work with |
| `chunk.strategy` | string | `semantic` | Chunking strategy for non-code documents; code files use `CodeAwareSplitter` automatically |

For **architecture-level queries** ("How does the ingestion pipeline connect to the vector store?"), also set:

| Parameter | Type | Recommended | Description |
|-----------|------|-------------|-------------|
| `rag.graph_rag` | bool | `true` | Enable GraphRAG entity graph |
| `rag.graph_rag_mode` | string | `local` | `local` — entity + relation descriptions |
| `rag.graph_rag_depth` | string | `light` | `light` — noun-phrase extraction without LLM at ingest (fast) |
