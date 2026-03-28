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
```yaml
rag:
  code_graph: true
```

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

For Python files, Axon uses an AST-aware splitter that respects syntactic boundaries:

- Functions and methods are never split mid-body
- Class definitions are kept together with their docstring and `__init__`
- Module-level code is chunked separately from class bodies

Enable via config:
```yaml
chunk:
  strategy: code   # or: recursive (default), semantic, markdown
```

The `code` strategy automatically detects Python, JavaScript/TypeScript, and other supported
languages and falls back to recursive splitting for unrecognised extensions.

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

```yaml
rag:
  hybrid_search: true     # BM25 lexical matching is especially effective for code
  rerank: true            # cross-encoder reranking improves symbol disambiguation
  code_graph: true        # enable structural graph + lexical boost
  graph_rag: false        # optional; enable for architecture-level questions
  top_k: 20               # wider candidate pool for the lexical booster to work with

chunk:
  strategy: code          # AST-aware splitting for Python; recursive fallback for others
```

For **architecture-level queries** ("How does the ingestion pipeline connect to the vector
store?"), also enable:
```yaml
rag:
  graph_rag: true
  graph_rag_mode: local   # entity + relation descriptions
  graph_rag_depth: light  # fast; uses noun-phrase extraction without LLM at ingest
```
