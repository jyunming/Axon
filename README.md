# Axon

<p align="center">
  <img src="docs/assets/axon-mark.png" alt="Axon" width="350" />
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

**A local-first RAG platform for humans and AI agents.**

Point Axon at your documents. Ask questions. Get answers — using a local LLM with no cloud, no API keys required.

---

## How it works

![Axon architecture](docs/assets/diagrams/arch-overview.png)

---

## Install

```bash
# From source (recommended for development)
git clone https://github.com/jyunming/Axon.git
cd Axon
pip install -e .

# Or install as a standalone CLI tool via pipx (no venv management needed)
pipx install git+https://github.com/jyunming/Axon.git
```

Pull a local model (or bring your own API key for Gemini / OpenAI):

```bash
ollama pull llama3.1:8b   # recommended — 4.7 GB, ~8 GB RAM
ollama pull phi3:mini     # minimal — 2.3 GB, ~4 GB RAM
```

> **Windows:** Use [Windows Terminal](https://aka.ms/terminal) and set `$env:PYTHONUTF8=1` in your PowerShell profile.

---

## Launch

| Command | Entry Point | Best For |
|---|---|---|
| `axon` | Interactive REPL | Day-to-day exploration |
| `axon-api` | FastAPI REST API | Agents, scripts, Copilot |
| `axon-mcp` | MCP Server | GitHub Copilot agent mode |

---

## GitHub Copilot Integration

There are two ways to connect Axon to GitHub Copilot — pick one or use both:

### Option A — VS Code Extension (Copilot Chat tools)

The VSIX ships with the repo — no download needed:

```
1. Extensions panel (Ctrl+Shift+X) → "..." → Install from VSIX...
2. Select:  integrations/vscode-axon/axon-copilot-0.9.0.vsix
3. Reload VS Code  (Ctrl+Shift+P → "Reload Window")
```

Start `axon-api`, then ask Copilot in chat:

```
Search my knowledge base for information about the authentication module.
Ingest my project docs at /path/to/docs
```

### Option B — MCP Server (Copilot agent mode)

Create `.vscode/mcp.json` in your workspace:

```json
{
  "servers": {
    "axon": {
      "type": "stdio",
      "command": "axon-mcp",
      "env": { "RAG_API_BASE": "http://localhost:8000" }
    }
  }
}
```

Create `.vscode/settings.json`:
```json
{ "chat.mcp.access": "all" }
```

Start `axon-api`, reload VS Code — Axon tools appear in Copilot agent mode (hammer icon).

> See **[Getting Started](GETTING_STARTED.md)** for full setup details, workflow diagrams, and per-entry-point examples.

---

## Interactive REPL

![Axon REPL](docs/assets/repl-demo.png)

![Axon REPL demo](docs/assets/repl-demo.gif)

---

## Graph Panel — Investigate Your Knowledge Base Visually

After asking a question, the **Axon Graph Panel** opens a split view directly inside VS Code — no browser, no extra tools:

```
┌──────────────────────┬──────────────────────────────────────────┐
│  Question            │  [ Knowledge Graph ]  [ Code Graph ]     │
│  ────────────────    │  ────────────────────────────────────── │
│  LLM-synthesised     │                                          │
│  answer with         │   ●─────────────◆                       │
│  inline citations    │   │   3D force-   │                      │
│  ────────────────    │   ▼   graph      ▼                       │
│  [1] main.py:142 ▸   │   ●              ●                       │
│  [2] api.py:55   ▸   │                                          │
│  [3] config.py   ▸   │  ← click any node or citation           │
│                      │     to jump to the source file           │
└──────────────────────┴──────────────────────────────────────────┘
```

**Two graph views — same panel:**
- **Knowledge Graph** — entity–relation graph built from **any document** (PDF, DOCX, Markdown…) during ingest. Requires `graph_rag: true` — **on by default**. Nodes are named entities (people, concepts, components); edges are extracted relations. Just ingest your documents and the graph is ready.
- **Code Graph** — structural file/class/function graph for source code (requires `code_graph: true` in `config.yaml`). Nodes are files, classes, and functions; edges are `IMPORTS` / `CONTAINS` / `CALLS` relationships. Click a node to jump to that definition.

**How to open it:**

```
Command Palette (Ctrl+Shift+P)
  → Axon: Show Graph for Query…   ← type a question
  → Axon: Show Graph for Selection ← select code, then run

Copilot Chat:
  @workspace show me the graph for how authentication works
  @workspace visualise the retrieval pipeline
```

![Axon Copilot demo](docs/assets/AxonCopilot.gif)

---

## Key capabilities

- **Hybrid search** — dense vector + BM25 keyword, fused for better precision than either alone
- **Multi-LLM** — Ollama (local), Gemini, OpenAI, vLLM; switch live from the REPL
- **Multi-embedding** — sentence-transformers, Ollama, FastEmbed
- **Vector stores** — ChromaDB (default), Qdrant, LanceDB
- **14+ file formats** — PDF, DOCX, XLSX, PPTX, EPUB, EML, MSG, LaTeX, Jupyter (.ipynb), Parquet, SQL, XML, RTF, JSONL, CSV, Markdown, HTML, plain text, images (BMP/PNG/TIF/PGM with VLM auto-captioning)
- **Adaptive chunking** — recursive, semantic, Markdown-aware, and cosine-semantic strategies
- **Project namespaces** — isolated knowledge bases per named project; nested projects search children automatically
- **Query transformations** — HyDE, multi-query, step-back, decomposition, contextual compression
- **RAPTOR + GraphRAG** — RAPTOR hierarchical summaries + entity/relation/community graph; disabled in the shipped config for fast first-run ingest; enable when your corpus is ready; interactive 3D graph panel in VS Code, or `/graph-viz` HTML export
- **Code graph** — structural file/class/function graph with `IMPORTS`/`CONTAINS` edges for code corpora; visualise alongside the knowledge graph in the VS Code panel
- **Reranking** — cross-encoder (BGE) reranking
- **Agent-ready** — FastAPI REST API + MCP server for Copilot agent mode

---

## Guides

| Guide | What it covers |
|---|---|
| **[Getting Started](GETTING_STARTED.md)** | Ingest/query workflow for every entry point — with diagrams |
| **[Setup Guide](SETUP.md)** | Full install for all platforms, models, VS Code extension config, MCP |
| **[Quick Reference](QUICKREF.md)** | All CLI flags, REPL commands, API endpoints |
| **[Model Guide](MODEL_GUIDE.md)** | Choosing an LLM and embedding model |
| **[Troubleshooting](TROUBLESHOOTING.md)** | Common errors and fixes |
| **[Development Guide](DEVELOPMENT.md)** | Running tests, contributing |
| **[SOTA Gaps](SOTA_ANALYSIS.md)** | What's not yet implemented and why |

---

## Security

File ingestion is restricted to a configurable base directory (`RAG_INGEST_BASE`, defaults to the current working directory). Requests outside this directory are rejected with 403. See [SECURITY.md](SECURITY.md) for details.

---

## License

MIT — see [LICENSE](LICENSE).
