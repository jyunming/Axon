# Axon

<p align="center">
  <img src="docs/assets/axon-mark.png" alt="Axon" width="300" />
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

**A local-first RAG platform for humans and AI agents.**

Axon lets you ingest your documents and ask questions about them using a local LLM — no cloud, no API keys required for the default setup. It also exposes a REST API and MCP server so agents (GitHub Copilot, custom bots) can query your knowledge base as a tool.

---

## Interactive REPL

![Axon REPL demo](docs/assets/repl-demo.gif)

---

## GitHub Copilot Integration

Use Axon as a tool directly inside VS Code with GitHub Copilot:

![Axon Copilot demo](docs/assets/AxonCopilot.gif)

---

## Web UI

![Axon Web UI](docs/assets/webapp-screenshot.png)

---

## Documentation

| Guide | What it covers |
|---|---|
| **[Getting Started](GETTING_STARTED.md)** | Core concepts, first ingest, REPL basics |
| **[Setup Guide](SETUP.md)** | Full install for Windows / Linux / macOS, all model options, MCP setup |
| **[Quick Reference](QUICKREF.md)** | All CLI flags, REPL slash commands, API endpoints |
| **[Model Guide](MODEL_GUIDE.md)** | Choosing an LLM and embedding model |
| **[Troubleshooting](TROUBLESHOOTING.md)** | Common errors and fixes |
| **[Development Guide](DEVELOPMENT.md)** | Running tests, contributing |
| **[SOTA Gaps](SOTA_ANALYSIS.md)** | What's not yet implemented and why |

---

## Key capabilities

- **Hybrid search** — dense vector + BM25 keyword, fused for better precision than either alone
- **Multi-LLM** — Ollama (local), Gemini, OpenAI, vLLM; switch live from the REPL
- **Multi-embedding** — sentence-transformers, Ollama, FastEmbed
- **Vector stores** — ChromaDB (default), Qdrant, LanceDB
- **Rich document support** — PDF, DOCX, HTML, CSV/TSV, Markdown, JSON, plain text, images (BMP/PNG/TIF/PGM with VLM auto-captioning)
- **Project namespaces** — isolated vector stores per named project; nested projects search children automatically
- **Query transformations** — HyDE, multi-query, step-back, decomposition, contextual compression
- **Advanced indexing** — RAPTOR hierarchical summaries, GraphRAG entity graph
- **Reranking** — cross-encoder (BGE) and LLM-based pointwise reranking
- **Agent-ready** — FastAPI REST API + MCP server for Copilot agent mode
- **Interactive REPL** — tab completion, streaming Markdown, `@file` context attachment, pinned status bar

---

## Quick start

### 1. Install

```bash
# Clone and install
git clone https://github.com/jyunming/Axon.git
cd Axon
pip install -e .
```

> **Windows:** Use [Windows Terminal](https://aka.ms/terminal) for correct rendering, and set `$env:PYTHONUTF8=1` in your PowerShell profile.

### 2. Pull a model (Ollama)

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
```

**Windows / macOS:** Download and run the installer from [ollama.com/download](https://ollama.com/download) — Ollama starts automatically.

```bash
# Pull a model (pick one)
ollama pull llama3.1:8b     # recommended (4.7 GB, ~6-8 GB RAM)
ollama pull phi3:mini        # minimal (2.3 GB, ~4-6 GB RAM)
```

### 3. Run

```bash
axon              # interactive REPL
axon-api          # REST API on :8000 (for agents / Copilot)
axon-ui           # Streamlit web UI on :8501
```

**First session:**
```
axon
/ingest ./my-documents/
You: What is the main topic of these documents?
```

> For Docker Compose, embedding model options, Qdrant, MCP/Copilot setup, and multi-user HPC deployment, see the **[Setup Guide](SETUP.md)**.

---

## Entry points

| Command | Purpose |
|---|---|
| `axon` | Interactive REPL — start here |
| `axon-api` | FastAPI server — for integrations and agents |
| `axon-ui` | Streamlit web UI |
| `axon-mcp` | MCP server — for GitHub Copilot agent mode |

---

## Security

File ingestion is restricted to a configurable base directory (`RAG_INGEST_BASE`, defaults to the current working directory). Requests for paths outside this directory are rejected with 403. See [SECURITY.md](SECURITY.md) for details.

---

## License

MIT — see [LICENSE](LICENSE).
