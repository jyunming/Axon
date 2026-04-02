# axon-rag

**Your documents, answerable. On your hardware.**

Drop in PDFs, code, spreadsheets, or URLs — ask anything, get cited answers from a local LLM. Nothing leaves your machine.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jyunming/Axon/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/axon-rag.svg)](https://pypi.org/project/axon-rag/)

---

![Axon REPL](https://raw.githubusercontent.com/jyunming/Axon/main/docs/assets/repl-animation.gif)

---

## Why Axon?

Most RAG tools make you choose between cloud power and data privacy. Axon runs entirely on your hardware — full capability, zero egress.

- **Private by default** — all inference runs locally via Ollama. No API key required, no upload, no telemetry.
- **Ingest anything** — 54 file formats (PDF, DOCX, PPTX, Jupyter, code, images, URLs) with SHA-256 dedup.
- **Works in your tools** — `@axon` in Copilot Chat, MCP server for Claude Code / Codex / Gemini CLI / Cursor, Graph panel in VS Code.
- **Built for teams** — share your knowledge base with signed, revocable read-only keys.
- **Production-grade retrieval** — hybrid search, reranking, HyDE, multi-query expansion, RAPTOR, GraphRAG.

---

## Install

```bash
pip install axon-rag
```

Requires Python 3.10+ and [Ollama](https://ollama.com) running locally.

### Optional extras

```bash
pip install axon-rag[ui]        # Streamlit web UI (axon-ui)
pip install axon-rag[chroma]    # ChromaDB vector store
pip install axon-rag[qdrant]    # Qdrant vector store
pip install axon-rag[graphrag]  # GraphRAG (networkx + leidenalg)
pip install axon-rag[loaders]   # Extra file loaders (EPUB, RTF, email)
pip install axon-rag[all]       # Everything above
```

---

## Quick Start

```bash
# Pull a model (first time only)
ollama pull llama3.2

# Launch the interactive REPL
axon

# Ingest a file or folder
axon ingest ~/docs/myreport.pdf
axon ingest ~/projects/myrepo/

# Ask a question
axon query "What are the key findings?"
```

---

## Entry Points

| Command | What it does |
|---------|-------------|
| `axon` | Interactive REPL — day-to-day exploration |
| `axon-api` | FastAPI REST server on port 8000 |
| `axon-mcp` | MCP stdio server — 30 tools for Claude Code, Codex, Gemini CLI, Cursor |
| `axon-ui` | Streamlit web UI on port 8501 (requires `[ui]` extra) |
| `axon-ext` | Install the bundled VS Code extension |

---

## VS Code Extension

![Axon Copilot](https://raw.githubusercontent.com/jyunming/Axon/main/docs/assets/AxonCopilot.gif)

Install the bundled extension to get the `@axon` chat participant, Knowledge Graph panel, Code Graph panel, and Governance dashboard directly inside VS Code alongside GitHub Copilot:

```bash
axon-ext
```

![Axon VS Code Graph Panel](https://raw.githubusercontent.com/jyunming/Axon/main/docs/assets/vscode-graph-panel.png)

---

## MCP Server

Connect any MCP-compatible agent to your local knowledge base:

```json
{
  "servers": {
    "axon": {
      "command": "axon-mcp"
    }
  }
}
```

30 tools available: ingest, query, search, project management, graph operations, sharing, and more.

---

## Configuration

On first run, Axon creates `~/.config/axon/config.yaml`. Key settings:

```yaml
llm:
  provider: ollama          # ollama | openai | gemini | grok | copilot
  model: llama3.2
  base_url: http://localhost:11434

embedding:
  provider: sentence_transformers
  model: BAAI/bge-small-en-v1.5

rag:
  top_k: 5
  hybrid: true
  rerank: false
  hyde: false
```

---

## Documentation

Full documentation is available on GitHub:

- [Getting Started](https://github.com/jyunming/Axon/blob/main/docs/GETTING_STARTED.md) — first-time walkthrough
- [Setup Guide](https://github.com/jyunming/Axon/blob/main/docs/SETUP.md) — install, models, VS Code, MCP
- [Admin Reference](https://github.com/jyunming/Axon/blob/main/docs/ADMIN_REFERENCE.md) — every endpoint, command, and config option
- [Advanced RAG](https://github.com/jyunming/Axon/blob/main/docs/ADVANCED_RAG.md) — HyDE, RAPTOR, GraphRAG, CRAG-Lite
- [Troubleshooting](https://github.com/jyunming/Axon/blob/main/docs/TROUBLESHOOTING.md) — common errors and fixes

---

## License

MIT — see [LICENSE](https://github.com/jyunming/Axon/blob/main/LICENSE)
