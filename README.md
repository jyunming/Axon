<div align="center">
  <img src="https://raw.githubusercontent.com/jyunming/Axon/main/docs/assets/brand/axon-wordmark.svg" alt="Axon" width="320" />
</div>

<div align="center">
  <img src="https://raw.githubusercontent.com/jyunming/Axon/main/docs/assets/repl-animation.gif" alt="Axon REPL" width="400" />

  <h3>Your documents, answerable. On your hardware.</h3>

  <p>
    Drop in PDFs, code, spreadsheets, or URLs — ask anything, get cited answers from a local LLM.<br/>
    Nothing leaves your machine.
  </p>

  [![PyPI version](https://img.shields.io/pypi/v/axon-rag.svg)](https://pypi.org/project/axon-rag/)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![CI](https://github.com/jyunming/Axon/actions/workflows/ci.yml/badge.svg)](https://github.com/jyunming/Axon/actions/workflows/ci.yml)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jyunming/Axon/blob/main/LICENSE)
  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

<div align="center">
  <img src="https://raw.githubusercontent.com/jyunming/Axon/main/docs/assets/repl-demo.png" alt="Axon REPL startup" width="820" />
</div>

---

## 🤔 Why Axon?

Most RAG tools make you choose between **cloud power** and **data privacy**. Axon runs entirely on your hardware — full capability, zero egress.

- 🔒 **Private by default** — all inference runs locally via Ollama or vLLM. No API key, no upload, no telemetry.
- 📄 **Ingest anything** — 54 file formats (PDF, DOCX, Jupyter, code, images, URLs) in one command. SHA-256 dedup skips unchanged files.
- 🤖 **Works in your tools** — `@axon` in Copilot Chat, MCP for Claude Code / Codex / Gemini CLI / Cursor, Graph panel in VS Code or your browser.
- 🤝 **Built for teams** — share your knowledge base with signed, revocable read-only keys. Sealed (AES-256-GCM encrypted) sharing works safely through OneDrive, Dropbox, and Google Drive. Per-user permissions, full audit trail, no extra infrastructure. [Quick setup →](#sealed-sharing-quick-start)
- 🕸️ **See your knowledge as a graph** — interactive 3D entity-relationship graph. Embedded webview in VS Code; opens in your browser everywhere else. Click any node to jump to the exact source line.
- 🔬 **Production-grade retrieval** — hybrid search, reranking, HyDE, multi-query expansion, and automatic web fallback. Zero manual tuning.

---

## ✨ Capabilities

<table>
<tr>
<td width="50%" valign="top">

### 🔍 Retrieval
- Hybrid semantic + keyword search
- HyDE, multi-query, step-back, query decomposition
- Sentence-window context retrieval
- BGE reranker for second-pass precision
- Web fallback via Brave Search (CRAG-Lite)
- Smart per-question query routing
- **Structured citations** — `sources` + `citations` arrays with character offsets (Claude / OpenAI compatible)

</td>
<td width="50%" valign="top">

### 🧠 Graph Intelligence
- **GraphRAG** — entity/relation/community graph (local / global / hybrid)
- **Dynamic Graph** — bi-temporal SQLite facts with `valid_at` + `invalid_at`
- **Federated** — weighted RRF over multiple backends, tunable per query
- **Point-in-time queries** (v0.3.2) — `--graph-retrieve "..." --graph-at TS`
- **Conflict inspection** (v0.3.2) — `--graph-conflicts` surfaces `status='conflicted'` facts
- **RAPTOR** — hierarchical corpus summaries · **Code Graph** via AST · 3D webview

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 📥 Ingest Everything
- **54 file formats** — PDF, DOCX, XLSX, PPTX, Jupyter, images, 24 code formats
- URL ingestion — any public web page
- SHA-256 dedup skips unchanged files
- Stale detection for modified sources
- 4 content-aware chunking strategies

</td>
<td width="50%" valign="top">

### 🔧 LLMs & Embeddings
- **Local:** Ollama, vLLM
- **Cloud:** OpenAI, Gemini, xAI Grok, GitHub Copilot, Ollama Cloud (API)
- Hot-swap provider and model — no restart needed
- Streaming on all providers
- 4 embedding providers; BGE-M3 for multilingual

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🏗️ Projects & Privacy
- Isolated knowledge base per project, with nesting
- Federated search across projects (`@projects`, `@mounts`, `@store`)
- **Strict offline / air-gapped mode** — zero outbound calls
- **AxonStore** — signed read-only sharing across OS users

</td>
<td width="50%" valign="top">

### ☁️ Cloud-Drive Sharing
**Sealed (AES-256-GCM encrypted) sharing works through any cloud sync drive.**
Files are ciphertext on disk — cloud providers see only encrypted bytes.

- OneDrive Personal / Business
- Dropbox
- Google Drive (Mirror mode)

→ [Sharing Guide](docs/SHARING.md) | [Quick Setup →](#sealed-sharing-quick-start)

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🛡️ Governance & Agents
- **Governance Console** — full audit trail of every query
- Graceful maintenance states: `normal → draining → readonly → offline`
- **REST API** — 70 endpoints with Swagger docs at `/docs`
- **MCP server** — 48 tools for Claude Code, Codex, Gemini, Cursor, Copilot
- **`@axon`** VS Code chat participant with Graph and Governance panels

</td>
</tr>
</table>

---

## ⚡ Quick Start

```bash
pip install "axon-rag[starter]"   # Python 3.10+. Includes UI, sealed sharing, extra loaders.
axon                              # First run auto-launches the setup wizard, then drops into the REPL.
```

That's it. The wizard configures your LLM provider, embedding model, and retrieval defaults; subsequent runs go straight to the REPL.

If something doesn't look right:

```bash
axon --doctor                     # Health checks: Python, Ollama, model pulled, store writable.
```

Local inference uses [Ollama](https://ollama.com) or vLLM (self-hosted). Cloud providers (OpenAI, Gemini, Grok, GitHub Copilot, Ollama Cloud) work via API keys.

**[→ Setup guide for VS Code, MCP, and cloud providers →](https://github.com/jyunming/Axon/blob/main/docs/SETUP.md)**

---

## 🔐 Sealed Sharing Quick Start

Share an encrypted knowledge base through OneDrive, Dropbox, or Google Drive. Cloud providers see only ciphertext.

```bash
pip install "axon-rag[sealed]"   # install sealed extra on both machines
```

**Owner (5 steps)**

```bash
axon --store-init "/path/to/OneDrive/AxonStore"  # 1. point store at sync folder
axon --store-bootstrap "your-passphrase"          # 2. bootstrap master key (once per machine)
axon --project-new research                       # 3. create project + ingest
axon --project research --ingest /docs
axon --project-seal research                      # 4. encrypt in place (≈1 s per 100 MB)
axon --share-generate research alice              # 5. print SEALED1:... string — send to grantee
```

**Grantee (3 steps)**

```bash
axon --store-init "/path/to/OneDrive/AxonStore"   # 1. same shared folder
axon --share-redeem "SEALED1:..."                  # 2. redeem — DEK stored in OS keyring
axon --project mounts/owner_research "question"   # 3. query; Axon decrypts to temp, wipes on exit
```

**[→ Full Sharing Guide](docs/SHARING.md)** — OneDrive setup, revocation, headless/Docker grantees, filesystem compatibility matrix.

---

## 🚀 Entry Points

| Command | Starts | Default Port | Best For |
|---------|--------|-------------|---------|
| `axon` | Interactive REPL | — | Day-to-day exploration, power users |
| `axon-api` | FastAPI REST server | `8000` | Agents, scripts, CI pipelines |
| `axon-mcp` | MCP stdio server | — | Any MCP-compatible agent (Claude Code, Codex, Gemini CLI, Cursor, Copilot…) |
| `axon-ui` | Streamlit UI | `8501` | Browser-based exploration |

---

## 🔌 VS Code + GitHub Copilot

<div align="center">
  <img src="https://raw.githubusercontent.com/jyunming/Axon/main/docs/assets/AxonCopilot.gif" alt="Axon Copilot integration" width="400" />
</div>

<br/>

<div align="center">
  <img src="https://raw.githubusercontent.com/jyunming/Axon/main/docs/assets/vscode-graph-panel.png" alt="Axon VS Code Graph Panel — answer, cited sources, and interactive 3D code graph" width="820" />
</div>

<br/>

Install the bundled VSIX to unlock the **`@axon` chat participant**, **Knowledge Graph panel**, **Code Graph panel**, and **Governance dashboard** — directly inside VS Code alongside Copilot.

```
Extensions panel  →  "..."  →  Install from VSIX...
→  run `axon-ext`  (or install from VSIX manually)
```

Or connect via MCP for Copilot agent mode — point `.vscode/mcp.json` at `axon-mcp` and all 48 tools appear in the agent hammer menu automatically.

> The VS Code extension surfaces **44 LM tools** to Copilot Chat, covering core RAG operations, sealed-store security, sharing, and governance.

**[Full setup guide →](https://github.com/jyunming/Axon/blob/main/docs/SETUP.md)**

---

## 🐍 Use Axon from Your Python Agent

Drop-in retrievers for LangChain and LlamaIndex agents — no REST round-trips, no extra process. Both wrap the same `AxonBrain.search_raw()` codepath the REST and REPL surfaces use, so hybrid search, reranking, HyDE, multi-query, and the GraphRAG budget apply automatically.

```python
# pip install "axon-rag[langchain]"
from axon import AxonBrain, AxonConfig
from axon.integrations.langchain import AxonRetriever

brain = AxonBrain(AxonConfig.from_yaml("config.yaml"))
retriever = AxonRetriever(brain=brain, top_k=5)

docs = retriever.invoke("what does the project do?")  # list[Document]
```

```python
# pip install "axon-rag[llama-index]"
from axon.integrations.llama_index import AxonLlamaRetriever

retriever = AxonLlamaRetriever(brain=brain, top_k=5)
nodes = retriever.retrieve("what does the project do?")  # list[NodeWithScore]
```

Per-call overrides (e.g. force HyDE for one question): `retriever.with_overrides({"hyde": True}).invoke(query)`.

---

## 📚 Documentation

**Getting started**

| | Guide | What it covers |
|-|-------|---------------|
| 🚀 | **[Getting Started](https://github.com/jyunming/Axon/blob/main/docs/GETTING_STARTED.md)** | First-time walkthrough — ingest, query, settings |
| ⚙️ | **[Setup Guide](https://github.com/jyunming/Axon/blob/main/docs/SETUP.md)** | Install, models, VS Code extension, MCP connection |
| 🔧 | **[Troubleshooting](https://github.com/jyunming/Axon/blob/main/docs/TROUBLESHOOTING.md)** | Common errors and platform-specific fixes |

**Reference**

| | Guide | What it covers |
|-|-------|---------------|
| 🔑 | **[Admin Reference](https://github.com/jyunming/Axon/blob/main/docs/ADMIN_REFERENCE.md)** | Every endpoint, REPL command, CLI flag, and config option |
| ⚡ | **[Quick Reference](https://github.com/jyunming/Axon/blob/main/docs/QUICKREF.md)** | Commands and flags at a glance |
| 📡 | **[API Reference](https://github.com/jyunming/Axon/blob/main/docs/API_REFERENCE.md)** | Full REST endpoint reference with request/response schemas |
| 🔌 | **[MCP Tools](https://github.com/jyunming/Axon/blob/main/docs/MCP_TOOLS.md)** | All 48 MCP tool signatures with parameter defaults |

**Deep dives**

| | Guide | What it covers |
|-|-------|---------------|
| 🤖 | **[Model Guide](https://github.com/jyunming/Axon/blob/main/docs/MODEL_GUIDE.md)** | Choosing LLM and embeddings; per-provider config examples |
| 🔬 | **[Advanced RAG](https://github.com/jyunming/Axon/blob/main/docs/ADVANCED_RAG.md)** | HyDE, RAPTOR, GraphRAG, CRAG-Lite — how each technique works |
| 🌐 | **[Web Search](https://github.com/jyunming/Axon/blob/main/docs/WEB_SEARCH.md)** | Brave Search integration, CRAG-Lite fallback setup |
| 🏝️ | **[Offline / Air-gap Guide](https://github.com/jyunming/Axon/blob/main/docs/OFFLINE_GUIDE.md)** | Full air-gap setup, model pre-download, local-assets-only mode |
| 💻 | **[Code RAG Guide](https://github.com/jyunming/Axon/blob/main/docs/CODE_RAG_GUIDE.md)** | Code graph retrieval and structural search |
| 🤝 | **[AxonStore](https://github.com/jyunming/Axon/blob/main/docs/AXON_STORE.md)** | Multi-user sharing, revocation, and the lease lifecycle |
| 🔐 | **[Sharing Guide](https://github.com/jyunming/Axon/blob/main/docs/SHARING.md)** | Plaintext and sealed sharing — which filesystems are safe, OneDrive/Dropbox/Google Drive setup, revocation |
| 📊 | **[Governance Console](https://github.com/jyunming/Axon/blob/main/docs/GOVERNANCE_CONSOLE.md)** | Audit trail, maintenance runbook, session management |
| 📈 | **[Evaluation Guide](https://github.com/jyunming/Axon/blob/main/docs/EVALUATION.md)** | RAGAS metrics, running evals, building testsets |
| 🛠️ | **[Development Guide](https://github.com/jyunming/Axon/blob/main/docs/DEVELOPMENT.md)** | Tests, contributing, pre-commit hooks, packaging & release |

---

## 🔒 Security

Ingestion is sandboxed to a configurable base directory (`RAG_INGEST_BASE`). Requests outside it are rejected with `403`. See [SECURITY.md](https://github.com/jyunming/Axon/blob/main/SECURITY.md).

## 📄 License

MIT — see [LICENSE](https://github.com/jyunming/Axon/blob/main/LICENSE).

