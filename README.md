<div align="center">
  <img src="https://raw.githubusercontent.com/jyunming/Axon/main/docs/assets/brand/axon-wordmark.svg" alt="Axon" width="400" />

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

  <br/>

  <img src="https://raw.githubusercontent.com/jyunming/Axon/main/docs/assets/repl-demo.png" alt="Axon REPL — startup banner, ingest, and a cited query" width="820" />
</div>

---

## 🤔 Why Axon?

Most RAG tools make you choose between **cloud power** and **data privacy**. Axon is local-first — full capability with zero egress when you run on Ollama or vLLM; cloud providers (OpenAI, Gemini, Grok, GitHub Copilot, Ollama Cloud) stay optional.

- 🔒 **Private by default** — local inference via Ollama or vLLM is the recommended path; cloud providers (OpenAI, Gemini, Grok, GitHub Copilot, Ollama Cloud) remain opt-in. No telemetry. An offline mode blocks model downloads and web search for air-gapped machines.
- 📄 **Ingest anything** — 54 file formats (PDF, DOCX, Jupyter, code, images, URLs) in one command. SHA-256 dedup skips unchanged files.
- 🤖 **Works in your tools** — `@axon` in Copilot Chat, MCP for Claude Code / Codex / Gemini CLI / Cursor, Graph panel in VS Code or your browser.
- 🤝 **Built for teams** — share your knowledge base with signed, revocable read-only keys. Sealed (AES-256-GCM encrypted) sharing works safely through OneDrive, Dropbox, and Google Drive. Read-only mounts, expiring keys, no extra infrastructure. [Quick setup →](#-sealed-sharing-quick-start)
- 🕸️ **See your knowledge as a graph** — interactive 3D entity-relationship graph. Embedded webview in VS Code; opens in your browser everywhere else. Click any node to inspect its supporting chunks and source excerpt.
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
- Delete by document or chunk ID — deleted text can be re-ingested cleanly
- Stale detection for modified sources
- 4 content-aware chunking strategies

</td>
<td width="50%" valign="top">

### 🔧 LLMs & Embeddings
- **Local:** Ollama, vLLM
- **Cloud (API key):** OpenAI, Gemini, xAI Grok, GitHub Copilot, Ollama Cloud
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
- **Offline / air-gapped mode** — local models only, no downloads
- **AxonStore** — signed read-only sharing across OS users

</td>
<td width="50%" valign="top">

### ☁️ Cloud-Drive Sharing
**Sealed (AES-256-GCM encrypted) sharing works through any cloud sync drive.**
Files are ciphertext on disk — cloud providers see only encrypted bytes.

- OneDrive Personal / Business
- Dropbox
- Google Drive (Mirror mode)

→ [Sharing Guide](https://github.com/jyunming/Axon/blob/main/docs/SHARING.md) | [Quick Setup →](#-sealed-sharing-quick-start)

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 🛡️ Operations & Agents
- Graceful maintenance states: `normal → draining → readonly → offline`
- **REST API** — 69 routes with Swagger docs at `/docs`
- **MCP server** — 18 focused tools for Claude Code, Codex, Gemini, Cursor, Copilot (destructive/admin operations stay human-only)
- **`@axon`** VS Code chat participant with Graph panels

</td>
</tr>
</table>

---

## ⚡ Quick Start

```bash
pip install "axon-rag[starter]"   # Python 3.10+. Sealed sharing + extra loaders. Web GUI ships with axon-api.
axon --ingest ./my-notes          # index a folder: local embeddings, no LLM calls
axon "What do my notes say about the release plan?"   # cited answer from a local Ollama model
```

The first question pulls the default model (`llama3.1:8b`) into [Ollama](https://ollama.com) if
it is missing, or use a cloud model: `axon --model gpt-4o-mini "…"` with `OPENAI_API_KEY` set.
`axon` on its own opens the REPL; on a fresh machine it runs the setup wizard first.

<details>
<summary><b>See it run — first-run wizard, ingest, and a cited query</b></summary>

<div align="center">
  <img src="https://raw.githubusercontent.com/jyunming/Axon/main/docs/assets/repl-animation.gif" alt="Axon REPL — first-run wizard, ingest, and a cited query in action" width="640" />
</div>

</details>

If something doesn't look right:

```bash
axon --doctor                     # Health checks: Python, Ollama, model pulled, store writable.
axon update                       # Check PyPI and upgrade the package + VS Code extension together.
```

Local inference uses [Ollama](https://ollama.com) or vLLM (self-hosted). Cloud providers (OpenAI, Gemini, Grok, GitHub Copilot, Ollama Cloud) work via API keys.

**[→ Getting Started — from install to a cited answer →](https://github.com/jyunming/Axon/blob/main/docs/GETTING_STARTED.md)**

---

## 🔐 Sealed Sharing Quick Start

Share an encrypted knowledge base through OneDrive, Dropbox, or Google Drive. Cloud providers see only ciphertext.

```bash
pip install "axon-rag[starter]"   # on both machines (or the smaller [sealed] extra)
```

**Owner**

```bash
axon --store-init "/path/to/OneDrive/axon"                  # 1. put the store in the synced folder
axon --project-new research --ingest /path/to/documents     # 2. create a project and index it
axon                                                        # 3. open the REPL for the sealed steps:
```

```
axon> /store bootstrap <passphrase>            first time only; later sessions: /store unlock <passphrase>
axon> /project seal research                   encrypt the project in place
axon> /share generate research alice           prints the share string — send it to alice
```

**Grantee**

```bash
axon --store-init "/path/to/OneDrive/axon"              # 1. the same synced folder
axon --share-redeem "<share string>"                     # 2. key goes to the OS keyring
axon --project mounts/owner_research "question"         # 3. decrypted to a temp cache, wiped on exit
```

**[→ Full Sharing Guide](https://github.com/jyunming/Axon/blob/main/docs/SHARING.md)** — OneDrive / Dropbox / Google Drive setup, revocation, expiry, filesystem compatibility matrix.

---

## 🚀 Entry Points

| Command | Starts | Default Port | Best For |
|---------|--------|-------------|---------|
| `axon` | Interactive REPL | — | Day-to-day exploration, power users |
| `axon-api` | FastAPI REST server **+ web GUI** | `8420` | Agents, scripts, CI pipelines, browser UI |
| `axon-mcp` | MCP stdio server | — | Any MCP-compatible agent (Claude Code, Codex, Gemini CLI, Cursor, Copilot…) |

**Browser UI:** start `axon-api` and open **<http://localhost:8420/gui/>**. No extra
command or dependency needed — the GUI ships with the server.

> **Single-instance routing.** If an `axon-api` server is already running, the
> `axon` CLI detects it (a quick `/health/ready` probe) and routes store-mutating
> commands — `--ingest` and project create / delete / switch — *through* that
> server instead of opening a second in-process copy of the same store. This
> avoids two processes racing on the vector-store files, skips reloading the
> embedding model on every CLI call, and keeps a single writer for both the store
> and the logs. Pass `--local` to force an in-process brain, or point the CLI at a
> specific server with the `AXON_API_BASE` environment variable (or the
> `api_host` / `api_port` config fields).
>
> Relatedly, `axon-api` itself refuses to start a **second** server on a store
> another `axon-api` is already serving (one writer per store) — set
> `AXON_ALLOW_MULTIPLE_SERVERS=1` if you really want two.

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

Install the bundled VSIX to unlock the **`@axon` chat participant**, **Knowledge Graph panel** and **Code Graph panel** — directly inside VS Code alongside Copilot.

```
Extensions panel  →  "..."  →  Install from VSIX...
→  run `axon-ext`  (or install from VSIX manually)
```

Or connect via MCP for Copilot agent mode — point `.vscode/mcp.json` at `axon-mcp` and all 18 tools appear in the agent hammer menu automatically.

> The VS Code extension surfaces **20 LM tools** to Copilot Chat — the 18 MCP tools (query, search, ingest, config, graph facts, sharing) plus `show_graph` and `ingest_image`.

**[VS Code and MCP setup →](https://github.com/jyunming/Axon/blob/main/docs/REFERENCE.md#10-vs-code-extension)**

---

## 🐍 Use Axon from Your Python Agent

Drop-in retrievers for LangChain and LlamaIndex agents — no REST round-trips, no extra process. Both wrap the same `AxonBrain.search_raw()` codepath the REST and REPL surfaces use, so hybrid search, reranking, HyDE, multi-query, and the GraphRAG budget apply automatically.

```python
# pip install "axon-rag[langchain]"
from axon import AxonBrain, AxonConfig
from axon.integrations.langchain import AxonRetriever

brain = AxonBrain(AxonConfig.load())   # ~/.config/axon/config.yaml, or .load("path.yaml")
retriever = AxonRetriever(brain=brain, top_k=5)

docs = retriever.invoke("what does the project do?")  # list[Document]
```

```python
# pip install "axon-rag[llama-index]"
from axon.integrations.llama_index import AxonLlamaRetriever

retriever = AxonLlamaRetriever(brain=brain, top_k=5)
nodes = retriever.retrieve("what does the project do?")  # list[NodeWithScore]
```

Per-call overrides (e.g. force HyDE for one question): `retriever.with_overrides({"hyde": True}).invoke(query)`. From async code, `await retriever.aretrieve(query, hyde=True, top_k=8)` accepts the same flags as kwargs without rebuilding the retriever (v0.4.1).

---

## 📚 Documentation

| | Guide | What it covers |
|-|-------|---------------|
| 🚀 | **[Getting Started](https://github.com/jyunming/Axon/blob/main/docs/GETTING_STARTED.md)** | Install, index a folder, get a cited answer — then the browser, VS Code, MCP and REST |
| 📖 | **[Reference](https://github.com/jyunming/Axon/blob/main/docs/REFERENCE.md)** | Every setting, CLI flag, REPL command, REST route, MCP tool and VS Code feature; retrieval, graphs, projects, offline mode, operations, the Python API |
| 🔐 | **[Sharing](https://github.com/jyunming/Axon/blob/main/docs/SHARING.md)** | Plaintext and sealed sharing, OneDrive / Dropbox / Google Drive, revocation and expiry |
| 🔧 | **[Troubleshooting](https://github.com/jyunming/Axon/blob/main/docs/TROUBLESHOOTING.md)** | Error messages and their fixes |
| 🛠️ | **[Contributing](https://github.com/jyunming/Axon/blob/main/CONTRIBUTING.md)** | Development setup, tests, evaluation, releases |

---

## 🔒 Security

Ingestion is sandboxed to a configurable base directory (`RAG_INGEST_BASE`). Requests outside it are rejected with `403`. See [SECURITY.md](https://github.com/jyunming/Axon/blob/main/SECURITY.md).

## 📄 License

MIT — see [LICENSE](https://github.com/jyunming/Axon/blob/main/LICENSE).

