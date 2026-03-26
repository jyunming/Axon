# Axon

<p align="center">
  <img src="docs/assets/axon-mark.png" alt="Axon" width="350" />
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

**A privacy-first, local AI knowledge engine featuring GraphRAG, RAPTOR, and Code-Aware structural search.**

Point Axon at your documents. Ask questions. Get answers — using a local LLM with no cloud, no API keys required.
Share knowledge bases across your team with HMAC-secured read-only project mounts. Monitor every query with the built-in Governance Console.

---

## Key capabilities

- **Hybrid search** — dense vector + BM25 keyword, fused for better precision than either alone; sentence-window retrieval for better context around matched chunks
- **Multi-LLM** — Ollama (local), Gemini, OpenAI, xAI Grok, vLLM, GitHub Copilot (incl. Claude models via Copilot); switch live from the REPL; streaming responses supported
- **Multi-embedding** — sentence-transformers, Ollama, FastEmbed (BGE-M3 for multilingual/long-doc)
- **Vector stores** — ChromaDB (default), Qdrant (local + cloud), LanceDB
- **54 file formats** — PDF, DOCX, XLSX, PPTX, EPUB, EML, MSG, LaTeX, Jupyter (.ipynb), Parquet, SQL, XML, RTF, JSONL, CSV, Markdown, HTML, plain text, images (BMP/PNG/TIF/PGM/JPEG with VLM auto-captioning), and 30+ source-code formats (.py, .ts, .js, .go, .rs, .java, .kt, .swift, .cpp, .c, .rb, .php, .sh and more)
- **URL ingestion** — ingest any public web page directly (`/ingest https://...`) alongside local files
- **Smart re-ingest** — SHA-256 hash deduplication skips unchanged files; stale-document detection flags chunks whose source has been modified
- **Adaptive chunking** — recursive, semantic, Markdown-aware, and cosine-semantic strategies
- **Projects** — isolated knowledge bases per named project; nested projects search children automatically; cross-project scopes (`@projects`, `@mounts`, `@store`) for federated search
- **Query transformations** — HyDE, multi-query, step-back, decomposition, contextual compression, CRAG-Lite; intelligent query routing selects the best strategy per query
- **Multi-turn sessions** — persistent conversation history with configurable session memory; discuss mode for iterative exploration
- **RAPTOR + GraphRAG** — RAPTOR hierarchical summaries + entity/relation/community graph with local/global/hybrid modes and light/standard/deep depth control; interactive 3D graph panel in VS Code or `/graph-viz` HTML export
- **Code graph** — structural file/class/function graph with `IMPORTS`/`CONTAINS`/`MENTIONED_IN` edges; visualise alongside the knowledge graph in VS Code
- **Reranking** — cross-encoder (BGE) reranking; CRAG-Lite corrective retrieval on low-confidence chunks
- **Strict offline mode** — lock to local-only assets; zero outbound network calls even for embedding or LLM
- **AxonStore sharing** — HMAC-secured read-only project mounts across OS users; lazy revocation via manifest tombstones
- **Governance Console** — operator dashboard with SQLite WAL audit trail; per-query event log, session tracking, write-lease monitoring; graceful maintenance states for zero-downtime draining
- **Agent-ready** — FastAPI REST API + MCP server (27 tools) for Copilot agent mode; OpenAI-compatible tool schema; `@axon` VS Code chat participant; interactive OpenAPI docs at `/docs`

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
ollama pull llama3.1:8b   # default — 4.7 GB, ~8 GB RAM
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
Show me the graph for how the retrieval pipeline works
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

Start `axon-api`, reload VS Code — Axon's 27 tools appear in Copilot agent mode (hammer icon).

> See **[Setup Guide](SETUP.md)** for full setup details, workflow diagrams, and per-entry-point examples.

---

## Interactive REPL

![Axon REPL](docs/assets/repl-demo.png)

![Axon REPL demo](docs/assets/repl-demo.gif)

---

## Graph Panel — Investigate Your Knowledge Base Visually

Invoke the **Axon Graph Panel** via the graph command or tool to open a split view directly inside VS Code — no browser, no extra tools:

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
- **Knowledge Graph** — entity–relation graph built from **any document** (PDF, DOCX, Markdown…) during ingest. Requires `graph_rag: true` (disabled in the shipped `config.yaml` by default; enable when your corpus is ready). Nodes are named entities (people, concepts, components); edges are extracted relations.
- **Code Graph** — structural file/class/function graph for source code (requires `code_graph: true` in `config.yaml`). Nodes are files, classes, and functions; edges are `IMPORTS` / `CONTAINS` / `MENTIONED_IN` relationships. Click a node to jump to that definition.

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

## AxonStore — Multi-User Knowledge Sharing

AxonStore lets you share a project's knowledge base with other OS users as a **read-only mount** — they can query it without being able to ingest or delete.

```
# Owner: initialise AxonStore and share a project
axon> /store-init /data
axon> /share my-project alice
→ share_string: eyJrZXlfaWQiOiAic2tfYTFiMmMzZDQiLCAidG9rZW4...

# Alice: redeem the share string on her machine
axon> /redeem eyJrZXlfaWQiOiAic2tfYTFiMmMzZDQiLCAidG9rZW4...
→ Mounted as: mounts/bob_my-project

# Alice can now query bob's project:
axon> /project mounts/bob_my-project
axon> What are the key themes in this knowledge base?
```

Share keys use HMAC-SHA256 to bind the token to `(key_id, project, grantee)` — a leaked token cannot be repurposed for a different project or user. Revocation is lazy: the owner revokes, and stale mounts are cleaned on Alice's next access.

```bash
# Revoke access at any time:
axon> /revoke sk_a1b2c3d4
→ Key revoked. Alice's mount will be removed on her next project access.
```

---

## Governance Console — Audit Every Query

The Governance Console gives operators a real-time view of all knowledge base activity:

```bash
# Start the API and open the governance dashboard:
GET /governance/overview      # active projects, session count, lease count
GET /governance/audit?limit=50  # per-query event log (project, surface, query, latency)
GET /governance/sessions      # active Copilot agent sessions (opened_at, last_query)
GET /registry/leases          # write-lease counts per project (safe to check before maintenance)
```

Every query emits a structured audit event — project, surface (API/MCP/VS Code/REPL), query text, and latency. Events are stored in a **SQLite WAL** database under the project directory, surviving server restarts.

**When to use it:**
- Verify which projects are being queried and by which surfaces
- Check active write leases before taking a project offline for maintenance
- Audit user activity in a shared AxonStore deployment

---

## Guides

| Guide | What it covers |
|---|---|
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
